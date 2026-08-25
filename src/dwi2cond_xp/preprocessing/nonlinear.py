"""FSL-compatible dense nonlinear tensor resampling and PPD reorientation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from .tensor_ops import decompose_tensor6
from .transforms import fsl_voxel_to_scaled_mm


@dataclass(frozen=True)
class NonlinearTensorResult:
    """Dense-warp tensor result and explicit numerical validity masks."""

    tensor: np.ndarray
    valid_mask: np.ndarray
    backward_jacobian: np.ndarray
    forward_jacobian: np.ndarray
    backward_jacobian_determinant: np.ndarray
    fold_mask: np.ndarray
    near_singular_mask: np.ndarray
    low_fa_mask: np.ndarray
    repeated_eigenvalue_mask: np.ndarray
    invalid_tensor_mask: np.ndarray


def _tensor6_to_matrix(tensor: np.ndarray) -> np.ndarray:
    matrix = np.empty(tensor.shape[:-1] + (3, 3), dtype=tensor.dtype)
    matrix[..., 0, 0] = tensor[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = tensor[..., 1]
    matrix[..., 0, 2] = matrix[..., 2, 0] = tensor[..., 2]
    matrix[..., 1, 1] = tensor[..., 3]
    matrix[..., 1, 2] = matrix[..., 2, 1] = tensor[..., 4]
    matrix[..., 2, 2] = tensor[..., 5]
    return matrix


def _matrix_to_tensor6(matrix: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 0, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 1],
            matrix[..., 1, 2],
            matrix[..., 2, 2],
        ),
        axis=-1,
    )


def fsl_displacement_gradient(displacement: np.ndarray) -> np.ndarray:
    """Differentiate a relative-mm FNIRT field like ``vecreg::sjgradient``.

    The result is indexed as ``[..., displacement_component, voxel_axis]``.
    FSL differentiates per target voxel, without dividing by voxel dimensions.
    """

    values = np.asarray(displacement, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("displacement must have shape (x, y, z, 3)")
    if any(size < 2 for size in values.shape[:3]) or not np.all(np.isfinite(values)):
        raise ValueError(
            "displacement must be finite with at least two voxels per axis"
        )
    gradient = np.empty(values.shape[:3] + (3, 3), dtype=np.float32)
    for component in range(3):
        field = values[..., component]
        for axis in range(3):
            destination = gradient[..., component, axis]
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            edge0 = [slice(None)] * 3
            edge1 = [slice(None)] * 3
            edge_last = [slice(None)] * 3
            edge_before_last = [slice(None)] * 3
            lower[axis] = slice(0, -2)
            upper[axis] = slice(2, None)
            interior = [slice(None)] * 3
            interior[axis] = slice(1, -1)
            destination[tuple(interior)] = np.float32(
                (field[tuple(upper)] - field[tuple(lower)]) / np.float32(2.0)
            )
            edge0[axis] = 0
            edge1[axis] = 1
            destination[tuple(edge0)] = np.float32(
                field[tuple(edge1)] - field[tuple(edge0)]
            )
            edge_last[axis] = -1
            edge_before_last[axis] = -2
            destination[tuple(edge_last)] = np.float32(
                field[tuple(edge_last)] - field[tuple(edge_before_last)]
            )
    return gradient


def fsl_warp_jacobians(
    displacement: np.ndarray,
    *,
    singular_tolerance: float = 1e-8,
    voxel_axis_directions: np.ndarray | None = None,
    voxel_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return backward/forward Jacobians and fold/singularity masks."""

    if not np.isfinite(singular_tolerance) or singular_tolerance <= 0:
        raise ValueError("singular_tolerance must be positive and finite")
    directions = (
        np.ones(3, dtype=np.float64)
        if voxel_axis_directions is None
        else np.asarray(voxel_axis_directions, dtype=np.float64)
    )
    if directions.shape != (3,) or not np.all(np.isin(directions, (-1.0, 1.0))):
        raise ValueError("voxel_axis_directions must contain three signs")
    dimensions = (
        np.ones(3, dtype=np.float64)
        if voxel_sizes is None
        else np.asarray(voxel_sizes, dtype=np.float64)
    )
    if (
        dimensions.shape != (3,)
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0.0)
    ):
        raise ValueError("voxel_sizes must contain three positive finite values")
    gradient = fsl_displacement_gradient(displacement).astype(np.float64)
    gradient *= (directions / dimensions).reshape((1, 1, 1, 1, 3))
    backward = gradient
    backward[..., 0, 0] += 1.0
    backward[..., 1, 1] += 1.0
    backward[..., 2, 2] += 1.0
    determinant = np.linalg.det(backward)
    fold = determinant <= 0.0
    near_singular = np.abs(determinant) <= singular_tolerance
    forward = np.full(backward.shape, np.nan, dtype=np.float64)
    invertible = ~near_singular & np.isfinite(determinant)
    forward[invertible] = np.linalg.inv(backward[invertible])
    return backward, forward, determinant, fold, near_singular


def _axial_fa(eigenvalues: np.ndarray) -> np.ndarray:
    mean = np.mean(eigenvalues, axis=-1)
    numerator = np.sum((eigenvalues - mean[..., None]) ** 2, axis=-1)
    denominator = np.sum(eigenvalues**2, axis=-1)
    result = np.zeros(mean.shape, dtype=np.float64)
    positive = denominator > 0.0
    result[positive] = np.sqrt(1.5 * numerator[positive] / denominator[positive])
    return result


def _ppd_rotation(
    forward: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    singular_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the two leading eigenvectors using the FSL PPD construction."""

    first = eigenvectors[..., :, 2]
    second = eigenvectors[..., :, 1]
    mapped_first = np.einsum("...ij,...j->...i", forward, first)
    mapped_second = np.einsum("...ij,...j->...i", forward, second)
    first_norm = np.linalg.norm(mapped_first, axis=-1)
    second_norm = np.linalg.norm(mapped_second, axis=-1)
    stable = (
        np.isfinite(first_norm)
        & np.isfinite(second_norm)
        & (first_norm > singular_tolerance)
        & (second_norm > singular_tolerance)
    )
    target_first = np.zeros_like(mapped_first)
    target_second = np.zeros_like(mapped_second)
    target_first[stable] = mapped_first[stable] / first_norm[stable, None]
    target_second[stable] = mapped_second[stable] / second_norm[stable, None]
    projection = (
        target_second
        - np.sum(target_first * target_second, axis=-1)[..., None] * target_first
    )
    projection_norm = np.linalg.norm(projection, axis=-1)
    stable &= np.isfinite(projection_norm) & (projection_norm > singular_tolerance)
    projection[stable] /= projection_norm[stable, None]
    source_third = np.cross(first, second)
    target_third = np.cross(target_first, projection)
    source_frame = np.stack((first, second, source_third), axis=-1)
    target_frame = np.stack((target_first, projection, target_third), axis=-1)
    rotation = np.einsum(
        "...ij,...kj->...ik", target_frame, source_frame, optimize=True
    )
    rotation[~stable] = np.nan
    return rotation, stable


def _fsl_pull_coordinates(
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    source_shape: tuple[int, int, int],
    source_affine: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    target_sampling = fsl_voxel_to_scaled_mm(reference_shape, reference_affine)
    source_sampling = fsl_voxel_to_scaled_mm(source_shape, source_affine)
    grid = np.indices(reference_shape, dtype=np.float64).reshape(3, -1)
    target_mm = target_sampling[:3, :3] @ grid + target_sampling[:3, 3, None]
    source_mm = target_mm + displacement.reshape(-1, 3).T
    return np.linalg.inv(source_sampling[:3, :3]) @ (
        source_mm - source_sampling[:3, 3, None]
    )


def resample_tensor_ppd_fsl(
    tensor: np.ndarray,
    source_affine: np.ndarray,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    displacement: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    singular_tolerance: float = 1e-8,
    repeated_eigenvalue_tolerance: float = 1e-8,
    low_fa_threshold: float = 0.05,
    compatibility_mode: str = "strict-fsl",
) -> NonlinearTensorResult:
    """Resample an FSL tensor through a dense FNIRT warp and apply PPD.

    ``displacement`` is the relative-mm backward field on the reference grid:
    each target voxel maps to the source FSL scaled-mm coordinate at the same
    grid location plus the stored displacement vector.
    """

    values = np.asarray(tensor, dtype=np.float32)
    target_shape = tuple(int(value) for value in reference_shape)
    warp = np.asarray(displacement, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 6:
        raise ValueError("tensor must have shape (x, y, z, 6)")
    if len(target_shape) != 3 or any(size <= 0 for size in target_shape):
        raise ValueError("reference_shape must contain three positive dimensions")
    if warp.shape != target_shape + (3,) or not np.all(np.isfinite(warp)):
        raise ValueError("displacement must be finite and match the reference grid")
    if not np.all(np.isfinite(source_affine)) or np.shape(source_affine) != (4, 4):
        raise ValueError("source_affine must be a finite 4x4 matrix")
    if not np.all(np.isfinite(reference_affine)) or np.shape(reference_affine) != (
        4,
        4,
    ):
        raise ValueError("reference_affine must be a finite 4x4 matrix")
    if repeated_eigenvalue_tolerance <= 0 or low_fa_threshold < 0:
        raise ValueError(
            "eigenvalue tolerance must be positive and FA threshold nonnegative"
        )
    if compatibility_mode not in ("strict-fsl", "robust"):
        raise ValueError("compatibility_mode must be strict-fsl or robust")

    if source_mask is None:
        source_valid = np.any(values[..., :3] != 0.0, axis=-1)
    else:
        source_valid = np.asarray(source_mask) > 0
        if source_valid.shape != values.shape[:3]:
            raise ValueError("source_mask must match the tensor grid")
    target_valid = np.ones(target_shape, dtype=bool)
    if reference_mask is not None:
        target_valid = np.asarray(reference_mask) > 0
        if target_valid.shape != target_shape:
            raise ValueError("reference_mask must match the reference grid")

    coordinates = _fsl_pull_coordinates(
        target_shape,
        np.asarray(reference_affine, dtype=np.float64),
        values.shape[:3],
        np.asarray(source_affine, dtype=np.float64),
        warp,
    )
    rounded = np.where(
        coordinates >= 0.0,
        np.floor(coordinates + 0.5),
        np.ceil(coordinates - 0.5),
    ).astype(np.int64)
    rounded_in_bounds = np.all(rounded >= 0, axis=0) & np.all(
        rounded < np.asarray(values.shape[:3])[:, None], axis=0
    )
    sampled_source_mask = np.zeros(coordinates.shape[1], dtype=bool)
    sampled_source_mask[rounded_in_bounds] = source_valid[
        tuple(rounded[:, rounded_in_bounds])
    ]
    sampled = np.empty(target_shape + (6,), dtype=np.float32)
    for component in range(6):
        sampled[..., component] = map_coordinates(
            values[..., component],
            coordinates,
            order=1,
            mode="grid-constant",
            cval=0.0,
            prefilter=False,
        ).reshape(target_shape)

    reference_sampling = fsl_voxel_to_scaled_mm(target_shape, reference_affine)
    voxel_axis_directions = np.sign(np.diag(reference_sampling)[:3])
    backward, forward, _, ppd_fold, ppd_near_singular = fsl_warp_jacobians(
        warp,
        singular_tolerance=singular_tolerance,
        voxel_axis_directions=voxel_axis_directions,
    )
    voxel_sizes = np.abs(np.diag(reference_sampling)[:3])
    _, _, determinant, fold, near_singular = fsl_warp_jacobians(
        warp,
        singular_tolerance=singular_tolerance,
        voxel_axis_directions=voxel_axis_directions,
        voxel_sizes=voxel_sizes,
    )
    matrices = _tensor6_to_matrix(sampled.astype(np.float64))
    finite_tensor = np.all(np.isfinite(matrices), axis=(-2, -1))
    eigenvalues = np.full(target_shape + (3,), np.nan, dtype=np.float64)
    eigenvectors = np.full(target_shape + (3, 3), np.nan, dtype=np.float64)
    eigenvalues[finite_tensor], eigenvectors[finite_tensor] = np.linalg.eigh(
        matrices[finite_tensor]
    )
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=-1), 1.0)
    repeated = (
        np.minimum(
            np.abs(eigenvalues[..., 2] - eigenvalues[..., 1]),
            np.abs(eigenvalues[..., 1] - eigenvalues[..., 0]),
        )
        <= repeated_eigenvalue_tolerance * scale
    )
    fa = _axial_fa(eigenvalues)
    low_fa = fa < low_fa_threshold
    rotation, ppd_stable = _ppd_rotation(
        forward, eigenvectors, singular_tolerance=singular_tolerance
    )
    active_support = target_valid & sampled_source_mask.reshape(target_shape)
    unsafe = active_support & (
        ~finite_tensor
        | fold
        | near_singular
        | ppd_fold
        | ppd_near_singular
        | ~ppd_stable
    )
    if compatibility_mode == "strict-fsl" and np.any(unsafe):
        raise ValueError(
            "strict-fsl nonlinear PPD encountered an invalid tensor or Jacobian; "
            "use robust mode only when explicit zero-filling is intended"
        )
    valid = (
        active_support
        & finite_tensor
        & ~fold
        & ~near_singular
        & ~ppd_fold
        & ~ppd_near_singular
        & ppd_stable
    )
    rotated = np.zeros_like(matrices)
    rotated[valid] = np.einsum(
        "...ij,...jk,...lk->...il",
        rotation[valid],
        matrices[valid],
        rotation[valid],
        optimize=True,
    )
    output = _matrix_to_tensor6(rotated).astype(np.float32)
    invalid_tensor = ~finite_tensor | ~np.all(np.isfinite(eigenvalues), axis=-1)
    return NonlinearTensorResult(
        tensor=output,
        valid_mask=valid,
        backward_jacobian=backward,
        forward_jacobian=forward,
        backward_jacobian_determinant=determinant,
        fold_mask=fold,
        near_singular_mask=near_singular,
        low_fa_mask=low_fa,
        repeated_eigenvalue_mask=repeated,
        invalid_tensor_mask=invalid_tensor,
    )


def register_tensor_nonlinear_nifti(
    tensor_file: str | Path,
    reference_file: str | Path,
    displacement_file: str | Path,
    output_file: str | Path,
    *,
    source_mask_file: str | Path | None = None,
    reference_mask_file: str | Path | None = None,
    output_mask_file: str | Path | None = None,
    warp_kind: str = "displacement",
    affine_matrix_file: str | Path | None = None,
    knot_spacing: tuple[int, int, int] | None = None,
    warp_resolution_mm: float = 10.0,
    final_subsampling: int = 2,
    workers: int = 8,
    derivative_base: str | None = None,
    compatibility_mode: str = "strict-fsl",
) -> dict[str, object]:
    """Apply a dense or coefficient FNIRT warp and write tensor derivatives."""

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    started = time.perf_counter()
    tensor_image = nib.load(str(tensor_file))
    reference_image = nib.load(str(reference_file))
    warp_image = nib.load(str(displacement_file))
    if tensor_image.shape[-1:] != (6,):
        raise ValueError("The input tensor must be a 4D six-component NIfTI")
    if len(reference_image.shape) != 3:
        raise ValueError("The reference image must be three-dimensional")
    tensor = np.asarray(tensor_image.dataobj, dtype=np.float32)
    if warp_kind == "displacement":
        if affine_matrix_file is not None:
            raise ValueError(
                "affine_matrix_file is only valid for an FNIRT coefficient warp"
            )
        if warp_image.shape != reference_image.shape + (3,) or not np.allclose(
            warp_image.affine, reference_image.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError("The displacement field must match the reference grid")
        displacement = np.asarray(warp_image.dataobj, dtype=np.float32)
        analytic_jacobian = None
        jacobian_contract = "finite-difference complete displacement"
    elif warp_kind == "coefficients":
        if affine_matrix_file is None:
            raise ValueError(
                "affine_matrix_file is required for an FNIRT coefficient warp"
            )
        from .fnirt import expand_fnirt_coefficients

        coefficients = np.asarray(warp_image.dataobj, dtype=np.float32)
        try:
            affine_matrix = np.loadtxt(affine_matrix_file, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise ValueError("The FNIRT affine matrix could not be read") from error
        expansion = expand_fnirt_coefficients(
            coefficients,
            reference_image.shape,
            reference_image.affine,
            affine_matrix,
            knot_spacing=knot_spacing,
            warp_resolution_mm=warp_resolution_mm,
            final_subsampling=final_subsampling,
        )
        displacement = expansion.displacement.astype(np.float32)
        analytic_jacobian = expansion.nonlinear_jacobian_determinant
        jacobian_contract = "FNIRT analytic nonlinear spline Jacobian"
    else:
        raise ValueError("warp_kind must be 'displacement' or 'coefficients'")
    source_mask = None
    if source_mask_file is not None:
        source_mask_image = nib.load(str(source_mask_file))
        if source_mask_image.shape != tensor_image.shape[:3] or not np.allclose(
            source_mask_image.affine, tensor_image.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError("The source mask must match the tensor grid")
        source_mask = np.asarray(source_mask_image.dataobj) > 0
    reference_mask = None
    if reference_mask_file is not None:
        reference_mask_image = nib.load(str(reference_mask_file))
        if reference_mask_image.shape != reference_image.shape or not np.allclose(
            reference_mask_image.affine,
            reference_image.affine,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError("The reference mask must match the reference grid")
        reference_mask = np.asarray(reference_mask_image.dataobj) > 0
    output_mask = None
    if output_mask_file is not None:
        output_mask_image = nib.load(str(output_mask_file))
        if output_mask_image.shape != reference_image.shape or not np.allclose(
            output_mask_image.affine,
            reference_image.affine,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError("The output mask must match the reference grid")
        output_mask = np.asarray(output_mask_image.dataobj) > 0
    loaded_at = time.perf_counter()
    result = resample_tensor_ppd_fsl(
        tensor,
        tensor_image.affine,
        reference_image.shape,
        reference_image.affine,
        displacement,
        source_mask=source_mask,
        reference_mask=reference_mask,
        compatibility_mode=compatibility_mode,
    )
    output_jacobian = (
        result.backward_jacobian_determinant
        if analytic_jacobian is None
        else analytic_jacobian
    )
    warped_at = time.perf_counter()
    output_tensor = result.tensor
    output_valid_mask = result.valid_mask
    if output_mask is not None:
        # 原始 dwi2cond 在 vecreg 完成后、tensor_decomp 前应用 T1 brain mask。
        output_tensor = result.tensor.copy()
        output_tensor[~output_mask] = 0.0
        output_valid_mask = result.valid_mask & output_mask
    derived = decompose_tensor6(
        output_tensor,
        output_valid_mask,
        requested=("FA", "V1"),
    )
    decomposed_at = time.perf_counter()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    name = output_path.name
    base = (
        derivative_base
        if derivative_base is not None
        else (name[:-7] if name.endswith(".nii.gz") else output_path.stem)
    )
    if not base or Path(base).name != base:
        raise ValueError("derivative_base must be a non-empty file-name stem")
    paths = {
        "tensor": output_path,
        "valid_mask": output_path.with_name(f"{base}_valid_mask.nii.gz"),
        "jacobian": output_path.with_name(f"{base}_jacobian.nii.gz"),
        "fa": output_path.with_name(f"{base}_FA.nii.gz"),
        "v1": output_path.with_name(f"{base}_V1.nii.gz"),
    }

    def save_like(values: np.ndarray, path: Path, dtype: np.dtype) -> None:
        header = reference_image.header.copy()
        header.set_data_dtype(dtype)
        image = nib.Nifti1Image(
            np.asarray(values, dtype=dtype), reference_image.affine, header
        )
        image.set_qform(
            reference_image.get_qform(), int(reference_image.header["qform_code"])
        )
        image.set_sform(
            reference_image.get_sform(), int(reference_image.header["sform_code"])
        )
        nib.save(image, str(path))

    output_values = (
        (output_tensor, paths["tensor"], np.dtype(np.float32)),
        (output_valid_mask, paths["valid_mask"], np.dtype(np.uint8)),
        (
            output_jacobian,
            paths["jacobian"],
            np.dtype(np.float32),
        ),
        (derived["FA"], paths["fa"], np.dtype(np.float32)),
        (derived["V1"], paths["v1"], np.dtype(np.float32)),
    )
    with ThreadPoolExecutor(max_workers=min(workers, len(output_values))) as executor:
        futures = [
            executor.submit(save_like, values, path, dtype)
            for values, path, dtype in output_values
        ]
        for future in futures:
            future.result()
    saved_at = time.perf_counter()
    finite_determinant = output_jacobian[np.isfinite(output_jacobian)]
    qa: dict[str, object] = {
        "status": "completed",
        "mode": "nonlinear",
        "warp_kind": warp_kind,
        "warp_contract": "reference-to-source relative-mm FSL displacement",
        "jacobian_contract": jacobian_contract,
        "tensor_component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
        "reorientation": "FSL vecreg preservation-of-principal-direction",
        "compatibility_mode": compatibility_mode,
        "output_mask_contract": (
            "none"
            if output_mask is None
            else "SimNIBS dwi2cond post-vecreg T1 brain mask"
        ),
        "workers": int(workers),
        "input_shape": list(tensor_image.shape),
        "output_shape": list(output_tensor.shape),
        "valid_voxels": int(np.count_nonzero(output_valid_mask)),
        "fold_voxels": int(np.count_nonzero(result.fold_mask)),
        "near_singular_voxels": int(np.count_nonzero(result.near_singular_mask)),
        "low_fa_voxels": int(np.count_nonzero(result.low_fa_mask & output_valid_mask)),
        "repeated_eigenvalue_voxels": int(
            np.count_nonzero(result.repeated_eigenvalue_mask & output_valid_mask)
        ),
        "invalid_tensor_voxels": int(np.count_nonzero(result.invalid_tensor_mask)),
        "jacobian_determinant_min": (
            None if finite_determinant.size == 0 else float(finite_determinant.min())
        ),
        "jacobian_determinant_max": (
            None if finite_determinant.size == 0 else float(finite_determinant.max())
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
        "stage_seconds": {
            "load": loaded_at - started,
            "warp_resample_ppd": warped_at - loaded_at,
            "tensor_decomposition": decomposed_at - warped_at,
            "parallel_output_write": saved_at - decomposed_at,
        },
        "wall_seconds": saved_at - started,
        "fallback": "none",
    }
    qa_path = output_path.with_name(f"{base}_nonlinear_qa.json")
    temporary = qa_path.with_suffix(qa_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(qa_path)
    return qa


def register_tensor_fnirt_nifti(
    fa_file: str | Path,
    tensor_file: str | Path,
    reference_file: str | Path,
    affine_matrix_file: str | Path,
    output_directory: str | Path,
    *,
    brain_mask_file: str | Path | None = None,
    workers: int = 8,
    compatibility_mode: str = "strict-fsl",
    progress: Callable[[int, str, int, int, float | None], None] | None = None,
) -> dict[str, object]:
    """Estimate the fixed FNIRT warp and run the SimNIBS tensor branch."""

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    from .fnirt import (
        SIMNIBS46_FNIRT_LEVELS,
        FnirtLevelImages,
        run_simnibs46_fnirt,
        warp_fnirt_moving,
    )

    started = time.perf_counter()
    fa_image = nib.load(str(fa_file))
    tensor_image = nib.load(str(tensor_file))
    reference_image = nib.load(str(reference_file))
    if len(fa_image.shape) != 3 or len(reference_image.shape) != 3:
        raise ValueError("FA and reference images must be three-dimensional")
    if tensor_image.shape != fa_image.shape + (6,) or not np.allclose(
        tensor_image.affine, fa_image.affine, rtol=0.0, atol=1e-5
    ):
        raise ValueError("The tensor and FA images must share one source grid")
    try:
        affine_matrix = np.loadtxt(affine_matrix_file, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ValueError("The FNIRT affine matrix could not be read") from error
    if affine_matrix.shape != (4, 4) or not np.all(np.isfinite(affine_matrix)):
        raise ValueError("The FNIRT affine matrix must be finite and 4x4")
    reference = np.asarray(reference_image.dataobj, dtype=np.float32)
    fa = np.asarray(fa_image.dataobj, dtype=np.float32)
    reference_voxel_sizes = tuple(
        float(value) for value in np.linalg.norm(reference_image.affine[:3, :3], axis=0)
    )
    result = run_simnibs46_fnirt(
        reference,
        fa,
        reference_image.affine,
        fa_image.affine,
        affine_matrix,
        workers=workers,
        progress=progress,
    )
    estimated_at = time.perf_counter()
    if progress is not None:
        progress(4, "finalize_write", 0, 3, None)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    coefficient_path = output / "FA2T1_warp.nii.gz"
    field_path = output / "FA2T1_field.nii.gz"
    jacobian_path = output / "FA2T1_jacobian.nii.gz"
    registered_fa_path = output / "DTI_FA_nonlin.nii.gz"
    tensor_path = output / "DTI_coregT1_tensor.nii.gz"

    coefficient_header = nib.Nifti1Header()
    coefficient_header.set_data_dtype(np.float32)
    coefficient_header.set_data_shape(result.coefficients.shape)
    coefficient_header.set_intent(2007, allow_unknown=True)
    coefficient_header["intent_p1"] = reference_voxel_sizes[0]
    coefficient_header["intent_p2"] = reference_voxel_sizes[1]
    coefficient_header["intent_p3"] = reference_voxel_sizes[2]
    coefficient_header["pixdim"][1:4] = result.expansion.knot_spacing
    coefficient_header["qform_code"] = 1
    coefficient_header["quatern_b"] = 0.0
    coefficient_header["quatern_c"] = 0.0
    coefficient_header["quatern_d"] = 0.0
    coefficient_header["qoffset_x"] = reference.shape[0]
    coefficient_header["qoffset_y"] = reference.shape[1]
    coefficient_header["qoffset_z"] = reference.shape[2]
    coefficient_header["sform_code"] = 1
    coefficient_header["srow_x"] = affine_matrix[0]
    coefficient_header["srow_y"] = affine_matrix[1]
    coefficient_header["srow_z"] = affine_matrix[2]
    coefficient_image = nib.Nifti1Image(
        result.coefficients.astype(np.float32),
        None,
        coefficient_header,
    )
    nib.save(coefficient_image, str(coefficient_path))

    def save_reference(values: np.ndarray, path: Path) -> None:
        header = reference_image.header.copy()
        header.set_data_dtype(np.float32)
        image = nib.Nifti1Image(
            np.asarray(values, dtype=np.float32),
            reference_image.affine,
            header,
        )
        image.set_qform(
            reference_image.get_qform(), int(reference_image.header["qform_code"])
        )
        image.set_sform(
            reference_image.get_sform(), int(reference_image.header["sform_code"])
        )
        nib.save(image, str(path))

    full_level = FnirtLevelImages(
        reference=reference,
        moving=fa,
        reference_mask=reference != 0.0,
        moving_mask=fa != 0.0,
        reference_voxel_sizes_mm=reference_voxel_sizes,
    )
    registered_fa = warp_fnirt_moving(
        full_level,
        reference_image.affine,
        fa_image.affine,
        affine_matrix,
        result.expansion.nonlinear_displacement,
        calculate_derivatives=False,
    ).values
    with ThreadPoolExecutor(max_workers=min(workers, 3)) as executor:
        futures = (
            executor.submit(save_reference, result.expansion.displacement, field_path),
            executor.submit(
                save_reference,
                result.expansion.nonlinear_jacobian_determinant,
                jacobian_path,
            ),
            executor.submit(save_reference, registered_fa, registered_fa_path),
        )
        for future in futures:
            future.result()
    written_at = time.perf_counter()
    if progress is not None:
        progress(4, "finalize_tensor", 1, 3, None)

    tensor_qa = register_tensor_nonlinear_nifti(
        tensor_file,
        reference_file,
        coefficient_path,
        tensor_path,
        warp_kind="coefficients",
        affine_matrix_file=affine_matrix_file,
        knot_spacing=result.expansion.knot_spacing,
        output_mask_file=brain_mask_file,
        workers=workers,
        compatibility_mode=compatibility_mode,
        derivative_base="DTI_coregT1",
    )
    completed_at = time.perf_counter()
    if progress is not None:
        progress(4, "finalize_qa", 2, 3, None)
    qa: dict[str, object] = {
        "status": "completed",
        "mode": "nonlinear",
        "algorithm": "SimNIBS 4.6 fixed FNIRT plus FSL vecreg PPD",
        "fallback": "none",
        "compatibility_mode": compatibility_mode,
        "workers": int(workers),
        "displacement_knot_spacing": list(result.expansion.knot_spacing),
        "levels": [
            {
                "level": index,
                "subsampling": specification.subsampling,
                "iterations": level_result.successful_iterations,
                "attempts": len(level_result.trace),
                "cost": level_result.cost,
                "status": level_result.status,
                "jacobian_min": result.jacobian_ranges[index - 1][0],
                "jacobian_max": result.jacobian_ranges[index - 1][1],
            }
            for index, (specification, level_result) in enumerate(
                zip(SIMNIBS46_FNIRT_LEVELS, result.levels, strict=True),
                start=1,
            )
        ],
        "outputs": {
            "coefficients": str(coefficient_path),
            "displacement": str(field_path),
            "fnirt_jacobian": str(jacobian_path),
            "registered_fa": str(registered_fa_path),
            "tensor": tensor_qa["outputs"]["tensor"],
            "tensor_valid_mask": tensor_qa["outputs"]["valid_mask"],
            "tensor_jacobian": tensor_qa["outputs"]["jacobian"],
            "tensor_fa": tensor_qa["outputs"]["fa"],
            "tensor_v1": tensor_qa["outputs"]["v1"],
        },
        "tensor_qa": tensor_qa,
        "stage_seconds": {
            "fnirt_estimation": estimated_at - started,
            "fnirt_output_write": written_at - estimated_at,
            "tensor_resample_ppd": completed_at - written_at,
        },
        "wall_seconds": completed_at - started,
    }
    qa_path = output / "nonlinear_registration_qa.json"
    temporary = qa_path.with_suffix(qa_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(qa_path)
    if progress is not None:
        progress(4, "finalize_complete", 3, 3, None)
    return qa
