"""Cross-platform spatial mapping and reorientation of diffusion tensors."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from .preprocessing.transforms import world_matrix_to_fsl


def tensor6_to_matrix(tensor: np.ndarray) -> np.ndarray:
    """Convert the FSL six-component order to symmetric 3x3 tensors."""
    tensor = np.asarray(tensor)
    if tensor.shape[-1] != 6:
        raise ValueError("The final tensor axis must contain six components")
    matrix = np.empty(tensor.shape[:-1] + (3, 3), dtype=tensor.dtype)
    matrix[..., 0, 0] = tensor[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = tensor[..., 1]
    matrix[..., 0, 2] = matrix[..., 2, 0] = tensor[..., 2]
    matrix[..., 1, 1] = tensor[..., 3]
    matrix[..., 1, 2] = matrix[..., 2, 1] = tensor[..., 4]
    matrix[..., 2, 2] = tensor[..., 5]
    return matrix


def matrix_to_tensor6(matrix: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 tensors to the FSL six-component order."""
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("The final two matrix axes must be 3x3")
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


def polar_rotation(linear: np.ndarray) -> np.ndarray:
    """Extract the affine linear part's orthogonal polar factor, matching vecreg."""
    linear = np.asarray(linear, dtype=np.float64)
    if linear.shape != (3, 3) or not np.all(np.isfinite(linear)):
        raise ValueError("The affine linear part must be a finite 3x3 matrix")
    if abs(np.linalg.det(linear)) < 1e-12:
        raise ValueError("The affine linear part is singular; tensor reorientation is undefined")
    u, _, vt = np.linalg.svd(linear)
    return u @ vt


def reorient_affine_tensor(tensor: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    """Apply ``R D R^T`` using the affine rotation component."""
    transform = np.asarray(world_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_transform must be 4x4")
    rotation = polar_rotation(transform[:3, :3])
    matrix = tensor6_to_matrix(np.asarray(tensor, dtype=np.float64))
    rotated = np.einsum("ij,...jk,lk->...il", rotation, matrix, rotation, optimize=True)
    return matrix_to_tensor6(rotated)


def register_tensor_affine(
    tensor_file: str | Path,
    reference_file: str | Path,
    output_file: str | Path,
    *,
    world_transform: np.ndarray | None = None,
    source_mask_file: str | Path | None = None,
    source_mask_mode: str | None = None,
    reference_mask_file: str | Path | None = None,
    output_valid_mask_file: str | Path | None = None,
    qa_file: str | Path | None = None,
    interpolation_order: int = 1,
    reorientation_transform: np.ndarray | None = None,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
    alignment_assumption: str | None = None,
    prepared_consumer: Callable[[np.ndarray, np.ndarray], None] | None = None,
) -> Path:
    """Affinely map a six-component tensor and apply finite-strain reorientation.

    ``world_transform`` maps input world coordinates to reference world
    coordinates. Without it, resampling uses only the two NIfTI affines and is
    valid only for data already aligned in physical space.
    """
    if interpolation_order not in (0, 1, 3):
        raise ValueError("interpolation_order must be 0, 1, or 3")
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if source_mask_mode not in (None, "fsl-vecreg"):
        raise ValueError("source_mask_mode must be None or 'fsl-vecreg'")
    if source_mask_file is not None and source_mask_mode is not None:
        raise ValueError("source_mask_file and source_mask_mode are mutually exclusive")
    profile_started = time.perf_counter()
    tensor_img = nib.load(str(tensor_file))
    reference_img = nib.load(str(reference_file))
    if tensor_img.shape[-1:] != (6,):
        raise ValueError("The input tensor must be a four-dimensional six-component NIfTI")
    transform = np.eye(4) if world_transform is None else np.asarray(world_transform, float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world_transform must be a finite 4x4 matrix")

    # scipy affine_transform consumes the inverse map from output to input voxels.
    output_to_input = (
        np.linalg.inv(tensor_img.affine)
        @ np.linalg.inv(transform)
        @ reference_img.affine
    )
    source = np.asarray(tensor_img.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(source)):
        raise ValueError("The input tensor contains NaN or Inf")
    target_shape = reference_img.shape[:3]
    sampled = np.empty(target_shape + (6,), dtype=np.float32)
    loaded_at = time.perf_counter()

    def sample_component(component: int) -> int:
        affine_transform(
            source[..., component],
            output_to_input[:3, :3],
            offset=output_to_input[:3, 3],
            output=sampled[..., component],
            output_shape=target_shape,
            order=interpolation_order,
            mode="grid-constant" if interpolation_order == 1 else "constant",
            cval=0.0,
            prefilter=interpolation_order > 1,
        )
        return component

    source_mask: np.ndarray | None = None
    if source_mask_file is not None or source_mask_mode is not None:
        if source_mask_mode == "fsl-vecreg":
            source_mask = (
                (source[..., 0] != 0)
                | (source[..., 1] != 0)
                | (source[..., 2] != 0)
            )
        else:
            source_mask_img = nib.load(str(source_mask_file))
            source_mask = np.asarray(source_mask_img.dataobj) != 0
            if not np.allclose(
                source_mask_img.affine,
                tensor_img.affine,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "The source mask must match the tensor grid exactly"
                )
        if source_mask.shape != tensor_img.shape[:3]:
            raise ValueError("The source-mask shape does not match the tensor")

    def sample_source_mask() -> np.ndarray | None:
        if source_mask is None:
            return None
        return affine_transform(
            source_mask.astype(np.uint8),
            output_to_input[:3, :3],
            offset=output_to_input[:3, 3],
            output_shape=target_shape,
            order=0,
            mode="grid-constant",
            cval=0,
            prefilter=False,
        ) > 0

    if workers == 1:
        for completed, component in enumerate(range(6), start=1):
            sample_component(component)
            if progress is not None:
                progress(completed, 8)
        sampled_source_mask = sample_source_mask()
    else:
        task_count = 6 + int(source_mask is not None)
        with ThreadPoolExecutor(max_workers=min(workers, task_count)) as executor:
            component_futures = [
                executor.submit(sample_component, component) for component in range(6)
            ]
            source_mask_future = (
                None if source_mask is None else executor.submit(sample_source_mask)
            )
            for completed, future in enumerate(component_futures, start=1):
                future.result()
                if progress is not None:
                    progress(completed, 8)
            sampled_source_mask = (
                None if source_mask_future is None else source_mask_future.result()
            )
    resampled_at = time.perf_counter()
    orientation_transform = (
        world_matrix_to_fsl(
            transform,
            tensor_img.shape[:3],
            tensor_img.affine,
            reference_img.shape[:3],
            reference_img.affine,
        )
        if reorientation_transform is None
        else np.asarray(reorientation_transform, dtype=np.float64)
    )
    if orientation_transform.shape != (4, 4) or not np.all(
        np.isfinite(orientation_transform)
    ):
        raise ValueError("reorientation_transform must be a finite 4x4 matrix")
    rotation = polar_rotation(orientation_transform[:3, :3])
    if not np.allclose(rotation, np.eye(3), rtol=0, atol=1e-12):
        # Whole-head 3x3 expansion needs several GiB; z-blocking bounds memory use.
        rotation32 = rotation.astype(np.float32)
        for z0 in range(0, target_shape[2], 8):
            z1 = min(z0 + 8, target_shape[2])
            matrices = tensor6_to_matrix(sampled[:, :, z0:z1])
            rotated = np.einsum(
                "ij,...jk,lk->...il",
                rotation32,
                matrices,
                rotation32,
                optimize=True,
            )
            sampled[:, :, z0:z1] = matrix_to_tensor6(rotated)
    reoriented_at = time.perf_counter()
    if progress is not None:
        progress(7, 8)

    valid = np.all(np.isfinite(sampled), axis=-1) & np.any(sampled != 0, axis=-1)
    if sampled_source_mask is not None:
        valid &= sampled_source_mask
    if reference_mask_file is not None:
        reference_mask_img = nib.load(str(reference_mask_file))
        reference_mask = np.asarray(reference_mask_img.dataobj) != 0
        if reference_mask.shape != target_shape or not np.allclose(
            reference_mask_img.affine, reference_img.affine
        ):
            raise ValueError("The reference mask must match the reference grid exactly")
        valid &= reference_mask
    sampled[~valid] = 0
    masked_at = time.perf_counter()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    output_img = nib.Nifti1Image(sampled, reference_img.affine, header)
    output_img.set_qform(reference_img.get_qform(), int(reference_img.header["qform_code"]))
    output_img.set_sform(reference_img.get_sform(), int(reference_img.header["sform_code"]))

    name = output_path.name
    base = name[:-7] if name.endswith(".nii.gz") else output_path.stem
    valid_path = (
        Path(output_valid_mask_file)
        if output_valid_mask_file is not None
        else output_path.with_name(f"{base}_valid_mask.nii.gz")
    )
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    valid_header = reference_img.header.copy()
    valid_header.set_data_dtype(np.uint8)
    valid_image = nib.Nifti1Image(
        valid.astype(np.uint8), reference_img.affine, valid_header
    )

    def timed_save(image: nib.spatialimages.SpatialImage, path: Path) -> float:
        write_started = time.perf_counter()
        nib.save(image, str(path))
        return time.perf_counter() - write_started

    # The main tensor, valid mask, and read-only downstream consumers have no data
    # dependencies, so parallel execution can hide gzip wall time.
    sampled.setflags(write=False)
    valid.setflags(write=False)
    consumer_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as output_executor:
        tensor_future = output_executor.submit(timed_save, output_img, output_path)
        valid_future = output_executor.submit(timed_save, valid_image, valid_path)
        if prepared_consumer is not None:
            prepared_consumer(sampled, valid)
        consumer_seconds = time.perf_counter() - consumer_started
        tensor_write_seconds = tensor_future.result()
        valid_write_seconds = valid_future.result()
    outputs_completed_at = time.perf_counter()
    if progress is not None:
        progress(8, 8)

    qa_path = (
        Path(qa_file)
        if qa_file is not None
        else output_path.with_name(f"{base}_registration_qa.json")
    )
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa = {
        "input_shape": list(tensor_img.shape),
        "output_shape": list(sampled.shape),
        "valid_voxels": int(np.count_nonzero(valid)),
        "world_transform": transform.tolist(),
        "reorientation_transform": orientation_transform.tolist(),
        "polar_rotation": rotation.tolist(),
        "output_to_input_voxel": output_to_input.tolist(),
        "interpolation_order": interpolation_order,
        "workers": workers,
        "tensor_component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
        "valid_mask": str(valid_path),
        "alignment_assumption": alignment_assumption,
        "source_mask_mode": source_mask_mode,
        "stage_seconds": {
            "load": loaded_at - profile_started,
            "component_resampling": resampled_at - loaded_at,
            "tensor_reorientation": reoriented_at - resampled_at,
            "masking": masked_at - reoriented_at,
            "tensor_gzip_write_parallel": tensor_write_seconds,
            "valid_mask_gzip_write_parallel": valid_write_seconds,
            "prepared_consumer_parallel": consumer_seconds,
            "output_parallel_critical_path": outputs_completed_at - masked_at,
            "total_before_qa_json": outputs_completed_at - profile_started,
        },
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def make_charm_brain_mask(
    labeling_file: str | Path,
    output_file: str | Path,
    *,
    reference_file: str | Path | None = None,
) -> Path:
    """Create a CHARM brain mask from the official dwi2cond label range 1..499."""
    labeling_img = nib.load(str(labeling_file))
    labeling = np.asarray(labeling_img.dataobj)
    mask = ((labeling >= 1) & (labeling <= 499)).astype(np.uint8)
    reference_img = labeling_img
    if reference_file is not None:
        reference_img = nib.load(str(reference_file))
        if reference_img.shape[:3] != labeling_img.shape[:3] or not np.allclose(
            reference_img.affine, labeling_img.affine
        ):
            raise ValueError("CHARM labeling does not match the reference T1 grid")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(mask, reference_img.affine, header)
    image.set_qform(reference_img.get_qform(), int(reference_img.header["qform_code"]))
    image.set_sform(reference_img.get_sform(), int(reference_img.header["sform_code"]))
    nib.save(image, str(output_path))
    return output_path
