"""SimNIBS 4.6 dwi2cond T1 linear-registration pipeline."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from ..registration import register_tensor_affine
from .flirt_registration import FlirtRegistrationResult, register_flirt_affine
from .image_ops import (
    apply_positive_mask,
    binarize_positive,
    edge_strength,
    gaussian_smooth,
    lower_threshold,
    upper_threshold,
)
from .resampling import resample_image
from .tensor_ops import decompose_tensor6
from .transforms import (
    fsl_matrix_to_world,
    fsl_voxel_to_scaled_mm,
    world_matrix_to_fsl,
)


ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class _CharmT1Inputs:
    """Store T1 image objects and computed arrays required for registration."""

    t1: nib.spatialimages.SpatialImage
    corrected: nib.spatialimages.SpatialImage
    brain_mask: np.ndarray
    t1_brain: np.ndarray
    brain_rim: np.ndarray
    reference_weight: np.ndarray


def _save_like(
    values: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output: Path,
    dtype: np.dtype | type[np.generic],
) -> Path:
    """Write an array using the full reference T1 geometry."""

    output.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(values, dtype=dtype), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(output))
    return output


def _same_grid(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
) -> bool:
    """Determine whether two inputs share the same spatial grid."""

    return first.shape[:3] == second.shape[:3] and np.allclose(
        first.affine, second.affine, rtol=0.0, atol=1e-5
    )


def _build_charm_t1_inputs(
    t1_file: str | Path,
    labeling_file: str | Path,
    bias_corrected_file: str | Path,
) -> _CharmT1Inputs:
    """Reproduce T1 brain, reference-weight, and brain-edge QA construction in memory."""

    t1 = nib.load(str(t1_file))
    labeling = nib.load(str(labeling_file))
    corrected = nib.load(str(bias_corrected_file))
    if len(t1.shape) != 3 or len(labeling.shape) != 3 or len(corrected.shape) != 3:
        raise ValueError("T1, CHARM labeling, and bias-corrected T1 must be 3D")
    if not _same_grid(t1, labeling) or not _same_grid(t1, corrected):
        raise ValueError("CHARM T1 inputs must share the same spatial grid")

    labels = np.asarray(labeling.dataobj, dtype=np.float32)
    brain_mask = binarize_positive(upper_threshold(lower_threshold(labels, 1.0), 499.0))
    corrected_values = np.asarray(corrected.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(corrected_values)):
        raise ValueError("Bias-corrected T1 contains NaN or Inf")
    t1_brain = apply_positive_mask(corrected_values, brain_mask)
    voxel_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(t1.affine))
    rim = binarize_positive(lower_threshold(edge_strength(brain_mask, voxel_sizes), 0.3))

    t1_values = np.asarray(t1.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(t1_values)):
        raise ValueError("T1 contains NaN or Inf")
    reference_weight = gaussian_smooth(
        binarize_positive(t1_values), 1.0, voxel_sizes
    )
    return _CharmT1Inputs(
        t1=t1,
        corrected=corrected,
        brain_mask=brain_mask,
        t1_brain=t1_brain,
        brain_rim=rim,
        reference_weight=reference_weight,
    )


def _write_charm_t1_inputs(
    inputs: _CharmT1Inputs, output_directory: str | Path
) -> dict[str, Path]:
    """Write the computed T1 inputs using upstream filenames."""

    output = Path(output_directory)
    return {
        "brain_mask": _save_like(
            inputs.brain_mask,
            inputs.t1,
            output / "T1_brainmask.nii.gz",
            np.uint8,
        ),
        "t1_brain": _save_like(
            inputs.t1_brain,
            inputs.corrected,
            output / "T1_brain.nii.gz",
            np.float32,
        ),
        "brain_rim": _save_like(
            inputs.brain_rim,
            inputs.t1,
            output / "T1_brainrim_QA.nii.gz",
            np.uint8,
        ),
        "reference_weight": _save_like(
            inputs.reference_weight,
            inputs.corrected,
            output / "T1_mask.nii.gz",
            np.float32,
        ),
    }


def prepare_charm_t1_inputs(
    t1_file: str | Path,
    labeling_file: str | Path,
    bias_corrected_file: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Reproduce and write the script's T1 brain, reference weight, and brain-edge QA."""

    return _write_charm_t1_inputs(
        _build_charm_t1_inputs(t1_file, labeling_file, bias_corrected_file),
        output_directory,
    )


def _load_finite_3d(path: str | Path, name: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Read a finite three-dimensional float32 image."""

    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"{name} must be a 3D NIfTI")
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    return image, values


def _estimate(
    reference_image: nib.Nifti1Image,
    reference: np.ndarray,
    reference_weight: np.ndarray,
    moving_image: nib.Nifti1Image,
    moving: np.ndarray,
    *,
    degrees_of_freedom: int,
    workers: int,
    progress: ProgressCallback | None,
) -> FlirtRegistrationResult:
    """Estimate a scaled-mm transform using the FSL default schedule."""

    reference_sampling = fsl_voxel_to_scaled_mm(reference.shape, reference_image.affine)
    moving_sampling = fsl_voxel_to_scaled_mm(moving.shape, moving_image.affine)
    qsform = world_matrix_to_fsl(
        np.eye(4),
        moving.shape,
        moving_image.affine,
        reference.shape,
        reference_image.affine,
    )
    return register_flirt_affine(
        reference,
        moving,
        reference_weight,
        np.ones_like(moving, dtype=np.float32),
        reference_sampling,
        moving_sampling,
        degrees_of_freedom=degrees_of_freedom,
        qsform_matrix=qsform,
        workers=workers,
        progress=progress,
    )


def _resample_qa_images(
    moving: np.ndarray,
    moving_image: nib.Nifti1Image,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    qa_world: np.ndarray,
    sse_file: str | Path | None,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Generate independent FA and optional SSE QA images in parallel."""

    started = time.perf_counter()
    sse_image: nib.Nifti1Image | None = None
    sse: np.ndarray | None = None
    if sse_file is not None:
        sse_image, sse = _load_finite_3d(sse_file, "DTI SSE")
        if not _same_grid(moving_image, sse_image):
            raise ValueError("DTI SSE must match the DTI FA grid")

    def sample(values: np.ndarray, image: nib.Nifti1Image) -> np.ndarray:
        return resample_image(
            values,
            image.affine,
            reference_shape,
            reference_affine,
            qa_world,
            interpolation="linear",
        )

    with ThreadPoolExecutor(max_workers=2 if sse is not None else 1) as executor:
        fa_future = executor.submit(sample, moving, moving_image)
        sse_future = (
            None if sse is None else executor.submit(sample, sse, sse_image)
        )
        return (
            fa_future.result(),
            None if sse_future is None else sse_future.result(),
            time.perf_counter() - started,
        )


def _run_t1_registration_nifti_impl(
    tensor_file: str | Path,
    fa_file: str | Path,
    t1_file: str | Path,
    labeling_file: str | Path,
    bias_corrected_file: str | Path,
    output_directory: str | Path,
    *,
    sse_file: str | Path | None = None,
    degrees_of_freedom: int = 12,
    workers: int = 8,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the SimNIBS 4.6 rigid/affine T1 registration and QA output pipeline."""

    if degrees_of_freedom not in (6, 12):
        raise ValueError("degrees_of_freedom must be 6 or 12")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prepared = prepare_charm_t1_inputs(
        t1_file, labeling_file, bias_corrected_file, output
    )
    prepared_at = time.perf_counter()
    if progress is not None:
        progress("prepare_t1", 1, 1)

    reference_image, reference = _load_finite_3d(prepared["t1_brain"], "T1 brain")
    _, reference_weight = _load_finite_3d(prepared["reference_weight"], "T1 weight")
    moving_image, moving = _load_finite_3d(fa_file, "DTI FA")
    estimate_arguments = (
        reference_image,
        reference,
        reference_weight,
        moving_image,
        moving,
    )
    registration_started = time.perf_counter()
    if degrees_of_freedom == 6:
        primary = _estimate(
            *estimate_arguments,
            degrees_of_freedom=6,
            workers=workers,
            progress=(
                None
                if progress is None
                else lambda stage, done, total: progress(
                    f"primary_{stage}", done, total
                )
            ),
        )
        qa_registration = primary
    else:
        # The two FLIRT runs have no data dependency, so execute their schedules in parallel.
        with ThreadPoolExecutor(max_workers=2) as registration_executor:
            primary_future = registration_executor.submit(
                _estimate,
                *estimate_arguments,
                degrees_of_freedom=12,
                workers=workers,
                progress=None,
            )
            qa_future = registration_executor.submit(
                _estimate,
                *estimate_arguments,
                degrees_of_freedom=6,
                workers=workers,
                progress=None,
            )
            primary = primary_future.result()
            qa_registration = qa_future.result()
        if progress is not None:
            progress("primary_complete", 1, 1)
            progress("qa_registration_complete", 1, 1)
    registered_at = time.perf_counter()
    primary_world = fsl_matrix_to_world(
        primary.matrix,
        moving.shape,
        moving_image.affine,
        reference.shape,
        reference_image.affine,
    )
    np.savetxt(output / "FA2T1.mat", primary.matrix, fmt="%.10g")
    np.savetxt(output / "FA2T1_world.mat", primary_world, fmt="%.17g")

    qa_world = fsl_matrix_to_world(
        qa_registration.matrix,
        moving.shape,
        moving_image.affine,
        reference.shape,
        reference_image.affine,
    )
    np.savetxt(output / "FA2T1_QA.mat", qa_registration.matrix, fmt="%.10g")

    tensor_output = output / "DTI_coregT1_tensor.nii.gz"
    tensor_metrics: dict[str, object] = {}

    def consume_registered_tensor(tensor: np.ndarray, valid: np.ndarray) -> None:
        """Decompose the same read-only memory array while the main tensor is compressed."""

        decomposition_started = time.perf_counter()
        decomposed, eigenvalue_range = decompose_tensor6(
            tensor,
            valid,
            requested=("FA", "V1"),
            return_eigenvalue_range=True,
        )
        decomposed_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as output_executor:
            output_futures = [
                output_executor.submit(
                    _save_like,
                    decomposed[suffix],
                    reference_image,
                    output / f"DTI_coregT1_{suffix}.nii.gz",
                    np.float32,
                )
                for suffix in ("FA", "V1")
            ]
            for future in output_futures:
                future.result()
        tensor_metrics.update(
            {
                "decomposition_seconds": decomposed_at - decomposition_started,
                "derived_write_seconds": time.perf_counter() - decomposed_at,
                "eigenvalue_range": eigenvalue_range,
                "valid_voxels": int(np.count_nonzero(valid)),
                "finite_tensor_components": int(np.count_nonzero(np.isfinite(tensor))),
            }
        )

    postprocess_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as qa_branch_executor:
        qa_images_future = qa_branch_executor.submit(
            _resample_qa_images,
            moving,
            moving_image,
            reference.shape,
            reference_image.affine,
            qa_world,
            sse_file,
        )
        tensor_registration_started = time.perf_counter()
        register_tensor_affine(
            tensor_file,
            prepared["t1_brain"],
            tensor_output,
            world_transform=primary_world,
            source_mask_mode="fsl-vecreg",
            reference_mask_file=prepared["brain_mask"],
            output_valid_mask_file=output / "DTI_coregT1_valid_mask.nii.gz",
            qa_file=output / "DTI_coregT1_tensor_registration_qa.json",
            interpolation_order=1,
            reorientation_transform=primary.matrix,
            workers=workers,
            alignment_assumption=f"automatic_flirt_{degrees_of_freedom}dof",
            prepared_consumer=consume_registered_tensor,
        )
        tensor_registered_at = time.perf_counter()
        derived_saved_at = tensor_registered_at
        registered_fa, registered_sse, qa_resample_seconds = qa_images_future.result()
        qa_collected_at = time.perf_counter()
    if progress is not None:
        progress("tensor", 1, 1)

    brain_mask = np.asarray(nib.load(str(prepared["brain_mask"])).dataobj) > 0
    registered_fa[~brain_mask] = 0.0
    qa_outputs = [
        (
            registered_fa,
            output / "DTI_FA_6dof_QA.nii.gz",
        )
    ]
    if registered_sse is not None:
        registered_sse[~brain_mask] = 0.0
        qa_outputs.append(
            (
                registered_sse,
                output / "DTI_SSE_6dof_QA.nii.gz",
            )
        )
    with ThreadPoolExecutor(max_workers=len(qa_outputs)) as output_executor:
        output_futures = [
            output_executor.submit(
                _save_like,
                values,
                reference_image,
                path,
                np.float32,
            )
            for values, path in qa_outputs
        ]
        for future in output_futures:
            future.result()
    qa_saved_at = time.perf_counter()
    if progress is not None:
        progress("qa", 1, 1)

    report: dict[str, object] = {
        "status": "completed",
        "mode": "rigid" if degrees_of_freedom == 6 else "affine",
        "workers": workers,
        "primary": {
            "fsl_scaled_mm_matrix": primary.matrix.tolist(),
            "world_matrix": primary_world.tolist(),
            "cost": primary.cost,
            "evaluations": primary.evaluations,
            "candidate_count": primary.candidate_count,
        },
        "qa_6dof": {
            "fsl_scaled_mm_matrix": qa_registration.matrix.tolist(),
            "world_matrix": qa_world.tolist(),
            "cost": qa_registration.cost,
            "evaluations": qa_registration.evaluations,
        },
        "output_grid_shape": list(reference.shape),
        "valid_voxels": tensor_metrics["valid_voxels"],
        "finite_tensor_components": tensor_metrics["finite_tensor_components"],
        "symmetric_tensor": True,
        "eigenvalue_min": tensor_metrics["eigenvalue_range"][0],
        "eigenvalue_max": tensor_metrics["eigenvalue_range"][1],
        "stage_seconds": {
            "prepare_t1": prepared_at - started,
            "input_load_before_registration": registration_started - prepared_at,
            "dual_flirt_registration": registered_at - registration_started,
            "matrix_write_and_setup": postprocess_started - registered_at,
            "tensor_registration": tensor_registered_at - tensor_registration_started,
            "tensor_load_and_decomposition": tensor_metrics["decomposition_seconds"],
            "derived_fa_v1_gzip_write": tensor_metrics["derived_write_seconds"],
            "qa_resampling_parallel_branch": qa_resample_seconds,
            "qa_branch_tail_wait": qa_collected_at - derived_saved_at,
            "qa_mask_and_gzip_write": qa_saved_at - qa_collected_at,
            "postprocess_critical_path": qa_saved_at - postprocess_started,
        },
        "wall_seconds": qa_saved_at - started,
        "fallback": "none",
    }
    temporary = output / ".t1_registration_qa.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output / "t1_registration_qa.json")
    return report


def run_t1_registration_nifti(
    tensor_file: str | Path,
    fa_file: str | Path,
    t1_file: str | Path,
    labeling_file: str | Path,
    bias_corrected_file: str | Path,
    output_directory: str | Path,
    *,
    sse_file: str | Path | None = None,
    degrees_of_freedom: int = 12,
    workers: int = 8,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the T1 registration pipeline with explicit candidate parallelism."""

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return _run_t1_registration_nifti_impl(
        tensor_file,
        fa_file,
        t1_file,
        labeling_file,
        bias_corrected_file,
        output_directory,
        sse_file=sse_file,
        degrees_of_freedom=degrees_of_freedom,
        workers=int(workers),
        progress=progress,
    )
