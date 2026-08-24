"""SimNIBS 4.6 legacy two-pass DWI correction pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import sys
from pathlib import Path
import shutil
from time import perf_counter

import nibabel as nib
import numpy as np
from scipy.linalg import polar

from ..gradients import load_gradients
from ..nifti_fit import fit_dti_nifti
from ._numba import set_available_numba_threads
from .brain_mask import write_bet_brain_mask
from .flirt_registration import register_flirt_nosearch_mutual_information
from .nomoco import _prepare_fitting_input, _write_masked_brain
from .orientation import write_fsl_reoriented
from .resampling import resample_image
from .rigid import (
    _intensity_center_scaled_mm,
    _isotropic_resample,
    _optimize_one_stage,
    write_aligned_b0_mean,
)
from .transforms import fsl_matrix_to_world


LegacyProgress = Callable[[str, int, int], None]


def _optimize_stage_payload(
    payload: tuple[
        int,
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        float,
        np.ndarray,
        int,
    ],
) -> tuple[int, np.ndarray, float, int]:
    """Run an MCFLIRT stage without cross-volume dependencies in a separate worker."""

    position, fixed, moving, spacing, initial, multiplier, center, dof = payload
    matrix, cost, count = _optimize_one_stage(
        fixed,
        moving,
        spacing,
        initial,
        multiplier,
        center,
        dof,
    )
    return position, matrix, cost, count


def _float32_mean(volumes: Sequence[np.ndarray], indices: np.ndarray) -> np.ndarray:
    """Compute an FSL-style float32 volume mean in input order."""

    if indices.size == 0:
        raise ValueError("At least one diffusion-weighted volume is required")
    total = np.zeros(volumes[0].shape, dtype=np.float32)
    for index in indices:
        total += volumes[int(index)]
    return np.asarray(total / np.float32(indices.size), dtype=np.float32)


def _register_mcflirt_series(
    volumes: Sequence[np.ndarray],
    reference: np.ndarray,
    affine: np.ndarray,
    *,
    degrees_of_freedom: int,
    workers: int,
    stages_mm: Sequence[float] = (8.0, 4.0, 4.0),
    max_evaluations: int = 1200,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Reproduce MCFLIRT's three-stage per-volume optimization with an external reference."""

    if degrees_of_freedom not in (6, 12):
        raise ValueError("Legacy MCFLIRT degrees of freedom must be 6 or 12")
    if workers <= 0:
        raise ValueError("The worker count must be a positive integer")
    if not stages_mm or any(value <= 0 for value in stages_mm):
        raise ValueError("Registration stages must contain positive spacings")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    if not volumes:
        raise ValueError("The DWI must contain at least one volume")
    spatial_shape = reference.shape
    if any(volume.shape != spatial_shape for volume in volumes):
        raise ValueError("All DWI volumes and the reference must share one grid")
    if not np.all(np.isfinite(reference)) or any(
        not np.all(np.isfinite(volume)) for volume in volumes
    ):
        raise ValueError("Legacy registration inputs must contain only finite values")

    voxel_sizes = nib.affines.voxel_sizes(affine)
    matrices = [np.eye(4, dtype=np.float64) for _ in volumes]
    evaluations = [0 for _ in volumes]
    costs = [0.0 for _ in volumes]
    total = len(stages_mm) * len(volumes)
    done = 0
    reference_cache: dict[float, np.ndarray] = {}
    volume_cache: dict[float, list[np.ndarray]] = {}
    center_cache: dict[float, list[np.ndarray]] = {}
    multipliers = (0.8, 0.8, 0.1)
    for stage_index, spacing in enumerate(stages_mm):
        key = float(spacing)
        if key not in reference_cache:
            reference_cache[key] = _isotropic_resample(reference, voxel_sizes, spacing)
            volume_cache[key] = [
                _isotropic_resample(volume, voxel_sizes, spacing) for volume in volumes
            ]
            center_cache[key] = [
                _intensity_center_scaled_mm(volume, spacing)
                for volume in volume_cache[key]
            ]
        fixed = reference_cache[key]
        coarse = volume_cache[key]
        centers = center_cache[key]
        stage_output = [matrix.copy() for matrix in matrices]

        def optimize(position: int) -> tuple[int, np.ndarray, float, int]:
            matrix, cost, count = _optimize_one_stage(
                fixed,
                coarse[position],
                spacing,
                matrices[position],
                multipliers[min(stage_index, 2)],
                centers[position],
                degrees_of_freedom,
            )
            return position, matrix, cost, count

        positions = list(range(len(volumes)))
        if stage_index == 0 or workers == 1:
            results = map(optimize, positions)
            executor = None
        elif sys.platform.startswith("linux"):
            executor = ProcessPoolExecutor(max_workers=min(workers, len(volumes)))
            payloads = [
                (
                    position,
                    fixed,
                    coarse[position],
                    spacing,
                    matrices[position],
                    multipliers[min(stage_index, 2)],
                    centers[position],
                    degrees_of_freedom,
                )
                for position in positions
            ]
            results = executor.map(_optimize_stage_payload, payloads)
        else:
            # Spawn entry points keep Windows and macOS cross-platform safe; Linux
            # benchmarks use forked processes to bypass the GIL.
            executor = ThreadPoolExecutor(max_workers=min(workers, len(volumes)))
            results = executor.map(optimize, positions)
        try:
            for position, matrix, cost, count in results:
                stage_output[position] = matrix
                costs[position] = cost
                evaluations[position] += count
                if evaluations[position] > max_evaluations:
                    raise RuntimeError(
                        f"Legacy registration exceeded the evaluation limit for DWI volume {position}"
                    )
                # MCFLIRT propagates the current result to the next volume only in
                # the first 8 mm stage; the source boundary condition preserves an
                # independent identity starting point for the last volume.
                if stage_index == 0 and position + 1 < len(volumes) - 1:
                    matrices[position + 1] = matrix.copy()
                done += 1
                if progress is not None:
                    progress(done, total)
        finally:
            if executor is not None:
                executor.shutdown()
        matrices = stage_output
    return matrices, evaluations, costs


def _resample_series(
    volumes: Sequence[np.ndarray],
    affine: np.ndarray,
    fsl_matrices: Sequence[np.ndarray],
    *,
    interpolation: str,
    workers: int,
    displacement: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    """Resample in volume order; matrix parallelism preserves each volume's numeric order."""

    def sample(item: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
        index, volume = item
        world = fsl_matrix_to_world(
            fsl_matrices[index], volume.shape, affine, volume.shape, affine
        )
        return index, resample_image(
            volume,
            affine,
            volume.shape,
            affine,
            world,
            interpolation=interpolation,
            reference_to_moving_displacement=displacement,
        )

    items = list(enumerate(volumes))
    if interpolation == "sinc":
        # The sinc kernel is already parallel over output points; keep the outer
        # volume loop sequential to avoid nested thread pools.
        set_available_numba_threads(workers)
        results = map(sample, items)
        executor = None
    elif workers == 1:
        results = map(sample, items)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(items)))
        results = executor.map(sample, items)
    ordered: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in items]
    try:
        for done, (index, volume) in enumerate(results, start=1):
            ordered[index] = volume
            if progress is not None:
                progress(done, len(items))
    finally:
        if executor is not None:
            executor.shutdown()
    return ordered


def _rotate_bvecs(
    bvals: np.ndarray, bvecs: np.ndarray, world_matrices: Sequence[np.ndarray]
) -> np.ndarray:
    """Generate explicit corrected b-vectors using the final affine's finite-strain rotation."""

    rotated = np.asarray(bvecs, dtype=np.float64).copy()
    for index, (bvalue, matrix) in enumerate(zip(bvals, world_matrices, strict=True)):
        if bvalue == 0:
            continue
        rotation, _ = polar(np.asarray(matrix, dtype=np.float64)[:3, :3])
        vector = rotation @ rotated[index]
        length = np.linalg.norm(vector)
        if not np.isfinite(length) or length <= 0:
            raise RuntimeError(f"Cannot rotate the b-vector for DWI volume {index}")
        rotated[index] = vector / length
    return rotated


def _save_nifti(values: np.ndarray, template: nib.spatialimages.SpatialImage, path: Path) -> None:
    """Save a float32 NIfTI image using the template geometry."""

    header = template.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), template.affine, header)
    image.set_qform(template.get_qform(), int(template.header["qform_code"]))
    image.set_sform(template.get_sform(), int(template.header["sform_code"]))
    nib.save(image, str(path))


def _load_displacement(
    displacement_file: str | Path, template: nib.spatialimages.SpatialImage
) -> np.ndarray:
    """Read an xyz moving-world displacement aligned with the DWI reference grid."""

    image = nib.load(str(displacement_file))
    displacement = np.asarray(image.dataobj, dtype=np.float64)
    if displacement.shape != template.shape[:3] + (3,):
        raise ValueError("The fieldmap displacement must match the DWI grid and end in xyz")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("The fieldmap displacement must contain only finite values")
    if not np.allclose(image.affine, template.affine, rtol=0.0, atol=1e-5):
        raise ValueError("The fieldmap displacement affine must match the DWI grid")
    return displacement


def run_legacy_nifti(
    data_file: str | Path,
    bvals_file: str | Path,
    bvecs_file: str | Path,
    output_directory: str | Path,
    *,
    grad_dev_file: str | Path | None = None,
    bvec_mode: str = "compat46",
    fieldmap_displacement_file: str | Path | None = None,
    shell: float = 1000.0,
    tolerance: float = 100.0,
    z_chunk: int = 4,
    voxel_batch: int = 4096,
    workers: int = 8,
    bet_backend: str = "optimized",
    max_evaluations: int = 1200,
    progress: LegacyProgress | None = None,
) -> dict[str, object]:
    """Execute SimNIBS 4.6 legacy two-pass correction, single resampling, and WLS fitting."""

    if bvec_mode not in ("compat46", "corrected"):
        raise ValueError("bvec_mode must be compat46 or corrected")
    if workers <= 0:
        raise ValueError("The worker count must be a positive integer")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "materialized": output / "DWIraw.nii",
        "bvals": output / "DWIbvals",
        "bvecs": output / "DWIbvecs",
        "nodif": output / "nodif.nii.gz",
        "mask": output / "nodif_brain_mask.nii.gz",
        "brain": output / "nodif_brain.nii.gz",
        "corrected": output / "DWI_corr.nii",
        "dwi_for_fit": output / "DWIforfit.nii",
        "mean": output / "DWI_corr_mean.nii.gz",
        "transforms": output / "DWI_corr.mat",
        "mean_transform": output / "meanDWI2nodif.mat",
        "fit": output / "DTI.nii.gz",
        "tensor": output / "DTI_tensor.nii.gz",
        "valid": output / "DTI_valid_mask.nii.gz",
        "fit_qa": output / "DTI_qa.json",
        "qa": output / "legacy_qa.json",
    }
    started_total = perf_counter()
    stage_seconds: dict[str, float] = {}
    started = perf_counter()
    fitting_input, input_strategy = _prepare_fitting_input(
        data_file, paths["materialized"], z_chunk=max(8, z_chunk)
    )
    shutil.copyfile(bvals_file, paths["bvals"])
    bvals, original_bvecs = load_gradients(bvals_file, bvecs_file)
    image = nib.load(str(fitting_input), mmap=True)
    if bvals.size != image.shape[3]:
        raise ValueError("The DWI fourth axis does not match bvals")
    b0_indices = np.flatnonzero(bvals == 0)
    diffusion_indices = np.flatnonzero(bvals > 0)
    if b0_indices.size == 0:
        raise ValueError("Legacy compat46 mode requires at least one exact b=0 volume")
    if diffusion_indices.size == 0:
        raise ValueError("Legacy compat46 mode requires at least one b>0 volume")
    volumes = [
        np.asarray(image.dataobj[..., index], dtype=np.float32)
        for index in range(image.shape[3])
    ]
    if any(not np.all(np.isfinite(volume)) for volume in volumes):
        raise ValueError("The DWI contains NaN or Inf")
    stage_seconds["normalize_input"] = perf_counter() - started
    if progress is not None:
        progress("normalize_input", 1, 1)

    started = perf_counter()
    write_aligned_b0_mean(
        fitting_input,
        paths["bvals"],
        paths["nodif"],
        b0_threshold=0.0,
        workers=workers,
        progress=(None if progress is None else lambda done, total: progress("align_b0", done, total)),
    )
    write_bet_brain_mask(paths["nodif"], paths["mask"], workers=workers, backend=bet_backend)
    _write_masked_brain(paths["nodif"], paths["mask"], paths["brain"])
    nodif = np.asarray(nib.load(str(paths["nodif"])).dataobj, dtype=np.float32)
    stage_seconds["nodif_and_mask"] = perf_counter() - started

    started = perf_counter()
    raw_mean = _float32_mean(volumes, diffusion_indices)
    pass1, pass1_evaluations, pass1_costs = _register_mcflirt_series(
        volumes,
        raw_mean,
        image.affine,
        degrees_of_freedom=6,
        workers=workers,
        max_evaluations=max_evaluations,
        progress=(None if progress is None else lambda done, total: progress("legacy_pass1_6dof", done, total)),
    )
    diffusion_volumes = [volumes[int(index)] for index in diffusion_indices]
    pass1_images = _resample_series(
        diffusion_volumes,
        image.affine,
        [pass1[int(index)] for index in diffusion_indices],
        interpolation="linear",
        workers=workers,
    )
    pass1_mean = _float32_mean(
        pass1_images, np.arange(len(pass1_images), dtype=np.int64)
    )
    stage_seconds["pass1_6dof"] = perf_counter() - started

    started = perf_counter()
    pass2, pass2_evaluations, pass2_costs = _register_mcflirt_series(
        volumes,
        pass1_mean,
        image.affine,
        degrees_of_freedom=12,
        workers=workers,
        max_evaluations=max_evaluations,
        progress=(None if progress is None else lambda done, total: progress("legacy_pass2_12dof", done, total)),
    )
    pass2_images = _resample_series(
        diffusion_volumes,
        image.affine,
        [pass2[int(index)] for index in diffusion_indices],
        interpolation="sinc",
        workers=workers,
    )
    corrected_mean_pre_nodif = _float32_mean(
        pass2_images, np.arange(len(pass2_images), dtype=np.int64)
    )
    stage_seconds["pass2_12dof"] = perf_counter() - started

    started = perf_counter()
    sampling = np.diag([*nib.affines.voxel_sizes(image.affine), 1.0])
    mean_registration = register_flirt_nosearch_mutual_information(
        nodif,
        corrected_mean_pre_nodif,
        sampling,
        sampling,
        degrees_of_freedom=12,
        workers=workers,
    )
    np.savetxt(paths["mean_transform"], mean_registration.matrix, fmt="%.10g")
    final_matrices = [mean_registration.matrix @ matrix for matrix in pass2]
    direct_prefix_count = int(b0_indices[-1]) + 1
    direct_b0, b0_evaluations, b0_costs = _register_mcflirt_series(
        volumes[:direct_prefix_count],
        nodif,
        image.affine,
        degrees_of_freedom=6,
        workers=workers,
        max_evaluations=max_evaluations,
        progress=(None if progress is None else lambda done, total: progress("direct_b0_6dof", done, total)),
    )
    for index in b0_indices:
        final_matrices[int(index)] = direct_b0[int(index)]
    stage_seconds["compose_transforms"] = perf_counter() - started

    displacement = None
    if fieldmap_displacement_file is not None:
        displacement = _load_displacement(fieldmap_displacement_file, image)

    started = perf_counter()
    corrected_volumes = _resample_series(
        volumes,
        image.affine,
        final_matrices,
        interpolation="sinc",
        workers=workers,
        displacement=displacement,
        progress=(None if progress is None else lambda done, total: progress("final_resample", done, total)),
    )
    corrected = np.stack(corrected_volumes, axis=3)
    corrected_mean = _float32_mean(corrected_volumes, diffusion_indices)
    _save_nifti(corrected, image, paths["corrected"])
    _save_nifti(np.maximum(corrected, np.float32(0.0)), image, paths["dwi_for_fit"])
    _save_nifti(corrected_mean, image, paths["mean"])
    paths["transforms"].mkdir(exist_ok=True)
    world_matrices = []
    for index, matrix in enumerate(final_matrices):
        np.savetxt(paths["transforms"] / f"MAT_{index:04d}", matrix, fmt="%.10g")
        world_matrices.append(
            fsl_matrix_to_world(matrix, image.shape[:3], image.affine, image.shape[:3], image.affine)
        )
    output_bvecs = (
        original_bvecs
        if bvec_mode == "compat46"
        else _rotate_bvecs(bvals, original_bvecs, world_matrices)
    )
    if bvec_mode == "compat46":
        shutil.copyfile(bvecs_file, paths["bvecs"])
    else:
        np.savetxt(paths["bvecs"], output_bvecs.T, fmt="%.10g")
    stage_seconds["final_single_resample"] = perf_counter() - started

    normalized_grad_dev = None
    if grad_dev_file is not None:
        normalized_grad_dev = output / "grad_dev.nii"
        write_fsl_reoriented(grad_dev_file, normalized_grad_dev, float32=True)
    started = perf_counter()
    fit_dti_nifti(
        paths["dwi_for_fit"],
        paths["bvals"],
        paths["bvecs"],
        paths["mask"],
        paths["fit"],
        grad_dev_file=normalized_grad_dev,
        shell=shell,
        tolerance=tolerance,
        b0_threshold=0.0,
        z_chunk=z_chunk,
        voxel_batch=voxel_batch,
        workers=workers,
        progress=(None if progress is None else lambda done, total, _z: progress("fit_dti", done, total)),
        valid_mask_file=paths["valid"],
        qa_file=paths["fit_qa"],
    )
    paths["fit"].replace(paths["tensor"])
    stage_seconds["fit_dti"] = perf_counter() - started
    fit_report = json.loads(paths["fit_qa"].read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "status": "completed",
        "mode": "legacy",
        "algorithm_contract": "simnibs-4.6-compat46",
        "bvec_mode": bvec_mode,
        "bvec_contract": "copied_original" if bvec_mode == "compat46" else "finite_strain_rotated",
        "workers": workers,
        "input_shape": list(image.shape),
        "b0_indices": b0_indices.tolist(),
        "diffusion_indices": diffusion_indices.tolist(),
        "fitting_input": input_strategy,
        "interpolation": {
            "formal_output_passes_per_volume": 1,
            "formal_output_kernel": "fsl_hanning_sinc_width_7",
            "pass1_mean_only": "trilinear",
            "pass2_mean_only": "sinc",
            "fieldmap_composed_before_sampling": fieldmap_displacement_file is not None,
        },
        "registration": {
            "pass1_dof": 6,
            "pass2_dof": 12,
            "direct_b0_dof": 6,
            "mean_to_nodif": "flirt_nosearch_mutualinfo_12dof",
            "pass1_evaluations": pass1_evaluations,
            "pass2_evaluations": pass2_evaluations,
            "direct_b0_evaluations": b0_evaluations,
            "direct_b0_evaluated_prefix_count": direct_prefix_count,
            "pass1_costs": pass1_costs,
            "pass2_costs": pass2_costs,
            "direct_b0_costs": b0_costs,
            "mean_to_nodif_cost": mean_registration.cost,
            "mean_to_nodif_evaluations": mean_registration.evaluations,
        },
        "stage_seconds": stage_seconds,
        "wall_seconds": perf_counter() - started_total,
        "artifacts": {
            "corrected_dwi": paths["corrected"].name,
            "dwi_for_fit": paths["dwi_for_fit"].name,
            "corrected_mean": paths["mean"].name,
            "transforms": paths["transforms"].name,
            "mean_to_nodif_transform": paths["mean_transform"].name,
            "bvals": paths["bvals"].name,
            "bvecs": paths["bvecs"].name,
            "nodif": paths["nodif"].name,
            "nodif_brain_mask": paths["mask"].name,
            "tensor": paths["tensor"].name,
            "fa": Path(fit_report["derived_outputs"]["FA"]).name,
            "sse": Path(fit_report["derived_outputs"]["sse"]).name,
            "valid_mask": paths["valid"].name,
        },
    }
    paths["qa"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
