"""SimNIBS 4.6 ``nomoco`` raw-DWI pipeline."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
from time import perf_counter

import nibabel as nib
import numpy as np

from ..nifti_fit import fit_dti_nifti
from ..gradients import load_gradients
from .brain_mask import write_bet_brain_mask
from .orientation import (
    fsl_canonical_orientation, reorient_bvecs_fsl,
    validate_fsl_raw_sampling, write_fsl_reoriented,
)
from .rigid import write_aligned_b0_mean


NomocoProgress = Callable[[str, int, int], None]
_IDENTITY_ORIENTATION = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])


def _can_reuse_raw_preparation(data_file: str | Path) -> bool:
    """仅复用本来就需要重新物化的输入，保留直接 mmap 路径的原始 header。"""

    source = Path(data_file)
    image = nib.load(str(source), mmap=True)
    return not (
        source.suffix.lower() == ".nii"
        and image.get_data_dtype() == np.dtype(np.float32)
        and np.array_equal(
            fsl_canonical_orientation(image.affine), _IDENTITY_ORIENTATION
        )
    )


def _prepare_fitting_input(
    data_file: str | Path,
    materialized_file: Path,
    *,
    z_chunk: int = 8,
) -> tuple[Path, dict[str, object]]:
    """Validate a raw DWI that can be read directly, applying only a storage-order rearrangement when necessary."""

    source = Path(data_file)
    image = nib.load(str(source), mmap=True)
    if len(image.shape) != 4:
        raise ValueError("DWI data must be a four-dimensional NIfTI")
    validate_fsl_raw_sampling(image)
    if z_chunk <= 0:
        raise ValueError("The validation z-chunk must be a positive integer")

    uncompressed = source.suffix.lower() == ".nii"
    float32 = image.get_data_dtype() == np.dtype(np.float32)
    orientation = fsl_canonical_orientation(image.affine)
    orientation_identity = bool(np.array_equal(orientation, _IDENTITY_ORIENTATION))
    validation: dict[str, object] = {
        "source_compression": "none" if uncompressed else "compressed",
        "source_dtype": str(image.get_data_dtype()),
        "orientation_identity": orientation_identity,
        "finite": None,
        "nonnegative": None,
    }
    if uncompressed and float32 and orientation_identity:
        finite = True
        nonnegative = True
        for z_start in range(0, image.shape[2], z_chunk):
            z_stop = min(z_start + z_chunk, image.shape[2])
            block = np.asanyarray(
                image.dataobj[:, :, z_start:z_stop, :], dtype=np.float32
            )
            if not np.all(np.isfinite(block)):
                finite = False
                break
            if np.any(block < 0):
                nonnegative = False
        validation["finite"] = finite
        validation["nonnegative"] = nonnegative
        if finite:
            if source.resolve() != materialized_file.resolve():
                shutil.copyfile(source, materialized_file)
            validation.update(
                {
                    "strategy": "validated_input_mmap",
                    "materialized": True,
                    "materialization": "byte_copy",
                    "validation_z_chunk": z_chunk,
                }
            )
            return materialized_file, validation

    write_fsl_reoriented(
        source,
        materialized_file,
        float32=True,
        nonnegative=False,
    )
    validation.update(
        {
            "strategy": "single_decode_materialization",
            "materialized": True,
            "validation_z_chunk": z_chunk,
        }
    )
    return materialized_file, validation


def _prepare_bvecs(
    data_file: str | Path,
    bvals_file: str | Path,
    bvecs_file: str | Path,
    output_file: Path,
    *,
    component_mode: str = "physical",
) -> np.ndarray:
    """同步 DWI 存储重排的 FSL 梯度基；恒等转换保留原文件字节。"""

    if component_mode not in ("physical", "simnibs46-literal"):
        raise ValueError("component_mode must be physical or simnibs46-literal")
    _, original = load_gradients(bvals_file, bvecs_file)
    transformed = (
        reorient_bvecs_fsl(original, nib.load(str(data_file)).affine)
        if component_mode == "physical" else original
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if np.array_equal(transformed, original):
        if Path(bvecs_file).resolve() != output_file.resolve():
            shutil.copyfile(bvecs_file, output_file)
    else:
        np.savetxt(output_file, transformed.T, fmt="%.17g")
    return transformed


def _write_masked_brain(
    image_file: Path, mask_file: Path, output_file: Path
) -> None:
    """Write a brain-extracted b0 image using FSL ``-mas`` semantics."""

    image = nib.load(str(image_file))
    mask = nib.load(str(mask_file))
    values = np.asarray(image.dataobj, dtype=np.float32)
    mask_values = np.asarray(mask.dataobj) > 0.0
    output_values = np.where(mask_values, values, 0.0).astype(np.float32)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(output_values, image.affine, header)
    output.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output.set_sform(image.get_sform(), int(image.header["sform_code"]))
    nib.save(output, str(output_file))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write the pipeline JSON atomically."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_nomoco_nifti(
    data_file: str | Path,
    bvals_file: str | Path,
    bvecs_file: str | Path,
    output_directory: str | Path,
    *,
    grad_dev_file: str | Path | None = None,
    shell: float | None = None,
    tolerance: float = 100.0,
    b0_threshold: float = 0.0,
    z_chunk: int = 4,
    voxel_batch: int = 4096,
    workers: int = 8,
    compatibility_mode: str = "strict-fsl",
    component_mode: str = "physical",
    bet_backend: str = "optimized",
    progress: NomocoProgress | None = None,
    _prepared_directory: Path | None = None,
) -> dict[str, object]:
    """Execute the raw-DWI path without motion/eddy correction in SimNIBS 4.6 order.

    b0 registration is used only to construct the reference mean and brain mask;
    the diffusion volumes themselves receive no motion, eddy, or fieldmap correction.
    The original b0, BET, and registration stages preserve negative values; clamp to zero only at the fitting boundary after all corrections are complete.
    """

    if workers <= 0:
        raise ValueError("The worker count must be a positive integer")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "materialized": output / "DWIraw.nii",
        "dwi_for_fit": output / "DWIforfit.nii",
        "bvals": output / "DWIbvals",
        "bvecs": output / "DWIbvecs",
        "nodif": output / "nodif.nii.gz",
        "nodif_mask": output / "nodif_brain_mask.nii.gz",
        "nodif_brain": output / "nodif_brain.nii.gz",
        "b0_qa": output / "nodif_qa.json",
        "fit_base": output / "DTI.nii.gz",
        "tensor": output / "DTI_tensor.nii.gz",
        "valid_mask": output / "DTI_valid_mask.nii.gz",
        "fit_qa": output / "DTI_qa.json",
        "qa": output / "nomoco_qa.json",
    }
    normalized_grad_dev = None
    if grad_dev_file is not None:
        normalized_grad_dev = output / "grad_dev.nii"
    stage_seconds: dict[str, float] = {}

    preparation = None
    if _prepared_directory is not None:
        preparation = json.loads(
            (_prepared_directory / "preparation_qa.json").read_text(encoding="utf-8")
        )
        expected = {
            "status": "completed",
            "workers": workers,
            "b0_threshold": b0_threshold,
            "component_mode": component_mode,
            "bet_backend": bet_backend,
            "bet_fractional_threshold": 0.2,
            "data": str(Path(data_file).resolve()),
            "bvals": str(Path(bvals_file).resolve()),
            "bvecs": str(Path(bvecs_file).resolve()),
            "grad_dev": (
                None if grad_dev_file is None else str(Path(grad_dev_file).resolve())
            ),
        }
        if (
            not _can_reuse_raw_preparation(data_file)
            or any(preparation.get(key) != value for key, value in expected.items())
        ):
            raise ValueError("Raw QA preparation does not match the nomoco inputs and parameters")

    started = perf_counter()
    if preparation is None:
        fitting_input, input_strategy = _prepare_fitting_input(
            data_file,
            paths["materialized"],
            z_chunk=max(8, z_chunk),
        )
        _prepare_bvecs(data_file, bvals_file, bvecs_file, paths["bvecs"], component_mode=component_mode)
        if normalized_grad_dev is not None:
            write_fsl_reoriented(
                grad_dev_file,
                normalized_grad_dev,
                float32=True,
                component_kind="grad-dev",
                component_mode=component_mode,
            )
    else:
        # producer 已通过 DAG 依赖与文件指纹检查；独立复制避免后续写入别名。
        fitting_input = paths["materialized"]
        shutil.copyfile(_prepared_directory / "DWIraw.nii", fitting_input)
        shutil.copyfile(_prepared_directory / "DWIbvecs", paths["bvecs"])
        if normalized_grad_dev is not None:
            shutil.copyfile(_prepared_directory / "grad_dev.nii", normalized_grad_dev)
        input_strategy = {
            "strategy": "reused_raw_qa_preparation",
            "materialized": True,
            "materialization": "byte_copy",
            "producer": "fit_raw_dti_qa",
        }
    shutil.copyfile(bvals_file, paths["bvals"])
    stage_seconds["normalize_input"] = perf_counter() - started
    if progress is not None:
        progress("normalize_input", 1, 1)

    started = perf_counter()
    if preparation is None:
        write_aligned_b0_mean(
            fitting_input,
            paths["bvals"],
            paths["nodif"],
            b0_threshold=b0_threshold,
            workers=workers,
            progress=(
                None
                if progress is None
                else lambda done, total: progress("align_b0", done, total)
            ),
            qa_file=paths["b0_qa"],
        )
    else:
        shutil.copyfile(_prepared_directory / "nodif.nii.gz", paths["nodif"])
        shutil.copyfile(_prepared_directory / "nodif_qa.json", paths["b0_qa"])
        if progress is not None:
            progress("align_b0", 1, 1)
    stage_seconds["align_b0"] = perf_counter() - started

    started = perf_counter()
    if preparation is None:
        bet_result = write_bet_brain_mask(
            paths["nodif"],
            paths["nodif_mask"],
            workers=workers,
            backend=bet_backend,
        )
        bet_report = {
            "backend": bet_backend,
            "passes": bet_result.passes,
            "mask_voxels": int(np.count_nonzero(bet_result.mask)),
        }
    else:
        shutil.copyfile(_prepared_directory / "nodif_brain_mask.nii.gz", paths["nodif_mask"])
        bet_report = preparation["bet"]
    _write_masked_brain(paths["nodif"], paths["nodif_mask"], paths["nodif_brain"])
    stage_seconds["brain_mask"] = perf_counter() - started
    if progress is not None:
        progress("brain_mask", 1, 1)

    started = perf_counter()
    write_fsl_reoriented(
        fitting_input,
        paths["dwi_for_fit"],
        float32=True,
        nonnegative=True,
    )
    fit_dti_nifti(
        paths["dwi_for_fit"],
        paths["bvals"],
        paths["bvecs"],
        paths["nodif_mask"],
        paths["fit_base"],
        grad_dev_file=normalized_grad_dev,
        shell=shell,
        tolerance=tolerance,
        b0_threshold=b0_threshold,
        z_chunk=z_chunk,
        voxel_batch=voxel_batch,
        workers=workers,
        compatibility_mode=compatibility_mode,
        progress=(
            None
            if progress is None
            else lambda done, total, _z: progress("fit_dti", done, total)
        ),
        valid_mask_file=paths["valid_mask"],
        qa_file=paths["fit_qa"],
    )
    paths["fit_base"].replace(paths["tensor"])
    stage_seconds["fit_dti"] = perf_counter() - started

    fit_report = json.loads(paths["fit_qa"].read_text(encoding="utf-8"))
    artifacts = {
        "bvals": paths["bvals"].name,
        "bvecs": paths["bvecs"].name,
        "nodif": paths["nodif"].name,
        "nodif_brain_mask": paths["nodif_mask"].name,
        "nodif_brain": paths["nodif_brain"].name,
        "tensor": paths["tensor"].name,
        "fa": Path(fit_report["derived_outputs"]["FA"]).name,
        "sse": Path(fit_report["derived_outputs"]["sse"]).name,
        "valid_mask": paths["valid_mask"].name,
        "b0_qa": paths["b0_qa"].name,
        "fit_qa": paths["fit_qa"].name,
    }
    artifacts["dwi_for_fit"] = paths["dwi_for_fit"].name
    artifacts["raw_reoriented"] = paths["materialized"].name
    if normalized_grad_dev is not None:
        artifacts["grad_dev"] = normalized_grad_dev.name
    report: dict[str, object] = {
        "status": "completed",
        "mode": "nomoco",
        "algorithm_contract": "simnibs-4.6-nomoco",
        "correction": {
            "motion": "not_applied_to_dwi",
            "eddy_current": "not_applied",
            "susceptibility": "not_applied",
        },
        "workers": workers,
        "shell": shell,
        "shell_tolerance": tolerance,
        "b0_threshold": b0_threshold,
        "compatibility_mode": compatibility_mode,
        "component_mode": component_mode,
        "fitting_input": input_strategy,
        "stage_seconds": stage_seconds,
        "total_seconds": sum(stage_seconds.values()),
        "bet": bet_report,
        "fit": {
            "masked_voxels": fit_report["masked_voxels"],
            "valid_fitted_voxels": fit_report["valid_fitted_voxels"],
        },
        "artifacts": artifacts,
    }
    _write_json(paths["qa"], report)
    return report
