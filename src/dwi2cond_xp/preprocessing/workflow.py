"""Assemble fixed preprocessing, T1 registration, unified QA, and FEM smoke into an explicit DAG."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from collections.abc import Callable

import nibabel as nib
import numpy as np

from .. import __version__
from ..nifti_fit import fit_dti_nifti
from ..simulation import run_tdcs
from .eddy import run_eddy_nifti
from .legacy import run_legacy_nifti
from .nomoco import run_nomoco_nifti
from .nonlinear import register_tensor_fnirt_nifti
from .orientation import write_fsl_reoriented
from .pipeline import ArtifactContract, PipelineRunner, StageDefinition, StageRunResult
from .qa import PipelineQaInputs, build_pipeline_qa
from .t1_registration import run_t1_registration_nifti
from .tensor_ops import decompose_tensor6
from .topup_eddy import run_topup_eddy_nifti


WorkflowProgress = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class Dwi2CondPipelineConfig:
    """Freeze every explicit input and mode required by one complete compatible path."""

    data: Path | None
    bvals: Path | None
    bvecs: Path | None
    m2m_directory: Path
    output_directory: Path
    prefit_tensor: Path | None = None
    preprocessing_mode: str = "legacy"
    t1_mode: str = "nonlinear"
    grad_dev: Path | None = None
    dwi_brain_mask: Path | None = None
    reverse_phase_encoding: Path | None = None
    susceptibility_field: Path | None = None
    fieldmap_corrected_mask: Path | None = None
    fieldmap_magnitude: Path | None = None
    fieldmap_radians_per_second: Path | None = None
    fieldmap_dwell_milliseconds: float | None = None
    readout_seconds: float | None = None
    phase_encoding_direction: str | None = None
    random_seed: int = 1
    workers: int = 8
    fit_compatibility_mode: str = "strict-fsl"
    publish_to_m2m: bool = True
    fem_smoke: str = "none"
    solver: str = "pardiso"


@dataclass(frozen=True)
class Dwi2CondPipelineResult:
    """Return all stage manifests, the final tensor, and aggregated QA."""

    stages: tuple[StageRunResult, ...]
    final_tensor: Path
    qa_manifest: Path


def _required_m2m_files(m2m: Path, *, require_fem: bool = False) -> dict[str, Path]:
    """按实际启用阶段解析 CHARM 输入，避免无 FEM 时要求无关文件。"""

    subject = m2m.name.removeprefix("m2m_")
    mesh = m2m / f"{subject}.msh"
    if not mesh.is_file():
        candidates = sorted(m2m.glob("*.msh"))
        if len(candidates) == 1:
            mesh = candidates[0]
    eeg = m2m / "eeg_positions" / "EEG10-10_UI_Jurak_2007.csv"
    if not eeg.is_file():
        candidates = sorted((m2m / "eeg_positions").glob("*10-10*.csv"))
        if candidates:
            eeg = candidates[0]
    files: dict[str, Path] = {
        "t1": m2m / "T1.nii.gz",
        "labeling": m2m / "segmentation" / "labeling.nii.gz",
        "bias_corrected": m2m / "segmentation" / "T1_bias_corrected.nii.gz",
    }
    if require_fem:
        files.update(
            {
                "final_tissues": m2m / "final_tissues.nii.gz",
                "mesh": mesh,
                "eeg": eeg,
            }
        )
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required CHARM pipeline inputs: " + ", ".join(missing)
        )
    return files


def _validate_config(config: Dwi2CondPipelineConfig) -> dict[str, Path]:
    """Reject invalid combinations once before expensive computation."""

    if config.preprocessing_mode not in ("prefit", "nomoco", "legacy", "eddy"):
        raise ValueError("preprocessing_mode must be prefit, nomoco, legacy, or eddy")
    if config.t1_mode not in ("rigid", "affine", "nonlinear"):
        raise ValueError("t1_mode must be rigid, affine, or nonlinear")
    if config.fem_smoke not in ("none", "dry-run", "run"):
        raise ValueError("fem_smoke must be none, dry-run, or run")
    if config.workers < 1:
        raise ValueError("workers must be positive")
    if config.fit_compatibility_mode not in ("strict-fsl", "robust"):
        raise ValueError("fit_compatibility_mode must be strict-fsl or robust")
    if config.preprocessing_mode == "prefit":
        if config.prefit_tensor is None or not config.prefit_tensor.is_file():
            raise FileNotFoundError(f"Missing pre-fitted tensor: {config.prefit_tensor}")
        if any(
            value is not None
            for value in (
                config.data,
                config.bvals,
                config.bvecs,
                config.grad_dev,
                config.dwi_brain_mask,
                config.reverse_phase_encoding,
                config.susceptibility_field,
                config.fieldmap_corrected_mask,
                config.fieldmap_magnitude,
                config.fieldmap_radians_per_second,
                config.fieldmap_dwell_milliseconds,
                config.readout_seconds,
                config.phase_encoding_direction,
            )
        ):
            raise ValueError("prefit mode cannot be combined with raw-DWI inputs")
    else:
        for path in (config.data, config.bvals, config.bvecs):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing pipeline input: {path}")
    if config.preprocessing_mode == "eddy":
        if config.reverse_phase_encoding is None:
            if config.dwi_brain_mask is None or not config.dwi_brain_mask.is_file():
                raise ValueError(
                    "eddy mode requires dwi_brain_mask unless reverse PE is provided"
                )
        elif not config.reverse_phase_encoding.is_file():
            raise FileNotFoundError(
                f"Missing reverse phase-encoding input: {config.reverse_phase_encoding}"
            )
        if config.readout_seconds is None or config.phase_encoding_direction is None:
            raise ValueError("eddy mode requires readout_seconds and PE direction")
        if config.reverse_phase_encoding is not None:
            if config.susceptibility_field is not None:
                raise ValueError(
                    "reverse PE TOPUP and an external susceptibility field are mutually exclusive"
                )
            if config.phase_encoding_direction not in ("x", "x-", "y", "y-"):
                raise ValueError("the fixed TOPUP subset supports x/x-/y/y- only")
    elif config.susceptibility_field is not None:
        if config.preprocessing_mode != "legacy":
            raise ValueError("a prepared susceptibility field is only valid for legacy or eddy")
    if config.susceptibility_field is not None and not config.susceptibility_field.is_file():
        raise FileNotFoundError(
            f"Missing susceptibility field: {config.susceptibility_field}"
        )
    if (
        config.fieldmap_corrected_mask is not None
        and not config.fieldmap_corrected_mask.is_file()
    ):
        raise FileNotFoundError(
            f"Missing corrected fieldmap mask: {config.fieldmap_corrected_mask}"
        )
    raw_fieldmap_values = (
        config.fieldmap_magnitude,
        config.fieldmap_radians_per_second,
        config.fieldmap_dwell_milliseconds,
    )
    raw_fieldmap = any(value is not None for value in raw_fieldmap_values)
    if raw_fieldmap and not all(value is not None for value in raw_fieldmap_values):
        raise ValueError(
            "raw fieldmap requires magnitude, radians-per-second field, and dwell"
        )
    if raw_fieldmap:
        if config.preprocessing_mode != "legacy":
            raise ValueError("raw fieldmap inputs require legacy preprocessing")
        if config.phase_encoding_direction is None:
            raise ValueError("raw fieldmap inputs require a PE direction")
        if config.susceptibility_field is not None or config.fieldmap_corrected_mask is not None:
            raise ValueError("raw and prepared fieldmap inputs are mutually exclusive")
        if float(config.fieldmap_dwell_milliseconds) <= 0.0:
            raise ValueError("fieldmap dwell must be positive")
        for path in (config.fieldmap_magnitude, config.fieldmap_radians_per_second):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing raw fieldmap input: {path}")
    if config.fieldmap_corrected_mask is not None:
        if config.preprocessing_mode != "legacy" or config.susceptibility_field is None:
            raise ValueError(
                "fieldmap_corrected_mask requires legacy mode and a susceptibility field"
            )
    if config.preprocessing_mode == "legacy" and config.susceptibility_field is not None:
        if config.fieldmap_corrected_mask is None:
            raise ValueError(
                "legacy fieldmap correction requires the corrected fieldmap mask"
            )
    return _required_m2m_files(
        config.m2m_directory, require_fem=config.fem_smoke != "none"
    )


def _write_nonnegative_nifti(source: Path, destination: Path) -> Path:
    """在所有校正结束后执行与 ``fslmaths -thr 0`` 相同的截零。"""

    image = nib.load(str(source))
    values = np.asarray(image.dataobj, dtype=np.float32)
    np.maximum(values, np.float32(0.0), out=values)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(values, image.affine, header)
    output.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output.set_sform(image.get_sform(), int(image.header["sform_code"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(destination))
    return destination


def _sha256(path: Path) -> str:
    """分块计算发布产物哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_tensor_to_m2m(source: Path, m2m: Path, version: str) -> dict[str, object]:
    """原子发布官方默认发现路径，并写入来源与哈希。"""

    destination = m2m / "DTI_coregT1_tensor.nii.gz"
    temporary = m2m / ".DTI_coregT1_tensor.nii.gz.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    provenance = m2m / "DTI_coregT1_tensor.provenance.json"
    payload: dict[str, object] = {
        "status": "completed",
        "implementation": "dwi2cond-xp",
        "version": version,
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "sha256": _sha256(destination),
    }
    temporary_json = m2m / ".DTI_coregT1_tensor.provenance.json.tmp"
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(provenance)
    return payload


def _save_array_like(
    values: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_file: Path,
    dtype: np.dtype,
) -> Path:
    """保存与参考图像完全一致的 qform/sform 几何。"""

    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(values, dtype=dtype), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(output_file))
    return output_file


def _import_prefit_tensor(source: Path, output: Path) -> dict[str, object]:
    """复现官方 pre-fitted tensor 的 copy、reorient 与 tensor_decomp。"""

    output.mkdir(parents=True, exist_ok=True)
    tensor_file = write_fsl_reoriented(
        source,
        output / "DTI_tensor.nii.gz",
        float32=True,
        nonnegative=False,
    )
    image = nib.load(str(tensor_file))
    tensor = np.asarray(image.dataobj, dtype=np.float32)
    if tensor.ndim != 4 or tensor.shape[3] != 6:
        raise ValueError("pre-fitted tensor must have shape XxYxZx6")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("pre-fitted tensor contains NaN or Inf")
    decomposition = decompose_tensor6(tensor, semantics="fslmaths")
    _save_array_like(decomposition["FA"], image, output / "DTI_FA.nii.gz", np.float32)
    valid = decomposition["L1"] > 0
    _save_array_like(
        valid.astype(np.uint8),
        image,
        output / "DTI_valid_mask.nii.gz",
        np.uint8,
    )
    report: dict[str, object] = {
        "status": "completed",
        "mode": "prefit",
        "algorithm_contract": "SimNIBS-4.6-prefit-fslmaths-tensor_decomp",
        "input": str(source.resolve()),
        "tensor": str(tensor_file.resolve()),
        "valid_voxels": int(np.count_nonzero(valid)),
    }
    (output / "prefit_qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_dwi2cond_pipeline(
    config: Dwi2CondPipelineConfig,
    *,
    progress: WorkflowProgress | None = None,
) -> Dwi2CondPipelineResult:
    """Run the full DAG without switching to reference or another mode on optimized failure."""

    m2m = _validate_config(config)
    root = config.output_directory.resolve()
    preprocess = root / "preprocess"
    registration = root / "registration"
    nonlinear = root / "nonlinear"
    qa_directory = root / "qa"
    fem_root = root / "fem"
    preprocess.mkdir(parents=True, exist_ok=True)
    stages: list[StageDefinition] = []
    version = str(__version__)

    def subprogress(stage: str) -> Callable[[str, int, int], None] | None:
        if progress is None:
            return None

        def report(phase: str, done: int, total: int) -> None:
            progress(f"{stage}:{phase}", done, total, "running")

        return report

    if config.preprocessing_mode == "prefit":
        preprocess_name = "import_prefit_tensor"
        dti_directory = preprocess / "prefit"

        def run_preprocess() -> dict[str, object]:
            return _import_prefit_tensor(config.prefit_tensor, dti_directory)

        preprocess_outputs = (
            ArtifactContract(
                dti_directory / "DTI_tensor.nii.gz", "nifti", ndim=4, final_axis=6
            ),
            ArtifactContract(dti_directory / "DTI_FA.nii.gz", "nifti", ndim=3),
            ArtifactContract(
                dti_directory / "DTI_valid_mask.nii.gz", "nifti", ndim=3
            ),
            ArtifactContract(dti_directory / "prefit_qa.json", "json"),
        )
        corrected_dwi = None
        fit_bvals = None
        fit_bvecs = None
        dwi_mask = dti_directory / "DTI_valid_mask.nii.gz"
    elif config.preprocessing_mode == "nomoco":
        preprocess_name = "preprocess_nomoco"

        def run_preprocess() -> dict[str, object]:
            report = run_nomoco_nifti(
                config.data,
                config.bvals,
                config.bvecs,
                preprocess,
                grad_dev_file=config.grad_dev,
                workers=config.workers,
                compatibility_mode=config.fit_compatibility_mode,
                progress=subprogress(preprocess_name),
            )
            return {"status": report["status"], "mode": report["mode"]}

        preprocess_outputs = (
            ArtifactContract(preprocess / "DTI_tensor.nii.gz", "nifti", ndim=4, final_axis=6),
            ArtifactContract(preprocess / "DTI_FA.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "DTI_sse.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "DTI_valid_mask.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "nodif_brain_mask.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "nomoco_qa.json", "json"),
        )
        corrected_dwi = config.data
        fit_bvals = preprocess / "DWIbvals"
        fit_bvecs = preprocess / "DWIbvecs"
        dwi_mask = preprocess / "nodif_brain_mask.nii.gz"
    elif config.preprocessing_mode == "legacy":
        preprocess_name = "preprocess_legacy"

        def run_preprocess() -> dict[str, object]:
            report = run_legacy_nifti(
                config.data,
                config.bvals,
                config.bvecs,
                preprocess,
                grad_dev_file=config.grad_dev,
                fieldmap_displacement_file=config.susceptibility_field,
                fieldmap_corrected_mask_file=config.fieldmap_corrected_mask,
                fieldmap_magnitude_file=config.fieldmap_magnitude,
                fieldmap_radians_per_second_file=config.fieldmap_radians_per_second,
                fieldmap_dwell_milliseconds=config.fieldmap_dwell_milliseconds,
                fieldmap_phase_encoding_direction=config.phase_encoding_direction,
                workers=config.workers,
                compatibility_mode=config.fit_compatibility_mode,
                progress=subprogress(preprocess_name),
            )
            return {"status": report["status"], "mode": report["mode"]}

        preprocess_outputs = (
            ArtifactContract(preprocess / "DWI_corr.nii", "nifti", ndim=4),
            ArtifactContract(preprocess / "DTI_tensor.nii.gz", "nifti", ndim=4, final_axis=6),
            ArtifactContract(preprocess / "DTI_FA.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "DTI_sse.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "DTI_valid_mask.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "nodif_brain_mask.nii.gz", "nifti", ndim=3),
            ArtifactContract(preprocess / "legacy_qa.json", "json"),
        )
        if config.fieldmap_magnitude is not None:
            preprocess_outputs += (
                ArtifactContract(
                    preprocess / "fieldmap" / "displacement_world_mm.nii.gz",
                    "nifti",
                    ndim=4,
                    final_axis=3,
                ),
                ArtifactContract(
                    preprocess / "fieldmap" / "corrected_mask.nii.gz",
                    "nifti",
                    ndim=3,
                ),
                ArtifactContract(
                    preprocess / "fieldmap" / "fieldmap_qa.json", "json"
                ),
            )
        corrected_dwi = preprocess / "DWI_corr.nii"
        fit_bvals = preprocess / "DWIbvals"
        fit_bvecs = preprocess / "DWIbvecs"
        dwi_mask = preprocess / "nodif_brain_mask.nii.gz"
    else:
        preprocess_name = (
            "preprocess_topup_eddy"
            if config.reverse_phase_encoding is not None
            else "preprocess_eddy"
        )
        eddy_directory = preprocess / "eddy"
        dti_directory = preprocess / "dti"

        def run_preprocess() -> dict[str, object]:
            if config.reverse_phase_encoding is not None:
                report = run_topup_eddy_nifti(
                    config.data,
                    config.bvals,
                    config.bvecs,
                    config.reverse_phase_encoding,
                    preprocess,
                    readout_seconds=float(config.readout_seconds),
                    phase_encoding_direction=str(config.phase_encoding_direction),
                    random_seed=config.random_seed,
                    workers=config.workers,
                    progress=subprogress(preprocess_name),
                )
            else:
                report = run_eddy_nifti(
                    config.data,
                    config.bvals,
                    config.bvecs,
                    config.dwi_brain_mask,
                    eddy_directory,
                    readout_seconds=float(config.readout_seconds),
                    phase_encoding_direction=str(config.phase_encoding_direction),
                    susceptibility_field_file=config.susceptibility_field,
                    random_seed=config.random_seed,
                    workers=config.workers,
                    progress=subprogress(preprocess_name),
                )
            return {"status": report["status"], "algorithm": report["algorithm"]}

        eddy_outputs = [
            ArtifactContract(eddy_directory / "DWIraw.nii", "nifti", ndim=4),
            ArtifactContract(eddy_directory / "nodif_brain_mask.nii.gz", "nifti", ndim=3),
            ArtifactContract(eddy_directory / "corrected_dwi.nii.gz", "nifti", ndim=4),
            ArtifactContract(eddy_directory / "eddy_output_mask.nii.gz", "nifti", ndim=3),
            ArtifactContract(eddy_directory / "outlier_free_data.nii.gz", "nifti", ndim=4),
            ArtifactContract(eddy_directory / "rotated_bvecs", "text"),
            ArtifactContract(eddy_directory / "bvals", "text"),
            ArtifactContract(eddy_directory / "eddy_parameters.txt", "text"),
            ArtifactContract(eddy_directory / "outlier_map.txt", "text"),
            ArtifactContract(eddy_directory / "eddy_qa.json", "json"),
        ]
        if config.susceptibility_field is not None:
            eddy_outputs.append(
                ArtifactContract(
                    eddy_directory / "susceptibility_field_hz.nii.gz",
                    "nifti",
                    ndim=3,
                )
            )
        if config.reverse_phase_encoding is not None:
            eddy_outputs.extend(
                (
                    ArtifactContract(preprocess / "topup_eddy_qa.json", "json"),
                    ArtifactContract(preprocess / "topup" / "field_hz.nii.gz", "nifti", ndim=3),
                    ArtifactContract(preprocess / "topup" / "field_coefficients.nii.gz", "nifti", ndim=3),
                    ArtifactContract(preprocess / "topup" / "movement_parameters.txt", "text"),
                    ArtifactContract(
                        preprocess
                        / "topup_preparation"
                        / "topup_corrected_b0_brain_mask.nii.gz",
                        "nifti",
                        ndim=3,
                    ),
                )
            )
        preprocess_outputs = tuple(eddy_outputs)
        corrected_dwi = eddy_directory / "corrected_dwi.nii.gz"
        dwi_for_fit = dti_directory / "DWIforfit.nii.gz"
        fit_bvals = eddy_directory / "bvals"
        fit_bvecs = eddy_directory / "rotated_bvecs"
        dwi_mask = eddy_directory / "nodif_brain_mask.nii.gz"

        def run_fit() -> dict[str, object]:
            dti_directory.mkdir(parents=True, exist_ok=True)
            _write_nonnegative_nifti(corrected_dwi, dwi_for_fit)
            normalized_grad_dev = None
            if config.grad_dev is not None:
                normalized_grad_dev = write_fsl_reoriented(
                    config.grad_dev,
                    dti_directory / "grad_dev.nii",
                    float32=True,
                )
            fit_dti_nifti(
                dwi_for_fit,
                fit_bvals,
                fit_bvecs,
                dwi_mask,
                dti_directory / "DTI.nii.gz",
                grad_dev_file=normalized_grad_dev,
                workers=config.workers,
                compatibility_mode=config.fit_compatibility_mode,
                valid_mask_file=dti_directory / "DTI_valid_mask.nii.gz",
                qa_file=dti_directory / "DTI_qa.json",
            )
            (dti_directory / "DTI.nii.gz").replace(
                dti_directory / "DTI_tensor.nii.gz"
            )
            return {"status": "completed"}

    preprocess_inputs = (
        [config.prefit_tensor]
        if config.preprocessing_mode == "prefit"
        else [config.data, config.bvals, config.bvecs]
    )
    for optional in (
        config.grad_dev,
        config.dwi_brain_mask if config.preprocessing_mode == "eddy" else None,
        config.reverse_phase_encoding,
        config.susceptibility_field,
        config.fieldmap_corrected_mask,
        config.fieldmap_magnitude,
        config.fieldmap_radians_per_second,
    ):
        if optional is not None:
            preprocess_inputs.append(optional)
    stages.append(
        StageDefinition(
            preprocess_name,
            run_preprocess,
            inputs=tuple(path for path in preprocess_inputs if path is not None),
            outputs=preprocess_outputs,
            parameters={
                "mode": config.preprocessing_mode,
                "workers": config.workers,
                "fit_compatibility_mode": config.fit_compatibility_mode,
                "random_seed": config.random_seed,
                "readout_seconds": config.readout_seconds,
                "phase_encoding_direction": config.phase_encoding_direction,
                "reverse_phase_encoding": config.reverse_phase_encoding is not None,
                "susceptibility_field": config.susceptibility_field is not None,
                "fieldmap_corrected_mask": config.fieldmap_corrected_mask is not None,
                "raw_fieldmap": config.fieldmap_magnitude is not None,
                "fieldmap_dwell_milliseconds": config.fieldmap_dwell_milliseconds,
            },
            implementation_version=version,
        )
    )

    if config.preprocessing_mode == "eddy":
        fit_inputs = [corrected_dwi, fit_bvals, fit_bvecs, dwi_mask]
        fit_outputs = [
            ArtifactContract(dwi_for_fit, "nifti", ndim=4),
            ArtifactContract(dti_directory / "DTI_tensor.nii.gz", "nifti", ndim=4, final_axis=6),
            ArtifactContract(dti_directory / "DTI_FA.nii.gz", "nifti", ndim=3),
            ArtifactContract(dti_directory / "DTI_sse.nii.gz", "nifti", ndim=3),
            ArtifactContract(dti_directory / "DTI_valid_mask.nii.gz", "nifti", ndim=3),
            ArtifactContract(dti_directory / "DTI_qa.json", "json"),
        ]
        if config.grad_dev is not None:
            fit_inputs.append(config.grad_dev)
            fit_outputs.append(
                ArtifactContract(dti_directory / "grad_dev.nii", "nifti", ndim=4, final_axis=9)
            )
        stages.append(
            StageDefinition(
                "fit_dti",
                run_fit,
                inputs=tuple(fit_inputs),
                outputs=tuple(fit_outputs),
                dependencies=(preprocess_name,),
                parameters={"workers": config.workers, "fit": "FSL-WLS"},
                implementation_version=version,
            )
        )
        dti = dti_directory
        fit_dependency = "fit_dti"
    elif config.preprocessing_mode == "prefit":
        dti = dti_directory
        fit_dependency = preprocess_name
    else:
        dti = preprocess
        fit_dependency = preprocess_name

    def run_registration() -> dict[str, object]:
        sse_file = (
            None
            if config.preprocessing_mode == "prefit"
            else dti / "DTI_sse.nii.gz"
        )
        report = run_t1_registration_nifti(
            dti / "DTI_tensor.nii.gz",
            dti / "DTI_FA.nii.gz",
            m2m["t1"],
            m2m["labeling"],
            m2m["bias_corrected"],
            registration,
            sse_file=sse_file,
            degrees_of_freedom=6 if config.t1_mode == "rigid" else 12,
            workers=config.workers,
            register_tensor_output=config.t1_mode != "nonlinear",
            progress=subprogress("register_t1"),
        )
        return {"status": report["status"], "mode": report["mode"]}

    registration_outputs = [
        ArtifactContract(registration / "FA2T1.mat", "text"),
        ArtifactContract(registration / "T1_brain.nii.gz", "nifti", ndim=3),
        ArtifactContract(registration / "T1_brainmask.nii.gz", "nifti", ndim=3),
        ArtifactContract(registration / "t1_registration_qa.json", "json"),
    ]
    if config.t1_mode != "nonlinear":
        registration_outputs.extend(
            (
                ArtifactContract(registration / "DTI_coregT1_tensor.nii.gz", "nifti", ndim=4, final_axis=6),
                ArtifactContract(registration / "DTI_coregT1_FA.nii.gz", "nifti", ndim=3),
                ArtifactContract(registration / "DTI_coregT1_V1.nii.gz", "nifti", ndim=4, final_axis=3),
                ArtifactContract(registration / "DTI_coregT1_valid_mask.nii.gz", "nifti", ndim=3),
            )
        )
    registration_inputs = [
        dti / "DTI_tensor.nii.gz",
        dti / "DTI_FA.nii.gz",
        m2m["t1"],
        m2m["labeling"],
        m2m["bias_corrected"],
    ]
    if config.preprocessing_mode != "prefit":
        registration_inputs.insert(2, dti / "DTI_sse.nii.gz")
    stages.append(
        StageDefinition(
            "register_t1",
            run_registration,
            inputs=tuple(registration_inputs),
            outputs=tuple(registration_outputs),
            dependencies=(fit_dependency,),
            parameters={
                "mode": "rigid" if config.t1_mode == "rigid" else "affine",
                "workers": config.workers,
                "register_tensor_output": config.t1_mode != "nonlinear",
            },
            implementation_version=version,
        )
    )

    final_directory = registration
    final_dependency = "register_t1"
    if config.t1_mode == "nonlinear":

        def run_nonlinear() -> dict[str, object]:
            report = register_tensor_fnirt_nifti(
                dti / "DTI_FA.nii.gz",
                dti / "DTI_tensor.nii.gz",
                registration / "T1_brain.nii.gz",
                registration / "FA2T1.mat",
                nonlinear,
                brain_mask_file=registration / "T1_brainmask.nii.gz",
                workers=config.workers,
                compatibility_mode=config.fit_compatibility_mode,
                progress=(
                    None
                    if progress is None
                    else lambda level, phase, done, total, value: progress(
                        f"register_nonlinear:level_{level}:{phase}",
                        done,
                        total,
                        (
                            phase
                            if value is None
                            else f"{phase}; value={value:.6g}"
                        ),
                    )
                ),
            )
            return {"status": report["status"], "mode": report["mode"]}

        stages.append(
            StageDefinition(
                "register_nonlinear",
                run_nonlinear,
                inputs=(
                    dti / "DTI_FA.nii.gz",
                    dti / "DTI_tensor.nii.gz",
                    registration / "T1_brain.nii.gz",
                    registration / "T1_brainmask.nii.gz",
                    registration / "FA2T1.mat",
                ),
                outputs=(
                    ArtifactContract(nonlinear / "FA2T1_warp.nii.gz", "nifti", ndim=4, final_axis=3),
                    ArtifactContract(nonlinear / "FA2T1_field.nii.gz", "nifti", ndim=4, final_axis=3),
                    ArtifactContract(nonlinear / "FA2T1_jacobian.nii.gz", "nifti", ndim=3),
                    ArtifactContract(nonlinear / "DTI_coregT1_tensor.nii.gz", "nifti", ndim=4, final_axis=6),
                    ArtifactContract(nonlinear / "DTI_coregT1_FA.nii.gz", "nifti", ndim=3),
                    ArtifactContract(nonlinear / "DTI_coregT1_V1.nii.gz", "nifti", ndim=4, final_axis=3),
                    ArtifactContract(nonlinear / "DTI_coregT1_valid_mask.nii.gz", "nifti", ndim=3),
                    ArtifactContract(nonlinear / "nonlinear_registration_qa.json", "json"),
                ),
                dependencies=("register_t1",),
                parameters={"workers": config.workers, "subsamp": [8, 4, 2, 2]},
                implementation_version=version,
            )
        )
        final_directory = nonlinear
        final_dependency = "register_nonlinear"

    final_tensor = final_directory / "DTI_coregT1_tensor.nii.gz"
    publish_dependency: str | None = None
    if config.publish_to_m2m:
        published_tensor = config.m2m_directory / "DTI_coregT1_tensor.nii.gz"
        published_provenance = (
            config.m2m_directory / "DTI_coregT1_tensor.provenance.json"
        )

        def run_publish() -> dict[str, object]:
            return _publish_tensor_to_m2m(
                final_tensor, config.m2m_directory, version
            )

        stages.append(
            StageDefinition(
                "publish_tensor",
                run_publish,
                inputs=(final_tensor,),
                outputs=(
                    ArtifactContract(
                        published_tensor, "nifti", ndim=4, final_axis=6
                    ),
                    ArtifactContract(published_provenance, "json"),
                ),
                dependencies=(final_dependency,),
                parameters={"official_m2m_contract": True},
                implementation_version=version,
            )
        )
        publish_dependency = "publish_tensor"
    fem_dependencies: list[str] = []
    fem_manifests: dict[str, Path] = {}
    if config.fem_smoke != "none":
        for mode in ("scalar", "vn", "dir", "mc"):
            stage_name = f"fem_{mode}"
            manifest = fem_root / mode / "dwi2cond_xp_simulation.json"
            fem_manifests[mode] = manifest

            def run_fem(active_mode: str = mode) -> dict[str, object]:
                report = run_tdcs(
                    config.m2m_directory,
                    fem_root,
                    mode=active_mode,
                    tensor_file=None if active_mode == "scalar" else final_tensor,
                    solver=config.solver,
                    cpus=config.workers,
                    dry_run=config.fem_smoke == "dry-run",
                )
                return {"status": report["status"], "mode": active_mode}

            stages.append(
                StageDefinition(
                    stage_name,
                    run_fem,
                    inputs=(
                        m2m["mesh"],
                        m2m["t1"],
                        m2m["final_tissues"],
                        m2m["eeg"],
                        *(() if mode == "scalar" else (final_tensor,)),
                    ),
                    outputs=(ArtifactContract(manifest, "json"),),
                    dependencies=(final_dependency,),
                    parameters={
                        "mode": mode,
                        "solver": config.solver,
                        "cpus": config.workers,
                        "dry_run": config.fem_smoke == "dry-run",
                    },
                    backend="simnibs-4.6.0",
                    implementation_version=version,
                )
            )
            fem_dependencies.append(stage_name)

    def run_qa() -> dict[str, object]:
        if config.preprocessing_mode == "prefit":
            qa_directory.mkdir(parents=True, exist_ok=True)
            payload: dict[str, object] = {
                "status": "completed",
                "mode": "prefit",
                "input_tensor": str(config.prefit_tensor.resolve()),
                "final_tensor": str(final_tensor.resolve()),
                "final_tensor_sha256": _sha256(final_tensor),
                "registered_fa": str(
                    (final_directory / "DTI_coregT1_FA.nii.gz").resolve()
                ),
                "registered_v1": str(
                    (final_directory / "DTI_coregT1_V1.nii.gz").resolve()
                ),
                "valid_mask": str(
                    (final_directory / "DTI_coregT1_valid_mask.nii.gz").resolve()
                ),
            }
            (qa_directory / "pipeline_qa.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            return payload
        report = build_pipeline_qa(
            PipelineQaInputs(
                bvals=fit_bvals,
                original_bvecs=config.bvecs,
                brain_mask=registration / "T1_brainmask.nii.gz",
                dwi_brain_mask=dwi_mask,
                fa=final_directory / "DTI_coregT1_FA.nii.gz",
                tensor=final_tensor,
                valid_mask=final_directory / "DTI_coregT1_valid_mask.nii.gz",
                raw_dwi=config.data,
                corrected_dwi=corrected_dwi,
                rotated_bvecs=(
                    fit_bvecs if config.preprocessing_mode in ("legacy", "eddy") else None
                ),
                t1=m2m["t1"],
                registered_fa=final_directory / "DTI_coregT1_FA.nii.gz",
                v1=final_directory / "DTI_coregT1_V1.nii.gz",
                field_hz=(
                    (
                        preprocess / "topup" / "field_hz.nii.gz"
                        if config.reverse_phase_encoding is not None
                        else config.susceptibility_field
                    )
                    if config.preprocessing_mode == "eddy"
                    else None
                ),
                jacobian=(
                    nonlinear / "FA2T1_jacobian.nii.gz"
                    if config.t1_mode == "nonlinear"
                    else None
                ),
                eddy_parameters=(
                    preprocess / "eddy" / "eddy_parameters.txt"
                    if config.preprocessing_mode == "eddy"
                    else None
                ),
                outlier_map=(
                    preprocess / "eddy" / "outlier_map.txt"
                    if config.preprocessing_mode == "eddy"
                    else None
                ),
                readout_seconds=config.readout_seconds,
                fem_manifests=fem_manifests,
            ),
            qa_directory,
            progress=(
                None
                if progress is None
                else lambda phase, done, total: progress(
                    f"pipeline_qa:{phase}", done, total, "running"
                )
            ),
        )
        return {"status": report["status"]}

    if config.preprocessing_mode == "prefit":
        qa_inputs = [
            config.prefit_tensor,
            final_directory / "DTI_coregT1_FA.nii.gz",
            final_tensor,
            final_directory / "DTI_coregT1_valid_mask.nii.gz",
            final_directory / "DTI_coregT1_V1.nii.gz",
            *fem_manifests.values(),
        ]
        qa_outputs = (ArtifactContract(qa_directory / "pipeline_qa.json", "json"),)
    else:
        qa_inputs = [
            fit_bvals,
            config.bvecs,
            registration / "T1_brainmask.nii.gz",
            dwi_mask,
            final_directory / "DTI_coregT1_FA.nii.gz",
            final_tensor,
            final_directory / "DTI_coregT1_valid_mask.nii.gz",
            config.data,
            corrected_dwi,
            m2m["t1"],
            final_directory / "DTI_coregT1_V1.nii.gz",
            *fem_manifests.values(),
        ]
        qa_outputs = (
            ArtifactContract(qa_directory / "pipeline_qa.json", "json"),
            ArtifactContract(qa_directory / "raw_b0_mean.nii.gz", "nifti", ndim=3),
            ArtifactContract(qa_directory / "raw_mean_dwi.nii.gz", "nifti", ndim=3),
            ArtifactContract(qa_directory / "corrected_b0_mean.nii.gz", "nifti", ndim=3),
            ArtifactContract(qa_directory / "corrected_mean_dwi.nii.gz", "nifti", ndim=3),
            ArtifactContract(qa_directory / "dti_fa_t1_overlay.png", "file"),
        )
    stages.append(
        StageDefinition(
            "pipeline_qa",
            run_qa,
            inputs=tuple(path for path in qa_inputs if path is not None),
            outputs=qa_outputs,
            dependencies=(
                final_dependency,
                *((publish_dependency,) if publish_dependency is not None else ()),
                *fem_dependencies,
            ),
            parameters={
                "fem_smoke": config.fem_smoke,
                "preprocessing_mode": config.preprocessing_mode,
                "t1_mode": config.t1_mode,
            },
            implementation_version=version,
        )
    )

    package_source = Path(__file__).resolve().parents[1]
    implementation_files = tuple(sorted(package_source.rglob("*.py")))
    runner = PipelineRunner(
        root / "manifests",
        progress=progress,
        implementation_files=implementation_files,
    )
    results = runner.run(stages)
    # Aggregated QA has already read and validated upstream values; validate only
    # the final published artifacts here to avoid decoding the large DWI again.
    runner.validate_final_outputs(stages[-1:])
    return Dwi2CondPipelineResult(
        stages=results,
        final_tensor=final_tensor,
        qa_manifest=qa_directory / "pipeline_qa.json",
    )


__all__ = [
    "Dwi2CondPipelineConfig",
    "Dwi2CondPipelineResult",
    "run_dwi2cond_pipeline",
]
