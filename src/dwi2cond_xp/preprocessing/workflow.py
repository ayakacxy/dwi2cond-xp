"""Assemble fixed preprocessing, T1 registration, unified QA, and FEM smoke into an explicit DAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from .. import __version__
from ..nifti_fit import fit_dti_nifti
from ..simulation import run_tdcs
from .eddy import run_eddy_nifti
from .legacy import run_legacy_nifti
from .nomoco import run_nomoco_nifti
from .nonlinear import register_tensor_fnirt_nifti
from .pipeline import ArtifactContract, PipelineRunner, StageDefinition, StageRunResult
from .qa import PipelineQaInputs, build_pipeline_qa
from .t1_registration import run_t1_registration_nifti


WorkflowProgress = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class Dwi2CondPipelineConfig:
    """Freeze every explicit input and mode required by one complete compatible path."""

    data: Path
    bvals: Path
    bvecs: Path
    m2m_directory: Path
    output_directory: Path
    preprocessing_mode: str = "nomoco"
    t1_mode: str = "affine"
    grad_dev: Path | None = None
    dwi_brain_mask: Path | None = None
    susceptibility_field: Path | None = None
    readout_seconds: float | None = None
    phase_encoding_direction: str | None = None
    random_seed: int = 1
    workers: int = 8
    fem_smoke: str = "none"
    solver: str = "pardiso"


@dataclass(frozen=True)
class Dwi2CondPipelineResult:
    """Return all stage manifests, the final tensor, and aggregated QA."""

    stages: tuple[StageRunResult, ...]
    final_tensor: Path
    qa_manifest: Path


def _required_m2m_files(m2m: Path) -> dict[str, Path]:
    """Resolve CHARM inputs once to avoid repeated glob operations across stages."""

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
    files = {
        "t1": m2m / "T1.nii.gz",
        "labeling": m2m / "segmentation" / "labeling.nii.gz",
        "bias_corrected": m2m / "segmentation" / "T1_bias_corrected.nii.gz",
        "final_tissues": m2m / "final_tissues.nii.gz",
        "mesh": mesh,
        "eeg": eeg,
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required CHARM pipeline inputs: " + ", ".join(missing)
        )
    return files


def _validate_config(config: Dwi2CondPipelineConfig) -> dict[str, Path]:
    """Reject invalid combinations once before expensive computation."""

    if config.preprocessing_mode not in ("nomoco", "legacy", "eddy"):
        raise ValueError("preprocessing_mode must be nomoco, legacy, or eddy")
    if config.t1_mode not in ("rigid", "affine", "nonlinear"):
        raise ValueError("t1_mode must be rigid, affine, or nonlinear")
    if config.fem_smoke not in ("none", "dry-run", "run"):
        raise ValueError("fem_smoke must be none, dry-run, or run")
    if config.workers < 1:
        raise ValueError("workers must be positive")
    for path in (config.data, config.bvals, config.bvecs):
        if not path.is_file():
            raise FileNotFoundError(f"Missing pipeline input: {path}")
    if config.preprocessing_mode == "eddy":
        if config.dwi_brain_mask is None or not config.dwi_brain_mask.is_file():
            raise ValueError("eddy mode requires dwi_brain_mask")
        if config.readout_seconds is None or config.phase_encoding_direction is None:
            raise ValueError("eddy mode requires readout_seconds and PE direction")
    elif config.susceptibility_field is not None:
        if config.preprocessing_mode != "legacy":
            raise ValueError("a prepared susceptibility field is only valid for legacy or eddy")
    if config.susceptibility_field is not None and not config.susceptibility_field.is_file():
        raise FileNotFoundError(
            f"Missing susceptibility field: {config.susceptibility_field}"
        )
    return _required_m2m_files(config.m2m_directory)


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

    if config.preprocessing_mode == "nomoco":
        preprocess_name = "preprocess_nomoco"

        def run_preprocess() -> dict[str, object]:
            report = run_nomoco_nifti(
                config.data,
                config.bvals,
                config.bvecs,
                preprocess,
                grad_dev_file=config.grad_dev,
                workers=config.workers,
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
                workers=config.workers,
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
        corrected_dwi = preprocess / "DWI_corr.nii"
        fit_bvals = preprocess / "DWIbvals"
        fit_bvecs = preprocess / "DWIbvecs"
        dwi_mask = preprocess / "nodif_brain_mask.nii.gz"
    else:
        preprocess_name = "preprocess_eddy"
        eddy_directory = preprocess / "eddy"
        dti_directory = preprocess / "dti"

        def run_preprocess() -> dict[str, object]:
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

        preprocess_outputs = (
            ArtifactContract(eddy_directory / "corrected_dwi.nii.gz", "nifti", ndim=4),
            ArtifactContract(eddy_directory / "outlier_free_data.nii.gz", "nifti", ndim=4),
            ArtifactContract(eddy_directory / "rotated_bvecs", "text"),
            ArtifactContract(eddy_directory / "bvals", "text"),
            ArtifactContract(eddy_directory / "eddy_parameters.txt", "text"),
            ArtifactContract(eddy_directory / "outlier_map.txt", "text"),
            ArtifactContract(eddy_directory / "eddy_qa.json", "json"),
        )
        corrected_dwi = eddy_directory / "outlier_free_data.nii.gz"
        fit_bvals = eddy_directory / "bvals"
        fit_bvecs = eddy_directory / "rotated_bvecs"
        dwi_mask = Path(config.dwi_brain_mask)

        def run_fit() -> dict[str, object]:
            dti_directory.mkdir(parents=True, exist_ok=True)
            fit_dti_nifti(
                corrected_dwi,
                fit_bvals,
                fit_bvecs,
                dwi_mask,
                dti_directory / "DTI.nii.gz",
                grad_dev_file=config.grad_dev,
                workers=config.workers,
                valid_mask_file=dti_directory / "DTI_valid_mask.nii.gz",
                qa_file=dti_directory / "DTI_qa.json",
            )
            (dti_directory / "DTI.nii.gz").replace(
                dti_directory / "DTI_tensor.nii.gz"
            )
            return {"status": "completed"}

    preprocess_inputs = [config.data, config.bvals, config.bvecs]
    for optional in (
        config.grad_dev,
        config.dwi_brain_mask if config.preprocessing_mode == "eddy" else None,
        config.susceptibility_field,
    ):
        if optional is not None:
            preprocess_inputs.append(optional)
    stages.append(
        StageDefinition(
            preprocess_name,
            run_preprocess,
            inputs=tuple(preprocess_inputs),
            outputs=preprocess_outputs,
            parameters={
                "mode": config.preprocessing_mode,
                "workers": config.workers,
                "random_seed": config.random_seed,
                "readout_seconds": config.readout_seconds,
                "phase_encoding_direction": config.phase_encoding_direction,
                "susceptibility_field": config.susceptibility_field is not None,
            },
            implementation_version=version,
        )
    )

    if config.preprocessing_mode == "eddy":
        stages.append(
            StageDefinition(
                "fit_dti",
                run_fit,
                inputs=(corrected_dwi, fit_bvals, fit_bvecs, dwi_mask),
                outputs=(
                    ArtifactContract(dti_directory / "DTI_tensor.nii.gz", "nifti", ndim=4, final_axis=6),
                    ArtifactContract(dti_directory / "DTI_FA.nii.gz", "nifti", ndim=3),
                    ArtifactContract(dti_directory / "DTI_sse.nii.gz", "nifti", ndim=3),
                    ArtifactContract(dti_directory / "DTI_valid_mask.nii.gz", "nifti", ndim=3),
                    ArtifactContract(dti_directory / "DTI_qa.json", "json"),
                ),
                dependencies=(preprocess_name,),
                parameters={"workers": config.workers, "fit": "FSL-WLS"},
                implementation_version=version,
            )
        )
        dti = dti_directory
        fit_dependency = "fit_dti"
    else:
        dti = preprocess
        fit_dependency = preprocess_name

    def run_registration() -> dict[str, object]:
        report = run_t1_registration_nifti(
            dti / "DTI_tensor.nii.gz",
            dti / "DTI_FA.nii.gz",
            m2m["t1"],
            m2m["labeling"],
            m2m["bias_corrected"],
            registration,
            sse_file=dti / "DTI_sse.nii.gz",
            degrees_of_freedom=6 if config.t1_mode == "rigid" else 12,
            workers=config.workers,
            progress=subprogress("register_t1"),
        )
        return {"status": report["status"], "mode": report["mode"]}

    stages.append(
        StageDefinition(
            "register_t1",
            run_registration,
            inputs=(
                dti / "DTI_tensor.nii.gz",
                dti / "DTI_FA.nii.gz",
                dti / "DTI_sse.nii.gz",
                m2m["t1"],
                m2m["labeling"],
                m2m["bias_corrected"],
            ),
            outputs=(
                ArtifactContract(registration / "FA2T1.mat", "text"),
                ArtifactContract(registration / "T1_brain.nii.gz", "nifti", ndim=3),
                ArtifactContract(registration / "T1_brainmask.nii.gz", "nifti", ndim=3),
                ArtifactContract(registration / "DTI_coregT1_tensor.nii.gz", "nifti", ndim=4, final_axis=6),
                ArtifactContract(registration / "DTI_coregT1_FA.nii.gz", "nifti", ndim=3),
                ArtifactContract(registration / "DTI_coregT1_V1.nii.gz", "nifti", ndim=4, final_axis=3),
                ArtifactContract(registration / "DTI_coregT1_valid_mask.nii.gz", "nifti", ndim=3),
                ArtifactContract(registration / "t1_registration_qa.json", "json"),
            ),
            dependencies=(fit_dependency,),
            parameters={
                "mode": "rigid" if config.t1_mode == "rigid" else "affine",
                "workers": config.workers,
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
                workers=config.workers,
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
                    config.susceptibility_field
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
    stages.append(
        StageDefinition(
            "pipeline_qa",
            run_qa,
            inputs=tuple(qa_inputs),
            outputs=(
                ArtifactContract(qa_directory / "pipeline_qa.json", "json"),
                ArtifactContract(qa_directory / "raw_b0_mean.nii.gz", "nifti", ndim=3),
                ArtifactContract(qa_directory / "raw_mean_dwi.nii.gz", "nifti", ndim=3),
                ArtifactContract(qa_directory / "corrected_b0_mean.nii.gz", "nifti", ndim=3),
                ArtifactContract(qa_directory / "corrected_mean_dwi.nii.gz", "nifti", ndim=3),
                ArtifactContract(qa_directory / "dti_fa_t1_overlay.png", "file"),
            ),
            dependencies=(final_dependency, *fem_dependencies),
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
