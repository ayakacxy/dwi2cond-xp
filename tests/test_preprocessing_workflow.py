"""Verify the topology, cache, and invalid-combination gates of the full P11 workflow."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from types import SimpleNamespace

from dwi2cond_xp.preprocessing import workflow
from dwi2cond_xp.cli import main


def _nifti(path: Path, shape: tuple[int, ...], value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.full(shape, value, dtype=np.float32), np.eye(4)),
        path,
    )


def _json(path: Path) -> None:
    path.write_text('{"status":"completed"}\n', encoding="utf-8")


def _fixture(tmp_path: Path) -> workflow.Dwi2CondPipelineConfig:
    _nifti(tmp_path / "dwi.nii.gz", (3, 3, 3, 2))
    (tmp_path / "bvals").write_text("0 1000\n", encoding="utf-8")
    (tmp_path / "bvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
    m2m = tmp_path / "m2m_subject"
    _nifti(m2m / "T1.nii.gz", (3, 3, 3))
    _nifti(m2m / "final_tissues.nii.gz", (3, 3, 3))
    _nifti(m2m / "segmentation" / "labeling.nii.gz", (3, 3, 3))
    _nifti(m2m / "segmentation" / "T1_bias_corrected.nii.gz", (3, 3, 3))
    (m2m / "subject.msh").write_text("mesh\n", encoding="utf-8")
    eeg = m2m / "eeg_positions" / "EEG10-10_UI_Jurak_2007.csv"
    eeg.parent.mkdir(parents=True)
    eeg.write_text("Electrode\n", encoding="utf-8")
    return workflow.Dwi2CondPipelineConfig(
        data=tmp_path / "dwi.nii.gz",
        bvals=tmp_path / "bvals",
        bvecs=tmp_path / "bvecs",
        m2m_directory=m2m,
        output_directory=tmp_path / "output",
        preprocessing_mode="nomoco",
        t1_mode="affine",
        workers=8,
    )


def test_complete_workflow_writes_stage_manifests_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    calls = {"preprocess": 0, "registration": 0, "qa": 0}

    def fake_nomoco(_data: Path, _bvals: Path, _bvecs: Path, output: Path, **_kwargs):
        calls["preprocess"] += 1
        output.mkdir(parents=True, exist_ok=True)
        for name, shape in (
            ("DTI_tensor.nii.gz", (3, 3, 3, 6)),
            ("DTI_FA.nii.gz", (3, 3, 3)),
            ("DTI_sse.nii.gz", (3, 3, 3)),
            ("DTI_valid_mask.nii.gz", (3, 3, 3)),
            ("nodif_brain_mask.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        (output / "DWIbvals").write_text("0 1000\n", encoding="utf-8")
        (output / "DWIbvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
        _json(output / "nomoco_qa.json")
        return {"status": "completed", "mode": "nomoco"}

    def fake_registration(*_args, **kwargs):
        calls["registration"] += 1
        output = Path(_args[5])
        output.mkdir(parents=True, exist_ok=True)
        (output / "FA2T1.mat").write_text("1 0 0 0\n", encoding="utf-8")
        for name, shape in (
            ("T1_brain.nii.gz", (3, 3, 3)),
            ("T1_brainmask.nii.gz", (3, 3, 3)),
            ("DTI_coregT1_tensor.nii.gz", (3, 3, 3, 6)),
            ("DTI_coregT1_FA.nii.gz", (3, 3, 3)),
            ("DTI_coregT1_V1.nii.gz", (3, 3, 3, 3)),
            ("DTI_coregT1_valid_mask.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        _json(output / "t1_registration_qa.json")
        return {"status": "completed", "mode": "affine"}

    def fake_qa(_inputs, output: Path, **_kwargs):
        calls["qa"] += 1
        output.mkdir(parents=True, exist_ok=True)
        _json(output / "pipeline_qa.json")
        for name in (
            "raw_b0_mean.nii.gz",
            "raw_mean_dwi.nii.gz",
            "corrected_b0_mean.nii.gz",
            "corrected_mean_dwi.nii.gz",
        ):
            _nifti(output / name, (3, 3, 3))
        (output / "dti_fa_t1_overlay.png").write_bytes(b"png")
        return {"status": "completed"}

    monkeypatch.setattr(workflow, "run_nomoco_nifti", fake_nomoco)
    monkeypatch.setattr(workflow, "run_t1_registration_nifti", fake_registration)
    monkeypatch.setattr(workflow, "build_pipeline_qa", fake_qa)

    first = workflow.run_dwi2cond_pipeline(config)
    second = workflow.run_dwi2cond_pipeline(config)

    assert [stage.status for stage in first.stages] == [
        "completed",
        "completed",
        "completed",
    ]
    assert [stage.status for stage in second.stages] == ["cached", "cached", "cached"]
    assert calls == {"preprocess": 1, "registration": 1, "qa": 1}
    assert first.final_tensor.is_file()
    assert json.loads(first.qa_manifest.read_text())["status"] == "completed"
    manifests = sorted((config.output_directory / "manifests").glob("*.json"))
    assert [path.stem for path in manifests] == [
        "pipeline_qa",
        "preprocess_nomoco",
        "register_t1",
    ]


def test_workflow_rejects_eddy_without_explicit_acquisition_contract(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    invalid = workflow.Dwi2CondPipelineConfig(
        **{**config.__dict__, "preprocessing_mode": "eddy"}
    )

    with pytest.raises(ValueError, match="dwi_brain_mask"):
        workflow.run_dwi2cond_pipeline(invalid)


def test_run_pipeline_cli_preserves_explicit_modes_and_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[workflow.Dwi2CondPipelineConfig] = []

    def fake_run(config, *, progress):
        captured.append(config)
        progress("preprocess_eddy", 1, 1, "completed")
        return SimpleNamespace(
            qa_manifest=tmp_path / "qa.json",
            final_tensor=tmp_path / "tensor.nii.gz",
        )

    monkeypatch.setattr(workflow, "run_dwi2cond_pipeline", fake_run)
    status = main(
        [
            "run-pipeline",
            str(tmp_path / "dwi.nii.gz"),
            str(tmp_path / "bvals"),
            str(tmp_path / "bvecs"),
            str(tmp_path / "m2m_subject"),
            str(tmp_path / "output"),
            "--preprocessing-mode",
            "eddy",
            "--t1-mode",
            "nonlinear",
            "--dwi-brain-mask",
            str(tmp_path / "mask.nii.gz"),
            "--readout-seconds",
            "0.05",
            "--phase-encoding-direction",
            "y-",
            "--workers",
            "8",
            "--fem-smoke",
            "dry-run",
            "--progress",
            "off",
        ]
    )

    assert status == 0
    assert captured[0].preprocessing_mode == "eddy"
    assert captured[0].t1_mode == "nonlinear"
    assert captured[0].phase_encoding_direction == "y-"
    assert captured[0].workers == 8
    assert captured[0].fem_smoke == "dry-run"


def _install_common_workflow_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, int, int, str]]:
    """Install fast stage doubles that write only the official artifact contract."""

    def fake_registration(*args, **kwargs):
        output = Path(args[5])
        output.mkdir(parents=True, exist_ok=True)
        (output / "FA2T1.mat").write_text("1 0 0 0\n", encoding="utf-8")
        for name, shape in (
            ("T1_brain.nii.gz", (3, 3, 3)),
            ("T1_brainmask.nii.gz", (3, 3, 3)),
            ("DTI_coregT1_tensor.nii.gz", (3, 3, 3, 6)),
            ("DTI_coregT1_FA.nii.gz", (3, 3, 3)),
            ("DTI_coregT1_V1.nii.gz", (3, 3, 3, 3)),
            ("DTI_coregT1_valid_mask.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        _json(output / "t1_registration_qa.json")
        if kwargs.get("progress") is not None:
            kwargs["progress"]("registration", 1, 1)
        return {"status": "completed", "mode": "affine"}

    def fake_nonlinear(*args, **kwargs):
        output = Path(args[4])
        output.mkdir(parents=True, exist_ok=True)
        for name, shape in (
            ("FA2T1_warp.nii.gz", (3, 3, 3, 3)),
            ("FA2T1_field.nii.gz", (3, 3, 3, 3)),
            ("FA2T1_jacobian.nii.gz", (3, 3, 3)),
            ("DTI_coregT1_tensor.nii.gz", (3, 3, 3, 6)),
            ("DTI_coregT1_FA.nii.gz", (3, 3, 3)),
            ("DTI_coregT1_V1.nii.gz", (3, 3, 3, 3)),
            ("DTI_coregT1_valid_mask.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        _json(output / "nonlinear_registration_qa.json")
        if kwargs.get("progress") is not None:
            kwargs["progress"](1, "complete", 1, 1, 0.0)
        return {"status": "completed", "mode": "nonlinear"}

    def fake_fem(m2m, output, *, mode, **kwargs):
        manifest = Path(output) / mode / "dwi2cond_xp_simulation.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"status": "planned", "mode": mode}) + "\n",
            encoding="utf-8",
        )
        return {"status": "planned", "mode": mode}

    def fake_qa(_inputs, output, **kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        _json(output / "pipeline_qa.json")
        for name in (
            "raw_b0_mean.nii.gz",
            "raw_mean_dwi.nii.gz",
            "corrected_b0_mean.nii.gz",
            "corrected_mean_dwi.nii.gz",
        ):
            _nifti(output / name, (3, 3, 3))
        (output / "dti_fa_t1_overlay.png").write_bytes(b"png")
        if kwargs.get("progress") is not None:
            kwargs["progress"]("complete", 8, 8)
        return {"status": "completed"}

    monkeypatch.setattr(workflow, "run_t1_registration_nifti", fake_registration)
    monkeypatch.setattr(workflow, "register_tensor_fnirt_nifti", fake_nonlinear)
    monkeypatch.setattr(workflow, "run_tdcs", fake_fem)
    monkeypatch.setattr(workflow, "build_pipeline_qa", fake_qa)
    events: list[tuple[str, int, int, str]] = []
    return events


def test_eddy_nonlinear_and_all_fem_modes_form_one_explicit_dag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    mask = tmp_path / "eddy-mask.nii.gz"
    field = tmp_path / "field.nii.gz"
    _nifti(mask, (3, 3, 3))
    _nifti(field, (3, 3, 3))
    events = _install_common_workflow_stubs(monkeypatch)

    def fake_eddy(*args, **kwargs):
        output = Path(args[4])
        output.mkdir(parents=True, exist_ok=True)
        _nifti(output / "corrected_dwi.nii.gz", (3, 3, 3, 2))
        _nifti(output / "outlier_free_data.nii.gz", (3, 3, 3, 2))
        (output / "rotated_bvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
        (output / "bvals").write_text("0 1000\n", encoding="utf-8")
        (output / "eddy_parameters.txt").write_text("0 0\n0 0\n", encoding="utf-8")
        (output / "outlier_map.txt").write_text("0\n0\n", encoding="utf-8")
        _json(output / "eddy_qa.json")
        kwargs["progress"]("iteration", 1, 1)
        return {"status": "completed", "algorithm": "eddy"}

    def fake_fit(_data, _bvals, _bvecs, _mask, tensor, **kwargs):
        directory = Path(tensor).parent
        _nifti(Path(tensor), (3, 3, 3, 6))
        _nifti(directory / "DTI_FA.nii.gz", (3, 3, 3))
        _nifti(directory / "DTI_sse.nii.gz", (3, 3, 3))
        _nifti(Path(kwargs["valid_mask_file"]), (3, 3, 3))
        _json(Path(kwargs["qa_file"]))

    monkeypatch.setattr(workflow, "run_eddy_nifti", fake_eddy)
    monkeypatch.setattr(workflow, "fit_dti_nifti", fake_fit)
    complete = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "eddy",
            "t1_mode": "nonlinear",
            "dwi_brain_mask": mask,
            "susceptibility_field": field,
            "readout_seconds": 0.05,
            "phase_encoding_direction": "y-",
            "fem_smoke": "dry-run",
        }
    )
    result = workflow.run_dwi2cond_pipeline(complete, progress=lambda *args: events.append(args))

    assert [stage.name for stage in result.stages] == [
        "preprocess_eddy",
        "fit_dti",
        "register_t1",
        "register_nonlinear",
        "fem_scalar",
        "fem_vn",
        "fem_dir",
        "fem_mc",
        "pipeline_qa",
    ]
    assert all(stage.status == "completed" for stage in result.stages)
    assert any(name.startswith("preprocess_eddy:") for name, *_ in events)
    assert any(name.startswith("register_nonlinear:") for name, *_ in events)
    assert any(name.startswith("pipeline_qa:") for name, *_ in events)


def test_legacy_rigid_mode_forwards_optional_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    grad = tmp_path / "grad.nii.gz"
    field = tmp_path / "legacy-field.nii.gz"
    _nifti(grad, (3, 3, 3, 9))
    _nifti(field, (3, 3, 3))
    _install_common_workflow_stubs(monkeypatch)
    seen: dict[str, object] = {}

    def fake_legacy(*args, **kwargs):
        output = Path(args[3])
        seen.update(kwargs)
        output.mkdir(parents=True, exist_ok=True)
        _nifti(output / "DWI_corr.nii", (3, 3, 3, 2))
        for name, shape in (
            ("DTI_tensor.nii.gz", (3, 3, 3, 6)),
            ("DTI_FA.nii.gz", (3, 3, 3)),
            ("DTI_sse.nii.gz", (3, 3, 3)),
            ("DTI_valid_mask.nii.gz", (3, 3, 3)),
            ("nodif_brain_mask.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        (output / "DWIbvals").write_text("0 1000\n", encoding="utf-8")
        (output / "DWIbvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
        _json(output / "legacy_qa.json")
        kwargs["progress"]("legacy", 1, 1)
        return {"status": "completed", "mode": "legacy"}

    monkeypatch.setattr(workflow, "run_legacy_nifti", fake_legacy)
    complete = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "legacy",
            "t1_mode": "rigid",
            "grad_dev": grad,
            "susceptibility_field": field,
        }
    )
    result = workflow.run_dwi2cond_pipeline(complete, progress=lambda *_args: None)

    assert result.stages[0].name == "preprocess_legacy"
    assert seen["grad_dev_file"] == grad
    assert seen["fieldmap_displacement_file"] == field


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"preprocessing_mode": "bad"}, "preprocessing_mode"),
        ({"t1_mode": "bad"}, "t1_mode"),
        ({"fem_smoke": "bad"}, "fem_smoke"),
        ({"workers": 0}, "workers"),
        ({"preprocessing_mode": "nomoco", "susceptibility_field": Path("missing")}, "only valid"),
        ({"preprocessing_mode": "eddy", "dwi_brain_mask": None}, "dwi_brain_mask"),
    ),
)
def test_workflow_rejects_invalid_modes_before_computation(
    tmp_path: Path,
    changes: dict[str, object],
    error: str,
) -> None:
    config = _fixture(tmp_path)
    invalid = workflow.Dwi2CondPipelineConfig(**{**config.__dict__, **changes})
    with pytest.raises(ValueError, match=error):
        workflow.run_dwi2cond_pipeline(invalid)


def test_workflow_rejects_missing_files_and_resolves_m2m_fallbacks(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config.data.unlink()
    with pytest.raises(FileNotFoundError, match="Missing pipeline input"):
        workflow.run_dwi2cond_pipeline(config)

    config = _fixture(tmp_path / "fallback")
    mesh = config.m2m_directory / "subject.msh"
    mesh.rename(config.m2m_directory / "only.msh")
    eeg = config.m2m_directory / "eeg_positions" / "EEG10-10_UI_Jurak_2007.csv"
    eeg.rename(eeg.with_name("custom-10-10.csv"))
    resolved = workflow._required_m2m_files(config.m2m_directory)
    assert resolved["mesh"].name == "only.msh"
    assert resolved["eeg"].name == "custom-10-10.csv"
    resolved["t1"].unlink()
    with pytest.raises(FileNotFoundError, match="t1"):
        workflow._required_m2m_files(config.m2m_directory)


def test_eddy_requires_both_readout_and_phase_encoding(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    mask = tmp_path / "mask.nii.gz"
    _nifti(mask, (3, 3, 3))
    invalid = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "eddy",
            "dwi_brain_mask": mask,
            "readout_seconds": None,
            "phase_encoding_direction": "y",
        }
    )
    with pytest.raises(ValueError, match="readout_seconds"):
        workflow.run_dwi2cond_pipeline(invalid)


def test_missing_legacy_field_is_rejected_after_valid_mode_check(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    invalid = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "legacy",
            "susceptibility_field": tmp_path / "missing-field.nii.gz",
        }
    )
    with pytest.raises(FileNotFoundError, match="Missing susceptibility field"):
        workflow.run_dwi2cond_pipeline(invalid)
