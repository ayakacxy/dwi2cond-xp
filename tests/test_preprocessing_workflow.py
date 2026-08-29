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
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "artifacts": {"report": "producer-artifact.nii.gz"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


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
            ("DWIraw.nii", (3, 3, 3, 2)),
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
            ("DTI_FA_6dof_QA.nii.gz", (3, 3, 3)),
            ("DTI_SSE_6dof_QA.nii.gz", (3, 3, 3)),
            ("DTIraw_FA_6dof_QA.nii.gz", (3, 3, 3)),
            ("DTIraw_SSE_6dof_QA.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        (output / "FA2T1_raw_QA.mat").write_text("1 0 0 0\n", encoding="utf-8")
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

    def fake_fit(_data, _bvals, _bvecs, _mask, tensor, **kwargs):
        directory = Path(tensor).parent
        _nifti(Path(tensor), (3, 3, 3, 6))
        _nifti(directory / "DTI_FA.nii.gz", (3, 3, 3))
        _nifti(directory / "DTI_sse.nii.gz", (3, 3, 3))
        _nifti(Path(kwargs["valid_mask_file"]), (3, 3, 3))
        _json(Path(kwargs["qa_file"]))

    def fake_aligned(data, _bvals, target, **kwargs):
        image = nib.load(data)
        _nifti(Path(target), image.shape[:3])
        if kwargs.get("progress") is not None:
            kwargs["progress"](1, 1)
        return Path(target)

    def fake_bet(_source, target, **_kwargs):
        _nifti(Path(target), (3, 3, 3))
        return SimpleNamespace()

    monkeypatch.setattr(workflow, "run_nomoco_nifti", fake_nomoco)
    monkeypatch.setattr(workflow, "fit_dti_nifti", fake_fit)
    monkeypatch.setattr(workflow, "write_aligned_b0_mean", fake_aligned)
    monkeypatch.setattr(workflow, "write_bet_brain_mask", fake_bet)
    monkeypatch.setattr(workflow, "run_t1_registration_nifti", fake_registration)
    monkeypatch.setattr(workflow, "build_pipeline_qa", fake_qa)

    first = workflow.run_dwi2cond_pipeline(config)
    second = workflow.run_dwi2cond_pipeline(config)
    robust_config = workflow.Dwi2CondPipelineConfig(
        **{**config.__dict__, "fit_compatibility_mode": "robust"}
    )
    robust = workflow.run_dwi2cond_pipeline(robust_config)
    strict_again = workflow.run_dwi2cond_pipeline(config)

    assert [stage.status for stage in first.stages] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert [stage.status for stage in second.stages] == [
        "cached",
        "cached",
        "cached",
        "cached",
        "cached",
    ]
    assert [stage.status for stage in robust.stages] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert [stage.status for stage in strict_again.stages] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert calls == {"preprocess": 3, "registration": 3, "qa": 3}
    assert first.final_tensor.is_file()
    assert json.loads(first.qa_manifest.read_text())["status"] == "completed"
    manifests = sorted((config.output_directory / "manifests").glob("*.json"))
    assert [path.stem for path in manifests] == [
        "fit_raw_dti_qa",
        "pipeline_qa",
        "preprocess_nomoco",
        "publish_tensor",
        "register_t1",
    ]


def test_workflow_eddy_default_mask_does_not_require_external_file(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    compatible = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "eddy",
            "readout_seconds": 0.05,
            "phase_encoding_direction": "y",
        }
    )

    resolved = workflow._validate_config(compatible)

    assert resolved["t1"] == compatible.m2m_directory / "T1.nii.gz"


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


def test_run_prefit_pipeline_cli_uses_official_import_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[workflow.Dwi2CondPipelineConfig] = []

    def fake_run(config, *, progress):
        captured.append(config)
        progress("import_prefit_tensor", 1, 1, "completed")
        return SimpleNamespace(
            qa_manifest=tmp_path / "qa.json",
            final_tensor=tmp_path / "tensor.nii.gz",
        )

    monkeypatch.setattr(workflow, "run_dwi2cond_pipeline", fake_run)
    status = main(
        [
            "run-prefit-pipeline",
            str(tmp_path / "DTI_tensor.nii.gz"),
            str(tmp_path / "m2m_subject"),
            str(tmp_path / "output"),
            "--t1-mode",
            "affine",
            "--no-publish-to-m2m",
            "--progress",
            "off",
        ]
    )

    assert status == 0
    assert captured[0].preprocessing_mode == "prefit"
    assert captured[0].prefit_tensor == tmp_path / "DTI_tensor.nii.gz"
    assert captured[0].data is None
    assert captured[0].bvals is None
    assert captured[0].bvecs is None
    assert captured[0].t1_mode == "affine"
    assert captured[0].publish_to_m2m is False


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
            ("DTI_FA_6dof_QA.nii.gz", (3, 3, 3)),
            ("DTI_SSE_6dof_QA.nii.gz", (3, 3, 3)),
            ("DTIraw_FA_6dof_QA.nii.gz", (3, 3, 3)),
            ("DTIraw_SSE_6dof_QA.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        (output / "FA2T1_raw_QA.mat").write_text("1 0 0 0\n", encoding="utf-8")
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
            ("DTI_FA_nonlin.nii.gz", (3, 3, 3)),
            ("DTI_coregT1_jacobian.nii.gz", (3, 3, 3)),
        ):
            _nifti(output / name, shape)
        _json(output / "DTI_coregT1_nonlinear_qa.json")
        _json(output / "nonlinear_registration_qa.json")
        if kwargs.get("progress") is not None:
            kwargs["progress"](1, "complete", 1, 1, 0.0)
        return {"status": "completed", "mode": "nonlinear"}

    def fake_fem(m2m, output, *, mode, **kwargs):
        root = Path(output) / "dry-run" if kwargs.get("dry_run") else Path(output)
        manifest = root / mode / "dwi2cond_xp_simulation.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "status": "planned",
                    "mode": mode,
                    "output_directory": str(manifest.parent),
                    "outputs": [],
                    "masked_subject_volumes": [],
                    "artifacts": [],
                }
            )
            + "\n",
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

    def fake_fit(_data, _bvals, _bvecs, _mask, tensor, **kwargs):
        directory = Path(tensor).parent
        _nifti(Path(tensor), (3, 3, 3, 6))
        _nifti(directory / "DTI_FA.nii.gz", (3, 3, 3))
        _nifti(directory / "DTI_sse.nii.gz", (3, 3, 3))
        _nifti(Path(kwargs["valid_mask_file"]), (3, 3, 3))
        _json(Path(kwargs["qa_file"]))

    def fake_aligned(data, _bvals, target, **kwargs):
        image = nib.load(data)
        _nifti(Path(target), image.shape[:3])
        if kwargs.get("progress") is not None:
            kwargs["progress"](1, 1)
        return Path(target)

    def fake_bet(_source, target, **_kwargs):
        _nifti(Path(target), (3, 3, 3))
        return SimpleNamespace()

    monkeypatch.setattr(workflow, "run_t1_registration_nifti", fake_registration)
    monkeypatch.setattr(workflow, "fit_dti_nifti", fake_fit)
    monkeypatch.setattr(workflow, "write_aligned_b0_mean", fake_aligned)
    monkeypatch.setattr(workflow, "write_bet_brain_mask", fake_bet)
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
    field = tmp_path / "field.nii.gz"
    _nifti(field, (3, 3, 3))
    events = _install_common_workflow_stubs(monkeypatch)

    def fake_aligned(data, _bvals, target, **kwargs):
        image = nib.load(data)
        _nifti(Path(target), image.shape[:3])
        kwargs["progress"](1, 1)
        return Path(target)

    def fake_bet(_source, target, **_kwargs):
        _nifti(Path(target), (3, 3, 3))
        return SimpleNamespace()

    def fake_eddy(*args, **kwargs):
        assert Path(args[0]).name == "DWIraw.nii"
        assert Path(args[3]).name == "nodif_brain_mask.nii.gz"
        assert "eddy_mask_preparation" in str(args[3])
        output = Path(args[4])
        output.mkdir(parents=True, exist_ok=True)
        _nifti(output / "DWIraw.nii", (3, 3, 3, 2))
        _nifti(output / "nodif_brain_mask.nii.gz", (3, 3, 3))
        _nifti(output / "corrected_dwi.nii.gz", (3, 3, 3, 2), value=2.0)
        _nifti(output / "outlier_free_data.nii.gz", (3, 3, 3, 2))
        _nifti(output / "eddy_output_mask.nii.gz", (3, 3, 3))
        _nifti(output / "susceptibility_field_hz.nii.gz", (3, 3, 3))
        (output / "rotated_bvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
        (output / "bvals").write_text("0 1000\n", encoding="utf-8")
        (output / "eddy_parameters.txt").write_text("0 0\n0 0\n", encoding="utf-8")
        (output / "outlier_map.txt").write_text("0\n0\n", encoding="utf-8")
        _json(output / "eddy_qa.json")
        kwargs["progress"]("iteration", 1, 1)
        return {"status": "completed", "algorithm": "eddy"}

    def fake_fit(_data, _bvals, _bvecs, _mask, tensor, **kwargs):
        if "raw_dti_qa" not in str(_data):
            assert np.all(np.asarray(nib.load(_data).dataobj) == 2.0)
        directory = Path(tensor).parent
        _nifti(Path(tensor), (3, 3, 3, 6))
        _nifti(directory / "DTI_FA.nii.gz", (3, 3, 3))
        _nifti(directory / "DTI_sse.nii.gz", (3, 3, 3))
        _nifti(Path(kwargs["valid_mask_file"]), (3, 3, 3))
        _json(Path(kwargs["qa_file"]))

    monkeypatch.setattr(workflow, "write_aligned_b0_mean", fake_aligned)
    monkeypatch.setattr(workflow, "write_bet_brain_mask", fake_bet)
    monkeypatch.setattr(workflow, "run_eddy_nifti", fake_eddy)
    monkeypatch.setattr(workflow, "fit_dti_nifti", fake_fit)
    complete = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "eddy",
            "t1_mode": "nonlinear",
            "dwi_brain_mask": None,
            "susceptibility_field": field,
            "readout_seconds": 0.05,
            "phase_encoding_direction": "y-",
            "fem_smoke": "dry-run",
        }
    )
    result = workflow.run_dwi2cond_pipeline(complete, progress=lambda *args: events.append(args))

    assert [stage.name for stage in result.stages] == [
        "fit_raw_dti_qa",
        "preprocess_eddy",
        "fit_dti",
        "register_t1",
        "register_nonlinear",
        "publish_tensor",
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
    qa_manifest = json.loads(
        (complete.output_directory / "manifests/pipeline_qa.json").read_text()
    )
    qa_input_paths = {Path(item["path"]) for item in qa_manifest["inputs"]}
    assert {
        complete.output_directory / "preprocess/raw_dti_qa/DWIraw.nii",
        complete.output_directory
        / "preprocess/raw_dti_qa/nodif_brain_mask.nii.gz",
        complete.output_directory / "preprocess/eddy/rotated_bvecs",
        complete.output_directory / "preprocess/eddy/eddy_parameters.txt",
        complete.output_directory / "preprocess/eddy/outlier_map.txt",
        complete.output_directory / "nonlinear/FA2T1_jacobian.nii.gz",
        field,
    } <= qa_input_paths
    preprocess_manifest = json.loads(
        (config.output_directory / "manifests" / "preprocess_eddy.json").read_text()
    )
    assert (
        preprocess_manifest["parameters"]["eddy_mask_source"]
        == "official-exact-b0-alignment-bet-0.2"
    )


def test_legacy_rigid_mode_forwards_optional_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    grad = tmp_path / "grad.nii.gz"
    field = tmp_path / "legacy-field.nii.gz"
    corrected_mask = tmp_path / "legacy-corrected-mask.nii.gz"
    _nifti(grad, (3, 3, 3, 9))
    _nifti(field, (3, 3, 3))
    _nifti(corrected_mask, (3, 3, 3))
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
            "fieldmap_corrected_mask": corrected_mask,
        }
    )
    result = workflow.run_dwi2cond_pipeline(complete, progress=lambda *_args: None)

    assert result.stages[0].name == "fit_raw_dti_qa"
    assert result.stages[1].name == "preprocess_legacy"
    assert seen["grad_dev_file"] == grad
    assert seen["fieldmap_displacement_file"] == field
    assert seen["fieldmap_corrected_mask_file"] == corrected_mask


def test_legacy_raw_fieldmap_is_part_of_the_cached_workflow_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    magnitude = tmp_path / "fieldmap-magnitude.nii.gz"
    radians = tmp_path / "fieldmap-radians-per-second.nii.gz"
    _nifti(magnitude, (3, 3, 3))
    _nifti(radians, (3, 3, 3), value=10.0)
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
        _nifti(output / "fieldmap/displacement_world_mm.nii.gz", (3, 3, 3, 3))
        _nifti(output / "fieldmap/corrected_mask.nii.gz", (3, 3, 3))
        _json(output / "fieldmap/fieldmap_qa.json")
        return {"status": "completed", "mode": "legacy"}

    monkeypatch.setattr(workflow, "run_legacy_nifti", fake_legacy)
    complete = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "legacy",
            "fieldmap_magnitude": magnitude,
            "fieldmap_radians_per_second": radians,
            "fieldmap_dwell_milliseconds": 0.75,
            "phase_encoding_direction": "y-",
        }
    )

    first = workflow.run_dwi2cond_pipeline(complete)
    second = workflow.run_dwi2cond_pipeline(complete)

    assert first.stages[0].status == "completed"
    assert second.stages[0].status == "cached"
    assert seen["fieldmap_magnitude_file"] == magnitude
    assert seen["fieldmap_radians_per_second_file"] == radians
    assert seen["fieldmap_dwell_milliseconds"] == 0.75
    assert seen["fieldmap_phase_encoding_direction"] == "y-"


def test_workflow_rejects_incomplete_or_mixed_raw_fieldmap_contract(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    magnitude = tmp_path / "fieldmap-magnitude.nii.gz"
    radians = tmp_path / "fieldmap-radians-per-second.nii.gz"
    prepared = tmp_path / "prepared-field.nii.gz"
    corrected_mask = tmp_path / "prepared-mask.nii.gz"
    for path in (magnitude, radians, prepared, corrected_mask):
        _nifti(path, (3, 3, 3))

    incomplete = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "legacy",
            "fieldmap_magnitude": magnitude,
        }
    )
    with pytest.raises(ValueError, match="raw fieldmap requires"):
        workflow.run_dwi2cond_pipeline(incomplete)

    mixed = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "legacy",
            "fieldmap_magnitude": magnitude,
            "fieldmap_radians_per_second": radians,
            "fieldmap_dwell_milliseconds": 0.75,
            "phase_encoding_direction": "y",
            "susceptibility_field": prepared,
            "fieldmap_corrected_mask": corrected_mask,
        }
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        workflow.run_dwi2cond_pipeline(mixed)


def test_prefit_tensor_forms_complete_registration_publish_and_qa_dag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _fixture(tmp_path)
    tensor = tmp_path / "input_tensor.nii.gz"
    values = np.zeros((3, 3, 3, 6), dtype=np.float32)
    values[..., 0] = 1.5e-3
    values[..., 3] = 0.7e-3
    values[..., 5] = 0.4e-3
    nib.save(nib.Nifti1Image(values, np.eye(4)), tensor)
    _install_common_workflow_stubs(monkeypatch)
    config = workflow.Dwi2CondPipelineConfig(
        data=None,
        bvals=None,
        bvecs=None,
        prefit_tensor=tensor,
        m2m_directory=raw.m2m_directory,
        output_directory=raw.output_directory,
        preprocessing_mode="prefit",
        t1_mode="affine",
    )

    result = workflow.run_dwi2cond_pipeline(config)

    assert [stage.name for stage in result.stages] == [
        "import_prefit_tensor",
        "register_t1",
        "publish_tensor",
        "pipeline_qa",
    ]
    imported = config.output_directory / "preprocess/prefit/DTI_tensor.nii.gz"
    assert imported.is_file()
    assert np.count_nonzero(
        np.asarray(nib.load(imported.with_name("DTI_FA.nii.gz")).dataobj)
    ) == 27
    assert (config.m2m_directory / "DTI_coregT1_tensor.nii.gz").is_file()
    provenance = json.loads(
        (config.m2m_directory / "DTI_coregT1_tensor.provenance.json").read_text()
    )
    assert provenance["implementation"] == "dwi2cond-xp"


def test_reverse_pe_workflow_and_grad_dev_form_one_complete_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path)
    reverse = tmp_path / "reverse.nii.gz"
    grad = tmp_path / "grad-dev.nii.gz"
    _nifti(reverse, (3, 3, 3, 2))
    _nifti(grad, (3, 3, 3, 9))
    _install_common_workflow_stubs(monkeypatch)

    def fake_topup_eddy(*args, **kwargs):
        output = Path(args[4])
        eddy = output / "eddy"
        for name, shape in (
            ("DWIraw.nii", (3, 3, 3, 2)),
            ("nodif_brain_mask.nii.gz", (3, 3, 3)),
            ("corrected_dwi.nii.gz", (3, 3, 3, 2)),
            ("eddy_output_mask.nii.gz", (3, 3, 3)),
            ("outlier_free_data.nii.gz", (3, 3, 3, 2)),
        ):
            _nifti(eddy / name, shape)
        (eddy / "rotated_bvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
        (eddy / "bvals").write_text("0 1000\n", encoding="utf-8")
        (eddy / "eddy_parameters.txt").write_text("0 0\n0 0\n", encoding="utf-8")
        (eddy / "outlier_map.txt").write_text("0\n0\n", encoding="utf-8")
        _json(eddy / "eddy_qa.json")
        _json(output / "topup_eddy_qa.json")
        _nifti(output / "topup/field_hz.nii.gz", (3, 3, 3))
        _nifti(output / "topup/field_coefficients.nii.gz", (3, 3, 3))
        (output / "topup/movement_parameters.txt").write_text(
            "0 0 0 0 0 0\n", encoding="utf-8"
        )
        _nifti(
            output / "topup_preparation/topup_corrected_b0_brain_mask.nii.gz",
            (3, 3, 3),
        )
        return {"status": "completed", "algorithm": "topup-eddy"}

    def fake_fit(_data, _bvals, _bvecs, _mask, tensor, **kwargs):
        directory = Path(tensor).parent
        _nifti(Path(tensor), (3, 3, 3, 6))
        _nifti(directory / "DTI_FA.nii.gz", (3, 3, 3))
        _nifti(directory / "DTI_sse.nii.gz", (3, 3, 3))
        _nifti(Path(kwargs["valid_mask_file"]), (3, 3, 3))
        _json(Path(kwargs["qa_file"]))

    monkeypatch.setattr(workflow, "run_topup_eddy_nifti", fake_topup_eddy)
    monkeypatch.setattr(workflow, "fit_dti_nifti", fake_fit)
    complete = workflow.Dwi2CondPipelineConfig(
        **{
            **config.__dict__,
            "preprocessing_mode": "eddy",
            "reverse_phase_encoding": reverse,
            "grad_dev": grad,
            "readout_seconds": 0.05,
            "phase_encoding_direction": "y",
        }
    )

    result = workflow.run_dwi2cond_pipeline(complete)

    assert result.stages[0].name == "fit_raw_dti_qa"
    assert result.stages[1].name == "preprocess_topup_eddy"
    assert (complete.output_directory / "preprocess/dti/grad_dev.nii").is_file()
    assert all(stage.status == "completed" for stage in result.stages)


def test_prefit_import_rejects_invalid_tensor_payloads(tmp_path: Path) -> None:
    invalid_shape = tmp_path / "invalid-shape.nii.gz"
    nonfinite = tmp_path / "nonfinite.nii.gz"
    _nifti(invalid_shape, (2, 2, 2, 5))
    values = np.zeros((2, 2, 2, 6), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(values, np.eye(4)), nonfinite)

    with pytest.raises(ValueError, match="shape"):
        workflow._import_prefit_tensor(invalid_shape, tmp_path / "shape-output")
    with pytest.raises(ValueError, match="NaN or Inf"):
        workflow._import_prefit_tensor(nonfinite, tmp_path / "finite-output")


def test_workflow_covers_all_new_input_contract_failures(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    tensor = tmp_path / "prefit.nii.gz"
    reverse = tmp_path / "reverse.nii.gz"
    field = tmp_path / "field.nii.gz"
    corrected_mask = tmp_path / "corrected-mask.nii.gz"
    external_mask = tmp_path / "external-mask.nii.gz"
    magnitude = tmp_path / "magnitude.nii.gz"
    radians = tmp_path / "radians.nii.gz"
    for path in (tensor, field, corrected_mask, external_mask, magnitude, radians):
        _nifti(path, (3, 3, 3, 6) if path == tensor else (3, 3, 3))
    _nifti(reverse, (3, 3, 3, 2))

    cases = (
        ({"fit_compatibility_mode": "bad"}, ValueError, "fit_compatibility_mode"),
        ({"solver": "mumps"}, ValueError, "solver is only consumed"),
        (
            {
                "preprocessing_mode": "prefit",
                "prefit_tensor": tensor,
                "data": None,
                "bvals": None,
                "bvecs": None,
                "fit_compatibility_mode": "robust",
            },
            ValueError,
            "not consumed by prefit",
        ),
        (
            {"preprocessing_mode": "nomoco", "prefit_tensor": tensor},
            ValueError,
            "only consumed by prefit",
        ),
        (
            {"preprocessing_mode": "nomoco", "random_seed": 9},
            ValueError,
            "only consumed by eddy",
        ),
        (
            {"preprocessing_mode": "prefit", "prefit_tensor": tmp_path / "missing"},
            FileNotFoundError,
            "pre-fitted tensor",
        ),
        (
            {"preprocessing_mode": "prefit", "prefit_tensor": tensor},
            ValueError,
            "raw-DWI inputs",
        ),
        (
            {
                "preprocessing_mode": "eddy",
                "reverse_phase_encoding": tmp_path / "missing-reverse",
                "readout_seconds": 0.05,
                "phase_encoding_direction": "y",
            },
            FileNotFoundError,
            "reverse phase-encoding",
        ),
        (
            {
                "preprocessing_mode": "eddy",
                "dwi_brain_mask": tmp_path / "missing-mask",
                "readout_seconds": 0.05,
                "phase_encoding_direction": "y",
            },
            FileNotFoundError,
            "external EDDY brain mask",
        ),
        (
            {
                "preprocessing_mode": "eddy",
                "reverse_phase_encoding": reverse,
                "dwi_brain_mask": external_mask,
                "readout_seconds": 0.05,
                "phase_encoding_direction": "y",
            },
            ValueError,
            "cannot use an external DWI mask",
        ),
        (
            {
                "preprocessing_mode": "legacy",
                "reverse_phase_encoding": reverse,
            },
            ValueError,
            "requires eddy preprocessing",
        ),
        (
            {"preprocessing_mode": "nomoco", "dwi_brain_mask": external_mask},
            ValueError,
            "only consumed by eddy",
        ),
        (
            {"preprocessing_mode": "nomoco", "readout_seconds": 0.05},
            ValueError,
            "only consumed by eddy",
        ),
        (
            {"preprocessing_mode": "nomoco", "phase_encoding_direction": "y"},
            ValueError,
            "not consumed by nomoco",
        ),
        (
            {
                "preprocessing_mode": "legacy",
                "phase_encoding_direction": "y",
            },
            ValueError,
            "only consumed by raw fieldmap",
        ),
        (
            {
                "preprocessing_mode": "eddy",
                "reverse_phase_encoding": reverse,
                "susceptibility_field": field,
                "readout_seconds": 0.05,
                "phase_encoding_direction": "y",
            },
            ValueError,
            "mutually exclusive",
        ),
        (
            {
                "preprocessing_mode": "eddy",
                "reverse_phase_encoding": reverse,
                "readout_seconds": 0.001,
                "phase_encoding_direction": "y",
            },
            ValueError,
            "within \\[0.01, 0.2\\]",
        ),
        (
            {
                "preprocessing_mode": "eddy",
                "reverse_phase_encoding": reverse,
                "readout_seconds": 0.05,
                "phase_encoding_direction": "z",
            },
            ValueError,
            "TOPUP subset",
        ),
        (
            {
                "preprocessing_mode": "legacy",
                "fieldmap_corrected_mask": tmp_path / "missing-mask",
            },
            FileNotFoundError,
            "corrected fieldmap mask",
        ),
        (
            {
                "fieldmap_magnitude": magnitude,
                "fieldmap_radians_per_second": radians,
                "fieldmap_dwell_milliseconds": 0.5,
                "phase_encoding_direction": "y",
            },
            ValueError,
            "legacy preprocessing",
        ),
        (
            {
                "preprocessing_mode": "legacy",
                "fieldmap_magnitude": magnitude,
                "fieldmap_radians_per_second": radians,
                "fieldmap_dwell_milliseconds": 0.5,
            },
            ValueError,
            "PE direction",
        ),
        (
            {
                "preprocessing_mode": "legacy",
                "fieldmap_magnitude": magnitude,
                "fieldmap_radians_per_second": radians,
                "fieldmap_dwell_milliseconds": 0.0,
                "phase_encoding_direction": "y",
            },
            ValueError,
            "dwell",
        ),
        (
            {
                "preprocessing_mode": "legacy",
                "fieldmap_magnitude": tmp_path / "missing-magnitude",
                "fieldmap_radians_per_second": radians,
                "fieldmap_dwell_milliseconds": 0.5,
                "phase_encoding_direction": "y",
            },
            FileNotFoundError,
            "raw fieldmap input",
        ),
        (
            {
                "preprocessing_mode": "legacy",
                "fieldmap_corrected_mask": corrected_mask,
            },
            ValueError,
            "requires legacy mode",
        ),
        (
            {"preprocessing_mode": "legacy", "susceptibility_field": field},
            ValueError,
            "requires the corrected fieldmap mask",
        ),
    )

    for changes, error_type, message in cases:
        invalid = workflow.Dwi2CondPipelineConfig(
            **{**config.__dict__, **changes}
        )
        with pytest.raises(error_type, match=message):
            workflow._validate_config(invalid)


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"preprocessing_mode": "bad"}, "preprocessing_mode"),
        ({"t1_mode": "bad"}, "t1_mode"),
        ({"fem_smoke": "bad"}, "fem_smoke"),
        ({"workers": 0}, "workers"),
        ({"preprocessing_mode": "nomoco", "susceptibility_field": Path("missing")}, "only valid"),
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
    resolved = workflow._required_m2m_files(config.m2m_directory, require_fem=True)
    assert resolved["mesh"].name == "only.msh"
    assert resolved["eeg"].name == "custom-10-10.csv"
    resolved["t1"].unlink()
    with pytest.raises(FileNotFoundError, match="t1"):
        workflow._required_m2m_files(config.m2m_directory)

    config = _fixture(tmp_path / "missing-sentinel")
    config.m2m_directory.joinpath("final_tissues.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match="final_tissues"):
        workflow._required_m2m_files(config.m2m_directory, require_fem=False)


def test_fem_rejects_non_charm_directory_name(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    renamed = tmp_path / "subject"
    config.m2m_directory.rename(renamed)
    with pytest.raises(ValueError, match=r"m2m_<subject>"):
        workflow._required_m2m_files(renamed, require_fem=True)


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


def test_tensor_publication_rejects_copy_and_provenance_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tensor.nii.gz"
    source.write_bytes(b"tensor")
    m2m = tmp_path / "m2m"
    m2m.mkdir()

    hashes = iter(("source", "destination"))
    monkeypatch.setattr(workflow, "_sha256", lambda _path: next(hashes))
    with pytest.raises(RuntimeError, match="does not match its source"):
        workflow._publish_tensor_to_m2m(source, m2m, "0.3.0")

    hashes = iter(("same", "same", "changed"))
    monkeypatch.setattr(workflow, "_sha256", lambda _path: next(hashes))
    with pytest.raises(RuntimeError, match="provenance SHA-256 is inconsistent"):
        workflow._publish_tensor_to_m2m(source, m2m, "0.3.0")

    monkeypatch.setattr(workflow, "_sha256", lambda _path: "same")
    monkeypatch.setattr(workflow.json, "loads", lambda _text: {"sha256": "changed"})
    with pytest.raises(RuntimeError, match="provenance SHA-256 is inconsistent"):
        workflow._publish_tensor_to_m2m(source, m2m, "0.3.0")


def test_simnibs_runtime_identity_records_missing_distribution(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow.importlib.metadata,
        "distribution",
        lambda _name: (_ for _ in ()).throw(
            workflow.importlib.metadata.PackageNotFoundError("simnibs")
        ),
    )
    identity = workflow._simnibs_runtime_identity()
    assert identity["distribution_version"] == "not-installed"
    assert identity["cache_safe"] is False
    with pytest.raises(ValueError, match="Unsupported FEM solver identity"):
        workflow._simnibs_runtime_identity("unknown")


def test_simnibs_runtime_identity_records_installed_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in (
        "simnibs/simulation/run_simnibs.py",
        "simnibs/simulation/sim_struct.py",
        "simnibs/simulation/fem.py",
        "simnibs/simulation/pardiso.py",
        "simnibs/utils/cond_utils.py",
        "simnibs/mesh_tools/mesh_io.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")

    class Distribution:
        version = "4.6.0"

        def locate_file(self, relative: str) -> Path:
            return tmp_path / relative

        def read_text(self, name: str) -> str | None:
            return "record\n" if name == "RECORD" else None

    monkeypatch.setattr(
        workflow.importlib.metadata, "distribution", lambda _name: Distribution()
    )
    native = tmp_path / "libmkl_rt.so"
    native.write_bytes(b"mkl")
    monkeypatch.setattr(
        workflow,
        "_solver_native_identity",
        lambda _solver: [
            {
                "path": str(native),
                "size_bytes": native.stat().st_size,
                "sha256": workflow._sha256(native),
            }
        ],
    )

    identity = workflow._simnibs_runtime_identity()
    assert identity["distribution_version"] == "4.6.0"
    assert identity["record_sha256"] == workflow.hashlib.sha256(b"record\n").hexdigest()
    assert identity["cache_safe"] is True

    before = identity["module_files"]["simnibs/simulation/fem.py"]["sha256"]
    (tmp_path / "simnibs/simulation/fem.py").write_text("changed\n", encoding="utf-8")
    after = workflow._simnibs_runtime_identity()["module_files"][
        "simnibs/simulation/fem.py"
    ]["sha256"]
    assert after != before

    (tmp_path / "simnibs/simulation/fem.py").unlink()
    incomplete = workflow._simnibs_runtime_identity()
    assert incomplete["module_files"]["simnibs/simulation/fem.py"] is None
    assert incomplete["cache_safe"] is False


def test_tensor_publication_failure_restores_previous_valid_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "new-tensor.nii.gz"
    source.write_bytes(b"new-tensor")
    m2m = tmp_path / "m2m"
    m2m.mkdir()
    tensor = m2m / "DTI_coregT1_tensor.nii.gz"
    provenance = m2m / "DTI_coregT1_tensor.provenance.json"
    tensor.write_bytes(b"old-tensor")
    provenance.write_text('{"sha256":"old"}\n', encoding="utf-8")
    original_replace = Path.replace

    def fail_provenance_commit(path: Path, target: Path):
        if path.name.startswith(".DTI_coregT1_tensor.provenance.json") and path.name.endswith(
            ".tmp"
        ):
            raise OSError("injected provenance commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_provenance_commit)
    with pytest.raises(OSError, match="injected provenance"):
        workflow._publish_tensor_to_m2m(source, m2m, "0.3.0")

    assert tensor.read_bytes() == b"old-tensor"
    assert provenance.read_text(encoding="utf-8") == '{"sha256":"old"}\n'
