from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dwi2cond_xp import cli
from dwi2cond_xp import preprocessing
from dwi2cond_xp.preprocessing import workflow


class _Progress:
    instances: list["_Progress"] = []

    def __init__(self, *, total: int, **kwargs) -> None:
        del kwargs
        self.total = total
        self.n = 0
        self.closed = False
        self.phase = ""
        self.instances.append(self)

    def update(self, amount: int) -> None:
        self.n += amount

    def set_postfix_str(self, phase: str, *, refresh: bool = True) -> None:
        del refresh
        self.phase = phase

    def close(self) -> None:
        self.closed = True

    def write(self, _message: str) -> None:
        pass


def test_disabled_lazy_progress_does_not_import_tqdm() -> None:
    progress = cli._LazyTqdm()(total=2, initial=1, disable=True)
    progress.update(1)
    progress.set_postfix_str("complete", refresh=False)
    progress.close()
    assert progress.n == 2


def test_enabled_lazy_progress_loads_tqdm() -> None:
    progress = cli._LazyTqdm()(total=0, disable=False)
    progress.close()


def test_lazy_cli_callable_imports_once(monkeypatch) -> None:
    imports: list[tuple[str, str | None]] = []
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def implementation(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "loaded"

    def fake_import(module: str, package: str | None):
        imports.append((module, package))
        return SimpleNamespace(implementation=implementation)

    monkeypatch.setattr(cli, "import_module", fake_import)
    lazy = cli._LazyCallable(".module", "implementation")
    assert lazy(1, option=2) == "loaded"
    assert lazy(3) == "loaded"
    assert imports == [(".module", "dwi2cond_xp")]
    assert calls == [((1,), {"option": 2}), ((3,), {})]


def test_preprocessing_lazy_exports_and_missing_attribute() -> None:
    preprocessing.__dict__.pop("run_topup_nifti", None)
    from dwi2cond_xp.preprocessing.topup import run_topup_nifti

    assert "run_topup_nifti" in dir(preprocessing)
    assert preprocessing.run_topup_nifti is run_topup_nifti
    assert preprocessing.__dict__["run_topup_nifti"] is run_topup_nifti
    with pytest.raises(AttributeError, match="missing_export"):
        getattr(preprocessing, "missing_export")


def test_pipeline_qa_rejects_invalid_and_duplicate_fem_manifest_modes() -> None:
    defaults = cli._build_parser().parse_args(
        ["pipeline-qa", "bvals", "bvecs", "mask", "fa", "tensor", "valid", "out"]
    )
    assert defaults.b0_threshold == 0.0
    positional = ["bvals", "bvecs", "mask", "fa", "tensor", "valid", "output"]
    with pytest.raises(ValueError, match="scalar/vn/dir/mc=PATH"):
        cli.main(["pipeline-qa", *positional, "--fem-manifest", "bad"])
    with pytest.raises(ValueError, match="Duplicate FEM manifest"):
        cli.main(
            [
                "pipeline-qa",
                *positional,
                "--fem-manifest",
                "scalar=first.json",
                "--fem-manifest",
                "scalar=second.json",
            ]
        )


def test_select_shell_and_fit_routes(monkeypatch, capsys) -> None:
    selected: dict[str, object] = {}

    def fake_select(*args, **kwargs):
        selected["call"] = (args, kwargs)
        return np.array([0, 2, 4])

    def fake_fit(*args, progress, **kwargs):
        selected["fit"] = (args, kwargs)
        progress(3, 3, 1)

    monkeypatch.setattr(cli, "select_shell_nifti", fake_select)
    monkeypatch.setattr(cli, "fit_dti_nifti", fake_fit)
    monkeypatch.setattr(cli, "tqdm", _Progress)

    assert cli.main(["select-shell", "dwi", "bval", "bvec", "out", "ob", "ov"]) == 0
    assert cli.main(["fit-dti", "dwi", "bval", "bvec", "mask", "tensor"]) == 0
    assert selected["call"][1]["shell"] == 1000.0
    assert selected["fit"][1]["workers"] == 1
    assert _Progress.instances[-1].n == 3
    assert _Progress.instances[-1].closed
    assert "selected 3 volumes" in capsys.readouterr().out


def test_nomoco_route_uses_eight_workers_and_visible_stages(
    monkeypatch, capsys
) -> None:
    calls: dict[str, object] = {}

    def fake_nomoco(*args, progress, **kwargs):
        calls["call"] = (args, kwargs)
        progress("align_b0", 1, 2)
        progress("align_b0", 2, 2)
        progress("fit_dti", 3, 3)
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_nomoco_nifti", fake_nomoco)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    assert cli.main(["preprocess-nomoco", "dwi", "bval", "bvec", "out"]) == 0
    assert calls["call"][1]["workers"] == 8
    assert _Progress.instances[-2].n == 2
    assert _Progress.instances[-1].n == 3
    assert all(instance.closed for instance in _Progress.instances[-2:])
    assert "nomoco_qa.json" in capsys.readouterr().out


def test_legacy_route_uses_compat46_and_visible_stages(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_legacy(*args, progress, **kwargs):
        calls["call"] = (args, kwargs)
        progress("legacy_pass1_6dof", 1, 2)
        progress("legacy_pass1_6dof", 2, 2)
        progress("final_resample", 4, 4)
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_legacy_nifti", fake_legacy)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    assert cli.main(["preprocess-legacy", "dwi", "bval", "bvec", "out"]) == 0
    assert calls["call"][1]["workers"] == 8
    assert calls["call"][1]["bvec_mode"] == "compat46"
    assert _Progress.instances[-2].n == 2
    assert _Progress.instances[-1].n == 4
    assert all(instance.closed for instance in _Progress.instances[-2:])
    assert "legacy_qa.json" in capsys.readouterr().out


def test_fieldmap_route_preserves_units_direction_and_worker_contract(
    monkeypatch, capsys
) -> None:
    calls: dict[str, object] = {}

    def fake_fieldmap(*args, progress, **kwargs):
        calls["call"] = (args, kwargs)
        progress("registration", 2, 4)
        progress("complete", 4, 4)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_fieldmap_nifti", fake_fieldmap)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    assert (
        cli.main(
            [
                "prepare-fieldmap",
                "magnitude",
                "field-rad-s",
                "b0-brain",
                "output",
                "--dwell-ms",
                "0.5",
                "--phase-encoding-direction",
                "y-",
                "--magnitude-mask",
                "magnitude-mask",
                "--b0-mask",
                "b0-mask",
                "--no-median-filter",
                "--progress",
                "off",
            ]
        )
        == 0
    )
    args, kwargs = calls["call"]
    assert args == ("magnitude", "field-rad-s", "b0-brain", "output")
    assert kwargs["dwell_milliseconds"] == 0.5
    assert kwargs["phase_encoding_direction"] == "y-"
    assert kwargs["magnitude_mask_file"] == "magnitude-mask"
    assert kwargs["b0_mask_file"] == "b0-mask"
    assert kwargs["workers"] == 8
    assert kwargs["median_filter"] is False
    assert _Progress.instances[-1].n == 4
    assert _Progress.instances[-1].closed
    assert "fieldmap_qa.json" in capsys.readouterr().out


def test_standalone_eddy_accepts_z_phase_encoding(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_eddy(*args, progress, **kwargs):
        calls["call"] = (args, kwargs)
        progress("complete", 1, 1)
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_eddy_nifti", fake_eddy)
    monkeypatch.setattr(cli, "tqdm", _Progress)

    status = cli.main(
        [
            "prepare-eddy",
            "dwi",
            "bvals",
            "bvecs",
            "mask",
            "output",
            "--readout-seconds",
            "0.05",
            "--phase-encoding-direction",
            "z-",
            "--progress",
            "off",
        ]
    )

    assert status == 0
    assert calls["call"][1]["phase_encoding_direction"] == "z-"
    assert "eddy_qa.json" in capsys.readouterr().out


def test_registration_and_mesh_routes(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_register(*args, progress, **kwargs):
        calls["register"] = (args, kwargs)
        progress(8, 8)

    def fake_mesh(*args, **kwargs):
        calls["mesh"] = (args, kwargs)

    monkeypatch.setattr(cli, "register_tensor_affine", fake_register)
    monkeypatch.setattr(cli, "tensor_to_mesh_conductivity", fake_mesh)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    transform = tmp_path / "affine.txt"
    np.savetxt(transform, np.eye(4))
    cond = tmp_path / "conductivity.json"
    cond.write_text(json.dumps({"1": 0.12, "2": 0.27}), encoding="utf-8")

    assert (
        cli.main(
            [
                "register-tensor",
                "tensor",
                "reference",
                "output",
                "--world-transform",
                str(transform),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "tensor-to-mesh",
                "tensor",
                "mesh",
                "output",
                "--mode",
                "vn",
                "--cond-json",
                str(cond),
                "--vn-singular-policy",
                "regularize",
                "--eigensystem-mode",
                "simnibs46-literal",
            ]
        )
        == 0
    )
    assert np.array_equal(calls["register"][1]["world_transform"], np.eye(4))
    assert calls["register"][1]["alignment_assumption"] == "external_world_transform"
    assert calls["mesh"][1]["scalar_conductivity"] == {1: 0.12, 2: 0.27}
    assert calls["mesh"][1]["vn_singular_policy"] == "regularize"
    assert calls["mesh"][1]["eigensystem_mode"] == "simnibs46-literal"


def test_automatic_t1_registration_route_uses_charm_contract(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    calls: dict[str, object] = {}
    dti = tmp_path / "dti"
    m2m = tmp_path / "m2m_subject"
    segmentation = m2m / "segmentation"
    dti.mkdir()
    segmentation.mkdir(parents=True)
    for path in (
        dti / "DTI_tensor.nii.gz",
        dti / "DTI_FA.nii.gz",
        dti / "DTI_sse.nii.gz",
        m2m / "T1.nii.gz",
        m2m / "final_tissues.nii.gz",
        segmentation / "labeling.nii.gz",
        segmentation / "T1_bias_corrected.nii.gz",
    ):
        path.touch()

    def fake_t1(*args, progress, **kwargs):
        calls["call"] = (args, kwargs)
        progress("prepare_t1", 1, 1)
        progress("primary_complete", 1, 1)
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_t1_registration_nifti", fake_t1)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    output = tmp_path / "output"
    assert (
        cli.main(["register-t1", str(dti), str(m2m), str(output), "--mode", "rigid"])
        == 0
    )
    assert calls["call"][1]["degrees_of_freedom"] == 6
    assert calls["call"][1]["workers"] == 8
    assert calls["call"][1]["sse_file"] == dti / "DTI_sse.nii.gz"
    assert all(instance.closed for instance in _Progress.instances[-2:])
    assert "t1_registration_qa.json" in capsys.readouterr().out
    assert (
        cli.main(
            [
                "register-t1",
                str(dti),
                str(m2m),
                str(output),
                "--progress",
                "off",
            ]
        )
        == 0
    )


def test_automatic_t1_registration_requires_charm_outputs(tmp_path: Path) -> None:
    dti = tmp_path / "dti"
    m2m = tmp_path / "m2m"
    dti.mkdir()
    m2m.mkdir()
    with pytest.raises(FileNotFoundError, match="final_tissues"):
        cli.main(["register-t1", str(dti), str(m2m), str(tmp_path / "out")])


def test_nonlinear_t1_registration_route_uses_fixed_contract(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    calls: dict[str, object] = {}

    def fake_nonlinear(*args, progress, **kwargs):
        calls["call"] = (args, kwargs)
        for level in range(1, 5):
            progress(level, "gradient", 1, 5, None)
            progress(level, "hessian", 1, 5, None)
            progress(level, "pcg", 7, 500, 1.0e-3)
            progress(level, "lm", 1, 5, 0.5)
            progress(level, "topology", 1, 2, 0.5)
            progress(level, "topology", 2, 2, 1.0)
            progress(level, "complete", 1, 1, 0.0)
        progress(4, "finalize_write", 0, 3, None)
        progress(4, "finalize_tensor", 1, 3, None)
        progress(4, "finalize_qa", 2, 3, None)
        progress(4, "finalize_complete", 3, 3, None)
        return {"status": "completed"}

    monkeypatch.setattr(cli, "register_tensor_fnirt_nifti", fake_nonlinear)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    first_progress = len(_Progress.instances)
    output = tmp_path / "nonlinear"
    assert (
        cli.main(
            [
                "register-t1-nonlinear",
                "fa.nii.gz",
                "tensor.nii.gz",
                "T1_brain.nii.gz",
                "FA2T1.mat",
                str(output),
                "--brain-mask",
                "T1_brainmask.nii.gz",
                "--workers",
                "8",
            ]
        )
        == 0
    )
    assert calls["call"] == (
        (
            "fa.nii.gz",
            "tensor.nii.gz",
            "T1_brain.nii.gz",
            "FA2T1.mat",
            str(output),
        ),
            {
                "brain_mask_file": "T1_brainmask.nii.gz",
                "workers": 8,
                "compatibility_mode": "strict-fsl",
            },
    )
    created = _Progress.instances[first_progress:]
    assert created[0].n == 4
    assert created[0].closed
    assert len(created) == 6
    assert all(progress.n == 5 and progress.closed for progress in created[1:5])
    assert all(progress.total == 5 for progress in created[1:5])
    assert created[5].total == 3
    assert created[5].n == 3
    assert created[5].closed
    assert "nonlinear_registration_qa.json" in capsys.readouterr().out


def test_nonlinear_progress_closes_replaced_and_failed_detail_bars(
    monkeypatch, tmp_path: Path
) -> None:
    def fail_after_progress(*args, progress, **kwargs):
        del args, kwargs
        progress(1, "gradient", 1, 5, None)
        progress(2, "gradient", 1, 5, None)
        progress(4, "finalize_write", 0, 3, None)
        raise RuntimeError("expected progress failure")

    monkeypatch.setattr(cli, "register_tensor_fnirt_nifti", fail_after_progress)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    first_progress = len(_Progress.instances)
    with pytest.raises(RuntimeError, match="expected progress failure"):
        cli.main(
            [
                "register-t1-nonlinear",
                "fa.nii.gz",
                "tensor.nii.gz",
                "T1_brain.nii.gz",
                "FA2T1.mat",
                str(tmp_path / "nonlinear"),
                "--brain-mask",
                "T1_brainmask.nii.gz",
            ]
        )
    created = _Progress.instances[first_progress:]
    assert len(created) == 4
    assert all(progress.closed for progress in created)


def test_pipeline_fnirt_progress_reuses_one_bar_per_level(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_pipeline(config, *, progress):
        del config
        progress("register_nonlinear", 0, 1, "running")
        progress("register_nonlinear:level_1:gradient", 1, 5, "gradient")
        progress("register_nonlinear:level_1:hessian", 1, 5, "hessian")
        progress("register_nonlinear:level_1:pcg", 11, 500, "pcg; value=0.01")
        progress("register_nonlinear:level_1:lm", 1, 5, "lm; value=1")
        progress("register_nonlinear:level_1:complete", 1, 1, "complete")
        progress("register_nonlinear:level_4:finalize_write", 0, 3, "running")
        progress("register_nonlinear:level_4:finalize_tensor", 1, 3, "running")
        progress("register_nonlinear:level_4:finalize_qa", 2, 3, "running")
        progress("register_nonlinear:level_4:finalize_complete", 3, 3, "running")
        progress("register_nonlinear", 1, 1, "completed")
        return type(
            "PipelineResult",
            (),
            {
                "qa_manifest": tmp_path / "pipeline_qa.json",
                "final_tensor": tmp_path / "tensor.nii.gz",
            },
        )()

    monkeypatch.setattr(workflow, "run_dwi2cond_pipeline", fake_pipeline)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    first_progress = len(_Progress.instances)
    assert (
        cli.main(
            [
                "run-pipeline",
                "dwi.nii.gz",
                "bvals",
                "bvecs",
                "m2m_subject",
                str(tmp_path / "output"),
                "--t1-mode",
                "nonlinear",
            ]
        )
        == 0
    )
    created = _Progress.instances[first_progress:]
    assert len(created) == 3
    assert created[0].n == 1
    assert created[0].closed
    assert created[1].total == 5
    assert created[1].n == 5
    assert created[1].closed
    assert created[2].total == 3
    assert created[2].n == 3
    assert created[2].closed


def test_pipeline_progress_closes_each_replaced_bar_after_failure(
    monkeypatch, tmp_path: Path
) -> None:
    def fail_after_progress(config, *, progress):
        del config
        progress("register_nonlinear", 0, 1, "running")
        progress("register_nonlinear:level_1:gradient", 1, 5, "gradient")
        progress("register_nonlinear:level_2:gradient", 1, 5, "gradient")
        progress("other:detail", 1, 2, "running")
        progress("register_nonlinear:level_4:finalize_write", 0, 3, "running")
        raise RuntimeError("expected pipeline failure")

    monkeypatch.setattr(workflow, "run_dwi2cond_pipeline", fail_after_progress)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    first_progress = len(_Progress.instances)
    with pytest.raises(RuntimeError, match="expected pipeline failure"):
        cli.main(
            [
                "run-pipeline",
                "dwi.nii.gz",
                "bvals",
                "bvecs",
                "m2m_subject",
                str(tmp_path / "output"),
                "--t1-mode",
                "nonlinear",
            ]
        )
    created = _Progress.instances[first_progress:]
    assert len(created) == 5
    assert all(progress.closed for progress in created)


def test_pipeline_qa_progress_reuses_one_bar_for_all_phases(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_pipeline(config, *, progress):
        del config
        progress("pipeline_qa", 0, 1, "running")
        for done, phase in enumerate(
            (
                "core_inputs",
                "raw_dwi",
                "corrected_dwi",
                "gradients",
                "tensor",
                "field_motion_eddy",
                "registration_fem",
                "complete",
            ),
            start=1,
        ):
            progress(f"pipeline_qa:{phase}", done, 8, "running")
        progress("pipeline_qa", 1, 1, "completed")
        return type(
            "PipelineResult",
            (),
            {
                "qa_manifest": tmp_path / "pipeline_qa.json",
                "final_tensor": tmp_path / "tensor.nii.gz",
            },
        )()

    monkeypatch.setattr(workflow, "run_dwi2cond_pipeline", fake_pipeline)
    monkeypatch.setattr(cli, "tqdm", _Progress)
    first_progress = len(_Progress.instances)
    assert (
        cli.main(
            [
                "run-pipeline",
                "dwi.nii.gz",
                "bvals",
                "bvecs",
                "m2m_subject",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    created = _Progress.instances[first_progress:]
    assert len(created) == 2
    assert created[0].n == 1
    assert created[0].closed
    assert created[1].total == 8
    assert created[1].n == 8
    assert created[1].closed


def test_mask_simulation_and_leadfield_routes(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "make_charm_brain_mask",
        lambda *args, **kwargs: calls.update(mask=(args, kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "run_tdcs",
        lambda *args, **kwargs: {
            "status": "planned",
            "output_directory": "/tmp/tdcs",
        },
    )

    def fake_leadfield(*args, progress, **kwargs):
        calls["leadfield"] = (args, kwargs)
        progress(2, 2, "validated")
        return {"status": "planned", "output_directory": "/tmp/leadfield"}

    monkeypatch.setattr(cli, "run_tdcs_leadfield", fake_leadfield)
    monkeypatch.setattr(cli, "tqdm", _Progress)

    assert cli.main(["charm-brain-mask", "labels", "mask"]) == 0
    assert (
        cli.main(["simulate-tdcs", "m2m_test", "out", "--mode", "scalar", "--dry-run"])
        == 0
    )
    assert (
        cli.main(["simulate-leadfield", "m2m_test", "out", "--mode", "vn", "--dry-run"])
        == 0
    )
    assert calls["mask"][0] == ("labels", "mask")
    assert calls["leadfield"][1]["export_matrix"] is True
    assert _Progress.instances[-1].phase == "validated"
    assert "planned: /tmp/leadfield" in capsys.readouterr().out


def test_plotting_routes(monkeypatch, tmp_path: Path) -> None:
    reports: dict[str, object] = {}

    def fake_montage(*args, **kwargs):
        reports["montage"] = (args, kwargs)
        return {"output_png": str(tmp_path / "montage.png")}

    def fake_compare(*args, **kwargs):
        reports["compare"] = (args, kwargs)
        return {"slice_index": 17}

    monkeypatch.setattr(cli, "plot_montage_schematic", fake_montage)
    monkeypatch.setattr(cli, "plot_field_comparison", fake_compare)

    assert cli.main(["plot-montage", str(tmp_path / "montage.png")]) == 0
    assert (
        cli.main(
            [
                "compare-fields",
                "scalar.nii.gz",
                "vn.nii.gz",
                "dir.nii.gz",
                "mc.nii.gz",
                "T1.nii.gz",
                "mask.nii.gz",
                str(tmp_path / "fields.png"),
                "--view",
                "magnitude",
            ]
        )
        == 0
    )
    assert reports["montage"][1]["montage_name"] == "standard_1020"
    assert reports["compare"][1]["view"] == "magnitude"
    assert reports["compare"][0][0]["mc"] == "mc.nii.gz"


def test_module_entrypoint_delegates_to_cli(monkeypatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 0)
    with np.testing.assert_raises_regex(SystemExit, "0"):
        runpy.run_module("dwi2cond_xp.__main__", run_name="__main__")


def test_unimplemented_command_is_rejected(monkeypatch) -> None:
    parser = SimpleNamespace(
        parse_args=lambda argv: SimpleNamespace(command="future-command")
    )
    monkeypatch.setattr(cli, "_build_parser", lambda: parser)
    with pytest.raises(RuntimeError, match="not implemented"):
        cli.main([])


def test_fit_route_can_disable_progress(monkeypatch) -> None:
    calls = []

    def fake_fit(*args, progress, **kwargs):
        del args, kwargs
        progress(1, 1, 1)
        calls.append("fit")

    monkeypatch.setattr(cli, "fit_dti_nifti", fake_fit)
    assert (
        cli.main(
            ["fit-dti", "dwi", "bval", "bvec", "mask", "tensor", "--progress", "off"]
        )
        == 0
    )
    assert calls == ["fit"]


def test_nomoco_route_can_disable_progress(monkeypatch) -> None:
    calls = []

    def fake_nomoco(*args, progress, **kwargs):
        del args, kwargs
        progress("fit_dti", 1, 1)
        calls.append("nomoco")
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_nomoco_nifti", fake_nomoco)
    assert (
        cli.main(
            ["preprocess-nomoco", "dwi", "bval", "bvec", "out", "--progress", "off"]
        )
        == 0
    )
    assert calls == ["nomoco"]


def test_legacy_route_can_disable_progress(monkeypatch) -> None:
    calls = []

    def fake_legacy(*args, progress, **kwargs):
        del args, kwargs
        progress("fit_dti", 1, 1)
        calls.append("legacy")
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_legacy_nifti", fake_legacy)
    assert (
        cli.main(
            ["preprocess-legacy", "dwi", "bval", "bvec", "out", "--progress", "off"]
        )
        == 0
    )
    assert calls == ["legacy"]
