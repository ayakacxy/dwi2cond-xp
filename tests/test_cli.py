from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np

from dwi2cond_xp import cli


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

    def set_postfix_str(self, phase: str) -> None:
        self.phase = phase

    def close(self) -> None:
        self.closed = True


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

    assert cli.main(
        ["register-tensor", "tensor", "reference", "output", "--world-transform", str(transform)]
    ) == 0
    assert cli.main(
        ["tensor-to-mesh", "tensor", "mesh", "output", "--mode", "mc", "--cond-json", str(cond)]
    ) == 0
    assert np.array_equal(calls["register"][1]["world_transform"], np.eye(4))
    assert calls["register"][1]["alignment_assumption"] == "external_world_transform"
    assert calls["mesh"][1]["scalar_conductivity"] == {1: 0.12, 2: 0.27}


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
    assert cli.main(["simulate-tdcs", "m2m_test", "out", "--mode", "scalar", "--dry-run"]) == 0
    assert cli.main(["simulate-leadfield", "m2m_test", "out", "--mode", "vn", "--dry-run"]) == 0
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
    assert cli.main(
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
    ) == 0
    assert reports["montage"][1]["montage_name"] == "standard_1020"
    assert reports["compare"][1]["view"] == "magnitude"
    assert reports["compare"][0][0]["mc"] == "mc.nii.gz"


def test_module_entrypoint_delegates_to_cli(monkeypatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 0)
    with np.testing.assert_raises_regex(SystemExit, "0"):
        runpy.run_module("dwi2cond_xp.__main__", run_name="__main__")
