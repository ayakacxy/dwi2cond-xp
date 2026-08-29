from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from dwi2cond_xp import simnibs_adapter


class _ElementData:
    sampled_values = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 2.0, 0.0, 2.0],
            [3.0, 0.0, 0.0, 3.0, 0.0, 3.0],
        ]
    )

    def __init__(self, value, name, mesh=None) -> None:
        self.value = np.asarray(value)
        self.name = name
        self.mesh = mesh
        self.triangles_assigned = False

    @classmethod
    def from_data_grid(cls, *args, **kwargs):
        del args, kwargs
        return cls(cls.sampled_values, "sampled")

    def assign_triangle_values(self) -> None:
        self.triangles_assigned = True


def _install_fake_simnibs(monkeypatch, tmp_path: Path):
    mesh = SimpleNamespace(
        elm=SimpleNamespace(
            nr=3,
            tag1=np.array([1, 7, 2]),
            get_tetrahedra=lambda: np.array([0, 2]),
        ),
        elmdata=[],
        elements_volumes_and_areas=lambda: SimpleNamespace(
            value=np.array([1.0, 5.0, 2.0])
        ),
    )
    mesh_io = ModuleType("simnibs.mesh_tools.mesh_io")
    mesh_io.ElementData = _ElementData
    mesh_io.read_msh = lambda path: mesh
    mesh_io.write_msh = lambda loaded, path: Path(path).write_text("mesh", encoding="utf-8")
    mesh_tools = ModuleType("simnibs.mesh_tools")
    mesh_tools.mesh_io = mesh_io
    cond_utils = ModuleType("simnibs.utils.cond_utils")
    cond_utils.standard_cond = lambda: []
    utils = ModuleType("simnibs.utils")
    utils.cond_utils = cond_utils
    simnibs = ModuleType("simnibs")
    simnibs.mesh_tools = mesh_tools
    simnibs.utils = utils
    for name, module in {
        "simnibs": simnibs,
        "simnibs.mesh_tools": mesh_tools,
        "simnibs.mesh_tools.mesh_io": mesh_io,
        "simnibs.utils": utils,
        "simnibs.utils.cond_utils": cond_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return mesh


def test_tensor_to_mesh_writes_conductivity_and_qa(monkeypatch, tmp_path: Path) -> None:
    mesh = _install_fake_simnibs(monkeypatch, tmp_path)
    image = SimpleNamespace(shape=(2, 2, 2, 6), dataobj=np.zeros((2, 2, 2, 6)), affine=np.eye(4))
    monkeypatch.setattr(simnibs_adapter.nib, "load", lambda path: image)
    monkeypatch.setattr(simnibs_adapter, "correct_fsl_tensor_basis", lambda matrix, affine: matrix)

    def fake_conversion(matrices, tags, scalar, **kwargs):
        assert matrices.shape == (2, 3, 3)
        assert tags.tolist() == [1, 2]
        assert scalar == {1: 0.1, 2: 0.2}
        return np.repeat(np.eye(3)[None, :, :], 2, axis=0), {"mode": kwargs["mode"]}

    monkeypatch.setattr(simnibs_adapter, "tensors_to_conductivity", fake_conversion)
    output = tmp_path / "conductivity.msh"
    qa = tmp_path / "qa.json"
    result = simnibs_adapter.tensor_to_mesh_conductivity(
        "tensor.nii.gz",
        "head.msh",
        output,
        mode="vn",
        scalar_conductivity={1: 0.1, 2: 0.2},
        vn_singular_policy="regularize",
        qa_file=qa,
    )

    assert result == output
    assert output.read_text(encoding="utf-8") == "mesh"
    assert mesh.elmdata[-1].value.shape == (3, 9)
    assert mesh.elmdata[-1].triangles_assigned
    report = json.loads(qa.read_text(encoding="utf-8"))
    assert report["tetrahedra"] == 2
    assert report["mode"] == "vn"
    assert report["vn_singular_policy"] == "regularize"


def test_tensor_to_mesh_counts_true_tetrahedra_in_boolean_mask(
    monkeypatch, tmp_path: Path
) -> None:
    mesh = _install_fake_simnibs(monkeypatch, tmp_path)
    mesh.elm.get_tetrahedra = lambda: np.array([True, False, True])
    image = SimpleNamespace(
        shape=(2, 2, 2, 6), dataobj=np.zeros((2, 2, 2, 6)), affine=np.eye(4)
    )
    monkeypatch.setattr(simnibs_adapter.nib, "load", lambda path: image)
    monkeypatch.setattr(simnibs_adapter, "correct_fsl_tensor_basis", lambda matrix, affine: matrix)
    monkeypatch.setattr(
        simnibs_adapter,
        "tensors_to_conductivity",
        lambda matrices, *_args, **_kwargs: (
            np.repeat(np.eye(3)[None], matrices.shape[0], axis=0),
            {},
        ),
    )
    qa = tmp_path / "bool_qa.json"

    simnibs_adapter.tensor_to_mesh_conductivity(
        "tensor", "mesh", tmp_path / "bool.msh", qa_file=qa
    )

    assert json.loads(qa.read_text(encoding="utf-8"))["tetrahedra"] == 2


def test_tensor_to_mesh_rejects_non_six_component_image(monkeypatch, tmp_path: Path) -> None:
    _install_fake_simnibs(monkeypatch, tmp_path)
    image = SimpleNamespace(shape=(2, 2, 2, 5), dataobj=np.zeros((2, 2, 2, 5)), affine=np.eye(4))
    monkeypatch.setattr(simnibs_adapter.nib, "load", lambda path: image)
    with pytest.raises(ValueError, match="six components"):
        simnibs_adapter.tensor_to_mesh_conductivity("tensor", "mesh", tmp_path / "out.msh")


def test_tensor_to_mesh_requires_simnibs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "simnibs", None)
    monkeypatch.delitem(sys.modules, "simnibs.mesh_tools", raising=False)
    monkeypatch.delitem(sys.modules, "simnibs.mesh_tools.mesh_io", raising=False)
    with pytest.raises(RuntimeError, match="requires SimNIBS"):
        simnibs_adapter.tensor_to_mesh_conductivity("tensor", "mesh", tmp_path / "out.msh")


def test_tensor_to_mesh_uses_standard_conductivities(monkeypatch, tmp_path: Path) -> None:
    _install_fake_simnibs(monkeypatch, tmp_path)
    cond_utils = sys.modules["simnibs.utils.cond_utils"]
    cond_utils.standard_cond = lambda: [SimpleNamespace(value=0.1), SimpleNamespace(value=0.2)]
    image = SimpleNamespace(
        shape=(2, 2, 2, 6), dataobj=np.zeros((2, 2, 2, 6)), affine=np.eye(4)
    )
    monkeypatch.setattr(simnibs_adapter.nib, "load", lambda path: image)

    def fake_conversion(matrices, tags, scalar, **kwargs):
        assert scalar == {1: 0.1, 2: 0.2}
        return np.repeat(np.eye(3)[None], 2, axis=0), {"mode": kwargs["mode"]}

    monkeypatch.setattr(simnibs_adapter, "tensors_to_conductivity", fake_conversion)
    output = tmp_path / "standard.msh"
    assert simnibs_adapter.tensor_to_mesh_conductivity(
        "tensor", "mesh", output, correct_fsl=False
    ) == output
    assert output.with_suffix(".conductivity.json").is_file()
