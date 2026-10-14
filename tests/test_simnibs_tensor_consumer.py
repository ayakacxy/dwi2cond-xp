from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.conductivity import fsl_tensor_basis
from dwi2cond_xp.registration import matrix_to_tensor6
from dwi2cond_xp.simnibs_tensor_consumer import (
    _clone_function, bind_simnibs_tensor_consumer, describe_simnibs_tensor_consumer,
)


def _tensor_input(path: Path, angle: float = 45.0) -> tuple[Path, np.ndarray]:
    """构建含空间变化及不同组织张量的非等距倾斜网格。"""

    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians), 0],
                         [np.sin(radians), np.cos(radians), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([1, 2, 3])
    base = np.array([[0.002, 0.0002, 0.0001],
                     [0.0002, 0.001, -0.00015], [0.0001, -0.00015, 0.0005]])
    scale = 1 + np.indices((5, 5, 5)).sum(axis=0) / 10.0
    tensors = scale[..., None, None] * base
    image = nib.Nifti1Image(matrix_to_tensor6(tensors), affine)
    nib.save(image, path)
    return path, tensors


@pytest.mark.parametrize("affine", [np.eye(4), np.diag([-1, 2, 3, 1])])
def test_no_consumer_rewrite_when_basis_already_agrees(tmp_path, affine):
    source = tmp_path / "tensor.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 6)), affine), source)
    output = tmp_path / "output"
    report = describe_simnibs_tensor_consumer(source)
    assert report["consumer_tensor"] == str(source)
    assert not output.exists()
    assert not report["tensor_file_rewritten"]


def test_literal_modes_are_independent_and_never_create_consumer_file(tmp_path):
    source, _ = _tensor_input(tmp_path / "tensor.nii.gz")
    output = tmp_path / "output"
    report = describe_simnibs_tensor_consumer(
        source, basis_mode="simnibs46-literal", eigensystem_mode="simnibs46-literal"
    )
    assert not output.exists()
    assert report["adapter"] == "upstream-unmodified"
    report = describe_simnibs_tensor_consumer(source, basis_mode="simnibs46-literal")
    assert report["eigensystem_mode"] == "stable"
    assert report["adapter"] == "instance-scoped-method"
    report = describe_simnibs_tensor_consumer(source, eigensystem_mode="simnibs46-literal")
    assert report["basis_mode"] == "physical"
    assert report["adapter"] == "instance-scoped-method"


def test_consumer_rejects_shear_and_unknown_eigensystem(tmp_path):
    source, _ = _tensor_input(tmp_path / "tensor.nii.gz")
    with pytest.raises(ValueError, match="eigensystem_mode"):
        describe_simnibs_tensor_consumer(source, eigensystem_mode="bad")
    affine = np.eye(4)
    affine[0, 1] = 0.2
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 6)), affine), source)
    with pytest.raises(ValueError, match="sheared grids"):
        describe_simnibs_tensor_consumer(source)
    assert not (tmp_path / "shear-output").exists()


def test_function_clone_does_not_change_original_globals():
    original = _clone_function
    original_globals = dict(original.__globals__)
    replacement = object()
    clone = _clone_function(original, {"FunctionType": replacement})
    assert original.__globals__ == original_globals
    assert clone.__globals__["FunctionType"] is replacement
    assert clone.__code__ is original.__code__


def _guard_cond(mesh, cond_list, anisotropy_volume=None, **kwargs):
    """最小第三方接口替身仅用于拒绝及标量转发测试。"""
    if anisotropy_volume is None:
        return np.asarray(cond_list)
    return _get_sorted_eigenv(anisotropy_volume)  # noqa: F821


def _guard_consumer_method(self, *args, **kwargs):
    return simnibs.utils.cond_utils.cond2elmdata(*args, **kwargs)  # noqa: F821


def _install_guard_simnibs(monkeypatch, *, version="4.6.0", cond=None, valid_globals=True):
    """隔离模拟模块，保证错误路径测试不修改真实 SimNIBS 的全局方法。"""
    simnibs_module = ModuleType("simnibs")
    simnibs_module.__version__ = version
    cond_module = ModuleType("simnibs.utils.cond_utils")
    cond_module.cond2elmdata = _guard_cond if cond is None else cond
    utils_module = ModuleType("simnibs.utils")
    utils_module.cond_utils = cond_module
    sim_struct_module = ModuleType("simnibs.sim_struct")
    namespace = {"simnibs": simnibs_module if valid_globals else object()}
    method = _clone_function(_guard_consumer_method, namespace)
    sim_struct_module.SimuList = type("SimuList", (), {"cond2elmdata": method})
    simnibs_module.utils = utils_module
    simnibs_module.sim_struct = sim_struct_module
    for name, module in (
        ("simnibs", simnibs_module), ("simnibs.utils", utils_module),
        ("simnibs.utils.cond_utils", cond_module),
        ("simnibs.sim_struct", sim_struct_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return simnibs_module, method


def test_consumer_rejects_incompatible_simnibs_version_before_binding(tmp_path, monkeypatch):
    source, _ = _tensor_input(tmp_path / "tensor.nii.gz")
    _install_guard_simnibs(monkeypatch, version="4.7.0")
    original = object()
    consumer = SimpleNamespace(cond2elmdata=original)
    with pytest.raises(RuntimeError, match="requires SimNIBS 4.6.0"):
        bind_simnibs_tensor_consumer(consumer, source)
    assert consumer.cond2elmdata is original


@pytest.mark.parametrize("mismatch", ["eigen_hook", "method_globals"])
def test_consumer_rejects_changed_conductivity_hook_atomically(tmp_path, monkeypatch, mismatch):
    source, _ = _tensor_input(tmp_path / "tensor.nii.gz")

    def changed_cond(mesh, cond_list, **kwargs):
        return cond_list

    _install_guard_simnibs(
        monkeypatch, cond=changed_cond if mismatch == "eigen_hook" else None,
        valid_globals=mismatch != "method_globals",
    )
    original = object()
    consumer = SimpleNamespace(cond2elmdata=original)
    with pytest.raises(RuntimeError, match="conductivity hook differs"):
        bind_simnibs_tensor_consumer(consumer, source)
    assert consumer.cond2elmdata is original


@pytest.mark.parametrize("mismatch", ["missing_static_call", "changed_static_call"])
def test_consumer_rejects_changed_leadfield_hook_atomically(tmp_path, monkeypatch, mismatch):
    source, _ = _tensor_input(tmp_path / "tensor.nii.gz")
    _install_guard_simnibs(monkeypatch)

    def missing(self, w_elec):
        return w_elec

    def changed(self, w_elec):
        return SimuList.cond2elmdata(self, "changed argument")  # noqa: F821

    consumer = type("Leadfield", (), {
        "run": missing if mismatch == "missing_static_call" else changed,
        "cond2elmdata": _guard_consumer_method,
    })()
    original_method, original_run = consumer.cond2elmdata.__func__, consumer.run.__func__
    with pytest.raises(RuntimeError, match="static conductivity call differs"):
        bind_simnibs_tensor_consumer(consumer, source, leadfield=True)
    assert consumer.cond2elmdata.__func__ is original_method
    assert consumer.run.__func__ is original_run


@pytest.mark.parametrize("eigensystem_mode", ["stable", "simnibs46-literal"])
def test_consumer_scalar_call_without_anisotropy_keeps_values(tmp_path, monkeypatch, eigensystem_mode):
    source, _ = _tensor_input(tmp_path / "tensor.nii.gz")
    fake_simnibs, original_method = _install_guard_simnibs(monkeypatch)
    consumer = fake_simnibs.sim_struct.SimuList()
    report = bind_simnibs_tensor_consumer(consumer, source, eigensystem_mode=eigensystem_mode)
    # 标量路径没有 affine 或 anisotropy_volume，不得触发物理张量变换。
    np.testing.assert_array_equal(consumer.cond2elmdata(object(), [0.126, 0.275]), [0.126, 0.275])
    assert consumer.cond2elmdata.__func__ is not original_method
    assert fake_simnibs.sim_struct.SimuList.cond2elmdata is original_method
    assert report["leadfield_static_call_adapted"] is False


def test_consumer_rejects_tensor_image_with_wrong_components(tmp_path):
    source = tmp_path / "wrong_tensor.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 5)), np.eye(4)), source)
    with pytest.raises(ValueError, match="4-D six-component"):
        describe_simnibs_tensor_consumer(source)


def test_literal_consumer_binding_leaves_instance_and_source_unchanged(tmp_path, monkeypatch):
    """两个 literal 选项同时启用时，消费实例保持上游原方法。"""
    source, _ = _tensor_input(tmp_path / "tensor.nii.gz")
    original_bytes = source.read_bytes()
    fake_simnibs, original_method = _install_guard_simnibs(monkeypatch)
    consumer = fake_simnibs.sim_struct.SimuList()
    report = bind_simnibs_tensor_consumer(
        consumer, source, basis_mode="simnibs46-literal",
        eigensystem_mode="simnibs46-literal",
    )
    assert report["adapter"] == "upstream-unmodified"
    assert consumer.cond2elmdata.__func__ is original_method
    assert "cond2elmdata" not in consumer.__dict__
    assert source.read_bytes() == original_bytes


@pytest.mark.skipif(
    os.environ.get("DWI2COND_TEST_SIMNIBS") != "1",
    reason="Set DWI2COND_TEST_SIMNIBS=1 in the SimNIBS environment for upstream A/B",
)
@pytest.mark.parametrize("mode", ["vn", "dir", "mc"])
@pytest.mark.parametrize("angle", [45.0, 90.0])
@pytest.mark.parametrize("tensor_kind", ["anisotropic", "isotropic"])
def test_real_simnibs_consumers_match_physical_cond2elmdata(tmp_path, mode, angle, tensor_kind):
    """真实网格、SESSION 准备及 leadfield 静态消费调用逐元素验证。"""

    from simnibs import sim_struct
    from simnibs.mesh_tools import mesh_io
    from simnibs.utils import cond_utils
    from dwi2cond_xp.leadfield import build_tdcs_leadfield
    from dwi2cond_xp.simulation import build_tdcs_session

    source, tensors = _tensor_input(tmp_path / "tensor.nii.gz", angle)
    image = nib.load(source)
    if tensor_kind == "isotropic":
        tensors = (0.001 * np.eye(3)) * np.ones((5, 5, 5, 1, 1))
        nib.save(nib.Nifti1Image(matrix_to_tensor6(tensors), image.affine), source)
        image = nib.load(source)
    original_bytes = source.read_bytes()
    # 用两套四面体在不同组织及不同插值位置测试完整九分量输出。
    points = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
                       [2, 2, 2], [3, 2, 2], [2, 3, 2], [2, 2, 3]], dtype=float)
    world = points @ image.affine[:3, :3].T + image.affine[:3, 3]
    elements = mesh_io.Elements(
        triangles=np.array([[1, 2, 3], [5, 6, 7]]),
        tetrahedra=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
    )
    elements.tag1[:] = [1001, 1002, 1, 2]
    elements.tag2[:] = elements.tag1
    mesh = mesh_io.Msh(mesh_io.Nodes(world), elements)
    subject = tmp_path / "m2m_test"
    (subject / "eeg_positions").mkdir(parents=True)
    mesh_file = subject / "test.msh"
    mesh_io.write_msh(mesh, str(mesh_file))
    cap = subject / "eeg_positions/EEG10-10_UI_Jurak_2007.csv"
    cap.write_text("Electrode,0,0,0,C3\nElectrode,1,0,0,C4\n")
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), image.affine), subject / "T1.nii.gz")
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), image.affine), subject / "final_tissues.nii.gz")
    contract = {"subpath": str(subject), "head_mesh": str(mesh_file),
                "eeg_cap": str(cap), "mode": mode, "tensor": {"path": str(source)}}
    upstream_method = sim_struct.SimuList.cond2elmdata
    upstream_run = sim_struct.TDCSLEADFIELD.run
    upstream_eigen = cond_utils._get_sorted_eigenv
    cond_globals = dict(cond_utils.cond2elmdata.__globals__)
    method_globals = dict(upstream_method.__globals__)
    run_globals = dict(upstream_run.__globals__)
    session = build_tdcs_session(contract, tmp_path / "simulation")
    session._prepare()
    tdcs = session.poslists[0]
    leadfield = build_tdcs_leadfield(contract, tmp_path / "leadfield")
    leadfield._prepare()
    assert tdcs.fn_tensor_nifti == session.fname_tensor == str(source)
    assert leadfield.fn_tensor_nifti == leadfield.fname_tensor == str(source)
    basis = fsl_tensor_basis(image.affine)
    physical = matrix_to_tensor6(np.einsum("ij,...jk,lk->...il", basis, tensors, basis))
    expected = cond_utils.cond2elmdata(
        mesh, [entry.value for entry in tdcs.cond],
        anisotropy_volume=physical, affine=image.affine, correct_FSL=False,
        normalize=mode == "vn", excentricity_scaling=0.0 if mode == "mc" else None,
    )
    actual_tdcs = tdcs.cond2elmdata(mesh)
    static_consumer = leadfield.run.__func__.__globals__["SimuList"]
    assert static_consumer is not sim_struct.SimuList
    assert leadfield.run.__func__.__code__ is upstream_run.__code__
    actual_leadfield = static_consumer.cond2elmdata(leadfield, mesh)
    np.testing.assert_allclose(actual_tdcs.value, expected.value, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual_leadfield.value, expected.value, rtol=1e-12, atol=1e-14)
    if tensor_kind == "isotropic":
        eigenvalues = np.linalg.eigvalsh(actual_tdcs.value.reshape(-1, 3, 3))
        np.testing.assert_allclose(eigenvalues[:, 2], eigenvalues[:, 0], rtol=1e-12, atol=1e-14)
    assert cond_utils._get_sorted_eigenv is upstream_eigen
    assert sim_struct.SimuList.cond2elmdata is upstream_method
    assert sim_struct.TDCSLEADFIELD.run is upstream_run
    assert cond_utils.cond2elmdata.__globals__ == cond_globals
    assert upstream_method.__globals__ == method_globals
    assert upstream_run.__globals__ == run_globals
    untouched_session = sim_struct.SESSION()
    assert untouched_session.add_tdcslist().cond2elmdata.__func__ is upstream_method
    assert sim_struct.TDCSLEADFIELD().run.__func__ is upstream_run
    assert source.read_bytes() == original_bytes
    if tensor_kind == "anisotropic" and angle == 45.0:
        literal_session = build_tdcs_session(
            contract, tmp_path / "literal-simulation",
            basis_mode="simnibs46-literal", eigensystem_mode="simnibs46-literal",
        )
        literal_session._prepare()
        literal_tdcs = literal_session.poslists[0]
        literal_leadfield = build_tdcs_leadfield(
            contract, tmp_path / "literal-leadfield",
            basis_mode="simnibs46-literal", eigensystem_mode="simnibs46-literal",
        )
        literal_leadfield._prepare()
        literal_expected = cond_utils.cond2elmdata(
            mesh, [entry.value for entry in tdcs.cond],
            anisotropy_volume=image.dataobj, affine=image.affine,
            normalize=mode == "vn", excentricity_scaling=0.0 if mode == "mc" else None,
        )
        assert literal_tdcs.cond2elmdata.__func__ is upstream_method
        assert literal_leadfield.run.__func__ is upstream_run
        np.testing.assert_array_equal(literal_tdcs.cond2elmdata(mesh).value, literal_expected.value)
        np.testing.assert_array_equal(upstream_method(literal_leadfield, mesh).value, literal_expected.value)
    (tmp_path / "basis_comparison.json").write_text(json.dumps({
        "mode": mode, "angle": angle, "tensor_kind": tensor_kind,
        "session_max_abs_error": float(np.max(np.abs(actual_tdcs.value - expected.value))),
        "leadfield_max_abs_error": float(np.max(np.abs(actual_leadfield.value - expected.value))),
        "session_tensor": session.fname_tensor,
        "leadfield_tensor": leadfield.fname_tensor,
        "num_elements": len(expected.value),
        "consumer_metadata": session._dwi2cond_xp_tensor_basis,
    }, indent=2) + "\n")
