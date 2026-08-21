import sys
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.simulation import (
    build_tdcs_session,
    mask_subject_volume_outputs,
    run_tdcs,
    validate_simulation_inputs,
)


class _FakeTdcs:
    def __init__(self) -> None:
        self.electrodes = []

    def add_electrode(self):
        electrode = type("Electrode", (), {})()
        self.electrodes.append(electrode)
        return electrode


class _FakeSession:
    def __init__(self) -> None:
        self.tdcs = _FakeTdcs()

    def add_tdcslist(self):
        return self.tdcs


def _make_subject(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal CHARM directory containing only required inputs."""
    subpath = tmp_path / "m2m_test"
    (subpath / "eeg_positions").mkdir(parents=True)
    (subpath / "test.msh").write_text("", encoding="utf-8")
    (subpath / "eeg_positions" / "test_10-10.csv").write_text("", encoding="utf-8")
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    t1 = nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), affine)
    nib.save(t1, subpath / "T1.nii.gz")
    nib.save(t1, subpath / "final_tissues.nii.gz")
    tensor_file = tmp_path / "tensor.nii.gz"
    tensor = nib.Nifti1Image(np.zeros((3, 4, 5, 6), dtype=np.float32), affine)
    nib.save(tensor, tensor_file)
    return subpath, tensor_file


def test_scalar_dry_run_does_not_require_tensor(tmp_path: Path) -> None:
    subpath, _ = _make_subject(tmp_path)
    manifest = run_tdcs(
        subpath, tmp_path / "simulation", mode="scalar", dry_run=True
    )
    assert manifest["status"] == "planned"
    assert manifest["input"]["tensor"] is None
    assert manifest["map_to_subject_volume"] is True
    assert (tmp_path / "simulation/scalar/dwi2cond_xp_simulation.json").is_file()


def test_anisotropic_mode_requires_tensor(tmp_path: Path) -> None:
    subpath, _ = _make_subject(tmp_path)
    with pytest.raises(ValueError, match="fallback is forbidden"):
        validate_simulation_inputs(subpath, mode="vn", tensor_file=None)


def test_anisotropic_mode_uses_official_subject_tensor(tmp_path: Path) -> None:
    subpath, tensor_file = _make_subject(tmp_path)
    official_tensor = subpath / "DTI_coregT1_tensor.nii.gz"
    tensor_file.replace(official_tensor)
    contract = validate_simulation_inputs(subpath, mode="vn", tensor_file=None)
    assert contract["tensor"]["path"] == str(official_tensor.resolve())


def test_tensor_grid_must_match_t1(tmp_path: Path) -> None:
    subpath, _ = _make_subject(tmp_path)
    wrong_tensor = tmp_path / "wrong_tensor.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.zeros((3, 4, 4, 6), dtype=np.float32), np.eye(4)
        ),
        wrong_tensor,
    )
    with pytest.raises(ValueError, match="Tensor shape"):
        validate_simulation_inputs(subpath, mode="vn", tensor_file=wrong_tensor)


def test_anisotropic_dry_run_records_contract(tmp_path: Path) -> None:
    subpath, tensor_file = _make_subject(tmp_path)
    manifest = run_tdcs(
        subpath,
        tmp_path / "simulation",
        mode="vn",
        tensor_file=tensor_file,
        dry_run=True,
    )
    assert manifest["status"] == "planned"
    assert manifest["solver"] == "pardiso"
    assert manifest["volume_tissues"] == [1, 2, 3]
    assert manifest["input"]["tensor"]["shape"] == [3, 4, 5, 6]
    assert manifest["input"]["tensor"]["component_order"] == [
        "Dxx",
        "Dxy",
        "Dxz",
        "Dyy",
        "Dyz",
        "Dzz",
    ]


def test_subject_volume_mask_excludes_skull_and_scalp(tmp_path: Path) -> None:
    """Formal volume output retains only WM, GM, and CSF."""
    output = tmp_path / "simulation" / "subject_volumes"
    output.mkdir(parents=True)
    affine = np.eye(4)
    tissues = np.array([1, 2, 3, 7, 5], dtype=np.uint16).reshape(5, 1, 1)
    tissue_file = tmp_path / "final_tissues.nii.gz"
    field_file = output / "test_E.nii.gz"
    nib.save(nib.Nifti1Image(tissues, affine), tissue_file)
    nib.save(
        nib.Nifti1Image(np.ones((5, 1, 1, 3), dtype=np.float32), affine),
        field_file,
    )

    masked = mask_subject_volume_outputs(
        tmp_path / "simulation", tissue_file, (1, 2, 3)
    )

    field = np.asanyarray(nib.load(field_file).dataobj)
    assert masked == [str(field_file.resolve())]
    assert np.all(field[:3] == 1)
    assert np.all(field[3:] == 0)


def test_build_tdcs_session_maps_tensor_and_montage(monkeypatch, tmp_path: Path) -> None:
    simnibs = ModuleType("simnibs")
    simnibs.__version__ = "4.6.0"
    sim_struct = ModuleType("simnibs.sim_struct")
    sim_struct.SESSION = _FakeSession
    simnibs.sim_struct = sim_struct
    monkeypatch.setitem(sys.modules, "simnibs", simnibs)
    monkeypatch.setitem(sys.modules, "simnibs.sim_struct", sim_struct)
    contract = {
        "subpath": "/tmp/m2m_test",
        "eeg_cap": "/tmp/cap.csv",
        "mode": "vn",
        "tensor": {"path": "/tmp/tensor.nii.gz"},
    }

    session = build_tdcs_session(contract, tmp_path / "output")

    assert session.map_to_vol is True
    assert session.tissues_in_niftis == [1, 2, 3]
    assert session.fname_tensor == "/tmp/tensor.nii.gz"
    assert session.tdcs.currents == [0.001, -0.001]
    assert session.tdcs.anisotropy_type == "vn"
    assert [electrode.centre for electrode in session.tdcs.electrodes] == ["C3", "C4"]
    assert all(electrode.dimensions == [50.0, 50.0] for electrode in session.tdcs.electrodes)


def test_build_tdcs_session_rejects_wrong_simnibs_version(monkeypatch, tmp_path: Path) -> None:
    simnibs = ModuleType("simnibs")
    simnibs.__version__ = "4.5.0"
    sim_struct = ModuleType("simnibs.sim_struct")
    sim_struct.SESSION = _FakeSession
    simnibs.sim_struct = sim_struct
    monkeypatch.setitem(sys.modules, "simnibs", simnibs)
    monkeypatch.setitem(sys.modules, "simnibs.sim_struct", sim_struct)
    with pytest.raises(RuntimeError, match="4.6.0"):
        build_tdcs_session(
            {"subpath": "x", "eeg_cap": "x", "mode": "scalar", "tensor": None},
            tmp_path / "output",
        )
