import json
import sys
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp import simulation
from dwi2cond_xp.simulation import (
    _discover_subject_files,
    _json_safe,
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
    assert (
        tmp_path / "simulation/dry-run/scalar/dwi2cond_xp_simulation.json"
    ).is_file()


def test_scalar_mode_rejects_unused_tensor(tmp_path: Path) -> None:
    subpath, tensor = _make_subject(tmp_path)
    with pytest.raises(ValueError, match="must not be provided"):
        validate_simulation_inputs(subpath, mode="scalar", tensor_file=tensor)


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


def test_simulation_rejects_nonfinite_tensor_and_tissue_grid_mismatch(tmp_path: Path) -> None:
    subpath, tensor_file = _make_subject(tmp_path)
    tensor_image = nib.load(tensor_file)
    tensor_values = np.asarray(tensor_image.dataobj).copy()
    tensor_values[0, 0, 0, 2] = np.inf
    nib.save(
        nib.Nifti1Image(tensor_values, tensor_image.affine, tensor_image.header),
        tensor_file,
    )
    with pytest.raises(ValueError, match="Tensor data contains NaN or Inf"):
        validate_simulation_inputs(subpath, mode="vn", tensor_file=tensor_file)

    tissues = subpath / "final_tissues.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.zeros((3, 4, 5, 1), dtype=np.uint8), tensor_image.affine
        ),
        tissues,
    )
    assert validate_simulation_inputs(
        subpath, mode="scalar", tensor_file=None
    )["grid"]["shape"] == [
        3,
        4,
        5,
    ]
    nib.save(
        nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), np.diag([2, 1, 1, 1])),
        tissues,
    )
    with pytest.raises(ValueError, match="share one three-dimensional grid"):
        validate_simulation_inputs(subpath, mode="scalar", tensor_file=None)


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


def _install_simnibs(monkeypatch, *, version="4.6.0", runner=None) -> ModuleType:
    simnibs = ModuleType("simnibs")
    simnibs.__version__ = version
    sim_struct = ModuleType("simnibs.sim_struct")
    sim_struct.SESSION = _FakeSession
    simnibs.sim_struct = sim_struct
    if runner is not None:
        simnibs.run_simnibs = runner
    monkeypatch.setitem(sys.modules, "simnibs", simnibs)
    monkeypatch.setitem(sys.modules, "simnibs.sim_struct", sim_struct)
    return simnibs


def test_discovery_prefers_named_files_and_falls_back(tmp_path: Path) -> None:
    named = tmp_path / "m2m_named"
    (named / "eeg_positions").mkdir(parents=True)
    (named / "named.msh").write_text("")
    official_cap = named / "eeg_positions/EEG10-10_UI_Jurak_2007.csv"
    official_cap.write_text("")
    mesh, _, _, cap = _discover_subject_files(named)
    assert mesh == named / "named.msh"
    assert cap == official_cap

    fallback = tmp_path / "subject"
    (fallback / "eeg_positions").mkdir(parents=True)
    (fallback / "only.msh").write_text("")
    alternate_cap = fallback / "eeg_positions/a_10-10_cap.csv"
    alternate_cap.write_text("")
    mesh, _, _, cap = _discover_subject_files(fallback)
    assert mesh == fallback / "only.msh"
    assert cap == alternate_cap


def test_validate_inputs_rejects_invalid_and_incomplete_subjects(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode must"):
        validate_simulation_inputs(tmp_path, mode="bad", tensor_file=None)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_simulation_inputs(tmp_path / "missing", mode="scalar", tensor_file=None)
    incomplete = tmp_path / "m2m_incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="CHARM head model is incomplete"):
        validate_simulation_inputs(incomplete, mode="scalar", tensor_file=None)


def test_validate_inputs_rejects_missing_tensor_and_affine(tmp_path: Path) -> None:
    subpath, _ = _make_subject(tmp_path)
    with pytest.raises(FileNotFoundError, match="Tensor file does not exist"):
        validate_simulation_inputs(subpath, mode="vn", tensor_file=tmp_path / "absent.nii.gz")
    wrong_affine = tmp_path / "affine.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((3, 4, 5, 6), dtype=np.float32), np.diag([2, 1, 1, 1])),
        wrong_affine,
    )
    with pytest.raises(ValueError, match="affine"):
        validate_simulation_inputs(subpath, mode="vn", tensor_file=wrong_affine)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"current_ma": 0}, "positive"),
        ({"thickness": 0}, "positive"),
        ({"dimensions": (10, 0)}, "positive"),
        ({"shape": "circle"}, "shape"),
        ({"solver": "default"}, "solver"),
        ({"volume_tissues": ()}, "volume_tissues"),
        ({"volume_tissues": (1, 0)}, "volume_tissues"),
    ],
)
def test_build_session_rejects_invalid_parameters(monkeypatch, tmp_path, kwargs, message):
    _install_simnibs(monkeypatch)
    contract = {"subpath": "x", "eeg_cap": "x", "mode": "scalar", "tensor": None}
    with pytest.raises(ValueError, match=message):
        build_tdcs_session(contract, tmp_path / "out", **kwargs)


def test_build_session_rejects_missing_simnibs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "simnibs", None)
    with pytest.raises(RuntimeError, match="simnibs==4.6.0"):
        build_tdcs_session(
            {"subpath": "x", "eeg_cap": "x", "mode": "scalar", "tensor": None},
            tmp_path / "out",
        )


def test_json_safe_converts_nested_scientific_values(tmp_path: Path) -> None:
    value = {
        "items": (np.int64(3), tmp_path, object()),
        4: np.float32(2.5),
        "none": None,
    }
    result = _json_safe(value)
    assert result["items"][0] == 3
    assert result["items"][1] == str(tmp_path)
    assert isinstance(result["items"][2], str)
    assert result["4"] == pytest.approx(2.5)


def test_mask_outputs_handles_absent_singleton_and_invalid_grids(tmp_path: Path) -> None:
    tissue_file = tmp_path / "tissues.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), tissue_file)
    assert mask_subject_volume_outputs(tmp_path / "absent", tissue_file, (1,)) == []

    output = tmp_path / "run/subject_volumes"
    output.mkdir(parents=True)
    singleton_tissues = tmp_path / "singleton.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2, 1), dtype=np.uint8), np.eye(4)), singleton_tissues
    )
    field = output / "field.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), field)
    assert mask_subject_volume_outputs(tmp_path / "run", singleton_tissues, (1,)) == [str(field.resolve())]

    invalid_tissues = tmp_path / "invalid.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2), dtype=np.uint8), np.eye(4)), invalid_tissues)
    with pytest.raises(ValueError, match="three-dimensional"):
        mask_subject_volume_outputs(tmp_path / "run", invalid_tissues, (1,))

    wrong_field = output / "wrong.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 2, 2), dtype=np.float32), np.eye(4)), wrong_field)
    with pytest.raises(ValueError, match="does not match"):
        mask_subject_volume_outputs(tmp_path / "run", tissue_file, (1,))


def test_run_tdcs_rejects_cpu_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        run_tdcs(tmp_path, tmp_path / "out", mode="scalar", cpus=0)


def test_dry_run_validates_static_montage_before_writing(tmp_path: Path) -> None:
    subpath, _ = _make_subject(tmp_path)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="must be positive"):
        run_tdcs(
            subpath,
            output,
            mode="scalar",
            current_ma=0.0,
            dry_run=True,
        )
    assert not output.exists()
    with pytest.raises(ValueError, match="distinct nonempty"):
        run_tdcs(
            subpath,
            output,
            mode="scalar",
            anode="C3",
            cathode="C3",
            dry_run=True,
        )
    with pytest.raises(ValueError, match="at least one SimNIBS output field"):
        run_tdcs(
            subpath,
            output,
            mode="scalar",
            fields="",
            dry_run=True,
        )


def test_run_tdcs_completes_and_masks_outputs(monkeypatch, tmp_path: Path) -> None:
    subpath, tensor = _make_subject(tmp_path)
    tissues = np.array([1, 2, 3, 5] * 15, dtype=np.uint8).reshape(3, 4, 5)
    nib.save(nib.Nifti1Image(tissues, np.eye(4)), subpath / "final_tissues.nii.gz")

    def runner(session, cpus):
        assert cpus == 2
        root = Path(session.pathfem)
        mesh = root / "mesh.msh"
        mesh.write_text("mesh\n", encoding="utf-8")
        output = root / "subject_volumes"
        output.mkdir(parents=True)
        data = np.ones((3, 4, 5, 3), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), output / "field_E.nii.gz")
        return [mesh]

    _install_simnibs(monkeypatch, runner=runner)
    stale = tmp_path / "out/vn/stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    result = run_tdcs(
        subpath, tmp_path / "out", mode="vn", tensor_file=tensor, cpus=2
    )
    assert result["status"] == "completed"
    assert result["outputs"] == [str(tmp_path / "out/vn/mesh.msh")]
    assert result["artifacts"]
    assert not stale.exists()
    assert all("sha256" in artifact for artifact in result["artifacts"])
    masked = np.asanyarray(nib.load(result["masked_subject_volumes"][0]).dataobj)
    assert np.all(masked[tissues == 5] == 0)


def test_run_tdcs_records_solver_failure(monkeypatch, tmp_path: Path) -> None:
    subpath, _ = _make_subject(tmp_path)

    def runner(session, cpus):
        raise RuntimeError("FEM failed")

    _install_simnibs(monkeypatch, runner=runner)
    with pytest.raises(RuntimeError, match="FEM failed"):
        run_tdcs(subpath, tmp_path / "out", mode="scalar")
    manifest = json.loads(
        (tmp_path / "out/scalar/dwi2cond_xp_simulation.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    assert manifest["failed_phase"] == "solve"


def test_run_tdcs_records_build_and_postprocess_failures(monkeypatch, tmp_path: Path) -> None:
    subpath, _ = _make_subject(tmp_path)
    _install_simnibs(monkeypatch, runner=lambda _session, _cpus: [])
    monkeypatch.setattr(
        simulation,
        "build_tdcs_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad montage")),
    )
    build_root = tmp_path / "build_failure"
    with pytest.raises(ValueError, match="bad montage"):
        run_tdcs(subpath, build_root, mode="scalar")
    build_manifest = json.loads(
        (build_root / "scalar/dwi2cond_xp_simulation.json").read_text()
    )
    assert build_manifest["status"] == "failed"
    assert build_manifest["failed_phase"] == "build_session"

    monkeypatch.undo()
    subpath, _ = _make_subject(tmp_path / "post")

    def runner(session, cpus):
        output = Path(session.pathfem) / "subject_volumes"
        output.mkdir(parents=True)
        nib.save(
            nib.Nifti1Image(np.ones((2, 2, 2, 3), dtype=np.float32), np.eye(4)),
            output / "bad_grid_E.nii.gz",
        )
        return []

    _install_simnibs(monkeypatch, runner=runner)
    post_root = tmp_path / "post_failure"
    with pytest.raises(ValueError, match="does not match"):
        run_tdcs(subpath, post_root, mode="scalar")
    post_manifest = json.loads(
        (post_root / "scalar/dwi2cond_xp_simulation.json").read_text()
    )
    assert post_manifest["status"] == "failed"
    assert post_manifest["failed_phase"] == "postprocess_subject_volumes"

    monkeypatch.undo()
    subpath, _ = _make_subject(tmp_path / "empty")

    def empty_volume_runner(session, cpus):
        assert cpus == 8
        mesh = Path(session.pathfem) / "result.msh"
        mesh.write_text("mesh\n", encoding="utf-8")
        return [str(mesh)]

    _install_simnibs(monkeypatch, runner=empty_volume_runner)
    with pytest.raises(ValueError, match="no subject-volume output"):
        run_tdcs(subpath, tmp_path / "empty-output", mode="scalar")


def test_run_tdcs_records_completed_manifest_serialization_failure(
    monkeypatch, tmp_path: Path
) -> None:
    subpath, _ = _make_subject(tmp_path)

    def runner(session, cpus):
        assert cpus == 8
        root = Path(session.pathfem)
        mesh = root / "result.msh"
        mesh.write_text("mesh\n", encoding="utf-8")
        subject_volumes = root / "subject_volumes"
        subject_volumes.mkdir()
        nib.save(
            nib.Nifti1Image(
                np.ones((3, 4, 5, 3), dtype=np.float32), np.eye(4)
            ),
            subject_volumes / "field_E.nii.gz",
        )
        return [str(mesh)]

    _install_simnibs(monkeypatch, runner=runner)
    original_write = simulation._write_manifest
    failed_once = False

    def fail_completed(path, payload):
        nonlocal failed_once
        if payload.get("status") == "completed" and not failed_once:
            failed_once = True
            raise OSError("injected completed manifest failure")
        original_write(path, payload)

    monkeypatch.setattr(simulation, "_write_manifest", fail_completed)
    with pytest.raises(OSError, match="completed manifest"):
        run_tdcs(subpath, tmp_path / "out", mode="scalar")
    manifest = json.loads(
        (tmp_path / "out/scalar/dwi2cond_xp_simulation.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["failed_phase"] == "write_completed_manifest"
