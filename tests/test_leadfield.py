import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.leadfield import (
    LEADFIELD_DATASET,
    _decode_text,
    build_tdcs_leadfield,
    run_tdcs_leadfield,
    validate_and_export_leadfield,
)


def _make_subject(tmp_path: Path) -> tuple[Path, Path]:
    """Create the minimal CHARM input contract required for a lead-field dry run."""
    subpath = tmp_path / "m2m_test"
    (subpath / "eeg_positions").mkdir(parents=True)
    (subpath / "test.msh").write_text("", encoding="utf-8")
    (subpath / "eeg_positions" / "test_10-10.csv").write_text("", encoding="utf-8")
    affine = np.eye(4)
    t1 = nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), affine)
    nib.save(t1, subpath / "T1.nii.gz")
    nib.save(t1, subpath / "final_tissues.nii.gz")
    tensor_file = tmp_path / "tensor.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((3, 4, 5, 6), dtype=np.float32), affine),
        tensor_file,
    )
    return subpath, tensor_file


def _make_leadfield_hdf5(path: Path, *, reference_matches: bool = True) -> np.ndarray:
    """Create a minimal HDF5 matching the SimNIBS three-axis contract."""
    values = np.arange(24, dtype=np.float64).reshape(2, 4, 3) / 10.0
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(LEADFIELD_DATASET, data=values)
        dataset.attrs["electrode_names"] = np.asarray(
            ["REF", "E1", "E2"], dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["reference_electrode"] = "REF" if reference_matches else "BAD"
        dataset.attrs["field"] = "E"
        dataset.attrs["units"] = "V/m"
        dataset.attrs["current"] = "1A"
        dataset.attrs["d_type"] = "element_data"
        dataset.attrs["interpolation"] = "None"
        handle.create_dataset("mesh_leadfield/elm/tag1", data=[1, 1, 2, 3])
        handle.create_dataset(
            "mesh_leadfield/nodes/node_coord", data=np.zeros((6, 3))
        )
    return values


def test_leadfield_dry_run_defaults_to_pardiso_and_npy(tmp_path: Path) -> None:
    subpath, tensor_file = _make_subject(tmp_path)
    manifest = run_tdcs_leadfield(
        subpath,
        tmp_path / "leadfields",
        mode="vn",
        tensor_file=tensor_file,
        dry_run=True,
    )
    assert manifest["status"] == "planned"
    assert manifest["leadfield"]["solver"] == "pardiso"
    assert manifest["leadfield"]["basis_current"] == "1A"


def test_validate_and_export_leadfield_preserves_axis_contract(tmp_path: Path) -> None:
    hdf5_file = tmp_path / "leadfield.hdf5"
    values = _make_leadfield_hdf5(hdf5_file)
    matrix_file = tmp_path / "leadfield.npy"
    report = validate_and_export_leadfield(
        hdf5_file,
        matrix_file=matrix_file,
        roi_labels=(1, 2),
        avoid_labels=(3,),
        roi_mask_file=tmp_path / "roi.npy",
        avoid_mask_file=tmp_path / "avoid.npy",
    )
    matrix = np.load(matrix_file, mmap_mode="r")
    assert matrix.shape == (12, 2)
    assert np.array_equal(matrix[:, 0], values[0].reshape(-1))
    assert np.array_equal(matrix[:, 1], values[1].reshape(-1))
    assert report["reference_electrode"] == "REF"
    assert report["active_electrodes"] == ["E1", "E2"]
    assert report["basis_current"] == "1A"
    assert np.array_equal(np.load(tmp_path / "roi.npy"), [True, True, True, False])
    assert np.array_equal(np.load(tmp_path / "avoid.npy"), [False, False, False, True])


def test_reference_must_be_first_electrode(tmp_path: Path) -> None:
    hdf5_file = tmp_path / "bad_reference.hdf5"
    _make_leadfield_hdf5(hdf5_file, reference_matches=False)
    with pytest.raises(ValueError, match="first"):
        validate_and_export_leadfield(hdf5_file)


def test_decode_text_handles_numpy_void_bytes() -> None:
    assert _decode_text(np.void(b"REF")) == "REF"


class _FakeLeadfield:
    def __init__(self) -> None:
        self.electrode = SimpleNamespace()


def _install_fake_simnibs(monkeypatch, *, version: str = "4.6.0", runner=None):
    simnibs = ModuleType("simnibs")
    simnibs.__version__ = version
    sim_struct = ModuleType("simnibs.sim_struct")
    sim_struct.TDCSLEADFIELD = _FakeLeadfield
    simnibs.sim_struct = sim_struct
    if runner is not None:
        simnibs.run_simnibs = runner
    monkeypatch.setitem(sys.modules, "simnibs", simnibs)
    monkeypatch.setitem(sys.modules, "simnibs.sim_struct", sim_struct)
    return simnibs


def _contract(*, tensor: bool = True) -> dict:
    return {
        "head_mesh": "/tmp/head.msh",
        "subpath": "/tmp/m2m_test",
        "eeg_cap": "/tmp/cap.csv",
        "mode": "vn" if tensor else "scalar",
        "tensor": {"path": "/tmp/tensor.nii.gz"} if tensor else None,
    }


def test_build_leadfield_maps_volume_and_surface_contract(monkeypatch, tmp_path: Path) -> None:
    _install_fake_simnibs(monkeypatch)
    volume = build_tdcs_leadfield(
        _contract(),
        tmp_path,
        field="J",
        shape="rect",
        dimensions=(11.0, 12.0),
        thickness=3.0,
        tissues=(1, 2, 3),
        solver="default",
    )
    assert volume.field == "J"
    assert volume.fname_tensor == "/tmp/tensor.nii.gz"
    assert volume.interpolation is None
    assert volume.tissues == [1, 2, 3]
    assert volume.electrode.dimensions == [11.0, 12.0]
    assert volume.solver_options is None

    surface = build_tdcs_leadfield(
        _contract(tensor=False), tmp_path, interpolation="middle-gm", solver="pardiso"
    )
    assert surface.interpolation == "middle gm"
    assert surface.tissues == []
    assert surface.interpolation_tissue == [2]
    assert surface.solver_options == "pardiso"
    assert not hasattr(surface, "fname_tensor")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"field": "D"}, "field"),
        ({"interpolation": "volume"}, "interpolation"),
        ({"shape": "circle"}, "shape"),
        ({"dimensions": (0.0, 1.0)}, "positive"),
        ({"thickness": 0.0}, "positive"),
        ({"solver": "mumps"}, "solver"),
        ({"tissues": ()}, "at least one"),
        ({"interpolation": "middle-gm", "interpolation_tissues": ()}, "requires"),
    ],
)
def test_build_leadfield_rejects_invalid_parameters(monkeypatch, tmp_path, kwargs, message):
    _install_fake_simnibs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        build_tdcs_leadfield(_contract(), tmp_path, **kwargs)


def test_build_leadfield_rejects_missing_or_wrong_simnibs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "simnibs", None)
    with pytest.raises(RuntimeError, match="simnibs==4.6.0"):
        build_tdcs_leadfield(_contract(), tmp_path)
    _install_fake_simnibs(monkeypatch, version="4.5.0")
    with pytest.raises(RuntimeError, match="found 4.5.0"):
        build_tdcs_leadfield(_contract(), tmp_path)


def test_axis_contract_accepts_node_data_and_byte_scalars(tmp_path: Path) -> None:
    path = tmp_path / "nodes.hdf5"
    values = np.zeros((1, 2, 3), dtype=np.float64)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(LEADFIELD_DATASET, data=values)
        dataset.attrs["electrode_names"] = np.asarray([b"REF", b"E1"])
        dataset.attrs["reference_electrode"] = np.bytes_(b"REF")
        dataset.attrs["d_type"] = "node_data"
        handle.create_dataset("mesh_leadfield/nodes/node_coord", data=np.zeros((2, 3)))
    report = validate_and_export_leadfield(path)
    assert report["d_type"] == "node_data"
    assert report["field"] == "unknown"
    assert report["root_mean_square"] == 0.0


def test_axis_contract_rejects_malformed_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(KeyError, match="missing dataset"):
        validate_and_export_leadfield(path)

    cases = [
        (np.zeros((2, 3)), ["REF", "E1", "E2"], "element_data", 3, "shape"),
        (np.zeros((2, 4, 3)), ["REF", "E1"], "element_data", 4, "count"),
        (np.zeros((2, 4, 3)), ["REF", "E1", "E2"], "voxel_data", 4, "Unsupported"),
        (np.zeros((2, 4, 3)), ["REF", "E1", "E2"], "element_data", 3, "spatial"),
    ]
    for index, (values, names, d_type, count, message) in enumerate(cases):
        candidate = tmp_path / f"bad_{index}.hdf5"
        with h5py.File(candidate, "w") as handle:
            dataset = handle.create_dataset(LEADFIELD_DATASET, data=values)
            dataset.attrs["electrode_names"] = names
            dataset.attrs["reference_electrode"] = "REF"
            dataset.attrs["d_type"] = d_type
            handle.create_dataset("mesh_leadfield/elm/tag1", data=np.ones(count))
        with pytest.raises(ValueError, match=message):
            validate_and_export_leadfield(candidate)


def test_export_rejects_invalid_masks_and_cleans_partial_matrix(tmp_path: Path) -> None:
    node_file = tmp_path / "nodes.hdf5"
    with h5py.File(node_file, "w") as handle:
        dataset = handle.create_dataset(LEADFIELD_DATASET, data=np.zeros((1, 2, 3)))
        dataset.attrs["electrode_names"] = ["REF", "E1"]
        dataset.attrs["reference_electrode"] = "REF"
        dataset.attrs["d_type"] = "node_data"
        handle.create_dataset("mesh_leadfield/nodes/node_coord", data=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="element_data only"):
        validate_and_export_leadfield(node_file, roi_labels=(1,), roi_mask_file=tmp_path / "r.npy")

    path = tmp_path / "leadfield.hdf5"
    _make_leadfield_hdf5(path)
    with pytest.raises(ValueError, match="missing tissue tags"):
        validate_and_export_leadfield(path, roi_labels=(9,), roi_mask_file=tmp_path / "r.npy")
    with pytest.raises(ValueError, match="roi_mask_file"):
        validate_and_export_leadfield(path, roi_labels=(1,))

    with h5py.File(path, "r+") as handle:
        handle[LEADFIELD_DATASET][1, 0, 0] = np.nan
    matrix = tmp_path / "partial.npy"
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_and_export_leadfield(path, matrix_file=matrix)
    assert not matrix.exists()
    assert not (tmp_path / "partial.npy.tmp").exists()


def test_export_progress_and_without_matrix(tmp_path: Path) -> None:
    path = tmp_path / "leadfield.hdf5"
    _make_leadfield_hdf5(path)
    calls = []
    report = validate_and_export_leadfield(
        path, progress=lambda current, total, stage: calls.append((current, total, stage))
    )
    assert calls == [(1, 2, "validate_export"), (2, 2, "validate_export")]
    assert "matrix_file" not in report


def test_run_leadfield_rejects_preflight_errors(tmp_path: Path) -> None:
    subpath, tensor = _make_subject(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        run_tdcs_leadfield(subpath, tmp_path / "out", mode="vn", tensor_file=tensor, cpus=0)
    with pytest.raises(ValueError, match="middle-gm"):
        run_tdcs_leadfield(
            subpath,
            tmp_path / "out",
            mode="vn",
            tensor_file=tensor,
            interpolation="middle-gm",
            roi_labels=(1,),
        )
    with pytest.raises(ValueError, match="included"):
        run_tdcs_leadfield(
            subpath, tmp_path / "out", mode="vn", tensor_file=tensor, tissues=(1,), roi_labels=(2,)
        )
    with pytest.raises(FileNotFoundError, match="EEG cap"):
        run_tdcs_leadfield(
            subpath, tmp_path / "out", mode="vn", tensor_file=tensor, eeg_cap=tmp_path / "missing.csv"
        )


def test_run_leadfield_refuses_existing_hdf5(tmp_path: Path) -> None:
    subpath, tensor = _make_subject(tmp_path)
    output = tmp_path / "out" / "vn"
    output.mkdir(parents=True)
    (output / "old.hdf5").write_bytes(b"")
    with pytest.raises(FileExistsError, match="already contains"):
        run_tdcs_leadfield(subpath, tmp_path / "out", mode="vn", tensor_file=tensor)
    assert not (output / "dwi2cond_xp_leadfield.json").exists()


def test_run_leadfield_completes_and_records_failure(monkeypatch, tmp_path: Path) -> None:
    subpath, tensor = _make_subject(tmp_path)

    def successful_runner(leadfield, cpus):
        assert cpus == 3
        output = Path(leadfield.pathfem)
        output.mkdir(parents=True, exist_ok=True)
        _make_leadfield_hdf5(output / "result.hdf5")

    _install_fake_simnibs(monkeypatch, runner=successful_runner)
    result = run_tdcs_leadfield(
        subpath,
        tmp_path / "success",
        mode="vn",
        tensor_file=tensor,
        cpus=3,
        roi_labels=(1,),
        export_matrix=False,
    )
    assert result["status"] == "completed"
    assert result["qa"]["roi_mask_labels"] == [1]
    assert "matrix_file" not in result["qa"]

    def failing_runner(leadfield, cpus):
        raise RuntimeError("solver failed")

    _install_fake_simnibs(monkeypatch, runner=failing_runner)
    with pytest.raises(RuntimeError, match="solver failed"):
        run_tdcs_leadfield(
            subpath, tmp_path / "failure", mode="vn", tensor_file=tensor
        )
    manifest_path = tmp_path / "failure/vn/dwi2cond_xp_leadfield.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"


def test_run_leadfield_requires_exactly_one_output(monkeypatch, tmp_path: Path) -> None:
    subpath, tensor = _make_subject(tmp_path)

    def no_output(leadfield, cpus):
        Path(leadfield.pathfem).mkdir(parents=True, exist_ok=True)

    _install_fake_simnibs(monkeypatch, runner=no_output)
    with pytest.raises(RuntimeError, match="found 0"):
        run_tdcs_leadfield(subpath, tmp_path / "out", mode="vn", tensor_file=tensor)
