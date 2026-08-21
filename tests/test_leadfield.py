from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.leadfield import (
    LEADFIELD_DATASET,
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
