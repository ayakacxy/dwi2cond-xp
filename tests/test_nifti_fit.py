from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.nifti_fit import _fit_z_block, fit_dti_nifti, select_shell_nifti
from dwi2cond_xp.tensor_fit import form_design_matrix


def _gradients() -> tuple[np.ndarray, np.ndarray]:
    directions = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1],
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return np.concatenate([np.zeros(2), np.full(9, 1000.0)]), np.vstack(
        [np.zeros((2, 3)), directions]
    )


def _write_fixture(tmp_path: Path):
    bvals, bvecs = _gradients()
    tensor = np.array([1.4e-3, 8e-5, -4e-5, 7e-4, 6e-5, 4e-4])
    design = form_design_matrix(bvals, bvecs)
    signal = np.exp(-(design[:, :6] @ tensor + design[:, 6] * -np.log(1000.0)))
    data = np.zeros((2, 1, 2, bvals.size), dtype=np.float32)
    data[0, 0, 0] = signal
    data[1, 0, 0] = np.nan
    mask = np.zeros((2, 1, 2), dtype=np.uint8)
    mask[:, :, 0] = 1
    grad = np.zeros((2, 1, 2, 9), dtype=np.float32)
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    paths = {
        "data": tmp_path / "data.nii.gz",
        "mask": tmp_path / "mask.nii.gz",
        "grad": tmp_path / "grad.nii.gz",
        "bvals": tmp_path / "bvals",
        "bvecs": tmp_path / "bvecs",
    }
    nib.save(nib.Nifti1Image(data, affine), paths["data"])
    nib.save(nib.Nifti1Image(mask, affine), paths["mask"])
    nib.save(nib.Nifti1Image(grad, affine), paths["grad"])
    np.savetxt(paths["bvals"], bvals[None, :])
    np.savetxt(paths["bvecs"], bvecs.T)
    return paths, bvals, bvecs


def test_select_shell_writes_data_and_gradients(tmp_path: Path) -> None:
    paths, bvals, _ = _write_fixture(tmp_path)
    output = tmp_path / "selected.nii"
    selected = select_shell_nifti(
        paths["data"],
        paths["bvals"],
        paths["bvecs"],
        output,
        tmp_path / "selected.bval",
        tmp_path / "selected.bvec",
    )
    assert selected.tolist() == list(range(bvals.size))
    assert nib.load(output).shape == (2, 1, 2, bvals.size)
    assert np.loadtxt(tmp_path / "selected.bvec").shape == (3, bvals.size)


def test_select_shell_rejects_bad_image_contract(tmp_path: Path) -> None:
    paths, bvals, bvecs = _write_fixture(tmp_path)
    bad = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), bad)
    with pytest.raises(ValueError, match="four-dimensional"):
        select_shell_nifti(bad, paths["bvals"], paths["bvecs"], "x", "y", "z")
    np.savetxt(paths["bvals"], bvals[:-1][None, :])
    np.savetxt(paths["bvecs"], bvecs[:-1].T)
    with pytest.raises(ValueError, match="fourth axis"):
        select_shell_nifti(paths["data"], paths["bvals"], paths["bvecs"], "x", "y", "z")


def test_serial_fit_writes_custom_outputs_and_progress(tmp_path: Path) -> None:
    paths, _, _ = _write_fixture(tmp_path)
    progress: list[tuple[int, int, int]] = []
    output = tmp_path / "tensor.nii"
    valid = tmp_path / "custom_valid.nii.gz"
    qa_file = tmp_path / "custom_qa.json"
    result = fit_dti_nifti(
        paths["data"],
        paths["bvals"],
        paths["bvecs"],
        paths["mask"],
        output,
        grad_dev_file=paths["grad"],
        workers=1,
        z_chunk=1,
        voxel_batch=1,
        progress=lambda done, total, z: progress.append((done, total, z)),
        valid_mask_file=valid,
        qa_file=qa_file,
    )
    report = json.loads(qa_file.read_text(encoding="utf-8"))
    assert result == output
    assert report["valid_fitted_voxels"] == 1
    assert report["nonfinite_voxels"] == 1
    assert np.asarray(nib.load(valid).dataobj).sum() == 1
    assert progress[-1] == (2, 2, 2)
    for path in report["derived_outputs"].values():
        assert Path(path).is_file()


def test_fit_z_block_covers_invalid_and_empty_blocks(tmp_path: Path) -> None:
    paths, bvals, bvecs = _write_fixture(tmp_path)
    selected = np.arange(bvals.size)
    block = _fit_z_block(
        str(paths["data"]),
        str(paths["mask"]),
        str(paths["grad"]),
        selected,
        bvals,
        bvecs,
        0,
        1,
        1,
    )
    assert block[6:] == (2, 1, 0, 0)
    assert block[5].sum() == 1
    empty = _fit_z_block(
        str(paths["data"]),
        str(paths["mask"]),
        None,
        selected,
        bvals,
        bvecs,
        1,
        2,
        2,
    )
    assert empty[6:] == (0, 0, 0, 0)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_fit_rejects_nonpositive_execution_sizes(tmp_path: Path, bad_value: int) -> None:
    paths, _, _ = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="must be positive"):
        fit_dti_nifti(
            paths["data"], paths["bvals"], paths["bvecs"], paths["mask"], "out", z_chunk=bad_value
        )


def test_fit_rejects_spatial_and_gradient_mismatches(tmp_path: Path) -> None:
    paths, bvals, bvecs = _write_fixture(tmp_path)
    bad_mask = tmp_path / "bad_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1)), np.eye(4)), bad_mask)
    with pytest.raises(ValueError, match="mask shape"):
        fit_dti_nifti(paths["data"], paths["bvals"], paths["bvecs"], bad_mask, "out")
    shifted_mask = tmp_path / "shifted_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 1, 2)), np.diag([2, 1, 1, 1])), shifted_mask)
    with pytest.raises(ValueError, match="mask affine"):
        fit_dti_nifti(paths["data"], paths["bvals"], paths["bvecs"], shifted_mask, "out")
    bad_grad = tmp_path / "bad_grad.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 1, 2, 8)), np.diag([1, 1, 1.2, 1])), bad_grad)
    with pytest.raises(ValueError, match="grad_dev shape"):
        fit_dti_nifti(
            paths["data"], paths["bvals"], paths["bvecs"], paths["mask"], "out", grad_dev_file=bad_grad
        )
    shifted_grad = tmp_path / "shifted_grad.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 1, 2, 9)), np.eye(4)), shifted_grad)
    with pytest.raises(ValueError, match="grad_dev affine"):
        fit_dti_nifti(
            paths["data"], paths["bvals"], paths["bvecs"], paths["mask"], "out", grad_dev_file=shifted_grad
        )
    np.savetxt(paths["bvals"], bvals[:-1][None, :])
    np.savetxt(paths["bvecs"], bvecs[:-1].T)
    with pytest.raises(ValueError, match="fourth axis"):
        fit_dti_nifti(paths["data"], paths["bvals"], paths["bvecs"], paths["mask"], "out")


def test_fit_rejects_three_dimensional_dwi(tmp_path: Path) -> None:
    paths, _, _ = _write_fixture(tmp_path)
    bad_data = tmp_path / "bad_data.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 1, 2)), np.eye(4)), bad_data)
    with pytest.raises(ValueError, match="four-dimensional"):
        fit_dti_nifti(bad_data, paths["bvals"], paths["bvecs"], paths["mask"], "out")


def test_parallel_fit_reports_completed_blocks(tmp_path: Path) -> None:
    paths, _, _ = _write_fixture(tmp_path)
    progress = []
    output = tmp_path / "parallel_tensor.nii.gz"
    fit_dti_nifti(
        paths["data"],
        paths["bvals"],
        paths["bvecs"],
        paths["mask"],
        output,
        workers=2,
        z_chunk=1,
        voxel_batch=1,
        progress=lambda done, total, z: progress.append((done, total, z)),
    )
    assert output.is_file()
    assert len(progress) == 2
    assert progress[-1][0] == 2
