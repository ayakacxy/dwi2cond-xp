from pathlib import Path
import json
import os
import subprocess

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.nifti_fit import fit_dti_nifti
from dwi2cond_xp.tensor_fit import form_design_matrix


FSL_DTIFIT = Path(os.environ.get("FSL_DTIFIT", "/path/not/configured/dtifit"))


def _gradient_fixture():
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


@pytest.mark.skipif(
    not FSL_DTIFIT.is_file(),
    reason="FSL reference disabled; set FSL_DTIFIT to a local dtifit executable",
)
def test_small_nifti_matches_fsl_wls_with_grad_dev(tmp_path):
    bvals, bvecs = _gradient_fixture()
    shape = (2, 2, 2)
    grad = np.zeros(shape + (9,), dtype=np.float32)
    grad[..., 0] = np.linspace(-0.01, 0.01, np.prod(shape)).reshape(shape)
    grad[..., 4] = 0.005
    grad[..., 8] = -0.004

    base_tensor = np.array([1.4e-3, 8e-5, -4e-5, 7e-4, 6e-5, 4e-4])
    data = np.empty(shape + (bvals.size,), dtype=np.float32)
    for index in np.ndindex(shape):
        tensor = base_tensor * (1.0 + 0.01 * np.ravel_multi_index(index, shape))
        design = form_design_matrix(bvals, bvecs, grad[index][None, :])[0]
        data[index] = np.exp(
            -(design[:, :6] @ tensor + design[:, 6] * -np.log(1000.0))
        )

    affine = np.array(
        [[-1.25, 0, 0, 1.25], [0, 1.25, 0, -1.25], [0, 0, 1.25, -1.25], [0, 0, 0, 1]]
    )
    data_file = tmp_path / "data.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    grad_file = tmp_path / "grad_dev.nii.gz"
    bvals_file = tmp_path / "bvals"
    bvecs_file = tmp_path / "bvecs"
    nib.save(nib.Nifti1Image(data, affine), data_file)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), mask_file)
    nib.save(nib.Nifti1Image(grad, affine), grad_file)
    np.savetxt(bvals_file, bvals[None, :], fmt="%.8g")
    np.savetxt(bvecs_file, bvecs.T, fmt="%.12g")

    ours_file = tmp_path / "ours_tensor.nii.gz"
    fit_dti_nifti(
        data_file,
        bvals_file,
        bvecs_file,
        mask_file,
        ours_file,
        grad_dev_file=grad_file,
        z_chunk=1,
        voxel_batch=3,
        workers=2,
    )

    fsl_prefix = tmp_path / "fsl"
    environment = os.environ.copy()
    environment["FSLDIR"] = str(FSL_DTIFIT.parent.parent)
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    subprocess.run(
        [
            str(FSL_DTIFIT),
            "-k",
            str(data_file),
            "-m",
            str(mask_file),
            "-r",
            str(bvecs_file),
            "-b",
            str(bvals_file),
            "-o",
            str(fsl_prefix),
            "--wls",
            "--save_tensor",
            "--sse",
            f"--gradnonlin={grad_file}",
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    ours = np.asarray(nib.load(ours_file).dataobj)
    reference = np.asarray(nib.load(tmp_path / "fsl_tensor.nii.gz").dataobj)
    assert ours.shape == reference.shape == shape + (6,)
    assert np.allclose(ours, reference, rtol=2e-5, atol=2e-8)
    for suffix in ("FA", "MD", "MO", "L1", "L2", "L3", "S0", "sse"):
        ours_map = np.asarray(nib.load(tmp_path / f"ours_tensor_{suffix}.nii.gz").dataobj)
        fsl_map = np.asarray(nib.load(tmp_path / f"fsl_{suffix}.nii.gz").dataobj)
        assert np.allclose(ours_map, fsl_map, rtol=2e-5, atol=2e-7), suffix
    for suffix in ("V1", "V2", "V3"):
        ours_vector = np.asarray(
            nib.load(tmp_path / f"ours_tensor_{suffix}.nii.gz").dataobj
        )
        fsl_vector = np.asarray(nib.load(tmp_path / f"fsl_{suffix}.nii.gz").dataobj)
        # Eigenvector sign is arbitrary, so compare axial angle rather than sign.
        axial_dot = np.abs(np.sum(ours_vector * fsl_vector, axis=-1))
        assert np.allclose(axial_dot, 1.0, rtol=0, atol=2e-5), suffix


def test_invalid_voxel_is_explicitly_recorded(tmp_path):
    """All-nonpositive signals must be zeroed and reported without NaN."""
    bvals, bvecs = _gradient_fixture()
    shape = (2, 1, 1)
    data = np.full(shape + (bvals.size,), 100.0, dtype=np.float32)
    data[1, 0, 0, :] = 0.0
    affine = np.eye(4)
    data_file = tmp_path / "data.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    bvals_file = tmp_path / "bvals"
    bvecs_file = tmp_path / "bvecs"
    output_file = tmp_path / "tensor.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), data_file)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), mask_file)
    np.savetxt(bvals_file, bvals[None, :], fmt="%.8g")
    np.savetxt(bvecs_file, bvecs.T, fmt="%.12g")

    fit_dti_nifti(
        data_file,
        bvals_file,
        bvecs_file,
        mask_file,
        output_file,
        z_chunk=1,
        workers=2,
    )

    tensor = np.asarray(nib.load(output_file).dataobj)
    valid = np.asarray(nib.load(tmp_path / "tensor_valid_mask.nii.gz").dataobj)
    qa = json.loads((tmp_path / "tensor_qa.json").read_text(encoding="utf-8"))
    assert np.all(np.isfinite(tensor))
    assert np.count_nonzero(tensor[1, 0, 0]) == 0
    assert valid[:, 0, 0].tolist() == [1, 0]
    assert qa["valid_fitted_voxels"] == 1
    assert qa["all_nonpositive_voxels"] == 1
