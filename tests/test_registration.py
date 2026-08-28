import json
import os
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.registration import (
    matrix_to_tensor6,
    make_charm_brain_mask,
    polar_rotation,
    register_tensor_affine,
    reorient_affine_tensor,
    tensor6_to_matrix,
)
from dwi2cond_xp.preprocessing.transforms import world_matrix_to_fsl


FSL_VECREG = Path(os.environ.get("FSL_VECREG", "/path/not/configured/vecreg"))


def test_tensor_component_roundtrip():
    tensor = np.array([[1.0, 0.2, 0.3, 2.0, 0.4, 3.0]])
    assert np.array_equal(matrix_to_tensor6(tensor6_to_matrix(tensor)), tensor)


def test_affine_reorientation_uses_polar_rotation():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rotation @ np.diag([2.0, 3.0, 4.0])
    assert np.allclose(polar_rotation(transform[:3, :3]), rotation)
    tensor = np.array([[3.0, 0.0, 0.0, 2.0, 0.0, 1.0]])
    rotated = reorient_affine_tensor(tensor, transform)
    assert np.allclose(rotated, [[2.0, 0.0, 0.0, 3.0, 0.0, 1.0]])


def test_identity_registration_writes_valid_mask_and_qa(tmp_path):
    shape = (3, 4, 2)
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[1:, :, :, 0] = 1.0
    tensor[1:, :, :, 3] = 2.0
    tensor[1:, :, :, 5] = 3.0
    mask = np.any(tensor != 0, axis=-1).astype(np.uint8)
    affine = np.diag([-1.25, 1.25, 1.25, 1.0])
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    output_file = tmp_path / "registered.nii.gz"
    progress = []
    nib.save(nib.Nifti1Image(tensor, affine), tensor_file)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), reference_file)
    nib.save(nib.Nifti1Image(mask, affine), mask_file)

    register_tensor_affine(
        tensor_file,
        reference_file,
        output_file,
        source_mask_file=mask_file,
        progress=lambda current, total: progress.append((current, total)),
    )

    registered = np.asarray(nib.load(output_file).dataobj)
    valid = np.asarray(nib.load(tmp_path / "registered_valid_mask.nii.gz").dataobj)
    qa = json.loads(
        (tmp_path / "registered_registration_qa.json").read_text(encoding="utf-8")
    )
    assert np.array_equal(registered, tensor)
    assert np.array_equal(valid, mask)
    assert qa["valid_voxels"] == int(mask.sum())
    assert progress[-1] == (8, 8)


def test_fsl_vecreg_source_mask_rounds_negative_half_away_from_zero(tmp_path):
    shape = (2, 2, 2)
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[..., 0] = 3.0
    tensor[..., 3] = 2.0
    tensor[..., 5] = 1.0
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    output_file = tmp_path / "registered.nii.gz"
    valid_file = tmp_path / "valid.nii.gz"
    nib.save(nib.Nifti1Image(tensor, np.eye(4)), tensor_file)
    nib.save(
        nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)),
        reference_file,
    )
    transform = np.eye(4)
    transform[0, 3] = 0.5

    register_tensor_affine(
        tensor_file,
        reference_file,
        output_file,
        world_transform=transform,
        source_mask_mode="fsl-vecreg",
        output_valid_mask_file=valid_file,
        interpolation_order=1,
    )

    valid = np.asarray(nib.load(valid_file).dataobj) != 0
    assert not np.any(valid[0])
    assert np.all(valid[1])


def test_charm_brain_mask_uses_official_label_range(tmp_path):
    labels = np.array([[[0, 1, 499, 500, 1000]]], dtype=np.int16)
    affine = np.eye(4)
    labeling = tmp_path / "labeling.nii.gz"
    output = tmp_path / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), labeling)
    make_charm_brain_mask(labeling, output, reference_file=labeling)
    mask = np.asarray(nib.load(output).dataobj)
    assert mask.ravel().tolist() == [0, 1, 1, 0, 0]


def test_tensor_conversion_and_rotation_validate_inputs():
    with pytest.raises(ValueError, match="six components"):
        tensor6_to_matrix(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="3x3"):
        matrix_to_tensor6(np.zeros((2, 6)))
    with pytest.raises(ValueError, match="finite 3x3"):
        polar_rotation(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="singular"):
        polar_rotation(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="4x4"):
        reorient_affine_tensor(np.zeros((1, 6)), np.eye(3))


def _registration_files(tmp_path, *, tensor_shape=(2, 2, 2, 6)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    tensor = np.zeros(tensor_shape, dtype=np.float32)
    if tensor_shape[-1:] == (6,):
        tensor[..., 0] = 3
        tensor[..., 3] = 2
        tensor[..., 5] = 1
    nib.save(nib.Nifti1Image(tensor, np.eye(4)), tensor_file)
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)), reference_file)
    return tensor_file, reference_file


def test_registration_rejects_invalid_parameters_and_tensor(tmp_path):
    tensor_file, reference_file = _registration_files(tmp_path)
    with pytest.raises(ValueError, match="interpolation_order"):
        register_tensor_affine(tensor_file, reference_file, tmp_path / "o.nii.gz", interpolation_order=2)
    with pytest.raises(ValueError, match="workers"):
        register_tensor_affine(tensor_file, reference_file, tmp_path / "o.nii.gz", workers=0)
    bad_tensor, _ = _registration_files(tmp_path / "bad", tensor_shape=(2, 2, 2, 5))
    with pytest.raises(ValueError, match="six-component"):
        register_tensor_affine(bad_tensor, reference_file, tmp_path / "o.nii.gz")
    transform = np.eye(4)
    transform[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 4x4"):
        register_tensor_affine(
            tensor_file, reference_file, tmp_path / "o.nii.gz", world_transform=transform
        )
    with pytest.raises(ValueError, match="reorientation_transform"):
        register_tensor_affine(
            tensor_file,
            reference_file,
            tmp_path / "o.nii.gz",
            reorientation_transform=np.eye(3),
        )
    with pytest.raises(ValueError, match="source_mask_mode"):
        register_tensor_affine(
            tensor_file,
            reference_file,
            tmp_path / "o.nii.gz",
            source_mask_mode="unknown",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        register_tensor_affine(
            tensor_file,
            reference_file,
            tmp_path / "o.nii.gz",
            source_mask_file=tensor_file,
            source_mask_mode="fsl-vecreg",
        )


def test_rotated_registration_uses_masks_progress_and_custom_outputs(tmp_path):
    tensor_file, reference_file = _registration_files(tmp_path)
    source_mask = tmp_path / "source_mask.nii.gz"
    reference_mask = tmp_path / "reference_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), source_mask)
    target_mask = np.ones((2, 2, 2), dtype=np.uint8)
    target_mask[0, 0, 0] = 0
    nib.save(nib.Nifti1Image(target_mask, np.eye(4)), reference_mask)
    transform = np.eye(4)
    transform[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform[0, 3] = 1
    progress = []
    valid = tmp_path / "custom_valid.nii.gz"
    qa = tmp_path / "custom_qa.json"
    output = register_tensor_affine(
        tensor_file,
        reference_file,
        tmp_path / "rotated.nii.gz",
        world_transform=transform,
        source_mask_file=source_mask,
        reference_mask_file=reference_mask,
        output_valid_mask_file=valid,
        qa_file=qa,
        interpolation_order=0,
        workers=2,
        progress=lambda current, total: progress.append((current, total)),
        alignment_assumption="test transform",
    )
    assert output.is_file() and valid.is_file() and qa.is_file()
    assert progress == [(1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)]
    assert json.loads(qa.read_text())["alignment_assumption"] == "test transform"


def test_registration_rejects_mask_grid_mismatches(tmp_path):
    tensor_file, reference_file = _registration_files(tmp_path)
    source_mask = tmp_path / "source_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 2, 2), dtype=np.uint8), np.eye(4)), source_mask)
    with pytest.raises(ValueError, match="source-mask shape"):
        register_tensor_affine(
            tensor_file, reference_file, tmp_path / "o.nii.gz", source_mask_file=source_mask
        )
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.diag([2, 1, 1, 1])),
        source_mask,
    )
    with pytest.raises(ValueError, match="source mask must match"):
        register_tensor_affine(
            tensor_file, reference_file, tmp_path / "o.nii.gz", source_mask_file=source_mask
        )
    reference_mask = tmp_path / "reference_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.diag([2, 1, 1, 1])),
        reference_mask,
    )
    with pytest.raises(ValueError, match="reference mask"):
        register_tensor_affine(
            tensor_file,
            reference_file,
            tmp_path / "o.nii.gz",
            reference_mask_file=reference_mask,
        )


def test_charm_mask_rejects_reference_grid_mismatch(tmp_path):
    labels = tmp_path / "labels.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), labels)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 3)), np.eye(4)), reference)
    with pytest.raises(ValueError, match="does not match"):
        make_charm_brain_mask(labels, tmp_path / "mask.nii.gz", reference_file=reference)


def test_registration_rejects_nonfinite_tensor_before_writing_outputs(tmp_path):
    tensor_file, reference_file = _registration_files(tmp_path)
    image = nib.load(tensor_file)
    values = np.asarray(image.dataobj).copy()
    values[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(values, image.affine, image.header), tensor_file)
    output = tmp_path / "nonfinite.nii.gz"

    with pytest.raises(ValueError, match="input tensor contains NaN or Inf"):
        register_tensor_affine(tensor_file, reference_file, output)

    assert not output.exists()
    assert not (tmp_path / "nonfinite_valid_mask.nii.gz").exists()
    assert not (tmp_path / "nonfinite_registration_qa.json").exists()


def test_registration_masks_follow_fsl_nonzero_semantics(tmp_path):
    tensor_file, reference_file = _registration_files(tmp_path)
    source_mask = tmp_path / "source_negative.nii.gz"
    reference_mask = tmp_path / "reference_negative.nii.gz"
    mask = np.zeros((2, 2, 2), dtype=np.float32)
    mask[0, 1, 1] = -1.0
    nib.save(nib.Nifti1Image(mask, np.eye(4)), source_mask)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), reference_mask)
    output = tmp_path / "negative_mask.nii.gz"

    register_tensor_affine(
        tensor_file,
        reference_file,
        output,
        source_mask_file=source_mask,
        reference_mask_file=reference_mask,
        interpolation_order=0,
    )

    valid = np.asarray(nib.load(tmp_path / "negative_mask_valid_mask.nii.gz").dataobj)
    assert np.count_nonzero(valid) == 1
    assert valid[0, 1, 1] == 1


def test_world_transform_reorientation_uses_fsl_scaled_mm_basis(tmp_path):
    shape = (1, 1, 1)
    source_affine = np.array(
        [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]],
        dtype=float,
    )
    tensor = np.array([3.0, 0.0, 0.0, 2.0, 0.0, 1.0], dtype=np.float32).reshape(
        shape + (6,)
    )
    tensor_file = tmp_path / "basis_tensor.nii.gz"
    reference_file = tmp_path / "basis_reference.nii.gz"
    output = tmp_path / "basis_output.nii.gz"
    nib.save(nib.Nifti1Image(tensor, source_affine), tensor_file)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), reference_file)

    register_tensor_affine(
        tensor_file,
        reference_file,
        output,
        world_transform=np.eye(4),
        interpolation_order=0,
    )

    result = np.asarray(nib.load(output).dataobj)[0, 0, 0]
    np.testing.assert_allclose(result, [2.0, 0.0, 0.0, 3.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.skipif(
    not FSL_VECREG.is_file(),
    reason="FSL reference disabled; set FSL_VECREG to a local vecreg executable",
)
def test_world_transform_basis_matches_real_fsl_vecreg(tmp_path):
    shape = (5, 5, 5)
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[..., 0] = 3.0
    tensor[..., 3] = 2.0
    tensor[..., 5] = 1.0
    source_affine = np.array(
        [[0.0, -1.0, 0.0, 4.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    reference_affine = np.eye(4)
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    ours_file = tmp_path / "ours.nii.gz"
    fsl_file = tmp_path / "fsl.nii.gz"
    matrix_file = tmp_path / "transform.mat"
    nib.save(nib.Nifti1Image(tensor, source_affine), tensor_file)
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.float32), reference_affine),
        reference_file,
    )
    world = np.eye(4)
    np.savetxt(
        matrix_file,
        world_matrix_to_fsl(
            world,
            shape,
            source_affine,
            shape,
            reference_affine,
        ),
    )

    register_tensor_affine(
        tensor_file,
        reference_file,
        ours_file,
        world_transform=world,
        interpolation_order=1,
    )
    environment = os.environ.copy()
    environment["FSLDIR"] = str(FSL_VECREG.parent.parent)
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    subprocess.run(
        [
            str(FSL_VECREG),
            "-i",
            str(tensor_file),
            "-o",
            str(fsl_file),
            "-r",
            str(reference_file),
            "-t",
            str(matrix_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    ours = np.asarray(nib.load(ours_file).dataobj)
    reference = np.asarray(nib.load(fsl_file).dataobj)
    common = np.any(ours != 0.0, axis=-1) & np.any(reference != 0.0, axis=-1)
    assert np.count_nonzero(common) == np.prod(shape)
    np.testing.assert_allclose(ours[common], reference[common], rtol=0.0, atol=1e-6)
