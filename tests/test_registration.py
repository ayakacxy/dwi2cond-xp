import json

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
    nib.save(nib.Nifti1Image(tensor, affine), tensor_file)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), reference_file)
    nib.save(nib.Nifti1Image(mask, affine), mask_file)

    register_tensor_affine(
        tensor_file,
        reference_file,
        output_file,
        source_mask_file=mask_file,
    )

    registered = np.asarray(nib.load(output_file).dataobj)
    valid = np.asarray(nib.load(tmp_path / "registered_valid_mask.nii.gz").dataobj)
    qa = json.loads(
        (tmp_path / "registered_registration_qa.json").read_text(encoding="utf-8")
    )
    assert np.array_equal(registered, tensor)
    assert np.array_equal(valid, mask)
    assert qa["valid_voxels"] == int(mask.sum())


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
    bad_tensor, _ = _registration_files(tmp_path / "bad", tensor_shape=(2, 2, 2, 5))
    with pytest.raises(ValueError, match="six-component"):
        register_tensor_affine(bad_tensor, reference_file, tmp_path / "o.nii.gz")
    transform = np.eye(4)
    transform[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 4x4"):
        register_tensor_affine(
            tensor_file, reference_file, tmp_path / "o.nii.gz", world_transform=transform
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
