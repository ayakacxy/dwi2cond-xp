import json

import nibabel as nib
import numpy as np

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
