import os
from pathlib import Path
import shutil
import subprocess

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing import (
    copy_nifti_geometry,
    decompose_tensor6,
    fsl_canonical_orientation,
    reorient_bvecs_voxel,
    reorient_spatial_array,
    reorient_tensor6_voxel,
    voxel_basis_transform,
    write_fsl_reoriented,
)
from dwi2cond_xp.registration import matrix_to_tensor6, tensor6_to_matrix


def _permuted_affine(*, positive: bool) -> np.ndarray:
    x_sign = -2.0 if positive else 2.0
    return np.array(
        [
            [0.0, 0.0, 4.0, 10.0],
            [x_sign, 0.0, 0.0, 20.0],
            [0.0, -3.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _write_image(path: Path, values: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    nib.save(image, path)


@pytest.mark.parametrize(
    ("positive", "expected_codes"), [(True, ("R", "A", "S")), (False, ("L", "A", "S"))]
)
def test_fsl_canonical_reorientation_preserves_world_samples(
    tmp_path, positive, expected_codes
):
    values = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    affine = _permuted_affine(positive=positive)
    source = tmp_path / "source.nii.gz"
    output = tmp_path / "nested" / "output.nii.gz"
    _write_image(source, values, affine)

    orientation = fsl_canonical_orientation(affine)
    write_fsl_reoriented(source, output)
    result = nib.load(output)
    new_voxel_to_old = nib.orientations.inv_ornt_aff(orientation, values.shape[:3])

    assert nib.aff2axcodes(result.affine) == expected_codes
    assert np.array_equal(
        np.asarray(result.dataobj), reorient_spatial_array(values, orientation)
    )
    assert np.allclose(result.affine, affine @ new_voxel_to_old)
    assert int(result.header["qform_code"]) == 1
    assert int(result.header["sform_code"]) == 2
    assert result.get_data_dtype() == np.dtype(np.float32)


def test_reorientation_uses_qform_when_sform_is_unset(tmp_path):
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = _permuted_affine(positive=True)
    source = tmp_path / "source.nii"
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, 1)
    image.set_sform(np.eye(4), 0)
    nib.save(image, source)

    output = write_fsl_reoriented(source, tmp_path / "output.nii")
    result = nib.load(output)

    assert nib.aff2axcodes(result.affine) == ("R", "A", "S")
    assert int(result.header["qform_code"]) == 1
    assert int(result.header["sform_code"]) == 0
    assert result.get_data_dtype() == np.dtype(np.int16)


def test_reorientation_can_fuse_float_conversion_and_nonnegative_threshold(tmp_path):
    values = np.array([[[-2, 3], [4, -5]]], dtype=np.int16)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    source = tmp_path / "source.nii.gz"
    _write_image(source, values, affine)

    output = write_fsl_reoriented(
        source,
        tmp_path / "output.nii",
        float32=True,
        nonnegative=True,
    )
    result = nib.load(output)

    assert result.get_data_dtype() == np.dtype(np.float32)
    assert np.array_equal(np.asarray(result.dataobj), np.maximum(values, 0).astype(np.float32))


def test_voxel_basis_transform_preserves_vectors_and_tensors():
    affine = _permuted_affine(positive=True)
    orientation = fsl_canonical_orientation(affine)
    basis = voxel_basis_transform(orientation)
    bvecs = np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 0.75]])
    transformed_bvecs = reorient_bvecs_voxel(bvecs, orientation)
    assert np.allclose(transformed_bvecs, (basis @ bvecs.T).T)

    tensor_matrix = np.array([[4.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 1.0]])
    tensor = matrix_to_tensor6(tensor_matrix)
    transformed = tensor6_to_matrix(reorient_tensor6_voxel(tensor, orientation))
    assert np.allclose(transformed, basis @ tensor_matrix @ basis.T)
    assert np.allclose(np.linalg.eigvalsh(transformed), np.linalg.eigvalsh(tensor_matrix))


def test_tensor_decomposition_outputs_fsl_maps_and_respects_mask():
    tensor = np.zeros((2, 1, 1, 6), dtype=np.float32)
    tensor[0, 0, 0] = [4.0, 0.0, 0.0, 2.0, 0.0, 1.0]
    tensor[1, 0, 0] = [9.0, 0.0, 0.0, 3.0, 0.0, 1.0]
    outputs = decompose_tensor6(tensor, np.array([[[True]], [[False]]]))
    assert set(outputs) == {"FA", "MD", "MO", "L1", "L2", "L3", "V1", "V2", "V3"}
    assert outputs["L1"].ravel().tolist() == [4.0, 0.0]
    assert outputs["L2"].ravel().tolist() == [2.0, 0.0]
    assert outputs["L3"].ravel().tolist() == [1.0, 0.0]
    assert outputs["MD"][0, 0, 0] == pytest.approx(7.0 / 3.0)
    assert np.array_equal(outputs["V1"][0, 0, 0], [1.0, 0.0, 0.0])
    assert all(values.dtype == np.float32 for values in outputs.values())

    all_selected = decompose_tensor6(tensor)
    assert all_selected["L1"][1, 0, 0] == 9.0
    selected, value_range = decompose_tensor6(
        tensor,
        requested=("FA", "V1"),
        return_eigenvalue_range=True,
    )
    assert set(selected) == {"FA", "V1"}
    assert np.array_equal(selected["FA"], all_selected["FA"])
    assert np.array_equal(selected["V1"], all_selected["V1"])
    assert value_range == (1.0, 9.0)

    nonpositive = np.array([[[[-1.0, 0.0, 0.0, -2.0, 0.0, -3.0]]]])
    suppressed = decompose_tensor6(nonpositive)
    assert all(np.count_nonzero(values) == 0 for values in suppressed.values())


def test_tensor_decomposition_validation_errors():
    with pytest.raises(ValueError, match="six components"):
        decompose_tensor6(np.array(1.0))
    with pytest.raises(ValueError, match="validity mask"):
        decompose_tensor6(np.zeros((2, 2, 6)), np.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="semantics"):
        decompose_tensor6(np.zeros((2, 2, 6)), semantics="unknown")
    with pytest.raises(ValueError, match="Unknown tensor decomposition"):
        decompose_tensor6(np.zeros((2, 2, 6)), requested=("unknown",))
    tensor = np.zeros((1, 1, 1, 6))
    tensor[..., 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        decompose_tensor6(tensor)


def test_orientation_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="finite 4x4"):
        fsl_canonical_orientation(np.eye(3))
    singular = np.eye(4)
    singular[2] = singular[1]
    with pytest.raises(ValueError, match="nonsingular"):
        fsl_canonical_orientation(singular)
    dropped_axis = np.eye(4)
    dropped_axis[2, :3] = 1e-13
    with pytest.raises(ValueError, match="nonsingular"):
        fsl_canonical_orientation(dropped_axis)
    with pytest.raises(ValueError, match="at least three"):
        reorient_spatial_array(np.zeros((2, 2)), np.array([[0, 1], [1, 1], [2, 1]]))
    with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
        reorient_spatial_array(np.zeros((2, 2, 2)), np.eye(3))
    with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
        voxel_basis_transform(np.eye(3))
    orientation = np.array([[0, 1], [1, 1], [2, 1]])
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        reorient_bvecs_voxel(np.zeros((3, 2)), orientation)
    with pytest.raises(ValueError, match="NaN or Inf"):
        reorient_bvecs_voxel(np.full((1, 3), np.nan), orientation)
    with pytest.raises(ValueError, match="six components"):
        reorient_tensor6_voxel(np.zeros((2, 5)), orientation)

    invalid = tmp_path / "invalid.nii.gz"
    image = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    image.set_qform(np.eye(4), 0)
    image.set_sform(np.eye(4), 0)
    nib.save(image, invalid)
    with pytest.raises(ValueError, match="valid qform or sform"):
        write_fsl_reoriented(invalid, tmp_path / "output.nii.gz")


def test_copy_nifti_geometry_preserves_destination_data(tmp_path):
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "destination.nii.gz"
    output = tmp_path / "nested" / "copied.nii.gz"
    source_affine = np.array(
        [[-1.5, 0, 0, 8], [0, 2.0, 0, 9], [0, 0, 2.5, 10], [0, 0, 0, 1]],
        dtype=float,
    )
    source_image = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), source_affine)
    source_image.set_qform(source_affine, 1)
    source_image.set_sform(source_affine, 2)
    nib.save(source_image, source)
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    _write_image(destination, values, np.eye(4))

    copy_nifti_geometry(source, destination, output)
    result = nib.load(output)

    assert np.array_equal(np.asarray(result.dataobj), values)
    assert np.allclose(result.get_qform(), source_affine)
    assert np.allclose(result.get_sform(), source_affine)
    assert result.header.get_zooms() == source_image.header.get_zooms()
    assert result.get_data_dtype() == np.dtype(np.int16)

    in_place = copy_nifti_geometry(source, destination)
    assert in_place == destination
    assert np.array_equal(np.asarray(nib.load(in_place).dataobj), values)


def test_copy_geometry_dimension_and_unset_form_contracts(tmp_path):
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "destination.nii.gz"
    source_image = nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    source_image.set_qform(np.eye(4), 0)
    source_image.set_sform(np.eye(4), 0)
    nib.save(source_image, source)
    destination_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.ones((3, 3, 4)), destination_affine), destination)

    with pytest.raises(ValueError, match="matching image shapes"):
        copy_nifti_geometry(source, destination, tmp_path / "bad.nii.gz")
    output = copy_nifti_geometry(
        source, destination, tmp_path / "without_dimensions.nii.gz", copy_dimensions=False
    )
    result = nib.load(output)
    assert result.shape == (3, 3, 4)
    assert int(result.header["qform_code"]) == 0
    assert int(result.header["sform_code"]) == 0


def _fsl_program(name: str) -> Path | None:
    configured = os.environ.get("FSLMATHS")
    if configured:
        candidate = Path(configured).with_name(name)
        if candidate.is_file():
            return candidate
    resolved = shutil.which(name)
    return None if resolved is None else Path(resolved)


def _fsl_environment(program: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["FSLDIR"] = str(program.parent.parent)
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    environment["PATH"] = f"{program.parent}:{environment.get('PATH', '')}"
    return environment


def test_real_fsl_reorientation_and_geometry_ab(tmp_path):
    reorient = _fsl_program("fslreorient2std")
    cpgeom = _fsl_program("fslcpgeom")
    if reorient is None or cpgeom is None:
        pytest.skip("Set FSLMATHS or PATH to run the real FSL orientation A/B")
    values = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    source = tmp_path / "source.nii.gz"
    _write_image(source, values, _permuted_affine(positive=False))
    fsl_output = tmp_path / "fsl.nii.gz"
    our_output = tmp_path / "ours.nii.gz"
    subprocess.run(
        [str(reorient), str(source), str(fsl_output)],
        check=True,
        env=_fsl_environment(reorient),
        capture_output=True,
        text=True,
    )
    write_fsl_reoriented(source, our_output)
    fsl_image = nib.load(fsl_output)
    our_image = nib.load(our_output)
    assert np.array_equal(np.asarray(our_image.dataobj), np.asarray(fsl_image.dataobj))
    assert np.allclose(our_image.get_qform(), fsl_image.get_qform())
    assert np.allclose(our_image.get_sform(), fsl_image.get_sform())
    assert our_image.header.get_zooms() == fsl_image.header.get_zooms()

    geometry_source = tmp_path / "geometry.nii.gz"
    destination = tmp_path / "destination.nii.gz"
    fsl_geometry = tmp_path / "fsl_geometry.nii.gz"
    our_geometry = tmp_path / "our_geometry.nii.gz"
    _write_image(geometry_source, np.zeros((2, 3, 4)), np.diag([-1.5, 2.0, 2.5, 1.0]))
    _write_image(destination, np.arange(24, dtype=np.int16).reshape(2, 3, 4), np.eye(4))
    shutil.copyfile(destination, fsl_geometry)
    subprocess.run(
        [str(cpgeom), str(geometry_source), str(fsl_geometry)],
        check=True,
        env=_fsl_environment(cpgeom),
        capture_output=True,
        text=True,
    )
    copy_nifti_geometry(geometry_source, destination, our_geometry)
    fsl_geometry_image = nib.load(fsl_geometry)
    our_geometry_image = nib.load(our_geometry)
    assert np.array_equal(
        np.asarray(our_geometry_image.dataobj), np.asarray(fsl_geometry_image.dataobj)
    )
    assert np.allclose(our_geometry_image.get_qform(), fsl_geometry_image.get_qform())
    assert np.allclose(our_geometry_image.get_sform(), fsl_geometry_image.get_sform())
    assert our_geometry_image.header.get_zooms() == fsl_geometry_image.header.get_zooms()


def test_real_fsl_tensor_decomposition_ab(tmp_path):
    fslmaths = _fsl_program("fslmaths")
    if fslmaths is None:
        pytest.skip("Set FSLMATHS or PATH to run the real FSL tensor A/B")
    matrices = np.array(
        [
            [[4.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]],
            [[3.0, -0.1, 0.2], [-0.1, 1.5, 0.1], [0.2, 0.1, 0.8]],
        ],
        dtype=np.float32,
    )
    tensor = np.empty((2, 2, 2, 6), dtype=np.float32)
    tensor[0] = matrix_to_tensor6(matrices[0])
    tensor[1] = matrix_to_tensor6(matrices[1])
    tensor_file = tmp_path / "tensor.nii.gz"
    prefix = tmp_path / "fsl_tensor"
    _write_image(tensor_file, tensor, np.eye(4))
    subprocess.run(
        [str(fslmaths), str(tensor_file), "-tensor_decomp", str(prefix)],
        check=True,
        env=_fsl_environment(fslmaths),
        capture_output=True,
        text=True,
    )
    ours = decompose_tensor6(tensor)
    for suffix in ("FA", "MD", "MO", "L1", "L2", "L3"):
        reference = np.asarray(nib.load(tmp_path / f"fsl_tensor_{suffix}.nii.gz").dataobj)
        assert np.allclose(ours[suffix], reference, rtol=1e-6, atol=1e-7)
    for suffix in ("V1", "V2", "V3"):
        reference = np.asarray(nib.load(tmp_path / f"fsl_tensor_{suffix}.nii.gz").dataobj)
        dot = np.sum(ours[suffix] * reference, axis=-1)
        assert np.allclose(np.abs(dot), 1.0, rtol=1e-6, atol=1e-6)
