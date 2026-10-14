"""审计反例及独立的坐标、分辨率和张量物理不变量。"""

import itertools
import os
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import dwi2cond_xp.preprocessing.t1_registration as t1_registration
from dwi2cond_xp.preprocessing.fnirt import expand_fnirt_coefficients
from dwi2cond_xp.preprocessing.nonlinear import (
    register_tensor_nonlinear_nifti,
    resample_tensor_ppd_fsl,
)
from dwi2cond_xp.preprocessing.orientation import (
    reorient_bvecs_fsl,
    validate_fsl_raw_sampling,
    write_fsl_reoriented,
)
from dwi2cond_xp.preprocessing.rigid import write_aligned_b0_mean
from dwi2cond_xp.preprocessing.topup import fsl_coefficient_shape
from dwi2cond_xp.preprocessing.transforms import (
    fsl_matrix_to_world,
    fsl_voxel_to_scaled_mm,
    world_matrix_to_fsl,
)
from dwi2cond_xp.registration import (
    matrix_to_tensor6,
    register_tensor_affine,
    tensor6_to_matrix,
)


def _basis(affine):
    axes = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    axes[:, 0] *= -1 if np.linalg.det(affine[:3, :3]) > 0 else 1
    return axes


def _write(path, values, affine, *, qform=None):
    image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine)
    image.set_qform(affine if qform is None else qform, 1)
    image.set_sform(affine, 2)
    nib.save(image, path)
    return image


def _tensor(shape):
    return np.broadcast_to(np.array([3, 0, 0, 2, 0, 1], dtype=np.float32), shape + (6,)).copy()


def test_all_storage_permutations_preserve_world_tensor_gradient_and_graddev(tmp_path):
    rotation = Rotation.from_euler("xyz", [17, -21, 13], degrees=True).as_matrix()
    matrix = np.array([[3, 0.2, -0.1], [0.2, 2, 0.3], [-0.1, 0.3, 1]], dtype=np.float32)
    graddev = np.array([[0.1, 0.2, -0.3], [0.4, 0.5, 0.6], [-0.7, 0.8, 0.9]], dtype=np.float32)
    vectors = np.array([[0, 0, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=np.float64)
    for index, (axes, signs) in enumerate(itertools.product(itertools.permutations(range(3)), itertools.product((-1, 1), repeat=3))):
        affine = np.eye(4)
        affine[:3, :3] = rotation @ (np.eye(3)[:, axes] * signs * np.array([1.3, 2.1, 2.7]))
        tensor = np.broadcast_to(matrix_to_tensor6(matrix), (2, 3, 4, 6)).copy()
        deviation = np.broadcast_to(graddev.T.reshape(9), (2, 3, 4, 9)).copy()
        tensor_file = tmp_path / f"tensor_{index}.nii.gz"
        dev_file = tmp_path / f"dev_{index}.nii.gz"
        _write(tensor_file, tensor, affine)
        _write(dev_file, deviation, affine)
        tensor_out = write_fsl_reoriented(tensor_file, tmp_path / f"tensor_out_{index}.nii.gz", component_kind="tensor")
        dev_out = write_fsl_reoriented(dev_file, tmp_path / f"dev_out_{index}.nii.gz", component_kind="grad-dev")
        output_image = nib.load(tensor_out)
        old_basis, new_basis = _basis(affine), _basis(output_image.affine)
        result_matrix = tensor6_to_matrix(np.asarray(output_image.dataobj)[0, 0, 0])
        result_dev = np.asarray(nib.load(dev_out).dataobj)[0, 0, 0].reshape(3, 3).T
        np.testing.assert_allclose(new_basis @ result_matrix @ new_basis.T, old_basis @ matrix @ old_basis.T, atol=3e-7)
        np.testing.assert_allclose(new_basis @ result_dev @ new_basis.T, old_basis @ graddev @ old_basis.T, atol=2e-7)
        converted = reorient_bvecs_fsl(vectors, affine)
        np.testing.assert_allclose(converted @ new_basis.T, vectors @ old_basis.T, atol=8e-8)
        np.testing.assert_allclose(np.linalg.norm(converted, axis=1), np.linalg.norm(vectors, axis=1), atol=1e-15)
        np.testing.assert_allclose(np.linalg.eigvalsh(result_matrix), np.linalg.eigvalsh(matrix), atol=3e-7)


def test_literal_storage_mode_preserves_channels_and_rejects_ambiguous_types(tmp_path):
    affine = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    source = tmp_path / "tensor.nii.gz"
    _write(source, _tensor((2, 3, 4)), affine)
    output = write_fsl_reoriented(source, tmp_path / "literal.nii.gz", component_kind="tensor", component_mode="simnibs46-literal")
    np.testing.assert_array_equal(np.asarray(nib.load(output).dataobj)[0, 0, 0], [3, 0, 0, 2, 0, 1])
    for kwargs in ({"component_kind": "unknown"}, {"component_mode": "unknown"}, {"component_kind": "grad-dev"}):
        with pytest.raises(ValueError):
            write_fsl_reoriented(source, tmp_path / "invalid.nii.gz", **kwargs)


def test_header_scaled_mm_conversion_uses_pixdim_and_roundtrips():
    shape = (9, 8, 7)
    moving = np.diag([-2, 3, 4, 1])
    reference = np.diag([1.5, 2.5, 3.5, 1])
    np.testing.assert_array_equal(fsl_voxel_to_scaled_mm(shape, moving, (1, 1, 1)), np.eye(4))
    fsl = np.eye(4)
    fsl[:3, 3] = [1.2, -0.4, 0.7]
    world = fsl_matrix_to_world(fsl, shape, moving, shape, reference, moving_voxel_sizes=(1, 1, 1), reference_voxel_sizes=(0.7, 0.8, 0.9))
    recovered = world_matrix_to_fsl(world, shape, moving, shape, reference, moving_voxel_sizes=(1, 1, 1), reference_voxel_sizes=(0.7, 0.8, 0.9))
    np.testing.assert_allclose(recovered, fsl, atol=2e-15)
    for dimensions in ((1, 0, 1), (1, 2), (np.nan, 1, 1)):
        with pytest.raises(ValueError, match="voxel_sizes"):
            fsl_voxel_to_scaled_mm(shape, moving, dimensions)


def test_raw_sampling_guard_rejects_mismatch_before_standalone_motion(tmp_path):
    source = tmp_path / "raw.nii.gz"
    image = _write(source, np.ones((5, 5, 5, 2)), np.diag([-2, 1, 1, 1]), qform=np.diag([-1, 1, 1, 1]))
    with pytest.raises(ValueError, match="header pixdim"):
        validate_fsl_raw_sampling(image)
    with pytest.raises(ValueError, match="header pixdim"):
        write_aligned_b0_mean(source, tmp_path / "missing_bvals", tmp_path / "output.nii.gz")
    valid = nib.Nifti1Image(np.ones((5, 5, 5)), np.diag([-2, 1, 1, 1]))
    validate_fsl_raw_sampling(valid)


def test_flirt_nifti_producer_keeps_header_sampling_separate_from_world(tmp_path, monkeypatch):
    shape = (5, 5, 5)
    moving = _write(tmp_path / "moving.nii.gz", np.ones(shape), np.diag([-2, 1, 1, 1]), qform=np.diag([-1, 1, 1, 1]))
    reference = _write(tmp_path / "reference.nii.gz", np.ones(shape), np.diag([-1, 1, 1, 1]))
    captured = {}

    def record(*args, **kwargs):
        captured["reference_sampling"] = args[4]
        captured["moving_sampling"] = args[5]
        captured["initial_transform"] = kwargs["qsform_matrix"]
        return object()

    monkeypatch.setattr(t1_registration, "register_flirt_affine", record)
    t1_registration._estimate(reference, np.ones(shape), np.ones(shape), moving, np.ones(shape), degrees_of_freedom=6, workers=1, progress=None)
    np.testing.assert_array_equal(captured["reference_sampling"], np.eye(4))
    np.testing.assert_array_equal(captured["moving_sampling"], np.eye(4))
    np.testing.assert_array_equal(captured["initial_transform"], np.diag([2, 1, 1, 1]))


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("dimensions", [(1, 1, 1), (2, 2, 2), (2, 3, 4)])
def test_physical_ppd_is_invariant_to_grid_resolution_and_handedness(dimensions, positive):
    shape = (9, 9, 9)
    affine = np.diag([dimensions[0] * (1 if positive else -1), dimensions[1], dimensions[2], 1])
    sampling = fsl_voxel_to_scaled_mm(shape, affine)
    x = sampling[0, 0] * np.indices(shape, dtype=np.float32)[0] + sampling[0, 3]
    warp = np.zeros(shape + (3,), dtype=np.float32)
    warp[..., 1] = 0.1 * x
    result = resample_tensor_ppd_fsl(_tensor(shape), affine, shape, affine, warp)
    expected_jacobian = np.eye(3)
    expected_jacobian[1, 0] = 0.1
    np.testing.assert_allclose(result.backward_jacobian, np.broadcast_to(expected_jacobian, result.backward_jacobian.shape), atol=5e-8)
    center = tensor6_to_matrix(result.tensor[4, 4, 4])
    eigenvalues, eigenvectors = np.linalg.eigh(center)
    np.testing.assert_allclose(eigenvalues, [1, 2, 3], atol=3e-7)
    expected_direction = np.array([1, -0.1, 0]) / np.sqrt(1.01)
    assert abs(np.dot(eigenvectors[:, -1], expected_direction)) > 1 - 2e-7
    np.testing.assert_allclose(np.linalg.det(result.backward_jacobian), result.backward_jacobian_determinant, atol=1e-15)
    assert not np.any(result.fold_mask)


def test_physical_ppd_fold_qa_and_rotation_use_the_same_jacobian():
    shape = (7, 7, 7)
    affine = np.diag([-2, 2, 2, 1])
    warp = np.zeros(shape + (3,), dtype=np.float32)
    warp[..., 0] = -1.5 * np.indices(shape, dtype=np.float32)[0]
    physical = resample_tensor_ppd_fsl(_tensor(shape), affine, shape, affine, warp)
    np.testing.assert_allclose(physical.backward_jacobian[..., 0, 0], 0.25, atol=0)
    assert not np.any(physical.fold_mask)
    literal = resample_tensor_ppd_fsl(_tensor(shape), affine, shape, affine, warp, jacobian_mode="simnibs46-literal", compatibility_mode="robust")
    assert not np.any(literal.valid_mask)
    with pytest.raises(ValueError, match="jacobian_mode"):
        resample_tensor_ppd_fsl(_tensor(shape), affine, shape, affine, warp, jacobian_mode="unknown")


def _fsl_environment():
    environment = os.environ.copy()
    environment["FSLDIR"] = "/usr/local/fsl"
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    return environment


FSL_VECREG = Path(os.environ.get("FSL_VECREG", "/path/not/configured/vecreg"))
FSL_CONVERTWARP = Path(os.environ.get("FSL_CONVERTWARP", "/path/not/configured/convertwarp"))


@pytest.mark.skipif(not FSL_VECREG.is_file(), reason="FSL vecreg unavailable")
def test_header_pixdim_sform_mismatch_matches_real_vecreg(tmp_path):
    shape = (9, 8, 7)
    source_affine = np.diag([-2, 1, 1, 1])
    reference_affine = np.diag([-1, 1, 1, 1])
    values = _tensor(shape)
    values[..., 0] += 0.2 * np.indices(shape, dtype=np.float32)[0]
    source = tmp_path / "source.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    _write(source, values, source_affine, qform=reference_affine)
    _write(reference, np.ones(shape), reference_affine)
    transform_file = tmp_path / "identity.mat"
    np.savetxt(transform_file, np.eye(4))
    subprocess.run([str(FSL_VECREG), "-i", str(source), "-r", str(reference), "-o", str(tmp_path / "fsl.nii.gz"), "-t", str(transform_file)], check=True, capture_output=True, env=_fsl_environment())
    transform = fsl_matrix_to_world(np.eye(4), shape, source_affine, shape, reference_affine, moving_voxel_sizes=(1, 1, 1), reference_voxel_sizes=(1, 1, 1))
    register_tensor_affine(source, reference, tmp_path / "ours.nii.gz", world_transform=transform, source_mask_mode="fsl-vecreg")
    np.testing.assert_array_equal(np.asarray(nib.load(tmp_path / "ours.nii.gz").dataobj), np.asarray(nib.load(tmp_path / "fsl.nii.gz").dataobj))
    warp = tmp_path / "zero_warp.nii.gz"
    _write(warp, np.zeros(shape + (3,)), reference_affine)
    register_tensor_nonlinear_nifti(source, reference, warp, tmp_path / "dense.nii.gz")
    np.testing.assert_array_equal(np.asarray(nib.load(tmp_path / "dense.nii.gz").dataobj), np.asarray(nib.load(tmp_path / "fsl.nii.gz").dataobj))


@pytest.mark.skipif(not FSL_VECREG.is_file() or not FSL_CONVERTWARP.is_file(), reason="FSL vecreg/convertwarp unavailable")
def test_two_mm_coefficient_and_dense_warp_ppd_are_equivalent(tmp_path):
    shape = (9, 9, 9)
    spacing = (2, 2, 2)
    affine = np.diag([-2, 2, 2, 1])
    coefficients = np.zeros(fsl_coefficient_shape(shape, spacing) + (3,), dtype=np.float32)
    coefficients[..., 1] = 0.4 * (np.indices(coefficients.shape[:3])[0] - 1)
    reference = tmp_path / "reference.nii.gz"
    source = tmp_path / "tensor.nii.gz"
    coefficient_file = tmp_path / "coefficients.nii.gz"
    dense_file = tmp_path / "dense.nii.gz"
    matrix_file = tmp_path / "identity.mat"
    _write(reference, np.ones(shape), affine)
    _write(source, _tensor(shape), affine)
    np.savetxt(matrix_file, np.eye(4))
    header = nib.Nifti1Header()
    header.set_data_shape(coefficients.shape)
    header.set_data_dtype(np.float32)
    header.set_intent(2007, allow_unknown=True)
    header["intent_p1"], header["intent_p2"], header["intent_p3"] = (2, 2, 2)
    header["pixdim"][1:4] = spacing
    header["qform_code"] = 1
    header["qoffset_x"], header["qoffset_y"], header["qoffset_z"] = shape
    header["sform_code"] = 1
    header["srow_x"], header["srow_y"], header["srow_z"] = np.eye(4)[:3]
    nib.save(nib.Nifti1Image(coefficients, None, header), coefficient_file)
    subprocess.run([str(FSL_CONVERTWARP), f"--ref={reference}", f"--warp1={coefficient_file}", f"--out={dense_file}", "--relout"], check=True, capture_output=True, env=_fsl_environment())
    dense = np.asarray(nib.load(dense_file).dataobj)
    # convertwarp 的输出省略 qform/sform；把已指定的 reference 网格显式写回。
    _write(dense_file, dense, affine)
    expansion = expand_fnirt_coefficients(coefficients, shape, affine, np.eye(4), knot_spacing=spacing)
    expanded_dense_file = tmp_path / "expanded_dense.nii.gz"
    _write(expanded_dense_file, expansion.displacement, affine)
    # convertwarp 先用 float32 存 absolute-mm 再减坐标，误差由网格最大毫米坐标限定。
    absolute_mm_rounding = np.finfo(np.float32).eps * 2 * (max(shape) - 1)
    np.testing.assert_allclose(expansion.displacement, dense, rtol=0, atol=absolute_mm_rounding)
    for kind, warp_file in (("coefficients", coefficient_file), ("displacement", dense_file)):
        subprocess.run([str(FSL_VECREG), "-i", str(source), "-r", str(reference), "-w", str(warp_file), "-o", str(tmp_path / f"fsl_{kind}.nii.gz")], check=True, capture_output=True, env=_fsl_environment())
        options = {} if kind == "displacement" else {"affine_matrix_file": matrix_file, "knot_spacing": spacing}
        for mode in ("physical", "simnibs46-literal"):
            register_tensor_nonlinear_nifti(source, reference, warp_file, tmp_path / f"{kind}_{mode}.nii.gz", warp_kind=kind, jacobian_mode=mode, **options)
    fsl_dense = np.asarray(nib.load(tmp_path / "fsl_displacement.nii.gz").dataobj)
    np.testing.assert_allclose(np.asarray(nib.load(tmp_path / "fsl_coefficients.nii.gz").dataobj), fsl_dense, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(nib.load(tmp_path / "displacement_simnibs46-literal.nii.gz").dataobj), fsl_dense, rtol=0, atol=1e-6)
    for mode in ("physical", "simnibs46-literal"):
        dense_output = np.asarray(nib.load(tmp_path / f"displacement_{mode}.nii.gz").dataobj)
        coefficient_output = np.asarray(nib.load(tmp_path / f"coefficients_{mode}.nii.gz").dataobj)
        # 相同 float32 位移应逐位相同；另用 convertwarp 独立展开核对内部支持。
        register_tensor_nonlinear_nifti(source, reference, expanded_dense_file, tmp_path / f"expanded_{mode}.nii.gz", jacobian_mode=mode)
        expanded_output = np.asarray(nib.load(tmp_path / f"expanded_{mode}.nii.gz").dataobj)
        np.testing.assert_array_equal(coefficient_output, expanded_output)
        np.testing.assert_allclose(coefficient_output[2:-2, 2:-2, 2:-2], dense_output[2:-2, 2:-2, 2:-2], rtol=0, atol=1e-6)
        _, directions = np.linalg.eigh(tensor6_to_matrix(coefficient_output[4, 4, 4]))
        angle = np.degrees(np.arctan2(abs(directions[1, -1]), abs(directions[0, -1])))
        assert angle == pytest.approx(np.degrees(np.arctan(0.1 if mode == "physical" else 0.2)), abs=2e-5)


def test_fnirt_coefficient_path_explicitly_rejects_unhandled_header_mismatch(tmp_path):
    shape = (5, 5, 5)
    source, reference, coefficient = (tmp_path / name for name in ("source.nii.gz", "reference.nii.gz", "coefficient.nii.gz"))
    _write(source, _tensor(shape), np.diag([-1, 1, 1, 1]))
    _write(reference, np.ones(shape), np.diag([-2, 1, 1, 1]), qform=np.diag([-1, 1, 1, 1]))
    _write(coefficient, np.zeros(fsl_coefficient_shape(shape, (2, 2, 2)) + (3,)), np.eye(4))
    matrix_file = tmp_path / "identity.mat"
    np.savetxt(matrix_file, np.eye(4))
    with pytest.raises(ValueError, match="header pixdim"):
        register_tensor_nonlinear_nifti(source, reference, coefficient, tmp_path / "output.nii.gz", warp_kind="coefficients", affine_matrix_file=matrix_file, knot_spacing=(2, 2, 2))
