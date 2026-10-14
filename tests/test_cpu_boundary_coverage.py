"""算法修复与数组优化新增入口的失败边界及独立数值对照。"""

import json

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.conductivity import fsl_tensor_basis
from dwi2cond_xp.nifti_fit import _save_derived_maps
from dwi2cond_xp.preprocessing import eddy, fieldmap, legacy, nomoco, tensor_ops
from dwi2cond_xp.tensor_fit import _reference_normal_equations, form_design_matrix


def test_literal_basis_rejects_linearly_dependent_nonzero_spatial_axes():
    affine = np.eye(4)
    affine[:3, :3] = [[1, 1, 0], [0, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="linearly independent"):
        fsl_tensor_basis(affine, basis_mode="simnibs46-literal")


def test_derived_map_private_executor_matches_serial_image_and_calibration(tmp_path):
    """自建写出线程与串行输出保持完整图像及校准字段一致。"""
    reference = nib.Nifti1Image(np.ones((2, 3, 4)), np.diag([1.2, 2.3, 3.4, 1.0]))
    reference.set_qform(reference.affine, 1)
    reference.set_sform(reference.affine, 2)
    outputs = {
        "L1": np.full(reference.shape, 0.002),
        "FA": np.full(reference.shape, 0.4),
        "V1": np.broadcast_to(np.array([1.0, 0.0, 0.0]), reference.shape + (3,)),
    }
    serial = _save_derived_maps(outputs, reference, tmp_path / "serial.nii.gz", workers=1)
    threaded = _save_derived_maps(outputs, reference, tmp_path / "threaded.nii.gz", workers=3)
    for suffix in outputs:
        first, second = nib.load(serial[suffix]), nib.load(threaded[suffix])
        np.testing.assert_array_equal(np.asarray(first.dataobj), np.asarray(second.dataobj))
        assert first.header.binaryblock == second.header.binaryblock
        np.testing.assert_array_equal(first.affine, second.affine)


@pytest.mark.parametrize("runner", ["eddy", "legacy", "nomoco_bvecs"])
def test_component_mode_rejection_precedes_reading_or_publication(tmp_path, runner):
    output = tmp_path / "must_not_publish"
    with pytest.raises(ValueError, match="component_mode must be"):
        if runner == "eddy":
            eddy.run_eddy_nifti(
                "missing-dwi", "missing-bvals", "missing-bvecs", "missing-mask", output,
                readout_seconds=0.05, phase_encoding_direction="y", component_mode="bad",
            )
        elif runner == "legacy":
            legacy.run_legacy_nifti(
                "missing-dwi", "missing-bvals", "missing-bvecs", output, component_mode="bad",
            )
        else:
            nomoco._prepare_bvecs(
                "missing-dwi", "missing-bvals", "missing-bvecs", output, component_mode="bad",
            )
    assert not output.exists()


@pytest.mark.parametrize("affine", [np.ones((3, 3)), np.full((4, 4), np.nan), np.diag([1, 0, 1, 1])])
def test_radiological_flip_rejects_invalid_spatial_geometry(affine):
    with pytest.raises(ValueError, match="finite 4x4|nonsingular spatial axes"):
        fieldmap._radiological_flip_required(affine)


def test_nomoco_reordered_gradient_file_matches_explicit_basis_mapping(tmp_path):
    """三轴循环重排后方向由两个 FSL 基底决定，检查落盘文件精度。"""
    affine = np.eye(4)
    affine[:3, :3] = [[0, 0, 2], [3, 0, 0], [0, 4, 0]]
    source = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4)), affine), source)
    bvals, bvecs = tmp_path / "bvals", tmp_path / "bvecs"
    original = np.vstack([np.zeros(3), np.eye(3)])
    np.savetxt(bvals, [0, 1000, 1000, 1000])
    np.savetxt(bvecs, original.T)
    output = tmp_path / "prepared/DWIbvecs"
    actual = nomoco._prepare_bvecs(source, bvals, bvecs, output)
    mapping = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    expected = original @ mapping.T
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.loadtxt(output).T, expected)
    np.testing.assert_array_equal(np.linalg.norm(actual, axis=1), np.linalg.norm(original, axis=1))


def test_nomoco_reused_preparation_emits_progress_without_recomputing_b0(tmp_path, monkeypatch):
    """复用准备数据仍报告所有阶段进度，并执行真实第二次拟合。"""
    directions = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
        [0, 1, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1],
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvals_values = np.r_[0, np.full(9, 1000.0)]
    bvec_values = np.vstack([np.zeros(3), directions])
    tensor = np.array([0.0014, 0, 0, 0.0007, 0, 0.0004])
    signal = 1000 * np.exp(-(form_design_matrix(bvals_values, bvec_values)[:, :6] @ tensor))
    data = np.broadcast_to(signal, (2, 2, 2, len(signal))).astype(np.float32)
    source, bvals, bvecs = tmp_path / "dwi.nii.gz", tmp_path / "bvals", tmp_path / "bvecs"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    np.savetxt(bvals, bvals_values)
    np.savetxt(bvecs, bvec_values.T)
    preparation = tmp_path / "preparation"
    preparation.mkdir()
    nib.save(nib.Nifti1Image(data, np.eye(4)), preparation / "DWIraw.nii")
    (preparation / "DWIbvecs").write_bytes(bvecs.read_bytes())
    nib.save(nib.Nifti1Image(data[..., 0], np.eye(4)), preparation / "nodif.nii.gz")
    nib.save(nib.Nifti1Image(np.ones(data.shape[:3], dtype=np.uint8), np.eye(4)), preparation / "nodif_brain_mask.nii.gz")
    (preparation / "nodif_qa.json").write_text('{"status":"completed"}')
    metadata = {
        "status": "completed", "workers": 1, "b0_threshold": 0.0,
        "component_mode": "physical", "bet_backend": "optimized",
        "bet_fractional_threshold": 0.2, "data": str(source.resolve()),
        "bvals": str(bvals.resolve()), "bvecs": str(bvecs.resolve()), "grad_dev": None,
        "bet": {"backend": "optimized", "passes": 0, "mask_voxels": 8},
    }
    (preparation / "preparation_qa.json").write_text(json.dumps(metadata))

    def unexpected(*args, **kwargs):
        raise AssertionError("Accepted b0/BET preparation must not be recomputed")

    monkeypatch.setattr(nomoco, "write_aligned_b0_mean", unexpected)
    monkeypatch.setattr(nomoco, "write_bet_brain_mask", unexpected)
    progress = []
    output = tmp_path / "nomoco"
    report = nomoco.run_nomoco_nifti(
        source, bvals, bvecs, output, workers=1, _prepared_directory=preparation,
        progress=lambda phase, done, total: progress.append((phase, done, total)),
    )
    assert report["status"] == "completed"
    assert ("align_b0", 1, 1) in progress
    assert {item[0] for item in progress} == {"normalize_input", "align_b0", "brain_mask", "fit_dti"}
    assert (output / "nodif.nii.gz").read_bytes() == (preparation / "nodif.nii.gz").read_bytes()
    fitted = np.asarray(nib.load(output / "DTI_tensor.nii.gz").dataobj)
    np.testing.assert_allclose(fitted, np.broadcast_to(tensor, fitted.shape), rtol=2e-6, atol=1e-10)


def test_voxelwise_reference_normal_equations_match_explicit_weighted_products():
    """每体素独立设计矩阵的规约与显式加权正规方程一致。"""
    random = np.random.default_rng(43)
    design = random.normal(size=(3, 11, 7))
    weights = random.uniform(0.1, 2, size=(3, 11))
    signal = random.normal(size=(3, 11))
    normal, rhs = _reference_normal_equations(design, weights, signal)
    expected_normal = np.stack([row.T @ (weight[:, None] * row) for row, weight in zip(design, weights)])
    expected_rhs = np.stack([-row.T @ (weight * log) for row, weight, log in zip(design, weights, signal)])
    np.testing.assert_allclose(normal, expected_normal, rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(rhs, expected_rhs, rtol=2e-15, atol=1e-14)


@pytest.mark.parametrize("workers", [0, 1.5])
@pytest.mark.parametrize("representation", ["matrix", "components"])
def test_batched_eigensystem_rejects_invalid_worker_count(workers, representation):
    with pytest.raises(ValueError, match="positive integer"):
        if representation == "matrix":
            tensor_ops._eigh_batched(np.eye(3)[None], workers=workers)
        else:
            tensor_ops._eigh_tensor6_batched(np.ones((1, 6)), workers=workers)


@pytest.mark.parametrize("count,workers", [(3, 1), (100_001, 3)])
def test_matrix_eigensystem_blocks_match_analytic_rotated_spectrum(count, workers):
    """串行及跨块 LAPACK 保持谱和矩阵重建，含最后不完整片段。"""
    angle = 0.37
    basis = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    matrix = basis @ np.diag([1, 2, 4]) @ basis.T
    matrices = np.broadcast_to(matrix, (count, 3, 3)).copy()
    values, vectors = tensor_ops._eigh_batched(matrices, workers=workers)
    np.testing.assert_allclose(values, np.broadcast_to([1, 2, 4], values.shape), rtol=2e-15, atol=2e-15)
    rebuilt = (vectors * values[:, None, :]) @ vectors.transpose(0, 2, 1)
    np.testing.assert_allclose(rebuilt, matrices, rtol=2e-15, atol=2e-15)
