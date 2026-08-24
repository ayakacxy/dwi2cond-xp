from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

import dwi2cond_xp.preprocessing.rigid as rigid_module
from dwi2cond_xp.preprocessing.rigid import (
    estimate_rigid_transform,
    normalized_correlation_cost,
    resample_rigid,
    rigid_world_matrix,
    write_aligned_b0_mean,
)


def _phantom(shape: tuple[int, int, int] = (33, 31, 29)) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64)
    fractions = (
        (1.0, (0.28, 0.38, 0.42), (0.12, 0.18, 0.14)),
        (0.7, (0.70, 0.67, 0.63), (0.09, 0.13, 0.10)),
        (0.45, (0.25, 0.76, 0.72), (0.07, 0.08, 0.14)),
        (0.3, (0.78, 0.24, 0.27), (0.10, 0.06, 0.08)),
    )
    values = np.zeros(shape, dtype=np.float64)
    for amplitude, center, width in fractions:
        exponent = sum(
            (
                (grid[axis] - center[axis] * (shape[axis] - 1))
                / (width[axis] * shape[axis])
            )
            ** 2
            for axis in range(3)
        )
        values += amplitude * np.exp(-exponent)
    values = (values * 1000.0).astype(np.float32)
    values[values < 0.5] = 0
    return values


def _affine() -> np.ndarray:
    return np.array(
        [[-2.0, 0, 0, 32.0], [0, 2.0, 0, -30.0], [0, 0, 2.0, -28.0], [0, 0, 0, 1]]
    )


def test_rigid_matrix_is_centered_and_validates_inputs() -> None:
    center = np.array([3.0, 4.0, 5.0])
    parameters = np.array([0.0, 0.0, np.pi / 2, 1.0, 2.0, 3.0])
    transform = rigid_world_matrix(parameters, center)
    assert np.allclose(nib.affines.apply_affine(transform, center), center + [1, 2, 3])
    with pytest.raises(ValueError, match="six-element"):
        rigid_world_matrix(np.zeros(5), center)
    with pytest.raises(ValueError, match="rotation center"):
        rigid_world_matrix(np.zeros(6), np.array([0.0, np.nan, 0.0]))
    gimbal = rigid_world_matrix(np.array([0, np.pi / 2, 0, 0, 0, 0]), np.zeros(3))
    recovered = rigid_module._rigid_parameters(gimbal, np.zeros(3))
    assert np.allclose(rigid_world_matrix(recovered, np.zeros(3)), gimbal)


def test_resample_identity_and_validation() -> None:
    values = _phantom((9, 8, 7))
    affine = _affine()
    actual = resample_rigid(values, affine, values.shape, affine, np.eye(4))
    assert np.array_equal(actual, values)
    with pytest.raises(ValueError, match="three-dimensional"):
        resample_rigid(values[..., None], affine, values.shape, affine, np.eye(4))
    with pytest.raises(ValueError, match="positive dimensions"):
        resample_rigid(values, affine, (9, 0, 7), affine, np.eye(4))
    bad = np.eye(4)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 4x4"):
        resample_rigid(values, bad, values.shape, affine, np.eye(4))
    with pytest.raises(ValueError, match="Interpolation order"):
        resample_rigid(values, affine, values.shape, affine, np.eye(4), order=2)
    with pytest.raises(ValueError, match="Interpolation mode"):
        resample_rigid(values, affine, values.shape, affine, np.eye(4), mode="mirror")


def test_normalized_correlation_cost_contract() -> None:
    values = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    assert normalized_correlation_cost(values, values) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="identical shapes"):
        normalized_correlation_cost(values, values[:2])
    assert normalized_correlation_cost(np.ones((1, 1, 2)), np.ones((1, 1, 2))) == 1.0
    assert normalized_correlation_cost(np.ones((3, 3, 3)), np.ones((3, 3, 3))) == 1.0
    assert normalized_correlation_cost(np.ones((4, 4, 4)), np.ones((4, 4, 4))) == 1.0


def test_internal_fsl_cost_and_search_degenerate_paths(monkeypatch) -> None:
    zeros = np.zeros((3, 3, 3), dtype=np.float32)
    assert np.array_equal(
        rigid_module._intensity_center_scaled_mm(zeros, 2.0), np.zeros(3)
    )
    objective = rigid_module._SmoothedNormCorr(zeros, zeros, 2.0, np.zeros(3))
    assert objective(np.array([0, 0, 0, 100, 0, 0], dtype=float)) == 1.0
    assert objective(np.zeros(6)) == 1.0
    constant_objective = rigid_module._SmoothedNormCorr(
        np.zeros((4, 4, 4), dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        2.0,
        np.zeros(3),
    )
    assert constant_objective(np.zeros(6)) == 1.0
    assert rigid_module._quadratic_minimum(0, 1, 2, 0, 1, 2) is None
    assert rigid_module._inside_candidate(0, 1, 2, 0, 1, 2) != 1

    with pytest.raises(RuntimeError, match="failed to bracket"):
        rigid_module._line_search(
            np.zeros(6),
            0,
            np.ones(6) * 0.01,
            lambda point: -float(point[0]),
            0.0,
        )

    monkeypatch.setattr(rigid_module, "_quadratic_minimum", lambda *args: 0.05)
    point, cost = rigid_module._line_search(
        np.zeros(6),
        0,
        np.ones(6) * 0.01,
        lambda candidate: {
            0.0: 1.0,
            0.1: 2.0,
            -0.16: 0.5,
            0.05: 0.2,
        }.get(round(float(candidate[0]), 2), 0.3),
        0.0,
    )
    assert np.isfinite(point[0]) and np.isfinite(cost)

    point, cost = rigid_module._line_search(
        np.zeros(6),
        0,
        np.ones(6) * 0.01,
        lambda candidate: {
            0.0: 1.0,
            0.1: 2.0,
            -0.16: 0.5,
            0.05: 1.5,
        }.get(round(float(candidate[0]), 2), 0.3),
        0.0,
    )
    assert np.isfinite(point[0]) and np.isfinite(cost)


def test_fused_smoothed_sampler_is_bitwise_equal_to_scipy_reference() -> None:
    rng = np.random.default_rng(12)
    moving = rng.normal(size=(7, 8, 9)).astype(np.float32)
    reference_shape = (6, 7, 8)
    spacing = 2.0
    smooth_voxels = 0.5
    upper = np.asarray(moving.shape, dtype=np.float64) - 1.0001
    parameters = np.array([0.02, -0.03, 0.01, 0.4, -0.2, 0.3])
    inverse = np.linalg.inv(
        rigid_world_matrix(parameters, np.array([5.0, 6.0, 7.0]))
    )
    actual_weights = np.zeros(reference_shape, dtype=np.float32)
    actual_samples = np.zeros(reference_shape, dtype=np.float32)
    actual_count = rigid_module._sample_smoothed_linear(
        moving,
        inverse,
        spacing,
        smooth_voxels,
        upper,
        actual_weights,
        actual_samples,
    )

    grid = np.indices(reference_shape, dtype=np.float64).reshape(3, -1)
    coordinates = (
        inverse[:3, :3] @ (grid * spacing) + inverse[:3, 3, None]
    ) / spacing
    valid = np.all(
        (coordinates >= 0.0) & (coordinates <= upper[:, None]), axis=0
    )
    expected_samples = map_coordinates(
        moving,
        coordinates[:, valid],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    selected = coordinates[:, valid]
    expected_weights = np.ones(expected_samples.size, dtype=np.float32)
    for axis in range(3):
        low = selected[axis] < smooth_voxels
        expected_weights[low] *= selected[axis, low] / smooth_voxels
        distance_to_high = upper[axis] - selected[axis]
        high = distance_to_high < smooth_voxels
        expected_weights[high] *= distance_to_high[high] / smooth_voxels
    np.maximum(expected_weights, 0.0, out=expected_weights)

    assert actual_count == np.count_nonzero(valid)
    assert np.array_equal(actual_samples.reshape(-1)[valid], expected_samples)
    assert np.array_equal(actual_weights.reshape(-1)[valid], expected_weights)
    assert np.count_nonzero(actual_weights.reshape(-1)[~valid]) == 0

    weighted_fixed = np.empty(reference_shape, dtype=np.float32)
    weighted_moving = np.empty(reference_shape, dtype=np.float32)
    fixed_square = np.empty(reference_shape, dtype=np.float32)
    moving_square = np.empty(reference_shape, dtype=np.float32)
    cross = np.empty(reference_shape, dtype=np.float32)
    reference = rng.normal(size=reference_shape).astype(np.float32)
    rigid_module._build_normcorr_products(
        actual_weights,
        reference,
        actual_samples,
        weighted_fixed,
        weighted_moving,
        fixed_square,
        moving_square,
        cross,
    )
    expected_products = (
        np.multiply(actual_weights, reference),
        np.multiply(actual_weights, actual_samples),
    )
    assert np.array_equal(weighted_fixed, expected_products[0])
    assert np.array_equal(weighted_moving, expected_products[1])
    assert np.array_equal(fixed_square, np.multiply(weighted_fixed, reference))
    assert np.array_equal(moving_square, np.multiply(weighted_moving, actual_samples))
    assert np.array_equal(cross, np.multiply(weighted_fixed, actual_samples))

    for count in (5, 10, 72, 129, 384):
        values = rng.normal(size=count).astype(np.float32)
        assert rigid_module._numpy_pairwise_sum_float32(
            values, 0, count
        ) == np.sum(values, dtype=np.float32)

    arrays = [rng.normal(size=(13, 17, 72)).astype(np.float32) for _ in range(6)]
    actual_sums = rigid_module._fsl_six_sums(*arrays)
    for values, actual in zip(arrays, actual_sums, strict=True):
        expected = np.sum(values, axis=0, dtype=np.float32)
        expected = np.sum(expected, axis=0, dtype=np.float32)
        expected = np.sum(expected, dtype=np.float32)
        assert actual == expected

def test_estimate_rigid_transform_recovers_synthetic_motion() -> None:
    reference = _phantom()
    affine = _affine()
    center = nib.affines.apply_affine(affine, (np.asarray(reference.shape) - 1) / 2)
    truth_parameters = np.array([0.025, -0.02, 0.03, 1.5, -1.0, 0.8])
    truth = rigid_world_matrix(truth_parameters, center)
    moving = resample_rigid(
        reference, affine, reference.shape, affine, np.linalg.inv(truth)
    )
    result = estimate_rigid_transform(
        reference,
        moving,
        affine,
        stages_mm=(4.0, 2.0),
        max_evaluations=500,
    )
    assert result.success
    corners = np.array(
        [
            [0, 0, 0],
            [32, 0, 0],
            [0, 30, 0],
            [0, 0, 28],
            [32, 30, 28],
        ],
        dtype=float,
    )
    corners_world = nib.affines.apply_affine(affine, corners)
    displacement = np.linalg.norm(
        nib.affines.apply_affine(result.world_transform, corners_world)
        - nib.affines.apply_affine(truth, corners_world),
        axis=1,
    )
    assert np.max(displacement) < 1.0
    assert result.cost < 0.01
    assert result.evaluations > 0


def test_estimate_rigid_transform_validation_and_evaluation_limit() -> None:
    values = _phantom((9, 9, 9))
    affine = np.eye(4)
    cases = [
        (values[..., None], values, affine, {}, "matching three-dimensional"),
        (values, np.full_like(values, np.nan), affine, {}, "only finite"),
        (values, values, np.full((4, 4), np.nan), {}, "finite 4x4"),
        (values, values, affine, {"stages_mm": ()}, "positive spacings"),
        (values, values, affine, {"max_evaluations": 0}, "must be positive"),
        (
            values,
            values,
            affine,
            {"initial_parameters": np.zeros(5)},
            "finite six-element",
        ),
    ]
    for reference, moving, spatial, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            estimate_rigid_transform(reference, moving, spatial, **kwargs)

    result = estimate_rigid_transform(
        values, values, affine, stages_mm=(2.0,), max_evaluations=1
    )
    assert not result.success
    assert "Maximum" in result.message


def _write_dwi(tmp_path):
    reference = _phantom((17, 17, 17))
    affine = np.eye(4)
    center = (np.asarray(reference.shape) - 1) / 2
    shift = rigid_world_matrix(np.array([0, 0, 0, 1.0, 0, 0]), center)
    moved_left = resample_rigid(
        reference, affine, reference.shape, affine, np.linalg.inv(shift)
    )
    moved_right = resample_rigid(reference, affine, reference.shape, affine, shift)
    data = np.stack(
        [moved_left, reference * 0.5, reference, reference * 0.6, moved_right], axis=3
    )
    dwi = tmp_path / "dwi.nii.gz"
    bvals = tmp_path / "bvals"
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    nib.save(image, dwi)
    np.savetxt(bvals, [[5, 1000, 5, 1000, 5]])
    return dwi, bvals, reference, data


def test_write_aligned_b0_mean_outputs_qa_and_progress(tmp_path) -> None:
    dwi, bvals, reference, data = _write_dwi(tmp_path)
    output = tmp_path / "aligned.nii.gz"
    qa_file = tmp_path / "custom.json"
    progress = []
    result = write_aligned_b0_mean(
        dwi,
        bvals,
        output,
        stages_mm=(4.0, 2.0),
        max_evaluations=300,
        workers=2,
        progress=lambda done, total: progress.append((done, total)),
        qa_file=qa_file,
    )
    aligned = np.asarray(nib.load(result).dataobj)
    unaligned = np.mean(data[..., [0, 2, 4]], axis=3)
    assert np.linalg.norm(aligned - reference) < np.linalg.norm(unaligned - reference)
    assert progress == [
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
        (7, 7),
    ]
    image = nib.load(result)
    assert image.get_data_dtype() == np.dtype(np.float32)
    assert int(image.header["qform_code"]) == 1
    assert int(image.header["sform_code"]) == 2
    qa = json.loads(qa_file.read_text(encoding="utf-8"))
    assert qa["reference_volume_index"] == 2
    assert qa["registration"] == "mcflirt_6dof_compat46"
    assert qa["workers"] == 2
    assert qa["volumes"][1]["message"] == "reference volume"


def test_write_aligned_b0_mean_validation_and_failure(tmp_path) -> None:
    dwi, bvals, _, data = _write_dwi(tmp_path)
    output = tmp_path / "aligned.nii.gz"
    write_aligned_b0_mean(dwi, bvals, output, stages_mm=(2.0,), max_evaluations=200)
    assert (tmp_path / "aligned_qa.json").is_file()

    uncompressed = tmp_path / "dwi.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), uncompressed)
    uncompressed_output = tmp_path / "uncompressed_aligned.nii.gz"
    write_aligned_b0_mean(
        uncompressed,
        bvals,
        uncompressed_output,
        stages_mm=(2.0,),
        max_evaluations=200,
        workers=1,
    )
    assert np.array_equal(
        np.asarray(nib.load(output).dataobj),
        np.asarray(nib.load(uncompressed_output).dataobj),
    )

    three_d = tmp_path / "three.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)), three_d)
    with pytest.raises(ValueError, match="four-dimensional"):
        write_aligned_b0_mean(three_d, bvals, output)
    bad_bvals = tmp_path / "bad_bvals"
    np.savetxt(bad_bvals, [[0, 1000]])
    with pytest.raises(ValueError, match="fourth axis"):
        write_aligned_b0_mean(dwi, bad_bvals, output)
    with pytest.raises(ValueError, match="worker count"):
        write_aligned_b0_mean(dwi, bvals, output, workers=0)

    reference_nan = data.copy()
    reference_nan[0, 0, 0, 2] = np.nan
    nib.save(nib.Nifti1Image(reference_nan, np.eye(4)), dwi)
    with pytest.raises(ValueError, match="b0 volumes"):
        write_aligned_b0_mean(dwi, bvals, output)
    moving_nan = data.copy()
    moving_nan[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(moving_nan, np.eye(4)), dwi)
    with pytest.raises(ValueError, match="b0 volumes"):
        write_aligned_b0_mean(dwi, bvals, output)

    nib.save(nib.Nifti1Image(data, np.eye(4)), dwi)
    with pytest.raises(RuntimeError, match="evaluation limit"):
        write_aligned_b0_mean(dwi, bvals, output, stages_mm=(2.0,), max_evaluations=1)

    all_b0 = tmp_path / "all_b0"
    np.savetxt(all_b0, [np.zeros(5, dtype=int)], fmt="%d")
    write_aligned_b0_mean(
        dwi, all_b0, tmp_path / "all_aligned.nii.gz", stages_mm=(2.0,)
    )
