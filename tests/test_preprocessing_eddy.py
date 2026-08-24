"""Tests for the source-faithful fixed EDDY primitives."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing import (
    EddySliceStatistics,
    EddySphericalGP,
    detect_eddy_slice_outliers,
    eddy_parameter_derivatives,
    eddy_slice_statistics,
    estimate_spherical_gp_hyperparameters,
    fit_spherical_gp_weights,
    invert_eddy_displacement,
    predict_spherical_gp,
    prepare_eddy_susceptibility_field,
    quadratic_eddy_field,
    rotate_bvecs_eddy,
    run_eddy_b0_iterations,
    run_eddy_dwi_iterations,
    run_eddy_nifti,
    select_fsl_gp_voxels,
    transform_eddy_model_to_scan,
    transform_eddy_scan_to_model,
)
from dwi2cond_xp.preprocessing.eddy import (
    _apply_eddy_dwi_location_reference,
    _eddy_model_to_scan_geometry,
    _glibc_rand_values,
)
from dwi2cond_xp.preprocessing.topup import _topup_movement_matrix
from dwi2cond_xp.preprocessing import eddy


def test_numba_executor_avoids_concurrent_workqueue_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[int] = []
    monkeypatch.setattr(eddy, "get_num_threads", lambda: 6)
    monkeypatch.setattr(eddy, "threading_layer", lambda: "workqueue")
    monkeypatch.setattr(eddy, "set_available_numba_threads", selected.append)
    monkeypatch.setattr(eddy, "set_num_threads", selected.append)

    executor, previous = eddy._create_numba_executor(8, 24)
    assert executor is None
    assert previous == 6
    assert selected == [8]

    eddy._shutdown_numba_executor(executor, previous)
    assert selected == [8, 6]


def test_numba_executor_uses_single_threaded_workers_when_backend_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {}
    selected: list[int] = []

    class FakeExecutor:
        def __init__(self, *, max_workers, initializer):
            state["max_workers"] = max_workers
            initializer()

        def shutdown(self) -> None:
            state["shutdown"] = True

    monkeypatch.setattr(eddy, "get_num_threads", lambda: 6)
    monkeypatch.setattr(eddy, "threading_layer", lambda: "omp")
    monkeypatch.setattr(eddy, "set_num_threads", selected.append)
    monkeypatch.setattr(eddy, "ThreadPoolExecutor", FakeExecutor)

    executor, previous = eddy._create_numba_executor(8, 3)
    assert executor is not None
    assert previous == 6
    assert state == {"max_workers": 3}
    assert selected == [1]

    eddy._shutdown_numba_executor(executor, previous)
    assert state["shutdown"] is True
    assert selected == [1]


def _fixture_directions(count: int = 24) -> np.ndarray:
    """Recreate the deterministic public EDDY fixture directions."""

    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    radius = np.sqrt(1.0 - z * z)
    return np.vstack((radius * np.cos(angle), radius * np.sin(angle), z))


def test_quadratic_eddy_field_uses_fsl_physical_coordinate_order() -> None:
    parameters = np.arange(1.0, 11.0)
    field = quadratic_eddy_field((3, 3, 3), (2.0, 3.0, 4.0), parameters)
    assert field[1, 1, 1] == 10.0
    x, y, z = -2.0, 3.0, 4.0
    expected = np.dot(
        parameters,
        np.array((x, y, z, x * x, y * y, z * z, x * y, x * z, y * z, 1.0)),
    )
    assert field[0, 2, 2] == expected


def test_rotate_bvecs_matches_fsl_inverse_movement_contract() -> None:
    bvecs = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    bvals = np.array([0.0, 1000.0])
    movements = np.zeros((2, 6), dtype=np.float64)
    movements[1, 5] = np.pi / 2.0
    rotated = rotate_bvecs_eddy(bvecs, bvals, movements, (5, 5, 5), (2.0, 2.0, 2.0))
    np.testing.assert_allclose(rotated[:, 0], 0.0)
    np.testing.assert_allclose(rotated[:, 1], (0.0, 1.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(np.linalg.norm(rotated[:, 1]), 1.0)


def test_dwi_location_reference_matches_relative_rigid_matrices() -> None:
    movements = np.asarray(
        (
            (0.1, -0.2, 0.3, 0.001, -0.002, 0.003),
            (-0.4, 0.5, -0.6, -0.004, 0.005, -0.006),
            (0.7, -0.8, 0.9, 0.007, -0.008, 0.009),
        ),
        dtype=np.float64,
    )
    shape = (26, 26, 18)
    voxel_sizes = (2.0, 2.0, 2.5)
    referenced = _apply_eddy_dwi_location_reference(
        movements, shape, voxel_sizes
    )
    np.testing.assert_allclose(referenced[0], 0.0, rtol=0.0, atol=1.0e-15)
    reference_inverse = np.linalg.inv(
        _topup_movement_matrix(movements[0], shape, voxel_sizes)
    )
    for index in range(movements.shape[0]):
        expected = (
            _topup_movement_matrix(movements[index], shape, voxel_sizes)
            @ reference_inverse
        )
        actual = _topup_movement_matrix(referenced[index], shape, voxel_sizes)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-7)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_inverse_displacement_matches_fsl_line_crossing(axis: int) -> None:
    forward = np.full((5, 6, 7), 0.2, dtype=np.float32)
    mask = np.ones(forward.shape, dtype=np.uint8)
    inverse, output_mask = invert_eddy_displacement(forward, axis, mask)
    np.testing.assert_allclose(inverse, -0.2, atol=3.0e-8, rtol=0.0)
    first = [slice(None)] * 3
    first[axis] = 0
    assert not np.any(output_mask[tuple(first)])
    remaining = [slice(None)] * 3
    remaining[axis] = slice(1, None)
    assert np.all(output_mask[tuple(remaining)])


def test_eddy_parameter_derivatives_have_fsl_volume_parameter_order() -> None:
    grid = np.indices((6, 6, 6), dtype=np.float32)
    prediction = np.asarray(10.0 + grid[0] + 2.0 * grid[1] + 3.0 * grid[2])
    result = eddy_parameter_derivatives(
        prediction,
        np.zeros(6),
        np.zeros(9),
        (2.0, 2.0, 2.0),
        1,
        1,
        0.05,
    )
    assert result.derivatives.shape == (6, 6, 6, 15)
    assert np.all(np.isfinite(result.derivatives))
    translation = result.derivatives[3, 3, 3, :3]
    assert np.all(translation < 0.0)
    assert translation[1] / translation[0] == pytest.approx(2.0, abs=0.01)
    assert translation[2] / translation[0] == pytest.approx(3.0, abs=0.01)


def test_b0_derivatives_skip_the_unused_eddy_columns() -> None:
    prediction = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    result = eddy_parameter_derivatives(
        prediction,
        np.zeros(6),
        np.zeros(9),
        (2.0, 2.0, 2.0),
        1,
        1,
        0.05,
        number_of_parameters=6,
    )
    assert result.derivatives.shape == (6, 6, 6, 6)


def test_scan_to_model_identity_preserves_values_and_jacobian() -> None:
    grid = np.indices((6, 7, 8), dtype=np.float32)
    observed = np.asarray(10.0 + grid[0] + 2.0 * grid[1] + 3.0 * grid[2])
    result = transform_eddy_scan_to_model(
        observed,
        np.zeros(6),
        np.zeros(9),
        (2.0, 2.0, 2.0),
        1,
        1,
        0.05,
    )
    np.testing.assert_allclose(result.values, observed, atol=2.0e-2, rtol=0.0)
    np.testing.assert_array_equal(result.jacobian, 1.0)
    np.testing.assert_array_equal(result.mask, 1)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_geometry_only_path_is_bitwise_equal_to_complete_transform(axis: int) -> None:
    prediction = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    movement = np.asarray((0.1, -0.2, 0.3, 0.001, -0.002, 0.003))
    eddy = np.asarray(
        (0.01, -0.02, 0.03, 0.0001, -0.0002, 0.0003, 0.0004, -0.0005, 0.0006)
    )
    complete = transform_eddy_model_to_scan(
        prediction, movement, eddy, (2.0, 2.5, 3.0), axis, -1, 0.05
    )
    coordinates, jacobian = _eddy_model_to_scan_geometry(
        prediction.shape, movement, eddy, (2.0, 2.5, 3.0), axis, -1, 0.05
    )
    np.testing.assert_array_equal(coordinates, complete.coordinates)
    np.testing.assert_array_equal(jacobian, complete.jacobian)


def test_no_topup_dwi_iteration_returns_complete_parameter_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directions = _fixture_directions(8)
    grid = np.indices((7, 7, 7), dtype=np.float32)
    scans = np.empty((7, 7, 7, 8), dtype=np.float32)
    for volume in range(8):
        scans[..., volume] = np.asarray(
            100.0
            + directions[0, volume] * grid[0]
            + directions[1, volume] * grid[1]
            + directions[2, volume] * grid[2],
            dtype=np.float32,
        )
    initial_movement = np.zeros((8, 6), dtype=np.float64)
    initial_movement[0, 0] = 0.01
    initial_eddy = np.zeros((8, 9), dtype=np.float64)
    initial_eddy[0, 0] = 0.001
    monkeypatch.setattr(
        eddy,
        "detect_eddy_slice_outliers",
        lambda statistics, **_kwargs: eddy.EddyOutlierResult(
            np.zeros(statistics.mean_difference.shape, dtype=bool),
            np.zeros(statistics.mean_difference.shape),
            np.zeros(statistics.mean_difference.shape),
        ),
    )
    result = run_eddy_dwi_iterations(
        scans,
        directions,
        np.ones(scans.shape[:3], dtype=np.uint8),
        (2.0, 2.0, 2.0),
        1,
        1,
        0.05,
        number_of_iterations=1,
        number_of_hyperparameter_voxels=40,
        random_seed=1,
        workers=1,
        replace_outliers=True,
        initial_movement_parameters=initial_movement,
        initial_quadratic_ec_parameters=initial_eddy,
    )
    assert result.movement_parameters.shape == (8, 6)
    assert result.quadratic_ec_parameters.shape == (8, 10)
    assert result.unwarped_scans.shape == scans.shape
    assert result.rotated_bvecs.shape == directions.shape
    assert len(result.iterations) == 1
    assert result.iterations[0].updates.shape == (8, 15)
    assert np.all(np.isfinite(result.unwarped_scans))
    np.testing.assert_array_equal(initial_movement[0], (0.01, 0, 0, 0, 0, 0))
    np.testing.assert_array_equal(initial_eddy[0], (0.001, 0, 0, 0, 0, 0, 0, 0, 0))

    repeated = run_eddy_dwi_iterations(
        scans,
        directions,
        np.ones(scans.shape[:3], dtype=np.uint8),
        (2.0, 2.0, 2.0),
        1,
        1,
        0.05,
        number_of_iterations=1,
        number_of_hyperparameter_voxels=40,
        random_seed=1,
        workers=1,
        initial_quadratic_ec_parameters=np.zeros((8, 10)),
    )
    assert repeated.quadratic_ec_parameters.shape == (8, 10)


def test_no_topup_b0_iteration_returns_movement_only_state() -> None:
    grid = np.indices((7, 7, 7), dtype=np.float32)
    first = np.asarray(100.0 + grid[0] + 2.0 * grid[1], dtype=np.float32)
    second = np.asarray(100.0 + grid[0] + 2.0 * grid[1] + 0.1, dtype=np.float32)
    scans = np.stack((first, second), axis=3)
    initial = np.zeros((2, 6), dtype=np.float64)
    result = run_eddy_b0_iterations(
        scans,
        np.ones(scans.shape[:3], dtype=np.uint8),
        (2.0, 2.0, 2.0),
        1,
        1,
        0.05,
        number_of_iterations=1,
        workers=1,
        initial_movement_parameters=initial,
    )
    assert result.movement_parameters.shape == (2, 6)
    assert result.unwarped_scans.shape == scans.shape
    assert len(result.iterations) == 1
    assert result.iterations[0].updates.shape == (2, 6)
    np.testing.assert_allclose(result.movement_parameters[0], 0.0, atol=1.0e-12)


def test_prepare_susceptibility_field_is_public_and_shape_checked() -> None:
    prepared = prepare_eddy_susceptibility_field(
        np.zeros((4, 5, 6), dtype=np.float32)
    )
    assert prepared.values_hz.shape == (4, 5, 6)
    assert prepared.interpolation_coefficients.shape == (4, 5, 6)


def test_eddy_nifti_writes_structured_fixed_subset_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dwi2cond_xp.preprocessing.eddy as eddy_module

    root = tmp_path
    scans = np.ones((4, 5, 6, 4), dtype=np.float32)
    affine = np.diag((2.0, 2.0, 2.5, 1.0))
    dwi_file = root / "dwi.nii.gz"
    mask_file = root / "mask.nii.gz"
    bvals_file = root / "bvals"
    bvecs_file = root / "bvecs"
    nib.save(nib.Nifti1Image(scans, affine), dwi_file)
    nib.save(
        nib.Nifti1Image(np.ones(scans.shape[:3], dtype=np.uint8), affine), mask_file
    )
    np.savetxt(bvals_file, np.asarray((0.0, 1000.0, 1000.0, 1000.0))[None, :])
    np.savetxt(
        bvecs_file,
        np.asarray(
            (
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )
    fake = SimpleNamespace(
        corrected_scans=scans + np.float32(1.0),
        movement_parameters=np.zeros((4, 6)),
        quadratic_ec_parameters=np.zeros((4, 10)),
        rotated_bvecs=np.loadtxt(bvecs_file),
        outlier_map=np.asarray(
            (
                (False, False, False, False, False, False),
                (False, False, True, False, False, False),
                (False, False, False, False, False, False),
                (False, False, False, False, False, False),
            )
        ),
        outlier_free_scans=scans.copy(),
        scale_factor=1.25,
        shell_pe_translation_mm=0.125,
        shell_alignment_parameters=np.arange(6, dtype=np.float64),
        dwi_registration=SimpleNamespace(iterations=()),
    )
    monkeypatch.setattr(
        eddy_module, "run_simnibs46_eddy", lambda *args, **kwargs: fake
    )
    output = root / "eddy"
    report = run_eddy_nifti(
        dwi_file,
        bvals_file,
        bvecs_file,
        mask_file,
        output,
        readout_seconds=0.05,
        phase_encoding_direction="y",
        workers=8,
    )
    assert report["status"] == "complete"
    assert report["outlier_slices"] == 1
    assert nib.load(output / "corrected_dwi.nii.gz").shape == scans.shape
    assert nib.load(output / "outlier_free_data.nii.gz").shape == scans.shape
    assert np.loadtxt(output / "eddy_parameters.txt").shape == (4, 16)
    assert np.loadtxt(output / "rotated_bvecs").shape == (3, 4)
    assert np.loadtxt(output / "outlier_map.txt", dtype=np.uint8).sum() == 1
    qa = json.loads((output / "eddy_qa.json").read_text(encoding="utf-8"))
    assert qa["workers"] == 8


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("initial_movement_parameters", np.zeros((2, 5)), "shape.*N, 6"),
        ("initial_quadratic_ec_parameters", np.zeros((2, 8)), "shape.*N, 9 or 10"),
    ),
)
def test_dwi_iteration_rejects_invalid_initial_state(
    keyword: str, value: np.ndarray, message: str
) -> None:
    arguments = {
        "fsl_scaled_scans": np.ones((2, 2, 2, 2), dtype=np.float32),
        "bvecs": np.eye(3, 2),
        "brain_mask": np.ones((2, 2, 2), dtype=np.uint8),
        "voxel_sizes_mm": (2.0, 2.0, 2.0),
        "phase_encoding_axis": 1,
        "phase_encoding_sign": 1,
        "readout_seconds": 0.05,
        "random_seed": 1,
        keyword: value,
    }
    with pytest.raises(ValueError, match=message):
        run_eddy_dwi_iterations(**arguments)


def test_slice_statistics_and_default_replacement_manager_find_dropout() -> None:
    predicted = np.full((20, 20, 6, 3), 100.0, dtype=np.float32)
    observed = predicted.copy()
    for volume in range(3):
        for slice_index in range(6):
            observed[:, :, slice_index, volume] += np.float32(
                0.01 * (volume * 6 + slice_index)
            )
    observed[:, :, 2, 1] = 0.0
    mask = np.ones((20, 20, 6), dtype=np.uint8)
    statistics = eddy_slice_statistics(observed, predicted, mask)
    assert statistics.voxel_count[1, 2] == 400
    assert statistics.mean_difference[1, 2] == -100.0
    assert statistics.mean_squared_difference[1, 2] == 10000.0
    result = detect_eddy_slice_outliers(statistics)
    assert np.argwhere(result.outlier_map).tolist() == [[1, 2]]
    assert result.n_standard_deviations[1, 2] < -4.0
    repeated = detect_eddy_slice_outliers(
        statistics, previous_outlier_map=result.outlier_map
    )
    assert np.argwhere(repeated.outlier_map).tolist() == [[1, 2]]


def test_leave_one_out_prediction_skips_the_target_validity_check() -> None:
    scans = np.asarray((1.0, 2.0, 3.0), dtype=np.float32).reshape(1, 1, 1, 3)
    weights = np.zeros((3, 3), dtype=np.float64)
    weights[1, 0] = 0.5
    weights[1, 2] = 0.5
    model = EddySphericalGP(np.zeros(3), np.eye(3), weights)
    included = predict_spherical_gp(scans, model)
    excluded = predict_spherical_gp(scans, model, exclude_target=True)
    assert included[0, 0, 0, 1] == 0.0
    assert excluded[0, 0, 0, 1] == 2.0


def test_new_spherical_gp_matches_fsl_debug_matrices() -> None:
    model = fit_spherical_gp_weights(
        _fixture_directions(), np.array((12.287603, 0.621164, -8.341233))
    )
    # FSL 6.0.4 --debug=3 values; tolerance covers its six-digit hyperparameter log.
    np.testing.assert_allclose(
        model.covariance[0, :5],
        (216989.1293, 93978.31092, 102485.3863, 109319.5093, 38099.67311),
        atol=0.11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        model.prediction_weights[0, :5],
        (
            0.999999996,
            4.77240469e-12,
            2.17299068e-10,
            -2.29931629e-11,
            1.34301792e-10,
        ),
        atol=1.2e-10,
        rtol=0.0,
    )


def test_spherical_gp_prediction_preserves_fsl_zero_validity_rule() -> None:
    model = fit_spherical_gp_weights(
        np.eye(3, dtype=np.float64)[:, :2], np.array((1.0, 0.0, -2.0))
    )
    scans = np.array([[[[2.0, 4.0]], [[2.0, 2.0]]]], dtype=np.float32)
    predicted = predict_spherical_gp(scans, model)
    assert predicted.shape == scans.shape
    assert np.all(np.isfinite(predicted))
    # Equal shell values become zero after mean correction, so FSL marks that voxel invalid.
    np.testing.assert_array_equal(predicted[0, 1, 0], 0.0)


def test_fsl_gp_voxel_selector_reproduces_glibc_rand() -> None:
    np.testing.assert_array_equal(
        _glibc_rand_values(1, 5),
        (1804289383, 846930886, 1681692777, 1714636915, 1957747793),
    )
    scans = np.arange(4 * 5 * 3 * 2, dtype=np.float32).reshape((4, 5, 3, 2))
    data, coordinates = select_fsl_gp_voxels(
        scans, np.ones((4, 5, 3), dtype=np.uint8), number_of_voxels=3, random_seed=1
    )
    np.testing.assert_array_equal(coordinates, ((3, 1, 0), (3, 3, 1), (2, 2, 0)))
    np.testing.assert_array_equal(
        data, scans[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], :]
    )


def test_spherical_gp_hyperparameter_estimator_runs_fsl_simplex() -> None:
    directions = _fixture_directions(8)
    rng = np.random.default_rng(14)
    data = rng.normal(size=(40, 8))
    result = estimate_spherical_gp_hyperparameters(
        data, directions, maximum_iterations=500
    )
    assert result.converged
    assert 0 < result.iterations < 500
    assert np.all(np.isfinite(result.hyperparameters))
    assert np.isfinite(result.cost)

    shrink_data = np.random.default_rng(2).normal(size=(12, 8))
    shrink = estimate_spherical_gp_hyperparameters(
        shrink_data, directions, maximum_iterations=40
    )
    assert np.all(np.isfinite(shrink.hyperparameters))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: quadratic_eddy_field((0, 2, 2), (1.0, 1.0, 1.0), np.zeros(9)), "shape"),
        (lambda: quadratic_eddy_field((2, 2, 2), (1.0, -1.0, 1.0), np.zeros(9)), "voxel"),
        (lambda: quadratic_eddy_field((2, 2, 2), (1.0, 1.0, 1.0), np.zeros(8)), "parameters"),
        (
            lambda: rotate_bvecs_eddy(
                np.zeros((2, 1)), np.zeros(1), np.zeros((1, 6)), (2, 2, 2), (1.0, 1.0, 1.0)
            ),
            "bvecs",
        ),
        (
            lambda: detect_eddy_slice_outliers(
                EddySliceStatistics(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
            ),
            "eligible",
        ),
    ],
)
def test_eddy_primitives_reject_invalid_contracts(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_eddy_remaining_gp_transform_and_outlier_validation_paths() -> None:
    with pytest.raises(ValueError, match="uint32"):
        eddy._glibc_rand_values(-1, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        eddy._glibc_rand_values(1, -1)
    scans = np.ones((2, 2, 2, 2), dtype=np.float32)
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="shapes"):
        select_fsl_gp_voxels(np.ones((2, 2, 2)), mask, number_of_voxels=1, random_seed=1)
    with pytest.raises(ValueError, match="fit within"):
        select_fsl_gp_voxels(scans, mask, number_of_voxels=9, random_seed=1)
    original_limit = eddy.GP_VOXEL_SELECTION_ATTEMPT_LIMIT
    eddy.GP_VOXEL_SELECTION_ATTEMPT_LIMIT = 1
    try:
        with pytest.raises(RuntimeError, match="unable to select"):
            select_fsl_gp_voxels(scans, mask, number_of_voxels=2, random_seed=1)
    finally:
        eddy.GP_VOXEL_SELECTION_ATTEMPT_LIMIT = original_limit
    with pytest.raises(ValueError, match="GP data"):
        eddy.spherical_gp_cv_cost(np.ones((2, 3)), np.ones((3, 2)), np.zeros(3))
    assert np.isfinite(
        eddy.spherical_gp_cv_cost(
            np.asarray([[1.0, 2.0], [2.5, 1.5]]), np.eye(3, 2), np.zeros(3)
        )
    )
    original_cholesky = eddy.np.linalg.cholesky
    eddy.np.linalg.cholesky = lambda _matrix: (_ for _ in ()).throw(
        np.linalg.LinAlgError("singular")
    )
    try:
        assert eddy.spherical_gp_cv_cost(
            np.asarray([[1.0, 2.0], [2.5, 1.5]]), np.eye(3, 2), np.zeros(3)
        ) == np.finfo(np.float64).max
    finally:
        eddy.np.linalg.cholesky = original_cholesky
    with pytest.raises(ValueError, match="selected shell data"):
        estimate_spherical_gp_hyperparameters(np.ones((2, 1)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="bvecs"):
        estimate_spherical_gp_hyperparameters(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="fudge factor"):
        estimate_spherical_gp_hyperparameters(np.asarray([[1.0, 2.0], [2.0, 1.0]]), np.eye(3, 2), error_variance_fudge_factor=0.5)
    with pytest.raises(ValueError, match="positive directional variance"):
        estimate_spherical_gp_hyperparameters(np.ones((2, 2)), np.eye(3, 2))
    with pytest.raises(ValueError, match="hyperparameters"):
        eddy.spherical_gp_covariance(np.eye(3, 2), np.ones(2))
    with pytest.raises(ValueError, match=r"shape \(3, N\)"):
        eddy.spherical_gp_covariance(np.ones((2, 2)), np.ones(3))
    bad_vectors = np.eye(3, 2)
    bad_vectors[:, 1] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        eddy.spherical_gp_covariance(bad_vectors, np.ones(3))
    with pytest.raises(ValueError, match="hyperparameters"):
        eddy._spherical_gp_covariance_from_angles_fsl_order(
            np.eye(2), np.ones(2)
        )

    model = EddySphericalGP(np.zeros(2), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="matching the GP"):
        predict_spherical_gp(np.ones((2, 2, 2, 3)), model)
    assert predict_spherical_gp(scans.astype(np.float64), model).dtype == np.float32
    invalid_scans = scans.copy()
    invalid_scans[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        predict_spherical_gp(invalid_scans, model)

    with pytest.raises(ValueError, match="3D grid"):
        invert_eddy_displacement(np.ones((2, 2)), 0, np.ones((2, 2)))
    with pytest.raises(ValueError, match="phase_encoding_axis"):
        invert_eddy_displacement(np.ones((2, 2, 2)), 3, mask)
    invalid_field = np.ones((2, 2, 2))
    invalid_field[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        invert_eddy_displacement(invalid_field, 0, mask)
    with pytest.raises(ValueError, match="finite 3D"):
        prepare_eddy_susceptibility_field(np.ones((2, 2)))

    base = EddySliceStatistics(np.asarray([[0.0, 1.0], [2.0, 3.0]]), np.asarray([[1.0, 2.0], [3.0, 5.0]]), np.full((2, 2), 300))
    with pytest.raises(ValueError, match="matching shape"):
        detect_eddy_slice_outliers(EddySliceStatistics(np.zeros((2, 2)), np.zeros((2, 1)), np.ones((2, 2))))
    with pytest.raises(ValueError, match="threshold"):
        detect_eddy_slice_outliers(base, threshold_standard_deviations=0)
    with pytest.raises(ValueError, match="previous_outlier_map"):
        detect_eddy_slice_outliers(base, previous_outlier_map=np.zeros((1, 1)))
    positive = detect_eddy_slice_outliers(base, threshold_standard_deviations=0.1, consider_positive=True, consider_squared=True)
    assert positive.outlier_map.shape == (2, 2)
    constant = EddySliceStatistics(
        np.ones((2, 2)), np.ones((2, 2)), np.full((2, 2), 300)
    )
    with pytest.raises(ValueError, match="variance must be nonzero"):
        detect_eddy_slice_outliers(constant)
    with pytest.raises(ValueError, match="matching shape"):
        eddy_slice_statistics(np.ones((2, 2, 2)), np.ones((2, 2, 2)), np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="mask must be"):
        eddy_slice_statistics(scans, scans, np.ones((2, 2)))
    invalid_observed = scans.copy()
    invalid_observed[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        eddy_slice_statistics(invalid_observed, scans, mask)


def test_eddy_reference_bvec_and_field_offset_contracts() -> None:
    movements = np.zeros((3, 6))
    ec = np.zeros((3, 10))
    bvals = np.full(3, 1000.0)
    bvecs = np.eye(3)
    common = ((3, 3, 3), (2.0, 2.0, 2.0), 1, 1, 0.05)
    invalid_calls = (
        lambda: eddy._separate_eddy_field_offset_from_movement(np.zeros((3, 5)), ec, bvals, bvecs, *common),
        lambda: eddy._separate_eddy_field_offset_from_movement(movements, np.zeros((3, 9)), bvals, bvecs, *common),
        lambda: eddy._separate_eddy_field_offset_from_movement(movements, ec, np.ones(2), bvecs, *common),
        lambda: eddy._separate_eddy_field_offset_from_movement(movements, ec, bvals, np.zeros((3, 3)), *common),
        lambda: _apply_eddy_dwi_location_reference(np.zeros((3, 5)), common[0], common[1]),
        lambda: _apply_eddy_dwi_location_reference(np.full((3, 6), np.nan), common[0], common[1]),
        lambda: _apply_eddy_dwi_location_reference(movements, common[0], common[1], reference=3),
        lambda: rotate_bvecs_eddy(bvecs, np.ones(2), movements, common[0], common[1]),
        lambda: rotate_bvecs_eddy(bvecs, bvals, np.zeros((2, 6)), common[0], common[1]),
        lambda: rotate_bvecs_eddy(np.full((3, 3), np.nan), bvals, movements, common[0], common[1]),
        lambda: rotate_bvecs_eddy(np.zeros((3, 3)), bvals, movements, common[0], common[1]),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()
    collinear = np.tile(np.asarray([[1.0], [0.0], [0.0]]), (1, 3))
    with pytest.raises(ValueError, match="cannot fit"):
        eddy._separate_eddy_field_offset_from_movement(
            movements, ec, bvals, collinear, *common
        )
    translated = eddy.apply_eddy_shell_pe_translation(
        movements, 0.1, common[0], common[1], 1
    )
    assert translated.shape == movements.shape


def test_eddy_nifti_rejects_all_header_and_table_errors(tmp_path: Path) -> None:
    affine = np.eye(4)
    scans = np.ones((2, 2, 2, 3), dtype=np.float32)
    dwi = tmp_path / "dwi.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    bvals = tmp_path / "bvals"
    bvecs = tmp_path / "bvecs"
    nib.save(nib.Nifti1Image(scans, affine), dwi)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), affine), mask)
    np.savetxt(bvals, np.asarray([[0, 1000, 1000]]))
    np.savetxt(bvecs, np.eye(3))

    common = (dwi, bvals, bvecs, mask, tmp_path / "out")
    for kwargs in (
        {"readout_seconds": 0.05, "phase_encoding_direction": "z"},
        {"readout_seconds": 0.0, "phase_encoding_direction": "y"},
        {"readout_seconds": 0.05, "phase_encoding_direction": "y", "workers": 0},
        {"readout_seconds": 0.05, "phase_encoding_direction": "y", "random_seed": -1},
    ):
        with pytest.raises(ValueError):
            run_eddy_nifti(*common, **kwargs)

    bad_dwi = tmp_path / "bad-dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), affine), bad_dwi)
    with pytest.raises(ValueError, match="four-dimensional"):
        run_eddy_nifti(bad_dwi, bvals, bvecs, mask, tmp_path / "a", readout_seconds=0.05, phase_encoding_direction="y")
    bad_mask = tmp_path / "bad-mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 2, 2)), affine), bad_mask)
    with pytest.raises(ValueError, match="spatial shape"):
        run_eddy_nifti(dwi, bvals, bvecs, bad_mask, tmp_path / "b", readout_seconds=0.05, phase_encoding_direction="y")
    shifted = affine.copy()
    shifted[0, 3] = 1.0
    shifted_mask = tmp_path / "shifted-mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), shifted), shifted_mask)
    with pytest.raises(ValueError, match="share one affine"):
        run_eddy_nifti(dwi, bvals, bvecs, shifted_mask, tmp_path / "c", readout_seconds=0.05, phase_encoding_direction="y")
    np.savetxt(tmp_path / "short-bvals", np.asarray([[0, 1000]]))
    with pytest.raises(ValueError, match="one value per"):
        run_eddy_nifti(dwi, tmp_path / "short-bvals", bvecs, mask, tmp_path / "d", readout_seconds=0.05, phase_encoding_direction="y")
    np.savetxt(tmp_path / "bad-bvecs", np.ones((2, 3)))
    with pytest.raises(ValueError, match=r"shape \(3, N\)"):
        run_eddy_nifti(dwi, bvals, tmp_path / "bad-bvecs", mask, tmp_path / "e", readout_seconds=0.05, phase_encoding_direction="y")
    bad_field = tmp_path / "bad-field.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 2, 2)), affine), bad_field)
    with pytest.raises(ValueError, match="field must match"):
        run_eddy_nifti(dwi, bvals, bvecs, mask, tmp_path / "f", readout_seconds=0.05, phase_encoding_direction="y", susceptibility_field_file=bad_field)
    shifted_field = tmp_path / "shifted-field.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), shifted), shifted_field)
    with pytest.raises(ValueError, match="field and DWI"):
        run_eddy_nifti(dwi, bvals, bvecs, mask, tmp_path / "g", readout_seconds=0.05, phase_encoding_direction="y", susceptibility_field_file=shifted_field)


def test_eddy_transform_derivative_and_shell_contracts() -> None:
    image = np.ones((3, 3, 3), dtype=np.float32)
    movement = np.zeros(6)
    ec = np.zeros(10)
    common = (movement, ec, (2.0, 2.0, 2.0), 1, 1, 0.05)
    invalid_model = (
        lambda: transform_eddy_model_to_scan(np.ones((2, 2)), *common),
        lambda: transform_eddy_model_to_scan(image, movement, ec, (2, 2, 2), 3, 1, 0.05),
        lambda: transform_eddy_model_to_scan(image, movement, ec, (2, 2, 2), 1, 1, 0.0),
        lambda: transform_eddy_scan_to_model(np.ones((2, 2)), *common),
        lambda: transform_eddy_scan_to_model(image, movement, ec, (2, 2, 2), 1, 0, 0.05),
        lambda: transform_eddy_scan_to_model(image, movement, ec, (2, 2, 2), 1, 1, -1.0),
    )
    for call in invalid_model:
        with pytest.raises(ValueError):
            call()
    prepared = prepare_eddy_susceptibility_field(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="scan grid"):
        transform_eddy_model_to_scan(image, *common, susceptibility_field_hz=prepared)

    derivative_errors = (
        lambda: eddy_parameter_derivatives(image, np.zeros(5), ec, (2, 2, 2), 1, 1, 0.05),
        lambda: eddy_parameter_derivatives(image, movement, ec, (2, 2, 2), 1, 1, 0.05, number_of_parameters=7),
        lambda: eddy_parameter_derivatives(image, movement, np.zeros(9), (2, 2, 2), 1, 1, 0.05, number_of_parameters=16),
    )
    for call in derivative_errors:
        with pytest.raises(ValueError):
            call()
    derivatives = eddy_parameter_derivatives(
        image, movement, ec, (2, 2, 2), 1, 1, 0.05, number_of_parameters=6
    )
    with pytest.raises(ValueError, match="observed scan"):
        eddy.eddy_gauss_newton_update(derivatives, np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="parameter_mask"):
        eddy.eddy_gauss_newton_update(derivatives, image, np.ones((2, 2, 2)))
    empty = eddy.EddyDerivativeResult(
        eddy.EddyTransformResult(
            derivatives.base.values,
            np.zeros_like(derivatives.base.mask),
            derivatives.base.jacobian,
            derivatives.base.coordinates,
            derivatives.base.spatial_gradient,
        ),
        derivatives.derivatives,
    )
    with pytest.raises(ValueError, match="no valid voxels"):
        eddy.eddy_gauss_newton_update(empty, image)

    grid = np.indices((3, 3, 3), dtype=np.float32)
    base = np.asarray(100.0 + grid[0] + 2.0 * grid[1] + 0.5 * grid[2], dtype=np.float32)
    b0 = np.stack((base, base + np.float32(0.1)), axis=3)
    dwi = np.stack((np.roll(base, 1, axis=1), np.roll(base, 1, axis=1) + np.float32(0.1)), axis=3)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    shell_errors = (
        lambda: eddy.estimate_eddy_shell_pe_translation(np.ones((3, 3, 3)), dwi, mask, mask, (2, 2, 2), 1),
        lambda: eddy.estimate_eddy_shell_pe_translation(b0, np.full_like(dwi, np.nan), mask, mask, (2, 2, 2), 1),
        lambda: eddy.estimate_eddy_shell_pe_translation(b0, dwi, np.ones((2, 2, 2)), mask, (2, 2, 2), 1),
        lambda: eddy.estimate_eddy_shell_pe_translation(b0, dwi, mask, mask, (2, 2, 2), 2),
        lambda: eddy.estimate_eddy_shell_pe_translation(b0, dwi, mask, mask, (2, 2, 2), 1, maximum_iterations=0),
        lambda: eddy.estimate_eddy_shell_rigid_alignment(np.ones((3, 3, 3)), dwi, mask, mask, (2, 2, 2)),
        lambda: eddy.estimate_eddy_shell_rigid_alignment(b0, np.full_like(dwi, np.nan), mask, mask, (2, 2, 2)),
        lambda: eddy.estimate_eddy_shell_rigid_alignment(b0, dwi, np.ones((2, 2, 2)), mask, (2, 2, 2)),
        lambda: eddy.estimate_eddy_shell_rigid_alignment(b0, dwi, mask, mask, (2, 2, 2), maximum_iterations=0),
        lambda: eddy.apply_eddy_shell_rigid_alignment(np.zeros((2, 5)), np.zeros(6), (3, 3, 3), (2, 2, 2)),
        lambda: eddy.apply_eddy_shell_rigid_alignment(np.zeros((2, 6)), np.zeros(5), (3, 3, 3), (2, 2, 2)),
        lambda: eddy.apply_eddy_shell_pe_translation(np.zeros((2, 5)), 0.0, (3, 3, 3), (2, 2, 2), 1),
        lambda: eddy.apply_eddy_shell_pe_translation(np.zeros((2, 6)), 0.0, (3, 3, 3), (2, 2, 2), 2),
    )
    for call in shell_errors:
        with pytest.raises(ValueError):
            call()
    assert np.isfinite(
        eddy.estimate_eddy_shell_pe_translation(
            b0, dwi, mask, mask, (2, 2, 2), 1,
            maximum_iterations=4,
        )
    )
    rigid = eddy.estimate_eddy_shell_rigid_alignment(
        b0, dwi, mask, mask, (2, 2, 2), maximum_iterations=2
    )
    assert rigid.shape == (6,)
    early = eddy.estimate_eddy_shell_rigid_alignment(
        b0,
        dwi,
        mask,
        mask,
        (2, 2, 2),
        maximum_iterations=2,
        fractional_cost_tolerance=1.0e20,
    )
    assert early.shape == (6,)


def test_shell_pe_simplex_expansion_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sample(_coefficients, mask, translation, _axis):
        return np.full(mask.shape, translation, dtype=np.float32), np.asarray(mask)

    def fake_mi(reference, moving, *_args):
        target = 5.0 if float(np.mean(reference)) < 1.5 else -5.0
        return -float((np.mean(moving) - target) ** 2)

    monkeypatch.setattr(eddy, "_sample_fsl_zeropad_cubic_translation", fake_sample)
    monkeypatch.setattr(eddy, "_soft_mutual_information_fsl_order", fake_mi)
    b0 = np.ones((2, 2, 2, 2), dtype=np.float32)
    dwi = np.full_like(b0, 2.0)
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    value = eddy.estimate_eddy_shell_pe_translation(
        b0, dwi, mask, mask, (1.0, 1.0, 1.0), 1, maximum_iterations=2
    )
    assert np.isfinite(value)


def test_shell_rigid_simplex_expansion_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled_costs = [5.0, *([10.0] * 6), 4.0, 3.0]
    calls = 0

    def fake_mi(*_args):
        nonlocal calls
        value = scheduled_costs[min(calls // 2, len(scheduled_costs) - 1)]
        calls += 1
        return -value

    monkeypatch.setattr(eddy, "_soft_mutual_information_fsl_order", fake_mi)
    b0 = np.ones((2, 2, 2, 2), dtype=np.float32)
    dwi = np.full_like(b0, 2.0)
    mask = np.ones((2, 2, 2), dtype=np.uint8)

    parameters = eddy.estimate_eddy_shell_rigid_alignment(
        b0, dwi, mask, mask, (1.0, 1.0, 1.0), maximum_iterations=1
    )
    assert parameters.shape == (6,)
    assert parameters[0] == -2.0


def test_eddy_iteration_and_complete_runner_validation_paths() -> None:
    scans = np.ones((3, 3, 3, 2), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    b0_common = (scans, mask, (2, 2, 2), 1, 1, 0.05)
    for kwargs in (
        {"fsl_scaled_scans": np.ones((3, 3, 3))},
        {"brain_mask": np.zeros((3, 3, 3))},
        {"number_of_iterations": 0},
        {"workers": 0},
        {"initial_movement_parameters": np.zeros((2, 5))},
    ):
        arguments = {
            "fsl_scaled_scans": b0_common[0],
            "brain_mask": b0_common[1],
            "voxel_sizes_mm": b0_common[2],
            "phase_encoding_axis": b0_common[3],
            "phase_encoding_sign": b0_common[4],
            "readout_seconds": b0_common[5],
            **kwargs,
        }
        with pytest.raises(ValueError):
            run_eddy_b0_iterations(**arguments)

    directions = np.eye(3, 2)
    dwi_arguments = {
        "fsl_scaled_scans": scans,
        "bvecs": directions,
        "brain_mask": mask,
        "voxel_sizes_mm": (2, 2, 2),
        "phase_encoding_axis": 1,
        "phase_encoding_sign": 1,
        "readout_seconds": 0.05,
        "random_seed": 1,
    }
    for changes in (
        {"fsl_scaled_scans": np.ones((3, 3, 3))},
        {"bvecs": np.ones((2, 2))},
        {"bvals": np.asarray([1000.0, -1.0])},
        {"brain_mask": np.zeros((3, 3, 3))},
        {"number_of_iterations": 0},
        {"workers": 0},
        {"susceptibility_field_hz": np.zeros((3, 3, 3))},
    ):
        with pytest.raises(ValueError):
            run_eddy_dwi_iterations(**{**dwi_arguments, **changes})

    complete_scans = np.ones((3, 3, 3, 3), dtype=np.float32)
    complete_bvals = np.asarray([0.0, 1000.0, 1000.0])
    complete_bvecs = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    complete = {
        "scans": complete_scans,
        "bvals": complete_bvals,
        "bvecs": complete_bvecs,
        "brain_mask": mask,
        "voxel_sizes_mm": (2, 2, 2),
        "phase_encoding_axis": 1,
        "phase_encoding_sign": 1,
        "readout_seconds": 0.05,
        "random_seed": 1,
    }
    invalid = (
        {"scans": np.ones((3, 3, 3, 2))},
        {"bvecs": np.ones((2, 3))},
        {"scans": np.full((3, 3, 3, 3), np.nan)},
        {"bvecs": np.full((3, 3), np.nan)},
        {"brain_mask": np.zeros((3, 3, 3))},
        {"workers": 0},
        {"bvals": np.asarray([1000.0, 1000.0, 1000.0])},
        {"bvals": np.asarray([0.0, 1000.0, 1200.0])},
        {"scans": np.zeros((3, 3, 3, 3))},
    )
    for changes in invalid:
        with pytest.raises(ValueError):
            eddy.run_simnibs46_eddy(**{**complete, **changes})


def test_complete_eddy_runner_covers_single_b0_serial_and_pe_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans = np.ones((3, 3, 3, 3), dtype=np.float32)
    bvals = np.asarray([0.0, 1000.0, 1000.0])
    bvecs = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    mask = np.ones((3, 3, 3), dtype=np.uint8)

    def identity_transform(observed, *_args, **_kwargs):
        values = np.asarray(observed, dtype=np.float32)
        return eddy.EddyTransformResult(
            values,
            np.ones(values.shape, dtype=np.uint8),
            np.ones(values.shape, dtype=np.float32),
            np.zeros(values.shape + (3,), dtype=np.float32),
            np.zeros(values.shape + (3,), dtype=np.float32),
        )

    def fake_dwi(values, vectors, *_args, **_kwargs):
        count = values.shape[3]
        return eddy.EddyDwiRegistrationResult(
            np.zeros((count, 6)),
            np.zeros((count, 10)),
            np.asarray(values),
            np.asarray(vectors),
            (),
            np.ones(values.shape[:3], dtype=np.uint8),
        )

    monkeypatch.setattr(eddy, "transform_eddy_scan_to_model", identity_transform)
    monkeypatch.setattr(eddy, "run_eddy_dwi_iterations", fake_dwi)
    monkeypatch.setattr(eddy, "estimate_eddy_shell_pe_translation", lambda *_args: 0.0)
    monkeypatch.setattr(eddy, "estimate_eddy_shell_rigid_alignment", lambda *_args: np.zeros(6))
    result = eddy.run_simnibs46_eddy(
        scans,
        bvals,
        bvecs,
        mask,
        (2.0, 2.0, 2.0),
        1,
        1,
        0.05,
        random_seed=1,
        workers=1,
        replace_outliers=False,
        align_shells_post_eddy=False,
    )
    assert result.corrected_scans.shape == scans.shape


def test_outlier_volume_selection_and_restore_are_explicit() -> None:
    assert eddy._eddy_flagged_volumes(np.zeros((2, 3), dtype=bool)) == ()
    outliers = np.zeros((2, 3), dtype=bool)
    outliers[1, 2] = True
    assert eddy._eddy_flagged_volumes(outliers) == (1,)

    scans = np.zeros((2, 2, 3, 1), dtype=np.float32)
    original = np.full_like(scans, 4.0)
    stored = np.asarray([[True, False, False]])
    current = np.asarray([False, True, False])
    predicted = np.full((2, 2, 3), 9.0, dtype=np.float32)
    replacement_mask = np.ones((2, 2, 3), dtype=np.uint8)
    eddy._apply_eddy_outlier_slices(
        scans, original, stored, current, predicted, replacement_mask, 0
    )
    assert np.all(scans[:, :, 0, 0] == 4.0)
    assert np.all(scans[:, :, 1, 0] == 9.0)
    assert np.array_equal(stored[0], current)
