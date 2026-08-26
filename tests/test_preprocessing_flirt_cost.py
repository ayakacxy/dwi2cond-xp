from __future__ import annotations

import numpy as np
import pytest

import dwi2cond_xp.preprocessing.flirt_cost as flirt_cost_module
from dwi2cond_xp.preprocessing import (
    FlirtWeightedCorrelationRatio,
    FlirtWeightedMutualInformation,
    flirt_intensity_cog,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(44)
    shape = (11, 10, 9)
    reference = random.normal(30.0, 8.0, shape).astype(np.float32)
    moving = (
        np.roll(reference, 1, axis=0) + random.normal(0.0, 0.7, shape)
    ).astype(np.float32)
    reference_weight = random.uniform(0.2, 1.0, shape).astype(np.float32)
    moving_weight = random.uniform(0.1, 1.0, shape).astype(np.float32)
    return reference, moving, reference_weight, moving_weight


def _evaluator() -> FlirtWeightedMutualInformation:
    reference, moving, reference_weight, moving_weight = _fixture()
    return FlirtWeightedMutualInformation(
        reference,
        moving,
        reference_weight,
        moving_weight,
        np.eye(4),
        np.eye(4),
        bins=16,
    )


def _correlation_evaluator() -> FlirtWeightedCorrelationRatio:
    reference, moving, reference_weight, moving_weight = _fixture()
    return FlirtWeightedCorrelationRatio(
        reference,
        moving,
        reference_weight,
        moving_weight,
        np.eye(4),
        np.eye(4),
        bins=16,
    )


def test_weighted_mi_matches_fsl_604_frozen_oracle() -> None:
    evaluator = _evaluator()
    cases = [
        ((0.0, 0.0, 0.0), -0.1337633133),
        ((0.2, -0.3, 0.4), -0.03535103798),
        ((-1.1, 0.35, 0.7), -0.08464455605),
        ((20.0, 0.0, 0.0), 0.0),
    ]
    for translation, expected in cases:
        transform = np.eye(4)
        transform[:3, 3] = translation
        assert evaluator(transform) == pytest.approx(expected, abs=1.5e-6)


def test_candidate_parallelism_preserves_values_and_order() -> None:
    evaluator = _evaluator()
    transforms = np.repeat(np.eye(4)[None], 7, axis=0)
    transforms[:, 0, 3] = [0.0, 0.2, -1.1, 0.5, -0.7, 1.2, 20.0]
    sequential = evaluator.evaluate_many(transforms, workers=1)
    parallel = evaluator.evaluate_many(transforms, workers=8)
    direct = np.asarray([evaluator(transform) for transform in transforms])
    assert np.array_equal(sequential, direct)
    assert np.array_equal(parallel, direct)
    assert evaluator.evaluate_many(transforms[:1], workers=8) == pytest.approx(direct[:1])


def test_weighted_correlation_ratio_preserves_fsl_order_in_parallel() -> None:
    evaluator = _correlation_evaluator()
    cases = [
        ((0.0, 0.0, 0.0), 1.0297900438308716),
        ((0.2, -0.3, 0.4), 1.0573623180389404),
        ((-1.1, 0.35, 0.7), 0.9750391840934753),
        ((20.0, 0.0, 0.0), 1.0),
    ]
    transforms = []
    for translation, expected in cases:
        transform = np.eye(4)
        transform[:3, 3] = translation
        transforms.append(transform)
        serial = evaluator(transform)
        assert serial == pytest.approx(expected, abs=1e-7)
        assert evaluator.evaluate_ordered_parallel(transform) == serial
        assert evaluator.evaluate_ordered_parallel(transform) == serial
    direct = np.asarray([evaluator(transform) for transform in transforms])
    assert np.array_equal(evaluator.evaluate_many(np.asarray(transforms), workers=8), direct)
    assert np.array_equal(evaluator.evaluate_many(np.asarray(transforms[:1])), direct[:1])


def test_correlation_workset_and_unit_weight_paths() -> None:
    values = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    weight = np.zeros_like(values)
    weight[[0, 2, 3], 1, 2] = [0.25, 0.5, 1.0]
    bins = np.arange(64, dtype=np.int32).reshape(4, 4, 4) % 4
    offsets, active_x, active_weights, active_bins = (
        flirt_cost_module._ordered_weight_workset(weight, bins)
    )
    row = 2 * 4 + 1
    assert offsets[row : row + 2].tolist() == [0, 3]
    assert active_x.tolist() == [0, 2, 3]
    assert active_weights.tolist() == pytest.approx([0.25, 0.5, 1.0])
    assert active_bins.tolist() == [bins[0, 1, 2], bins[2, 1, 2], bins[3, 1, 2]]

    evaluator = FlirtWeightedCorrelationRatio(
        values,
        values,
        np.ones_like(values),
        np.ones_like(values),
        np.eye(4),
        np.eye(4),
        bins=4,
    )
    assert evaluator._unit_moving_weight
    assert np.isfinite(evaluator(np.eye(4)))
    assert evaluator.evaluate_ordered_parallel(np.eye(4)) == evaluator(np.eye(4))


def test_python_kernels_cover_the_frozen_reduction_logic() -> None:
    mutual_information = _evaluator()
    correlation_ratio = _correlation_evaluator()
    identity = np.eye(4)
    far = np.eye(4)
    far[0, 3] = 20.0
    for transform in (identity, far):
        mapping = np.linalg.inv(transform)
        mutual_value = flirt_cost_module._weighted_mi_kernel.py_func(
            mutual_information.reference,
            mutual_information.moving,
            mutual_information.reference_weight,
            mutual_information.moving_weight,
            mutual_information._unit_moving_weight,
            mutual_information._reference_first_x,
            mutual_information._reference_last_x,
            mutual_information._reference_bins,
            mapping,
            mutual_information._moving_min,
            mutual_information._moving_max,
            mutual_information._moving_voxel_sizes,
            mutual_information.bins,
            np.float32(mutual_information.smooth_size),
            np.float32(mutual_information.fuzzy_fraction),
        )
        assert float(mutual_value) == pytest.approx(mutual_information(transform))

        correlation_value = flirt_cost_module._weighted_correlation_ratio_kernel.py_func(
            correlation_ratio.reference,
            correlation_ratio.moving,
            correlation_ratio.reference_weight,
            correlation_ratio.moving_weight,
            correlation_ratio._unit_moving_weight,
            correlation_ratio._reference_first_x,
            correlation_ratio._reference_last_x,
            correlation_ratio._reference_bins,
            mapping,
            correlation_ratio._moving_voxel_sizes,
            correlation_ratio.bins,
            np.float32(correlation_ratio.smooth_size),
        )
        assert float(correlation_value) == pytest.approx(correlation_ratio(transform))

        sampled_values = np.empty(correlation_ratio._active_x.size, dtype=np.float32)
        sampled_weights = np.empty(correlation_ratio._active_x.size, dtype=np.float32)
        parallel_value = (
            flirt_cost_module._weighted_correlation_ratio_ordered_parallel_kernel.py_func(
                correlation_ratio.reference,
                correlation_ratio.moving,
                correlation_ratio.moving_weight,
                correlation_ratio._unit_moving_weight,
                correlation_ratio._reference_first_x,
                correlation_ratio._reference_last_x,
                correlation_ratio._row_offsets,
                correlation_ratio._active_x,
                correlation_ratio._active_reference_weights,
                correlation_ratio._active_reference_bins,
                sampled_values,
                sampled_weights,
                mapping,
                correlation_ratio._moving_voxel_sizes,
                correlation_ratio.bins,
                np.float32(correlation_ratio.smooth_size),
            )
        )
        assert float(parallel_value) == pytest.approx(correlation_ratio(transform))


def test_python_kernels_cover_sparse_and_defensive_weight_paths() -> None:
    values = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    values[2, 2, 2] = np.float32(63.0)
    values[2, 1, 1] = np.float32(0.0)
    reference_weight = np.ones_like(values)
    reference_weight[:, 0, 0] = 0.0
    reference_weight[:2, 1:, :] = 0.0
    moving_weight = -np.ones_like(values)
    mutual_information = FlirtWeightedMutualInformation(
        values,
        values,
        reference_weight,
        moving_weight,
        np.eye(4),
        np.eye(4),
        bins=4,
        fuzzy_fraction=0.1,
    )
    correlation_ratio = FlirtWeightedCorrelationRatio(
        values,
        values,
        reference_weight,
        moving_weight,
        np.eye(4),
        np.eye(4),
        bins=4,
    )
    mapping = np.eye(4)
    assert np.isfinite(
        flirt_cost_module._weighted_mi_kernel.py_func(
            mutual_information.reference,
            mutual_information.moving,
            mutual_information.reference_weight,
            mutual_information.moving_weight,
            False,
            mutual_information._reference_first_x,
            mutual_information._reference_last_x,
            mutual_information._reference_bins,
            mapping,
            np.float32(16.0),
            np.float32(63.0),
            mutual_information._moving_voxel_sizes,
            mutual_information.bins,
            np.float32(1.0),
            np.float32(0.1),
        )
    )
    assert np.isfinite(
        flirt_cost_module._weighted_correlation_ratio_kernel.py_func(
            correlation_ratio.reference,
            correlation_ratio.moving,
            correlation_ratio.reference_weight,
            correlation_ratio.moving_weight,
            False,
            correlation_ratio._reference_first_x,
            correlation_ratio._reference_last_x,
            correlation_ratio._reference_bins,
            mapping,
            correlation_ratio._moving_voxel_sizes,
            correlation_ratio.bins,
            np.float32(1.0),
        )
    )
    for shift in (-3.0, 0.0, 2.0):
        shifted_mapping = np.eye(4)
        shifted_mapping[0, 3] = shift
        sampled_values = np.empty(correlation_ratio._active_x.size, dtype=np.float32)
        sampled_weights = np.empty(correlation_ratio._active_x.size, dtype=np.float32)
        assert np.isfinite(
            flirt_cost_module._weighted_correlation_ratio_ordered_parallel_kernel.py_func(
                correlation_ratio.reference,
                correlation_ratio.moving,
                correlation_ratio.moving_weight,
                False,
                correlation_ratio._reference_first_x,
                correlation_ratio._reference_last_x,
                correlation_ratio._row_offsets,
                correlation_ratio._active_x,
                correlation_ratio._active_reference_weights,
                correlation_ratio._active_reference_bins,
                sampled_values,
                sampled_weights,
                shifted_mapping,
                correlation_ratio._moving_voxel_sizes,
                correlation_ratio.bins,
                np.float32(1.0),
            )
        )


def test_constant_correlation_reference_uses_nonzero_bin_range() -> None:
    values = np.ones((3, 3, 3), dtype=np.float32)
    evaluator = FlirtWeightedCorrelationRatio(
        values,
        values,
        values,
        values,
        np.eye(4),
        np.eye(4),
        bins=4,
    )
    assert np.isfinite(evaluator(np.eye(4)))


def test_intensity_cog_matches_fsl_604_frozen_oracle() -> None:
    _, moving, _, _ = _fixture()
    center = flirt_intensity_cog(moving, np.eye(4))
    assert center == pytest.approx([5.00856231441, 4.52202296181, 3.9904571742], abs=5e-12)
    assert flirt_cost_module._voxel_intensity_cog.py_func(
        np.asfortranarray(moving)
    ) == pytest.approx(center, abs=5e-12)
    large = np.arange(11 * 10 * 10, dtype=np.float32).reshape(11, 10, 10)
    assert np.all(np.isfinite(flirt_cost_module._voxel_intensity_cog.py_func(large)))
    np.testing.assert_array_equal(
        flirt_cost_module._voxel_intensity_cog.py_func(np.ones((3, 3, 3), dtype=np.float32)),
        np.zeros(3),
    )


def test_reference_and_constant_intensity_bin_preparation() -> None:
    values = np.ones((3, 3, 3), dtype=np.float32)
    evaluator = FlirtWeightedMutualInformation(
        values,
        values,
        values,
        values,
        np.eye(4),
        np.eye(4),
        bins=4,
    )
    assert evaluator(np.eye(4)) == pytest.approx(0.0, abs=3e-7)


def test_interpolation_and_range_helpers_follow_fsl_edges() -> None:
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    interpolated = flirt_cost_module._trilinear.py_func(
        values, np.float32(0.25), np.float32(0.5), np.float32(0.75)
    )
    assert interpolated == pytest.approx(4.5)
    valid = flirt_cost_module._find_range_x.py_func(
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0),
        3,
        np.float32(2.9999),
        np.float32(2.9999),
        np.float32(2.9999),
    )
    assert valid == (0, 2)
    empty = flirt_cost_module._find_range_x.py_func(
        np.float32(0.0),
        np.float32(4.0),
        np.float32(0.0),
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0),
        3,
        np.float32(2.9999),
        np.float32(2.9999),
        np.float32(2.9999),
    )
    assert empty == (1, 0)
    rounded_endpoint = flirt_cost_module._find_range_x.py_func(
        np.float32(45.633274),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(2.0856845),
        np.float32(0.0),
        np.float32(0.0),
        10,
        np.float32(66.49012),
        np.float32(2.9999),
        np.float32(2.9999),
    )
    assert rounded_endpoint == (0, 10)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference", np.zeros((2, 2, 2, 1)), "three-dimensional"),
        ("moving", np.zeros((1, 2, 2)), "size at least two"),
        ("reference", np.full((2, 2, 2), np.nan), "finite"),
        ("reference_weight", np.zeros((3, 2, 2)), "must match"),
        ("moving_weight", np.zeros((3, 2, 2)), "must match"),
        ("bins", 1, "at least two"),
        ("bins", 2.5, "integer"),
        ("smooth_size", 0.0, "positive"),
        ("smooth_size", np.nan, "positive"),
        ("fuzzy_fraction", 0.0, r"in \(0, 0.5\]"),
        ("fuzzy_fraction", 0.6, r"in \(0, 0.5\]"),
        ("reference_sampling", np.eye(3), "finite 4x4"),
        ("moving_sampling", np.diag([0.0, 1.0, 1.0, 1.0]), "invertible"),
    ],
)
def test_evaluator_validation(field: str, value: object, message: str) -> None:
    reference, moving, reference_weight, moving_weight = _fixture()
    arguments = {
        "reference": reference,
        "moving": moving,
        "reference_weight": reference_weight,
        "moving_weight": moving_weight,
        "reference_sampling": np.eye(4),
        "moving_sampling": np.eye(4),
        "bins": 16,
        "smooth_size": 1.0,
        "fuzzy_fraction": 0.5,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        FlirtWeightedMutualInformation(**arguments)


def test_transform_batch_validation() -> None:
    evaluator = _evaluator()
    with pytest.raises(ValueError, match="shape"):
        evaluator.evaluate_many(np.eye(4))
    with pytest.raises(ValueError, match="positive integer"):
        evaluator.evaluate_many(np.eye(4)[None], workers=0)
    singular = np.eye(4)[None]
    singular[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        evaluator.evaluate_many(singular)
    nonfinite = np.eye(4)[None]
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate_many(nonfinite)
    nonfinite = np.eye(4)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 4x4"):
        evaluator(nonfinite)


def test_correlation_evaluator_validation() -> None:
    reference, moving, reference_weight, moving_weight = _fixture()
    arguments = {
        "reference": reference,
        "moving": moving,
        "reference_weight": reference_weight,
        "moving_weight": moving_weight,
        "reference_sampling": np.eye(4),
        "moving_sampling": np.eye(4),
        "bins": 16,
        "smooth_size": 1.0,
    }
    for field, value, message in [
        ("reference_weight", np.zeros((3, 2, 2)), "must match"),
        ("moving_weight", np.zeros((3, 2, 2)), "must match"),
        ("bins", 1, "at least two"),
        ("bins", 2.5, "integer"),
        ("smooth_size", 0.0, "positive"),
        ("reference_sampling", np.eye(3), "finite 4x4"),
        ("moving_sampling", np.diag([0.0, 1.0, 1.0, 1.0]), "invertible"),
    ]:
        invalid = dict(arguments)
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            FlirtWeightedCorrelationRatio(**invalid)

    evaluator = FlirtWeightedCorrelationRatio(**arguments)
    with pytest.raises(ValueError, match="shape"):
        evaluator.evaluate_many(np.eye(4))
    with pytest.raises(ValueError, match="positive integer"):
        evaluator.evaluate_many(np.eye(4)[None], workers=0)
    singular = np.eye(4)[None]
    singular[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        evaluator.evaluate_many(singular)
    nonfinite = np.eye(4)[None]
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate_many(nonfinite)
