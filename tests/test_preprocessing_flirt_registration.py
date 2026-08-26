from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import dwi2cond_xp.preprocessing.flirt_registration as registration_module
from dwi2cond_xp.preprocessing import (
    FlirtOptimizationResult,
    register_flirt_affine,
    register_flirt_nosearch_mutual_information,
)


def _asymmetric_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (24, 23, 22)
    xvalue, yvalue, zvalue = np.indices(shape, dtype=np.float32)
    first = np.exp(
        -(
            (xvalue - 7.0) ** 2 / 15.0
            + (yvalue - 12.0) ** 2 / 35.0
            + (zvalue - 9.0) ** 2 / 20.0
        )
    )
    second = np.exp(
        -(
            (xvalue - 17.0) ** 2 / 10.0
            + (yvalue - 6.0) ** 2 / 12.0
            + (zvalue - 15.0) ** 2 / 9.0
        )
    )
    volume = np.asarray(
        first * np.float32(90.0)
        + second * np.float32(50.0),
        dtype=np.float32,
    )
    weight = np.zeros(shape, dtype=np.float32)
    weight[3:-3, 3:-3, 3:-3] = 1.0
    return volume, weight, np.diag([2.0, 2.0, 2.0, 1.0])


@pytest.mark.parametrize("degrees_of_freedom", [6, 12])
def test_default_schedule_recovers_synthetic_translation(degrees_of_freedom: int) -> None:
    reference, reference_weight, sampling = _asymmetric_volume()
    moving = np.zeros_like(reference)
    moving[:-1] = reference[1:]
    moving_weight = np.zeros_like(reference_weight)
    moving_weight[:-1] = reference_weight[1:]
    progress: list[tuple[str, int, int]] = []
    result = register_flirt_affine(
        reference,
        moving,
        reference_weight,
        moving_weight,
        sampling,
        sampling,
        degrees_of_freedom=degrees_of_freedom,
        workers=8,
        progress=lambda stage, current, total: progress.append((stage, current, total)),
    )
    expected = np.eye(4)
    expected[0, 3] = 2.0
    assert result.cost < 2e-4
    assert result.evaluations > 1_000
    assert result.candidate_count >= 4
    assert np.max(np.abs(result.matrix - expected)) < 0.11
    assert progress[-1] == ("complete", 1, 1)
    assert any(stage == "search" for stage, _, _ in progress)


def test_registration_scaling_and_matrix_helpers() -> None:
    assert registration_module._base_scale(np.eye(4), np.eye(4)) == 1.0
    assert registration_module._base_scale(
        np.diag([13.0, 13.0, 13.0, 1.0]), np.eye(4)
    ) == pytest.approx(13.0 / 8.0)
    assert registration_module._base_scale(
        np.diag([0.7, 0.7, 0.7, 1.0]), np.diag([13.0, 13.0, 13.0, 1.0])
    ) == pytest.approx(np.float32(0.7))

    grid = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    assert registration_module._interpolate_grid(grid, 0.5, 0.5, 0.5) == pytest.approx(6.5)
    assert registration_module._interpolate_grid(grid, 2.0, 2.0, 2.0) == 26.0
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [0.3, 1.4, 0.8], [1.2, 0.6, 1.9], [2.0, 2.0, 2.0]],
        dtype=np.float32,
    )
    scalar = np.asarray(
        [registration_module._interpolate_grid(grid, *point) for point in coordinates]
    )
    np.testing.assert_array_equal(
        registration_module._interpolate_grid_many(grid, coordinates), scalar
    )

    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 3.0
    assert registration_module._rms_deviation(first, first, 10.0) == 0.0
    assert registration_module._rms_deviation(second, first, 10.0) == 3.0


def test_distinct_candidate_replacement_and_ordering() -> None:
    identity = np.eye(4)
    nearby = np.eye(4)
    nearby[0, 3] = 0.1
    distant = np.eye(4)
    distant[0, 3] = 3.0
    first = registration_module._Candidate(0.5, identity)
    candidates = [(first, first)]
    registration_module._add_distinct(
        candidates,
        registration_module._Candidate(0.6, nearby),
        first,
        1.0,
    )
    assert candidates[0][0].cost == 0.5
    registration_module._add_distinct(
        candidates,
        registration_module._Candidate(0.4, nearby),
        first,
        1.0,
    )
    assert candidates[0][0].cost == 0.4
    registration_module._add_distinct(
        candidates,
        registration_module._Candidate(0.3, distant),
        first,
        1.0,
    )
    assert [item[0].cost for item in candidates] == [0.3, 0.4]


def test_search_threshold_and_defensive_minimum_fallback() -> None:
    constant = np.ones((11, 11, 11), dtype=np.float32)
    assert registration_module._fine_cost_threshold(constant) > np.float32(1.0)
    descending = -np.arange(11**3, dtype=np.float32).reshape(11, 11, 11)
    assert registration_module._fine_grid_minima(descending) == [1330]
    centered = np.ones((11, 11, 11), dtype=np.float32)
    centered[5, 5, 5] = 0.0
    assert 5 * 121 + 5 * 11 + 5 in registration_module._fine_grid_minima(centered)


def test_stage_and_parallel_helpers_compute_missing_centers(monkeypatch) -> None:
    class Evaluator:
        moving = np.ones((3, 3, 3), dtype=np.float32)
        moving_sampling = np.eye(4)

        def __call__(self, matrix: np.ndarray) -> float:
            return float(np.sum(np.square(matrix - np.eye(4))))

        def evaluate_ordered_parallel(self, matrix: np.ndarray) -> float:
            return self(matrix)

    def fake_optimize(parameters, tolerance, cost, **kwargs):
        return FlirtOptimizationResult(parameters, cost(parameters), 1, 1)

    evaluator = Evaluator()
    monkeypatch.setattr(registration_module, "flirt_brent_optimize", fake_optimize)
    result = registration_module._stage_optimize(
        evaluator,
        np.eye(4),
        np.eye(4),
        6,
        1.0,
        1,
        voxel_parallel=True,
    )
    np.testing.assert_array_equal(result.parameters, np.eye(4))
    candidates, evaluations = registration_module._parallel_optimize(
        evaluator,
        np.eye(4),
        [np.eye(4)],
        6,
        1.0,
        1,
        8,
    )
    assert candidates[0].cost == 0.0
    assert evaluations == 1


def test_batched_request_guard_rejects_missing_result() -> None:
    def missing_result():
        if False:
            yield np.zeros(1)
        return None

    with pytest.raises(RuntimeError, match="without a result"):
        registration_module._run_batched_requests(
            [missing_result()],
            lambda indices, values: np.zeros((len(indices), 4, 4)),
            SimpleNamespace(evaluate_many=lambda matrices, workers: np.zeros(len(matrices))),
            1,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("degrees_of_freedom", 5, "between six and twelve"),
        ("degrees_of_freedom", 13, "between six and twelve"),
        ("workers", 0, "positive integer"),
        ("workers", 1.5, "positive integer"),
        ("cost_function", "unsupported", "correlation_ratio or mutual_information"),
        (
            "search_cost_function",
            "unsupported",
            "correlation_ratio or mutual_information",
        ),
        ("initial_matrix", np.eye(3), "finite invertible 4x4"),
        ("initial_matrix", np.diag([0.0, 1.0, 1.0, 1.0]), "finite invertible 4x4"),
        ("qsform_matrix", np.full((4, 4), np.nan), "finite invertible 4x4"),
    ],
)
def test_registration_rejects_invalid_options(keyword: str, value: object, message: str) -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    arguments = {
        "reference": volume,
        "moving": volume,
        "reference_weight": np.ones_like(volume),
        "moving_weight": np.ones_like(volume),
        "reference_sampling": np.eye(4),
        "moving_sampling": np.eye(4),
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        register_flirt_affine(**arguments)


def test_nosearch_mutual_information_uses_full_default_schedule(monkeypatch) -> None:
    volume = np.ones((3, 3, 3), dtype=np.float32)
    captured = {}

    def register(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return registration_module.FlirtRegistrationResult(np.eye(4), 0.0, 1, 1)

    monkeypatch.setattr(registration_module, "register_flirt_affine", register)
    result = register_flirt_nosearch_mutual_information(
        volume,
        volume,
        np.eye(4),
        np.eye(4),
        degrees_of_freedom=12,
        initial_matrix=np.eye(4),
        workers=2,
    )
    assert captured["kwargs"]["degrees_of_freedom"] == 12
    assert captured["kwargs"]["workers"] == 2
    assert captured["kwargs"]["cost_function"] == "mutual_information"
    assert captured["kwargs"]["search_cost_function"] == "correlation_ratio"
    assert captured["kwargs"]["nosearch"] is True
    assert captured["kwargs"]["use_weights"] is False
    np.testing.assert_array_equal(captured["args"][2], np.ones_like(volume))
    np.testing.assert_array_equal(captured["args"][3], np.ones_like(volume))
    assert np.array_equal(result.matrix, np.eye(4))


@pytest.mark.parametrize("maximum_dof", [6, 7])
def test_nosearch_executes_reduced_translation_and_scale_schedule(
    monkeypatch: pytest.MonkeyPatch, maximum_dof: int
) -> None:
    volume = np.ones((3, 3, 3), dtype=np.float32)
    level = SimpleNamespace(
        moving=volume,
        reference=volume,
        moving_sampling=np.eye(4),
        reference_sampling=np.eye(4),
        requested_scale=8.0,
    )
    progress: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        registration_module,
        "_level_evaluator",
        lambda *_args, **_kwargs: lambda matrix: float(np.sum(matrix)),
    )
    monkeypatch.setattr(
        registration_module,
        "flirt_intensity_cog",
        lambda *_args: np.zeros(3, dtype=np.float64),
    )
    monkeypatch.setattr(
        registration_module,
        "flirt_brent_optimize",
        lambda parameters, *_args, **_kwargs: SimpleNamespace(
            parameters=np.ones_like(parameters), evaluations=2
        ),
    )

    def optimize(_evaluator, _initial, matrices, *_args, **_kwargs):
        return [registration_module._Candidate(0.0, matrices[0])], 3

    monkeypatch.setattr(registration_module, "_parallel_optimize", optimize)
    optimized, preoptimized, evaluations = registration_module._search(
        level,
        np.eye(4),
        maximum_dof,
        1,
        lambda *event: progress.append(event),
        "correlation_ratio",
        True,
    )

    assert len(optimized) == len(preoptimized) == 1
    assert evaluations == 6
    assert progress == [("search", 1, 1)]
    if maximum_dof == 7:
        np.testing.assert_allclose(np.diag(preoptimized[0].matrix)[:3], 2.0)
    else:
        np.testing.assert_allclose(np.diag(preoptimized[0].matrix)[:3], 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"degrees_of_freedom": 5}, "between six and twelve"),
        ({"workers": 0}, "positive integer"),
        ({"initial_matrix": np.eye(3)}, "finite invertible"),
        ({"initial_matrix": np.diag([0.0, 1.0, 1.0, 1.0])}, "finite invertible"),
    ],
)
def test_nosearch_mutual_information_validation(kwargs, message) -> None:
    values = np.ones((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        register_flirt_nosearch_mutual_information(
            values, values, np.eye(4), np.eye(4), **kwargs
        )
