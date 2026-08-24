from __future__ import annotations

import numpy as np
import pytest

import dwi2cond_xp.preprocessing.flirt_optimizer as optimizer_module
from dwi2cond_xp.preprocessing import (
    FlirtWeightedMutualInformation,
    flirt_brent_optimize,
    optimize_flirt_stage,
)


def _objective(parameters: np.ndarray) -> float:
    xvalue, yvalue, zvalue = parameters
    return (
        (xvalue - 1.25) ** 2
        + 2.0 * (yvalue + 0.75) ** 2
        + 0.5 * (zvalue - 2.0) ** 2
        + 0.1 * xvalue * yvalue
    )


def test_brent_optimizer_matches_fsl_604_oracle() -> None:
    result = flirt_brent_optimize(
        np.array([-2.0, 3.0, -1.0]),
        np.array([0.01, 0.02, 0.03]),
        _objective,
    )
    assert result.parameters == pytest.approx(
        [1.28887484968, -0.782221879344, 2.0], abs=5e-12
    )
    assert result.cost == pytest.approx(-0.0972308591008, abs=2e-13)
    assert result.iterations == 23
    assert result.evaluations == 52


def test_batched_request_trajectory_is_bitwise_equal_to_direct_optimizer() -> None:
    parameters = np.array([-2.0, 3.0, -1.0])
    tolerances = np.array([0.01, 0.02, 0.03])
    direct = flirt_brent_optimize(parameters, tolerances, _objective)
    requests = optimizer_module.flirt_brent_requests(
        parameters, tolerances
    )
    try:
        candidate = next(requests)
        while True:
            candidate = requests.send(_objective(candidate))
    except StopIteration as completed:
        batched = completed.value
    np.testing.assert_array_equal(batched.parameters, direct.parameters)
    assert batched.cost == direct.cost
    assert batched.iterations == direct.iterations
    assert batched.evaluations == direct.evaluations


@pytest.mark.parametrize(
    ("parameters", "tolerances", "kwargs", "message"),
    [
        (np.array([]), np.array([]), {}, "non-empty finite"),
        (np.array([np.nan]), np.ones(1), {}, "non-empty finite"),
        (np.zeros(1), np.ones(2), {}, "matching"),
        (np.zeros(1), np.ones(1), {"parameter_count": 0}, "parameter_count"),
        (np.zeros(1), np.ones(1), {"major_iterations": 0}, "major_iterations"),
        (np.zeros(1), np.ones(1), {"bound_guesses": ()}, "bound_guesses"),
        (
            np.zeros(1),
            np.ones(1),
            {"bound_guesses": (np.nan,)},
            "bound_guesses",
        ),
    ],
)
def test_batched_request_validation(parameters, tolerances, kwargs, message) -> None:
    requests = optimizer_module.flirt_brent_requests(
        parameters, tolerances, **kwargs
    )
    with pytest.raises(ValueError, match=message):
        next(requests)


def test_batched_request_rejects_nonfinite_cost() -> None:
    requests = optimizer_module.flirt_brent_requests(np.zeros(1), np.ones(1))
    next(requests)
    with pytest.raises(ValueError, match="cost must return finite"):
        requests.send(np.nan)


def test_parameter_subset_and_custom_bound_sequence() -> None:
    result = flirt_brent_optimize(
        np.array([-2.0, 3.0, -1.0]),
        np.array([0.01, 0.02, 0.03]),
        _objective,
        parameter_count=2,
        major_iterations=2,
        bound_guesses=(3.0,),
    )
    assert result.parameters[2] == -1.0
    assert result.cost < _objective(np.array([-2.0, 3.0, -1.0]))


def test_scalar_point_helpers_cover_quadratic_and_golden_paths() -> None:
    valid, point = optimizer_module._quadratic_minimum(
        *(np.float32(value) for value in (-1.0, 0.0, 2.0, 1.0, 0.0, 4.0))
    )
    assert valid and point == pytest.approx(0.0)
    maximum, _ = optimizer_module._quadratic_minimum(
        *(np.float32(value) for value in (-1.0, 0.0, 1.0, -1.0, 0.0, -1.0))
    )
    assert not maximum
    linear, _ = optimizer_module._quadratic_minimum(
        *(np.float32(value) for value in (-1.0, 0.0, 1.0, -1.0, 0.0, 1.0))
    )
    assert not linear
    assert optimizer_module._golden_point(
        np.float32(-3.0), np.float32(0.0), np.float32(1.0)
    ) < 0
    assert optimizer_module._golden_point(
        np.float32(-1.0), np.float32(0.0), np.float32(3.0)
    ) > 0
    golden = optimizer_module._next_point(
        *(np.float32(value) for value in (-1.0, 0.0, 1.0, -1.0, 0.0, -1.0))
    )
    assert -1.0 < golden < 1.0


def test_initial_bound_zero_sentinels_and_inner_candidate_paths(monkeypatch) -> None:
    evaluated: list[float] = []

    def quadratic(value: np.float32) -> np.float32:
        evaluated.append(float(value))
        return np.float32((float(value) - 2.0) ** 2 + 1.0)

    result = optimizer_module._initial_bound(
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        quadratic,
    )
    assert all(np.isfinite(result))
    assert len(evaluated) >= 3

    monkeypatch.setattr(
        optimizer_module,
        "_quadratic_minimum",
        lambda *arguments: (True, np.float32(0.5)),
    )

    def lower_inner(value: np.float32) -> np.float32:
        return np.float32(0.25 if value == np.float32(0.5) else 0.5)

    inner = optimizer_module._initial_bound(
        np.float32(0.0),
        np.float32(1.0),
        np.float32(2.0),
        np.float32(1.0),
        lower_inner,
    )
    assert inner[1] == np.float32(0.5)

    candidates = iter((np.float32(0.5), np.float32(2.0)))
    monkeypatch.setattr(
        optimizer_module,
        "_quadratic_minimum",
        lambda *arguments: (True, next(candidates)),
    )

    def higher_inner(value: np.float32) -> np.float32:
        if value == np.float32(2.6):
            return np.float32(0.5)
        return np.float32(1.5 if value == np.float32(0.5) else 2.0)

    bracket = optimizer_module._initial_bound(
        np.float32(0.0),
        np.float32(1.0),
        np.float32(2.0),
        np.float32(1.0),
        higher_inner,
    )
    assert all(np.isfinite(bracket))


def test_batched_initial_bound_requests_cover_zero_sentinels_and_lower_inner(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        optimizer_module,
        "_quadratic_minimum",
        lambda *arguments: (True, np.float32(0.5)),
    )
    request = optimizer_module._initial_bound_requests(
        np.zeros(1),
        np.ones(1),
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
    )
    np.testing.assert_array_equal(next(request), [1.0])
    np.testing.assert_array_equal(request.send(4.0), [0.0])
    np.testing.assert_array_equal(request.send(1.0), np.asarray([-1.6], dtype=np.float32))
    np.testing.assert_array_equal(request.send(0.5), [0.5])
    with pytest.raises(StopIteration) as completed:
        request.send(0.25)
    assert completed.value.value[1] == np.float32(0.5)

    higher = optimizer_module._initial_bound_requests(
        np.zeros(1),
        np.ones(1),
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
    )
    next(higher)
    higher.send(4.0)
    higher.send(1.0)
    higher.send(0.5)
    assert np.all(np.isfinite(higher.send(2.0)))
    higher.close()


def test_direction_optimizer_separates_candidate_from_first_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        optimizer_module, "_next_point", lambda x1, *arguments: np.float32(x1)
    )
    point, value, iterations = optimizer_module._optimize_direction(
        np.zeros(1),
        np.ones(1),
        np.asarray([0.01]),
        lambda candidate: np.float32((candidate[0] - 1.0) ** 2 + 1.0),
        np.float32(0.0),
        np.float32(1.0),
        max_iterations=1,
    )
    assert np.all(np.isfinite(point))
    assert np.isfinite(value)
    assert iterations >= 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((np.array([]), np.array([]), _objective), "non-empty finite"),
        ((np.array([np.nan]), np.array([1.0]), lambda value: 0.0), "non-empty finite"),
        ((np.array([1.0]), np.array([1.0, 2.0]), lambda value: 0.0), "matching"),
    ],
)
def test_optimizer_vector_validation(arguments: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        flirt_brent_optimize(*arguments)


@pytest.mark.parametrize("count", [0, 4, 1.5])
def test_optimizer_parameter_count_validation(count: object) -> None:
    with pytest.raises(ValueError, match="parameter_count"):
        flirt_brent_optimize(
            np.zeros(3), np.ones(3), lambda value: 1.0, parameter_count=count
        )


def test_optimizer_remaining_validation() -> None:
    with pytest.raises(ValueError, match="major_iterations"):
        flirt_brent_optimize(np.zeros(1), np.ones(1), lambda value: 1.0, major_iterations=0)
    with pytest.raises(ValueError, match="bound_guesses"):
        flirt_brent_optimize(np.zeros(1), np.ones(1), lambda value: 1.0, bound_guesses=())
    with pytest.raises(ValueError, match="bound_guesses"):
        flirt_brent_optimize(
            np.zeros(1), np.ones(1), lambda value: 1.0, bound_guesses=(np.nan,)
        )
    with pytest.raises(ValueError, match="cost must return finite"):
        flirt_brent_optimize(np.zeros(1), np.ones(1), lambda value: np.nan)


def test_flirt_stage_uses_source_tolerances_and_preserves_inactive_parameters() -> None:
    coordinates = np.indices((7, 7, 7), dtype=np.float32)
    values = np.exp(-np.sum((coordinates - 3.0) ** 2, axis=0) / 5.0).astype(np.float32)
    weights = np.ones_like(values)
    evaluator = FlirtWeightedMutualInformation(
        values, values, weights, weights, np.eye(4), np.eye(4), bins=8
    )
    parameters = np.array([0.0] * 6 + [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    result = optimize_flirt_stage(
        evaluator,
        parameters,
        np.array([3.0, 3.0, 3.0]),
        degrees_of_freedom=6,
        requested_scale=1.0,
        major_iterations=1,
    )
    assert np.all(np.isfinite(result.parameters))
    assert np.array_equal(result.parameters[6:], parameters[6:])
    assert result.evaluations > 0


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("parameters", np.zeros(6), "twelve"),
        ("center", np.zeros(2), "three"),
        ("degrees_of_freedom", 5, "six and twelve"),
        ("degrees_of_freedom", 13, "six and twelve"),
        ("requested_scale", 0.0, "positive"),
        ("requested_scale", np.nan, "positive"),
    ],
)
def test_flirt_stage_validation(keyword: str, value: object, message: str) -> None:
    array = np.ones((3, 3, 3), dtype=np.float32)
    evaluator = FlirtWeightedMutualInformation(
        array, array, array, array, np.eye(4), np.eye(4), bins=4
    )
    arguments = {
        "evaluator": evaluator,
        "parameters": np.array([0.0] * 6 + [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "center": np.ones(3),
        "degrees_of_freedom": 6,
        "requested_scale": 1.0,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        optimize_flirt_stage(**arguments)
