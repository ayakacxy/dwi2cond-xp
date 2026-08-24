"""FSL MISCMATHS coordinate-wise Brent optimization."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
import math

import numpy as np

from .flirt_cost import FlirtWeightedMutualInformation
from .transforms import affine_matrix


@dataclass(frozen=True)
class FlirtOptimizationResult:
    """Final parameter vector, cost and one-dimensional iteration count."""

    parameters: np.ndarray
    cost: float
    iterations: int
    evaluations: int


CostRequest = Generator[np.ndarray, float, FlirtOptimizationResult]


def _quadratic_minimum(
    x1: np.float32,
    middle: np.float32,
    x2: np.float32,
    y1: np.float32,
    middle_value: np.float32,
    y2: np.float32,
) -> tuple[bool, np.float32]:
    coefficient = np.float32(
        np.float32(middle - x2) * np.float32(middle_value - y1)
        - np.float32(middle - x1) * np.float32(middle_value - y2)
    )
    linear = np.float32(
        -np.float32(np.float32(middle * middle) - np.float32(x2 * x2))
        * np.float32(middle_value - y1)
        + np.float32(np.float32(middle * middle) - np.float32(x1 * x1))
        * np.float32(middle_value - y2)
    )
    determinant = np.float32(
        np.float32(middle - x2)
        * np.float32(x2 - x1)
        * np.float32(x1 - middle)
    )
    if abs(determinant) > np.float32(1e-15) and np.float32(coefficient / determinant) < 0:
        return False, np.float32(0.0)
    if abs(coefficient) > np.float32(1e-15):
        return True, np.float32(-linear / np.float32(2.0 * coefficient))
    return False, np.float32(0.0)


def _golden_point(x1: np.float32, middle: np.float32, x2: np.float32) -> np.float32:
    ratio = np.float32(0.3819660)
    if abs(np.float32(x2 - middle)) > abs(np.float32(x1 - middle)):
        return np.float32(ratio * x2 + np.float32(1.0 - ratio) * middle)
    return np.float32(ratio * x1 + np.float32(1.0 - ratio) * middle)


def _next_point(
    x1: np.float32,
    middle: np.float32,
    x2: np.float32,
    y1: np.float32,
    middle_value: np.float32,
    y2: np.float32,
) -> np.float32:
    valid, candidate = _quadratic_minimum(x1, middle, x2, y1, middle_value, y2)
    if not valid or candidate < min(x1, x2) or candidate > max(x1, x2):
        return _golden_point(x1, middle, x2)
    return candidate


def _initial_bound(
    x1: np.float32,
    middle: np.float32,
    y1: np.float32,
    middle_value: np.float32,
    evaluate: Callable[[np.float32], np.float32],
) -> tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
    factor = np.float32(1.6)
    if y1 == 0:
        y1 = evaluate(x1)
    if middle_value == 0:
        middle_value = evaluate(middle)
    if y1 < middle_value:
        x1, middle = middle, x1
        y1, middle_value = middle_value, y1
    direction = np.float32(-1.0 if middle < x1 else 1.0)
    x2 = np.float32(middle + factor * np.float32(middle - x1))
    y2 = evaluate(x2)
    while middle_value > y2:
        maximum = np.float32(middle + np.float32(2.0 * factor) * np.float32(x2 - middle))
        valid, candidate = _quadratic_minimum(x1, middle, x2, y1, middle_value, y2)
        if (
            not valid
            or np.float32(candidate - x1) * direction < 0
            or np.float32(candidate - maximum) * direction > 0
        ):
            candidate = np.float32(middle + factor * np.float32(x2 - x1))
        candidate_value = evaluate(candidate)
        if np.float32(candidate - middle) * np.float32(candidate - x1) < 0:
            if candidate_value < middle_value:
                x2, y2 = middle, middle_value
                middle, middle_value = candidate, candidate_value
                break
            x1, y1 = candidate, candidate_value
        elif candidate_value > middle_value:
            x2, y2 = candidate, candidate_value
            break
        elif np.float32(candidate - x2) * direction < 0:
            x1, y1 = middle, middle_value
            middle, middle_value = candidate, candidate_value
        else:
            x1, y1 = middle, middle_value
            middle, middle_value = x2, y2
            x2, y2 = candidate, candidate_value
    return x1, middle, x2, y1, middle_value, y2


def _optimize_direction(
    point: np.ndarray,
    direction: np.ndarray,
    tolerance: np.ndarray,
    cost: Callable[[np.ndarray], np.float32],
    initial_value: np.float32,
    bound_guess: np.float32,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.float32, int]:
    unit_direction = direction / math.sqrt(float(np.sum(direction * direction)))
    direction_tolerance = np.float32(0.0)
    for index in range(tolerance.size):
        if abs(tolerance[index]) > 1e-15:
            direction_tolerance = np.float32(
                direction_tolerance + abs(unit_direction[index] / tolerance[index])
            )
    unit_tolerance = np.float32(abs(np.float32(1.0) / direction_tolerance))

    middle = np.float32(0.0)
    x1 = np.float32(bound_guess * unit_tolerance)

    def evaluate(distance: np.float32) -> np.float32:
        return cost(point + float(distance) * unit_direction)

    middle_value = initial_value
    if middle_value == 0:
        middle_value = evaluate(middle)
    y1 = evaluate(x1)
    x1, middle, x2, y1, middle_value, y2 = _initial_bound(
        x1, middle, y1, middle_value, evaluate
    )
    minimum_distance = np.float32(0.1 * unit_tolerance)
    iterations = 0
    while True:
        iterations += 1
        if iterations > max_iterations or abs(np.float32((x2 - x1) / unit_tolerance)) <= 1.0:
            break
        candidate = _next_point(x1, middle, x2, y1, middle_value, y2)
        direction_sign = np.float32(-1.0 if x2 < x1 else 1.0)
        if abs(np.float32(candidate - x1)) < minimum_distance:
            candidate = np.float32(x1 + direction_sign * minimum_distance)
        if abs(np.float32(candidate - x2)) < minimum_distance:
            candidate = np.float32(x2 - direction_sign * minimum_distance)
        if abs(np.float32(candidate - middle)) < minimum_distance:
            candidate = _golden_point(x1, middle, x2)
        if abs(np.float32(middle - x1)) < np.float32(0.4 * unit_tolerance):
            candidate = np.float32(middle + direction_sign * np.float32(0.5 * unit_tolerance))
        if abs(np.float32(middle - x2)) < np.float32(0.4 * unit_tolerance):
            candidate = np.float32(middle - direction_sign * np.float32(0.5 * unit_tolerance))
        candidate_value = evaluate(candidate)
        if np.float32(candidate - middle) * np.float32(x2 - middle) > 0:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        if candidate_value < middle_value:
            x2, y2 = middle, middle_value
            middle, middle_value = candidate, candidate_value
        else:
            x1, y1 = candidate, candidate_value
    return point + float(middle) * unit_direction, middle_value, iterations


def _request_cost(candidate: np.ndarray) -> Generator[np.ndarray, float, np.float32]:
    """Yield one candidate and receive its finite cost from the batch scheduler."""

    value = np.float32((yield candidate))
    if not np.isfinite(value):
        raise ValueError("cost must return finite values")
    return value


def _initial_bound_requests(
    point: np.ndarray,
    unit_direction: np.ndarray,
    x1: np.float32,
    middle: np.float32,
    y1: np.float32,
    middle_value: np.float32,
) -> Generator[
    np.ndarray,
    float,
    tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32],
]:
    """Run the same bracketing logic as ``_initial_bound`` via request/response."""

    factor = np.float32(1.6)
    if y1 == 0:
        y1 = yield from _request_cost(point + float(x1) * unit_direction)
    if middle_value == 0:
        middle_value = yield from _request_cost(point + float(middle) * unit_direction)
    if y1 < middle_value:
        x1, middle = middle, x1
        y1, middle_value = middle_value, y1
    direction = np.float32(-1.0 if middle < x1 else 1.0)
    x2 = np.float32(middle + factor * np.float32(middle - x1))
    y2 = yield from _request_cost(point + float(x2) * unit_direction)
    while middle_value > y2:
        maximum = np.float32(middle + np.float32(2.0 * factor) * np.float32(x2 - middle))
        valid, candidate = _quadratic_minimum(x1, middle, x2, y1, middle_value, y2)
        if (
            not valid
            or np.float32(candidate - x1) * direction < 0
            or np.float32(candidate - maximum) * direction > 0
        ):
            candidate = np.float32(middle + factor * np.float32(x2 - x1))
        candidate_value = yield from _request_cost(
            point + float(candidate) * unit_direction
        )
        if np.float32(candidate - middle) * np.float32(candidate - x1) < 0:
            if candidate_value < middle_value:
                x2, y2 = middle, middle_value
                middle, middle_value = candidate, candidate_value
                break
            x1, y1 = candidate, candidate_value
        elif candidate_value > middle_value:
            x2, y2 = candidate, candidate_value
            break
        elif np.float32(candidate - x2) * direction < 0:
            x1, y1 = middle, middle_value
            middle, middle_value = candidate, candidate_value
        else:
            x1, y1 = middle, middle_value
            middle, middle_value = x2, y2
            x2, y2 = candidate, candidate_value
    return x1, middle, x2, y1, middle_value, y2


def _optimize_direction_requests(
    point: np.ndarray,
    direction: np.ndarray,
    tolerance: np.ndarray,
    initial_value: np.float32,
    bound_guess: np.float32,
    max_iterations: int = 100,
) -> Generator[np.ndarray, float, tuple[np.ndarray, np.float32, int]]:
    """Run one FSL Brent coordinate direction in a batch-schedulable form."""

    unit_direction = direction / math.sqrt(float(np.sum(direction * direction)))
    direction_tolerance = np.float32(0.0)
    for index in range(tolerance.size):
        if abs(tolerance[index]) > 1e-15:
            direction_tolerance = np.float32(
                direction_tolerance + abs(unit_direction[index] / tolerance[index])
            )
    unit_tolerance = np.float32(abs(np.float32(1.0) / direction_tolerance))
    middle = np.float32(0.0)
    x1 = np.float32(bound_guess * unit_tolerance)
    middle_value = initial_value
    if middle_value == 0:
        middle_value = yield from _request_cost(point + float(middle) * unit_direction)
    y1 = yield from _request_cost(point + float(x1) * unit_direction)
    x1, middle, x2, y1, middle_value, y2 = yield from _initial_bound_requests(
        point, unit_direction, x1, middle, y1, middle_value
    )
    minimum_distance = np.float32(0.1 * unit_tolerance)
    iterations = 0
    while True:
        iterations += 1
        if iterations > max_iterations or abs(np.float32((x2 - x1) / unit_tolerance)) <= 1.0:
            break
        candidate = _next_point(x1, middle, x2, y1, middle_value, y2)
        direction_sign = np.float32(-1.0 if x2 < x1 else 1.0)
        if abs(np.float32(candidate - x1)) < minimum_distance:
            candidate = np.float32(x1 + direction_sign * minimum_distance)
        if abs(np.float32(candidate - x2)) < minimum_distance:
            candidate = np.float32(x2 - direction_sign * minimum_distance)
        if abs(np.float32(candidate - middle)) < minimum_distance:
            candidate = _golden_point(x1, middle, x2)
        if abs(np.float32(middle - x1)) < np.float32(0.4 * unit_tolerance):
            candidate = np.float32(middle + direction_sign * np.float32(0.5 * unit_tolerance))
        if abs(np.float32(middle - x2)) < np.float32(0.4 * unit_tolerance):
            candidate = np.float32(middle - direction_sign * np.float32(0.5 * unit_tolerance))
        candidate_value = yield from _request_cost(
            point + float(candidate) * unit_direction
        )
        if np.float32(candidate - middle) * np.float32(x2 - middle) > 0:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        if candidate_value < middle_value:
            x2, y2 = middle, middle_value
            middle, middle_value = candidate, candidate_value
        else:
            x1, y1 = candidate, candidate_value
    return point + float(middle) * unit_direction, middle_value, iterations


def flirt_brent_requests(
    parameters: np.ndarray,
    tolerance: np.ndarray,
    *,
    parameter_count: int | None = None,
    major_iterations: int = 4,
    bound_guesses: Sequence[float] = (10.0, 1.0),
) -> CostRequest:
    """Generate cost requests in exactly the same order as ``flirt_brent_optimize``."""

    point = np.asarray(parameters, dtype=np.float64)
    tolerances = np.asarray(tolerance, dtype=np.float64)
    if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("parameters must be a non-empty finite vector")
    if tolerances.shape != point.shape or not np.all(np.isfinite(tolerances)):
        raise ValueError("tolerance must be a finite vector matching parameters")
    count = point.size if parameter_count is None else parameter_count
    if not isinstance(count, (int, np.integer)) or not 1 <= count <= point.size:
        raise ValueError("parameter_count must be between one and the vector length")
    if not isinstance(major_iterations, (int, np.integer)) or major_iterations < 1:
        raise ValueError("major_iterations must be a positive integer")
    guesses = np.asarray(bound_guesses, dtype=np.float32)
    if guesses.ndim != 1 or guesses.size == 0 or not np.all(np.isfinite(guesses)):
        raise ValueError("bound_guesses must be a non-empty finite vector")
    inverse_tolerance = np.zeros_like(tolerances)
    valid = np.abs(tolerances) > 1e-15
    inverse_tolerance[valid] = np.abs(1.0 / tolerances[valid])
    inverse_tolerance /= tolerances.size
    directions = np.eye(point.size, dtype=np.float64)
    total_iterations = 0
    evaluation_count = 0
    value = np.float32(0.0)
    for major in range(int(major_iterations)):
        initial = point.copy()
        bound = guesses[min(major, guesses.size - 1)]
        for index in range(int(count)):
            request = _optimize_direction_requests(
                point,
                directions[:, index],
                tolerances,
                value,
                bound,
            )
            try:
                candidate = next(request)
                while True:
                    evaluation_count += 1
                    candidate = request.send((yield candidate))
            except StopIteration as completed:
                point, value, iterations = completed.value
            total_iterations += iterations
        average_tolerance = float(np.sum(np.abs((initial - point) * inverse_tolerance)))
        if average_tolerance < 1.0:
            break
    return FlirtOptimizationResult(point, float(value), total_iterations, evaluation_count)


def flirt_brent_optimize(
    parameters: np.ndarray,
    tolerance: np.ndarray,
    cost: Callable[[np.ndarray], float],
    *,
    parameter_count: int | None = None,
    major_iterations: int = 4,
    bound_guesses: Sequence[float] = (10.0, 1.0),
) -> FlirtOptimizationResult:
    """Run FSL's default coordinate-wise ``MISCMATHS::optimise`` strategy."""

    point = np.asarray(parameters, dtype=np.float64)
    tolerances = np.asarray(tolerance, dtype=np.float64)
    if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("parameters must be a non-empty finite vector")
    if tolerances.shape != point.shape or not np.all(np.isfinite(tolerances)):
        raise ValueError("tolerance must be a finite vector matching parameters")
    count = point.size if parameter_count is None else parameter_count
    if not isinstance(count, (int, np.integer)) or not 1 <= count <= point.size:
        raise ValueError("parameter_count must be between one and the vector length")
    if not isinstance(major_iterations, (int, np.integer)) or major_iterations < 1:
        raise ValueError("major_iterations must be a positive integer")
    guesses = np.asarray(bound_guesses, dtype=np.float32)
    if guesses.ndim != 1 or guesses.size == 0 or not np.all(np.isfinite(guesses)):
        raise ValueError("bound_guesses must be a non-empty finite vector")
    evaluation_count = 0

    def measured_cost(candidate: np.ndarray) -> np.float32:
        nonlocal evaluation_count
        value = np.float32(cost(candidate))
        evaluation_count += 1
        if not np.isfinite(value):
            raise ValueError("cost must return finite values")
        return value

    inverse_tolerance = np.zeros_like(tolerances)
    valid = np.abs(tolerances) > 1e-15
    inverse_tolerance[valid] = np.abs(1.0 / tolerances[valid])
    inverse_tolerance /= tolerances.size
    directions = np.eye(point.size, dtype=np.float64)
    total_iterations = 0
    value = np.float32(0.0)
    for major in range(int(major_iterations)):
        initial = point.copy()
        bound = guesses[min(major, guesses.size - 1)]
        for index in range(int(count)):
            point, value, iterations = _optimize_direction(
                point,
                directions[:, index],
                tolerances,
                measured_cost,
                value,
                bound,
            )
            total_iterations += iterations
        average_tolerance = float(np.sum(np.abs((initial - point) * inverse_tolerance)))
        if average_tolerance < 1.0:
            break
    return FlirtOptimizationResult(point, float(value), total_iterations, evaluation_count)


def optimize_flirt_stage(
    evaluator: FlirtWeightedMutualInformation,
    parameters: np.ndarray,
    center: np.ndarray,
    *,
    degrees_of_freedom: int,
    requested_scale: float,
    major_iterations: int = 4,
) -> FlirtOptimizationResult:
    """Optimize one FLIRT schedule stage with FSL tolerances and parameter order."""

    values = np.asarray(parameters, dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain twelve finite values")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must contain three finite values")
    if degrees_of_freedom not in range(6, 13):
        raise ValueError("degrees_of_freedom must be between six and twelve")
    if not np.isfinite(requested_scale) or requested_scale <= 0:
        raise ValueError("requested_scale must be positive and finite")
    tolerances = np.array(
        [0.005, 0.005, 0.005, 0.2, 0.2, 0.2, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001],
        dtype=np.float64,
    )
    tolerances *= requested_scale

    def parameter_cost(candidate: np.ndarray) -> float:
        transform = affine_matrix(candidate[:degrees_of_freedom], origin)
        return evaluator(transform)

    return flirt_brent_optimize(
        values,
        tolerances,
        parameter_cost,
        parameter_count=degrees_of_freedom,
        major_iterations=major_iterations,
    )
