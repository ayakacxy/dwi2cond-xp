"""FSL FLIRT default-schedule affine registration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from .flirt_cost import (
    FlirtWeightedCorrelationRatio,
    FlirtWeightedMutualInformation,
    flirt_intensity_cog,
)
from .flirt_optimizer import (
    CostRequest,
    FlirtOptimizationResult,
    flirt_brent_optimize,
    flirt_brent_requests,
)
from .flirt_pyramid import FlirtPyramidLevel, build_flirt_pyramid
from .transforms import affine_matrices, affine_matrix, decompose_affine


@dataclass(frozen=True)
class FlirtRegistrationResult:
    """Final scaled-mm matrix and deterministic optimization diagnostics."""

    matrix: np.ndarray
    cost: float
    evaluations: int
    candidate_count: int


@dataclass(frozen=True)
class _Candidate:
    cost: float
    matrix: np.ndarray


def _base_scale(reference_sampling: np.ndarray, moving_sampling: np.ndarray) -> float:
    """Return FLIRT's automatic coordinate scale from the two voxel grids."""

    reference_sizes = np.linalg.norm(reference_sampling[:3, :3], axis=0).astype(np.float32)
    moving_sizes = np.linalg.norm(moving_sampling[:3, :3], axis=0).astype(np.float32)
    maximum = np.float32(max(float(reference_sizes.max()), float(moving_sizes.max())))
    minimum = np.float32(min(float(reference_sizes.min()), float(moving_sizes.min())))
    scale = np.float32(1.0)
    if maximum > np.float32(12.0):
        scale = np.float32(maximum / np.float32(8.0))
    if minimum < np.float32(0.75):
        scale = minimum
    return float(scale)


def _level_evaluator(
    level: FlirtPyramidLevel,
    smooth_size: float,
    cost_function: str = "correlation_ratio",
) -> FlirtWeightedCorrelationRatio | FlirtWeightedMutualInformation:
    evaluator_class = (
        FlirtWeightedCorrelationRatio
        if cost_function == "correlation_ratio"
        else FlirtWeightedMutualInformation
    )
    return evaluator_class(
        level.reference,
        level.moving,
        level.reference_weight,
        level.moving_weight,
        level.reference_sampling,
        level.moving_sampling,
        bins=level.bins,
        smooth_size=smooth_size,
    )


def _stage_optimize(
    evaluator: FlirtWeightedCorrelationRatio,
    initial_matrix: np.ndarray,
    candidate: np.ndarray,
    degrees_of_freedom: int,
    scale: float,
    major_iterations: int,
    bound_guesses: Sequence[float] = (10.0, 1.0),
    voxel_parallel: bool = False,
    center: np.ndarray | None = None,
) -> FlirtOptimizationResult:
    if center is None:
        center = flirt_intensity_cog(evaluator.moving, evaluator.moving_sampling)
    parameters = decompose_affine(candidate, center)
    tolerances = np.array(
        [0.005, 0.005, 0.005, 0.2, 0.2, 0.2, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001],
        dtype=np.float64,
    )
    tolerances *= scale
    evaluate = (
        evaluator.evaluate_ordered_parallel
        if voxel_parallel and hasattr(evaluator, "evaluate_ordered_parallel")
        else evaluator
    )

    def cost(values: np.ndarray) -> float:
        return evaluate(affine_matrix(values[:degrees_of_freedom], center) @ initial_matrix)

    result = flirt_brent_optimize(
        parameters,
        tolerances,
        cost,
        parameter_count=degrees_of_freedom,
        major_iterations=major_iterations,
        bound_guesses=bound_guesses,
    )
    return FlirtOptimizationResult(
        affine_matrix(result.parameters[:degrees_of_freedom], center),
        result.cost,
        result.iterations,
        result.evaluations,
    )


def _run_batched_requests(
    requests: Sequence[CostRequest],
    matrices_for_requests: Callable[[list[int], np.ndarray], np.ndarray],
    evaluator: FlirtWeightedCorrelationRatio | FlirtWeightedMutualInformation,
    workers: int,
) -> list[FlirtOptimizationResult]:
    """Batch-evaluate independent Brent requests in lockstep while preserving trajectory order."""

    pending: dict[int, np.ndarray] = {}
    completed: list[FlirtOptimizationResult | None] = [None] * len(requests)
    for index, request in enumerate(requests):
        try:
            pending[index] = next(request)
        except StopIteration as result:
            completed[index] = result.value
    while pending:
        indices = list(pending)
        values = np.asarray([pending[index] for index in indices], dtype=np.float64)
        matrices = np.asarray(
            matrices_for_requests(indices, values), dtype=np.float64
        )
        costs = evaluator.evaluate_many(matrices, workers=workers)
        following: dict[int, np.ndarray] = {}
        for index, cost in zip(indices, costs, strict=True):
            try:
                following[index] = requests[index].send(float(cost))
            except StopIteration as result:
                completed[index] = result.value
        pending = following
    if any(result is None for result in completed):
        raise RuntimeError("batched FLIRT optimizer ended without a result")
    return [result for result in completed if result is not None]


def _parallel_optimize(
    evaluator: FlirtWeightedCorrelationRatio,
    initial_matrix: np.ndarray,
    matrices: Sequence[np.ndarray],
    degrees_of_freedom: int,
    scale: float,
    major_iterations: int,
    workers: int,
    bound_guesses: Sequence[float] = (10.0, 1.0),
    center: np.ndarray | None = None,
) -> tuple[list[_Candidate], int]:
    # Use voxel parallelism for one start; with two or more starts, use candidate
    # parallelism to occupy all eight workers.
    voxel_parallel = len(matrices) == 1
    if center is None:
        center = flirt_intensity_cog(evaluator.moving, evaluator.moving_sampling)

    def optimize(matrix: np.ndarray) -> FlirtOptimizationResult:
        return _stage_optimize(
            evaluator,
            initial_matrix,
            matrix,
            degrees_of_freedom,
            scale,
            major_iterations,
            bound_guesses,
            voxel_parallel,
            center,
        )

    if voxel_parallel:
        results = [optimize(matrix) for matrix in matrices]
    else:
        starts = [decompose_affine(matrix, center) for matrix in matrices]
        tolerances = np.array(
            [
                0.005,
                0.005,
                0.005,
                0.2,
                0.2,
                0.2,
                0.002,
                0.002,
                0.002,
                0.001,
                0.001,
                0.001,
            ],
            dtype=np.float64,
        )
        tolerances *= scale
        requests = [
            flirt_brent_requests(
                parameters,
                tolerances,
                parameter_count=degrees_of_freedom,
                major_iterations=major_iterations,
                bound_guesses=bound_guesses,
            )
            for parameters in starts
        ]
        raw_results = _run_batched_requests(
            requests,
            lambda _indices, values: affine_matrices(
                values[:, :degrees_of_freedom], center
            )
            @ initial_matrix,
            evaluator,
            workers,
        )
        results = [
            FlirtOptimizationResult(
                affine_matrix(result.parameters[:degrees_of_freedom], center),
                result.cost,
                result.iterations,
                result.evaluations,
            )
            for result in raw_results
        ]
    return (
        [_Candidate(result.cost, result.parameters) for result in results],
        sum(result.evaluations for result in results),
    )


def _interpolate_grid(values: np.ndarray, xvalue: float, yvalue: float, zvalue: float) -> np.float32:
    xindex, yindex, zindex = int(xvalue), int(yvalue), int(zvalue)
    dx, dy, dz = np.float32(xvalue - xindex), np.float32(yvalue - yindex), np.float32(zvalue - zindex)
    xnext = min(xindex + 1, values.shape[0] - 1)
    ynext = min(yindex + 1, values.shape[1] - 1)
    znext = min(zindex + 1, values.shape[2] - 1)
    first = np.float32(np.float32(values[xnext, yindex, zindex] - values[xindex, yindex, zindex]) * dx + values[xindex, yindex, zindex])
    second = np.float32(np.float32(values[xnext, yindex, znext] - values[xindex, yindex, znext]) * dx + values[xindex, yindex, znext])
    third = np.float32(np.float32(values[xnext, ynext, zindex] - values[xindex, ynext, zindex]) * dx + values[xindex, ynext, zindex])
    fourth = np.float32(np.float32(values[xnext, ynext, znext] - values[xindex, ynext, znext]) * dx + values[xindex, ynext, znext])
    fifth = np.float32(np.float32(third - first) * dy + first)
    sixth = np.float32(np.float32(fourth - second) * dy + second)
    return np.float32(np.float32(sixth - fifth) * dz + fifth)


def _interpolate_grid_many(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Interpolate a coarse-search coordinate batch at float32 boundaries using the scalar formula."""

    points = np.asarray(coordinates, dtype=np.float32)
    indices = points.astype(np.int32)
    xindex, yindex, zindex = indices[:, 0], indices[:, 1], indices[:, 2]
    dx = np.asarray(points[:, 0] - xindex, dtype=np.float32)
    dy = np.asarray(points[:, 1] - yindex, dtype=np.float32)
    dz = np.asarray(points[:, 2] - zindex, dtype=np.float32)
    xnext = np.minimum(xindex + 1, values.shape[0] - 1)
    ynext = np.minimum(yindex + 1, values.shape[1] - 1)
    znext = np.minimum(zindex + 1, values.shape[2] - 1)
    first = np.asarray(
        np.asarray(values[xnext, yindex, zindex] - values[xindex, yindex, zindex], dtype=np.float32)
        * dx
        + values[xindex, yindex, zindex],
        dtype=np.float32,
    )
    second = np.asarray(
        np.asarray(values[xnext, yindex, znext] - values[xindex, yindex, znext], dtype=np.float32)
        * dx
        + values[xindex, yindex, znext],
        dtype=np.float32,
    )
    third = np.asarray(
        np.asarray(values[xnext, ynext, zindex] - values[xindex, ynext, zindex], dtype=np.float32)
        * dx
        + values[xindex, ynext, zindex],
        dtype=np.float32,
    )
    fourth = np.asarray(
        np.asarray(values[xnext, ynext, znext] - values[xindex, ynext, znext], dtype=np.float32)
        * dx
        + values[xindex, ynext, znext],
        dtype=np.float32,
    )
    fifth = np.asarray(np.asarray(third - first, dtype=np.float32) * dy + first, dtype=np.float32)
    sixth = np.asarray(
        np.asarray(fourth - second, dtype=np.float32) * dy + second,
        dtype=np.float32,
    )
    return np.asarray(
        np.asarray(sixth - fifth, dtype=np.float32) * dz + fifth,
        dtype=np.float32,
    )


def _rms_deviation(first: np.ndarray, second: np.ndarray, radius: float) -> float:
    difference = first @ np.linalg.inv(second) - np.eye(4)
    linear = difference[:3, :3]
    translation = difference[:3, 3]
    return math.sqrt(
        float(translation @ translation)
        + radius * radius / 5.0 * float(np.trace(linear.T @ linear))
    )


def _add_distinct(
    candidates: list[tuple[_Candidate, _Candidate]],
    optimized: _Candidate,
    preoptimized: _Candidate,
    radius: float,
) -> None:
    for index, (stored, _) in enumerate(candidates):
        if _rms_deviation(optimized.matrix, stored.matrix, radius) < radius:
            if optimized.cost < stored.cost:
                candidates.pop(index)
                break
            return
    position = len(candidates)
    for index, (stored, _) in enumerate(candidates):
        if optimized.cost < stored.cost:
            position = index
            break
    candidates.insert(position, (optimized, preoptimized))


def _fine_cost_threshold(cost_grid: np.ndarray) -> np.float32:
    """Compute the fine-grid screening threshold using the default schedule's 20% rule."""

    minimum = np.float32(cost_grid.min())
    maximum = np.float32(cost_grid.max())
    threshold = min(
        np.float32(minimum + np.float32(0.2) * np.float32(maximum - minimum)),
        np.sort(cost_grid.ravel())[int(cost_grid.size * 0.2)],
    )
    if threshold <= minimum:
        threshold = max(
            np.float32(minimum * np.float32(1.0001)),
            np.float32(minimum * np.float32(0.9999)),
        )
    return threshold


def _fine_grid_minima(cost_grid: np.ndarray) -> list[int]:
    """Extract fine-search local minima over FSL's ten-cell centered range."""

    minima: list[int] = []
    for zindex in range(10):
        for yindex in range(10):
            for xindex in range(10):
                value = cost_grid[xindex, yindex, zindex]
                neighbourhood = cost_grid[
                    max(0, xindex - 1) : min(11, xindex + 2),
                    max(0, yindex - 1) : min(11, yindex + 2),
                    max(0, zindex - 1) : min(11, zindex + 2),
                ]
                if not np.any(neighbourhood < value):
                    minima.append(xindex * 121 + yindex * 11 + zindex)
    if not minima:
        minima = [int(np.argmin(cost_grid))]
    return minima


def _search(
    level: FlirtPyramidLevel,
    initial_matrix: np.ndarray,
    maximum_dof: int,
    workers: int,
    progress: Callable[[str, int, int], None] | None,
    cost_function: str = "correlation_ratio",
) -> tuple[list[_Candidate], list[_Candidate], int]:
    evaluator = _level_evaluator(level, 8.0, cost_function)
    center = flirt_intensity_cog(level.moving, level.moving_sampling)
    reference_center = flirt_intensity_cog(level.reference, level.reference_sampling)
    transformed_center = initial_matrix @ np.r_[center, 1.0]
    translation = reference_center - transformed_center[:3]
    coarse = np.linspace(-math.pi / 2.0, math.pi / 2.0, 4, dtype=np.float64)
    fine = np.linspace(-math.pi / 2.0, math.pi / 2.0, 11, dtype=np.float64)
    dof = min(maximum_dof, 7)
    reduced_count = 4 if dof > 6 else 3
    reduced_tolerance = np.array([0.016, 1.6, 1.6, 1.6]) if dof > 6 else np.full(3, 1.6)
    rotations = [(xvalue, yvalue, zvalue) for xvalue in coarse for yvalue in coarse for zvalue in coarse]

    coarse_parameters = [
        np.array(
            [*rotation, *translation, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        for rotation in rotations
    ]

    coarse_parameter_array = np.asarray(coarse_parameters)

    def reduced_matrices(indices: list[int], reduced: np.ndarray) -> np.ndarray:
        parameters = coarse_parameter_array[indices].copy()
        if dof > 6:
            parameters[:, 6:9] += reduced[:, :1]
            parameters[:, 3:6] += reduced[:, 1:4]
        else:
            parameters[:, 3:6] += reduced
        return affine_matrices(parameters[:, :dof], center) @ initial_matrix

    coarse_optimizers = [
        flirt_brent_requests(
            np.zeros(reduced_count),
            reduced_tolerance,
            major_iterations=4,
        )
        for _ in rotations
    ]
    coarse_raw = _run_batched_requests(
        coarse_optimizers, reduced_matrices, evaluator, workers
    )
    coarse_results: list[tuple[np.ndarray, int]] = []
    for reference_parameters, result in zip(
        coarse_parameters, coarse_raw, strict=True
    ):
        parameters = reference_parameters.copy()
        if dof > 6:
            parameters[6:9] += result.parameters[0]
            parameters[3:6] += result.parameters[1:4]
        else:
            parameters[3:6] += result.parameters
        coarse_results.append((parameters, result.evaluations))
    evaluations = sum(item[1] for item in coarse_results)
    translations = np.empty((4, 4, 4, 3), dtype=np.float32)
    scales = np.empty((4, 4, 4), dtype=np.float32)
    for flat_index, (parameters, _) in enumerate(coarse_results):
        xindex, remainder = divmod(flat_index, 16)
        yindex, zindex = divmod(remainder, 4)
        translations[xindex, yindex, zindex] = parameters[3:6]
        scales[xindex, yindex, zindex] = parameters[6]
    median_scale = np.sort(scales.ravel())[scales.size // 2]
    scales[:] = median_scale
    factor = np.float32(3.0 / 10.0)
    grid_indices = np.indices((11, 11, 11), dtype=np.float32).reshape(3, -1).T
    coordinates = np.asarray(grid_indices * factor, dtype=np.float32)
    fine_parameters = np.zeros((coordinates.shape[0], 12), dtype=np.float64)
    fine_parameters[:, :3] = fine[grid_indices.astype(np.int32)]
    for axis in range(3):
        fine_parameters[:, 3 + axis] = _interpolate_grid_many(
            translations[..., axis], coordinates
        )
    fine_parameters[:, 6:9] = _interpolate_grid_many(scales, coordinates)[:, None]
    if dof <= 6:
        fine_parameters[:, 6:9] = 1.0
    fine_matrices = (
        affine_matrices(fine_parameters[:, :dof], center) @ initial_matrix
    )
    fine_costs = evaluator.evaluate_many(fine_matrices, workers=workers).astype(np.float32)
    evaluations += len(fine_matrices)
    cost_grid = fine_costs.reshape(11, 11, 11)
    threshold = _fine_cost_threshold(cost_grid)
    selected = [index for index, value in enumerate(fine_costs) if value < threshold]

    selected_parameters = [fine_parameters[index].copy() for index in selected]

    selected_parameter_array = np.asarray(selected_parameters)

    def refined_matrices(indices: list[int], reduced: np.ndarray) -> np.ndarray:
        parameters = selected_parameter_array[indices].copy()
        if dof > 6:
            parameters[:, 6:9] += reduced[:, :1]
            parameters[:, 3:6] += reduced[:, 1:4]
        else:
            parameters[:, 3:6] += reduced
        return affine_matrices(parameters[:, :dof], center) @ initial_matrix

    refined_raw = _run_batched_requests(
        [
            flirt_brent_requests(
                np.zeros(reduced_count),
                reduced_tolerance,
                major_iterations=4,
            )
            for _ in selected
        ],
        refined_matrices,
        evaluator,
        workers,
    )
    refined: list[tuple[np.ndarray, float, int]] = []
    for reference_parameters, result in zip(
        selected_parameters, refined_raw, strict=True
    ):
        parameters = reference_parameters.copy()
        if dof > 6:
            parameters[6:9] += result.parameters[0]
            parameters[3:6] += result.parameters[1:4]
        else:
            parameters[3:6] += result.parameters
        refined.append((parameters, result.cost, result.evaluations))
    for index, (parameters, value, count) in zip(selected, refined, strict=True):
        fine_parameters[index] = parameters
        fine_costs[index] = value
        cost_grid.flat[index] = value
        evaluations += count
    minima = _fine_grid_minima(cost_grid)
    pre_matrices = affine_matrices(fine_parameters[minima, :dof], center)
    optimized, count = _parallel_optimize(
        evaluator,
        initial_matrix,
        pre_matrices,
        dof,
        level.requested_scale,
        4,
        workers,
        center=center,
    )
    evaluations += count
    distinct: list[tuple[_Candidate, _Candidate]] = []
    radius = float(np.min(np.linalg.norm(level.reference_sampling[:3, :3], axis=0)))
    for optimized_candidate, prematrix in zip(optimized, pre_matrices, strict=True):
        precost = float(evaluator(prematrix @ initial_matrix))
        evaluations += 1
        _add_distinct(
            distinct,
            optimized_candidate,
            _Candidate(precost, prematrix),
            radius,
        )
    if progress:
        progress("search", len(rotations) + len(fine_matrices) + len(selected), len(rotations) + len(fine_matrices) + len(selected))
    return [item[0] for item in distinct], [item[1] for item in distinct], evaluations


def _measure(
    evaluator: FlirtWeightedCorrelationRatio,
    initial_matrix: np.ndarray,
    candidates: Sequence[_Candidate],
    workers: int,
) -> tuple[list[_Candidate], int]:
    matrices = np.asarray([candidate.matrix @ initial_matrix for candidate in candidates])
    costs = evaluator.evaluate_many(matrices, workers=workers)
    return [
        _Candidate(float(cost), candidate.matrix) for cost, candidate in zip(costs, candidates, strict=True)
    ], len(candidates)


def register_flirt_affine(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_weight: np.ndarray,
    moving_weight: np.ndarray,
    reference_sampling: np.ndarray,
    moving_sampling: np.ndarray,
    *,
    degrees_of_freedom: int = 12,
    initial_matrix: np.ndarray | None = None,
    qsform_matrix: np.ndarray | None = None,
    workers: int = 8,
    cost_function: str = "correlation_ratio",
    progress: Callable[[str, int, int], None] | None = None,
) -> FlirtRegistrationResult:
    """Run FSL's default 3D FLIRT schedule with the specified upstream cost."""

    if degrees_of_freedom not in range(6, 13):
        raise ValueError("degrees_of_freedom must be between six and twelve")
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if cost_function not in ("correlation_ratio", "mutual_information"):
        raise ValueError(
            "cost_function must be correlation_ratio or mutual_information"
        )
    initial = np.eye(4) if initial_matrix is None else np.asarray(initial_matrix, dtype=np.float64)
    qsform = np.eye(4) if qsform_matrix is None else np.asarray(qsform_matrix, dtype=np.float64)
    for matrix, name in ((initial, "initial_matrix"), (qsform, "qsform_matrix")):
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
            raise ValueError(f"{name} must be a finite invertible 4x4 matrix")
    base_scale = _base_scale(reference_sampling, moving_sampling)
    coordinate_scale = np.diag([1.0 / base_scale] * 3 + [1.0])
    inverse_coordinate_scale = np.diag([base_scale] * 3 + [1.0])
    internal_initial = coordinate_scale @ initial @ inverse_coordinate_scale
    internal_qsform = coordinate_scale @ qsform @ inverse_coordinate_scale
    internal_reference_sampling = coordinate_scale @ np.asarray(
        reference_sampling, dtype=np.float64
    )
    internal_moving_sampling = coordinate_scale @ np.asarray(
        moving_sampling, dtype=np.float64
    )
    levels = build_flirt_pyramid(
        reference,
        moving,
        reference_weight,
        moving_weight,
        internal_reference_sampling,
        internal_moving_sampling,
    )
    optimized_search, preoptimized_search, evaluations = _search(
        levels[8],
        internal_initial,
        degrees_of_freedom,
        workers,
        progress,
        cost_function,
    )
    evaluator4 = _level_evaluator(levels[4], 4.0, cost_function)
    center4 = flirt_intensity_cog(levels[4].moving, levels[4].moving_sampling)
    measured_optimized, count = _measure(
        evaluator4, internal_initial, optimized_search, workers
    )
    evaluations += count
    measured_preoptimized, count = _measure(
        evaluator4, internal_initial, preoptimized_search, workers
    )
    evaluations += count
    pairs = sorted(zip(measured_optimized, measured_preoptimized, strict=True), key=lambda item: item[0].cost)
    starts = [item.matrix for item in [pair[0] for pair in pairs[:3]] + [pair[1] for pair in pairs[:3]]]
    starts.append(np.eye(4))
    candidates, count = _parallel_optimize(
        evaluator4,
        internal_initial,
        starts,
        min(degrees_of_freedom, 7),
        4.0,
        4,
        workers,
        center=center4,
    )
    evaluations += count
    candidates.sort(key=lambda item: item.cost)
    anchors = candidates[:4]
    expanded = list(anchors)
    delta = np.array([math.radians(9.0)] * 3 + [0.05] * 3 + [0.1] * 3 + [0.05] * 3)
    perturbations = [
        (np.array([sign if axis == index else 0.0 for axis in range(12)]), True)
        for index in range(3)
        for sign in (1.0, -1.0)
    ] + [
        (np.array([0.0] * 6 + [value] * 3 + [0.0] * 3), False)
        for value in (0.1, -0.1, 0.2, -0.2)
    ]
    perturbed_starts: list[np.ndarray] = []
    for perturbation, relative in perturbations:
        for anchor in anchors:
            parameters = decompose_affine(anchor.matrix, center4)
            parameters += perturbation * delta if relative else perturbation
            perturbed_starts.append(affine_matrix(parameters, center4))
    block, count = _parallel_optimize(
        evaluator4,
        internal_initial,
        perturbed_starts,
        min(degrees_of_freedom, 7),
        4.0,
        4,
        workers,
        center=center4,
    )
    expanded.extend(block)
    evaluations += count
    expanded.sort(key=lambda item: item.cost)

    evaluator2 = _level_evaluator(levels[2], 2.0, cost_function)
    center2 = flirt_intensity_cog(levels[2].moving, levels[2].moving_sampling)
    measured, count = _measure(evaluator2, internal_initial, expanded, workers)
    evaluations += count
    measured.sort(key=lambda item: item.cost)
    stage, count = _parallel_optimize(
        evaluator2,
        internal_initial,
        [measured[0].matrix],
        min(degrees_of_freedom, 7),
        2.0,
        4,
        workers,
        center=center2,
    )
    evaluations += count
    current = stage
    if degrees_of_freedom > 7:
        current, count = _parallel_optimize(
            evaluator2,
            internal_initial,
            [current[0].matrix],
            min(degrees_of_freedom, 9),
            2.0,
            1,
            workers,
            (1.0,),
            center2,
        )
        evaluations += count
    if degrees_of_freedom > 9:
        current, count = _parallel_optimize(
            evaluator2,
            internal_initial,
            [current[0].matrix],
            degrees_of_freedom,
            2.0,
            2,
            workers,
            (1.0,),
            center2,
        )
        evaluations += count

    evaluator1 = _level_evaluator(levels[1], 1.0, cost_function)
    center1 = flirt_intensity_cog(levels[1].moving, levels[1].moving_sampling)
    relative_qsform = internal_qsform @ np.linalg.inv(internal_initial)
    final_starts = [current[0].matrix, relative_qsform]
    final, count = _parallel_optimize(
        evaluator1,
        internal_initial,
        final_starts,
        degrees_of_freedom,
        levels[1].requested_scale,
        1,
        workers,
        (1.0,),
        center1,
    )
    evaluations += count
    qcost = float(evaluator1(internal_qsform))
    evaluations += 1
    final.append(_Candidate(qcost, relative_qsform))
    final.sort(key=lambda item: item.cost)
    if progress:
        progress("complete", 1, 1)
    best = final[0]
    return FlirtRegistrationResult(
        inverse_coordinate_scale @ best.matrix @ internal_initial @ coordinate_scale,
        best.cost,
        evaluations,
        len(expanded),
    )


def register_flirt_nosearch_mutual_information(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_sampling: np.ndarray,
    moving_sampling: np.ndarray,
    *,
    degrees_of_freedom: int = 12,
    initial_matrix: np.ndarray | None = None,
    workers: int = 8,
) -> FlirtRegistrationResult:
    """Run the default optimization stage of ``flirt -nosearch -cost mutualinfo``."""

    if degrees_of_freedom not in range(6, 13):
        raise ValueError("degrees_of_freedom must be between six and twelve")
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    initial = np.eye(4) if initial_matrix is None else np.asarray(initial_matrix, dtype=np.float64)
    if (
        initial.shape != (4, 4)
        or not np.all(np.isfinite(initial))
        or abs(np.linalg.det(initial[:3, :3])) < 1e-12
    ):
        raise ValueError("initial_matrix must be a finite invertible 4x4 matrix")
    base_scale = _base_scale(reference_sampling, moving_sampling)
    coordinate_scale = np.diag([1.0 / base_scale] * 3 + [1.0])
    inverse_coordinate_scale = np.diag([base_scale] * 3 + [1.0])
    internal_initial = coordinate_scale @ initial @ inverse_coordinate_scale
    reference_sampling_internal = coordinate_scale @ np.asarray(reference_sampling, dtype=np.float64)
    moving_sampling_internal = coordinate_scale @ np.asarray(moving_sampling, dtype=np.float64)
    unit_reference = np.ones(np.asarray(reference).shape, dtype=np.float32)
    unit_moving = np.ones(np.asarray(moving).shape, dtype=np.float32)
    levels = build_flirt_pyramid(
        reference,
        moving,
        unit_reference,
        unit_moving,
        reference_sampling_internal,
        moving_sampling_internal,
    )

    def evaluator(scale: int) -> FlirtWeightedMutualInformation:
        level = levels[scale]
        return FlirtWeightedMutualInformation(
            level.reference,
            level.moving,
            level.reference_weight,
            level.moving_weight,
            level.reference_sampling,
            level.moving_sampling,
            bins=level.bins,
            smooth_size=float(scale),
        )

    stages = [
        (4, min(degrees_of_freedom, 7), 4, (10.0, 1.0)),
        (2, min(degrees_of_freedom, 7), 4, (10.0, 1.0)),
    ]
    if degrees_of_freedom > 7:
        stages.append((2, min(degrees_of_freedom, 9), 1, (1.0,)))
    if degrees_of_freedom > 9:
        stages.append((2, degrees_of_freedom, 2, (1.0,)))
    stages.append((1, degrees_of_freedom, 1, (1.0,)))
    evaluations = 0
    current = [_Candidate(float("inf"), np.eye(4))]
    for scale, dof, iterations, bounds in stages:
        level = levels[scale]
        block, count = _parallel_optimize(
            evaluator(scale),
            internal_initial,
            [current[0].matrix],
            dof,
            level.requested_scale,
            iterations,
            workers,
            bounds,
        )
        current = block
        evaluations += count
    best = current[0]
    return FlirtRegistrationResult(
        inverse_coordinate_scale @ best.matrix @ internal_initial @ coordinate_scale,
        best.cost,
        evaluations,
        1,
    )
