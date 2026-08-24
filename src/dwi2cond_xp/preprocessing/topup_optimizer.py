"""FSL MISCMATHS optimizers required by the fixed TOPUP subset."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np
from numba import njit, prange
from scipy.sparse import csc_matrix, issparse


class MatrixOperator(Protocol):
    """Minimal matrix operation contract used by FSL's PCG solver."""

    @property
    def shape(self) -> tuple[int, int]: ...

    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def diagonal(self) -> np.ndarray: ...


@dataclass(frozen=True)
class PcgResult:
    """Solution and convergence information from diagonal-preconditioned CG."""

    solution: np.ndarray
    iterations: int
    relative_residual: float
    converged: bool


@dataclass(frozen=True)
class TopupOptimizationTrace:
    """One attempted optimizer step, including rejected LM attempts."""

    iteration: int
    accepted: bool
    cost: float
    attempted_cost: float
    damping: float


@dataclass(frozen=True)
class TopupOptimizationResult:
    """Final parameters, cost, status and complete attempted-step trace."""

    parameters: np.ndarray
    cost: float
    status: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    trace: tuple[TopupOptimizationTrace, ...]


CostFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]
HessianFunction = Callable[[np.ndarray], MatrixOperator | np.ndarray]


@njit(cache=True)
def _dense_matvec_row_order(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Multiply a dense matrix while retaining ascending-column accumulation."""

    result = np.empty(vector.size, dtype=np.float64)
    for row in range(vector.size):
        value = 0.0
        for column in range(vector.size):
            value += matrix[row, column] * vector[column]
        result[row] = value
    return result


@njit(cache=True)
def _fsl_diagonal_pcg_dense(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    diagonal: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float, bool]:
    """Run the dense FSL PCG loop without repeated Python dispatch."""

    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    norm_rhs = np.linalg.norm(right_hand_side)
    if norm_rhs == 0.0:
        norm_rhs = 1.0
    relative_residual = np.linalg.norm(residual) / norm_rhs
    if relative_residual <= tolerance:
        return solution, 0, relative_residual, True

    direction = np.empty_like(right_hand_side)
    previous_rho = 0.0
    for iteration in range(1, max_iterations + 1):
        preconditioned = residual / diagonal
        rho = np.dot(residual, preconditioned)
        if iteration == 1:
            direction[:] = preconditioned
        else:
            direction *= rho / previous_rho
            direction += preconditioned
        product = _dense_matvec_row_order(matrix, direction)
        denominator = np.dot(direction, product)
        if denominator == 0.0 or not np.isfinite(denominator):
            raise np.linalg.LinAlgError("PCG encountered a singular search direction")
        alpha = rho / denominator
        solution += alpha * direction
        residual -= alpha * product
        relative_residual = np.linalg.norm(residual) / norm_rhs
        if relative_residual <= tolerance:
            return solution, iteration, relative_residual, True
        previous_rho = rho
    return solution, max_iterations, relative_residual, False


@njit(cache=True, parallel=True)
def _csr_matvec_row_order(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Compute CSR products by ascending column index while preserving FSL row sums."""

    result = np.empty(indptr.size - 1, dtype=np.float64)
    for row in prange(result.size):
        value = 0.0
        for offset in range(indptr[row], indptr[row + 1]):
            value += data[offset] * vector[indices[offset]]
        result[row] = value
    return result


@njit(cache=True, nogil=True)
def _fsl_diagonal_pcg_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    right_hand_side: np.ndarray,
    diagonal: np.ndarray,
    tolerance: float,
    max_iterations: int,
    progress_iteration: np.ndarray,
    progress_residual: np.ndarray,
) -> tuple[np.ndarray, int, float, bool]:
    """Run FSL's sparse diagonally preconditioned CG loop inside Numba."""

    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    norm_rhs = np.linalg.norm(right_hand_side)
    if norm_rhs == 0.0:
        norm_rhs = 1.0
    relative_residual = np.linalg.norm(residual) / norm_rhs
    progress_iteration[0] = 0
    progress_residual[0] = relative_residual
    if relative_residual <= tolerance:
        return solution, 0, relative_residual, True

    direction = np.empty_like(right_hand_side)
    previous_rho = 0.0
    for iteration in range(1, max_iterations + 1):
        preconditioned = residual / diagonal
        rho = np.dot(residual, preconditioned)
        if iteration == 1:
            direction[:] = preconditioned
        else:
            direction *= rho / previous_rho
            direction += preconditioned
        product = _csr_matvec_row_order(data, indices, indptr, direction)
        denominator = np.dot(direction, product)
        if denominator == 0.0 or not np.isfinite(denominator):
            raise np.linalg.LinAlgError("PCG encountered a singular search direction")
        alpha = rho / denominator
        solution += alpha * direction
        residual -= alpha * product
        relative_residual = np.linalg.norm(residual) / norm_rhs
        progress_iteration[0] = iteration
        progress_residual[0] = relative_residual
        if relative_residual <= tolerance:
            return solution, iteration, relative_residual, True
        previous_rho = rho
    return solution, max_iterations, relative_residual, False


def _as_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector.copy()


def _finite_cost(cost: CostFunction, parameters: np.ndarray) -> float:
    value = float(cost(parameters))
    if not np.isfinite(value):
        raise ValueError("cost function returned a non-finite value")
    return value


def _finite_gradient(gradient: GradientFunction, parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(gradient(parameters), dtype=np.float64)
    if values.shape != parameters.shape:
        raise ValueError("gradient shape must match the parameter vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("gradient returned non-finite values")
    return values.copy()


def _zero_cost_difference(old: float, new: float, tolerance: float) -> bool:
    epsilon = np.finfo(np.float64).eps
    return 2.0 * abs(old - new) <= tolerance * (abs(old) + abs(new) + epsilon)


def _zero_gradient(
    parameters: np.ndarray,
    gradient: np.ndarray,
    cost: float,
    tolerance: float,
) -> bool:
    scaled = np.abs(gradient) * np.maximum(np.abs(parameters), 1.0)
    return float(np.max(scaled)) / max(cost, 1.0) < tolerance


def fsl_diagonal_pcg(
    matrix: MatrixOperator | np.ndarray,
    right_hand_side: np.ndarray,
    *,
    tolerance: float = 1.0e-3,
    max_iterations: int = 500,
    dense_backend: str = "optimized",
    sparse_backend: str = "optimized",
    progress: Callable[[int, int, float], None] | None = None,
) -> PcgResult:
    """Solve ``A x = b`` using FSL's zero-start diagonal PCG sequence."""

    rhs = _as_vector(right_hand_side, name="right-hand side")
    if len(matrix.shape) != 2 or matrix.shape != (rhs.size, rhs.size):
        raise ValueError("matrix must be square and match the right-hand side")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if dense_backend not in ("reference", "optimized"):
        raise ValueError("dense backend must be reference or optimized")
    if sparse_backend not in ("reference", "optimized"):
        raise ValueError("sparse backend must be reference or optimized")
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    if diagonal.shape != rhs.shape or not np.all(np.isfinite(diagonal)):
        raise ValueError("matrix diagonal must be finite and match the system size")
    if np.any(diagonal == 0.0):
        raise np.linalg.LinAlgError("diagonal preconditioner contains a zero")

    if isinstance(matrix, np.ndarray) and dense_backend == "optimized":
        solution, iterations, relative_residual, converged = (
            _fsl_diagonal_pcg_dense(
                matrix,
                rhs,
                diagonal,
                tolerance,
                max_iterations,
            )
        )
        return PcgResult(solution, iterations, relative_residual, converged)

    if issparse(matrix) and sparse_backend == "optimized":
        csr = matrix.tocsr(copy=False)
        csr.sort_indices()
        progress_iteration = np.zeros(1, dtype=np.int64)
        progress_residual = np.ones(1, dtype=np.float64)

        def solve_csr() -> tuple[np.ndarray, int, float, bool]:
            return _fsl_diagonal_pcg_csr(
                np.asarray(csr.data, dtype=np.float64),
                np.asarray(csr.indices),
                np.asarray(csr.indptr),
                rhs,
                diagonal,
                tolerance,
                max_iterations,
                progress_iteration,
                progress_residual,
            )

        if progress is None:
            solution, iterations, relative_residual, converged = solve_csr()
        else:
            last_iteration = -1
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_csr)
                while not future.done():
                    iteration = int(progress_iteration[0])
                    if iteration != last_iteration:
                        progress(
                            iteration,
                            max_iterations,
                            float(progress_residual[0]),
                        )
                        last_iteration = iteration
                    time.sleep(0.1)
                solution, iterations, relative_residual, converged = future.result()
            progress(iterations, max_iterations, relative_residual)
        return PcgResult(solution, iterations, relative_residual, converged)

    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    norm_rhs = float(np.linalg.norm(rhs))
    if norm_rhs == 0.0:
        norm_rhs = 1.0
    relative_residual = float(np.linalg.norm(residual)) / norm_rhs
    if relative_residual <= tolerance:
        return PcgResult(solution, 0, relative_residual, True)

    direction = np.empty_like(rhs)
    previous_rho = 0.0
    for iteration in range(1, max_iterations + 1):
        preconditioned = residual / diagonal
        rho = float(np.dot(residual, preconditioned))
        if iteration == 1:
            direction[:] = preconditioned
        else:
            direction *= rho / previous_rho
            direction += preconditioned
        if isinstance(matrix, np.ndarray):
            product = _dense_matvec_row_order(matrix, direction)
        else:
            product = np.asarray(matrix @ direction, dtype=np.float64)
        denominator = float(np.dot(direction, product))
        if denominator == 0.0 or not np.isfinite(denominator):
            raise np.linalg.LinAlgError("PCG encountered a singular search direction")
        alpha = rho / denominator
        solution += alpha * direction
        residual -= alpha * product
        relative_residual = float(np.linalg.norm(residual)) / norm_rhs
        if progress is not None:
            progress(iteration, max_iterations, relative_residual)
        if relative_residual <= tolerance:
            return PcgResult(solution, iteration, relative_residual, True)
        previous_rho = rho
    return PcgResult(solution, max_iterations, relative_residual, False)


def _copy_hessian(
    matrix: MatrixOperator | np.ndarray, size: int
) -> csc_matrix | np.ndarray:
    if matrix.shape != (size, size):
        raise ValueError("Hessian must match the parameter-vector size")
    if issparse(matrix):
        copied = csc_matrix(matrix, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(copied.data)):
            raise ValueError("Hessian contains non-finite values")
        return copied
    copied = np.asarray(matrix, dtype=np.float64)
    if copied.shape != (size, size) or not np.all(np.isfinite(copied)):
        raise ValueError("Hessian must be finite and match the parameter-vector size")
    return copied.copy()


def _scale_hessian_diagonal(
    matrix: csc_matrix | np.ndarray,
    factor: float,
) -> None:
    if issparse(matrix):
        diagonal = np.asarray(matrix.diagonal(), dtype=np.float64) * factor
        matrix.setdiag(diagonal)
        matrix.sort_indices()
    else:
        indices = np.diag_indices_from(matrix)
        matrix[indices] *= factor


def fsl_levenberg_marquardt(
    initial_parameters: np.ndarray,
    cost: CostFunction,
    gradient: GradientFunction,
    hessian: HessianFunction,
    *,
    max_iterations: int,
    cost_tolerance: float = 1.0e-8,
    initial_damping: float = 0.1,
    damping_limit: float = 1.0e20,
    equation_tolerance: float = 1.0e-3,
    equation_max_iterations: int = 500,
    progress: Callable[[str, int, int, float | None], None] | None = None,
) -> TopupOptimizationResult:
    """Run FSL's Levenberg-Marquardt accept/reject and damping sequence."""

    parameters = _as_vector(initial_parameters, name="initial parameters")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    current_cost = _finite_cost(cost, parameters)
    function_evaluations = 1
    gradient_evaluations = 0
    hessian_evaluations = 0
    damping = float(initial_damping)
    old_damping = 0.0
    successful_iterations = 0
    previous_success = True
    current_gradient: np.ndarray | None = None
    current_hessian: csc_matrix | np.ndarray | None = None
    trace: list[TopupOptimizationTrace] = []
    status = "maximum_iterations"

    while previous_success is False or successful_iterations < max_iterations:
        if previous_success:
            successful_iterations += 1
            if progress is not None:
                progress("gradient", successful_iterations, max_iterations, None)
            current_gradient = _finite_gradient(gradient, parameters)
            if progress is not None:
                progress("hessian", successful_iterations, max_iterations, None)
            current_hessian = _copy_hessian(hessian(parameters), parameters.size)
            gradient_evaluations += 1
            hessian_evaluations += 1
        assert current_gradient is not None and current_hessian is not None
        factor = (1.0 + damping) / (1.0 + old_damping)
        _scale_hessian_diagonal(current_hessian, factor)
        inversion_failed = False
        try:
            solve = fsl_diagonal_pcg(
                current_hessian,
                current_gradient,
                tolerance=equation_tolerance,
                max_iterations=equation_max_iterations,
                progress=(
                    None
                    if progress is None
                    else lambda done, total, residual: progress(
                        "pcg", done, total, residual
                    )
                ),
            )
            step = -solve.solution
            attempted_cost = _finite_cost(cost, parameters + step)
            function_evaluations += 1
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            inversion_failed = True
            attempted_cost = float("inf")

        accepted = not inversion_failed and attempted_cost < current_cost
        if progress is not None:
            progress("lm", successful_iterations, max_iterations, attempted_cost)
        trace.append(
            TopupOptimizationTrace(
                successful_iterations,
                accepted,
                current_cost,
                attempted_cost,
                damping,
            )
        )
        if accepted:
            old_cost = current_cost
            parameters += step
            current_cost = attempted_cost
            damping /= 10.0
            old_damping = 0.0
            previous_success = True
            if _zero_cost_difference(old_cost, current_cost, cost_tolerance):
                status = "cost_converged"
                break
        else:
            previous_success = False
            old_damping = damping
            damping *= 10.0
            if damping > damping_limit:
                status = "damping_converged"
                break
    return TopupOptimizationResult(
        parameters,
        current_cost,
        status,
        successful_iterations,
        function_evaluations,
        gradient_evaluations,
        hessian_evaluations,
        tuple(trace),
    )


def fsl_scaled_conjugate_gradient(
    initial_parameters: np.ndarray,
    cost: CostFunction,
    gradient: GradientFunction,
    *,
    max_iterations: int,
    gradient_tolerance: float = 1.0e-8,
    initial_damping: float = 0.1,
    sigma: float = 1.0e-2,
) -> TopupOptimizationResult:
    """Run the scaled conjugate-gradient sequence used by FSL MISCMATHS."""

    parameters = _as_vector(initial_parameters, name="initial parameters")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    current_cost = _finite_cost(cost, parameters)
    function_evaluations = 1
    negative_gradient = -_finite_gradient(gradient, parameters)
    gradient_evaluations = 1
    direction = negative_gradient.copy()
    damping = float(initial_damping)
    damping_bar = 0.0
    success = True
    curvature = 0.0
    trace: list[TopupOptimizationTrace] = []
    status = "maximum_iterations"

    for iteration in range(1, max_iterations + 1):
        direction_norm_squared = float(np.dot(direction, direction))
        if direction_norm_squared == 0.0:
            status = "gradient_converged"
            break
        if success:
            sigma_k = sigma / np.sqrt(direction_norm_squared)
            displaced_gradient = _finite_gradient(
                gradient, parameters + sigma_k * direction
            )
            gradient_evaluations += 1
            hessian_direction = (displaced_gradient + negative_gradient) / sigma_k
            curvature = float(np.dot(direction, hessian_direction))
        adjustment = damping - damping_bar
        hessian_direction += adjustment * direction
        curvature += adjustment * direction_norm_squared
        if curvature <= 0.0:
            correction = damping - 2.0 * curvature / direction_norm_squared
            hessian_direction += correction * direction
            damping_bar = 2.0 * (damping - curvature / direction_norm_squared)
            curvature = damping * direction_norm_squared - curvature
            damping = damping_bar

        projection = float(np.dot(direction, negative_gradient))
        alpha = projection / curvature
        attempted_cost = _finite_cost(cost, parameters + alpha * direction)
        function_evaluations += 1
        quality = (
            2.0
            * curvature
            * (current_cost - attempted_cost)
            / (projection * projection)
        )
        accepted = quality >= 0.0
        trace.append(
            TopupOptimizationTrace(
                iteration, accepted, current_cost, attempted_cost, damping
            )
        )
        if accepted:
            current_cost = attempted_cost
            parameters += alpha * direction
            damping_bar = 0.0
            success = True
            old_negative_gradient = negative_gradient
            negative_gradient = -_finite_gradient(gradient, parameters)
            gradient_evaluations += 1
            if iteration % parameters.size == 0:
                direction = negative_gradient.copy()
            else:
                beta = (
                    float(np.dot(negative_gradient, negative_gradient))
                    - float(np.dot(old_negative_gradient, negative_gradient))
                ) / projection
                direction *= beta
                direction += negative_gradient
            if quality > 0.75:
                damping /= 2.0
        else:
            damping_bar = damping
            success = False
        if quality < 0.25:
            damping *= 4.0
        if _zero_gradient(
            parameters, negative_gradient, current_cost, gradient_tolerance
        ):
            status = "gradient_converged"
            break

    return TopupOptimizationResult(
        parameters,
        current_cost,
        status,
        len(trace),
        function_evaluations,
        gradient_evaluations,
        0,
        tuple(trace),
    )
