"""Tests for the FSL-compatible TOPUP optimizer primitives."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threading import Event
from types import SimpleNamespace

import dwi2cond_xp.preprocessing.topup_optimizer as optimizer_module
from dwi2cond_xp.preprocessing.topup_optimizer import (
    fsl_diagonal_pcg,
    fsl_levenberg_marquardt,
    fsl_scaled_conjugate_gradient,
)


def test_diagonal_pcg_preserves_fsl_iteration_sequence() -> None:
    matrix = csc_matrix(np.asarray([[4.0, 1.0], [1.0, 3.0]]))
    result = fsl_diagonal_pcg(matrix, np.asarray([1.0, 2.0]), tolerance=1.0e-12)

    np.testing.assert_allclose(result.solution, [1.0 / 11.0, 7.0 / 11.0])
    assert result.iterations == 2
    assert result.relative_residual < 1.0e-12
    assert result.converged


def test_diagonal_pcg_reports_iteration_limit() -> None:
    matrix = csc_matrix(np.asarray([[4.0, 1.0], [1.0, 3.0]]))
    result = fsl_diagonal_pcg(
        matrix, np.asarray([1.0, 2.0]), tolerance=1.0e-15, max_iterations=1
    )

    assert result.iterations == 1
    assert not result.converged
    assert result.relative_residual > 1.0e-15

    reference = fsl_diagonal_pcg(
        matrix,
        np.asarray([1.0, 2.0]),
        tolerance=1.0e-15,
        max_iterations=1,
        sparse_backend="reference",
    )
    assert reference.iterations == 1
    assert not reference.converged


def test_reference_dense_pcg_accepts_zero_residual() -> None:
    result = fsl_diagonal_pcg(
        np.eye(2), np.zeros(2), dense_backend="reference"
    )

    assert result.iterations == 0
    assert result.relative_residual == 0.0
    assert result.converged


def test_dense_pcg_optimized_backend_is_bitwise_equal_to_reference() -> None:
    matrix = np.asarray(
        [
            [6.0, 0.75, -0.25, 0.5],
            [0.75, 5.0, 0.4, -0.3],
            [-0.25, 0.4, 4.5, 0.2],
            [0.5, -0.3, 0.2, 3.5],
        ],
        dtype=np.float64,
    )
    right_hand_side = np.asarray([1.25, -2.0, 0.75, 3.0], dtype=np.float64)
    reference = fsl_diagonal_pcg(
        matrix,
        right_hand_side,
        tolerance=1.0e-14,
        dense_backend="reference",
    )
    optimized = fsl_diagonal_pcg(
        matrix,
        right_hand_side,
        tolerance=1.0e-14,
        dense_backend="optimized",
    )

    assert np.array_equal(optimized.solution, reference.solution)
    assert optimized.iterations == reference.iterations
    assert optimized.relative_residual == reference.relative_residual
    assert optimized.converged == reference.converged


def test_sparse_pcg_optimized_backend_is_bitwise_equal_to_reference() -> None:
    matrix = csc_matrix(
        np.asarray(
            [
                [6.0, 0.75, 0.0, 0.5],
                [0.75, 5.0, 0.4, 0.0],
                [0.0, 0.4, 4.5, 0.2],
                [0.5, 0.0, 0.2, 3.5],
            ],
            dtype=np.float64,
        )
    )
    right_hand_side = np.asarray([1.25, -2.0, 0.75, 3.0], dtype=np.float64)
    reference = fsl_diagonal_pcg(
        matrix,
        right_hand_side,
        tolerance=1.0e-14,
        sparse_backend="reference",
    )
    optimized = fsl_diagonal_pcg(
        matrix,
        right_hand_side,
        tolerance=1.0e-14,
        sparse_backend="optimized",
    )

    assert np.array_equal(optimized.solution, reference.solution)
    assert optimized.iterations == reference.iterations
    assert optimized.relative_residual == reference.relative_residual
    assert optimized.converged == reference.converged


def test_sparse_pcg_progress_preserves_reference_result() -> None:
    matrix = csc_matrix(
        np.asarray(
            [
                [6.0, 0.75, 0.0, 0.5],
                [0.75, 5.0, 0.4, 0.0],
                [0.0, 0.4, 4.5, 0.2],
                [0.5, 0.0, 0.2, 3.5],
            ],
            dtype=np.float64,
        )
    )
    right_hand_side = np.asarray([1.25, -2.0, 0.75, 3.0], dtype=np.float64)
    reference_events: list[tuple[int, int, float]] = []
    optimized_events: list[tuple[int, int, float]] = []
    reference = fsl_diagonal_pcg(
        matrix,
        right_hand_side,
        tolerance=1.0e-14,
        sparse_backend="reference",
        progress=lambda *event: reference_events.append(event),
    )
    optimized = fsl_diagonal_pcg(
        matrix,
        right_hand_side,
        tolerance=1.0e-14,
        sparse_backend="optimized",
        progress=lambda *event: optimized_events.append(event),
    )

    assert np.array_equal(optimized.solution, reference.solution)
    assert optimized.iterations == reference.iterations
    assert optimized.relative_residual == reference.relative_residual
    assert optimized.converged == reference.converged
    assert reference_events[-1] == (
        reference.iterations,
        500,
        reference.relative_residual,
    )
    assert optimized_events[-1] == (
        optimized.iterations,
        500,
        optimized.relative_residual,
    )


def test_sparse_pcg_progress_polls_a_running_kernel(monkeypatch) -> None:
    started = Event()
    release = Event()
    events: list[tuple[int, int, float]] = []

    def fake_kernel(
        _data,
        _indices,
        _indptr,
        right_hand_side,
        _diagonal,
        _tolerance,
        _max_iterations,
        progress_iteration,
        progress_residual,
    ):
        progress_iteration[0] = 1
        progress_residual[0] = 0.5
        started.set()
        assert release.wait(timeout=2.0)
        return np.zeros_like(right_hand_side), 1, 0.5, False

    def release_worker(_seconds: float) -> None:
        assert started.wait(timeout=2.0)
        release.set()

    monkeypatch.setattr(optimizer_module, "_fsl_diagonal_pcg_csr", fake_kernel)
    monkeypatch.setattr(optimizer_module.time, "sleep", release_worker)
    result = fsl_diagonal_pcg(
        csc_matrix(np.eye(2)),
        np.ones(2),
        progress=lambda *event: events.append(event),
    )

    assert result.iterations == 1
    assert events == [(1, 500, 0.5), (1, 500, 0.5)]


def test_levenberg_marquardt_matches_fsl_acceptance_trajectory() -> None:
    target = np.asarray([2.0, -1.0])
    weights = np.asarray([3.0, 5.0])

    def cost(parameters: np.ndarray) -> float:
        residual = parameters - target
        return float(0.5 * np.dot(weights * residual, residual))

    def gradient(parameters: np.ndarray) -> np.ndarray:
        return weights * (parameters - target)

    def hessian(_parameters: np.ndarray) -> csc_matrix:
        return csc_matrix(np.diag(weights))

    result = fsl_levenberg_marquardt(
        np.zeros(2), cost, gradient, hessian, max_iterations=5
    )

    assert result.status == "maximum_iterations"
    assert result.iterations == 5
    assert len(result.trace) == 5
    assert all(item.accepted for item in result.trace)
    np.testing.assert_allclose(
        [item.damping for item in result.trace],
        [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5],
    )
    assert result.cost < 1.0e-24


def test_levenberg_marquardt_reports_detailed_progress() -> None:
    events: list[tuple[str, int, int, float | None]] = []
    result = fsl_levenberg_marquardt(
        np.asarray([1.0]),
        lambda parameters: float(parameters[0] ** 2),
        lambda parameters: 2.0 * parameters,
        lambda _parameters: csc_matrix([[2.0]]),
        max_iterations=1,
        progress=lambda *event: events.append(event),
    )

    assert result.iterations == 1
    phases = [event[0] for event in events]
    assert phases[:2] == ["gradient", "hessian"]
    assert phases[-1] == "lm"
    assert phases[2:-1] and set(phases[2:-1]) == {"pcg"}
    assert events[-1][3] == result.trace[-1].attempted_cost


def test_levenberg_marquardt_rejections_do_not_consume_iterations() -> None:
    calls = 0

    def cost(parameters: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return float(parameters[0] ** 2)

    result = fsl_levenberg_marquardt(
        np.asarray([1.0]),
        cost,
        lambda parameters: np.asarray([2.0 * parameters[0]]),
        lambda _parameters: np.asarray([[-1.0]]),
        max_iterations=1,
        damping_limit=1.0,
    )

    assert result.status == "damping_converged"
    assert result.iterations == 1
    assert len(result.trace) == 2
    assert not any(item.accepted for item in result.trace)
    assert calls == result.function_evaluations


def test_scaled_conjugate_gradient_matches_quadratic_solution() -> None:
    matrix = np.asarray([[5.0, 1.0], [1.0, 3.0]])
    rhs = np.asarray([2.0, -4.0])

    def cost(parameters: np.ndarray) -> float:
        return float(0.5 * parameters @ matrix @ parameters - rhs @ parameters + 5.0)

    def gradient(parameters: np.ndarray) -> np.ndarray:
        return matrix @ parameters - rhs

    result = fsl_scaled_conjugate_gradient(
        np.zeros(2), cost, gradient, max_iterations=20
    )

    np.testing.assert_allclose(result.parameters, np.linalg.solve(matrix, rhs), atol=1e-8)
    assert result.status == "gradient_converged"
    assert any(item.accepted for item in result.trace)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: fsl_diagonal_pcg(np.eye(2), np.ones(3)), "matrix must be square"),
        (
            lambda: fsl_diagonal_pcg(np.diag([1.0, 0.0]), np.ones(2)),
            "zero",
        ),
    ],
)
def test_optimizer_contract_errors(call: object, message: str) -> None:
    with pytest.raises((ValueError, np.linalg.LinAlgError), match=message):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: fsl_diagonal_pcg(np.eye(2), np.ones((1, 2))), "one-dimensional"),
        (lambda: fsl_diagonal_pcg(np.eye(2), np.asarray([1.0, np.nan])), "finite"),
        (lambda: fsl_diagonal_pcg(np.eye(2), np.ones(2), tolerance=0.0), "tolerance"),
        (lambda: fsl_diagonal_pcg(np.eye(2), np.ones(2), max_iterations=0), "max_iterations"),
        (lambda: fsl_diagonal_pcg(np.eye(2), np.ones(2), dense_backend="bad"), "backend"),
        (lambda: fsl_diagonal_pcg(np.eye(2), np.ones(2), sparse_backend="bad"), "backend"),
        (
            lambda: fsl_levenberg_marquardt(
                np.ones(1), lambda _x: np.nan, lambda x: x, lambda _x: np.eye(1), max_iterations=1
            ),
            "cost function",
        ),
        (
            lambda: fsl_scaled_conjugate_gradient(
                np.ones(1), lambda x: float(x[0] ** 2), lambda _x: np.ones(2), max_iterations=1
            ),
            "gradient shape",
        ),
        (
            lambda: fsl_scaled_conjugate_gradient(
                np.ones(1), lambda x: float(x[0] ** 2), lambda _x: np.asarray([np.nan]), max_iterations=1
            ),
            "gradient returned",
        ),
        (
            lambda: fsl_levenberg_marquardt(
                np.ones(1), lambda x: float(x[0] ** 2), lambda x: x, lambda _x: np.eye(2), max_iterations=1
            ),
            "Hessian must match",
        ),
        (
            lambda: fsl_levenberg_marquardt(
                np.ones(1), lambda x: float(x[0] ** 2), lambda x: x, lambda _x: np.asarray([[np.nan]]), max_iterations=1
            ),
            "Hessian must be finite",
        ),
        (
            lambda: fsl_levenberg_marquardt(
                np.ones(1), lambda x: float(x[0] ** 2), lambda x: x, lambda _x: csc_matrix([[np.nan]]), max_iterations=1
            ),
            "Hessian contains",
        ),
        (
            lambda: fsl_levenberg_marquardt(
                np.ones(1), lambda x: float(x[0] ** 2), lambda x: x, lambda _x: np.eye(1), max_iterations=-1
            ),
            "max_iterations",
        ),
        (
            lambda: fsl_scaled_conjugate_gradient(
                np.ones(1), lambda x: float(x[0] ** 2), lambda x: x, max_iterations=-1
            ),
            "max_iterations",
        ),
    ),
)
def test_all_optimizer_validation_paths(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_pcg_zero_rhs_bad_operator_and_lm_inversion_failure() -> None:
    zero = fsl_diagonal_pcg(csc_matrix(np.eye(2)), np.zeros(2))
    assert zero.iterations == 0
    assert zero.converged

    bad_diagonal = SimpleNamespace(
        shape=(2, 2),
        diagonal=lambda: np.asarray([1.0, np.nan]),
    )
    with pytest.raises(ValueError, match="matrix diagonal"):
        fsl_diagonal_pcg(bad_diagonal, np.ones(2))

    class SingularOperator:
        shape = (2, 2)

        @staticmethod
        def diagonal() -> np.ndarray:
            return np.ones(2)

        def __matmul__(self, vector: np.ndarray) -> np.ndarray:
            return np.zeros_like(vector)

    singular = SingularOperator()
    with pytest.raises(np.linalg.LinAlgError, match="singular search direction"):
        fsl_diagonal_pcg(singular, np.ones(2))

    result = fsl_levenberg_marquardt(
        np.ones(1),
        lambda x: float(x[0] ** 2),
        lambda x: 2.0 * x,
        lambda _x: np.zeros((1, 1)),
        max_iterations=1,
        damping_limit=0.5,
    )
    assert result.status == "damping_converged"

    converged = fsl_levenberg_marquardt(
        np.ones(1),
        lambda x: float(x[0] ** 2),
        lambda x: 2.0 * x,
        lambda _x: np.asarray([[2.0]]),
        max_iterations=3,
        cost_tolerance=10.0,
    )
    assert converged.status == "cost_converged"


def test_scaled_conjugate_gradient_zero_rejected_and_negative_curvature_paths() -> None:
    stationary = fsl_scaled_conjugate_gradient(
        np.zeros(1), lambda _x: 1.0, lambda _x: np.zeros(1), max_iterations=2
    )
    assert stationary.status == "gradient_converged"

    rejected = fsl_scaled_conjugate_gradient(
        np.asarray([1.0]),
        lambda x: float((x[0] - 1.0) ** 2),
        lambda _x: np.asarray([1.0]),
        max_iterations=1,
        gradient_tolerance=0.0,
    )
    assert not rejected.trace[0].accepted

    concave = fsl_scaled_conjugate_gradient(
        np.asarray([1.0]),
        lambda x: float(5.0 - x[0] ** 2),
        lambda x: -2.0 * x,
        max_iterations=1,
        gradient_tolerance=0.0,
    )
    assert concave.iterations == 1
