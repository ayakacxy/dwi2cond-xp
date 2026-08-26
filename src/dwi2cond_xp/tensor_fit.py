"""FSL-compatible weighted least-squares diffusion tensor fitting."""

from __future__ import annotations

import numpy as np


def _validate_gradients(bvals: np.ndarray, bvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bvals = np.asarray(bvals, dtype=np.float64).reshape(-1)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.shape != (bvals.size, 3):
        raise ValueError("bvecs must be Nx3 and N must match bvals")
    return bvals, bvecs


def _normalize_bvecs_fsl(bvecs: np.ndarray) -> np.ndarray:
    """按 ``dtifit`` 的读取顺序单位化所有非零 b-vector。"""

    vectors = np.asarray(bvecs, dtype=np.float64).copy()
    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms != 0.0
    vectors[nonzero] /= norms[nonzero, None]
    return vectors


def _gradient_transform(grad_dev: np.ndarray) -> np.ndarray:
    """Construct I+L using the FSL grad_nonlin nine-component order."""
    grad_dev = np.asarray(grad_dev, dtype=np.float64)
    if grad_dev.ndim != 2 or grad_dev.shape[1] != 9:
        raise ValueError("grad_dev must be Vx9")
    transform = np.broadcast_to(np.eye(3), (grad_dev.shape[0], 3, 3)).copy()
    transform[:, 0, 0] += grad_dev[:, 0]
    transform[:, 0, 1] += grad_dev[:, 3]
    transform[:, 0, 2] += grad_dev[:, 6]
    transform[:, 1, 0] += grad_dev[:, 1]
    transform[:, 1, 1] += grad_dev[:, 4]
    transform[:, 1, 2] += grad_dev[:, 7]
    transform[:, 2, 0] += grad_dev[:, 2]
    transform[:, 2, 1] += grad_dev[:, 5]
    transform[:, 2, 2] += grad_dev[:, 8]
    return transform


def form_design_matrix(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    grad_dev: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the seven-column design matrix used by FSL dtifit.

    The first six columns are ``Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`` and the final
    column is ``-log(S0)``. Returns ``VxNx7`` with ``grad_dev`` and ``Nx7``
    otherwise.
    """
    bvals, bvecs = _validate_gradients(bvals, bvecs)
    bvecs = _normalize_bvecs_fsl(bvecs)
    if grad_dev is None:
        h = bvecs
        design = np.empty((bvals.size, 7), dtype=np.float64)
        scaled = bvals[:, None]
        design[:, 0] = scaled[:, 0] * h[:, 0] * h[:, 0]
        design[:, 1] = 2.0 * scaled[:, 0] * h[:, 0] * h[:, 1]
        design[:, 2] = 2.0 * scaled[:, 0] * h[:, 0] * h[:, 2]
        design[:, 3] = scaled[:, 0] * h[:, 1] * h[:, 1]
        design[:, 4] = 2.0 * scaled[:, 0] * h[:, 1] * h[:, 2]
        design[:, 5] = scaled[:, 0] * h[:, 2] * h[:, 2]
        design[:, 6] = 1.0
        return design

    transform = _gradient_transform(grad_dev)
    # 原始方向先按 dtifit 单位化；梯度非线性后的模长通过 b*g*g 自然保留。
    h = np.einsum("vij,nj->vni", transform, bvecs, optimize=True)
    scaled = bvals[None, :]
    design = np.empty((grad_dev.shape[0], bvals.size, 7), dtype=np.float64)
    design[:, :, 0] = scaled * h[:, :, 0] * h[:, :, 0]
    design[:, :, 1] = 2.0 * scaled * h[:, :, 0] * h[:, :, 1]
    design[:, :, 2] = 2.0 * scaled * h[:, :, 0] * h[:, :, 2]
    design[:, :, 3] = scaled * h[:, :, 1] * h[:, :, 1]
    design[:, :, 4] = 2.0 * scaled * h[:, :, 1] * h[:, :, 2]
    design[:, :, 5] = scaled * h[:, :, 2] * h[:, :, 2]
    design[:, :, 6] = 1.0
    return design


def _normal_equations(
    design: np.ndarray,
    weights: np.ndarray,
    log_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct batched WLS normal equations."""
    if design.ndim == 2:
        normal = np.einsum("ni,vn,nj->vij", design, weights, design, optimize=True)
        rhs = -np.einsum(
            "ni,vn,vn->vi", design, weights, log_signal, optimize=True
        )
    elif design.ndim == 3:
        normal = np.einsum("vni,vn,vnj->vij", design, weights, design, optimize=True)
        rhs = -np.einsum(
            "vni,vn,vn->vi", design, weights, log_signal, optimize=True
        )
    else:
        raise ValueError("The design matrix must be Nx7 or VxNx7")
    return normal, rhs


def _solve_wls(
    design: np.ndarray,
    weights: np.ndarray,
    log_signal: np.ndarray,
) -> np.ndarray:
    normal, rhs = _normal_equations(design, weights, log_signal)
    try:
        # NumPy 2 treats a 2-D rhs as a matrix; add a column to fix batched-vector semantics.
        return np.linalg.solve(normal, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("The DTI design matrix is singular; no fitting fallback is allowed") from exc


def fit_tensor_wls(
    signals: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    grad_dev: np.ndarray | None = None,
    *,
    compatibility_mode: str = "strict-fsl",
    return_sse: bool = False,
    return_metrics: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Fit diffusion tensors in batches using FSL 6.0.4 ``dtifit --wls`` semantics.

    ``signals`` is ``VxN``. The output order is fixed as
    ``Dxx,Dxy,Dxz,Dyy,Dyz,Dzz``. Computation uses float64; callers may cast to
    float32 when writing NIfTI to match the FSL file contract.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if compatibility_mode not in {"strict-fsl", "robust"}:
        raise ValueError("compatibility_mode must be strict-fsl or robust")
    if signals.ndim != 2:
        raise ValueError("signals must be VxN")
    bvals, bvecs = _validate_gradients(bvals, bvecs)
    if signals.shape[1] != bvals.size:
        raise ValueError("The signal-volume count does not match bvals/bvecs")
    if grad_dev is not None and np.asarray(grad_dev).shape != (signals.shape[0], 9):
        raise ValueError("grad_dev must contain the same number of voxels as signals")
    if compatibility_mode == "strict-fsl":
        if np.any(np.isinf(signals)):
            raise ValueError("strict-fsl rejects Inf because FSL dtifit aborts")
    elif not np.all(np.isfinite(signals)):
        raise ValueError("robust fitting requires finite signals")
    if compatibility_mode == "robust" and np.any(np.max(signals, axis=1) <= 0):
        raise ValueError("At least one fitted voxel has no positive signal")

    design = form_design_matrix(bvals, bvecs, grad_dev)
    positive = signals > 0
    weights = np.where(positive, signals * signals, 1.0)
    initial_log = np.zeros_like(signals)
    np.log(signals, out=initial_log, where=positive)
    initial = _solve_wls(design, weights, initial_log)

    if compatibility_mode == "strict-fsl":
        # NEWMAT MaximumAbsoluteValue and Sum skip NaN. All-NaN rows therefore
        # start at zero and are subsequently replaced by the 0.01*S0 floor.
        finite = np.isfinite(signals)
        max_signal = np.max(np.where(finite, np.abs(signals), 0.0), axis=1)
        finite_count = np.count_nonzero(finite, axis=1)
        mean_signal = np.divide(
            np.sum(np.where(finite, signals, 0.0), axis=1),
            finite_count,
            out=np.zeros(signals.shape[0], dtype=np.float64),
            where=finite_count != 0,
        )
    else:
        max_signal = np.max(np.abs(signals), axis=1)
        mean_signal = np.mean(signals, axis=1)
    max_log = 23.0 if compatibility_mode == "strict-fsl" else np.log(
        np.finfo(np.float64).max
    )
    s0 = max_signal.copy()
    valid_s0 = initial[:, 6] > -max_log
    s0[valid_s0] = np.exp(-initial[valid_s0, 6])
    s0 = np.where(s0 < mean_signal, max_signal, s0)

    floor = 0.01 * s0[:, None]
    use_signal = positive & ((signals / s0[:, None]) > 0.01)
    robust_signal = np.where(use_signal, signals, floor)
    robust_log = np.log(robust_signal)
    fitted = _solve_wls(design, weights, robust_log)
    tensor = fitted[:, :6]
    if not return_sse and not return_metrics:
        return tensor

    if design.ndim == 2:
        residual = np.einsum("ni,vi->vn", design, fitted) + robust_log
    else:
        residual = np.einsum("vni,vi->vn", design, fitted) + robust_log
    sse = np.sum(residual * residual, axis=1)
    if return_metrics:
        fitted_s0 = (
            np.exp(-fitted[:, 6])
            if compatibility_mode == "strict-fsl"
            else np.exp(np.clip(-fitted[:, 6], -max_log, max_log))
        )
        fitted_s0 = np.where(fitted_s0 < mean_signal, mean_signal, fitted_s0)
        return tensor, fitted_s0, sse
    return tensor, sse
