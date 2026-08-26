"""Diffusion-gradient input and single-shell selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _fsl_shell_groups(
    bvals: np.ndarray,
    *,
    shell_tolerance: float = 100.0,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Reproduce FSL EDDY's template, mean, and reassignment shell grouping."""

    values = np.asarray(bvals, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("b-values must be a nonempty finite array")
    if shell_tolerance <= 0.0:
        raise ValueError("shell tolerance must be positive")

    template_indices = [0]
    for index in range(1, values.size):
        if not any(
            abs(float(values[template]) - float(values[index]))
            < shell_tolerance
            for template in template_indices
        ):
            template_indices.append(index)

    means = np.empty(len(template_indices), dtype=np.float64)
    for group, template in enumerate(template_indices):
        selected = np.abs(values - values[template]) < shell_tolerance
        means[group] = float(np.mean(values[selected], dtype=np.float64))
    means.sort()

    groups = tuple(
        np.flatnonzero(np.abs(values - mean) <= shell_tolerance)
        for mean in means
    )
    assigned = sum(group.size for group in groups)
    if assigned != values.size:
        raise ValueError("FSL shell grouping found inconsistent b-values")
    return groups, means


def _is_single_fsl_shell(
    bvals: np.ndarray,
    *,
    shell_tolerance: float = 100.0,
) -> bool:
    """Return whether FSL EDDY grouping assigns every value to one shell."""

    try:
        groups, _means = _fsl_shell_groups(
            bvals, shell_tolerance=shell_tolerance
        )
    except ValueError:
        return False
    return len(groups) == 1 and groups[0].size == np.asarray(bvals).size


def load_gradients(
    bvals_file: str | Path,
    bvecs_file: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate FSL-style b-values and b-vectors."""
    bvals = np.asarray(np.loadtxt(bvals_file), dtype=np.float64).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvecs_file), dtype=np.float64)
    if bvecs.ndim != 2:
        raise ValueError("bvecs must be a two-dimensional array")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    elif bvecs.shape[1] != 3:
        raise ValueError("bvecs must be 3xN or Nx3")
    if bvals.size != bvecs.shape[0]:
        raise ValueError("bvals and bvecs contain different numbers of volumes")
    if not np.all(np.isfinite(bvals)) or not np.all(np.isfinite(bvecs)):
        raise ValueError("bvals/bvecs contain NaN or Inf")
    if np.any(bvals < 0):
        raise ValueError("bvals must not be negative")
    return bvals, bvecs


def select_dti_volumes(
    bvals: np.ndarray,
    *,
    shell: float = 1000.0,
    tolerance: float = 100.0,
    b0_threshold: float = 50.0,
) -> np.ndarray:
    """Select b=0 and one shell explicitly; never fit other shells silently."""
    bvals = np.asarray(bvals, dtype=np.float64).reshape(-1)
    if shell <= b0_threshold:
        raise ValueError("The target shell must be above the b=0 threshold")
    if tolerance <= 0 or b0_threshold < 0:
        raise ValueError("Shell tolerance must be positive and the b=0 threshold nonnegative")

    is_b0 = bvals <= b0_threshold
    is_shell = np.abs(bvals - shell) <= tolerance
    selected = np.flatnonzero(is_b0 | is_shell)
    if np.count_nonzero(is_b0) == 0:
        raise ValueError("No b=0 volume was found")
    if np.count_nonzero(is_shell) < 6:
        raise ValueError("The target shell has fewer than six directions")
    return selected


def validate_single_shell_volumes(
    bvals: np.ndarray,
    *,
    b0_threshold: float = 0.0,
    shell_tolerance: float = 100.0,
) -> np.ndarray:
    """验证官方调用者已准备好的单壳输入，并返回全部 volume。"""

    values = np.asarray(bvals, dtype=np.float64).reshape(-1)
    if b0_threshold < 0 or shell_tolerance <= 0:
        raise ValueError("b0 threshold must be nonnegative and tolerance positive")
    b0 = values <= b0_threshold
    diffusion = values > b0_threshold
    if np.count_nonzero(b0) == 0:
        raise ValueError("No b=0 volume was found")
    if np.count_nonzero(diffusion) < 6:
        raise ValueError("The single-shell input has fewer than six directions")
    shell_values = values[diffusion]
    if not _is_single_fsl_shell(
        shell_values, shell_tolerance=shell_tolerance
    ):
        raise ValueError(
            "Multishell input is not accepted implicitly; run select-shell first"
        )
    return np.arange(values.size, dtype=np.int64)
