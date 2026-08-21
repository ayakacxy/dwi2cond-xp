"""Diffusion-gradient input and single-shell selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
