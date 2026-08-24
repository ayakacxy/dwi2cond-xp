"""Finite tensor decomposition used by the dwi2cond FSL-compatible path."""

from __future__ import annotations

import numpy as np

from ..registration import tensor6_to_matrix


def decompose_tensor6(
    tensor: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    requested: tuple[str, ...] | None = None,
    return_eigenvalue_range: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], tuple[float | None, float | None]]:
    """Return FSL-style eigenvalue, eigenvector, FA, MD, and mode arrays."""

    values = np.asarray(tensor)
    if values.ndim < 1 or values.shape[-1] != 6:
        raise ValueError("The final tensor axis must contain six components")
    spatial_shape = values.shape[:-1]
    selected = np.ones(spatial_shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if selected.shape != spatial_shape:
        raise ValueError("The tensor validity mask must match the spatial shape")
    available = {
        "FA", "MD", "MO", "L1", "L2", "L3", "V1", "V2", "V3"
    }
    wanted = available if requested is None else set(requested)
    unknown = wanted - available
    if unknown:
        raise ValueError(f"Unknown tensor decomposition outputs: {sorted(unknown)}")
    flat = np.asarray(values[selected], dtype=np.float64)
    if not np.all(np.isfinite(flat)):
        raise ValueError("Selected tensor components contain NaN or Inf")
    matrices = tensor6_to_matrix(flat)
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = eigenvalues[:, ::-1]
    eigenvectors = eigenvectors[:, :, ::-1]
    # FSL writes decomposition results only when the largest eigenvalue is positive;
    # all other voxels remain zero.
    positive_l1 = eigenvalues[:, 0] > 0

    mean = np.mean(eigenvalues, axis=1)
    denominator = np.sum(eigenvalues * eigenvalues, axis=1)
    numerator = 1.5 * np.sum((eigenvalues - mean[:, None]) ** 2, axis=1)
    fa_values = np.sqrt(
        np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-10,
        )
    )
    outputs: dict[str, np.ndarray] = {}
    scalar_values: dict[str, np.ndarray] = {"FA": fa_values, "MD": mean}
    if "MO" in wanted:
        centered = eigenvalues - mean[:, None]
        e1, e2, e3 = centered[:, 2], centered[:, 1], centered[:, 0]
        mode_numerator = (
            (e1 + e2 - 2 * e3)
            * (2 * e1 - e2 - e3)
            * (e1 - 2 * e2 + e3)
        )
        mode_root = np.sqrt(
            np.maximum(
                e1 * e1 + e2 * e2 + e3 * e3 - e1 * e2 - e2 * e3 - e1 * e3,
                0.0,
            )
        )
        mode_denominator = 2.0 * mode_root**3
        scalar_values["MO"] = np.clip(
            np.divide(
                mode_numerator,
                mode_denominator,
                out=np.zeros_like(mode_numerator),
                where=mode_denominator != 0,
            ),
            -1.0,
            1.0,
        )
    for suffix, selected_values in scalar_values.items():
        if suffix not in wanted:
            continue
        output = np.zeros(spatial_shape, dtype=np.float32)
        output[selected] = np.where(positive_l1, selected_values, 0.0)
        outputs[suffix] = output
    for index in range(3):
        scalar_name = f"L{index + 1}"
        vector_name = f"V{index + 1}"
        if scalar_name in wanted:
            scalar = np.zeros(spatial_shape, dtype=np.float32)
            scalar[selected] = np.where(positive_l1, eigenvalues[:, index], 0.0)
            outputs[scalar_name] = scalar
        if vector_name in wanted:
            vector = np.zeros(spatial_shape + (3,), dtype=np.float32)
            vector[selected] = np.where(
                positive_l1[:, None], eigenvectors[:, :, index], 0.0
            )
            outputs[vector_name] = vector
    if not return_eigenvalue_range:
        return outputs
    value_range = (
        (None, None)
        if eigenvalues.size == 0
        else (float(eigenvalues.min()), float(eigenvalues.max()))
    )
    return outputs, value_range
