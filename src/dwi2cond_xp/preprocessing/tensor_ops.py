"""Finite tensor decomposition used by the dwi2cond FSL-compatible path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..registration import tensor6_to_matrix


def _eigh_batched(
    matrices: np.ndarray, workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """并行处理互相独立的张量，保留每个矩阵的 NumPy LAPACK 分解。"""

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    count = matrices.shape[0]
    if workers == 1 or count < 100_000:
        return np.linalg.eigh(matrices)
    chunk = min(262_144, max(1, (count + workers - 1) // workers))
    spans = tuple((start, min(start + chunk, count)) for start in range(0, count, chunk))
    eigenvalues = np.empty((count, 3), dtype=np.float64)
    eigenvectors = np.empty((count, 3, 3), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=min(workers, len(spans))) as executor:
        results = executor.map(
            np.linalg.eigh, (matrices[start:stop] for start, stop in spans)
        )
        for (start, stop), (values, vectors) in zip(spans, results, strict=True):
            eigenvalues[start:stop] = values
            eigenvectors[start:stop] = vectors
    return eigenvalues, eigenvectors


def _eigh_tensor6_batched(
    components: np.ndarray, workers: int = 1, *, principal_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """分块构造 float64 矩阵；线程直接写入独立输出片段，避免积压分解副本。"""

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    count = components.shape[0]
    eigenvalues = np.empty((count, 3), dtype=np.float64)
    vector_shape = (count, 3) if principal_only else (count, 3, 3)
    eigenvectors = np.empty(vector_shape, dtype=np.float64)
    chunk = min(262_144, max(1, (count + workers - 1) // workers))
    spans = tuple((start, min(start + chunk, count)) for start in range(0, count, chunk))

    def decompose(span: tuple[int, int]) -> None:
        start, stop = span
        matrices = tensor6_to_matrix(np.asarray(components[start:stop], dtype=np.float64))
        values, vectors = np.linalg.eigh(matrices)
        eigenvalues[start:stop] = values
        eigenvectors[start:stop] = vectors[:, :, -1] if principal_only else vectors

    if workers == 1 or count < 100_000:
        for span in spans:
            decompose(span)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(spans))) as executor:
            for _ in executor.map(decompose, spans):
                pass
    return eigenvalues, eigenvectors


def decompose_tensor6(
    tensor: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    semantics: str = "fslmaths",
    requested: tuple[str, ...] | None = None,
    return_eigenvalue_range: bool = False,
    workers: int = 1,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], tuple[float | None, float | None]]:
    """Decompose a six-component tensor according to ``fslmaths`` or ``dtifit`` semantics."""

    values = np.asarray(tensor)
    if values.ndim < 1 or values.shape[-1] != 6:
        raise ValueError("The final tensor axis must contain six components")
    spatial_shape = values.shape[:-1]
    selected = np.ones(spatial_shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if selected.shape != spatial_shape:
        raise ValueError("The tensor validity mask must match the spatial shape")
    if semantics not in {"fslmaths", "dtifit"}:
        raise ValueError("semantics must be fslmaths or dtifit")
    available = {
        "FA", "MD", "MO", "L1", "L2", "L3", "V1", "V2", "V3"
    }
    wanted = available if requested is None else set(requested)
    unknown = wanted - available
    if unknown:
        raise ValueError(f"Unknown tensor decomposition outputs: {sorted(unknown)}")
    flat = values[selected]
    if not np.all(np.isfinite(flat)):
        raise ValueError("Selected tensor components contain NaN or Inf")
    principal_only = not wanted.intersection({"V2", "V3"})
    eigenvalues, eigenvectors = _eigh_tensor6_batched(
        flat, workers, principal_only=principal_only,
    )
    eigenvalues = eigenvalues[:, ::-1]
    if not principal_only:
        eigenvectors = eigenvectors[:, :, ::-1]
    # fslmaths gates on the largest eigenvalue; dtifit writes fitted voxels inside the mask unconditionally.
    write_values = (
        eigenvalues[:, 0] > 0
        if semantics == "fslmaths"
        else np.ones(eigenvalues.shape[0], dtype=bool)
    )

    mean = (
        np.mean(eigenvalues, axis=1)
        if wanted.intersection({"FA", "MD", "MO"}) else None
    )
    scalar_values: dict[str, np.ndarray] = {}
    if "FA" in wanted:
        denominator = np.sum(eigenvalues * eigenvalues, axis=1)
        numerator = 1.5 * np.sum((eigenvalues - mean[:, None]) ** 2, axis=1)
        fa_denominator_threshold = 0.0 if semantics == "fslmaths" else 1e-10
        scalar_values["FA"] = np.sqrt(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > fa_denominator_threshold,
            )
        )
    if "MD" in wanted:
        scalar_values["MD"] = mean
    outputs: dict[str, np.ndarray] = {}
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
        output = np.zeros(spatial_shape, dtype=np.float32)
        output[selected] = np.where(write_values, selected_values, 0.0)
        outputs[suffix] = output
    for index in range(3):
        scalar_name = f"L{index + 1}"
        vector_name = f"V{index + 1}"
        if scalar_name in wanted:
            scalar = np.zeros(spatial_shape, dtype=np.float32)
            scalar[selected] = np.where(write_values, eigenvalues[:, index], 0.0)
            outputs[scalar_name] = scalar
        if vector_name in wanted:
            vector = np.zeros(spatial_shape + (3,), dtype=np.float32)
            vector[selected] = np.where(
                write_values[:, None],
                eigenvectors if principal_only else eigenvectors[:, :, index], 0.0,
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
