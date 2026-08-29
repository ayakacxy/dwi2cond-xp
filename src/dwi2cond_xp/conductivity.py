"""Pure NumPy conversion from diffusion to SimNIBS-style conductivity tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _sorted_eigensystem(tensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric tensor eigensystem ordered by descending eigenvalue."""
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    return eigenvalues[:, ::-1], eigenvectors[:, :, ::-1]


def _form_tensors(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Reconstruct tensors from eigenvectors stored by column."""
    return np.einsum(
        "nij,nj,nkj->nik", eigenvectors, eigenvalues, eigenvectors, optimize=True
    )


def _fix_eigenvalues(
    eigenvalues: np.ndarray,
    max_value: float,
    max_ratio: float,
    fallback: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply SimNIBS-compatible negativity, ceiling, and anisotropy corrections."""
    values = np.asarray(eigenvalues, dtype=np.float64).copy()
    invalid = np.all(values <= 0.0, axis=1) | np.all(np.isclose(values, 0), axis=1)
    values[invalid] = fallback
    large = values > max_value
    values[large] = max_value
    small = values < (values[:, 0] / max_ratio)[:, None]
    values[small[:, 1], 1] = values[small[:, 1], 0] / max_ratio
    values[small[:, 2], 2] = values[small[:, 2], 0] / max_ratio
    return values, {
        "negative_semidefinite_tensors": int(np.count_nonzero(invalid)),
        "capped_eigenvalues": int(np.count_nonzero(large)),
        "raised_eigenvalues": int(np.count_nonzero(small)),
    }


def _adjust_excentricity(eigenvalues: np.ndarray, scaling: float) -> np.ndarray:
    """Apply SimNIBS eccentricity scaling while preserving the determinant."""
    if not 0.0 <= scaling < 1.0:
        raise ValueError("excentricity scaling must be in [0, 1)")
    if np.any(eigenvalues < 0) or np.any(np.isclose(eigenvalues, 0)):
        raise ValueError("excentricity scaling requires strictly positive eigenvalues")
    excentricity = np.sqrt(
        1.0
        - (eigenvalues[:, [1, 2, 2]] / eigenvalues[:, [0, 0, 1]]) ** 2
    )
    if scaling < 0.5:
        scaled = 2.0 * excentricity * scaling
    elif scaling > 0.5:
        scaled = 2.0 * (1.0 - excentricity) * scaling + 2.0 * excentricity - 1.0
    else:
        scaled = excentricity
    result = np.ones_like(eigenvalues)
    result[:, 1] = np.sqrt(1.0 - scaled[:, 0] ** 2)
    result[:, 2] = np.sqrt(1.0 - scaled[:, 1] ** 2)
    result *= (
        np.prod(eigenvalues, axis=1) / np.prod(result, axis=1)
    )[:, None] ** (1.0 / 3.0)
    isotropic = np.isclose(eigenvalues[:, 0], eigenvalues[:, 2], rtol=1e-2)
    result[isotropic] = eigenvalues[isotropic]
    return result


def _anisotropic_intensity_scale(
    mean_determinant: Mapping[int, float],
    conductivities: Mapping[int, float],
) -> float:
    """Compute the intensity-correction scale from each tissue's mean determinant and reject degenerate inputs."""

    numerator = 0.0
    denominator = 0.0
    for tag, determinant in mean_determinant.items():
        root = determinant ** (1.0 / 3.0)
        numerator += conductivities[tag] * root
        denominator += root * root
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Anisotropic intensity denominator must be positive and finite")
    intensity_scale = numerator / denominator
    if not np.isfinite(intensity_scale) or intensity_scale <= 0.0:
        raise ValueError("Anisotropic intensity scale must be positive and finite")
    return float(intensity_scale)


def correct_fsl_tensor_basis(tensors: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert FSL tensor component directions to SimNIBS mesh world directions."""
    tensors = np.asarray(tensors, dtype=np.float64)
    affine = np.asarray(affine, dtype=np.float64)
    if tensors.shape[-2:] != (3, 3) or affine.shape != (4, 4):
        raise ValueError("tensors must be Nx3x3 and affine must be 4x4")
    linear = affine[:3, :3]
    norms = np.linalg.norm(linear, axis=0)
    if np.any(norms == 0):
        raise ValueError("affine contains a zero-length spatial axis")
    orientation = linear / norms[:, None]
    reflection = np.eye(3)
    if np.linalg.det(orientation) > 0:
        reflection[0, 0] = -1
    orientation = orientation @ reflection
    return np.einsum(
        "ij,njk,lk->nil", orientation, tensors, orientation, optimize=True
    )


def tensors_to_conductivity(
    tensors: np.ndarray,
    tissue_tags: np.ndarray,
    scalar_conductivity: Mapping[int, float] | Sequence[float],
    *,
    mode: str = "vn",
    anisotropic_tissues: Sequence[int] = (1, 2),
    weights: np.ndarray | None = None,
    max_ratio: float = 10.0,
    max_cond: float = 2.0,
    excentricity_scaling: float | None = None,
    correct_intensity: bool = True,
    vn_singular_policy: str = "error",
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert element- or voxel-sampled diffusion tensors to conductivity.

    ``mode`` supports SimNIBS ``dir``, ``vn``, and ``mc``. On a mesh,
    ``weights`` should contain tetrahedron volumes; voxel previews may omit
    them and use equal weights.
    """
    tensors = np.asarray(tensors, dtype=np.float64)
    tags = np.asarray(tissue_tags).reshape(-1)
    if tensors.shape != (tags.size, 3, 3):
        raise ValueError("tensors must be Nx3x3 and N must match tissue_tags")
    if mode not in {"dir", "vn", "mc"}:
        raise ValueError("mode must be dir, vn, or mc")
    if vn_singular_policy not in {"error", "regularize"}:
        raise ValueError("vn_singular_policy must be error or regularize")
    if mode != "vn" and vn_singular_policy != "error":
        raise ValueError("vn_singular_policy is only consumed by vn mode")
    if max_ratio < 1 or max_cond <= 0:
        raise ValueError("max_ratio must be >= 1 and max_cond must be > 0")
    element_weights = (
        np.ones(tags.size, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if element_weights.shape != tags.shape or np.any(element_weights <= 0):
        raise ValueError("weights must match the labels and be strictly positive")

    def conductivity_for(tag: int) -> float:
        value = (
            scalar_conductivity[tag]
            if isinstance(scalar_conductivity, Mapping)
            else scalar_conductivity[tag - 1]
        )
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Scalar conductivity for tissue {tag} must be finite and positive")
        return value

    output = np.zeros_like(tensors)
    anisotropic = set(int(tag) for tag in anisotropic_tissues)
    reports: dict[str, object] = {
        "mode": mode,
        "tissues": {},
        "vn_singular_policy": vn_singular_policy,
    }
    pending: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    mean_determinant: dict[int, float] = {}
    for tag in np.unique(tags):
        tag_int = int(tag)
        indices = np.flatnonzero(tags == tag)
        conductivity = conductivity_for(tag_int)
        if tag_int not in anisotropic:
            output[indices] = conductivity * np.eye(3)
            continue
        tissue_tensors = tensors[indices].copy()
        zero = np.all(np.isclose(tissue_tensors.reshape(-1, 9), 0), axis=1)
        if mode == "vn":
            tissue_tensors[zero] = conductivity * np.eye(3)
        eigenvalues, eigenvectors = _sorted_eigensystem(tissue_tensors)

        if mode == "vn":
            determinant_scale = np.abs(np.prod(eigenvalues, axis=1)) ** (1.0 / 3.0)
            singular = ~np.isfinite(determinant_scale) | (determinant_scale == 0.0)
            regularization = None
            if np.any(singular):
                if vn_singular_policy == "error":
                    raise ValueError(
                        f"VN determinant normalization is undefined for "
                        f"{int(np.count_nonzero(singular))} nonzero singular tensor(s) "
                        f"in tissue {tag_int}; use vn_singular_policy=regularize "
                        "for the explicit anisotropy-bound projection"
                    )
                eigenvalues[singular], regularization = _fix_eigenvalues(
                    eigenvalues[singular],
                    np.inf,
                    max_ratio,
                    conductivity,
                )
                determinant_scale = np.abs(np.prod(eigenvalues, axis=1)) ** (1.0 / 3.0)
                if np.any(~np.isfinite(determinant_scale)) or np.any(
                    determinant_scale <= 0.0
                ):
                    raise ValueError("VN singular-tensor regularization did not produce a positive determinant")
            eigenvalues /= determinant_scale[:, None]
            eigenvalues, first = _fix_eigenvalues(
                eigenvalues, max_cond, max_ratio, conductivity
            )
            eigenvalues /= np.prod(eigenvalues, axis=1)[:, None] ** (1.0 / 3.0)
            eigenvalues, second = _fix_eigenvalues(
                eigenvalues, max_cond, max_ratio, conductivity
            )
            if excentricity_scaling is not None:
                eigenvalues = _adjust_excentricity(eigenvalues, excentricity_scaling)
            output[indices] = _form_tensors(eigenvalues, eigenvectors) * conductivity
            reports["tissues"][str(tag_int)] = {
                "elements": int(indices.size),
                "zero_tensors": int(np.count_nonzero(zero)),
                "regularized_singular_tensors": int(np.count_nonzero(singular)),
                "singular_regularization_fix": regularization,
                "first_fix": first,
                "second_fix": second,
            }
            continue

        if correct_intensity:
            eigenvalues, first = _fix_eigenvalues(
                eigenvalues, 1e10, max_ratio, -1e-6
            )
            determinant = np.prod(eigenvalues, axis=1)
            current_mean_determinant = float(
                np.sum(determinant * element_weights[indices])
                / np.sum(element_weights[indices])
            )
            if not np.isfinite(current_mean_determinant) or current_mean_determinant <= 0.0:
                raise ValueError(
                    f"Anisotropic tissue {tag_int} has no positive finite mean determinant"
                )
            mean_determinant[tag_int] = current_mean_determinant
            pending[tag_int] = (indices, eigenvalues, eigenvectors)
            reports["tissues"][str(tag_int)] = {
                "elements": int(indices.size),
                "zero_tensors": int(np.count_nonzero(zero)),
                "first_fix": first,
            }
        else:
            eigenvalues, fixed = _fix_eigenvalues(
                eigenvalues, max_cond, max_ratio, conductivity
            )
            scaling = 0.0 if mode == "mc" else excentricity_scaling
            if scaling is not None:
                eigenvalues = _adjust_excentricity(eigenvalues, scaling)
            reconstructed = _form_tensors(eigenvalues, eigenvectors)
            reconstructed[zero] = conductivity * np.eye(3)
            output[indices] = reconstructed
            reports["tissues"][str(tag_int)] = {
                "elements": int(indices.size),
                "zero_tensors": int(np.count_nonzero(zero)),
                "fix": fixed,
            }

    if pending:
        intensity_scale = _anisotropic_intensity_scale(
            mean_determinant,
            {tag: conductivity_for(tag) for tag in mean_determinant},
        )
        reports["intensity_scale"] = float(intensity_scale)
        for tag, (indices, eigenvalues, eigenvectors) in pending.items():
            conductivity = conductivity_for(tag)
            eigenvalues, fixed = _fix_eigenvalues(
                intensity_scale * eigenvalues, max_cond, max_ratio, conductivity
            )
            scaling = 0.0 if mode == "mc" else excentricity_scaling
            if scaling is not None:
                eigenvalues = _adjust_excentricity(eigenvalues, scaling)
            reconstructed = _form_tensors(eigenvalues, eigenvectors)
            reconstructed[
                np.all(np.isclose(tensors[indices].reshape(-1, 9), 0), axis=1)
            ] = conductivity * np.eye(3)
            output[indices] = reconstructed
            reports["tissues"][str(tag)]["second_fix"] = fixed
            reports["tissues"][str(tag)]["mean_conductivity"] = float(
                np.sum(
                    np.prod(eigenvalues, axis=1) ** (1.0 / 3.0)
                    * element_weights[indices]
                )
                / np.sum(element_weights[indices])
            )
    return output, reports
