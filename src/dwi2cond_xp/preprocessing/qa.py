"""Aggregate the image, gradient, field, tensor, and FEM scientific QA required by P11."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from collections.abc import Callable
from typing import Mapping

import nibabel as nib
import numpy as np

from ..registration import tensor6_to_matrix


QaProgress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PipelineQaInputs:
    """List required core inputs and mode-specific optional inputs for unified QA."""

    bvals: Path
    original_bvecs: Path
    brain_mask: Path
    fa: Path
    tensor: Path
    valid_mask: Path
    dwi_brain_mask: Path | None = None
    raw_dwi: Path | None = None
    corrected_dwi: Path | None = None
    raw_registered_fa: Path | None = None
    raw_registered_sse: Path | None = None
    rotated_bvecs: Path | None = None
    sse: Path | None = None
    t1: Path | None = None
    registered_fa: Path | None = None
    v1: Path | None = None
    field_hz: Path | None = None
    jacobian: Path | None = None
    eddy_parameters: Path | None = None
    outlier_map: Path | None = None
    readout_seconds: float | None = None
    fem_manifests: Mapping[str, Path] = field(default_factory=dict)


def _load_finite(path: Path, *, ndim: int | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Read a finite NIfTI image and freeze it as a NumPy array before statistics."""

    image = nib.load(str(path), mmap=True)
    if ndim is not None and len(image.shape) != ndim:
        raise ValueError(f"{path} must be {ndim}D")
    values = np.asarray(image.dataobj)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains NaN or Inf")
    return image, values


def _save_like(values: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    """Atomically publish a float32 NIfTI while preserving reference-space metadata."""

    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    temporary = path.with_name(f".{path.name}.tmp.nii.gz")
    nib.save(image, str(temporary))
    temporary.replace(path)


def _masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Compute finite-value statistics within a mask using uniform conventions."""

    selected = np.asarray(values)[mask]
    if selected.size == 0:
        raise ValueError("QA mask selects no values")
    return {
        "count": int(selected.size),
        "min": float(np.min(selected)),
        "mean": float(np.mean(selected, dtype=np.float64)),
        "p99": float(np.quantile(selected, 0.99)),
        "max": float(np.max(selected)),
    }


def _load_bvecs(path: Path, volumes: int) -> np.ndarray:
    """Accept FSL 3xN b-vectors and reject ambiguous transposes."""

    vectors = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if vectors.shape != (3, volumes) or not np.all(np.isfinite(vectors)):
        raise ValueError(f"{path} must contain finite b-vectors with shape (3, N)")
    return vectors


def _mean_artifacts(
    path: Path,
    b0_indices: np.ndarray,
    diffusion_indices: np.ndarray,
    output: Path,
    prefix: str,
) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    """Decode DWI once and generate b0 and diffusion means without repeated gzip reads."""

    image, values = _load_finite(path, ndim=4)
    if values.shape[3] <= int(max(b0_indices.max(), diffusion_indices.max())):
        raise ValueError(f"{path} volume count does not match b-values")
    b0_mean = np.mean(values[..., b0_indices], axis=3, dtype=np.float64)
    diffusion_mean = np.mean(
        values[..., diffusion_indices], axis=3, dtype=np.float64
    )
    b0_path = output / f"{prefix}_b0_mean.nii.gz"
    diffusion_path = output / f"{prefix}_mean_dwi.nii.gz"
    _save_like(b0_mean, image, b0_path)
    _save_like(diffusion_mean, image, diffusion_path)
    return (
        {"b0_mean": str(b0_path), "mean_dwi": str(diffusion_path)},
        b0_mean,
        diffusion_mean,
    )


def _write_overlay(
    t1: np.ndarray,
    fa: np.ndarray,
    mask: np.ndarray,
    output: Path,
) -> dict[str, object]:
    """Write same-slice T1/FA overlays and record the in-mask correlation coefficient."""

    if t1.shape != fa.shape or t1.shape != mask.shape:
        raise ValueError("T1, registered FA and brain mask must share one grid")
    masked_t1 = np.asarray(t1[mask], dtype=np.float64)
    masked_fa = np.asarray(fa[mask], dtype=np.float64)
    if np.std(masked_t1) == 0.0 or np.std(masked_fa) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(masked_t1, masked_fa)[0, 1])
    slice_index = int(np.argmax(np.count_nonzero(mask, axis=(0, 1))))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.imshow(np.rot90(t1[:, :, slice_index]), cmap="gray", interpolation="nearest")
    overlay = np.ma.masked_where(
        ~np.rot90(mask[:, :, slice_index]), np.rot90(fa[:, :, slice_index])
    )
    image = axis.imshow(
        overlay,
        cmap="magma",
        alpha=0.55,
        vmin=0.0,
        vmax=max(0.8, float(np.quantile(masked_fa, 0.99))),
        interpolation="nearest",
    )
    axis.set_title(f"DTI-FA / T1 overlay, z={slice_index}")
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, label="FA")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {
        "path": str(output),
        "slice_index": slice_index,
        "masked_pearson_correlation": correlation,
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish the final QA JSON atomically."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_fem_manifest(
    path: str | Path,
    expected_mode: str,
    *,
    _tissue_cache: dict[Path, tuple[nib.Nifti1Image, np.ndarray]] | None = None,
) -> dict[str, object]:
    """Validate a real FEM manifest, mesh, and volume data under strict tissue masks."""

    if expected_mode not in ("scalar", "vn", "dir", "mc"):
        raise ValueError("expected FEM mode must be scalar, vn, dir, or mc")
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("status", "missing"))
    summary: dict[str, object] = {
        "status": status,
        "manifest": str(manifest_path),
        "completed": status == "completed",
    }
    if status != "completed":
        return summary
    if manifest.get("required_simnibs_version") != "4.6.0":
        raise ValueError(f"FEM manifest {manifest_path} does not require SimNIBS 4.6.0")
    input_contract = manifest.get("input")
    if not isinstance(input_contract, dict) or input_contract.get("mode") != expected_mode:
        raise ValueError(f"FEM manifest mode does not match {expected_mode}: {manifest_path}")
    mesh_outputs = [Path(item) for item in manifest.get("outputs", [])]
    if not mesh_outputs or any(not item.is_file() or item.stat().st_size == 0 for item in mesh_outputs):
        raise ValueError(f"FEM mesh outputs are missing or empty: {manifest_path}")
    volume_paths = [Path(item) for item in manifest.get("masked_subject_volumes", [])]
    if not volume_paths:
        raise ValueError(f"FEM manifest has no subject-volume output: {manifest_path}")
    final_tissues_path = Path(str(input_contract.get("final_tissues", ""))).resolve()
    cached_tissues = (
        None if _tissue_cache is None else _tissue_cache.get(final_tissues_path)
    )
    if cached_tissues is None:
        tissues_image, tissues = _load_finite(final_tissues_path)
        if tissues.ndim == 4 and tissues.shape[-1] == 1:
            tissues = tissues[..., 0]
        if tissues.ndim != 3:
            raise ValueError(
                "final_tissues must be 3D or trailing-singleton 4D: "
                f"{final_tissues_path}"
            )
        if _tissue_cache is not None:
            _tissue_cache[final_tissues_path] = (tissues_image, tissues)
    else:
        tissues_image, tissues = cached_tissues
    labels = tuple(int(value) for value in manifest.get("volume_tissues", ()))
    if not labels:
        raise ValueError(f"FEM manifest has no volume tissue labels: {manifest_path}")
    keep = np.isin(tissues, labels)
    volume_summaries: list[dict[str, object]] = []
    for volume_path in volume_paths:
        image, values = _load_finite(volume_path)
        if values.shape[:3] != tissues.shape or not np.allclose(
            image.affine, tissues_image.affine, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"FEM subject volume does not match final_tissues: {volume_path}")
        outside = np.asarray(values[~keep])
        max_outside = 0.0 if outside.size == 0 else float(np.max(np.abs(outside)))
        if max_outside != 0.0:
            raise ValueError(f"FEM subject volume is nonzero outside selected tissues: {volume_path}")
        volume_summaries.append(
            {
                "path": str(volume_path.resolve()),
                "shape": [int(value) for value in values.shape],
                "finite_values": int(values.size),
                "max_abs_outside_tissues": max_outside,
            }
        )
    summary.update(
        {
            "mesh_outputs": [str(item.resolve()) for item in mesh_outputs],
            "subject_volumes": volume_summaries,
            "volume_tissues": list(labels),
        }
    )
    return summary


def build_pipeline_qa(
    inputs: PipelineQaInputs,
    output_directory: str | Path,
    *,
    b0_threshold: float = 50.0,
    progress: QaProgress | None = None,
) -> dict[str, object]:
    """Generate all available P11 QA and explicitly list inapplicable mode-specific items."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bvals = np.asarray(np.loadtxt(inputs.bvals), dtype=np.float64).reshape(-1)
    if bvals.size < 2 or not np.all(np.isfinite(bvals)) or np.any(bvals < 0.0):
        raise ValueError("b-values must contain at least two finite nonnegative values")
    b0_indices = np.flatnonzero(bvals <= b0_threshold)
    diffusion_indices = np.flatnonzero(bvals > b0_threshold)
    if b0_indices.size == 0 or diffusion_indices.size == 0:
        raise ValueError("QA requires at least one b0 and one diffusion volume")
    original_bvecs = _load_bvecs(inputs.original_bvecs, bvals.size)

    _mask_image, raw_mask = _load_finite(inputs.brain_mask, ndim=3)
    mask = raw_mask != 0
    if np.count_nonzero(mask) == 0:
        raise ValueError("brain mask must contain at least one voxel")
    dwi_mask = mask
    if inputs.dwi_brain_mask is not None:
        _, raw_dwi_mask = _load_finite(inputs.dwi_brain_mask, ndim=3)
        dwi_mask = raw_dwi_mask != 0
        if np.count_nonzero(dwi_mask) == 0:
            raise ValueError("DWI brain mask must contain at least one voxel")
    _fa_image, fa_values = _load_finite(inputs.fa, ndim=3)
    _tensor_image, tensor_values = _load_finite(inputs.tensor, ndim=4)
    _, valid_values = _load_finite(inputs.valid_mask, ndim=3)
    if tensor_values.shape[-1] != 6:
        raise ValueError("tensor final axis must contain six components")
    spatial_shape = mask.shape
    if (
        fa_values.shape != spatial_shape
        or tensor_values.shape[:3] != spatial_shape
        or valid_values.shape != spatial_shape
    ):
        raise ValueError("mask, FA, tensor and valid mask must share one grid")
    valid = mask & (valid_values != 0)
    if np.count_nonzero(valid) == 0:
        raise ValueError("valid tensor mask selects no voxels")
    if progress is not None:
        progress("core_inputs", 1, 8)

    dwi_artifacts: dict[str, object] = {}
    if inputs.raw_dwi is not None:
        raw_artifacts, _raw_b0, raw_mean = _mean_artifacts(
            inputs.raw_dwi,
            b0_indices,
            diffusion_indices,
            output,
            "raw",
        )
        dwi_artifacts["raw"] = {
            "artifacts": raw_artifacts,
            "mean_dwi_stats": _masked_stats(raw_mean, dwi_mask),
        }
    else:
        dwi_artifacts["raw"] = {"status": "not_provided"}
    if progress is not None:
        progress("raw_dwi", 2, 8)
    if inputs.corrected_dwi is not None:
        corrected_artifacts, _corrected_b0, corrected_mean = _mean_artifacts(
            inputs.corrected_dwi,
            b0_indices,
            diffusion_indices,
            output,
            "corrected",
        )
        dwi_artifacts["corrected"] = {
            "artifacts": corrected_artifacts,
            "mean_dwi_stats": _masked_stats(corrected_mean, dwi_mask),
        }
    else:
        dwi_artifacts["corrected"] = {"status": "not_provided"}
    if progress is not None:
        progress("corrected_dwi", 3, 8)

    rotated_bvec_qa: dict[str, object]
    if inputs.rotated_bvecs is None:
        rotated_bvec_qa = {"status": "not_provided"}
    else:
        rotated = _load_bvecs(inputs.rotated_bvecs, bvals.size)
        original_norm = np.linalg.norm(original_bvecs[:, diffusion_indices], axis=0)
        rotated_norm = np.linalg.norm(rotated[:, diffusion_indices], axis=0)
        if np.any(original_norm == 0.0) or np.any(rotated_norm == 0.0):
            raise ValueError("diffusion b-vectors must be nonzero")
        cosine = np.sum(
            original_bvecs[:, diffusion_indices] * rotated[:, diffusion_indices],
            axis=0,
        ) / (original_norm * rotated_norm)
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        rotated_bvec_qa = {
            "status": "available",
            "mean_angle_degrees": float(np.mean(angles)),
            "p99_angle_degrees": float(np.quantile(angles, 0.99)),
            "max_angle_degrees": float(np.max(angles)),
            "max_unit_norm_error": float(np.max(np.abs(rotated_norm - 1.0))),
        }
    if progress is not None:
        progress("gradients", 4, 8)

    matrices = tensor6_to_matrix(np.asarray(tensor_values[valid], dtype=np.float64))
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    principal = eigenvectors[:, :, -1]
    v1_qa: dict[str, object] = {
        "computed_valid_vectors": int(principal.shape[0]),
        "max_norm_error": float(
            np.max(np.abs(np.linalg.norm(principal, axis=1) - 1.0))
        ),
    }
    if inputs.v1 is not None:
        _, stored_v1 = _load_finite(inputs.v1, ndim=4)
        if stored_v1.shape != spatial_shape + (3,):
            raise ValueError("V1 must share the tensor grid and have final axis 3")
        stored = np.asarray(stored_v1[valid], dtype=np.float64)
        stored_norm = np.linalg.norm(stored, axis=1)
        usable = stored_norm > 0.0
        cosine = np.abs(
            np.sum(stored[usable] * principal[usable], axis=1) / stored_norm[usable]
        )
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        v1_qa.update(
            {
                "stored_vectors": int(np.count_nonzero(usable)),
                "decomposition_axial_angle_mean_degrees": float(np.mean(angles)),
                "decomposition_axial_angle_max_degrees": float(np.max(angles)),
            }
        )
    if progress is not None:
        progress("tensor", 5, 8)

    field_qa: dict[str, object] = {"status": "not_provided"}
    if inputs.field_hz is not None:
        _, field_values = _load_finite(inputs.field_hz, ndim=3)
        field_mask = dwi_mask if field_values.shape == dwi_mask.shape else mask
        if field_values.shape != field_mask.shape:
            raise ValueError("field must share the DWI or tensor brain-mask grid")
        field_qa = {
            "status": "available",
            "field_hz": _masked_stats(field_values, field_mask),
        }
        if inputs.readout_seconds is not None:
            if not np.isfinite(inputs.readout_seconds) or inputs.readout_seconds <= 0.0:
                raise ValueError("readout_seconds must be positive and finite")
            field_qa["voxel_shift"] = _masked_stats(
                field_values * inputs.readout_seconds, field_mask
            )
    if inputs.jacobian is not None:
        _, jacobian_values = _load_finite(inputs.jacobian, ndim=3)
        if jacobian_values.shape != spatial_shape:
            raise ValueError("Jacobian and brain mask must share one grid")
        field_qa["jacobian"] = _masked_stats(jacobian_values, mask)
        field_qa["nonpositive_jacobian_voxels"] = int(
            np.count_nonzero((jacobian_values <= 0.0) & mask)
        )
        field_qa["status"] = "available"

    parameter_qa: dict[str, object] = {"status": "not_provided"}
    if inputs.eddy_parameters is not None:
        parameters = np.atleast_2d(np.loadtxt(inputs.eddy_parameters, dtype=np.float64))
        if parameters.shape[0] != bvals.size or not np.all(np.isfinite(parameters)):
            raise ValueError("eddy parameters must have one finite row per volume")
        parameter_qa = {
            "status": "available",
            "rows": int(parameters.shape[0]),
            "columns": int(parameters.shape[1]),
            "per_volume": parameters.tolist(),
        }
    if inputs.outlier_map is not None:
        outliers = np.atleast_2d(np.loadtxt(inputs.outlier_map, dtype=np.float64))
        parameter_qa["outlier_slices"] = int(np.count_nonzero(outliers))
        parameter_qa["outlier_map_shape"] = [int(value) for value in outliers.shape]
    if progress is not None:
        progress("field_motion_eddy", 6, 8)

    overlay: dict[str, object] = {"status": "not_provided"}
    if inputs.t1 is not None and inputs.registered_fa is not None:
        _, t1_values = _load_finite(inputs.t1, ndim=3)
        _, registered_fa = _load_finite(inputs.registered_fa, ndim=3)
        overlay = _write_overlay(
            t1_values,
            registered_fa,
            mask,
            output / "dti_fa_t1_overlay.png",
        )
        overlay["status"] = "available"

    sse_qa: dict[str, object] = {"status": "not_provided"}
    if inputs.sse is not None:
        _, sse_values = _load_finite(inputs.sse, ndim=3)
        if sse_values.shape != spatial_shape:
            raise ValueError("SSE and brain mask must share one grid")
        sse_qa = {"status": "available", "stats": _masked_stats(sse_values, valid)}

    raw_fit_qa: dict[str, object] = {"status": "not_provided"}
    if inputs.raw_registered_fa is not None or inputs.raw_registered_sse is not None:
        if inputs.raw_registered_fa is None or inputs.raw_registered_sse is None:
            raise ValueError("raw registered FA and SSE must be provided together")
        _, raw_fa_values = _load_finite(inputs.raw_registered_fa, ndim=3)
        _, raw_sse_values = _load_finite(inputs.raw_registered_sse, ndim=3)
        if raw_fa_values.shape != spatial_shape or raw_sse_values.shape != spatial_shape:
            raise ValueError("raw registered FA, SSE, and brain mask must share one grid")
        raw_fit_qa = {
            "status": "available",
            "fa": _masked_stats(raw_fa_values, mask),
            "sse": _masked_stats(raw_sse_values, mask),
            "fa_path": str(inputs.raw_registered_fa.resolve()),
            "sse_path": str(inputs.raw_registered_sse.resolve()),
        }

    fem: dict[str, object] = {}
    tissue_cache: dict[Path, tuple[nib.Nifti1Image, np.ndarray]] = {}
    for mode in ("scalar", "vn", "dir", "mc"):
        manifest_path = inputs.fem_manifests.get(mode)
        if manifest_path is None:
            fem[mode] = {"status": "not_provided"}
            continue
        fem[mode] = audit_fem_manifest(
            manifest_path, mode, _tissue_cache=tissue_cache
        )
    if progress is not None:
        progress("registration_fem", 7, 8)

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "b0_threshold": b0_threshold,
        "b0_indices": b0_indices.tolist(),
        "diffusion_indices": diffusion_indices.tolist(),
        "brain_mask": {
            "path": str(inputs.brain_mask.resolve()),
            "voxels": int(np.count_nonzero(mask)),
            "shape": [int(value) for value in mask.shape],
        },
        "dwi": dwi_artifacts,
        "fa": _masked_stats(fa_values, valid),
        "sse": sse_qa,
        "raw_fit": raw_fit_qa,
        "motion_eddy": parameter_qa,
        "bvec_rotation": rotated_bvec_qa,
        "susceptibility": field_qa,
        "registration_overlay": overlay,
        "tensor": {
            "component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
            "valid_voxels": int(np.count_nonzero(valid)),
            "invalid_voxels_in_brain": int(np.count_nonzero(mask & ~valid)),
            "eigenvalue_min": float(np.min(eigenvalues)),
            "eigenvalue_p01": float(np.quantile(eigenvalues, 0.01)),
            "eigenvalue_mean": float(np.mean(eigenvalues)),
            "eigenvalue_max": float(np.max(eigenvalues)),
            "nonpositive_eigenvalues": int(np.count_nonzero(eigenvalues <= 0.0)),
            "v1": v1_qa,
        },
        "fem_smoke": fem,
    }
    _atomic_json(output / "pipeline_qa.json", report)
    if progress is not None:
        progress("complete", 8, 8)
    return report


__all__ = ["PipelineQaInputs", "audit_fem_manifest", "build_pipeline_qa"]
