"""Finite 4D NIfTI operations used by the raw-DWI preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from scipy.ndimage import convolve1d


ProgressCallback = Callable[[int, int], None]


def _as_fsl_float(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim not in (3, 4):
        raise ValueError("FSL-compatible image operations require a 3D or 4D array")
    return result


def _compatible_operand(values: np.ndarray, operand: np.ndarray | float) -> np.ndarray:
    other = np.asarray(operand, dtype=np.float32)
    if other.ndim == 0:
        return other
    if values.ndim == 4 and other.shape == values.shape[:3]:
        return other[..., None]
    if other.shape != values.shape:
        raise ValueError("Image operands must have compatible 3D or 4D shapes")
    return other


def lower_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Zero values below the inclusive FSL ``-thr`` boundary."""

    result = _as_fsl_float(values).copy()
    result[np.isnan(result) | (result < np.float32(threshold))] = 0.0
    return result


def upper_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Zero values above the inclusive FSL ``-uthr`` boundary."""

    result = _as_fsl_float(values).copy()
    result[np.isnan(result) | (result > np.float32(threshold))] = 0.0
    return result


def binarize_positive(values: np.ndarray) -> np.ndarray:
    """Return the FSL ``-bin`` mask defined by strictly positive values."""

    return (_as_fsl_float(values) > 0.0).astype(np.float32)


def apply_positive_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the FSL ``-mas`` rule using ``mask > 0``."""

    source = _as_fsl_float(values)
    compatible = _compatible_operand(source, mask)
    return np.multiply(source, compatible > 0.0, dtype=np.float32)


def multiply(values: np.ndarray, operand: np.ndarray | float) -> np.ndarray:
    """Apply FSL-compatible scalar or image multiplication."""

    source = _as_fsl_float(values)
    return np.multiply(source, _compatible_operand(source, operand), dtype=np.float32)


def subtract(values: np.ndarray, operand: np.ndarray | float) -> np.ndarray:
    """Apply FSL-compatible scalar or image subtraction."""

    source = _as_fsl_float(values)
    return np.subtract(source, _compatible_operand(source, operand), dtype=np.float32)


def time_mean(values: np.ndarray) -> np.ndarray:
    """Reduce a 4D image with the FSL ``-Tmean`` float accumulation contract."""

    source = _as_fsl_float(values)
    if source.ndim != 4:
        raise ValueError("Time mean requires a four-dimensional image")
    return np.mean(source, axis=3, dtype=np.float64).astype(np.float32)


def merge_time(images: list[np.ndarray]) -> np.ndarray:
    """Merge 3D or 4D images along the time axis like ``fslmerge -t``."""

    if not images:
        raise ValueError("At least one image is required for time merging")
    prepared = []
    spatial_shape: tuple[int, ...] | None = None
    for image in images:
        values = _as_fsl_float(image)
        current_shape = values.shape[:3]
        if spatial_shape is None:
            spatial_shape = current_shape
        elif current_shape != spatial_shape:
            raise ValueError("Merged images must have identical spatial shapes")
        prepared.append(values[..., None] if values.ndim == 3 else values)
    return np.concatenate(prepared, axis=3, dtype=np.float32)


def extract_roi(
    values: np.ndarray,
    starts: tuple[int, ...],
    sizes: tuple[int, ...],
) -> np.ndarray:
    """Extract an FSL-style ROI where a size of ``-1`` means to the axis end."""

    source = _as_fsl_float(values)
    if len(starts) != source.ndim or len(sizes) != source.ndim:
        raise ValueError("ROI starts and sizes must match the image dimensions")
    slices = []
    for axis, (start, size) in enumerate(zip(starts, sizes)):
        if start < 0 or start >= source.shape[axis] or size == 0 or size < -1:
            raise ValueError("ROI bounds must select a nonempty in-image region")
        stop = source.shape[axis] if size == -1 else start + size
        if stop > source.shape[axis]:
            raise ValueError("ROI bounds must select a nonempty in-image region")
        slices.append(slice(start, stop))
    result = source[tuple(slices)].copy()
    return result[..., 0] if result.ndim == 4 and result.shape[3] == 1 else result


def masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    """Return the FSL ``-P`` rank percentile over masked nonzero voxels."""

    source = _as_fsl_float(values)
    compatible = _compatible_operand(source, mask)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("The percentile must be between 0 and 100")
    selected = source[
        np.broadcast_to(compatible >= 0.5, source.shape) & (source != 0.0)
    ]
    if selected.size == 0:
        raise ValueError("The percentile mask does not select any voxels")
    ordered = np.sort(selected, axis=None)
    index = min(int(ordered.size * percentile / 100.0), ordered.size - 1)
    return float(ordered[index])


def image_dimensions(values: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``dim1`` through ``dim4`` with an implicit singleton time axis."""

    source = _as_fsl_float(values)
    return (*source.shape[:3], 1 if source.ndim == 3 else source.shape[3])


def median_filter_box(values: np.ndarray) -> np.ndarray:
    """Apply the default FSL 3x3x3 ``-fmedian`` rank rule."""

    source = _as_fsl_float(values)
    volumes = source[..., None] if source.ndim == 3 else source
    result = np.empty_like(volumes)
    axis_counts = []
    for size in source.shape[:3]:
        counts = np.full(size, 3, dtype=np.uint8)
        if size == 1:
            counts[0] = 1
        else:
            counts[0] = 2
            counts[-1] = 2
        axis_counts.append(counts)
    counts = (
        axis_counts[0][:, None, None]
        * axis_counts[1][None, :, None]
        * axis_counts[2][None, None, :]
    )
    ranks = np.unique(counts // 2)
    for timepoint in range(volumes.shape[3]):
        current = volumes[..., timepoint]
        padded = np.pad(current, 1, mode="constant", constant_values=np.inf)
        neighbours = np.stack(
            [
                padded[
                    x_offset : x_offset + current.shape[0],
                    y_offset : y_offset + current.shape[1],
                    z_offset : z_offset + current.shape[2],
                ]
                for x_offset in range(3)
                for y_offset in range(3)
                for z_offset in range(3)
            ],
            axis=0,
        )
        neighbours.partition(ranks, axis=0)
        for rank in ranks:
            selected = counts // 2 == rank
            result[..., timepoint][selected] = neighbours[rank][selected]
    return result[..., 0] if source.ndim == 3 else result


def gaussian_smooth(
    values: np.ndarray,
    sigma_mm: float,
    voxel_sizes: tuple[float, float, float],
) -> np.ndarray:
    """Apply FSL ``-s`` separable Gaussian smoothing with edge renormalization."""

    source = _as_fsl_float(values)
    spacing = np.asarray(voxel_sizes, dtype=np.float64)
    if sigma_mm <= 0 or spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Gaussian sigma and three voxel sizes must be positive")
    result = source.copy()
    for axis in range(3):
        radius = int(np.ceil(sigma_mm * 4.0 / spacing[axis]))
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(
            -((offsets * spacing[axis]) ** 2) / (2.0 * sigma_mm * sigma_mm)
        ).astype(np.float32)
        numerator = convolve1d(result, weights, axis=axis, mode="constant", cval=0.0)
        denominator = convolve1d(
            np.ones(result.shape, dtype=np.float32),
            weights,
            axis=axis,
            mode="constant",
            cval=0.0,
        )
        result = np.divide(numerator, denominator, dtype=np.float32)
    return result


def edge_strength(
    values: np.ndarray, voxel_sizes: tuple[float, float, float]
) -> np.ndarray:
    """Apply the central-difference FSL ``-edge`` operation."""

    source = _as_fsl_float(values)
    spacing = np.asarray(voxel_sizes, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Edge strength requires three positive voxel sizes")
    result = source.copy()
    scale = np.float32(2.0 * np.sqrt(np.sum(1.0 / (spacing * spacing))))
    suffix = (slice(None),) if source.ndim == 4 else ()
    if source.shape[2] > 2:
        interior = (slice(1, -1), slice(1, -1), slice(1, -1), *suffix)
        dz = (source[1:-1, 1:-1, 2:, ...] - source[1:-1, 1:-1, :-2, ...]) / spacing[2]
        dy = (source[1:-1, 2:, 1:-1, ...] - source[1:-1, :-2, 1:-1, ...]) / spacing[1]
        dx = (source[2:, 1:-1, 1:-1, ...] - source[:-2, 1:-1, 1:-1, ...]) / spacing[0]
        result[interior] = np.sqrt(dx * dx + dy * dy + dz * dz) / scale
    else:
        interior = (slice(1, -1), slice(1, -1), slice(None), *suffix)
        dy = (source[1:-1, 2:, ...] - source[1:-1, :-2, ...]) / spacing[1]
        dx = (source[2:, 1:-1, ...] - source[:-2, 1:-1, ...]) / spacing[0]
        result[interior] = np.sqrt(dx * dx + dy * dy) / scale
    return result


def select_b0_indices(bvals: np.ndarray, *, threshold: float = 50.0) -> np.ndarray:
    """Select b0 volumes using an explicit inclusive threshold."""

    values = np.asarray(bvals, dtype=np.float64).reshape(-1)
    if threshold < 0:
        raise ValueError("The b0 threshold must be nonnegative")
    if not np.all(np.isfinite(values)):
        raise ValueError("bvals contain NaN or Inf")
    selected = np.flatnonzero(values <= threshold)
    if selected.size == 0:
        raise ValueError("No b0 volume was found")
    return selected


def read_dwi_z_block(
    image: nib.spatialimages.SpatialImage,
    z_start: int,
    z_stop: int,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    """Read one contiguous float32 DWI z-block, optionally clipping negatives."""

    if len(image.shape) != 4:
        raise ValueError("DWI data must be a four-dimensional NIfTI")
    if z_start < 0 or z_stop <= z_start or z_stop > image.shape[2]:
        raise ValueError("Invalid DWI z-block bounds")
    block = np.ascontiguousarray(
        np.asanyarray(image.dataobj[:, :, z_start:z_stop, :], dtype=np.float32)
    )
    if nonnegative:
        np.maximum(block, 0.0, out=block)
    return block


def _save_float32_like(
    values: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_file: str | Path,
) -> Path:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(
        values.astype(np.float32, copy=False), reference.affine, header
    )
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(output))
    return output


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_float32_copy(
    input_file: str | Path, output_file: str | Path
) -> Path:
    """Copy a NIfTI through the float processing type used by ``fslmaths``."""

    image = nib.load(str(input_file))
    values = _as_fsl_float(np.asanyarray(image.dataobj))
    return _save_float32_like(values, image, output_file)


def write_unaligned_b0_mean(
    data_file: str | Path,
    bvals_file: str | Path,
    output_file: str | Path,
    *,
    b0_threshold: float = 50.0,
    z_chunk: int = 8,
    progress: ProgressCallback | None = None,
    qa_file: str | Path | None = None,
) -> Path:
    """Write the direct b0 mean without motion correction or brain extraction.

    The function reads each 4D z-block once and never splits volumes into
    temporary NIfTI files. It is intentionally named ``unaligned`` because b0
    rigid registration is a separate algorithmic stage.
    """

    if z_chunk <= 0:
        raise ValueError("z_chunk must be a positive integer")
    image = nib.load(str(data_file), mmap=True)
    if len(image.shape) != 4:
        raise ValueError("DWI data must be a four-dimensional NIfTI")
    bvals = np.asarray(np.loadtxt(bvals_file), dtype=np.float64).reshape(-1)
    if bvals.size != image.shape[3]:
        raise ValueError("The DWI fourth axis does not match bvals")
    b0_indices = select_b0_indices(bvals, threshold=b0_threshold)

    mean = np.empty(image.shape[:3], dtype=np.float32)
    nonfinite_measurements = 0
    negative_measurements = 0
    for z_start in range(0, image.shape[2], z_chunk):
        z_stop = min(z_start + z_chunk, image.shape[2])
        block = read_dwi_z_block(image, z_start, z_stop)
        nonfinite_measurements += int(np.count_nonzero(~np.isfinite(block)))
        negative_measurements += int(np.count_nonzero(block < 0))
        b0 = block[..., b0_indices]
        if not np.all(np.isfinite(b0)):
            raise ValueError("b0 volumes contain NaN or Inf")
        mean[:, :, z_start:z_stop] = np.mean(b0, axis=3, dtype=np.float64).astype(
            np.float32
        )
        if progress is not None:
            progress(z_stop, image.shape[2])

    output = _save_float32_like(mean, image, output_file)
    qa_path = (
        output.with_name(f"{output.name.removesuffix('.nii.gz')}_qa.json")
        if qa_file is None
        else Path(qa_file)
    )
    _write_json(
        qa_path,
        {
            "status": "completed",
            "operation": "unaligned_b0_mean",
            "b0_alignment": "none",
            "b0_threshold": b0_threshold,
            "b0_indices": b0_indices.tolist(),
            "b0_values": bvals[b0_indices].tolist(),
            "input_shape": list(image.shape),
            "output_shape": list(mean.shape),
            "output_dtype": "float32",
            "nonfinite_measurements": nonfinite_measurements,
            "negative_measurements": negative_measurements,
        },
    )
    return output
