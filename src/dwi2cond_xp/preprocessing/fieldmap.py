"""Fixed SimNIBS 4.6 GRE fieldmap and FUGUE path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
from numba import njit, prange
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, convolve1d, label

from ._numba import set_available_numba_threads
from .brain_mask import bet_brain_mask
from .flirt_registration import FlirtRegistrationResult, register_flirt_affine
from .image_ops import masked_percentile, median_filter_box
from .orientation import write_fsl_reoriented
from .resampling import resample_image
from .transforms import (
    fsl_matrix_to_world,
    fsl_voxel_to_scaled_mm,
    world_matrix_to_fsl,
)


ProgressCallback = Callable[[str, int, int], None]
_DIRECTIONS = {
    "x": (0, 1),
    "x-": (0, -1),
    "y": (1, 1),
    "y-": (1, -1),
    "z": (2, 1),
    "z-": (2, -1),
}


@dataclass(frozen=True)
class FieldmapResult:
    """Store arrays, transforms, and unit contracts for the fixed P7 path."""

    field_radians_per_second: np.ndarray
    magnitude_brain: np.ndarray
    magnitude_mask: np.ndarray
    distorted_magnitude: np.ndarray
    dwi_to_fieldmap_fsl: np.ndarray
    field_dwi_radians_per_second: np.ndarray
    fieldmap_mask_dwi: np.ndarray
    voxel_shift: np.ndarray
    displacement_world_mm: np.ndarray
    corrected_b0: np.ndarray
    corrected_mask: np.ndarray
    registration: FlirtRegistrationResult


def phase_encoding_axis_sign(direction: str) -> tuple[int, int]:
    """Return the axis and sign for the six FUGUE/convertwarp strings."""

    try:
        return _DIRECTIONS[direction]
    except KeyError as error:
        raise ValueError("phase_encoding_direction must be x, x-, y, y-, z, or z-") from error


def prepare_radians_per_second(
    field: np.ndarray,
    mask: np.ndarray,
    *,
    median_filter: bool = True,
) -> tuple[np.ndarray, float]:
    """Reproduce optional ``-fmedian`` and in-mask ``-P 50`` offset removal."""

    values = np.asarray(field, dtype=np.float32)
    selected = np.asarray(mask) > 0
    if values.ndim != 3 or selected.shape != values.shape:
        raise ValueError("field and mask must be matching three-dimensional arrays")
    if not np.all(np.isfinite(values)) or not np.any(selected):
        raise ValueError("field must be finite and mask must select at least one voxel")
    filtered = median_filter_box(values) if median_filter else values.copy()
    offset = masked_percentile(filtered, selected.astype(np.float32), 50.0)
    return np.subtract(filtered, np.float32(offset), dtype=np.float32), offset


def voxel_shift_from_field(
    field_radians_per_second: np.ndarray,
    dwell_seconds: float,
    phase_encoding_direction: str,
) -> np.ndarray:
    """Generate an unsigned voxel-shift map using FSL ``fmap2pixshift_factor``."""

    field = np.asarray(field_radians_per_second, dtype=np.float32)
    axis, _ = phase_encoding_axis_sign(phase_encoding_direction)
    if field.ndim != 3 or not np.all(np.isfinite(field)):
        raise ValueError("field_radians_per_second must be a finite 3D array")
    if not np.isfinite(dwell_seconds) or dwell_seconds <= 0:
        raise ValueError("dwell_seconds must be positive and finite")
    factor = np.float32(field.shape[axis] * dwell_seconds / (2.0 * np.pi))
    return np.multiply(field, factor, dtype=np.float32)


def displacement_from_voxel_shift(
    voxel_shift: np.ndarray,
    affine: np.ndarray,
    phase_encoding_direction: str,
) -> np.ndarray:
    """Express the signed convertwarp voxel shift as a NIfTI world-mm pull displacement."""

    shift = np.asarray(voxel_shift, dtype=np.float32)
    matrix = np.asarray(affine, dtype=np.float64)
    axis, sign = phase_encoding_axis_sign(phase_encoding_direction)
    if shift.ndim != 3 or not np.all(np.isfinite(shift)):
        raise ValueError("voxel_shift must be a finite 3D array")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("affine must be a finite 4x4 matrix")
    direction = matrix[:3, axis] * float(sign)
    return np.multiply(shift[..., None], direction, dtype=np.float64)


def fill_head_mask(mask: np.ndarray) -> np.ndarray:
    """Reproduce FUGUE ``fill_head_mask`` closing and corner-background hole filling."""

    original = np.asarray(mask) > 0
    if original.ndim != 3 or not np.any(original):
        raise ValueError("mask must be a nonempty three-dimensional binary array")
    structure = np.ones((3, 3, 3), dtype=bool)
    closed = binary_erosion(
        binary_dilation(original, structure=structure, border_value=0),
        structure=structure,
        border_value=1,
    )
    background_labels, _ = label(~closed)
    corners = {
        int(background_labels[xindex, yindex, zindex])
        for xindex in (0, closed.shape[0] - 1)
        for yindex in (0, closed.shape[1] - 1)
        for zindex in (0, closed.shape[2] - 1)
    }
    filled = closed.copy()
    for component in np.unique(background_labels):
        if component != 0 and int(component) not in corners:
            filled[background_labels == component] = True
    return filled.astype(np.uint8)


def extrapolate_field_holes(
    field: np.ndarray, original_mask: np.ndarray, filled_mask: np.ndarray
) -> np.ndarray:
    """Fill in-head holes using FUGUE's nearest valid box and 0.2/0.6/0.2 smoothing."""

    values = np.asarray(field, dtype=np.float32)
    original = np.asarray(original_mask) > 0
    enlarged = np.asarray(filled_mask) > 0
    if values.ndim != 3 or original.shape != values.shape or enlarged.shape != values.shape:
        raise ValueError("field and masks must share a three-dimensional grid")
    holes = enlarged & ~original
    if not np.any(holes):
        return values.copy()
    extrapolated = values.copy()
    maximum_radius = min(size - 1 for size in values.shape)
    default = np.float32(np.median(values))
    for xindex, yindex, zindex in np.argwhere(holes):
        found = False
        for radius in range(1, maximum_radius):
            x0, x1 = max(0, xindex - radius), min(values.shape[0], xindex + radius + 1)
            y0, y1 = max(0, yindex - radius), min(values.shape[1], yindex + radius + 1)
            z0, z1 = max(0, zindex - radius), min(values.shape[2], zindex + radius + 1)
            valid = original[x0:x1, y0:y1, z0:z1] & ~holes[x0:x1, y0:y1, z0:z1]
            if np.any(valid):
                extrapolated[xindex, yindex, zindex] = np.mean(
                    values[x0:x1, y0:y1, z0:z1][valid], dtype=np.float32
                )
                found = True
                break
        if not found:
            extrapolated[xindex, yindex, zindex] = default
    kernel = np.asarray([0.2, 0.6, 0.2], dtype=np.float32)
    smoothed = extrapolated.copy()
    normalization = (original | holes).astype(np.float32)
    for axis in range(3):
        smoothed = convolve1d(smoothed, kernel, axis=axis, mode="constant", cval=0.0)
        normalization = convolve1d(
            normalization, kernel, axis=axis, mode="constant", cval=0.0
        )
    smoothed = np.divide(
        smoothed,
        normalization + np.float32(0.0001),
        dtype=np.float32,
    )
    extrapolated[holes] = smoothed[holes]
    return extrapolated


def _fsl_background_value(values: np.ndarray) -> float:
    """Return the tenth percentile over the two-layer NEWIMAGE ``backgroundval`` boundary."""

    volume = np.asarray(values, dtype=np.float32)
    edge_x = min(2, volume.shape[0] - 1)
    edge_y = min(2, volume.shape[1] - 1)
    edge_z = min(2, volume.shape[2] - 1)
    boundary: list[np.float32] = []
    for edge in range(edge_z):
        for xindex in range(edge_x, volume.shape[0] - edge_x):
            for yindex in range(edge_y, volume.shape[1] - edge_y):
                boundary.extend(
                    (volume[xindex, yindex, edge], volume[xindex, yindex, -1 - edge])
                )
    for edge in range(edge_y):
        for xindex in range(edge_x, volume.shape[0] - edge_x):
            for zindex in range(volume.shape[2]):
                boundary.extend(
                    (volume[xindex, edge, zindex], volume[xindex, -1 - edge, zindex])
                )
    for edge in range(edge_x):
        for yindex in range(volume.shape[1]):
            for zindex in range(volume.shape[2]):
                boundary.extend(
                    (volume[edge, yindex, zindex], volume[-1 - edge, yindex, zindex])
                )
    ordered = np.sort(np.asarray(boundary, dtype=np.float32))
    return float(ordered[ordered.size // 10])


def regularize_voxel_shift(
    voxel_shift: np.ndarray,
    mask: np.ndarray,
    phase_encoding_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce FUGUE's default hole fill and rigid extension along the PE axis."""

    shift = np.asarray(voxel_shift, dtype=np.float32)
    original = (np.asarray(mask) > 0).astype(np.uint8)
    if shift.ndim != 3 or original.shape != shift.shape:
        raise ValueError("voxel_shift and mask must share a three-dimensional grid")
    axis, sign = phase_encoding_axis_sign(phase_encoding_direction)
    filled = fill_head_mask(original)
    regularized = extrapolate_field_holes(shift, original, filled)
    shift_y = np.moveaxis(regularized, axis, 1)
    mask_y = np.moveaxis(filled, axis, 1)
    if sign < 0:
        shift_y = shift_y[:, ::-1, :]
        mask_y = mask_y[:, ::-1, :]
    shift_y = np.ascontiguousarray(shift_y)
    for xindex in range(shift_y.shape[0]):
        for zindex in range(shift_y.shape[2]):
            yindex = 0
            while yindex < shift_y.shape[1]:
                if mask_y[xindex, yindex, zindex] < 0.5:
                    stop = yindex + 1
                    while stop < shift_y.shape[1] and mask_y[xindex, stop, zindex] < 0.5:
                        stop += 1
                    first = yindex
                    start_value = (
                        shift_y[xindex, first - 1, zindex]
                        if first > 0
                        else np.float32(0.0)
                    )
                    if stop < shift_y.shape[1]:
                        end_value = shift_y[xindex, stop, zindex]
                        last = stop
                    else:
                        end_value = start_value
                        last = shift_y.shape[1] - 1
                    if first == 0:
                        start_value = end_value
                    width = last - first
                    if width == 0:
                        shift_y[xindex, first, zindex] = start_value
                    else:
                        for position in range(first, last + 1):
                            fraction = np.float32(position - first) / np.float32(width)
                            shift_y[xindex, position, zindex] = np.float32(
                                (end_value - start_value) * fraction + start_value
                            )
                    yindex = last
                yindex += 1
    if sign < 0:
        shift_y = shift_y[:, ::-1, :]
    return np.moveaxis(shift_y, 1, axis), filled


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _forward_warp_y(source: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Forward-warp along positive y in FUGUE image-space half-voxel allocation order."""

    output = np.zeros(source.shape, dtype=np.float32)
    line_count = source.shape[0] * source.shape[2]
    for line in prange(line_count):
        xindex = line // source.shape[2]
        zindex = line - xindex * source.shape[2]
        for yindex in range(source.shape[1]):
            center = np.float32(yindex) + shift[xindex, yindex, zindex]
            if yindex + 1 < source.shape[1]:
                next_center = (
                    np.float32(yindex + 1) + shift[xindex, yindex + 1, zindex]
                )
            else:
                next_center = center + np.float32(1.0)
            if yindex > 0:
                previous_center = (
                    np.float32(yindex - 1) + shift[xindex, yindex - 1, zindex]
                )
            else:
                previous_center = center - np.float32(1.0)
            right_edge = np.float32(0.5) * (center + next_center)
            left_edge = np.float32(0.5) * (center + previous_center)
            intensity = source[xindex, yindex, zindex]
            for target in range(source.shape[1]):
                for endpoint in range(2):
                    first = left_edge if endpoint == 0 else center
                    second = center if endpoint == 0 else right_edge
                    lower = min(first, second)
                    upper = max(first, second)
                    left = max(lower, np.float32(target) - np.float32(0.5))
                    right = min(upper, np.float32(target) + np.float32(0.5))
                    length = right - left
                    width = upper - lower
                    if length > np.float32(0.0) and width > np.float32(0.0):
                        output[xindex, target, zindex] = np.float32(
                            output[xindex, target, zindex]
                            + np.float32(0.5) * intensity * length / width
                        )
    return output


def forward_warp_magnitude(
    magnitude: np.ndarray,
    voxel_shift: np.ndarray,
    phase_encoding_direction: str,
    *,
    workers: int = 8,
) -> np.ndarray:
    """Reproduce the image-space forward warp from ``fugue -w --nokspace``."""

    source = np.asarray(magnitude, dtype=np.float32)
    shift = np.asarray(voxel_shift, dtype=np.float32)
    axis, sign = phase_encoding_axis_sign(phase_encoding_direction)
    if source.ndim != 3 or shift.shape != source.shape:
        raise ValueError("magnitude and voxel_shift must be matching 3D arrays")
    if workers < 1:
        raise ValueError("workers must be positive")
    source_y = np.moveaxis(source, axis, 1)
    shift_y = np.moveaxis(shift, axis, 1)
    if sign < 0:
        source_y = source_y[:, ::-1, :]
        shift_y = shift_y[:, ::-1, :]
    set_available_numba_threads(workers)
    warped = _forward_warp_y(
        np.ascontiguousarray(source_y), np.ascontiguousarray(shift_y)
    )
    if sign < 0:
        warped = warped[:, ::-1, :]
    return np.moveaxis(warped, 1, axis)


def run_fieldmap(
    magnitude: np.ndarray,
    field_radians_per_second: np.ndarray,
    magnitude_mask: np.ndarray,
    b0_brain: np.ndarray,
    b0_mask: np.ndarray,
    magnitude_affine: np.ndarray,
    b0_affine: np.ndarray,
    *,
    dwell_seconds: float,
    phase_encoding_direction: str,
    workers: int = 8,
    median_filter: bool = True,
    progress: ProgressCallback | None = None,
) -> FieldmapResult:
    """Execute the complete fixed GRE fieldmap path for rad/s input."""

    mag = np.asarray(magnitude, dtype=np.float32)
    mask = (np.asarray(magnitude_mask) > 0).astype(np.float32)
    b0 = np.asarray(b0_brain, dtype=np.float32)
    nodif_mask = (np.asarray(b0_mask) > 0).astype(np.uint8)
    if mag.ndim != 3 or mask.shape != mag.shape:
        raise ValueError("magnitude and magnitude_mask must share a 3D grid")
    if b0.ndim != 3 or nodif_mask.shape != b0.shape:
        raise ValueError("b0_brain and b0_mask must share a 3D grid")
    magnitude_brain = np.multiply(mag, mask, dtype=np.float32)
    field, _ = prepare_radians_per_second(
        field_radians_per_second, mask, median_filter=median_filter
    )
    shift_magnitude = voxel_shift_from_field(
        field, dwell_seconds, phase_encoding_direction
    )
    shift_magnitude, _ = regularize_voxel_shift(
        shift_magnitude, mask, phase_encoding_direction
    )
    distorted = forward_warp_magnitude(
        magnitude_brain,
        shift_magnitude,
        phase_encoding_direction,
        workers=workers,
    )
    if progress:
        progress("fieldmap_prepare", 1, 4)

    reference_sampling = fsl_voxel_to_scaled_mm(distorted.shape, magnitude_affine)
    moving_sampling = fsl_voxel_to_scaled_mm(b0.shape, b0_affine)
    qsform = world_matrix_to_fsl(
        np.eye(4), b0.shape, b0_affine, distorted.shape, magnitude_affine
    )
    registration = register_flirt_affine(
        distorted,
        b0,
        np.ones_like(distorted, dtype=np.float32),
        np.ones_like(b0, dtype=np.float32),
        reference_sampling,
        moving_sampling,
        degrees_of_freedom=6,
        qsform_matrix=qsform,
        workers=workers,
        cost_function="mutual_information",
    )
    dwi_to_fieldmap_world = fsl_matrix_to_world(
        registration.matrix,
        b0.shape,
        b0_affine,
        distorted.shape,
        magnitude_affine,
    )
    fieldmap_to_dwi_world = np.linalg.inv(dwi_to_fieldmap_world)
    if progress:
        progress("fieldmap_register", 2, 4)

    field_dwi = resample_image(
        field,
        magnitude_affine,
        b0.shape,
        b0_affine,
        fieldmap_to_dwi_world,
        interpolation="linear",
        cval=_fsl_background_value(field),
        linear_extrapolation="fsl",
    )
    mask_dwi_float = resample_image(
        mask,
        magnitude_affine,
        b0.shape,
        b0_affine,
        fieldmap_to_dwi_world,
        interpolation="linear",
        linear_extrapolation="fsl",
    )
    mask_dwi = (mask_dwi_float >= np.float32(0.5)).astype(np.uint8)
    shift = voxel_shift_from_field(field_dwi, dwell_seconds, phase_encoding_direction)
    shift, filled_mask_dwi = regularize_voxel_shift(
        shift, mask_dwi, phase_encoding_direction
    )
    shift = np.multiply(shift, filled_mask_dwi, dtype=np.float32)
    displacement = displacement_from_voxel_shift(
        shift, b0_affine, phase_encoding_direction
    )
    corrected = resample_image(
        b0,
        b0_affine,
        b0.shape,
        b0_affine,
        np.eye(4),
        interpolation="linear",
        reference_to_moving_displacement=displacement,
    )
    unwarped_b0_mask = resample_image(
        nodif_mask,
        b0_affine,
        b0.shape,
        b0_affine,
        np.eye(4),
        interpolation="sinc",
        reference_to_moving_displacement=displacement,
    )
    corrected_mask = (
        np.multiply(unwarped_b0_mask, mask_dwi, dtype=np.float32) > 0
    ).astype(np.uint8)
    corrected = np.multiply(corrected, corrected_mask, dtype=np.float32)
    if progress:
        progress("fieldmap_warp", 4, 4)
    return FieldmapResult(
        field,
        magnitude_brain,
        mask.astype(np.uint8),
        distorted,
        registration.matrix,
        field_dwi,
        mask_dwi,
        shift,
        displacement,
        corrected,
        corrected_mask,
        registration,
    )


def _load_3d(path: str | Path, name: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Read a finite three-dimensional NIfTI image."""

    image = nib.load(str(path))
    if len(image.shape) == 4 and image.shape[3] >= 1:
        values = np.asarray(image.dataobj[..., 0], dtype=np.float32)
    elif len(image.shape) == 3:
        values = np.asarray(image.dataobj, dtype=np.float32)
    else:
        raise ValueError(f"{name} must be 3D or have a nonempty fourth dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    return image, values


def _save_like(
    values: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output: Path,
    dtype: np.dtype | type[np.generic] = np.float32,
) -> None:
    """Write a NIfTI image while preserving the reference geometry."""

    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(values, dtype=dtype), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(output))


def run_fieldmap_nifti(
    magnitude_file: str | Path,
    field_radians_per_second_file: str | Path,
    b0_brain_file: str | Path,
    output_directory: str | Path,
    *,
    dwell_milliseconds: float,
    phase_encoding_direction: str,
    magnitude_mask_file: str | Path | None = None,
    b0_mask_file: str | Path | None = None,
    workers: int = 8,
    bet_backend: str = "optimized",
    median_filter: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the P7 NIfTI CLI and write a world-mm displacement directly composable by legacy."""

    started = perf_counter()
    if not np.isfinite(dwell_milliseconds) or dwell_milliseconds <= 0:
        raise ValueError("dwell_milliseconds must be positive and finite")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    prepared_magnitude = write_fsl_reoriented(
        magnitude_file, output / "magnitude_reoriented.nii.gz", float32=True
    )
    prepared_field = write_fsl_reoriented(
        field_radians_per_second_file,
        output / "field_reoriented_radians_per_second.nii.gz",
        float32=True,
    )
    prepared_b0 = write_fsl_reoriented(
        b0_brain_file, output / "b0_brain_reoriented.nii.gz", float32=True
    )
    magnitude_image, magnitude = _load_3d(prepared_magnitude, "magnitude")
    field_image, field = _load_3d(prepared_field, "fieldmap")
    b0_image, b0 = _load_3d(prepared_b0, "b0 brain")
    if magnitude.shape != field.shape or not np.allclose(
        magnitude_image.affine, field_image.affine, rtol=0.0, atol=1e-5
    ):
        raise ValueError("magnitude and fieldmap must share a grid")
    if magnitude_mask_file is None:
        bet = bet_brain_mask(
            magnitude,
            nib.affines.voxel_sizes(magnitude_image.affine),
            fractional_threshold=0.5,
            workers=workers,
            backend=bet_backend,
        )
        magnitude_mask = bet.mask
    else:
        prepared_magnitude_mask = write_fsl_reoriented(
            magnitude_mask_file,
            output / "magnitude_mask_reoriented.nii.gz",
        )
        mask_image, magnitude_mask = _load_3d(
            prepared_magnitude_mask, "magnitude mask"
        )
        if mask_image.shape[:3] != magnitude.shape or not np.allclose(
            mask_image.affine, magnitude_image.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError("magnitude mask must match magnitude")
    if b0_mask_file is None:
        b0_mask = b0 > 0
    else:
        prepared_b0_mask = write_fsl_reoriented(
            b0_mask_file,
            output / "b0_mask_reoriented.nii.gz",
        )
        mask_image, b0_mask = _load_3d(prepared_b0_mask, "b0 mask")
        if mask_image.shape[:3] != b0.shape or not np.allclose(
            mask_image.affine, b0_image.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError("b0 mask must match b0 brain")
    result = run_fieldmap(
        magnitude,
        field,
        magnitude_mask,
        b0,
        b0_mask,
        magnitude_image.affine,
        b0_image.affine,
        dwell_seconds=dwell_milliseconds / 1000.0,
        phase_encoding_direction=phase_encoding_direction,
        workers=workers,
        median_filter=median_filter,
        progress=progress,
    )
    field_outputs = {
        "field_radians_per_second": (result.field_radians_per_second, magnitude_image, np.float32),
        "magnitude_brain": (result.magnitude_brain, magnitude_image, np.float32),
        "magnitude_brain_mask": (result.magnitude_mask, magnitude_image, np.uint8),
        "magnitude_brain_distorted": (result.distorted_magnitude, magnitude_image, np.float32),
        "field_dwi_radians_per_second": (result.field_dwi_radians_per_second, b0_image, np.float32),
        "fieldmap_mask_dwi": (result.fieldmap_mask_dwi, b0_image, np.uint8),
        "voxel_shift": (result.voxel_shift, b0_image, np.float32),
        "displacement_world_mm": (result.displacement_world_mm, b0_image, np.float32),
        "corrected_b0": (result.corrected_b0, b0_image, np.float32),
        "corrected_mask": (result.corrected_mask, b0_image, np.uint8),
    }
    paths: dict[str, str] = {}
    for name, (values, reference, dtype) in field_outputs.items():
        path = output / f"{name}.nii.gz"
        _save_like(values, reference, path, dtype)
        paths[name] = str(path)
    matrix_path = output / "nodif2fieldmap.mat"
    np.savetxt(matrix_path, result.dwi_to_fieldmap_fsl, fmt="%.10g")
    paths["dwi_to_fieldmap_fsl"] = str(matrix_path)
    report: dict[str, object] = {
        "status": "complete",
        "algorithm": "simnibs-4.6-gre-fieldmap-fugue-rads-input",
        "fieldmap_input_units": "radians_per_second",
        "dwell_input_units": "milliseconds",
        "dwell_seconds": dwell_milliseconds / 1000.0,
        "voxel_shift_units": "voxels",
        "displacement_units": "nifti_world_millimeters",
        "phase_encoding_direction": phase_encoding_direction,
        "median_filter": median_filter,
        "workers": workers,
        "registration_cost": "mutual_information",
        "registration_degrees_of_freedom": 6,
        "registration_evaluations": result.registration.evaluations,
        "raw_storage_reorientation": "fslreorient2std-compatible-no-interpolation",
        "elapsed_seconds": perf_counter() - started,
        "outputs": paths,
    }
    report_path = output / "fieldmap_qa.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
