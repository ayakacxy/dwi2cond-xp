"""FSL TOPUP fixed-path primitives used by the SimNIBS 4.6 workflow.

This module intentionally follows the coefficient indexing and cubic B-spline
definitions in FSL 6.0.4 ``basisfield``.  It does not expose a generic TOPUP
command-line compatibility layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Callable
import json
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import csc_matrix

from ._numba import set_available_numba_threads
from .orientation import write_fsl_reoriented


@dataclass(frozen=True)
class TopupLevel:
    """Describe one immutable ``b02b0_nosubsamp.cnf`` resolution level."""

    warp_resolution_mm: float
    subsampling: int
    fwhm_mm: float
    max_iterations: int
    regularization_weight: float
    estimate_movements: bool
    minimizer: str


@dataclass(frozen=True)
class PreparedTopupScan:
    """Hold one scan and its reusable cubic interpolation coefficients."""

    values: np.ndarray
    interpolation_coefficients: np.ndarray


@dataclass(frozen=True)
class TopupFixedMovementState:
    """Cached images and derivatives for one fixed-movement field evaluation."""

    parameters: np.ndarray
    corrected_scans: np.ndarray
    alpha: np.ndarray
    axis_term: np.ndarray
    joint_mask: np.ndarray
    mean: np.ndarray
    mean_alpha: np.ndarray
    mean_axis_term: np.ndarray
    ssd: float


@dataclass(frozen=True)
class TopupMovingState:
    """Cached images and movement derivatives for one joint evaluation."""

    parameters: np.ndarray
    corrected_scans: np.ndarray
    alpha: np.ndarray
    axis_term: np.ndarray
    joint_mask: np.ndarray
    mean: np.ndarray
    mean_alpha: np.ndarray
    mean_axis_term: np.ndarray
    movement_derivatives: np.ndarray
    ssd: float


@dataclass(frozen=True)
class TopupRunResult:
    """Final fixed-subset TOPUP field, movements and per-level optimizer results."""

    field_coefficients: np.ndarray
    field_hz: np.ndarray
    movement_parameters: np.ndarray
    corrected_scans: np.ndarray
    joint_mask: np.ndarray
    level_results: tuple[object, ...]


@njit(cache=True)
def _periodic_cubic_deconvolve_line(line: np.ndarray) -> None:
    """Apply the cubic periodic IIR sweep used by FSL Splinterpolator."""

    pole = np.sqrt(3.0) - 2.0
    precision = 1.0e-8
    count = int(np.log(precision) / np.log(abs(pole)) + 1.5)
    count = min(count, line.size)
    initial = line[0]
    power = pole
    pointer = line.size - 1
    for _index in range(1, count):
        initial += power * line[pointer]
        pointer -= 1
        power *= pole
    line[0] = initial
    for index in range(1, line.size):
        line[index] += pole * line[index - 1]
    initial = pole * line[-1]
    power = pole * pole
    pointer = 0
    for _index in range(1, count):
        initial += power * line[pointer]
        pointer += 1
        power *= pole
    line[-1] = initial / (power - 1.0)
    for index in range(line.size - 2, -1, -1):
        line[index] = pole * (line[index + 1] - line[index])
    line *= 6.0


@njit(cache=True)
def _fsl_periodic_cubic_coefficients(values: np.ndarray) -> np.ndarray:
    """Create float32 spline coefficients in FSL's axis and storage order."""

    coefficients = values.copy()
    nx, ny, nz = coefficients.shape
    line_x = np.empty(nx, dtype=np.float64)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                line_x[x] = coefficients[x, y, z]
            _periodic_cubic_deconvolve_line(line_x)
            for x in range(nx):
                coefficients[x, y, z] = line_x[x]
    line_y = np.empty(ny, dtype=np.float64)
    for z in range(nz):
        for x in range(nx):
            for y in range(ny):
                line_y[y] = coefficients[x, y, z]
            _periodic_cubic_deconvolve_line(line_y)
            for y in range(ny):
                coefficients[x, y, z] = line_y[y]
    line_z = np.empty(nz, dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
            for z in range(nz):
                line_z[z] = coefficients[x, y, z]
            _periodic_cubic_deconvolve_line(line_z)
            for z in range(nz):
                coefficients[x, y, z] = line_z[z]
    return coefficients


@njit(cache=True, inline="always")
def _cubic_weight(distance: float) -> float:
    absolute = abs(distance)
    if absolute < 1.0:
        return 2.0 / 3.0 + 0.5 * absolute * absolute * (absolute - 2.0)
    if absolute < 2.0:
        remainder = 2.0 - absolute
        return remainder * remainder * remainder / 6.0
    return 0.0


@njit(cache=True, inline="always")
def _cubic_derivative_weight(distance: float) -> float:
    absolute = abs(distance)
    sign = -1.0 if distance < 0.0 else 1.0
    if absolute < 1.0:
        return sign * (1.5 * absolute * absolute - 2.0 * absolute)
    if absolute < 2.0:
        remainder = 2.0 - absolute
        return -0.5 * sign * remainder * remainder
    return 0.0


@njit(cache=True, parallel=True)
def _sample_fsl_periodic_cubic(
    coefficients: np.ndarray,
    displacement: np.ndarray,
    phase_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a PE-displaced volume and all three spatial derivatives."""

    source_nx, source_ny, source_nz = coefficients.shape
    nx, ny, nz = displacement.shape
    values = np.empty((nx, ny, nz), dtype=np.float32)
    derivative_x = np.empty_like(values)
    derivative_y = np.empty_like(values)
    derivative_z = np.empty_like(values)
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                coordinate_x = float(x)
                coordinate_y = float(y)
                coordinate_z = float(z)
                if phase_axis == 0:
                    coordinate_x += displacement[x, y, z]
                else:
                    coordinate_y += displacement[x, y, z]
                rounded_x = int(coordinate_x + 0.5)
                rounded_y = int(coordinate_y + 0.5)
                rounded_z = int(coordinate_z + 0.5)
                start_x = rounded_x - (1 if rounded_x < coordinate_x else 2)
                start_y = rounded_y - (1 if rounded_y < coordinate_y else 2)
                start_z = rounded_z - (1 if rounded_z < coordinate_z else 2)
                value = 0.0
                dx = 0.0
                dy = 0.0
                dz = 0.0
                for offset_z in range(4):
                    index_z = start_z + offset_z
                    weight_z = _cubic_weight(coordinate_z - index_z)
                    dweight_z = _cubic_derivative_weight(coordinate_z - index_z)
                    wrapped_z = index_z % source_nz
                    for offset_y in range(4):
                        index_y = start_y + offset_y
                        weight_y = _cubic_weight(coordinate_y - index_y)
                        dweight_y = _cubic_derivative_weight(coordinate_y - index_y)
                        wrapped_y = index_y % source_ny
                        weight_zy = weight_z * weight_y
                        for offset_x in range(4):
                            index_x = start_x + offset_x
                            weight_x = _cubic_weight(coordinate_x - index_x)
                            dweight_x = _cubic_derivative_weight(coordinate_x - index_x)
                            coefficient = coefficients[
                                index_x % source_nx, wrapped_y, wrapped_z
                            ]
                            value += coefficient * weight_x * weight_zy
                            dx += coefficient * dweight_x * weight_zy
                            dy += coefficient * weight_x * dweight_y * weight_z
                            dz += coefficient * weight_x * weight_y * dweight_z
                values[x, y, z] = value
                derivative_x[x, y, z] = dx
                derivative_y[x, y, z] = dy
                derivative_z[x, y, z] = dz
    return values, derivative_x, derivative_y, derivative_z


@njit(cache=True, parallel=True, nogil=True)
def _sample_fsl_periodic_cubic_affine(
    coefficients: np.ndarray,
    displacement: np.ndarray,
    phase_axis: int,
    pull_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample FSL's affine-plus-PE-displacement transform and derivatives."""

    source_nx, source_ny, source_nz = coefficients.shape
    nx, ny, nz = displacement.shape
    values = np.empty((nx, ny, nz), dtype=np.float32)
    derivative_x = np.empty_like(values)
    derivative_y = np.empty_like(values)
    derivative_z = np.empty_like(values)
    mask = np.empty((nx, ny, nz), dtype=np.uint8)
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                coordinate_x = (
                    pull_matrix[0, 0] * x
                    + pull_matrix[0, 1] * y
                    + pull_matrix[0, 2] * z
                    + pull_matrix[0, 3]
                )
                coordinate_y = (
                    pull_matrix[1, 0] * x
                    + pull_matrix[1, 1] * y
                    + pull_matrix[1, 2] * z
                    + pull_matrix[1, 3]
                )
                coordinate_z = (
                    pull_matrix[2, 0] * x
                    + pull_matrix[2, 1] * y
                    + pull_matrix[2, 2] * z
                    + pull_matrix[2, 3]
                )
                if phase_axis == 0:
                    coordinate_x = coordinate_x + displacement[x, y, z]
                else:
                    coordinate_y = coordinate_y + displacement[x, y, z]
                valid = True
                if phase_axis != 0 and not 0.0 <= coordinate_x <= source_nx - 1:
                    valid = False
                if phase_axis != 1 and not 0.0 <= coordinate_y <= source_ny - 1:
                    valid = False
                if not 0.0 <= coordinate_z <= source_nz - 1:
                    valid = False
                rounded_x = int(coordinate_x + 0.5)
                rounded_y = int(coordinate_y + 0.5)
                rounded_z = int(coordinate_z + 0.5)
                start_x = rounded_x - (1 if rounded_x < coordinate_x else 2)
                start_y = rounded_y - (1 if rounded_y < coordinate_y else 2)
                start_z = rounded_z - (1 if rounded_z < coordinate_z else 2)
                value = 0.0
                dx = 0.0
                dy = 0.0
                dz = 0.0
                for offset_z in range(4):
                    index_z = start_z + offset_z
                    weight_z = _cubic_weight(coordinate_z - index_z)
                    dweight_z = _cubic_derivative_weight(coordinate_z - index_z)
                    wrapped_z = index_z % source_nz
                    for offset_y in range(4):
                        index_y = start_y + offset_y
                        weight_y = _cubic_weight(coordinate_y - index_y)
                        dweight_y = _cubic_derivative_weight(coordinate_y - index_y)
                        wrapped_y = index_y % source_ny
                        weight_zy = weight_z * weight_y
                        for offset_x in range(4):
                            index_x = start_x + offset_x
                            weight_x = _cubic_weight(coordinate_x - index_x)
                            dweight_x = _cubic_derivative_weight(coordinate_x - index_x)
                            coefficient = coefficients[
                                index_x % source_nx, wrapped_y, wrapped_z
                            ]
                            value += coefficient * weight_x * weight_zy
                            dx += coefficient * dweight_x * weight_zy
                            dy += coefficient * weight_x * dweight_y * weight_z
                            dz += coefficient * weight_x * weight_y * dweight_z
                values[x, y, z] = value
                derivative_x[x, y, z] = dx
                derivative_y[x, y, z] = dy
                derivative_z[x, y, z] = dz
                mask[x, y, z] = 1 if valid else 0
    return values, derivative_x, derivative_y, derivative_z, mask


@njit(cache=True, parallel=True, nogil=True)
def _sample_fsl_periodic_cubic_affine_batch(
    coefficients: np.ndarray,
    displacements: np.ndarray,
    phase_axis: int,
    pull_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample independent scans in one dispatch with unchanged scan arithmetic."""

    source_nx, source_ny, source_nz, scan_count = coefficients.shape
    nx, ny, nz, _ = displacements.shape
    values = np.empty((nx, ny, nz, scan_count), dtype=np.float32)
    derivative_x = np.empty_like(values)
    derivative_y = np.empty_like(values)
    derivative_z = np.empty_like(values)
    mask = np.empty((nx, ny, nz, scan_count), dtype=np.uint8)
    for work_index in prange(scan_count * nz):
        scan_index = work_index // nz
        z = work_index - scan_index * nz
        pull_matrix = pull_matrices[scan_index]
        for y in range(ny):
            for x in range(nx):
                coordinate_x = (
                    pull_matrix[0, 0] * x
                    + pull_matrix[0, 1] * y
                    + pull_matrix[0, 2] * z
                    + pull_matrix[0, 3]
                )
                coordinate_y = (
                    pull_matrix[1, 0] * x
                    + pull_matrix[1, 1] * y
                    + pull_matrix[1, 2] * z
                    + pull_matrix[1, 3]
                )
                coordinate_z = (
                    pull_matrix[2, 0] * x
                    + pull_matrix[2, 1] * y
                    + pull_matrix[2, 2] * z
                    + pull_matrix[2, 3]
                )
                if phase_axis == 0:
                    coordinate_x = (
                        coordinate_x + displacements[x, y, z, scan_index]
                    )
                else:
                    coordinate_y = (
                        coordinate_y + displacements[x, y, z, scan_index]
                    )
                valid = True
                if phase_axis != 0 and not 0.0 <= coordinate_x <= source_nx - 1:
                    valid = False
                if phase_axis != 1 and not 0.0 <= coordinate_y <= source_ny - 1:
                    valid = False
                if not 0.0 <= coordinate_z <= source_nz - 1:
                    valid = False
                rounded_x = int(coordinate_x + 0.5)
                rounded_y = int(coordinate_y + 0.5)
                rounded_z = int(coordinate_z + 0.5)
                start_x = rounded_x - (1 if rounded_x < coordinate_x else 2)
                start_y = rounded_y - (1 if rounded_y < coordinate_y else 2)
                start_z = rounded_z - (1 if rounded_z < coordinate_z else 2)
                value = 0.0
                dx = 0.0
                dy = 0.0
                dz = 0.0
                for offset_z in range(4):
                    index_z = start_z + offset_z
                    weight_z = _cubic_weight(coordinate_z - index_z)
                    dweight_z = _cubic_derivative_weight(coordinate_z - index_z)
                    wrapped_z = index_z % source_nz
                    for offset_y in range(4):
                        index_y = start_y + offset_y
                        weight_y = _cubic_weight(coordinate_y - index_y)
                        dweight_y = _cubic_derivative_weight(coordinate_y - index_y)
                        wrapped_y = index_y % source_ny
                        weight_zy = weight_z * weight_y
                        for offset_x in range(4):
                            index_x = start_x + offset_x
                            weight_x = _cubic_weight(coordinate_x - index_x)
                            dweight_x = _cubic_derivative_weight(
                                coordinate_x - index_x
                            )
                            coefficient = coefficients[
                                index_x % source_nx,
                                wrapped_y,
                                wrapped_z,
                                scan_index,
                            ]
                            value += coefficient * weight_x * weight_zy
                            dx += coefficient * dweight_x * weight_zy
                            dy += coefficient * weight_x * dweight_y * weight_z
                            dz += coefficient * weight_x * weight_y * dweight_z
                values[x, y, z, scan_index] = value
                derivative_x[x, y, z, scan_index] = dx
                derivative_y[x, y, z, scan_index] = dy
                derivative_z[x, y, z, scan_index] = dz
                mask[x, y, z, scan_index] = 1 if valid else 0
    return values, derivative_x, derivative_y, derivative_z, mask


SIMNIBS46_TOPUP_LEVELS: Final[tuple[TopupLevel, ...]] = tuple(
    TopupLevel(*values)
    for values in zip(
        (20.0, 16.0, 14.0, 12.0, 10.0, 6.0, 4.0, 4.0, 4.0),
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
        (8.0, 6.0, 4.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0),
        (5, 5, 5, 5, 5, 10, 10, 20, 20),
        (5e-3, 1e-3, 1e-4, 1.5e-5, 5e-6, 5e-7, 5e-8, 5e-10, 1e-11),
        (True, True, True, True, True, False, False, False, False),
        ("levenberg_marquardt",) * 5 + ("scaled_conjugate_gradient",) * 4,
        strict=True,
    )
)

SPLINE_ORDER: Final = 3
REGULARIZATION_MODEL: Final = "bending_energy"
SCALE_IMAGES_INDIVIDUALLY: Final = True
SCALE_REGULARIZATION_BY_SSD: Final = True
HESSIAN_PRECISION: Final = "double"
IMAGE_INTERPOLATION: Final = "spline"


def validate_acquisition_parameters(
    acquisition_parameters: np.ndarray,
    *,
    number_of_volumes: int | None = None,
) -> np.ndarray:
    """Validate FSL ``acqp`` rows and return an owned float64 array.

    The fixed subset permits exactly one non-zero phase-encoding component per
    row.  TOPUP itself accepts more general vectors, but EDDY later requires the
    single-axis contract used by SimNIBS.
    """

    values = np.asarray(acquisition_parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("acquisition parameters must have shape (N, 4)")
    if number_of_volumes is not None and values.shape[0] != number_of_volumes:
        raise ValueError("one acquisition-parameter row is required per input volume")
    if not np.all(np.isfinite(values)):
        raise ValueError("acquisition parameters must be finite")
    directions = values[:, :3]
    if np.any(np.count_nonzero(directions, axis=1) != 1):
        raise ValueError("each acquisition row must select exactly one PE axis")
    if np.any(directions[:, 2] != 0.0):
        raise ValueError("FSL 6.0.4 TOPUP does not support phase encoding along z")
    if np.any(values[:, 3] <= 0.0):
        raise ValueError("total readout times must be positive")
    return values.copy()


def fsl_knot_spacing(
    warp_resolution_mm: float, voxel_sizes_mm: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Convert warp resolution to FSL integer voxel knot spacing.

    FSL uses ``round(warpres / voxel_size)`` with positive inputs and clamps a
    rounded zero to one.  NumPy's bankers rounding is deliberately not used.
    """

    if not np.isfinite(warp_resolution_mm) or warp_resolution_mm <= 0.0:
        raise ValueError("warp resolution must be positive and finite")
    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if voxel_sizes.shape != (3,) or not np.all(np.isfinite(voxel_sizes)):
        raise ValueError("voxel sizes must contain three finite values")
    if np.any(voxel_sizes <= 0.0):
        raise ValueError("voxel sizes must be positive")
    return tuple(
        max(1, int(np.floor(warp_resolution_mm / size + 0.5))) for size in voxel_sizes
    )


def fsl_coefficient_shape(
    field_shape: tuple[int, int, int], knot_spacing: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Return FSL's backward-compatible cubic coefficient matrix size."""

    if len(field_shape) != 3 or len(knot_spacing) != 3:
        raise ValueError("field shape and knot spacing must be three-dimensional")
    result = []
    for size, spacing in zip(field_shape, knot_spacing, strict=True):
        if size < 1 or spacing < 1:
            raise ValueError("field sizes and knot spacings must be positive")
        if spacing == 1:
            result.append(int(size))
        else:
            result.append(int(np.ceil((size + 1) / spacing)) + 2)
    return tuple(result)


def cubic_spline_value(
    voxel: float, coefficient_index: int, knot_spacing: int, derivative: int = 0
) -> float:
    """Evaluate FSL's cubic B-spline basis or first/second derivative."""

    if knot_spacing < 1:
        raise ValueError("knot spacing must be positive")
    if coefficient_index < 0:
        raise ValueError("coefficient index must be nonnegative")
    if derivative not in (0, 1, 2):
        raise ValueError("only derivative orders 0, 1, and 2 are supported")
    center = (
        coefficient_index
        if knot_spacing == 1
        else (coefficient_index - SPLINE_ORDER // 2) * knot_spacing
    )
    x = (float(voxel) - center) / knot_spacing
    absolute = abs(x)
    if derivative == 0:
        if absolute <= 1.0:
            return (2.0 / 3.0) + absolute * absolute * (absolute / 2.0 - 1.0)
        if absolute < 2.0:
            return ((2.0 - absolute) ** 3) / 6.0
        return 0.0
    if derivative == 1:
        if absolute < 1e-6:
            return 0.0
        sign = x / absolute
        if absolute <= 1.0:
            return sign * (1.5 * absolute * absolute - 2.0 * absolute) / knot_spacing
        if absolute < 2.0:
            return -sign * 0.5 * (2.0 - absolute) ** 2 / knot_spacing
        return 0.0
    denominator = float(knot_spacing * knot_spacing)
    if absolute <= 1.0:
        return (3.0 * absolute - 2.0) / denominator
    if absolute < 2.0:
        return (2.0 - absolute) / denominator
    return 0.0


@lru_cache(maxsize=64)
def spline_design_matrix(
    field_size: int,
    knot_spacing: int,
    *,
    derivative: int = 0,
) -> np.ndarray:
    """Build a dense 1D FSL spline design matrix in float64."""

    coefficient_size = fsl_coefficient_shape((field_size, 1, 1), (knot_spacing, 1, 1))[
        0
    ]
    matrix = np.empty((field_size, coefficient_size), dtype=np.float64)
    for voxel in range(field_size):
        for coefficient in range(coefficient_size):
            matrix[voxel, coefficient] = cubic_spline_value(
                voxel, coefficient, knot_spacing, derivative
            )
    return matrix


@lru_cache(maxsize=64)
def _sampled_cubic_kernel(knot_spacing: int, derivative: int) -> np.ndarray:
    size = (SPLINE_ORDER + 1) * knot_spacing - 1
    center = (size - 1) / 2.0
    coefficient = 0 if knot_spacing == 1 else SPLINE_ORDER // 2
    return np.asarray(
        [
            cubic_spline_value(index - center, coefficient, knot_spacing, derivative)
            for index in range(size)
        ],
        dtype=np.float64,
    )


def _kernel_overlap(first: np.ndarray, second: np.ndarray, shift: int) -> float:
    if shift >= 0:
        count = min(first.size - shift, second.size)
        if count <= 0:
            return 0.0
        return float(np.dot(first[shift : shift + count], second[:count]))
    return _kernel_overlap(second, first, -shift)


@lru_cache(maxsize=64)
def _sampled_cubic_self_overlaps(
    knot_spacing: int, derivative: int, overlap_limit: int
) -> np.ndarray:
    """Cache repeated 1D kernel overlaps used by every bending-matrix offset."""

    kernel = _sampled_cubic_kernel(knot_spacing, derivative)
    values = np.asarray(
        [
            _kernel_overlap(kernel, kernel, shift * knot_spacing)
            for shift in range(-overlap_limit, overlap_limit + 1)
        ],
        dtype=np.float64,
    )
    values.setflags(write=False)
    return values


@njit(cache=True)
def _assemble_bending_energy_coo(
    coefficient_shape: tuple[int, int, int],
    overlap_limit: tuple[int, int, int],
    helper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble FSL's sparse bending matrix in its original loop order."""

    nx, ny, nz = coefficient_shape
    limit_x, limit_y, limit_z = overlap_limit
    count = 0
    for coefficient_z in range(nz):
        for coefficient_y in range(ny):
            for coefficient_x in range(nx):
                for dz in range(-limit_z, limit_z + 1):
                    row_z = coefficient_z + dz
                    if not 0 <= row_z < nz:
                        continue
                    for dy in range(-limit_y, limit_y + 1):
                        row_y = coefficient_y + dy
                        if not 0 <= row_y < ny:
                            continue
                        for dx in range(-limit_x, limit_x + 1):
                            row_x = coefficient_x + dx
                            if 0 <= row_x < nx:
                                count += 1

    rows = np.empty(count, dtype=np.int64)
    columns = np.empty(count, dtype=np.int64)
    data = np.empty(count, dtype=np.float64)
    output = 0
    for coefficient_z in range(nz):
        for coefficient_y in range(ny):
            for coefficient_x in range(nx):
                column = coefficient_z * ny * nx + coefficient_y * nx + coefficient_x
                for dz in range(-limit_z, limit_z + 1):
                    row_z = coefficient_z + dz
                    if not 0 <= row_z < nz:
                        continue
                    for dy in range(-limit_y, limit_y + 1):
                        row_y = coefficient_y + dy
                        if not 0 <= row_y < ny:
                            continue
                        for dx in range(-limit_x, limit_x + 1):
                            row_x = coefficient_x + dx
                            if not 0 <= row_x < nx:
                                continue
                            rows[output] = row_z * ny * nx + row_y * nx + row_x
                            columns[output] = column
                            data[output] = helper[
                                dx + limit_x,
                                dy + limit_y,
                                dz + limit_z,
                            ]
                            output += 1
    return rows, columns, data


@lru_cache(maxsize=16)
def bending_energy_hessian(
    field_shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    knot_spacing: tuple[int, int, int],
) -> csc_matrix:
    """Build FSL's double-precision cubic bending-energy Hessian.

    The regularizer covers the complete spline field, including support beyond
    the image FOV.  Matrix indices use FSL's x-fastest coefficient order.
    """

    coefficient_shape = fsl_coefficient_shape(field_shape, knot_spacing)
    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if voxel_sizes.shape != (3,) or not np.all(np.isfinite(voxel_sizes)):
        raise ValueError("voxel sizes must contain three finite values")
    if np.any(voxel_sizes <= 0.0):
        raise ValueError("voxel sizes must be positive")
    overlaps: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    weights: dict[tuple[int, int, int], float] = {}
    overlap_limit = tuple(
        2 if spacing == 1 else SPLINE_ORDER for spacing in knot_spacing
    )
    for first_axis in range(3):
        for second_axis in range(first_axis, 3):
            derivatives = [0, 0, 0]
            derivatives[first_axis] += 1
            derivatives[second_axis] += 1
            key = tuple(derivatives)
            scale = voxel_sizes[first_axis] * voxel_sizes[second_axis]
            weights[key] = (2.0 if first_axis == second_axis else 4.0) / (scale * scale)
            overlaps[key] = tuple(
                _sampled_cubic_self_overlaps(
                    knot_spacing[axis], derivatives[axis], overlap_limit[axis]
                )
                for axis in range(3)
            )
    helper = np.empty(tuple(2 * limit + 1 for limit in overlap_limit), dtype=np.float64)
    for dz in range(-overlap_limit[2], overlap_limit[2] + 1):
        for dy in range(-overlap_limit[1], overlap_limit[1] + 1):
            for dx in range(-overlap_limit[0], overlap_limit[0] + 1):
                value = 0.0
                offsets = (dx, dy, dz)
                for key, axes in overlaps.items():
                    product = 1.0
                    for axis in range(3):
                        product *= axes[axis][offsets[axis] + overlap_limit[axis]]
                    value += weights[key] * product
                helper[
                    dx + overlap_limit[0],
                    dy + overlap_limit[1],
                    dz + overlap_limit[2],
                ] = value

    rows, columns, data = _assemble_bending_energy_coo(
        coefficient_shape, overlap_limit, helper
    )
    nx, ny, nz = coefficient_shape
    size = nx * ny * nz
    return csc_matrix((data, (rows, columns)), shape=(size, size), dtype=np.float64)


def bending_energy(
    coefficients: np.ndarray,
    field_shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    knot_spacing: tuple[int, int, int],
    *,
    hessian: csc_matrix | None = None,
) -> float:
    """Calculate FSL bending energy as ``0.5 * c.T @ H @ c``."""

    values = np.asarray(coefficients, dtype=np.float64)
    expected = fsl_coefficient_shape(field_shape, knot_spacing)
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"coefficients must be finite with shape {expected}")
    matrix = (
        bending_energy_hessian(field_shape, voxel_sizes_mm, knot_spacing)
        if hessian is None
        else hessian
    )
    vector = values.reshape(-1, order="F")
    return 0.5 * float(vector @ (matrix @ vector))


@njit(cache=True)
def _expand_coefficients_fsl_order(
    coefficients: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
) -> np.ndarray:
    """Expand coefficients in the same coefficient and voxel loop order as FSL."""

    field = np.zeros(
        (basis_x.shape[0], basis_y.shape[0], basis_z.shape[0]), dtype=np.float64
    )
    for coefficient_z in range(coefficients.shape[2]):
        first_z = 0
        while first_z < basis_z.shape[0] and basis_z[first_z, coefficient_z] == 0.0:
            first_z += 1
        last_z = basis_z.shape[0]
        while last_z > first_z and basis_z[last_z - 1, coefficient_z] == 0.0:
            last_z -= 1
        for coefficient_y in range(coefficients.shape[1]):
            first_y = 0
            while first_y < basis_y.shape[0] and basis_y[first_y, coefficient_y] == 0.0:
                first_y += 1
            last_y = basis_y.shape[0]
            while last_y > first_y and basis_y[last_y - 1, coefficient_y] == 0.0:
                last_y -= 1
            for coefficient_x in range(coefficients.shape[0]):
                value = coefficients[coefficient_x, coefficient_y, coefficient_z]
                if value == 0.0:
                    continue
                first_x = 0
                while (
                    first_x < basis_x.shape[0]
                    and basis_x[first_x, coefficient_x] == 0.0
                ):
                    first_x += 1
                last_x = basis_x.shape[0]
                while last_x > first_x and basis_x[last_x - 1, coefficient_x] == 0.0:
                    last_x -= 1
                for voxel_z in range(first_z, last_z):
                    weight_z = basis_z[voxel_z, coefficient_z]
                    for voxel_y in range(first_y, last_y):
                        weight_zy = weight_z * basis_y[voxel_y, coefficient_y]
                        for voxel_x in range(first_x, last_x):
                            # FSL precomputes the 3D kernel as z*y*x, then
                            # multiplies that stored value by the coefficient.
                            kernel = weight_zy * basis_x[voxel_x, coefficient_x]
                            field[voxel_x, voxel_y, voxel_z] += value * kernel
    return field


@njit(cache=True, parallel=True)
def _expand_coefficients_fsl_order_voxel_parallel(
    coefficients: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
) -> np.ndarray:
    """Expand independent voxels while retaining FSL's coefficient order."""

    nx = basis_x.shape[0]
    ny = basis_y.shape[0]
    nz = basis_z.shape[0]
    field = np.zeros((nx, ny, nz), dtype=np.float64)
    for voxel in prange(nx * ny * nz):
        voxel_x = voxel % nx
        voxel_y = (voxel // nx) % ny
        voxel_z = voxel // (nx * ny)
        value = 0.0
        for coefficient_z in range(coefficients.shape[2]):
            weight_z = basis_z[voxel_z, coefficient_z]
            if weight_z == 0.0:
                continue
            for coefficient_y in range(coefficients.shape[1]):
                weight_zy = weight_z * basis_y[voxel_y, coefficient_y]
                if weight_zy == 0.0:
                    continue
                for coefficient_x in range(coefficients.shape[0]):
                    kernel = weight_zy * basis_x[voxel_x, coefficient_x]
                    value += (
                        coefficients[coefficient_x, coefficient_y, coefficient_z]
                        * kernel
                    )
        field[voxel_x, voxel_y, voxel_z] = value
    return field


def expand_spline_coefficients(
    coefficients: np.ndarray,
    field_shape: tuple[int, int, int],
    knot_spacing: tuple[int, int, int],
    *,
    derivative_axis: int | None = None,
) -> np.ndarray:
    """Expand FSL cubic coefficients to a dense field or one voxel derivative."""

    values = np.asarray(coefficients, dtype=np.float64)
    expected = fsl_coefficient_shape(field_shape, knot_spacing)
    if values.shape != expected:
        raise ValueError(f"coefficient shape must be {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite")
    if derivative_axis is not None and derivative_axis not in (0, 1, 2):
        raise ValueError("derivative axis must be 0, 1, 2, or None")
    matrices = tuple(
        spline_design_matrix(
            field_shape[axis],
            knot_spacing[axis],
            derivative=1 if derivative_axis == axis else 0,
        )
        for axis in range(3)
    )
    return _expand_coefficients_fsl_order_voxel_parallel(values, *matrices)


def field_displacement_voxels(
    field_hz: np.ndarray, acquisition_row: np.ndarray
) -> np.ndarray:
    """Convert an off-resonance field to FSL pull displacement in voxels."""

    field = np.asarray(field_hz, dtype=np.float64)
    if field.ndim != 3 or not np.all(np.isfinite(field)):
        raise ValueError("field must be a finite 3D array")
    row = validate_acquisition_parameters(np.asarray(acquisition_row)[None, :])[0]
    return field[..., None] * row[:3] * row[3]


def field_jacobian(
    field_hz: np.ndarray,
    acquisition_row: np.ndarray,
    *,
    field_coefficients: np.ndarray | None = None,
    knot_spacing: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Return FSL's single-axis susceptibility Jacobian determinant.

    When coefficients are supplied, the derivative comes from the same spline
    basis as TOPUP.  ``numpy.gradient`` is intentionally not used as a fallback
    because it would change the algorithm at image boundaries.
    """

    field = np.asarray(field_hz, dtype=np.float64)
    if field.ndim != 3 or not np.all(np.isfinite(field)):
        raise ValueError("field must be a finite 3D array")
    row = validate_acquisition_parameters(np.asarray(acquisition_row)[None, :])[0]
    axis = int(np.flatnonzero(row[:3])[0])
    if field_coefficients is None or knot_spacing is None:
        raise ValueError(
            "field coefficients and knot spacing are required for Jacobian"
        )
    derivative = expand_spline_coefficients(
        field_coefficients,
        field.shape,
        knot_spacing,
        derivative_axis=axis,
    )
    return 1.0 + row[axis] * row[3] * derivative


def resample_topup_scan(
    scan: np.ndarray | PreparedTopupScan,
    field_hz: np.ndarray,
    acquisition_row: np.ndarray,
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the zero-motion TOPUP forward model to one scan.

    FSL uses cubic-spline interpolation, periodic extrapolation only along the
    phase-encoding axis, and zero validity outside the other two axes.  The
    returned mask includes the additional one-voxel non-PE frame used by TOPUP.
    """

    if isinstance(scan, PreparedTopupScan):
        values = scan.values
        coefficients = scan.interpolation_coefficients
    else:
        values = np.asarray(scan, dtype=np.float32)
        coefficients = _fsl_periodic_cubic_coefficients(values)
    field = np.asarray(field_hz, dtype=np.float64)
    determinant = np.asarray(jacobian, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape != field.shape
        or values.shape != determinant.shape
    ):
        raise ValueError("scan, field, and Jacobian must be matching 3D arrays")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(determinant)):
        raise ValueError("scan and Jacobian must be finite")
    row = validate_acquisition_parameters(np.asarray(acquisition_row)[None, :])[0]
    axis = int(np.flatnonzero(row[:3])[0])
    displacement = field * row[axis] * row[3]
    mask = np.ones(values.shape, dtype=np.uint8)
    for other_axis in range(3):
        if other_axis == axis:
            continue
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[other_axis] = 0
        upper[other_axis] = -1
        mask[tuple(lower)] = 0
        mask[tuple(upper)] = 0
    sampled, _dx, _dy, _dz = _sample_fsl_periodic_cubic(
        coefficients, displacement, axis
    )
    corrected = sampled * determinant
    corrected[mask == 0] = 0.0
    return corrected.astype(np.float32), mask


def prepare_topup_scan(scan: np.ndarray) -> PreparedTopupScan:
    """Precompute FSL-style cubic interpolation state once per prepared scan."""

    values = np.asarray(scan, dtype=np.float32)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("scan must be a finite 3D array")
    owned = np.ascontiguousarray(values)
    coefficients = _fsl_periodic_cubic_coefficients(owned)
    owned.setflags(write=False)
    coefficients.setflags(write=False)
    return PreparedTopupScan(owned, coefficients)


def fsl_regrid_topup_scan(
    scan: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Apply TOPUP's default one-voxel source-grid enlargement.

    FSL keeps the field and output on the original target grid, but constructs
    each interpolation source with one extra voxel along every axis.  The new
    sampling preserves the original centre-to-centre field of view minus the
    small epsilon used by ``TopupScan::ReGrid``.
    """

    values = np.asarray(scan, dtype=np.float32)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("scan must be a finite 3D array")
    sizes = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("voxel sizes must contain three positive finite values")
    target_shape = tuple(int(size) + 1 for size in values.shape)
    new_sizes = tuple(
        float(((values.shape[axis] - 1) * sizes[axis] - 1.0e-6) / target_shape[axis])
        for axis in range(3)
    )
    pull = np.eye(4, dtype=np.float64)
    for axis in range(3):
        pull[axis, axis] = new_sizes[axis] / sizes[axis]
    displacement = np.zeros(target_shape, dtype=np.float32)
    coefficients = _fsl_periodic_cubic_coefficients(np.ascontiguousarray(values))
    regridded, _dx, _dy, _dz, _mask = _sample_fsl_periodic_cubic_affine(
        coefficients,
        displacement,
        0,
        pull,
    )
    return np.ascontiguousarray(regridded), new_sizes


@njit(cache=True)
def _periodic_convolve_axis_fsl_order(
    values: np.ndarray, kernel: np.ndarray, axis: int
) -> np.ndarray:
    output = np.empty_like(values)
    radius = kernel.size // 2
    nx, ny, nz = values.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                result = np.float32(0.0)
                for offset in range(-radius, radius + 1):
                    sample_x = x
                    sample_y = y
                    sample_z = z
                    if axis == 0:
                        sample_x = (x + offset) % nx
                    elif axis == 1:
                        sample_y = (y + offset) % ny
                    else:
                        sample_z = (z + offset) % nz
                    result = np.float32(
                        result
                        + values[sample_x, sample_y, sample_z] * kernel[offset + radius]
                    )
                output[x, y, z] = result
    return output


def fsl_smooth_topup_scan(
    scan: np.ndarray,
    fwhm_mm: float,
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Apply FSL NEWIMAGE's periodic separable Gaussian smoothing."""

    values = np.asarray(scan, dtype=np.float32)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("scan must be a finite 3D array")
    if not np.isfinite(fwhm_mm) or fwhm_mm < 0.0:
        raise ValueError("FWHM must be nonnegative and finite")
    sizes = np.asarray(voxel_sizes_mm, dtype=np.float32)
    if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("voxel sizes must contain three positive finite values")
    if fwhm_mm == 0.0:
        return np.ascontiguousarray(values)
    sigma_mm = np.float32(fwhm_mm / np.sqrt(8.0 * np.log(2.0)))
    result = np.ascontiguousarray(values)
    for axis in range(3):
        sigma = np.float32(sigma_mm / sizes[axis])
        radius = int(np.float32(sigma - np.float32(0.001))) * 2 + 3
        unnormalized = np.empty(2 * radius + 1, dtype=np.float32)
        total = np.float32(0.0)
        for offset in range(-radius, radius + 1):
            value = np.float32(
                np.exp(-(offset * offset) / (2.0 * float(sigma * sigma)))
            )
            unnormalized[offset + radius] = value
            total = np.float32(total + value)
        kernel = unnormalized.astype(np.float64)
        kernel *= 1.0 / float(total)
        result = _periodic_convolve_axis_fsl_order(result, kernel, axis)
    return result


@njit(cache=True, parallel=True)
def _spline_jte_fsl_order(
    image: np.ndarray,
    mask: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
) -> np.ndarray:
    """Apply a spline transpose in FSL's coefficient and voxel order."""

    result = np.zeros(
        (basis_x.shape[1], basis_y.shape[1], basis_z.shape[1]), dtype=np.float64
    )
    coefficient_x_count = result.shape[0]
    coefficient_y_count = result.shape[1]
    count = result.size
    for coefficient in prange(count):
        coefficient_x = coefficient % coefficient_x_count
        coefficient_y = (coefficient // coefficient_x_count) % coefficient_y_count
        coefficient_z = coefficient // (coefficient_x_count * coefficient_y_count)
        value = 0.0
        for z in range(image.shape[2]):
            weight_z = basis_z[z, coefficient_z]
            if weight_z == 0.0:
                continue
            for y in range(image.shape[1]):
                weight_zy = weight_z * basis_y[y, coefficient_y]
                if weight_zy == 0.0:
                    continue
                for x in range(image.shape[0]):
                    if mask[x, y, z]:
                        value += (
                            basis_x[x, coefficient_x]
                            * weight_zy
                            * float(image[x, y, z])
                        )
        result[coefficient_x, coefficient_y, coefficient_z] = value
    return result


@njit(cache=True, parallel=True)
def _paired_spline_jte_fsl_order(
    direct_image: np.ndarray,
    derivative_image: np.ndarray,
    mask: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
    derivative_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply both field-gradient spline transposes in one voxel traversal."""

    result_shape = (basis_x.shape[1], basis_y.shape[1], basis_z.shape[1])
    direct_result = np.zeros(result_shape, dtype=np.float64)
    derivative_result = np.zeros(result_shape, dtype=np.float64)
    coefficient_x_count = result_shape[0]
    coefficient_y_count = result_shape[1]
    count = direct_result.size
    for coefficient in prange(count):
        coefficient_x = coefficient % coefficient_x_count
        coefficient_y = (coefficient // coefficient_x_count) % coefficient_y_count
        coefficient_z = coefficient // (coefficient_x_count * coefficient_y_count)
        direct_value = 0.0
        derivative_value = 0.0
        for z in range(direct_image.shape[2]):
            direct_weight_z = basis_z[z, coefficient_z]
            derivative_weight_z = derivative_z[z, coefficient_z]
            if direct_weight_z == 0.0 and derivative_weight_z == 0.0:
                continue
            for y in range(direct_image.shape[1]):
                direct_weight_zy = direct_weight_z * basis_y[y, coefficient_y]
                derivative_weight_zy = (
                    derivative_weight_z * derivative_y[y, coefficient_y]
                )
                if direct_weight_zy == 0.0 and derivative_weight_zy == 0.0:
                    continue
                for x in range(direct_image.shape[0]):
                    if mask[x, y, z]:
                        direct_value += (
                            basis_x[x, coefficient_x]
                            * direct_weight_zy
                            * float(direct_image[x, y, z])
                        )
                        derivative_value += (
                            derivative_x[x, coefficient_x]
                            * derivative_weight_zy
                            * float(derivative_image[x, y, z])
                        )
        direct_result[coefficient_x, coefficient_y, coefficient_z] = direct_value
        derivative_result[coefficient_x, coefficient_y, coefficient_z] = (
            derivative_value
        )
    return direct_result, derivative_result


@njit(cache=True, parallel=True)
def _movement_interaction_jte_fsl_order(
    movement_derivatives: np.ndarray,
    alpha_difference: np.ndarray,
    axis_difference: np.ndarray,
    mask: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
    derivative_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate independent movement-interaction columns in one traversal."""

    coefficient_x_count = basis_x.shape[1]
    coefficient_y_count = basis_y.shape[1]
    coefficient_z_count = basis_z.shape[1]
    coefficient_count = (
        coefficient_x_count * coefficient_y_count * coefficient_z_count
    )
    movement_count = movement_derivatives.shape[3]
    direct_result = np.zeros((coefficient_count, movement_count), dtype=np.float64)
    derivative_result = np.zeros_like(direct_result)
    for coefficient in prange(coefficient_count):
        coefficient_x = coefficient % coefficient_x_count
        coefficient_y = (coefficient // coefficient_x_count) % coefficient_y_count
        coefficient_z = coefficient // (coefficient_x_count * coefficient_y_count)
        direct_values = np.zeros(movement_count, dtype=np.float64)
        derivative_values = np.zeros(movement_count, dtype=np.float64)
        for z in range(alpha_difference.shape[2]):
            direct_weight_z = basis_z[z, coefficient_z]
            derivative_weight_z = derivative_z[z, coefficient_z]
            if direct_weight_z == 0.0 and derivative_weight_z == 0.0:
                continue
            for y in range(alpha_difference.shape[1]):
                direct_weight_zy = direct_weight_z * basis_y[y, coefficient_y]
                derivative_weight_zy = (
                    derivative_weight_z * derivative_y[y, coefficient_y]
                )
                if direct_weight_zy == 0.0 and derivative_weight_zy == 0.0:
                    continue
                for x in range(alpha_difference.shape[0]):
                    if mask[x, y, z]:
                        direct_weight = basis_x[x, coefficient_x] * direct_weight_zy
                        derivative_weight = (
                            derivative_x[x, coefficient_x] * derivative_weight_zy
                        )
                        alpha_value = alpha_difference[x, y, z]
                        axis_value = axis_difference[x, y, z]
                        for movement in range(movement_count):
                            movement_value = movement_derivatives[x, y, z, movement]
                            direct_image_value = np.float32(
                                movement_value * alpha_value
                            )
                            derivative_image_value = np.float32(
                                movement_value * axis_value
                            )
                            direct_values[movement] += direct_weight * float(
                                direct_image_value
                            )
                            derivative_values[movement] += derivative_weight * float(
                                derivative_image_value
                            )
        direct_result[coefficient] = direct_values
        derivative_result[coefficient] = derivative_values
    return direct_result, derivative_result


@njit(cache=True)
def _masked_sum_product_fsl_order(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray
) -> float:
    """Accumulate a float32 image product in FSL's z/y/x loop order."""

    result = 0.0
    for z in range(first.shape[2]):
        for y in range(first.shape[1]):
            for x in range(first.shape[0]):
                if mask[x, y, z]:
                    result += first[x, y, z] * second[x, y, z]
    return result


@njit(cache=True)
def _topup_ssd_fsl_order(
    corrected_scans: np.ndarray,
    mean: np.ndarray,
    joint_mask: np.ndarray,
) -> float:
    """Accumulate TOPUP's scan-major, z/y/x SSD without changing its order."""

    ssd = 0.0
    for scan_index in range(corrected_scans.shape[3]):
        for z in range(mean.shape[2]):
            for y in range(mean.shape[1]):
                for x in range(mean.shape[0]):
                    if joint_mask[x, y, z]:
                        value = np.float32(
                            mean[x, y, z] - corrected_scans[x, y, z, scan_index]
                        )
                        value64 = np.float64(value)
                        ssd += value64 * value64
    return ssd


@njit(cache=True, parallel=True)
def _field_hessian_fsl_order(
    alpha_squared: np.ndarray,
    alpha_axis: np.ndarray,
    axis_squared: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
    derivative_z: np.ndarray,
    support_x: np.ndarray,
    support_y: np.ndarray,
    support_z: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
) -> np.ndarray:
    """Assemble the four FSL Gauss-Newton spline products in one pass."""

    coefficient_x = basis_x.shape[1]
    coefficient_y = basis_y.shape[1]
    coefficient_z = basis_z.shape[1]
    count = coefficient_x * coefficient_y * coefficient_z
    result = np.zeros((count, count), dtype=np.float64)
    for pair_index in prange(pair_left.size):
        left = pair_left[pair_index]
        right = pair_right[pair_index]
        left_x = left % coefficient_x
        left_y = (left // coefficient_x) % coefficient_y
        left_z = left // (coefficient_x * coefficient_y)
        right_x = right % coefficient_x
        right_y = (right // coefficient_x) % coefficient_y
        right_z = right // (coefficient_x * coefficient_y)
        direct = 0.0
        interaction = 0.0
        transposed_interaction = 0.0
        derivative = 0.0
        z_start = max(support_z[left_z, 0], support_z[right_z, 0])
        z_stop = min(support_z[left_z, 1], support_z[right_z, 1])
        y_start = max(support_y[left_y, 0], support_y[right_y, 0])
        y_stop = min(support_y[left_y, 1], support_y[right_y, 1])
        x_start = max(support_x[left_x, 0], support_x[right_x, 0])
        x_stop = min(support_x[left_x, 1], support_x[right_x, 1])
        for z in range(z_start, z_stop):
            left_bz = basis_z[z, left_z]
            right_bz = basis_z[z, right_z]
            left_dz = derivative_z[z, left_z]
            right_dz = derivative_z[z, right_z]
            if (left_bz == 0.0 and left_dz == 0.0) or (
                right_bz == 0.0 and right_dz == 0.0
            ):
                continue
            for y in range(y_start, y_stop):
                left_by = basis_y[y, left_y]
                right_by = basis_y[y, right_y]
                left_dy = derivative_y[y, left_y]
                right_dy = derivative_y[y, right_y]
                if (left_by == 0.0 and left_dy == 0.0) or (
                    right_by == 0.0 and right_dy == 0.0
                ):
                    continue
                for x in range(x_start, x_stop):
                    left_b = left_bz * left_by * basis_x[x, left_x]
                    right_b = right_bz * right_by * basis_x[x, right_x]
                    left_d = left_dz * left_dy * derivative_x[x, left_x]
                    right_d = right_dz * right_dy * derivative_x[x, right_x]
                    direct += left_b * alpha_squared[x, y, z] * right_b
                    interaction += left_b * alpha_axis[x, y, z] * right_d
                    transposed_interaction += left_d * alpha_axis[x, y, z] * right_b
                    derivative += left_d * axis_squared[x, y, z] * right_d
        result[left, right] = direct + interaction + transposed_interaction + derivative
    return result


@njit(cache=True)
def _spline_union_support_bounds(
    basis: np.ndarray, derivative: np.ndarray
) -> np.ndarray:
    """Return half-open nonzero support bounds for each spline coefficient."""

    bounds = np.empty((basis.shape[1], 2), dtype=np.int64)
    for coefficient in range(basis.shape[1]):
        start = basis.shape[0]
        stop = 0
        for voxel in range(basis.shape[0]):
            if (
                basis[voxel, coefficient] != 0.0
                or derivative[voxel, coefficient] != 0.0
            ):
                start = min(start, voxel)
                stop = voxel + 1
        bounds[coefficient, 0] = start
        bounds[coefficient, 1] = stop
    return bounds


@lru_cache(maxsize=32)
def _spline_hessian_workset(
    field_shape: tuple[int, int, int],
    knot_spacing: tuple[int, int, int],
    phase_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache independent nonzero Hessian elements and their spline supports."""

    basis = tuple(
        spline_design_matrix(field_shape[axis], knot_spacing[axis]) for axis in range(3)
    )
    derivative_basis = tuple(
        spline_design_matrix(
            field_shape[axis],
            knot_spacing[axis],
            derivative=1 if axis == phase_axis else 0,
        )
        for axis in range(3)
    )
    coefficient_shape = tuple(matrix.shape[1] for matrix in basis)
    pair_left, pair_right = _spline_hessian_pairs(coefficient_shape)
    supports = tuple(
        _spline_union_support_bounds(basis[axis], derivative_basis[axis])
        for axis in range(3)
    )
    for array in (*supports, pair_left, pair_right):
        array.setflags(write=False)
    return (*supports, pair_left, pair_right)


@lru_cache(maxsize=64)
def _balanced_spline_hessian_workset(
    field_shape: tuple[int, int, int],
    knot_spacing: tuple[int, int, int],
    phase_axis: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distribute independent Hessian elements evenly across worker chunks."""

    support_x, support_y, support_z, pair_left, pair_right = (
        _spline_hessian_workset(field_shape, knot_spacing, phase_axis)
    )
    if workers <= 1:
        return support_x, support_y, support_z, pair_left, pair_right
    nx, ny, _nz = (support_x.shape[0], support_y.shape[0], support_z.shape[0])
    left_x = pair_left % nx
    left_y = (pair_left // nx) % ny
    left_z = pair_left // (nx * ny)
    right_x = pair_right % nx
    right_y = (pair_right // nx) % ny
    right_z = pair_right // (nx * ny)
    overlap_x = np.minimum(support_x[left_x, 1], support_x[right_x, 1]) - np.maximum(
        support_x[left_x, 0], support_x[right_x, 0]
    )
    overlap_y = np.minimum(support_y[left_y, 1], support_y[right_y, 1]) - np.maximum(
        support_y[left_y, 0], support_y[right_y, 0]
    )
    overlap_z = np.minimum(support_z[left_z, 1], support_z[right_z, 1]) - np.maximum(
        support_z[left_z, 0], support_z[right_z, 0]
    )
    work = overlap_x * overlap_y * overlap_z
    descending = np.argsort(-work, kind="stable")
    order = np.concatenate(tuple(descending[index::workers] for index in range(workers)))
    balanced_left = np.ascontiguousarray(pair_left[order])
    balanced_right = np.ascontiguousarray(pair_right[order])
    balanced_left.setflags(write=False)
    balanced_right.setflags(write=False)
    return support_x, support_y, support_z, balanced_left, balanced_right


@njit(cache=True)
def _spline_hessian_pairs(
    coefficient_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ordered list of potentially overlapping coefficient pairs."""

    pair_count = 0
    for left_z in range(coefficient_shape[2]):
        z_count = min(coefficient_shape[2], left_z + SPLINE_ORDER + 1) - max(
            0, left_z - SPLINE_ORDER
        )
        for left_y in range(coefficient_shape[1]):
            y_count = min(coefficient_shape[1], left_y + SPLINE_ORDER + 1) - max(
                0, left_y - SPLINE_ORDER
            )
            for left_x in range(coefficient_shape[0]):
                x_count = min(coefficient_shape[0], left_x + SPLINE_ORDER + 1) - max(
                    0, left_x - SPLINE_ORDER
                )
                pair_count += x_count * y_count * z_count
    pair_left = np.empty(pair_count, dtype=np.int64)
    pair_right = np.empty(pair_count, dtype=np.int64)
    output = 0
    nx, ny, nz = coefficient_shape
    for left in range(nx * ny * nz):
        left_x = left % nx
        left_y = (left // nx) % ny
        left_z = left // (nx * ny)
        for right_z in range(
            max(0, left_z - SPLINE_ORDER), min(nz, left_z + SPLINE_ORDER + 1)
        ):
            for right_y in range(
                max(0, left_y - SPLINE_ORDER), min(ny, left_y + SPLINE_ORDER + 1)
            ):
                for right_x in range(
                    max(0, left_x - SPLINE_ORDER), min(nx, left_x + SPLINE_ORDER + 1)
                ):
                    pair_left[output] = left
                    pair_right[output] = right_z * nx * ny + right_y * nx + right_x
                    output += 1
    return pair_left, pair_right


class TopupFixedMovementObjective:
    """FSL TOPUP data, gradient and regularizer terms with movements fixed."""

    def __init__(
        self,
        scans: np.ndarray,
        acquisition_parameters: np.ndarray,
        voxel_sizes_mm: tuple[float, float, float],
        warp_resolution_mm: float,
        regularization_weight: float,
        *,
        fixed_movements: np.ndarray | None = None,
        target_shape: tuple[int, int, int] | None = None,
        source_voxel_sizes_mm: tuple[float, float, float] | None = None,
    ) -> None:
        values = np.asarray(scans, dtype=np.float32)
        if values.ndim != 4 or values.shape[3] < 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                "scans must be a finite 4D array with at least two volumes"
            )
        rows = validate_acquisition_parameters(
            acquisition_parameters, number_of_volumes=values.shape[3]
        )
        axes = np.flatnonzero(rows[0, :3])
        row_axes = np.argmax(np.abs(rows[:, :3]), axis=1)
        if axes.size != 1 or np.any(row_axes != axes[0]):
            raise ValueError("the fixed objective requires one shared PE axis")
        if not np.isfinite(regularization_weight) or regularization_weight < 0.0:
            raise ValueError("regularization weight must be nonnegative and finite")
        self.scans = np.ascontiguousarray(values)
        self.acquisition_parameters = rows
        self.voxel_sizes_mm = tuple(float(value) for value in voxel_sizes_mm)
        self.field_shape = (
            tuple(int(value) for value in target_shape)
            if target_shape is not None
            else self.scans.shape[:3]
        )
        if len(self.field_shape) != 3 or any(value < 1 for value in self.field_shape):
            raise ValueError("target shape must contain three positive values")
        self.source_voxel_sizes_mm = (
            tuple(float(value) for value in source_voxel_sizes_mm)
            if source_voxel_sizes_mm is not None
            else self.voxel_sizes_mm
        )
        source_sizes = np.asarray(self.source_voxel_sizes_mm, dtype=np.float64)
        if (
            source_sizes.shape != (3,)
            or not np.all(np.isfinite(source_sizes))
            or np.any(source_sizes <= 0.0)
        ):
            raise ValueError("source voxel sizes must contain three positive values")
        self.knot_spacing = fsl_knot_spacing(warp_resolution_mm, self.voxel_sizes_mm)
        self.coefficient_shape = fsl_coefficient_shape(
            self.field_shape, self.knot_spacing
        )
        self.regularization_weight = float(regularization_weight)
        if fixed_movements is None:
            movements = np.zeros((self.scans.shape[3], 6), dtype=np.float64)
        else:
            movements = np.asarray(fixed_movements, dtype=np.float64)
            if movements.shape != (self.scans.shape[3], 6) or not np.all(
                np.isfinite(movements)
            ):
                raise ValueError("fixed movements must have shape (N, 6) and be finite")
            movements = movements.copy()
        self.fixed_movements = movements
        self.prepared_scans = tuple(
            prepare_topup_scan(self.scans[..., index])
            for index in range(self.scans.shape[3])
        )
        self.interpolation_coefficients = np.stack(
            tuple(
                prepared.interpolation_coefficients
                for prepared in self.prepared_scans
            ),
            axis=3,
        )
        self.phase_axis = int(axes[0])
        self.basis = tuple(
            spline_design_matrix(self.field_shape[axis], self.knot_spacing[axis])
            for axis in range(3)
        )
        self.derivative_basis = tuple(
            spline_design_matrix(
                self.field_shape[axis],
                self.knot_spacing[axis],
                derivative=1 if axis == self.phase_axis else 0,
            )
            for axis in range(3)
        )
        self.hessian_workset = _balanced_spline_hessian_workset(
            self.field_shape,
            self.knot_spacing,
            self.phase_axis,
            get_num_threads(),
        )
        self.regularization_hessian = bending_energy_hessian(
            self.field_shape, self.voxel_sizes_mm, self.knot_spacing
        )
        self._regularization_hessian_csr = self.regularization_hessian.tocsr()
        voxel_grid = np.indices(self.field_shape, dtype=np.float64).reshape(3, -1)
        self.homogeneous_grid = np.vstack(
            (voxel_grid, np.ones((1, voxel_grid.shape[1]), dtype=np.float64))
        )
        self._state: TopupFixedMovementState | None = None
        self._regularization_parameters: np.ndarray | None = None
        self._regularization_product: np.ndarray | None = None
        self._dense_regularization_hessian: np.ndarray | None = None

    @property
    def number_of_parameters(self) -> int:
        """Return the number of field coefficients at this resolution."""

        return int(np.prod(self.coefficient_shape))

    def _apply_regularization_hessian(self, parameters: np.ndarray) -> np.ndarray:
        """Reuse an exact sparse product shared by adjacent cost/gradient calls."""

        if self._regularization_parameters is None or not np.array_equal(
            parameters, self._regularization_parameters
        ):
            self._regularization_parameters = parameters.copy()
            self._regularization_product = np.asarray(
                self._regularization_hessian_csr @ parameters, dtype=np.float64
            )
        assert self._regularization_product is not None
        return self._regularization_product

    def _evaluate(self, parameters: np.ndarray) -> TopupFixedMovementState:
        vector = np.asarray(parameters, dtype=np.float64)
        if vector.shape != (self.number_of_parameters,) or not np.all(
            np.isfinite(vector)
        ):
            raise ValueError("parameters must be a finite field-coefficient vector")
        if self._state is not None and np.array_equal(vector, self._state.parameters):
            return self._state
        coefficients = vector.reshape(self.coefficient_shape, order="F")
        field = expand_spline_coefficients(
            coefficients, self.field_shape, self.knot_spacing
        ).astype(np.float32)
        field_derivative = expand_spline_coefficients(
            coefficients,
            self.field_shape,
            self.knot_spacing,
            derivative_axis=self.phase_axis,
        ).astype(np.float32)
        output_shape = (*self.field_shape, self.scans.shape[3])
        corrected_scans = np.empty(output_shape, dtype=np.float32)
        alpha = np.empty(output_shape, dtype=np.float32)
        axis_term = np.empty(output_shape, dtype=np.float32)
        displacements = np.empty(output_shape, dtype=np.float32)
        jacobians = np.empty(output_shape, dtype=np.float32)
        pull_matrices = np.empty((self.scans.shape[3], 4, 4), dtype=np.float64)
        source_scales = np.empty(self.scans.shape[3], dtype=np.float32)
        scales = np.empty(self.scans.shape[3], dtype=np.float32)
        for index in range(self.scans.shape[3]):
            row = self.acquisition_parameters[index]
            scale = np.float32(row[self.phase_axis] * row[3])
            scales[index] = scale
            jacobians[..., index] = np.float32(1.0) + scale * field_derivative
            pull_matrices[index] = _topup_source_pull_matrix(
                self.fixed_movements[index],
                self.scans.shape[:3],
                self.field_shape,
                self.source_voxel_sizes_mm,
                self.voxel_sizes_mm,
            )
            source_scale = np.float32(
                scale
                * self.voxel_sizes_mm[self.phase_axis]
                / self.source_voxel_sizes_mm[self.phase_axis]
            )
            source_scales[index] = source_scale
            displacements[..., index] = source_scale * field
        sampled, derivative_x, derivative_y, _derivative_z, masks = (
            _sample_fsl_periodic_cubic_affine_batch(
                self.interpolation_coefficients,
                displacements,
                self.phase_axis,
                pull_matrices,
            )
        )
        for index in range(self.scans.shape[3]):
            derivative = (
                derivative_x[..., index]
                if self.phase_axis == 0
                else derivative_y[..., index]
            )
            corrected_scans[..., index] = sampled[..., index] * jacobians[..., index]
            alpha[..., index] = (
                source_scales[index] * derivative * jacobians[..., index]
            )
            axis_term[..., index] = scales[index] * sampled[..., index]
            for other_axis in range(3):
                if other_axis == self.phase_axis:
                    continue
                lower = [slice(None)] * 3
                upper = [slice(None)] * 3
                lower[other_axis] = 0
                upper[other_axis] = -1
                masks[(*lower, index)] = 0
                masks[(*upper, index)] = 0
        joint_mask = np.prod(masks, axis=3, dtype=np.uint8)
        mean = corrected_scans[..., 0].copy()
        mean_alpha = alpha[..., 0].copy()
        mean_axis_term = axis_term[..., 0].copy()
        for index in range(1, self.scans.shape[3]):
            mean += corrected_scans[..., index]
            mean_alpha += alpha[..., index]
            mean_axis_term += axis_term[..., index]
        count = np.float32(self.scans.shape[3])
        mean /= count
        mean_alpha /= count
        mean_axis_term /= count
        valid_count = int(np.count_nonzero(joint_mask))
        ssd = _topup_ssd_fsl_order(corrected_scans, mean, joint_mask) / (
            valid_count * (self.scans.shape[3] - 1)
        )
        self._state = TopupFixedMovementState(
            vector.copy(),
            corrected_scans,
            alpha,
            axis_term,
            joint_mask,
            mean,
            mean_alpha,
            mean_axis_term,
            ssd,
        )
        return self._state

    def cost(self, parameters: np.ndarray) -> float:
        """Return FSL's SSD plus SSD-scaled bending-energy penalty."""

        state = self._evaluate(parameters)
        vector = state.parameters
        energy = 0.5 * float(vector @ self._apply_regularization_hessian(vector))
        return state.ssd + state.ssd * self.regularization_weight * energy

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Return FSL's analytic fixed-movement field gradient."""

        state = self._evaluate(parameters)
        field_term = np.zeros(self.field_shape, dtype=np.float32)
        derivative_term = np.zeros_like(field_term)
        for index in range(self.scans.shape[3]):
            difference = state.mean - state.corrected_scans[..., index]
            field_term += (state.mean_alpha - state.alpha[..., index]) * difference
            derivative_term += (
                state.mean_axis_term - state.axis_term[..., index]
            ) * difference
        direct, derivative = _paired_spline_jte_fsl_order(
            field_term,
            derivative_term,
            state.joint_mask,
            *self.basis,
            *self.derivative_basis,
        )
        valid_count = int(np.count_nonzero(state.joint_mask))
        scale = 2.0 / (valid_count * (self.scans.shape[3] - 1))
        result = scale * (direct + derivative)
        vector = state.parameters
        result = result.reshape(-1, order="F")
        result += (
            state.ssd
            * self.regularization_weight
            * self._apply_regularization_hessian(vector)
        )
        return result

    def hessian(self, parameters: np.ndarray) -> np.ndarray:
        """Return FSL's fixed-movement Gauss-Newton field Hessian."""

        state = self._evaluate(parameters)
        alpha_squared = np.zeros(self.field_shape, dtype=np.float32)
        alpha_axis = np.zeros_like(alpha_squared)
        axis_squared = np.zeros_like(alpha_squared)
        for index in range(self.scans.shape[3]):
            alpha_difference = state.mean_alpha - state.alpha[..., index]
            axis_difference = state.mean_axis_term - state.axis_term[..., index]
            alpha_squared += alpha_difference * alpha_difference
            alpha_axis += alpha_difference * axis_difference
            axis_squared += axis_difference * axis_difference
        mask = state.joint_mask.astype(bool)
        alpha_squared[~mask] = 0.0
        alpha_axis[~mask] = 0.0
        axis_squared[~mask] = 0.0

        data_hessian = _field_hessian_fsl_order(
            alpha_squared,
            alpha_axis,
            axis_squared,
            *self.basis,
            *self.derivative_basis,
            *self.hessian_workset,
        )
        valid_count = int(np.count_nonzero(state.joint_mask))
        scale = 2.0 / (valid_count * (self.scans.shape[3] - 1))
        result = scale * data_hessian
        if self._dense_regularization_hessian is None:
            self._dense_regularization_hessian = self.regularization_hessian.toarray()
        result += (
            state.ssd * self.regularization_weight * self._dense_regularization_hessian
        )
        return result


@njit(cache=True)
def _fsl_atan2_float32(y: np.float32, x: np.float32) -> np.float32:
    """Use the scalar libm float path selected by FSL's C++ overload."""

    return np.arctan2(y, x)


def _fsl_matrix_multiply_4x4(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two 4x4 matrices in NEWMAT's scalar accumulation order."""

    result = np.zeros((4, 4), dtype=np.float64)
    for row in range(4):
        for column in range(4):
            for inner in range(4):
                result[row, column] += left[row, inner] * right[inner, column]
    return result


def _fsl_affine_inverse_4x4(matrix: np.ndarray) -> np.ndarray:
    """Invert an affine 4x4 matrix with Armadillo 9.700 ``inv_tiny`` order."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be one finite 4x4 array")
    if not np.array_equal(values[3], np.asarray((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("matrix must have an affine final row")

    determinant = (
        -values[0, 2] * values[1, 1] * values[2, 0]
        + values[0, 1] * values[1, 2] * values[2, 0]
        + values[0, 2] * values[1, 0] * values[2, 1]
        - values[0, 0] * values[1, 2] * values[2, 1]
        - values[0, 1] * values[1, 0] * values[2, 2]
        + values[0, 0] * values[1, 1] * values[2, 2]
    )
    if abs(determinant) < np.finfo(np.float64).eps:
        raise ValueError("matrix is singular")

    inverse = np.zeros((4, 4), dtype=np.float64)
    inverse[0, 0] = (
        -values[1, 2] * values[2, 1] + values[1, 1] * values[2, 2]
    ) / determinant
    inverse[0, 1] = (
        values[0, 2] * values[2, 1] - values[0, 1] * values[2, 2]
    ) / determinant
    inverse[0, 2] = (
        -values[0, 2] * values[1, 1] + values[0, 1] * values[1, 2]
    ) / determinant
    inverse[1, 0] = (
        values[1, 2] * values[2, 0] - values[1, 0] * values[2, 2]
    ) / determinant
    inverse[1, 1] = (
        -values[0, 2] * values[2, 0] + values[0, 0] * values[2, 2]
    ) / determinant
    inverse[1, 2] = (
        values[0, 2] * values[1, 0] - values[0, 0] * values[1, 2]
    ) / determinant
    inverse[2, 0] = (
        -values[1, 1] * values[2, 0] + values[1, 0] * values[2, 1]
    ) / determinant
    inverse[2, 1] = (
        values[0, 1] * values[2, 0] - values[0, 0] * values[2, 1]
    ) / determinant
    inverse[2, 2] = (
        -values[0, 1] * values[1, 0] + values[0, 0] * values[1, 1]
    ) / determinant
    inverse[0, 3] = (
        values[0, 3] * values[1, 2] * values[2, 1]
        - values[0, 2] * values[1, 3] * values[2, 1]
        - values[0, 3] * values[1, 1] * values[2, 2]
        + values[0, 1] * values[1, 3] * values[2, 2]
        + values[0, 2] * values[1, 1] * values[2, 3]
        - values[0, 1] * values[1, 2] * values[2, 3]
    ) / determinant
    inverse[1, 3] = (
        values[0, 2] * values[1, 3] * values[2, 0]
        - values[0, 3] * values[1, 2] * values[2, 0]
        + values[0, 3] * values[1, 0] * values[2, 2]
        - values[0, 0] * values[1, 3] * values[2, 2]
        - values[0, 2] * values[1, 0] * values[2, 3]
        + values[0, 0] * values[1, 2] * values[2, 3]
    ) / determinant
    inverse[2, 3] = (
        values[0, 3] * values[1, 1] * values[2, 0]
        - values[0, 1] * values[1, 3] * values[2, 0]
        - values[0, 3] * values[1, 0] * values[2, 1]
        + values[0, 0] * values[1, 3] * values[2, 1]
        + values[0, 1] * values[1, 0] * values[2, 3]
        - values[0, 0] * values[1, 1] * values[2, 3]
    ) / determinant
    inverse[3, 3] = 1.0
    return inverse


def _topup_movement_matrix(
    movement: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Convert TOPUP's translation/rotation parameters to its rigid matrix."""

    parameters = np.asarray(movement, dtype=np.float64)
    if parameters.shape != (6,) or not np.all(np.isfinite(parameters)):
        raise ValueError("movement parameters must contain six finite values")
    return _cached_topup_movement_matrix(
        tuple(float(value) for value in parameters),
        tuple(int(value) for value in shape),
        tuple(float(value) for value in voxel_sizes_mm),
    ).copy()


def _topup_matrix_to_movement_parameters(
    matrix: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Convert one rigid matrix with FSL ``Matrix2MovePar`` arithmetic."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be one finite 4x4 array")

    # FSL's rotmat2euler stores every intermediate in float before writing the
    # three angles into its double-precision NEWMAT output vector.
    cy = np.float32(
        np.sqrt(values[0, 0] * values[0, 0] + values[0, 1] * values[0, 1])
    )
    if cy < np.float32(1.0e-4):
        cx = np.float32(values[1, 1])
        sx = np.float32(-values[2, 1])
        sy = np.float32(-values[0, 2])
        rx = _fsl_atan2_float32(sx, cx)
        ry = _fsl_atan2_float32(sy, np.float32(0.0))
        rz = np.float32(0.0)
    else:
        cz = np.float32(values[0, 0] / cy)
        sz = np.float32(values[0, 1] / cy)
        cx = np.float32(values[2, 2] / cy)
        sx = np.float32(values[1, 2] / cy)
        sy = np.float32(-values[0, 2])
        rx = _fsl_atan2_float32(sx, cx)
        ry = _fsl_atan2_float32(sy, cy)
        rz = _fsl_atan2_float32(sz, cz)

    movement = np.zeros(6, dtype=np.float64)
    movement[3:] = (rx, ry, rz)
    rotation_about_center = _topup_movement_matrix(
        movement, shape, voxel_sizes_mm
    )
    movement[:3] = values[:3, 3] - rotation_about_center[:3, 3]
    return movement


@lru_cache(maxsize=1024)
def _cached_topup_movement_matrix(
    parameters: tuple[float, float, float, float, float, float],
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Cache repeated FSL rigid matrices across rejected optimizer states."""

    parameter_values = np.asarray(parameters, dtype=np.float64)
    center = (
        0.5
        * (np.asarray(shape, dtype=np.float64) - 1.0)
        * np.asarray(voxel_sizes_mm, dtype=np.float64)
    )

    def axis_rotation(axis: int, angle: float) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        angles = np.zeros(3, dtype=np.float64)
        angles[axis] = angle
        theta = np.float32(np.linalg.norm(angles))
        if theta < np.float32(1.0e-8):
            return matrix
        first = angles / float(theta)
        second = np.asarray([-first[1], first[0], 0.0])
        if np.linalg.norm(second) <= 0.0:
            second = np.asarray([1.0, 0.0, 0.0])
        second /= np.linalg.norm(second)
        third = np.cross(first, second)
        third /= np.linalg.norm(third)
        basis = np.column_stack((second, third, first))
        planar = np.eye(3, dtype=np.float64)
        # FSL stores theta as float and resolves the float overloads of
        # cos/sin before assigning the results into its double NEWMAT matrix.
        cosine = np.float32(np.cos(theta))
        sine = np.float32(np.sin(theta))
        planar[0, 0] = cosine
        planar[1, 1] = planar[0, 0]
        planar[0, 1] = sine
        planar[1, 0] = -planar[0, 1]
        core = basis @ planar @ basis.T
        matrix[:3, :3] = core
        matrix[:3, 3] = (np.eye(3) - core) @ center
        return matrix

    matrix = axis_rotation(0, parameter_values[3])
    matrix = _fsl_matrix_multiply_4x4(
        matrix, axis_rotation(1, parameter_values[4])
    )
    matrix = _fsl_matrix_multiply_4x4(
        matrix, axis_rotation(2, parameter_values[5])
    )
    matrix[:3, 3] += parameter_values[:3]
    return matrix


@lru_cache(maxsize=4096)
def _cached_topup_inverse_movement_matrix(
    movement: tuple[float, float, float, float, float, float],
    source_shape: tuple[int, int, int],
    source_voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Cache the identical rigid inverse shared by source and target pulls."""

    movement_matrix = _topup_movement_matrix(
        np.asarray(movement, dtype=np.float64), source_shape, source_voxel_sizes_mm
    )
    return _fsl_affine_inverse_4x4(movement_matrix)


def _topup_source_pull_matrix(
    movement: np.ndarray,
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    source_voxel_sizes_mm: tuple[float, float, float],
    target_voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    parameters = np.asarray(movement, dtype=np.float64)
    return _cached_topup_source_pull_matrix(
        tuple(float(value) for value in parameters),
        tuple(int(value) for value in source_shape),
        tuple(int(value) for value in target_shape),
        tuple(float(value) for value in source_voxel_sizes_mm),
        tuple(float(value) for value in target_voxel_sizes_mm),
    ).copy()


@lru_cache(maxsize=4096)
def _cached_topup_source_pull_matrix(
    movement: tuple[float, float, float, float, float, float],
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    source_voxel_sizes_mm: tuple[float, float, float],
    target_voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Cache pull matrices reused by fixed motion and rejected field states."""

    del target_shape
    source_sampling = np.diag((*source_voxel_sizes_mm, 1.0))
    target_sampling = np.diag((*target_voxel_sizes_mm, 1.0))
    inverse_movement = _cached_topup_inverse_movement_matrix(
        movement, source_shape, source_voxel_sizes_mm
    )
    return np.linalg.inv(source_sampling) @ inverse_movement @ target_sampling


def _topup_target_pull_matrix(
    movement: np.ndarray,
    source_shape: tuple[int, int, int],
    source_voxel_sizes_mm: tuple[float, float, float],
    target_voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Return FSL's target-voxel movement matrix used by its derivative."""

    parameters = np.asarray(movement, dtype=np.float64)
    return _cached_topup_target_pull_matrix(
        tuple(float(value) for value in parameters),
        tuple(int(value) for value in source_shape),
        tuple(float(value) for value in source_voxel_sizes_mm),
        tuple(float(value) for value in target_voxel_sizes_mm),
    ).copy()


@lru_cache(maxsize=4096)
def _cached_topup_target_pull_matrix(
    movement: tuple[float, float, float, float, float, float],
    source_shape: tuple[int, int, int],
    source_voxel_sizes_mm: tuple[float, float, float],
    target_voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Cache target-grid derivative matrices without changing inversion math."""

    target_sampling = np.diag((*target_voxel_sizes_mm, 1.0))
    inverse_movement = _cached_topup_inverse_movement_matrix(
        movement, source_shape, source_voxel_sizes_mm
    )
    return np.linalg.inv(target_sampling) @ inverse_movement @ target_sampling


def _topup_voxel_pull_matrix(
    movement: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Return the legacy same-grid form used by focused oracle tests."""

    return _topup_source_pull_matrix(
        movement,
        shape,
        shape,
        voxel_sizes_mm,
        voxel_sizes_mm,
    )


class TopupMovingObjective(TopupFixedMovementObjective):
    """FSL TOPUP joint field/movement objective for one reverse-PE pair."""

    def __init__(
        self,
        scans: np.ndarray,
        acquisition_parameters: np.ndarray,
        voxel_sizes_mm: tuple[float, float, float],
        warp_resolution_mm: float,
        regularization_weight: float,
        *,
        target_shape: tuple[int, int, int] | None = None,
        source_voxel_sizes_mm: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__(
            scans,
            acquisition_parameters,
            voxel_sizes_mm,
            warp_resolution_mm,
            regularization_weight,
            target_shape=target_shape,
            source_voxel_sizes_mm=source_voxel_sizes_mm,
        )
        if self.scans.shape[3] != 2:
            raise ValueError("the moving fixed subset requires exactly two scans")
        self.movement_indices = (
            (1, 2, 3, 4, 5) if self.phase_axis == 0 else (0, 2, 3, 4, 5)
        )
        self._moving_state: TopupMovingState | None = None

    @property
    def number_of_parameters(self) -> int:
        """Return field coefficients plus five identifiable movement parameters."""

        return int(np.prod(self.coefficient_shape)) + 5

    @property
    def number_of_field_parameters(self) -> int:
        """Return the leading field-coefficient parameter count."""

        return int(np.prod(self.coefficient_shape))

    def _split_parameters(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(parameters, dtype=np.float64)
        if vector.shape != (self.number_of_parameters,) or not np.all(
            np.isfinite(vector)
        ):
            raise ValueError("parameters must be a finite joint parameter vector")
        movement = np.zeros(6, dtype=np.float64)
        movement[list(self.movement_indices)] = vector[
            self.number_of_field_parameters :
        ]
        return vector, movement

    def _evaluate_moving(self, parameters: np.ndarray) -> TopupMovingState:
        vector, movement = self._split_parameters(parameters)
        if self._moving_state is not None and np.array_equal(
            vector, self._moving_state.parameters
        ):
            return self._moving_state
        field_vector = vector[: self.number_of_field_parameters]
        coefficients = field_vector.reshape(self.coefficient_shape, order="F")
        field = expand_spline_coefficients(
            coefficients, self.field_shape, self.knot_spacing
        ).astype(np.float32)
        field_derivative = expand_spline_coefficients(
            coefficients,
            self.field_shape,
            self.knot_spacing,
            derivative_axis=self.phase_axis,
        ).astype(np.float32)
        output_shape = (*self.field_shape, 2)
        corrected_scans = np.empty(output_shape, dtype=np.float32)
        alpha = np.empty(output_shape, dtype=np.float32)
        axis_term = np.empty(output_shape, dtype=np.float32)
        displacements = np.empty(output_shape, dtype=np.float32)
        jacobians = np.empty(output_shape, dtype=np.float32)
        pull_matrices = np.empty((2, 4, 4), dtype=np.float64)
        source_scales = np.empty(2, dtype=np.float32)
        scales = np.empty(2, dtype=np.float32)
        movement_derivatives = np.empty((*self.field_shape, 5), dtype=np.float32)
        zero_movement = np.zeros(6, dtype=np.float64)
        movements = (zero_movement, movement)
        for index in range(2):
            row = self.acquisition_parameters[index]
            scale = np.float32(row[self.phase_axis] * row[3])
            scales[index] = scale
            jacobians[..., index] = np.float32(1.0) + scale * field_derivative
            pull_matrices[index] = _topup_source_pull_matrix(
                movements[index],
                self.scans.shape[:3],
                self.field_shape,
                self.source_voxel_sizes_mm,
                self.voxel_sizes_mm,
            )
            source_scale = np.float32(
                scale
                * self.voxel_sizes_mm[self.phase_axis]
                / self.source_voxel_sizes_mm[self.phase_axis]
            )
            source_scales[index] = source_scale
            displacements[..., index] = source_scale * field
        sampled, derivative_x, derivative_y, derivative_z, masks = (
            _sample_fsl_periodic_cubic_affine_batch(
                self.interpolation_coefficients,
                displacements,
                self.phase_axis,
                pull_matrices,
            )
        )
        for index in range(2):
            derivative = (
                derivative_x[..., index]
                if self.phase_axis == 0
                else derivative_y[..., index]
            )
            corrected_scans[..., index] = sampled[..., index] * jacobians[..., index]
            alpha[..., index] = (
                source_scales[index] * derivative * jacobians[..., index]
            )
            axis_term[..., index] = scales[index] * sampled[..., index]
            for other_axis in range(3):
                if other_axis == self.phase_axis:
                    continue
                lower = [slice(None)] * 3
                upper = [slice(None)] * 3
                lower[other_axis] = 0
                upper[other_axis] = -1
                masks[(*lower, index)] = 0
                masks[(*upper, index)] = 0

        base_pull = _topup_target_pull_matrix(
            movement,
            self.scans.shape[:3],
            self.source_voxel_sizes_mm,
            self.voxel_sizes_mm,
        )
        base_coordinates = (base_pull @ self.homogeneous_grid)[:3]
        derivative_images = (
            derivative_x[..., 1],
            derivative_y[..., 1],
            derivative_z[..., 1],
        )
        for output_index, parameter_index in enumerate(self.movement_indices):
            tiny = 1.0e-4 if parameter_index < 3 else 1.0e-5
            perturbed = movement.copy()
            perturbed[parameter_index] += tiny
            perturbed_pull = _topup_target_pull_matrix(
                perturbed,
                self.scans.shape[:3],
                self.source_voxel_sizes_mm,
                self.voxel_sizes_mm,
            )
            coordinate_change = (
                (perturbed_pull @ self.homogeneous_grid)[:3] - base_coordinates
            ).reshape((3, *self.field_shape), order="C")
            scale_factors = np.asarray(self.voxel_sizes_mm) / np.asarray(
                self.source_voxel_sizes_mm
            )
            movement_derivative = np.asarray(
                scale_factors[0] * coordinate_change[0] * derivative_images[0]
                + scale_factors[1] * coordinate_change[1] * derivative_images[1]
                + scale_factors[2] * coordinate_change[2] * derivative_images[2],
                dtype=np.float32,
            )
            movement_derivative /= np.float32(tiny)
            movement_derivative *= jacobians[..., 1]
            movement_derivatives[..., output_index] = movement_derivative

        joint_mask = np.prod(masks, axis=3, dtype=np.uint8)
        mean = corrected_scans[..., 0].copy()
        mean += corrected_scans[..., 1]
        mean /= np.float32(2.0)
        mean_alpha = alpha[..., 0].copy()
        mean_alpha += alpha[..., 1]
        mean_alpha /= np.float32(2.0)
        mean_axis_term = axis_term[..., 0].copy()
        mean_axis_term += axis_term[..., 1]
        mean_axis_term /= np.float32(2.0)
        valid_count = int(np.count_nonzero(joint_mask))
        ssd = _topup_ssd_fsl_order(corrected_scans, mean, joint_mask) / valid_count
        self._moving_state = TopupMovingState(
            vector.copy(),
            corrected_scans,
            alpha,
            axis_term,
            joint_mask,
            mean,
            mean_alpha,
            mean_axis_term,
            movement_derivatives,
            ssd,
        )
        return self._moving_state

    def cost(self, parameters: np.ndarray) -> float:
        """Return the joint movement/field TOPUP cost."""

        state = self._evaluate_moving(parameters)
        field_vector = state.parameters[: self.number_of_field_parameters]
        energy = 0.5 * float(
            field_vector @ self._apply_regularization_hessian(field_vector)
        )
        return state.ssd + state.ssd * self.regularization_weight * energy

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Return the joint analytic field and movement gradient."""

        state = self._evaluate_moving(parameters)
        field_term = np.zeros(self.field_shape, dtype=np.float32)
        derivative_term = np.zeros_like(field_term)
        for index in range(2):
            difference = state.mean - state.corrected_scans[..., index]
            field_term += (state.mean_alpha - state.alpha[..., index]) * difference
            derivative_term += (
                state.mean_axis_term - state.axis_term[..., index]
            ) * difference
        direct, derivative = _paired_spline_jte_fsl_order(
            field_term,
            derivative_term,
            state.joint_mask,
            *self.basis,
            *self.derivative_basis,
        )
        valid_count = int(np.count_nonzero(state.joint_mask))
        scale = 2.0 / valid_count
        field_gradient = scale * (direct + derivative)
        field_vector = state.parameters[: self.number_of_field_parameters]
        field_gradient = field_gradient.reshape(-1, order="F")
        field_gradient += (
            state.ssd
            * self.regularization_weight
            * self._apply_regularization_hessian(field_vector)
        )
        movement_gradient = np.empty(5, dtype=np.float64)
        difference = state.mean - state.corrected_scans[..., 1]
        for index in range(5):
            movement_gradient[index] = -scale * _masked_sum_product_fsl_order(
                difference,
                state.movement_derivatives[..., index],
                state.joint_mask,
            )
        return np.concatenate((field_gradient, movement_gradient))

    def hessian(self, parameters: np.ndarray) -> np.ndarray:
        """Return FSL's joint Gauss-Newton field/movement Hessian."""

        state = self._evaluate_moving(parameters)
        alpha_squared = np.zeros(self.field_shape, dtype=np.float32)
        alpha_axis = np.zeros_like(alpha_squared)
        axis_squared = np.zeros_like(alpha_squared)
        for index in range(2):
            alpha_difference = state.mean_alpha - state.alpha[..., index]
            axis_difference = state.mean_axis_term - state.axis_term[..., index]
            alpha_squared += alpha_difference * alpha_difference
            alpha_axis += alpha_difference * axis_difference
            axis_squared += axis_difference * axis_difference
        mask = state.joint_mask.astype(bool)
        for image in (alpha_squared, alpha_axis, axis_squared):
            image[~mask] = 0.0

        data_hessian = _field_hessian_fsl_order(
            alpha_squared,
            alpha_axis,
            axis_squared,
            *self.basis,
            *self.derivative_basis,
            *self.hessian_workset,
        )
        valid_count = int(np.count_nonzero(state.joint_mask))
        field_hessian = (2.0 / valid_count) * data_hessian
        if self._dense_regularization_hessian is None:
            self._dense_regularization_hessian = self.regularization_hessian.toarray()
        field_hessian += (
            state.ssd * self.regularization_weight * self._dense_regularization_hessian
        )

        movement_hessian = np.empty((5, 5), dtype=np.float64)
        for row in range(5):
            for column in range(row, 5):
                value = (
                    _masked_sum_product_fsl_order(
                        state.movement_derivatives[..., row],
                        state.movement_derivatives[..., column],
                        state.joint_mask,
                    )
                    / valid_count
                )
                movement_hessian[row, column] = value
                movement_hessian[column, row] = value
        alpha_difference = state.mean_alpha - state.alpha[..., 1]
        axis_difference = state.mean_axis_term - state.axis_term[..., 1]
        direct_columns, derivative_columns = _movement_interaction_jte_fsl_order(
            state.movement_derivatives,
            alpha_difference,
            axis_difference,
            state.joint_mask,
            *self.basis,
            *self.derivative_basis,
        )
        interaction = (-2.0 / valid_count) * (direct_columns + derivative_columns)
        result = np.block(
            [[field_hessian, interaction], [interaction.T, movement_hessian]]
        )
        return result


def fit_spline_coefficients(
    field_hz: np.ndarray,
    knot_spacing: tuple[int, int, int],
) -> np.ndarray:
    """Fit FSL cubic coefficients to a dense field during level changes."""

    field = np.asarray(field_hz, dtype=np.float32)
    if field.ndim != 3 or not np.all(np.isfinite(field)):
        raise ValueError("field must be a finite 3D array")
    coefficient_shape = fsl_coefficient_shape(field.shape, knot_spacing)
    result = field.astype(np.float64)
    for axis in range(3):
        design = spline_design_matrix(field.shape[axis], knot_spacing[axis])
        stabilizer = np.zeros((field.shape[axis], coefficient_shape[axis]))
        stabilizer[0, 0] = 2.0
        stabilizer[0, 1] = -1.0
        stabilizer[0, -1] = -1.0
        stabilizer[-1, 0] = -1.0
        stabilizer[-1, -2] = -1.0
        stabilizer[-1, -1] = 2.0
        augmented = np.vstack((design, 0.005 * stabilizer))
        projector = np.linalg.inv(augmented.T @ augmented) @ design.T
        moved = np.moveaxis(result, axis, 0)
        output_shape = (coefficient_shape[axis], *moved.shape[1:])
        converted = np.empty(output_shape, dtype=np.float64)
        for index in np.ndindex(moved.shape[1:]):
            converted[(slice(None), *index)] = projector @ moved[(slice(None), *index)]
        result = np.moveaxis(converted, 0, axis)
    return result


def run_simnibs46_topup(
    scans: np.ndarray,
    acquisition_parameters: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    *,
    progress: Callable[[int, str], None] | None = None,
) -> TopupRunResult:
    """Run the frozen nine-level SimNIBS 4.6 TOPUP subset."""

    from .topup_optimizer import (
        fsl_levenberg_marquardt,
        fsl_scaled_conjugate_gradient,
    )

    values = np.asarray(scans, dtype=np.float32)
    if values.ndim != 4 or values.shape[3] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("the SimNIBS fixed subset requires two finite 3D scans")
    rows = validate_acquisition_parameters(acquisition_parameters, number_of_volumes=2)
    scaled = np.ascontiguousarray(values.copy())
    input_means = np.empty(2, dtype=np.float64)
    for index in range(2):
        mean = float(np.mean(scaled[..., index], dtype=np.float64))
        if mean == 0.0:
            raise ValueError("TOPUP cannot scale an input volume with zero mean")
        input_means[index] = mean
        scaled[..., index] *= np.float32(100.0 / mean)

    regridded_scans: list[np.ndarray] = []
    regridded_voxel_sizes: tuple[float, float, float] | None = None
    for index in range(2):
        regridded, current_sizes = fsl_regrid_topup_scan(
            scaled[..., index], voxel_sizes_mm
        )
        if regridded_voxel_sizes is not None and current_sizes != regridded_voxel_sizes:
            raise RuntimeError("TOPUP source grids unexpectedly differ between scans")
        regridded_scans.append(regridded)
        regridded_voxel_sizes = current_sizes
    assert regridded_voxel_sizes is not None
    regridded_values = np.stack(regridded_scans, axis=3)

    coefficients: np.ndarray | None = None
    previous_spacing: tuple[int, int, int] | None = None
    movements = np.zeros((2, 6), dtype=np.float64)
    level_results: list[object] = []
    final_objective: TopupFixedMovementObjective | None = None
    for level_index, level in enumerate(SIMNIBS46_TOPUP_LEVELS, start=1):
        if progress is not None:
            progress(level_index, "preparing")
        smoothed = np.stack(
            tuple(
                fsl_smooth_topup_scan(
                    regridded_values[..., scan_index],
                    level.fwhm_mm,
                    regridded_voxel_sizes,
                )
                for scan_index in range(2)
            ),
            axis=3,
        )
        spacing = fsl_knot_spacing(level.warp_resolution_mm, voxel_sizes_mm)
        coefficient_shape = fsl_coefficient_shape(values.shape[:3], spacing)
        if coefficients is None:
            coefficients = np.zeros(coefficient_shape, dtype=np.float64)
        elif spacing != previous_spacing:
            old_field = expand_spline_coefficients(
                coefficients, values.shape[:3], previous_spacing
            ).astype(np.float32)
            coefficients = fit_spline_coefficients(old_field, spacing)

        if level.estimate_movements:
            objective: TopupFixedMovementObjective = TopupMovingObjective(
                smoothed,
                rows,
                voxel_sizes_mm,
                level.warp_resolution_mm,
                level.regularization_weight,
                target_shape=values.shape[:3],
                source_voxel_sizes_mm=regridded_voxel_sizes,
            )
            moving_objective = objective
            assert isinstance(moving_objective, TopupMovingObjective)
            starting = np.concatenate(
                (
                    coefficients.reshape(-1, order="F"),
                    movements[1, list(moving_objective.movement_indices)],
                )
            )
        else:
            objective = TopupFixedMovementObjective(
                smoothed,
                rows,
                voxel_sizes_mm,
                level.warp_resolution_mm,
                level.regularization_weight,
                fixed_movements=movements,
                target_shape=values.shape[:3],
                source_voxel_sizes_mm=regridded_voxel_sizes,
            )
            starting = coefficients.reshape(-1, order="F")
        if progress is not None:
            progress(level_index, "optimizing")
        if level.minimizer == "levenberg_marquardt":
            result = fsl_levenberg_marquardt(
                starting,
                objective.cost,
                objective.gradient,
                objective.hessian,
                max_iterations=level.max_iterations,
            )
        else:
            result = fsl_scaled_conjugate_gradient(
                starting,
                objective.cost,
                objective.gradient,
                max_iterations=level.max_iterations,
            )
        level_results.append(result)
        coefficients = result.parameters[: int(np.prod(coefficient_shape))].reshape(
            coefficient_shape, order="F"
        )
        if level.estimate_movements:
            assert isinstance(objective, TopupMovingObjective)
            movements[1, list(objective.movement_indices)] = result.parameters[
                objective.number_of_field_parameters :
            ]
        previous_spacing = spacing
        final_objective = objective
        if progress is not None:
            progress(level_index, "complete")

    assert coefficients is not None and final_objective is not None
    final_parameters = coefficients.reshape(-1, order="F")
    final_objective.cost(final_parameters)
    final_state = final_objective._state
    if final_state is None:
        raise RuntimeError("final TOPUP state was not evaluated")
    field_hz = expand_spline_coefficients(
        coefficients, values.shape[:3], previous_spacing
    ).astype(np.float32)
    corrected_scans = final_state.corrected_scans.copy()
    common_mean = np.mean(input_means, dtype=np.float64)
    corrected_scans *= np.float32(common_mean / 100.0)
    corrected_scans[final_state.joint_mask == 0, :] = np.float32(0.0)
    return TopupRunResult(
        coefficients,
        field_hz,
        movements,
        corrected_scans,
        final_state.joint_mask.copy(),
        tuple(level_results),
    )


def run_topup_nifti(
    forward_b0_file: str | Path,
    reverse_b0_file: str | Path,
    output_directory: str | Path,
    *,
    readout_seconds: float,
    phase_encoding_direction: str,
    workers: int = 8,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    """Run the fixed TOPUP subset and write its public NIfTI artifacts."""

    import nibabel as nib

    if not np.isfinite(readout_seconds) or readout_seconds <= 0.0:
        raise ValueError("readout seconds must be positive and finite")
    if workers < 1:
        raise ValueError("workers must be positive")
    directions = {
        "x": ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        "x-": ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "y": ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        "y-": ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
    }
    if phase_encoding_direction not in directions:
        raise ValueError("phase encoding direction must be x, x-, y, or y-")
    source_forward = nib.load(str(forward_b0_file))
    source_reverse = nib.load(str(reverse_b0_file))
    if len(source_forward.shape) != 3 or len(source_reverse.shape) != 3:
        raise ValueError("forward and reverse b0 images must share one 3D shape")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    prepared_forward = write_fsl_reoriented(
        forward_b0_file,
        output / "forward_b0_reoriented.nii.gz",
        float32=True,
    )
    prepared_reverse = write_fsl_reoriented(
        reverse_b0_file,
        output / "reverse_b0_reoriented.nii.gz",
        float32=True,
    )
    forward_image = nib.load(str(prepared_forward))
    reverse_image = nib.load(str(prepared_reverse))
    forward = np.asarray(forward_image.dataobj, dtype=np.float32)
    reverse = np.asarray(reverse_image.dataobj, dtype=np.float32)
    if forward.ndim != 3 or reverse.shape != forward.shape:
        raise ValueError("forward and reverse b0 images must share one 3D shape")
    if not np.allclose(forward_image.affine, reverse_image.affine, rtol=0.0, atol=1e-6):
        raise ValueError("forward and reverse b0 images must share one affine")
    voxel_sizes = tuple(float(value) for value in forward_image.header.get_zooms()[:3])
    rows = np.asarray(
        [
            (*direction, readout_seconds)
            for direction in directions[phase_encoding_direction]
        ],
        dtype=np.float64,
    )
    started = perf_counter()
    previous_workers = get_num_threads()
    try:
        set_available_numba_threads(workers)
        result = run_simnibs46_topup(
            np.stack((forward, reverse), axis=3),
            rows,
            voxel_sizes,
            progress=progress,
        )
    finally:
        set_num_threads(previous_workers)
    algorithm_seconds = perf_counter() - started

    header = forward_image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(result.field_hz, forward_image.affine, header),
        output / "field_hz.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(result.corrected_scans, forward_image.affine, header),
        output / "corrected_pair.nii.gz",
    )
    mask_header = forward_image.header.copy()
    mask_header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(
            result.joint_mask.astype(np.uint8), forward_image.affine, mask_header
        ),
        output / "joint_mask.nii.gz",
    )
    final_spacing = fsl_knot_spacing(
        SIMNIBS46_TOPUP_LEVELS[-1].warp_resolution_mm, voxel_sizes
    )
    coefficient_affine = np.eye(4, dtype=np.float64)
    coefficient_affine[0, 0] = final_spacing[0]
    coefficient_affine[1, 1] = final_spacing[1]
    coefficient_affine[2, 2] = final_spacing[2]
    coefficient_affine[:3, 3] = forward.shape
    coefficient_image = nib.Nifti1Image(
        result.field_coefficients.astype(np.float32), coefficient_affine
    )
    coefficient_image.set_qform(coefficient_affine, code=1)
    coefficient_image.set_sform(coefficient_affine, code=0)
    nib.save(coefficient_image, output / "field_coefficients.nii.gz")
    np.savetxt(
        output / "movement_parameters.txt", result.movement_parameters, fmt="%.10g"
    )
    uncorrected_difference = forward.astype(np.float64) - reverse.astype(np.float64)
    corrected_difference = result.corrected_scans[..., 0].astype(
        np.float64
    ) - result.corrected_scans[..., 1].astype(np.float64)
    report: dict[str, object] = {
        "status": "complete",
        "algorithm": "SimNIBS-4.6-b02b0_nosubsamp-fixed-subset",
        "algorithm_seconds": algorithm_seconds,
        "raw_storage_reorientation": "fslreorient2std-compatible-no-interpolation",
        "input_shape": list(forward.shape),
        "voxel_sizes_mm": list(voxel_sizes),
        "phase_encoding_direction": phase_encoding_direction,
        "readout_seconds": readout_seconds,
        "workers": workers,
        "field_hz_range": [float(result.field_hz.min()), float(result.field_hz.max())],
        "joint_mask_voxels": int(np.count_nonzero(result.joint_mask)),
        "uncorrected_pair_l2": float(np.linalg.norm(uncorrected_difference)),
        "corrected_pair_l2": float(np.linalg.norm(corrected_difference)),
        "level_costs": [float(level.cost) for level in result.level_results],
    }
    (output / "topup_qa.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def topup_pair_cost(
    scans: np.ndarray,
    corrected_scans: np.ndarray,
    masks: np.ndarray,
) -> float:
    """Calculate TOPUP's normalized across-scan SSD in FSL accumulation order."""

    original = np.asarray(scans)
    corrected = np.asarray(corrected_scans, dtype=np.float32)
    valid = np.asarray(masks, dtype=np.uint8)
    if (
        original.ndim != 4
        or corrected.shape != original.shape
        or valid.shape != original.shape
    ):
        raise ValueError("scans, corrected scans, and masks must share a 4D shape")
    if original.shape[3] < 2:
        raise ValueError("TOPUP requires at least two scans")
    joint_mask = np.prod(valid, axis=3, dtype=np.uint8)
    count = int(np.count_nonzero(joint_mask))
    if count == 0:
        raise ValueError("TOPUP joint mask is empty")
    mean = np.zeros(original.shape[:3], dtype=np.float32)
    for scan_index in range(original.shape[3]):
        mean += corrected[..., scan_index]
    mean /= np.float32(original.shape[3])
    ssd = _topup_ssd_fsl_order(corrected, mean, joint_mask)
    return ssd / (count * (original.shape[3] - 1))
