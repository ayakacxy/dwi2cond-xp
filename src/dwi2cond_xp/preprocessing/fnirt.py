"""FSL-compatible expansion of cubic FNIRT coefficient fields."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import bmat, csc_matrix, csr_matrix, hstack, tril

from ._numba import set_available_numba_threads
from .fnirt_topology import fsl_constrain_topology, fsl_good_fft_size
from .topup import (
    _spline_jte_fsl_order,
    bending_energy,
    bending_energy_hessian,
    cubic_spline_value,
    expand_spline_coefficients,
    fit_spline_coefficients,
    fsl_coefficient_shape,
    fsl_knot_spacing,
    spline_design_matrix,
)
from .transforms import fsl_voxel_to_scaled_mm


@dataclass(frozen=True)
class FnirtWarpExpansion:
    """Dense FNIRT field components and the analytic nonlinear Jacobian."""

    nonlinear_displacement: np.ndarray
    affine_displacement: np.ndarray
    displacement: np.ndarray
    nonlinear_jacobian: np.ndarray
    nonlinear_jacobian_determinant: np.ndarray
    knot_spacing: tuple[int, int, int]


@dataclass(frozen=True)
class FnirtLevel:
    """One frozen level of the SimNIBS 4.6 FNIRT invocation."""

    subsampling: int
    maximum_iterations: int
    input_fwhm_mm: float
    reference_fwhm_mm: float
    regularization_weight: float
    estimate_intensity: bool


@dataclass(frozen=True)
class FnirtPreparedImages:
    """Globally scaled FNIRT inputs and their implicit/explicit masks."""

    reference: np.ndarray
    moving: np.ndarray
    reference_mask: np.ndarray
    moving_mask: np.ndarray
    reference_mean: float
    moving_mean: float


@dataclass(frozen=True)
class FnirtLevelImages:
    """Smoothed inputs and reference-grid data for one FNIRT level."""

    reference: np.ndarray
    moving: np.ndarray
    reference_mask: np.ndarray
    moving_mask: np.ndarray
    reference_voxel_sizes_mm: tuple[float, float, float]


@dataclass(frozen=True)
class FnirtWarpedMoving:
    """One FNIRT pull result and every mask contributing to its SSD."""

    values: np.ndarray
    coordinates: np.ndarray
    derivatives_per_mm: np.ndarray
    data_mask: np.ndarray
    warped_moving_mask: np.ndarray
    mask: np.ndarray


@njit(cache=True, inline="always")
def _sample_trilinear_fsl(
    source: np.ndarray, x: np.float32, y: np.float32, z: np.float32
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Reproduce ``volume<float>::interp3partial`` with constant padding."""

    ix = int(np.floor(x))
    iy = int(np.floor(y))
    iz = int(np.floor(z))
    dx = np.float32(x - np.float32(ix))
    dy = np.float32(y - np.float32(iy))
    dz = np.float32(z - np.float32(iz))
    neighbours = np.empty(8, dtype=np.float32)
    output = 0
    for ox in range(2):
        for oy in range(2):
            for oz in range(2):
                sx, sy, sz = ix + ox, iy + oy, iz + oz
                neighbours[output] = (
                    source[sx, sy, sz]
                    if 0 <= sx < source.shape[0]
                    and 0 <= sy < source.shape[1]
                    and 0 <= sz < source.shape[2]
                    else np.float32(0.0)
                )
                output += 1
    v000, v001, v010, v011, v100, v101, v110, v111 = neighbours
    one_minus_z = np.float32(1.0 - dz)
    one_minus_y = np.float32(1.0 - dy)
    tmp11 = np.float32(one_minus_z * v000 + dz * v001)
    tmp12 = np.float32(one_minus_z * v010 + dz * v011)
    tmp13 = np.float32(one_minus_z * v100 + dz * v101)
    tmp14 = np.float32(one_minus_z * v110 + dz * v111)
    derivative_x = np.float32(
        one_minus_y * np.float32(tmp13 - tmp11) + dy * np.float32(tmp14 - tmp12)
    )
    derivative_y = np.float32(
        np.float32(1.0 - dx) * np.float32(tmp12 - tmp11)
        + dx * np.float32(tmp14 - tmp13)
    )
    tmp11 = np.float32(one_minus_y * v000 + dy * v010)
    tmp12 = np.float32(one_minus_y * v001 + dy * v011)
    tmp13 = np.float32(one_minus_y * v100 + dy * v110)
    tmp14 = np.float32(one_minus_y * v101 + dy * v111)
    tmp21 = np.float32(np.float32(1.0 - dx) * tmp11 + dx * tmp13)
    tmp22 = np.float32(np.float32(1.0 - dx) * tmp12 + dx * tmp14)
    derivative_z = np.float32(tmp22 - tmp21)
    value = np.float32(one_minus_z * tmp21 + dz * tmp22)
    return value, derivative_x, derivative_y, derivative_z


@njit(cache=True, inline="always")
def _sample_trilinear_value_fsl(
    source: np.ndarray, x: np.float32, y: np.float32, z: np.float32
) -> np.float32:
    """Reproduce ``q_tri_interpolation`` used by ``volume::interpolate``."""

    ix = int(np.floor(x))
    iy = int(np.floor(y))
    iz = int(np.floor(z))
    dx = np.float32(x - np.float32(ix))
    dy = np.float32(y - np.float32(iy))
    dz = np.float32(z - np.float32(iz))
    neighbours = np.empty(8, dtype=np.float32)
    output = 0
    for ox in range(2):
        for oy in range(2):
            for oz in range(2):
                sx, sy, sz = ix + ox, iy + oy, iz + oz
                neighbours[output] = (
                    source[sx, sy, sz]
                    if 0 <= sx < source.shape[0]
                    and 0 <= sy < source.shape[1]
                    and 0 <= sz < source.shape[2]
                    else np.float32(0.0)
                )
                output += 1
    v000, v001, v010, v011, v100, v101, v110, v111 = neighbours
    temp1 = np.float32(np.float32(v100 - v000) * dx + v000)
    temp2 = np.float32(np.float32(v101 - v001) * dx + v001)
    temp3 = np.float32(np.float32(v110 - v010) * dx + v010)
    temp4 = np.float32(np.float32(v111 - v011) * dx + v011)
    temp5 = np.float32(np.float32(temp3 - temp1) * dy + temp1)
    temp6 = np.float32(np.float32(temp4 - temp2) * dy + temp2)
    return np.float32(np.float32(temp6 - temp5) * dz + temp5)


@njit(cache=True)
def _warp_fnirt_fsl_order(
    moving: np.ndarray,
    moving_mask: np.ndarray,
    reference_mask: np.ndarray,
    inverse_affine_sampling: np.ndarray,
    moving_mm_to_voxel: np.ndarray,
    displacement: np.ndarray,
    moving_voxel_sizes_mm: np.ndarray,
    calculate_derivatives: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull moving data, derivatives, and masks in FSL's voxel loop order."""

    nx, ny, nz = reference_mask.shape
    values = np.empty((nx, ny, nz), dtype=np.float32)
    coordinates = np.empty((3, nx, ny, nz), dtype=np.float32)
    derivatives = np.empty((nx, ny, nz, 3), dtype=np.float32)
    data_mask = np.empty((nx, ny, nz), dtype=np.bool_)
    warped_mask = np.empty((nx, ny, nz), dtype=np.bool_)
    total_mask = np.empty((nx, ny, nz), dtype=np.bool_)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                float_x = np.float32(x)
                float_y = np.float32(y)
                float_z = np.float32(z)
                o1 = np.float32(
                    np.float32(
                        np.float32(
                            inverse_affine_sampling[0, 0] * float_x
                            + inverse_affine_sampling[0, 1] * float_y
                        )
                        + inverse_affine_sampling[0, 2] * float_z
                    )
                    + inverse_affine_sampling[0, 3]
                )
                o2 = np.float32(
                    np.float32(
                        np.float32(
                            inverse_affine_sampling[1, 0] * float_x
                            + inverse_affine_sampling[1, 1] * float_y
                        )
                        + inverse_affine_sampling[1, 2] * float_z
                    )
                    + inverse_affine_sampling[1, 3]
                )
                o3 = np.float32(
                    np.float32(
                        np.float32(
                            inverse_affine_sampling[2, 0] * float_x
                            + inverse_affine_sampling[2, 1] * float_y
                        )
                        + inverse_affine_sampling[2, 2] * float_z
                    )
                    + inverse_affine_sampling[2, 3]
                )
                o1 = np.float32(o1 + displacement[x, y, z, 0])
                o2 = np.float32(o2 + displacement[x, y, z, 1])
                o3 = np.float32(o3 + displacement[x, y, z, 2])
                c1 = np.float32(
                    np.float32(
                        np.float32(
                            moving_mm_to_voxel[0, 0] * o1
                            + moving_mm_to_voxel[0, 1] * o2
                        )
                        + moving_mm_to_voxel[0, 2] * o3
                    )
                    + moving_mm_to_voxel[0, 3]
                )
                c2 = np.float32(
                    np.float32(
                        np.float32(
                            moving_mm_to_voxel[1, 0] * o1
                            + moving_mm_to_voxel[1, 1] * o2
                        )
                        + moving_mm_to_voxel[1, 2] * o3
                    )
                    + moving_mm_to_voxel[1, 3]
                )
                c3 = np.float32(
                    np.float32(
                        np.float32(
                            moving_mm_to_voxel[2, 0] * o1
                            + moving_mm_to_voxel[2, 1] * o2
                        )
                        + moving_mm_to_voxel[2, 2] * o3
                    )
                    + moving_mm_to_voxel[2, 3]
                )
                coordinates[0, x, y, z] = c1
                coordinates[1, x, y, z] = c2
                coordinates[2, x, y, z] = c3
                if calculate_derivatives:
                    value, dx, dy, dz = _sample_trilinear_fsl(moving, c1, c2, c3)
                    values[x, y, z] = value
                    derivatives[x, y, z, 0] = np.float32(dx / moving_voxel_sizes_mm[0])
                    derivatives[x, y, z, 1] = np.float32(dy / moving_voxel_sizes_mm[1])
                    derivatives[x, y, z, 2] = np.float32(dz / moving_voxel_sizes_mm[2])
                else:
                    values[x, y, z] = _sample_trilinear_value_fsl(moving, c1, c2, c3)
                    derivatives[x, y, z, 0] = np.float32(0.0)
                    derivatives[x, y, z, 1] = np.float32(0.0)
                    derivatives[x, y, z, 2] = np.float32(0.0)
                inside = (
                    0.0 <= c1 <= moving.shape[0] - 1
                    and 0.0 <= c2 <= moving.shape[1] - 1
                    and 0.0 <= c3 <= moving.shape[2] - 1
                )
                data_mask[x, y, z] = inside
                sampled_mask = _sample_trilinear_value_fsl(moving_mask, c1, c2, c3)
                mask_value = np.uint8(sampled_mask) != 0
                warped_mask[x, y, z] = mask_value
                total_mask[x, y, z] = reference_mask[x, y, z] and inside and mask_value
    return values, coordinates, derivatives, data_mask, warped_mask, total_mask


@dataclass(frozen=True)
class FnirtIntensityMapping:
    """FNIRT global polynomial and local multiplicative bias field."""

    global_coefficients: np.ndarray
    bias_coefficients: np.ndarray
    bias_field: np.ndarray
    bias_knot_spacing: tuple[int, int, int]


@dataclass(frozen=True)
class FnirtCostEvaluation:
    """Separated terms of FNIRT's mean-SSD objective."""

    total: float
    mean_squared_difference: float
    displacement_regularization: float
    intensity_regularization: float
    voxel_count: int
    warped_moving: FnirtWarpedMoving
    scaled_reference: np.ndarray


@dataclass(frozen=True)
class FnirtGradientEvaluation:
    """FNIRT gradient in FSL's displacement, bias, and global order."""

    gradient: np.ndarray
    displacement_gradient: np.ndarray
    bias_gradient: np.ndarray
    global_gradient: np.ndarray
    cost: FnirtCostEvaluation


@dataclass(frozen=True)
class FnirtHessianEvaluation:
    """Sparse FNIRT Gauss-Newton Hessian in FSL parameter order."""

    hessian: csc_matrix
    displacement_hessian: csc_matrix
    intensity_hessian: csc_matrix
    cross_hessian: csc_matrix
    gradient: FnirtGradientEvaluation


@dataclass(frozen=True)
class FnirtLevelOptimization:
    """Optimized parameters and separated fields for one FNIRT level."""

    displacement_coefficients: np.ndarray
    intensity_mapping: FnirtIntensityMapping
    parameters: np.ndarray
    cost: float
    status: str
    successful_iterations: int
    trace: tuple[object, ...]


@dataclass(frozen=True)
class FnirtRunResult:
    """Complete frozen four-level FNIRT result before NIfTI serialization."""

    coefficients: np.ndarray
    intensity_mapping: FnirtIntensityMapping
    expansion: FnirtWarpExpansion
    levels: tuple[FnirtLevelOptimization, ...]
    jacobian_ranges: tuple[tuple[float, float], ...]


SIMNIBS46_FNIRT_LEVELS: Final[tuple[FnirtLevel, ...]] = tuple(
    FnirtLevel(*values)
    for values in zip(
        (8, 4, 2, 2),
        (5, 5, 5, 5),
        (6.0, 4.0, 2.0, 2.0),
        (4.0, 2.0, 0.0, 0.0),
        (240.0, 120.0, 60.0, 60.0),
        (True, True, True, False),
        strict=True,
    )
)


@njit(cache=True)
def _spm_like_mean_fsl_order(values: np.ndarray) -> float:
    """Accumulate x-fastest in the two passes used by FSL."""

    nx, ny, nz = values.shape
    mean = 0.0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                mean += values[x, y, z]
    mean /= nx * ny * nz
    threshold = 0.125 * mean
    mean = 0.0
    count = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if values[x, y, z] > threshold:
                    mean += values[x, y, z]
                    count += 1
    return mean / count if count else np.nan


def _fsl_smooth_kernel(sigma_voxels: float) -> np.ndarray:
    """Build ``newimage::gaussian_kernel1D`` with its float accumulators."""

    radius = int(sigma_voxels - 0.001) * 2 + 3
    values = np.empty(2 * radius + 1, dtype=np.float64)
    total = np.float32(0.0)
    sigma = np.float32(sigma_voxels)
    for output, offset in enumerate(range(-radius, radius + 1)):
        if sigma > np.float32(1e-6):
            value = np.float32(
                np.exp(-(offset * offset) / (2.0 * float(sigma) * float(sigma)))
            )
        else:
            value = np.float32(1.0 if offset == 0 else 0.0)
        values[output] = value
        total = np.float32(total + value)
    values *= 1.0 / float(total)
    return values


@njit(cache=True)
def _convolve_axis_fsl_serial(
    source: np.ndarray, kernel: np.ndarray, axis: int
) -> np.ndarray:
    nx, ny, nz = source.shape
    radius = kernel.size // 2
    output = np.empty_like(source)
    for voxel in range(nx * ny * nz):
        x = voxel % nx
        y = (voxel // nx) % ny
        z = voxel // (nx * ny)
        value = np.float32(0.0)
        for index in range(kernel.size):
            offset = index - radius
            sx, sy, sz = x, y, z
            if axis == 0:
                sx += offset
            elif axis == 1:
                sy += offset
            else:
                sz += offset
            if 0 <= sx < nx and 0 <= sy < ny and 0 <= sz < nz:
                value = np.float32(value + source[sx, sy, sz] * kernel[index])
        output[x, y, z] = value
    return output


@njit(cache=True, parallel=True)
def _convolve_axis_fsl_parallel(
    source: np.ndarray, kernel: np.ndarray, axis: int
) -> np.ndarray:
    nx, ny, nz = source.shape
    radius = kernel.size // 2
    output = np.empty_like(source)
    for voxel in prange(nx * ny * nz):
        x = voxel % nx
        y = (voxel // nx) % ny
        z = voxel // (nx * ny)
        value = np.float32(0.0)
        for index in range(kernel.size):
            offset = index - radius
            sx, sy, sz = x, y, z
            if axis == 0:
                sx += offset
            elif axis == 1:
                sy += offset
            else:
                sz += offset
            if 0 <= sx < nx and 0 <= sy < ny and 0 <= sz < nz:
                value = np.float32(value + source[sx, sy, sz] * kernel[index])
        output[x, y, z] = value
    return output


def fsl_fnirt_smooth(
    values: np.ndarray,
    sigma_mm: float,
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Reproduce the zero-padded separable ``newimage::smooth`` operation."""

    source = np.asarray(values, dtype=np.float32)
    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("FNIRT smoothing requires a finite 3D array")
    if (
        not np.isfinite(sigma_mm)
        or sigma_mm <= 0.0
        or voxel_sizes.shape != (3,)
        or not np.all(np.isfinite(voxel_sizes))
        or np.any(voxel_sizes <= 0.0)
    ):
        raise ValueError("sigma and three voxel sizes must be positive and finite")
    convolve = (
        _convolve_axis_fsl_parallel
        if source.size >= 100_000
        else _convolve_axis_fsl_serial
    )
    output = source
    for axis in range(3):
        output = convolve(
            output, _fsl_smooth_kernel(float(sigma_mm / voxel_sizes[axis])), axis
        )
    return output


def fsl_fnirt_bias_knot_spacing(
    voxel_sizes_mm: tuple[float, float, float],
    subsampling: int,
    *,
    bias_resolution_mm: float = 50.0,
) -> tuple[int, int, int]:
    """Return the fixed-path bias-field spacing at one pyramid level."""

    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if (
        voxel_sizes.shape != (3,)
        or not np.all(np.isfinite(voxel_sizes))
        or np.any(voxel_sizes <= 0.0)
        or not np.isfinite(bias_resolution_mm)
        or bias_resolution_mm <= 0.0
    ):
        raise ValueError("bias resolution and three voxel sizes must be positive")
    fsl_fnirt_subsampled_shape((1, 1, 1), subsampling)
    first_subsampling = SIMNIBS46_FNIRT_LEVELS[0].subsampling
    initial = tuple(
        int(np.floor(bias_resolution_mm / (size * first_subsampling) + 0.5))
        for size in voxel_sizes
    )
    if any(value < 1 for value in initial):
        raise ValueError("bias resolution is incompatible with the coarsest level")
    return tuple((first_subsampling // subsampling) * value for value in initial)


def _spline_matrix_with_coefficient_size(
    field_size: int, knot_spacing: int, coefficient_size: int
) -> np.ndarray:
    return np.asarray(
        [
            [
                cubic_spline_value(voxel, coefficient, knot_spacing)
                for coefficient in range(coefficient_size)
            ]
            for voxel in range(field_size)
        ],
        dtype=np.float64,
    )


def _spline_mapping_matrix(
    field_size: int,
    target_knot_spacing: int,
    target_coefficient_size: int,
    source_knot_spacing: int,
    source_coefficient_size: int,
) -> np.ndarray:
    target = _spline_matrix_with_coefficient_size(
        field_size, target_knot_spacing, target_coefficient_size
    )
    source = _spline_matrix_with_coefficient_size(
        field_size, source_knot_spacing, source_coefficient_size
    )
    return np.linalg.pinv(target.T @ target) @ (target.T @ source)


@njit(cache=True)
def _apply_spline_coefficient_maps(
    coefficients: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    map_z: np.ndarray,
) -> np.ndarray:
    """Apply x, y, then z maps in ``splinefield::ZoomField`` loop order."""

    temporary_x = np.empty(
        (map_x.shape[0], coefficients.shape[1], coefficients.shape[2]),
        dtype=np.float64,
    )
    for z in range(coefficients.shape[2]):
        for y in range(coefficients.shape[1]):
            for output_x in range(map_x.shape[0]):
                value = 0.0
                for input_x in range(coefficients.shape[0]):
                    value += map_x[output_x, input_x] * coefficients[input_x, y, z]
                temporary_x[output_x, y, z] = value
    temporary_y = np.empty(
        (map_x.shape[0], map_y.shape[0], coefficients.shape[2]),
        dtype=np.float64,
    )
    for z in range(coefficients.shape[2]):
        for x in range(map_x.shape[0]):
            for output_y in range(map_y.shape[0]):
                value = 0.0
                for input_y in range(coefficients.shape[1]):
                    value += map_y[output_y, input_y] * temporary_x[x, input_y, z]
                temporary_y[x, output_y, z] = value
    output = np.empty(
        (map_x.shape[0], map_y.shape[0], map_z.shape[0]), dtype=np.float64
    )
    for y in range(map_y.shape[0]):
        for x in range(map_x.shape[0]):
            for output_z in range(map_z.shape[0]):
                value = 0.0
                for input_z in range(coefficients.shape[2]):
                    value += map_z[output_z, input_z] * temporary_y[x, y, input_z]
                output[x, y, output_z] = value
    return output


def zoom_fnirt_spline_coefficients(
    coefficients: np.ndarray,
    old_shape: tuple[int, int, int],
    old_voxel_sizes_mm: tuple[float, float, float],
    old_knot_spacing: tuple[int, int, int],
    new_shape: tuple[int, int, int],
    new_voxel_sizes_mm: tuple[float, float, float],
    new_knot_spacing: tuple[int, int, int],
) -> np.ndarray:
    """Reproduce FNIRT's voxel-size zoom followed by knot-spacing zoom."""

    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != fsl_coefficient_shape(old_shape, old_knot_spacing):
        raise ValueError("coefficient shape does not match the old spline field")
    old_voxels = np.asarray(old_voxel_sizes_mm, dtype=np.float64)
    new_voxels = np.asarray(new_voxel_sizes_mm, dtype=np.float64)
    if (
        old_voxels.shape != (3,)
        or new_voxels.shape != (3,)
        or np.any(old_voxels <= 0.0)
        or np.any(new_voxels <= 0.0)
        or not np.all(np.isfinite(old_voxels))
        or not np.all(np.isfinite(new_voxels))
    ):
        raise ValueError("old and new voxel sizes must be positive and finite")
    intermediate_spacing = tuple(int(value) for value in old_knot_spacing)
    intermediate_shape = fsl_coefficient_shape(new_shape, intermediate_spacing)
    fake_source_spacing = []
    for axis in range(3):
        exact = old_voxels[axis] / new_voxels[axis] * float(intermediate_spacing[axis])
        rounded = int(np.floor(exact + 0.5))
        if rounded < 1 or abs(exact - rounded) > 1e-6:
            raise ValueError("voxel-size change cannot be represented by integer knots")
        fake_source_spacing.append(rounded)
    first_maps = tuple(
        _spline_mapping_matrix(
            new_shape[axis],
            intermediate_spacing[axis],
            intermediate_shape[axis],
            fake_source_spacing[axis],
            values.shape[axis],
        )
        for axis in range(3)
    )
    intermediate = _apply_spline_coefficient_maps(values, *first_maps)
    target_shape = fsl_coefficient_shape(new_shape, new_knot_spacing)
    second_maps = tuple(
        _spline_mapping_matrix(
            new_shape[axis],
            new_knot_spacing[axis],
            target_shape[axis],
            intermediate_spacing[axis],
            intermediate.shape[axis],
        )
        for axis in range(3)
    )
    return _apply_spline_coefficient_maps(intermediate, *second_maps)


def initialize_fnirt_intensity_mapping(
    reference_shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    level: FnirtLevel,
) -> FnirtIntensityMapping:
    """Initialize the order-five global mapping and zoomed constant bias field."""

    full_bias_spacing = tuple(
        value * SIMNIBS46_FNIRT_LEVELS[-1].subsampling
        for value in fsl_fnirt_bias_knot_spacing(
            voxel_sizes_mm, SIMNIBS46_FNIRT_LEVELS[-1].subsampling
        )
    )
    full_coefficients = fit_spline_coefficients(
        np.ones(reference_shape, dtype=np.float32), full_bias_spacing
    )
    level_shape = fsl_fnirt_subsampled_shape(reference_shape, level.subsampling)
    level_voxel_sizes = tuple(
        float(value) * level.subsampling for value in voxel_sizes_mm
    )
    level_spacing = fsl_fnirt_bias_knot_spacing(voxel_sizes_mm, level.subsampling)
    level_coefficients = zoom_fnirt_spline_coefficients(
        full_coefficients,
        reference_shape,
        voxel_sizes_mm,
        full_bias_spacing,
        level_shape,
        level_voxel_sizes,
        level_spacing,
    )
    bias = expand_spline_coefficients(
        level_coefficients, level_shape, level_spacing
    ).astype(np.float32)
    global_coefficients = np.zeros(5, dtype=np.float64)
    global_coefficients[1] = 1.0
    return FnirtIntensityMapping(
        global_coefficients=global_coefficients,
        bias_coefficients=level_coefficients,
        bias_field=bias,
        bias_knot_spacing=level_spacing,
    )


@njit(cache=True)
def _apply_fnirt_intensity_mapping_fsl_order(
    reference: np.ndarray,
    global_coefficients: np.ndarray,
    bias_field: np.ndarray,
) -> np.ndarray:
    nx, ny, nz = reference.shape
    output = np.empty_like(reference)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                mapped = np.float32(global_coefficients[0])
                product = 1.0
                for order in range(1, global_coefficients.size):
                    product *= reference[x, y, z]
                    mapped = np.float32(mapped + global_coefficients[order] * product)
                output[x, y, z] = np.float32(mapped * bias_field[x, y, z])
    return output


def apply_fnirt_intensity_mapping(
    reference: np.ndarray, mapping: FnirtIntensityMapping
) -> np.ndarray:
    """Scale a level reference using FSL's polynomial and local bias order."""

    values = np.asarray(reference, dtype=np.float32)
    if values.shape != mapping.bias_field.shape or not np.all(np.isfinite(values)):
        raise ValueError("reference must be finite and match the bias field")
    return _apply_fnirt_intensity_mapping_fsl_order(
        values,
        np.asarray(mapping.global_coefficients, dtype=np.float64),
        np.asarray(mapping.bias_field, dtype=np.float32),
    )


@njit(cache=True)
def _masked_ssd_fsl_order(
    moving: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> tuple[float, int]:
    nx, ny, nz = moving.shape
    total = 0.0
    count = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if mask[x, y, z]:
                    difference = np.float32(moving[x, y, z] - reference[x, y, z])
                    squared = np.float32(difference * difference)
                    total += squared
                    count += 1
    return total, count


def evaluate_fnirt_cost(
    level_images: FnirtLevelImages,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
    affine_matrix: np.ndarray,
    intensity_mapping: FnirtIntensityMapping,
    level: FnirtLevel,
    *,
    nonlinear_displacement: np.ndarray | None = None,
    displacement_coefficients: np.ndarray | None = None,
    displacement_knot_spacing: tuple[int, int, int] = (2, 2, 2),
    bias_regularization_weight: float = 10_000.0,
) -> FnirtCostEvaluation:
    """Evaluate the fixed FNIRT SSD, bending, and intensity terms."""

    warped = warp_fnirt_moving(
        level_images,
        reference_affine,
        moving_affine,
        affine_matrix,
        nonlinear_displacement,
        calculate_derivatives=False,
    )
    scaled_reference = apply_fnirt_intensity_mapping(
        level_images.reference, intensity_mapping
    )
    sum_squared, voxel_count = _masked_ssd_fsl_order(
        warped.values, scaled_reference, warped.mask
    )
    if voxel_count == 0:
        raise ValueError("FNIRT has no valid voxels at this level")
    mean_squared = float(sum_squared / voxel_count)

    displacement_regularization = 0.0
    if displacement_coefficients is not None:
        coefficients = np.asarray(displacement_coefficients, dtype=np.float64)
        expected = fsl_coefficient_shape(
            level_images.reference.shape, displacement_knot_spacing
        ) + (3,)
        if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
            raise ValueError(
                f"displacement_coefficients must be finite with shape {expected}"
            )
        energy = sum(
            bending_energy(
                coefficients[..., component],
                level_images.reference.shape,
                level_images.reference_voxel_sizes_mm,
                displacement_knot_spacing,
            )
            for component in range(3)
        )
        displacement_regularization = (
            mean_squared * level.regularization_weight * energy / voxel_count
        )
    if not np.isfinite(bias_regularization_weight) or bias_regularization_weight < 0:
        raise ValueError("bias_regularization_weight must be finite and nonnegative")
    intensity_regularization = 0.0
    if level.estimate_intensity:
        intensity_regularization = (
            bias_regularization_weight
            * bending_energy(
                intensity_mapping.bias_coefficients,
                level_images.reference.shape,
                level_images.reference_voxel_sizes_mm,
                intensity_mapping.bias_knot_spacing,
            )
            / voxel_count
        )
    total = mean_squared + displacement_regularization + intensity_regularization
    return FnirtCostEvaluation(
        total=total,
        mean_squared_difference=mean_squared,
        displacement_regularization=displacement_regularization,
        intensity_regularization=intensity_regularization,
        voxel_count=voxel_count,
        warped_moving=warped,
        scaled_reference=scaled_reference,
    )


@njit(cache=True)
def _fnirt_intensity_gradient_images_fsl_order(
    reference: np.ndarray,
    difference: np.ndarray,
    mask: np.ndarray,
    bias_field: np.ndarray,
    global_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build FSL's weighted reference and global intensity gradient."""

    nx, ny, nz = reference.shape
    weighted_reference = np.empty_like(reference)
    global_gradient = np.zeros(global_coefficients.size, dtype=np.float64)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                ref = reference[x, y, z]
                ref_power = np.float32(ref)
                weighted = np.float32(global_coefficients[0])
                for order in range(1, global_coefficients.size):
                    weighted = np.float32(
                        weighted + np.float32(ref_power * global_coefficients[order])
                    )
                    if order < global_coefficients.size - 1:
                        ref_power = np.float32(ref_power * ref)
                weighted_reference[x, y, z] = weighted
                if mask[x, y, z]:
                    product = 1.0
                    for order in range(global_coefficients.size):
                        global_gradient[order] -= (
                            product
                            * float(bias_field[x, y, z])
                            * float(difference[x, y, z])
                        )
                        product *= float(ref)
    return weighted_reference, global_gradient


def evaluate_fnirt_gradient(
    level_images: FnirtLevelImages,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
    affine_matrix: np.ndarray,
    intensity_mapping: FnirtIntensityMapping,
    level: FnirtLevel,
    *,
    nonlinear_displacement: np.ndarray | None = None,
    displacement_coefficients: np.ndarray | None = None,
    displacement_knot_spacing: tuple[int, int, int] = (2, 2, 2),
    bias_regularization_weight: float = 10_000.0,
    _cost_evaluation: FnirtCostEvaluation | None = None,
) -> FnirtGradientEvaluation:
    """Evaluate FSL's analytic mean-SSD and regularization gradient."""

    coefficient_shape = fsl_coefficient_shape(
        level_images.reference.shape, displacement_knot_spacing
    )
    coefficients = (
        np.zeros(coefficient_shape + (3,), dtype=np.float64)
        if displacement_coefficients is None
        else np.asarray(displacement_coefficients, dtype=np.float64)
    )
    if coefficients.shape != coefficient_shape + (3,) or not np.all(
        np.isfinite(coefficients)
    ):
        raise ValueError(
            "displacement_coefficients must match the level displacement field"
        )
    cost = _cost_evaluation
    if cost is None:
        cost = evaluate_fnirt_cost(
            level_images,
            reference_affine,
            moving_affine,
            affine_matrix,
            intensity_mapping,
            level,
            nonlinear_displacement=nonlinear_displacement,
            displacement_coefficients=coefficients,
            displacement_knot_spacing=displacement_knot_spacing,
            bias_regularization_weight=bias_regularization_weight,
        )
    derivative_warp = warp_fnirt_moving(
        level_images,
        reference_affine,
        moving_affine,
        affine_matrix,
        nonlinear_displacement,
        calculate_derivatives=True,
    )
    cost = FnirtCostEvaluation(
        total=cost.total,
        mean_squared_difference=cost.mean_squared_difference,
        displacement_regularization=cost.displacement_regularization,
        intensity_regularization=cost.intensity_regularization,
        voxel_count=cost.voxel_count,
        warped_moving=derivative_warp,
        scaled_reference=cost.scaled_reference,
    )
    difference = np.subtract(
        cost.warped_moving.values, cost.scaled_reference, dtype=np.float32
    )
    mask = cost.warped_moving.mask
    scale = 2.0 / cost.voxel_count
    displacement_basis = tuple(
        spline_design_matrix(size, spacing)
        for size, spacing in zip(
            level_images.reference.shape, displacement_knot_spacing, strict=True
        )
    )
    displacement_hessian = bending_energy_hessian(
        level_images.reference.shape,
        level_images.reference_voxel_sizes_mm,
        displacement_knot_spacing,
    )
    displacement_parts = []
    for component in range(3):
        data_image = np.multiply(
            difference,
            cost.warped_moving.derivatives_per_mm[..., component],
            dtype=np.float32,
        )
        data_gradient = _spline_jte_fsl_order(
            data_image, mask, *displacement_basis
        ).reshape(-1, order="F")
        regularization_gradient = displacement_hessian @ coefficients[
            ..., component
        ].reshape(-1, order="F")
        displacement_parts.append(
            scale * data_gradient
            + (
                cost.mean_squared_difference
                * level.regularization_weight
                / cost.voxel_count
            )
            * regularization_gradient
        )
    displacement_gradient = np.concatenate(displacement_parts)

    if not level.estimate_intensity:
        empty = np.empty(0, dtype=np.float64)
        return FnirtGradientEvaluation(
            gradient=displacement_gradient,
            displacement_gradient=displacement_gradient,
            bias_gradient=empty,
            global_gradient=empty,
            cost=cost,
        )

    weighted_reference, global_gradient = _fnirt_intensity_gradient_images_fsl_order(
        np.asarray(level_images.reference, dtype=np.float32),
        difference,
        mask,
        np.asarray(intensity_mapping.bias_field, dtype=np.float32),
        np.asarray(intensity_mapping.global_coefficients, dtype=np.float64),
    )
    bias_basis = tuple(
        spline_design_matrix(size, spacing)
        for size, spacing in zip(
            level_images.reference.shape,
            intensity_mapping.bias_knot_spacing,
            strict=True,
        )
    )
    bias_gradient = -_spline_jte_fsl_order(
        np.multiply(weighted_reference, difference, dtype=np.float32),
        mask,
        *bias_basis,
    ).reshape(-1, order="F")
    bias_hessian = bending_energy_hessian(
        level_images.reference.shape,
        level_images.reference_voxel_sizes_mm,
        intensity_mapping.bias_knot_spacing,
    )
    bias_gradient += (
        0.5
        * bias_regularization_weight
        * (bias_hessian @ intensity_mapping.bias_coefficients.reshape(-1, order="F"))
    )
    bias_gradient *= scale
    global_gradient *= scale
    gradient = np.concatenate((displacement_gradient, bias_gradient, global_gradient))
    return FnirtGradientEvaluation(
        gradient=gradient,
        displacement_gradient=displacement_gradient,
        bias_gradient=bias_gradient,
        global_gradient=global_gradient,
        cost=cost,
    )


@njit(cache=True, parallel=True)
def _fill_spline_basis_sparse_fsl_order(
    field_shape: tuple[int, int, int],
    basis_x_data: np.ndarray,
    basis_x_indices: np.ndarray,
    basis_x_indptr: np.ndarray,
    basis_y_data: np.ndarray,
    basis_y_indices: np.ndarray,
    basis_y_indptr: np.ndarray,
    basis_z_data: np.ndarray,
    basis_z_indices: np.ndarray,
    basis_z_indptr: np.ndarray,
    output_data: np.ndarray,
    output_indices: np.ndarray,
    output_indptr: np.ndarray,
    coefficient_shape: tuple[int, int, int],
    worker_count: int,
) -> None:
    """Fill a 3D spline CSR directly in SciPy's original Kronecker order."""

    nx, ny, nz = field_shape
    coefficient_x, coefficient_y, _ = coefficient_shape
    for lane in prange(worker_count):
        for voxel in range(lane, nx * ny * nz, worker_count):
            voxel_x = voxel % nx
            voxel_y = (voxel // nx) % ny
            voxel_z = voxel // (nx * ny)
            offset = output_indptr[voxel]
            for position_z in range(
                basis_z_indptr[voxel_z], basis_z_indptr[voxel_z + 1]
            ):
                coefficient_z = basis_z_indices[position_z]
                value_z = basis_z_data[position_z]
                for position_y in range(
                    basis_y_indptr[voxel_y], basis_y_indptr[voxel_y + 1]
                ):
                    coefficient_y_index = basis_y_indices[position_y]
                    coefficient_offset = (
                        coefficient_z * coefficient_y + coefficient_y_index
                    ) * coefficient_x
                    value_y = basis_y_data[position_y]
                    for position_x in range(
                        basis_x_indptr[voxel_x], basis_x_indptr[voxel_x + 1]
                    ):
                        output_indices[offset] = (
                            coefficient_offset + basis_x_indices[position_x]
                        )
                        output_data[offset] = value_z * (
                            value_y * basis_x_data[position_x]
                        )
                        offset += 1


@njit(cache=True, parallel=True)
def _fill_selected_spline_basis_sparse_fsl_order(
    voxel_indices: np.ndarray,
    field_shape: tuple[int, int, int],
    basis_x_data: np.ndarray,
    basis_x_indices: np.ndarray,
    basis_x_indptr: np.ndarray,
    basis_y_data: np.ndarray,
    basis_y_indices: np.ndarray,
    basis_y_indptr: np.ndarray,
    basis_z_data: np.ndarray,
    basis_z_indices: np.ndarray,
    basis_z_indptr: np.ndarray,
    output_data: np.ndarray,
    output_indices: np.ndarray,
    output_indptr: np.ndarray,
    coefficient_shape: tuple[int, int, int],
    worker_count: int,
) -> None:
    """Fill CSR only for active voxels in full original Kronecker order."""

    nx, ny, _ = field_shape
    coefficient_x, coefficient_y, _ = coefficient_shape
    for lane in prange(worker_count):
        for output_row in range(lane, voxel_indices.size, worker_count):
            voxel = voxel_indices[output_row]
            voxel_x = voxel % nx
            voxel_y = (voxel // nx) % ny
            voxel_z = voxel // (nx * ny)
            offset = output_indptr[output_row]
            for position_z in range(
                basis_z_indptr[voxel_z], basis_z_indptr[voxel_z + 1]
            ):
                coefficient_z = basis_z_indices[position_z]
                value_z = basis_z_data[position_z]
                for position_y in range(
                    basis_y_indptr[voxel_y], basis_y_indptr[voxel_y + 1]
                ):
                    coefficient_y_index = basis_y_indices[position_y]
                    coefficient_offset = (
                        coefficient_z * coefficient_y + coefficient_y_index
                    ) * coefficient_x
                    value_y = basis_y_data[position_y]
                    for position_x in range(
                        basis_x_indptr[voxel_x], basis_x_indptr[voxel_x + 1]
                    ):
                        output_indices[offset] = (
                            coefficient_offset + basis_x_indices[position_x]
                        )
                        output_data[offset] = value_z * (
                            value_y * basis_x_data[position_x]
                        )
                        offset += 1


def _spline_basis_axes(
    field_shape: tuple[int, int, int], knot_spacing: tuple[int, int, int]
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Build cached one-dimensional cubic-spline axis matrices."""

    return tuple(
        csr_matrix(spline_design_matrix(size, spacing))
        for size, spacing in zip(field_shape, knot_spacing, strict=True)
    )


def _spline_basis_sparse(
    field_shape: tuple[int, int, int],
    knot_spacing: tuple[int, int, int],
    workers: int = 1,
) -> csr_matrix:
    """Build CSR directly, bitwise identical to the x-fastest Kronecker basis."""

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    axes = _spline_basis_axes(field_shape, knot_spacing)
    counts = (
        np.diff(axes[0].indptr)[:, None, None]
        * np.diff(axes[1].indptr)[None, :, None]
        * np.diff(axes[2].indptr)[None, None, :]
    ).reshape(-1, order="F")
    output_indptr = np.empty(int(np.prod(field_shape)) + 1, dtype=np.int64)
    output_indptr[0] = 0
    np.cumsum(counts, out=output_indptr[1:])
    output_data = np.empty(int(output_indptr[-1]), dtype=np.float64)
    output_indices = np.empty(int(output_indptr[-1]), dtype=np.int32)
    previous_workers = get_num_threads()
    try:
        active_workers = min(workers, previous_workers)
        set_available_numba_threads(active_workers)
        _fill_spline_basis_sparse_fsl_order(
            field_shape,
            axes[0].data,
            axes[0].indices,
            axes[0].indptr,
            axes[1].data,
            axes[1].indices,
            axes[1].indptr,
            axes[2].data,
            axes[2].indices,
            axes[2].indptr,
            output_data,
            output_indices,
            output_indptr,
            tuple(axis.shape[1] for axis in axes),
            active_workers,
        )
    finally:
        set_num_threads(previous_workers)
    return csr_matrix(
        (output_data, output_indices, output_indptr),
        shape=(
            int(np.prod(field_shape)),
            int(np.prod(tuple(axis.shape[1] for axis in axes))),
        ),
    )


def _selected_spline_basis_sparse(
    field_shape: tuple[int, int, int],
    knot_spacing: tuple[int, int, int],
    selected: np.ndarray,
    workers: int,
) -> csr_matrix:
    """Build spline CSR directly for masked voxels without creating the full 3D basis."""

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    selection = np.asarray(selected, dtype=bool)
    if selection.shape != (int(np.prod(field_shape)),):
        raise ValueError("selected mask must match the flattened field")
    axes = _spline_basis_axes(field_shape, knot_spacing)
    voxel_indices = np.flatnonzero(selection)
    nx, ny, _ = field_shape
    voxel_x = voxel_indices % nx
    voxel_y = (voxel_indices // nx) % ny
    voxel_z = voxel_indices // (nx * ny)
    counts = (
        np.diff(axes[0].indptr)[voxel_x]
        * np.diff(axes[1].indptr)[voxel_y]
        * np.diff(axes[2].indptr)[voxel_z]
    )
    output_indptr = np.empty(voxel_indices.size + 1, dtype=np.int64)
    output_indptr[0] = 0
    np.cumsum(counts, out=output_indptr[1:])
    output_data = np.empty(int(output_indptr[-1]), dtype=np.float64)
    output_indices = np.empty(int(output_indptr[-1]), dtype=np.int32)
    previous_workers = get_num_threads()
    try:
        active_workers = min(workers, previous_workers)
        set_available_numba_threads(active_workers)
        _fill_selected_spline_basis_sparse_fsl_order(
            voxel_indices,
            field_shape,
            axes[0].data,
            axes[0].indices,
            axes[0].indptr,
            axes[1].data,
            axes[1].indices,
            axes[1].indptr,
            axes[2].data,
            axes[2].indices,
            axes[2].indptr,
            output_data,
            output_indices,
            output_indptr,
            tuple(axis.shape[1] for axis in axes),
            active_workers,
        )
    finally:
        set_num_threads(previous_workers)
    return csr_matrix(
        (output_data, output_indices, output_indptr),
        shape=(
            voxel_indices.size,
            int(np.prod(tuple(axis.shape[1] for axis in axes))),
        ),
    )


def _masked_sparse_jtj(
    left_basis: csr_matrix,
    right_basis: csr_matrix,
    voxel_weights: np.ndarray,
    mask: np.ndarray,
) -> csc_matrix:
    """Calculate a masked sparse ``J_left.T @ W @ J_right`` product."""

    selected = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    weights = np.asarray(voxel_weights, dtype=np.float64).reshape(-1, order="F")[
        selected
    ]
    left = left_basis[selected]
    right = right_basis[selected]
    return (left.T @ right.multiply(weights[:, None])).tocsc()


def _selected_sparse_jtj(
    left_basis: csr_matrix,
    right_basis: csr_matrix,
    voxel_weights: np.ndarray,
) -> csc_matrix:
    """Compute sparse ``J_left.T W J_right`` for spline bases with the same mask."""

    weights = np.asarray(voxel_weights, dtype=np.float64)
    if weights.shape != (left_basis.shape[0],):
        raise ValueError("selected voxel weights must match the basis rows")
    return (left_basis.T @ right_basis.multiply(weights[:, None])).tocsc()


@njit(cache=True)
def _sparse_cross_column_costs(
    left_indptr: np.ndarray,
    right_indices: np.ndarray,
    right_indptr: np.ndarray,
    column_count: int,
) -> np.ndarray:
    """Estimate multiplication work per right column from left-matrix nonzeros in shared rows."""

    costs = np.zeros(column_count, dtype=np.int64)
    for row in range(right_indptr.size - 1):
        row_cost = left_indptr[row + 1] - left_indptr[row]
        for position in range(right_indptr[row], right_indptr[row + 1]):
            costs[right_indices[position]] += row_cost
    return costs


def _selected_sparse_jtj_blocks(
    left_basis: csr_matrix,
    right_basis: csr_matrix,
    voxel_weights: tuple[np.ndarray, ...],
    workers: int,
) -> tuple[csc_matrix, ...]:
    """Assemble shared-spline-basis ``J.T W J`` blocks in parallel by disjoint coefficient columns."""

    if not voxel_weights:
        return ()
    if workers == 1 or right_basis.shape[1] < 2:
        return tuple(
            _selected_sparse_jtj(left_basis, right_basis, weights)
            for weights in voxel_weights
        )
    column_count = right_basis.shape[1]
    worker_count = min(workers, column_count)
    column_costs = _sparse_cross_column_costs(
        left_basis.indptr,
        right_basis.indices,
        right_basis.indptr,
        column_count,
    )
    cumulative_costs = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(column_costs))
    )
    boundaries = [0]
    for lane in range(1, worker_count):
        target = cumulative_costs[-1] * lane / worker_count
        candidate = int(np.searchsorted(cumulative_costs, target))
        minimum = boundaries[-1] + 1
        maximum = column_count - (worker_count - lane)
        boundaries.append(max(minimum, min(candidate, maximum)))
    boundaries.append(column_count)
    ranges = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))

    def build_chunk(bounds: tuple[int, int]) -> tuple[csc_matrix, ...]:
        """Compute a span of independent columns using SciPy's original sum order per column."""

        start, stop = bounds
        right = right_basis[:, start:stop]
        width = stop - start
        weighted = hstack(
            tuple(right.multiply(weights[:, None]) for weights in voxel_weights),
            format="csr",
        )
        product = (left_basis.T @ weighted).tocsc()
        return tuple(
            product[:, index * width : (index + 1) * width]
            for index in range(len(voxel_weights))
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunks = tuple(executor.map(build_chunk, ranges))
    return tuple(
        hstack(tuple(chunk[index] for chunk in chunks), format="csc")
        for index in range(len(voxel_weights))
    )


@njit(cache=True)
def _build_symmetric_spline_lower_pattern(
    coefficient_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate FSL's same-basis lower-triangular ``JtJ`` structure in a compiled kernel."""

    size_x, size_y, size_z = coefficient_shape
    coefficient_count = size_x * size_y * size_z
    indptr = np.empty(coefficient_count + 1, dtype=np.int64)
    counts = np.empty(coefficient_count, dtype=np.int64)
    column = 0
    for coefficient_z in range(size_z):
        for coefficient_y in range(size_y):
            for coefficient_x in range(size_x):
                count = 0
                for row_z in range(
                    max(0, coefficient_z - 3), min(size_z, coefficient_z + 4)
                ):
                    for row_y in range(
                        max(0, coefficient_y - 3), min(size_y, coefficient_y + 4)
                    ):
                        for row_x in range(
                            max(0, coefficient_x - 3),
                            min(size_x, coefficient_x + 4),
                        ):
                            row = row_z * size_y * size_x + row_y * size_x + row_x
                            if row >= column:
                                count += 1
                counts[column] = count
                column += 1
    indptr[0] = 0
    for index in range(coefficient_count):
        indptr[index + 1] = indptr[index] + counts[index]
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    column = 0
    for coefficient_z in range(size_z):
        for coefficient_y in range(size_y):
            for coefficient_x in range(size_x):
                offset = int(indptr[column])
                for row_z in range(
                    max(0, coefficient_z - 3), min(size_z, coefficient_z + 4)
                ):
                    for row_y in range(
                        max(0, coefficient_y - 3), min(size_y, coefficient_y + 4)
                    ):
                        for row_x in range(
                            max(0, coefficient_x - 3),
                            min(size_x, coefficient_x + 4),
                        ):
                            row = row_z * size_y * size_x + row_y * size_x + row_x
                            if row >= column:
                                indices[offset] = row
                                offset += 1
                column += 1
    return indptr, indices


@lru_cache(maxsize=32)
def _symmetric_spline_lower_pattern(
    coefficient_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Cache the fixed CSC structure of FSL's cubic-spline same-basis lower ``JtJ`` triangle."""

    return _build_symmetric_spline_lower_pattern(coefficient_shape)


@njit(cache=True, parallel=True)
def _symmetric_spline_jtj_values_fsl_order(
    basis_data: np.ndarray,
    basis_indices: np.ndarray,
    basis_indptr: np.ndarray,
    weights: np.ndarray,
    output_indices: np.ndarray,
    output_indptr: np.ndarray,
    worker_count: int,
) -> np.ndarray:
    """Accumulate the lower triangle in FSL column-spline premultiplication and x-fastest intersection order."""

    values = np.empty((output_indices.size, weights.shape[0]), dtype=np.float64)
    if weights.shape[0] == 6:
        for column in prange(output_indptr.size - 1):
            column_start = basis_indptr[column]
            column_stop = basis_indptr[column + 1]
            for output_index in range(output_indptr[column], output_indptr[column + 1]):
                row = output_indices[output_index]
                row_position = basis_indptr[row]
                row_stop = basis_indptr[row + 1]
                column_position = column_start
                value0 = 0.0
                value1 = 0.0
                value2 = 0.0
                value3 = 0.0
                value4 = 0.0
                value5 = 0.0
                while column_position < column_stop and row_position < row_stop:
                    column_voxel = basis_indices[column_position]
                    row_voxel = basis_indices[row_position]
                    if column_voxel < row_voxel:
                        column_position += 1
                    elif row_voxel < column_voxel:
                        row_position += 1
                    else:
                        column_basis = basis_data[column_position]
                        row_basis = basis_data[row_position]
                        value0 += (column_basis * weights[0, column_voxel]) * row_basis
                        value1 += (column_basis * weights[1, column_voxel]) * row_basis
                        value2 += (column_basis * weights[2, column_voxel]) * row_basis
                        value3 += (column_basis * weights[3, column_voxel]) * row_basis
                        value4 += (column_basis * weights[4, column_voxel]) * row_basis
                        value5 += (column_basis * weights[5, column_voxel]) * row_basis
                        column_position += 1
                        row_position += 1
                values[output_index, 0] = value0
                values[output_index, 1] = value1
                values[output_index, 2] = value2
                values[output_index, 3] = value3
                values[output_index, 4] = value4
                values[output_index, 5] = value5
        return values.T
    if weights.shape[0] == 1:
        for column in prange(output_indptr.size - 1):
            column_start = basis_indptr[column]
            column_stop = basis_indptr[column + 1]
            for output_index in range(output_indptr[column], output_indptr[column + 1]):
                row = output_indices[output_index]
                row_position = basis_indptr[row]
                row_stop = basis_indptr[row + 1]
                column_position = column_start
                value = 0.0
                while column_position < column_stop and row_position < row_stop:
                    column_voxel = basis_indices[column_position]
                    row_voxel = basis_indices[row_position]
                    if column_voxel < row_voxel:
                        column_position += 1
                    elif row_voxel < column_voxel:
                        row_position += 1
                    else:
                        value += (
                            basis_data[column_position] * weights[0, column_voxel]
                        ) * basis_data[row_position]
                        column_position += 1
                        row_position += 1
                values[output_index, 0] = value
        return values.T
    for column in prange(output_indptr.size - 1):
        column_start = basis_indptr[column]
        column_stop = basis_indptr[column + 1]
        for output_index in range(output_indptr[column], output_indptr[column + 1]):
            row = output_indices[output_index]
            row_position = basis_indptr[row]
            row_stop = basis_indptr[row + 1]
            column_position = column_start
            for block in range(weights.shape[0]):
                values[output_index, block] = 0.0
            while column_position < column_stop and row_position < row_stop:
                column_voxel = basis_indices[column_position]
                row_voxel = basis_indices[row_position]
                if column_voxel < row_voxel:
                    column_position += 1
                elif row_voxel < column_voxel:
                    row_position += 1
                else:
                    column_basis = basis_data[column_position]
                    row_basis = basis_data[row_position]
                    for block in range(weights.shape[0]):
                        values[output_index, block] += (
                            column_basis * weights[block, column_voxel]
                        ) * row_basis
                    column_position += 1
                    row_position += 1
    return values.T


def _symmetric_spline_jtj_blocks(
    basis: csr_matrix,
    voxel_weights: tuple[np.ndarray, ...],
    coefficient_shape: tuple[int, int, int],
    workers: int,
) -> tuple[csc_matrix, ...]:
    """Assemble multiple same-basis ``JtJ`` blocks using FSL local support and symmetry contracts."""

    matrix = basis.tocsc()
    pattern_indptr, pattern_indices = _symmetric_spline_lower_pattern(coefficient_shape)
    weight_matrix = np.vstack(
        tuple(np.asarray(values, dtype=np.float32) for values in voxel_weights)
    )
    previous_workers = get_num_threads()
    try:
        active_workers = min(workers, previous_workers)
        set_available_numba_threads(active_workers)
        block_values = _symmetric_spline_jtj_values_fsl_order(
            np.asarray(matrix.data, dtype=np.float64),
            np.asarray(matrix.indices),
            np.asarray(matrix.indptr),
            weight_matrix,
            pattern_indices,
            pattern_indptr,
            active_workers,
        )
    finally:
        set_num_threads(previous_workers)
    coefficient_count = matrix.shape[1]
    blocks = []
    for values in block_values:
        lower = csc_matrix(
            (values.copy(), pattern_indices.copy(), pattern_indptr.copy()),
            shape=(coefficient_count, coefficient_count),
        )
        lower.eliminate_zeros()
        blocks.append((lower + tril(lower, k=-1).T).tocsc())
    return tuple(blocks)


def evaluate_fnirt_hessian(
    level_images: FnirtLevelImages,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
    affine_matrix: np.ndarray,
    intensity_mapping: FnirtIntensityMapping,
    level: FnirtLevel,
    *,
    nonlinear_displacement: np.ndarray | None = None,
    displacement_coefficients: np.ndarray | None = None,
    displacement_knot_spacing: tuple[int, int, int] = (2, 2, 2),
    bias_regularization_weight: float = 10_000.0,
    workers: int = 1,
    _gradient_evaluation: FnirtGradientEvaluation | None = None,
    _displacement_basis: csr_matrix | None = None,
    _displacement_regularization: csc_matrix | None = None,
    _bias_basis: csr_matrix | None = None,
    _bias_regularization: csc_matrix | None = None,
) -> FnirtHessianEvaluation:
    """Evaluate FSL's sparse Gauss-Newton Hessian block construction."""

    gradient = _gradient_evaluation
    if gradient is None:
        gradient = evaluate_fnirt_gradient(
            level_images,
            reference_affine,
            moving_affine,
            affine_matrix,
            intensity_mapping,
            level,
            nonlinear_displacement=nonlinear_displacement,
            displacement_coefficients=displacement_coefficients,
            displacement_knot_spacing=displacement_knot_spacing,
            bias_regularization_weight=bias_regularization_weight,
        )
    cost = gradient.cost
    mask = cost.warped_moving.mask
    scale = 2.0 / cost.voxel_count
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    selected = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    selected_displacement_basis = (
        _selected_spline_basis_sparse(
            level_images.reference.shape,
            displacement_knot_spacing,
            selected,
            workers,
        )
        if _displacement_basis is None
        else _displacement_basis[selected]
    )
    partials = tuple(
        np.asarray(
            cost.warped_moving.derivatives_per_mm[..., component],
            dtype=np.float32,
        )
        for component in range(3)
    )
    regularization = (
        bending_energy_hessian(
            level_images.reference.shape,
            level_images.reference_voxel_sizes_mm,
            displacement_knot_spacing,
        )
        if _displacement_regularization is None
        else _displacement_regularization
    )
    regularization = (
        cost.mean_squared_difference * level.regularization_weight / cost.voxel_count
    ) * regularization
    displacement_blocks: list[list[csc_matrix | None]] = [
        [None, None, None],
        [None, None, None],
        [None, None, None],
    ]
    block_indices = tuple((row, column) for row in range(3) for column in range(row, 3))

    displacement_weights = tuple(
        np.multiply(partials[row], partials[column], dtype=np.float32).reshape(
            -1, order="F"
        )[selected]
        for row, column in block_indices
    )

    def assemble_displacement_hessian(
        blocks: tuple[csc_matrix, ...],
    ) -> csc_matrix:
        """Concatenate independently computed displacement Hessians in FSL 3x3 block order."""

        for (row, column), block in zip(block_indices, blocks, strict=True):
            block = scale * block
            if row == column:
                block = block + regularization
            block = block.tocsc()
            displacement_blocks[row][column] = block
            displacement_blocks[column][row] = block
        return bmat(displacement_blocks, format="csc")

    if not level.estimate_intensity:
        built_blocks = (
            _selected_sparse_jtj_blocks(
                selected_displacement_basis,
                selected_displacement_basis,
                displacement_weights,
                1,
            )
            if workers == 1
            else _symmetric_spline_jtj_blocks(
                selected_displacement_basis,
                displacement_weights,
                fsl_coefficient_shape(
                    level_images.reference.shape, displacement_knot_spacing
                ),
                workers,
            )
        )
        displacement_hessian = assemble_displacement_hessian(built_blocks)
        empty = csc_matrix((displacement_hessian.shape[0], 0))
        return FnirtHessianEvaluation(
            hessian=displacement_hessian,
            displacement_hessian=displacement_hessian,
            intensity_hessian=csc_matrix((0, 0)),
            cross_hessian=empty,
            gradient=gradient,
        )

    reference = np.asarray(level_images.reference, dtype=np.float32)
    weighted_reference, _ = _fnirt_intensity_gradient_images_fsl_order(
        reference,
        np.zeros_like(reference),
        mask,
        np.asarray(intensity_mapping.bias_field, dtype=np.float32),
        np.asarray(intensity_mapping.global_coefficients, dtype=np.float64),
    )
    selected_bias_basis = (
        _selected_spline_basis_sparse(
            level_images.reference.shape,
            intensity_mapping.bias_knot_spacing,
            selected,
            workers,
        )
        if _bias_basis is None
        else _bias_basis[selected]
    )
    bias_regularization = (
        bending_energy_hessian(
            level_images.reference.shape,
            level_images.reference_voxel_sizes_mm,
            intensity_mapping.bias_knot_spacing,
        )
        if _bias_regularization is None
        else _bias_regularization
    )
    local_weights = np.multiply(
        weighted_reference, weighted_reference, dtype=np.float32
    ).reshape(-1, order="F")[selected]
    bias = np.asarray(intensity_mapping.bias_field, dtype=np.float64)
    global_derivatives = np.empty(
        (reference.size, intensity_mapping.global_coefficients.size),
        dtype=np.float64,
    )
    reference_flat = reference.reshape(-1, order="F").astype(np.float64)
    bias_flat = bias.reshape(-1, order="F")
    power = np.ones(reference.size, dtype=np.float64)
    for order in range(global_derivatives.shape[1]):
        global_derivatives[:, order] = bias_flat * power
        power *= reference_flat
    global_valid = global_derivatives[selected]
    global_hessian = csc_matrix(global_valid.T @ global_valid)
    weighted_bias = np.multiply(
        weighted_reference,
        np.asarray(intensity_mapping.bias_field, dtype=np.float32),
        dtype=np.float32,
    )
    local_global_columns = []
    reference_power = np.ones_like(reference)
    for order in range(global_derivatives.shape[1]):
        weight = (
            weighted_bias
            if order == 0
            else np.multiply(weighted_bias, reference_power, dtype=np.float32)
        )
        local_global_columns.append(
            selected_bias_basis.T
            @ weight.reshape(-1, order="F")[selected].astype(np.float64)
        )
        if order < global_derivatives.shape[1] - 1:
            reference_power = np.multiply(reference_power, reference, dtype=np.float32)
    local_global = csc_matrix(np.column_stack(local_global_columns))
    cross_weights = tuple(
        np.multiply(partial, weighted_reference, dtype=np.float32).reshape(
            -1, order="F"
        )[selected]
        for partial in partials
    )
    built_blocks = (
        _selected_sparse_jtj_blocks(
            selected_displacement_basis,
            selected_displacement_basis,
            displacement_weights,
            1,
        )
        if workers == 1
        else _symmetric_spline_jtj_blocks(
            selected_displacement_basis,
            displacement_weights,
            fsl_coefficient_shape(
                level_images.reference.shape, displacement_knot_spacing
            ),
            workers,
        )
    )
    local_hessian = (
        _selected_sparse_jtj_blocks(
            selected_bias_basis,
            selected_bias_basis,
            (local_weights,),
            1,
        )[0]
        if workers == 1
        else _symmetric_spline_jtj_blocks(
            selected_bias_basis,
            (local_weights,),
            intensity_mapping.bias_coefficients.shape,
            workers,
        )[0]
    )
    local_cross_blocks = _selected_sparse_jtj_blocks(
        selected_displacement_basis,
        selected_bias_basis,
        cross_weights,
        workers,
    )
    displacement_hessian = assemble_displacement_hessian(built_blocks)
    local_hessian = (
        local_hessian + 0.5 * bias_regularization_weight * bias_regularization
    )
    intensity_hessian = scale * bmat(
        [
            [local_hessian, local_global],
            [local_global.T, global_hessian],
        ],
        format="csc",
    )
    cross_rows = []
    for partial, local_cross_block in zip(partials, local_cross_blocks, strict=True):
        local_cross = -local_cross_block
        global_cross_columns = []
        reference_power = np.ones_like(reference)
        bias_float = np.asarray(intensity_mapping.bias_field, dtype=np.float32)
        for order in range(global_derivatives.shape[1]):
            bias_power = (
                bias_float
                if order == 0
                else np.multiply(bias_float, reference_power, dtype=np.float32)
            )
            weight = -np.multiply(partial, bias_power, dtype=np.float32)
            global_cross_columns.append(
                selected_displacement_basis.T
                @ weight.reshape(-1, order="F")[selected].astype(np.float64)
            )
            if order < global_derivatives.shape[1] - 1:
                reference_power = np.multiply(
                    reference_power, reference, dtype=np.float32
                )
        global_cross = csc_matrix(np.column_stack(global_cross_columns))
        cross_rows.append([local_cross, global_cross])
    cross_hessian = scale * bmat(cross_rows, format="csc")
    hessian = bmat(
        [
            [displacement_hessian, cross_hessian],
            [cross_hessian.T, intensity_hessian],
        ],
        format="csc",
    )
    return FnirtHessianEvaluation(
        hessian=hessian,
        displacement_hessian=displacement_hessian,
        intensity_hessian=intensity_hessian,
        cross_hessian=cross_hessian,
        gradient=gradient,
    )


class _FnirtLevelObjective:
    """Adapt one fixed FNIRT level to FSL MISCMATHS optimizer callbacks."""

    def __init__(
        self,
        level_images: FnirtLevelImages,
        reference_affine: np.ndarray,
        moving_affine: np.ndarray,
        affine_matrix: np.ndarray,
        intensity_mapping: FnirtIntensityMapping,
        level: FnirtLevel,
        displacement_knot_spacing: tuple[int, int, int],
        bias_regularization_weight: float,
        workers: int = 1,
    ) -> None:
        self.level_images = level_images
        self.reference_affine = reference_affine
        self.moving_affine = moving_affine
        self.affine_matrix = affine_matrix
        self.template_mapping = intensity_mapping
        self.level = level
        self.displacement_knot_spacing = displacement_knot_spacing
        self.bias_regularization_weight = bias_regularization_weight
        self.workers = workers
        self.displacement_shape = fsl_coefficient_shape(
            level_images.reference.shape, displacement_knot_spacing
        )
        self.displacement_size = int(np.prod(self.displacement_shape))
        self.bias_size = intensity_mapping.bias_coefficients.size
        self._parameters: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None
        self._displacement: np.ndarray | None = None
        self._mapping: FnirtIntensityMapping | None = None
        self._cost_parameters: np.ndarray | None = None
        self._cost_evaluation: FnirtCostEvaluation | None = None
        self._gradient_parameters: np.ndarray | None = None
        self._gradient_evaluation: FnirtGradientEvaluation | None = None
        self._displacement_regularization: csc_matrix | None = None
        self._bias_regularization: csc_matrix | None = None

    def unpack(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, FnirtIntensityMapping]:
        """Decode FSL's x-fastest parameter blocks and cache dense fields."""

        values = np.asarray(parameters, dtype=np.float64)
        expected = 3 * self.displacement_size
        if self.level.estimate_intensity:
            expected += self.bias_size + self.template_mapping.global_coefficients.size
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"FNIRT parameters must be finite with shape ({expected},)"
            )
        if self._parameters is not None and np.array_equal(values, self._parameters):
            assert self._coefficients is not None
            assert self._displacement is not None
            assert self._mapping is not None
            return self._coefficients, self._displacement, self._mapping
        coefficients = np.empty(self.displacement_shape + (3,), dtype=np.float64)
        for component in range(3):
            start = component * self.displacement_size
            coefficients[..., component] = values[
                start : start + self.displacement_size
            ].reshape(self.displacement_shape, order="F")
        displacement = np.stack(
            [
                expand_spline_coefficients(
                    coefficients[..., component],
                    self.level_images.reference.shape,
                    self.displacement_knot_spacing,
                )
                for component in range(3)
            ],
            axis=3,
        )
        if self.level.estimate_intensity:
            bias_start = 3 * self.displacement_size
            bias_coefficients = values[
                bias_start : bias_start + self.bias_size
            ].reshape(self.template_mapping.bias_coefficients.shape, order="F")
            global_coefficients = values[bias_start + self.bias_size :].copy()
            bias_field = expand_spline_coefficients(
                bias_coefficients,
                self.level_images.reference.shape,
                self.template_mapping.bias_knot_spacing,
            ).astype(np.float32)
            mapping = FnirtIntensityMapping(
                global_coefficients=global_coefficients,
                bias_coefficients=bias_coefficients,
                bias_field=bias_field,
                bias_knot_spacing=self.template_mapping.bias_knot_spacing,
            )
        else:
            mapping = self.template_mapping
        self._parameters = values.copy()
        self._coefficients = coefficients
        self._displacement = displacement
        self._mapping = mapping
        return coefficients, displacement, mapping

    def cost(self, parameters: np.ndarray) -> float:
        """Return the current FSL objective value."""

        coefficients, displacement, mapping = self.unpack(parameters)
        evaluation = evaluate_fnirt_cost(
            self.level_images,
            self.reference_affine,
            self.moving_affine,
            self.affine_matrix,
            mapping,
            self.level,
            nonlinear_displacement=displacement,
            displacement_coefficients=coefficients,
            displacement_knot_spacing=self.displacement_knot_spacing,
            bias_regularization_weight=self.bias_regularization_weight,
        )
        self._cost_parameters = np.asarray(parameters, dtype=np.float64).copy()
        self._cost_evaluation = evaluation
        return evaluation.total

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Return the current FSL analytic gradient."""

        coefficients, displacement, mapping = self.unpack(parameters)
        values = np.asarray(parameters, dtype=np.float64)
        cached_cost = (
            self._cost_evaluation
            if self._cost_parameters is not None
            and np.array_equal(values, self._cost_parameters)
            else None
        )
        evaluation = evaluate_fnirt_gradient(
            self.level_images,
            self.reference_affine,
            self.moving_affine,
            self.affine_matrix,
            mapping,
            self.level,
            nonlinear_displacement=displacement,
            displacement_coefficients=coefficients,
            displacement_knot_spacing=self.displacement_knot_spacing,
            bias_regularization_weight=self.bias_regularization_weight,
            _cost_evaluation=cached_cost,
        )
        self._cost_parameters = None
        self._cost_evaluation = None
        self._gradient_parameters = values.copy()
        self._gradient_evaluation = evaluation
        return evaluation.gradient

    def hessian(self, parameters: np.ndarray) -> csc_matrix:
        """Return the current sparse FSL Gauss-Newton Hessian."""

        coefficients, displacement, mapping = self.unpack(parameters)
        values = np.asarray(parameters, dtype=np.float64)
        cached_gradient = (
            self._gradient_evaluation
            if self._gradient_parameters is not None
            and np.array_equal(values, self._gradient_parameters)
            else None
        )
        if self._displacement_regularization is None:

            def build_regularization(
                spacing: tuple[int, int, int],
            ) -> csc_matrix:
                """Prepare a fixed regularization matrix reused across LM iterations within a level."""

                return bending_energy_hessian(
                    self.level_images.reference.shape,
                    self.level_images.reference_voxel_sizes_mm,
                    spacing,
                )

            if self.level.estimate_intensity:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    displacement_future = executor.submit(
                        build_regularization, self.displacement_knot_spacing
                    )
                    bias_future = executor.submit(
                        build_regularization, mapping.bias_knot_spacing
                    )
                    self._displacement_regularization = displacement_future.result()
                    self._bias_regularization = bias_future.result()
            else:
                self._displacement_regularization = build_regularization(
                    self.displacement_knot_spacing
                )
        evaluation = evaluate_fnirt_hessian(
            self.level_images,
            self.reference_affine,
            self.moving_affine,
            self.affine_matrix,
            mapping,
            self.level,
            nonlinear_displacement=displacement,
            displacement_coefficients=coefficients,
            displacement_knot_spacing=self.displacement_knot_spacing,
            bias_regularization_weight=self.bias_regularization_weight,
            workers=self.workers,
            _gradient_evaluation=cached_gradient,
            _displacement_regularization=self._displacement_regularization,
            _bias_regularization=self._bias_regularization,
        )
        self._gradient_parameters = None
        self._gradient_evaluation = None
        return evaluation.hessian


def optimize_fnirt_level(
    level_images: FnirtLevelImages,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
    affine_matrix: np.ndarray,
    intensity_mapping: FnirtIntensityMapping,
    level: FnirtLevel,
    *,
    initial_displacement_coefficients: np.ndarray | None = None,
    displacement_knot_spacing: tuple[int, int, int] = (2, 2, 2),
    bias_regularization_weight: float = 10_000.0,
    workers: int = 1,
    progress: Callable[[str, int, int, float | None], None] | None = None,
) -> FnirtLevelOptimization:
    """Run FSL's LM sequence for one frozen FNIRT resolution level."""

    from .topup_optimizer import fsl_levenberg_marquardt

    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    objective = _FnirtLevelObjective(
        level_images,
        reference_affine,
        moving_affine,
        affine_matrix,
        intensity_mapping,
        level,
        displacement_knot_spacing,
        bias_regularization_weight,
        workers,
    )
    coefficients = (
        np.zeros(objective.displacement_shape + (3,), dtype=np.float64)
        if initial_displacement_coefficients is None
        else np.asarray(initial_displacement_coefficients, dtype=np.float64)
    )
    if coefficients.shape != objective.displacement_shape + (3,) or not np.all(
        np.isfinite(coefficients)
    ):
        raise ValueError("initial displacement coefficients have an invalid shape")
    blocks = [
        coefficients[..., component].reshape(-1, order="F") for component in range(3)
    ]
    if level.estimate_intensity:
        blocks.extend(
            (
                intensity_mapping.bias_coefficients.reshape(-1, order="F"),
                np.asarray(intensity_mapping.global_coefficients, dtype=np.float64),
            )
        )
    initial_parameters = np.concatenate(blocks)
    result = fsl_levenberg_marquardt(
        initial_parameters,
        objective.cost,
        objective.gradient,
        objective.hessian,
        max_iterations=level.maximum_iterations,
        equation_tolerance=1.0e-3,
        equation_max_iterations=500,
        progress=progress,
    )
    optimized_coefficients, _, optimized_mapping = objective.unpack(result.parameters)
    return FnirtLevelOptimization(
        displacement_coefficients=optimized_coefficients,
        intensity_mapping=optimized_mapping,
        parameters=result.parameters,
        cost=result.cost,
        status=result.status,
        successful_iterations=result.iterations,
        trace=result.trace,
    )


def _spline_basis_at_coordinates(
    coordinates: np.ndarray,
    coefficient_size: int,
    knot_spacing: int,
    *,
    derivative: int = 0,
) -> np.ndarray:
    """Compute basis functions in FSL spline order, including those outside the valid FOV."""

    return np.asarray(
        [
            [
                cubic_spline_value(
                    float(coordinate), coefficient, knot_spacing, derivative
                )
                for coefficient in range(coefficient_size)
            ]
            for coordinate in coordinates
        ],
        dtype=np.float64,
    )


def _fnirt_full_jacobian_range(
    coefficients: np.ndarray,
    reference_shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    affine_matrix: np.ndarray,
    knot_spacing: tuple[int, int, int],
) -> tuple[float, float]:
    """Reproduce ``fnirt_CF::JacobianRange``, including the initial affine."""

    basis = tuple(
        spline_design_matrix(reference_shape[axis], knot_spacing[axis])
        for axis in range(3)
    )
    derivatives = tuple(
        spline_design_matrix(reference_shape[axis], knot_spacing[axis], derivative=1)
        for axis in range(3)
    )
    expand = (
        _expand_fnirt_coefficients_parallel
        if np.prod(reference_shape) >= 100_000
        else _expand_fnirt_coefficients_serial
    )
    _, jacobian = expand(coefficients, *basis, *derivatives)
    jacobian /= np.asarray(voxel_sizes_mm, dtype=np.float64)
    jacobian += np.linalg.inv(affine_matrix)[:3, :3]
    determinant = np.linalg.det(jacobian)
    return float(np.min(determinant)), float(np.max(determinant))


def _constrain_fnirt_warpfield(
    coefficients: np.ndarray,
    reference_shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    affine_matrix: np.ndarray,
    knot_spacing: tuple[int, int, int],
    *,
    max_tries: int,
    minimum_jacobian: float = 0.01,
    maximum_jacobian: float = 100.0,
    progress: Callable[[int, int, float], None] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Reproduce the outer loops of ``constrain_warpfield`` and ``ForceJacobianRange``."""

    current = np.asarray(coefficients, dtype=np.float64).copy()
    current_range = _fnirt_full_jacobian_range(
        current,
        reference_shape,
        voxel_sizes_mm,
        affine_matrix,
        knot_spacing,
    )
    if progress is not None:
        progress(0, max_tries, current_range[0])
    for attempt in range(max_tries):
        if (
            current_range[0] >= minimum_jacobian
            and current_range[1] <= maximum_jacobian
        ):
            break
        extended_shape = tuple(fsl_good_fft_size(size) for size in reference_shape)
        offsets = tuple(
            (extended - original) // 2
            for extended, original in zip(extended_shape, reference_shape, strict=True)
        )
        basis = []
        derivatives = []
        for axis in range(3):
            coordinates = (
                np.arange(extended_shape[axis], dtype=np.float64) - offsets[axis]
            )
            basis.append(
                _spline_basis_at_coordinates(
                    coordinates,
                    current.shape[axis],
                    knot_spacing[axis],
                )
            )
            derivatives.append(
                _spline_basis_at_coordinates(
                    coordinates,
                    current.shape[axis],
                    knot_spacing[axis],
                    derivative=1,
                )
            )
        expand = (
            _expand_fnirt_coefficients_parallel
            if np.prod(extended_shape) >= 100_000
            else _expand_fnirt_coefficients_serial
        )
        nonlinear, _ = expand(current, *basis, *derivatives)
        sampling = np.diag((*voxel_sizes_mm, 1.0))
        affine_mapping = np.zeros((4, 4), dtype=np.float64)
        if np.max(np.abs(affine_matrix - np.eye(4, dtype=np.float64))) > 1.0e-6:
            affine_mapping = (
                np.linalg.inv(affine_matrix) - np.eye(4, dtype=np.float64)
            ) @ sampling
        grid = np.ogrid[: extended_shape[0], : extended_shape[1], : extended_shape[2]]
        relative = nonlinear.astype(np.float32)
        for component in range(3):
            relative[..., component] += np.asarray(
                affine_mapping[component, 0] * grid[0]
                + affine_mapping[component, 1] * grid[1]
                + affine_mapping[component, 2] * grid[2]
                + affine_mapping[component, 3],
                dtype=np.float32,
            )
            relative[..., component] += np.asarray(
                grid[component] * voxel_sizes_mm[component], dtype=np.float32
            )
        constrained = fsl_constrain_topology(
            relative,
            voxel_sizes_mm,
            minimum_jacobian=minimum_jacobian,
            maximum_jacobian=maximum_jacobian,
        )
        for component in range(3):
            constrained[..., component] -= np.asarray(
                grid[component] * voxel_sizes_mm[component], dtype=np.float32
            )
            constrained[..., component] -= np.asarray(
                affine_mapping[component, 0] * grid[0]
                + affine_mapping[component, 1] * grid[1]
                + affine_mapping[component, 2] * grid[2]
                + affine_mapping[component, 3],
                dtype=np.float32,
            )
        slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(offsets, reference_shape, strict=True)
        )
        cropped = constrained[slices]
        updated = np.empty_like(current)
        for component in range(3):
            updated[..., component] = fit_spline_coefficients(
                cropped[..., component], knot_spacing
            )
        previous_range = current_range
        current = updated
        current_range = _fnirt_full_jacobian_range(
            current,
            reference_shape,
            voxel_sizes_mm,
            affine_matrix,
            knot_spacing,
        )
        if progress is not None:
            progress(attempt + 1, max_tries, current_range[0])
        if (
            abs(previous_range[0] - current_range[0]) < 1.0e-6
            or abs(previous_range[1] - current_range[1]) < 1.0e-6
        ):
            break
    return current, current_range


def run_simnibs46_fnirt(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
    affine_matrix: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    workers: int = 1,
    progress: Callable[[int, str, int, int, float | None], None] | None = None,
) -> FnirtRunResult:
    """Run the exact four-level FNIRT schedule used by SimNIBS 4.6."""

    reference_transform = _validate_affine(reference_affine, "reference_affine")
    moving_transform = _validate_affine(moving_affine, "moving_affine")
    affine = _validate_affine(affine_matrix, "affine_matrix")
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    prepared = prepare_fnirt_images(
        reference,
        moving,
        reference_mask=reference_mask,
        moving_mask=moving_mask,
    )
    full_shape = prepared.reference.shape
    voxel_sizes = tuple(
        float(value) for value in np.linalg.norm(reference_transform[:3, :3], axis=0)
    )
    moving_voxel_sizes = tuple(
        float(value) for value in np.linalg.norm(moving_transform[:3, :3], axis=0)
    )
    displacement_spacing = fsl_fnirt_full_resolution_knot_spacing(voxel_sizes)
    if (
        np.array_equal(np.asarray(reference), np.asarray(moving))
        and np.array_equal(reference_transform, moving_transform)
        and np.all(np.abs(affine - np.eye(4)) < 1.0e-8)
    ):
        coefficients = np.zeros(
            fsl_coefficient_shape(full_shape, displacement_spacing) + (3,),
            dtype=np.float64,
        )
        bias_spacing = fsl_fnirt_bias_knot_spacing(voxel_sizes, 1)
        bias_coefficients = fit_spline_coefficients(
            np.ones(full_shape, dtype=np.float32), bias_spacing
        )
        mapping = FnirtIntensityMapping(
            global_coefficients=np.asarray(
                [0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            ),
            bias_coefficients=bias_coefficients,
            bias_field=np.ones(full_shape, dtype=np.float32),
            bias_knot_spacing=bias_spacing,
        )
        expansion = expand_fnirt_coefficients(
            coefficients,
            full_shape,
            reference_transform,
            affine,
            knot_spacing=displacement_spacing,
        )
        return FnirtRunResult(
            coefficients=coefficients,
            intensity_mapping=mapping,
            expansion=expansion,
            levels=(),
            jacobian_ranges=(),
        )
    coefficients: np.ndarray | None = None
    mapping: FnirtIntensityMapping | None = None
    previous_images: FnirtLevelImages | None = None
    previous_spacing: tuple[int, int, int] | None = None
    level_results = []
    jacobian_ranges = []
    for level_index, level in enumerate(SIMNIBS46_FNIRT_LEVELS, start=1):
        if progress is not None:
            progress(level_index, "prepare", 0, 1, None)
        level_images = prepare_fnirt_level_images(
            prepared,
            voxel_sizes,
            level,
            moving_voxel_sizes_mm=moving_voxel_sizes,
        )
        if coefficients is None:
            coefficient_shape = fsl_coefficient_shape(
                level_images.reference.shape, displacement_spacing
            )
            coefficients = np.zeros(coefficient_shape + (3,), dtype=np.float64)
            mapping = initialize_fnirt_intensity_mapping(full_shape, voxel_sizes, level)
        else:
            assert previous_images is not None
            assert mapping is not None
            assert previous_spacing is not None
            zoomed = np.empty(
                fsl_coefficient_shape(
                    level_images.reference.shape, displacement_spacing
                )
                + (3,),
                dtype=np.float64,
            )
            for component in range(3):
                zoomed[..., component] = zoom_fnirt_spline_coefficients(
                    coefficients[..., component],
                    previous_images.reference.shape,
                    previous_images.reference_voxel_sizes_mm,
                    displacement_spacing,
                    level_images.reference.shape,
                    level_images.reference_voxel_sizes_mm,
                    displacement_spacing,
                )
            coefficients = zoomed
            new_bias_spacing = fsl_fnirt_bias_knot_spacing(
                voxel_sizes, level.subsampling
            )
            bias_coefficients = zoom_fnirt_spline_coefficients(
                mapping.bias_coefficients,
                previous_images.reference.shape,
                previous_images.reference_voxel_sizes_mm,
                previous_spacing,
                level_images.reference.shape,
                level_images.reference_voxel_sizes_mm,
                new_bias_spacing,
            )
            mapping = FnirtIntensityMapping(
                global_coefficients=mapping.global_coefficients.copy(),
                bias_coefficients=bias_coefficients,
                bias_field=expand_spline_coefficients(
                    bias_coefficients,
                    level_images.reference.shape,
                    new_bias_spacing,
                ).astype(np.float32),
                bias_knot_spacing=new_bias_spacing,
            )
        if progress is not None:
            progress(level_index, "optimize", 0, level.maximum_iterations, None)
        result = optimize_fnirt_level(
            level_images,
            fsl_fnirt_level_affine(reference_transform, level.subsampling),
            moving_transform,
            affine,
            mapping,
            level,
            initial_displacement_coefficients=coefficients,
            displacement_knot_spacing=displacement_spacing,
            workers=workers,
            progress=(
                None
                if progress is None
                else lambda phase, done, total, value: progress(
                    level_index, phase, done, total, value
                )
            ),
        )
        if progress is not None:
            progress(level_index, "complete", 1, 1, result.cost)
        coefficients = result.displacement_coefficients
        mapping = result.intensity_mapping
        current_range = _fnirt_full_jacobian_range(
            coefficients,
            level_images.reference.shape,
            level_images.reference_voxel_sizes_mm,
            affine,
            displacement_spacing,
        )
        if current_range[0] < 0.01 or current_range[1] > 100.0:
            coefficients, current_range = _constrain_fnirt_warpfield(
                coefficients,
                level_images.reference.shape,
                level_images.reference_voxel_sizes_mm,
                affine,
                displacement_spacing,
                max_tries=5 if level_index == 1 else 10,
                progress=(
                    None
                    if progress is None
                    else lambda done, total, value: progress(
                        level_index, "topology", done, total, value
                    )
                ),
            )
            parameter_count = int(np.prod(coefficients.shape[:3]))
            constrained_parameters = result.parameters.copy()
            for component in range(3):
                start = component * parameter_count
                constrained_parameters[start : start + parameter_count] = coefficients[
                    ..., component
                ].reshape(-1, order="F")
            result = FnirtLevelOptimization(
                displacement_coefficients=coefficients,
                intensity_mapping=result.intensity_mapping,
                parameters=constrained_parameters,
                cost=result.cost,
                status=result.status,
                successful_iterations=result.successful_iterations,
                trace=result.trace,
            )
        jacobian_ranges.append(current_range)
        level_results.append(result)
        previous_images = level_images
        previous_spacing = mapping.bias_knot_spacing
    assert coefficients is not None
    assert mapping is not None
    assert previous_images is not None
    assert previous_spacing is not None
    full_coefficients = np.empty(
        fsl_coefficient_shape(full_shape, displacement_spacing) + (3,),
        dtype=np.float64,
    )
    for component in range(3):
        full_coefficients[..., component] = zoom_fnirt_spline_coefficients(
            coefficients[..., component],
            previous_images.reference.shape,
            previous_images.reference_voxel_sizes_mm,
            displacement_spacing,
            full_shape,
            voxel_sizes,
            displacement_spacing,
        )
    full_bias_spacing = fsl_fnirt_bias_knot_spacing(voxel_sizes, 1)
    full_bias_coefficients = zoom_fnirt_spline_coefficients(
        mapping.bias_coefficients,
        previous_images.reference.shape,
        previous_images.reference_voxel_sizes_mm,
        previous_spacing,
        full_shape,
        voxel_sizes,
        full_bias_spacing,
    )
    full_mapping = FnirtIntensityMapping(
        global_coefficients=mapping.global_coefficients.copy(),
        bias_coefficients=full_bias_coefficients,
        bias_field=expand_spline_coefficients(
            full_bias_coefficients, full_shape, full_bias_spacing
        ).astype(np.float32),
        bias_knot_spacing=full_bias_spacing,
    )
    expansion = expand_fnirt_coefficients(
        full_coefficients,
        full_shape,
        reference_transform,
        affine,
        knot_spacing=displacement_spacing,
    )
    return FnirtRunResult(
        coefficients=full_coefficients,
        intensity_mapping=full_mapping,
        expansion=expansion,
        levels=tuple(level_results),
        jacobian_ranges=tuple(jacobian_ranges),
    )


def fsl_spm_like_mean(values: np.ndarray) -> float:
    """Return the global mean used by FNIRT before optimization."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("FNIRT input must be a finite three-dimensional array")
    mean = float(_spm_like_mean_fsl_order(source))
    if not np.isfinite(mean) or mean == 0.0:
        raise ValueError("FNIRT cannot normalize an input with an empty foreground")
    return mean


def _fnirt_mask(values: np.ndarray, explicit_mask: np.ndarray | None) -> np.ndarray:
    mask = np.ones(values.shape, dtype=bool)
    if explicit_mask is not None:
        explicit = np.asarray(explicit_mask)
        if explicit.shape != values.shape or not np.all(np.isin(explicit, (0, 1))):
            raise ValueError("FNIRT masks must be binary and match their input image")
        mask &= explicit.astype(bool)
    mask &= np.abs(values.astype(np.float64)) >= 1e-16
    return mask


def prepare_fnirt_images(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
) -> FnirtPreparedImages:
    """Create masks before scaling both FNIRT inputs to a foreground mean of 100."""

    fixed = np.asarray(reference, dtype=np.float32)
    source = np.asarray(moving, dtype=np.float32)
    if fixed.ndim != 3 or source.ndim != 3:
        raise ValueError("FNIRT inputs must be three-dimensional")
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(source)):
        raise ValueError("FNIRT inputs must be finite")
    fixed_mask = _fnirt_mask(fixed, reference_mask)
    source_mask = _fnirt_mask(source, moving_mask)
    fixed_mean = fsl_spm_like_mean(fixed)
    source_mean = fsl_spm_like_mean(source)
    return FnirtPreparedImages(
        reference=np.multiply(fixed, np.float32(100.0 / fixed_mean), dtype=np.float32),
        moving=np.multiply(source, np.float32(100.0 / source_mean), dtype=np.float32),
        reference_mask=fixed_mask,
        moving_mask=source_mask,
        reference_mean=fixed_mean,
        moving_mean=source_mean,
    )


def fsl_fnirt_subsampled_shape(
    shape: tuple[int, int, int], subsampling: int
) -> tuple[int, int, int]:
    """Apply ``splinefield::next_size_down`` recursively."""

    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("shape must contain three positive dimensions")
    if not isinstance(subsampling, (int, np.integer)) or subsampling < 1:
        raise ValueError("subsampling must be a positive power of two")
    factor = int(subsampling)
    if factor & (factor - 1):
        raise ValueError("subsampling must be a positive power of two")
    output = [int(value) for value in shape]
    while factor > 1:
        output = [
            ((value + 1) // 2 if value % 2 else value // 2 + 1) for value in output
        ]
        factor //= 2
    return tuple(output)


def fsl_fnirt_level_affine(
    reference_affine: np.ndarray, subsampling: int
) -> np.ndarray:
    """Scale voxel axes while retaining the world coordinate of voxel zero."""

    affine = _validate_affine(reference_affine, "reference_affine")
    fsl_fnirt_subsampled_shape((1, 1, 1), subsampling)
    scale = np.diag((float(subsampling), float(subsampling), float(subsampling), 1.0))
    return affine @ scale


def _sample_reference_grid(values: np.ndarray, subsampling: int) -> np.ndarray:
    shape = fsl_fnirt_subsampled_shape(values.shape, subsampling)
    output = np.zeros(shape, dtype=values.dtype)
    valid_shape = tuple(
        min(shape[axis], (values.shape[axis] - 1) // subsampling + 1)
        for axis in range(3)
    )
    destination = tuple(slice(0, value) for value in valid_shape)
    source = tuple(slice(0, value * subsampling, subsampling) for value in valid_shape)
    output[destination] = values[source]
    return output


def prepare_fnirt_level_images(
    prepared: FnirtPreparedImages,
    voxel_sizes_mm: tuple[float, float, float],
    level: FnirtLevel,
    *,
    moving_voxel_sizes_mm: tuple[float, float, float] | None = None,
) -> FnirtLevelImages:
    """Build one pyramid level using the physical voxel sizes of reference and moving separately."""

    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if voxel_sizes.shape != (3,) or not np.all(np.isfinite(voxel_sizes)):
        raise ValueError("voxel_sizes_mm must contain three finite values")
    if np.any(voxel_sizes <= 0.0):
        raise ValueError("voxel_sizes_mm must be positive")
    moving_voxel_sizes = (
        voxel_sizes
        if moving_voxel_sizes_mm is None
        else np.asarray(moving_voxel_sizes_mm, dtype=np.float64)
    )
    if moving_voxel_sizes.shape != (3,) or not np.all(np.isfinite(moving_voxel_sizes)):
        raise ValueError("moving_voxel_sizes_mm must contain three finite values")
    if np.any(moving_voxel_sizes <= 0.0):
        raise ValueError("moving_voxel_sizes_mm must be positive")
    fixed = prepared.reference
    if level.reference_fwhm_mm > 0.0:
        fixed = fsl_fnirt_smooth(
            fixed,
            level.reference_fwhm_mm / np.sqrt(8.0 * np.log(2.0)),
            tuple(voxel_sizes),
        )
    source = prepared.moving
    if level.input_fwhm_mm > 0.0:
        sigma = level.input_fwhm_mm / np.sqrt(8.0 * np.log(2.0))
        weighted = fsl_fnirt_smooth(
            source * prepared.moving_mask,
            sigma,
            tuple(moving_voxel_sizes),
        )
        weights = fsl_fnirt_smooth(
            prepared.moving_mask.astype(np.float32),
            sigma,
            tuple(moving_voxel_sizes),
        )
        source = np.zeros_like(source)
        np.divide(weighted, weights, out=source, where=weights != 0.0)
        source[~prepared.moving_mask] = 0.0
    fixed_level = _sample_reference_grid(fixed, level.subsampling)
    fixed_mask = _sample_reference_grid(
        prepared.reference_mask.astype(np.float32), level.subsampling
    ) > np.float32(0.99)
    return FnirtLevelImages(
        reference=fixed_level,
        moving=source,
        reference_mask=fixed_mask,
        moving_mask=prepared.moving_mask,
        reference_voxel_sizes_mm=tuple(voxel_sizes * level.subsampling),
    )


def warp_fnirt_moving(
    level_images: FnirtLevelImages,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
    affine_matrix: np.ndarray,
    nonlinear_displacement: np.ndarray | None = None,
    *,
    calculate_derivatives: bool = True,
) -> FnirtWarpedMoving:
    """Apply FNIRT's affine plus relative-mm nonlinear pull transform."""

    target_shape = level_images.reference.shape
    reference = _validate_affine(reference_affine, "reference_affine")
    moving = _validate_affine(moving_affine, "moving_affine")
    affine = _validate_affine(affine_matrix, "affine_matrix")
    displacement = (
        np.zeros(target_shape + (3,), dtype=np.float64)
        if nonlinear_displacement is None
        else np.asarray(nonlinear_displacement, dtype=np.float64)
    )
    if displacement.shape != target_shape + (3,) or not np.all(
        np.isfinite(displacement)
    ):
        raise ValueError("nonlinear_displacement must match the level reference grid")
    if level_images.reference_mask.shape != target_shape:
        raise ValueError("reference_mask must match the level reference grid")
    if level_images.moving_mask.shape != level_images.moving.shape:
        raise ValueError("moving_mask must match the moving image")

    reference_sampling = fsl_voxel_to_scaled_mm(target_shape, reference)
    moving_sampling = fsl_voxel_to_scaled_mm(level_images.moving.shape, moving)
    inverse_affine_sampling = np.asarray(
        np.linalg.inv(affine) @ reference_sampling, dtype=np.float32
    )
    moving_mm_to_voxel = np.asarray(np.linalg.inv(moving_sampling), dtype=np.float32)
    moving_voxel_sizes = np.asarray(
        np.linalg.norm(moving[:3, :3], axis=0), dtype=np.float32
    )
    warped, coordinates, derivatives, data_mask, warped_mask, total_mask = (
        _warp_fnirt_fsl_order(
            np.asarray(level_images.moving, dtype=np.float32),
            level_images.moving_mask.astype(np.float32),
            level_images.reference_mask,
            inverse_affine_sampling,
            moving_mm_to_voxel,
            displacement.astype(np.float32),
            moving_voxel_sizes,
            calculate_derivatives,
        )
    )
    return FnirtWarpedMoving(
        values=warped,
        coordinates=coordinates,
        derivatives_per_mm=derivatives,
        data_mask=data_mask,
        warped_moving_mask=warped_mask,
        mask=total_mask,
    )


@njit(cache=True)
def _expand_fnirt_coefficients_serial(
    coefficients: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
    derivative_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse all independent FNIRT outputs with FSL's coefficient order."""

    nx, ny, nz = basis_x.shape[0], basis_y.shape[0], basis_z.shape[0]
    field = np.empty((nx, ny, nz, 3), dtype=np.float64)
    gradient = np.empty((nx, ny, nz, 3, 3), dtype=np.float64)
    for voxel in range(nx * ny * nz):
        voxel_x = voxel % nx
        voxel_y = (voxel // nx) % ny
        voxel_z = voxel // (nx * ny)
        for component in range(3):
            value = 0.0
            dx = 0.0
            dy = 0.0
            dz = 0.0
            for coefficient_z in range(coefficients.shape[2]):
                weight_z = basis_z[voxel_z, coefficient_z]
                dweight_z = derivative_z[voxel_z, coefficient_z]
                if weight_z == 0.0 and dweight_z == 0.0:
                    continue
                for coefficient_y in range(coefficients.shape[1]):
                    weight_y = basis_y[voxel_y, coefficient_y]
                    dweight_y = derivative_y[voxel_y, coefficient_y]
                    weight_zy = weight_z * weight_y
                    dweight_zy = weight_z * dweight_y
                    dz_weight_y = dweight_z * weight_y
                    if weight_zy == 0.0 and dweight_zy == 0.0 and dz_weight_y == 0.0:
                        continue
                    for coefficient_x in range(coefficients.shape[0]):
                        coefficient = coefficients[
                            coefficient_x, coefficient_y, coefficient_z, component
                        ]
                        weight_x = basis_x[voxel_x, coefficient_x]
                        value += coefficient * (weight_zy * weight_x)
                        dx += coefficient * (
                            weight_zy * derivative_x[voxel_x, coefficient_x]
                        )
                        dy += coefficient * (dweight_zy * weight_x)
                        dz += coefficient * (dz_weight_y * weight_x)
            field[voxel_x, voxel_y, voxel_z, component] = value
            gradient[voxel_x, voxel_y, voxel_z, component, 0] = dx
            gradient[voxel_x, voxel_y, voxel_z, component, 1] = dy
            gradient[voxel_x, voxel_y, voxel_z, component, 2] = dz
    return field, gradient


@njit(cache=True, parallel=True)
def _expand_fnirt_coefficients_parallel(
    coefficients: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
    derivative_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallelize independent voxels without changing any voxel reduction."""

    nx, ny, nz = basis_x.shape[0], basis_y.shape[0], basis_z.shape[0]
    field = np.empty((nx, ny, nz, 3), dtype=np.float64)
    gradient = np.empty((nx, ny, nz, 3, 3), dtype=np.float64)
    for voxel in prange(nx * ny * nz):
        voxel_x = voxel % nx
        voxel_y = (voxel // nx) % ny
        voxel_z = voxel // (nx * ny)
        for component in range(3):
            value = 0.0
            dx = 0.0
            dy = 0.0
            dz = 0.0
            for coefficient_z in range(coefficients.shape[2]):
                weight_z = basis_z[voxel_z, coefficient_z]
                dweight_z = derivative_z[voxel_z, coefficient_z]
                if weight_z == 0.0 and dweight_z == 0.0:
                    continue
                for coefficient_y in range(coefficients.shape[1]):
                    weight_y = basis_y[voxel_y, coefficient_y]
                    dweight_y = derivative_y[voxel_y, coefficient_y]
                    weight_zy = weight_z * weight_y
                    dweight_zy = weight_z * dweight_y
                    dz_weight_y = dweight_z * weight_y
                    if weight_zy == 0.0 and dweight_zy == 0.0 and dz_weight_y == 0.0:
                        continue
                    for coefficient_x in range(coefficients.shape[0]):
                        coefficient = coefficients[
                            coefficient_x, coefficient_y, coefficient_z, component
                        ]
                        weight_x = basis_x[voxel_x, coefficient_x]
                        value += coefficient * (weight_zy * weight_x)
                        dx += coefficient * (
                            weight_zy * derivative_x[voxel_x, coefficient_x]
                        )
                        dy += coefficient * (dweight_zy * weight_x)
                        dz += coefficient * (dz_weight_y * weight_x)
            field[voxel_x, voxel_y, voxel_z, component] = value
            gradient[voxel_x, voxel_y, voxel_z, component, 0] = dx
            gradient[voxel_x, voxel_y, voxel_z, component, 1] = dy
            gradient[voxel_x, voxel_y, voxel_z, component, 2] = dz
    return field, gradient


def fsl_fnirt_full_resolution_knot_spacing(
    voxel_sizes_mm: tuple[float, float, float],
    *,
    warp_resolution_mm: float = 10.0,
    final_subsampling: int = 2,
) -> tuple[int, int, int]:
    """Return FNIRT's output coefficient spacing on the full reference grid.

    FNIRT first rounds ``warpres / voxel_size`` on the original grid and then
    performs integer division by the final subsampling factor when it writes
    the full-resolution coefficient field.  This reproduces
    ``basisfield::FullResKsp`` rather than rounding at the final pyramid level.
    """

    if not isinstance(final_subsampling, (int, np.integer)) or final_subsampling < 1:
        raise ValueError("final_subsampling must be a positive integer")
    coarse_spacing = fsl_knot_spacing(warp_resolution_mm, voxel_sizes_mm)
    spacing = tuple(value // int(final_subsampling) for value in coarse_spacing)
    if any(value == 0 for value in spacing):
        raise ValueError(
            "FNIRT subsampling, warp resolution and voxel size are incompatible"
        )
    return spacing


def _validate_affine(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(values[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be homogeneous")
    if abs(float(np.linalg.det(values[:3, :3]))) <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must be invertible")
    return values


def _fnirt_affine_displacement(
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """Expand the affine part using ``FnirtFileReader::add_affine_part``."""

    sampling = fsl_voxel_to_scaled_mm(reference_shape, reference_affine)
    mapping = np.zeros((4, 4), dtype=np.float64)
    if np.max(np.abs(affine_matrix - np.eye(4, dtype=np.float64))) > 1.0e-6:
        mapping = (
            np.linalg.inv(affine_matrix) - np.eye(4, dtype=np.float64)
        ) @ sampling
    x, y, z = np.ogrid[: reference_shape[0], : reference_shape[1], : reference_shape[2]]
    output = np.empty(reference_shape + (3,), dtype=np.float64)
    for component in range(3):
        output[..., component] = (
            mapping[component, 0] * x
            + mapping[component, 1] * y
            + mapping[component, 2] * z
            + mapping[component, 3]
        )
    return output


def expand_fnirt_coefficients(
    coefficients: np.ndarray,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    affine_matrix: np.ndarray,
    *,
    knot_spacing: tuple[int, int, int] | None = None,
    warp_resolution_mm: float = 10.0,
    final_subsampling: int = 2,
) -> FnirtWarpExpansion:
    """Expand an FSL cubic coefficient warp without changing its arithmetic.

    The returned dense displacement includes the inverse affine contribution
    used by ``FnirtFileReader``.  The analytic Jacobian and determinant contain
    only the nonlinear spline contribution, matching FNIRT ``--jout`` and
    ``SaveJacobian``.  Tensor PPD must instead differentiate ``displacement``
    because VECREG operates on the complete pull field.
    """

    target_shape = tuple(int(value) for value in reference_shape)
    if len(target_shape) != 3 or any(value < 2 for value in target_shape):
        raise ValueError("reference_shape must contain three dimensions of size >= 2")
    reference = _validate_affine(reference_affine, "reference_affine")
    affine = _validate_affine(affine_matrix, "affine_matrix")
    voxel_sizes = tuple(
        float(value) for value in np.linalg.norm(reference[:3, :3], axis=0)
    )
    spacing = (
        fsl_fnirt_full_resolution_knot_spacing(
            voxel_sizes,
            warp_resolution_mm=warp_resolution_mm,
            final_subsampling=final_subsampling,
        )
        if knot_spacing is None
        else tuple(int(value) for value in knot_spacing)
    )
    if len(spacing) != 3 or any(value < 1 for value in spacing):
        raise ValueError("knot_spacing must contain three positive integers")
    values = np.asarray(coefficients, dtype=np.float64)
    expected = fsl_coefficient_shape(target_shape, spacing) + (3,)
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"coefficients must be finite with shape {expected}")

    basis = tuple(
        spline_design_matrix(target_shape[axis], spacing[axis]) for axis in range(3)
    )
    derivatives = tuple(
        spline_design_matrix(target_shape[axis], spacing[axis], derivative=1)
        for axis in range(3)
    )
    expand = (
        _expand_fnirt_coefficients_parallel
        if np.prod(target_shape) >= 100_000
        else _expand_fnirt_coefficients_serial
    )
    nonlinear, nonlinear_jacobian = expand(values, *basis, *derivatives)
    nonlinear_jacobian /= np.asarray(voxel_sizes, dtype=np.float64)
    for axis in range(3):
        nonlinear_jacobian[..., axis, axis] += 1.0
    affine_displacement = _fnirt_affine_displacement(target_shape, reference, affine)
    return FnirtWarpExpansion(
        nonlinear_displacement=nonlinear,
        affine_displacement=affine_displacement,
        displacement=nonlinear + affine_displacement,
        nonlinear_jacobian=nonlinear_jacobian,
        nonlinear_jacobian_determinant=np.linalg.det(nonlinear_jacobian),
        knot_spacing=spacing,
    )
