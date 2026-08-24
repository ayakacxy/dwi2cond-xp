"""FSL FLIRT weighted mutual-information cost evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from threading import local

from numba import njit, prange
import numpy as np

from ._numba import set_available_numba_threads


def _finite_4x4(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if abs(np.linalg.det(values[:3, :3])) < 1e-12:
        raise ValueError(f"{name} must be invertible")
    return values


def _float_volume(volume: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) < 2:
        raise ValueError(f"{name} must be a three-dimensional array of size at least two")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    # FSL volume memory uses x as the contiguous axis; Fortran order matches the
    # z/y/x traversal contract.
    return np.asfortranarray(values)


def _weight_x_ranges(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero weight bounds per y/z row, using empty intervals for zero rows."""

    nonzero = weight != np.float32(0.0)
    valid_rows = np.any(nonzero, axis=0)
    first = np.argmax(nonzero, axis=0).astype(np.int32)
    last = (weight.shape[0] - 1 - np.argmax(nonzero[::-1], axis=0)).astype(np.int32)
    first[~valid_rows] = 0
    last[~valid_rows] = -1
    return np.asfortranarray(first), np.asfortranarray(last)


def _ordered_weight_workset(
    weight: np.ndarray, reference_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack nonzero reference-weight voxels in FSL z/y/x order."""

    xsize = weight.shape[0]
    flattened_weight = weight.ravel(order="F")
    active_indices = np.flatnonzero(flattened_weight)
    row_counts = np.bincount(
        active_indices // xsize, minlength=weight.shape[1] * weight.shape[2]
    )
    row_offsets = np.empty(row_counts.size + 1, dtype=np.int64)
    row_offsets[0] = 0
    np.cumsum(row_counts, out=row_offsets[1:])
    return (
        row_offsets,
        np.asarray(active_indices % xsize, dtype=np.int32),
        np.asarray(flattened_weight[active_indices], dtype=np.float32),
        np.asarray(reference_bins.ravel(order="F")[active_indices], dtype=np.int32),
    )


@njit(cache=True, nogil=True)
def _find_range_x(
    o1: np.float32,
    o2: np.float32,
    o3: np.float32,
    a11: np.float32,
    a21: np.float32,
    a31: np.float32,
    xb1: int,
    xb2: np.float32,
    yb2: np.float32,
    zb2: np.float32,
) -> tuple[int, int]:
    xmin0 = np.float32(0.0)
    xmax0 = np.float32(xb1)
    origins = (o1, o2, o3)
    slopes = (a11, a21, a31)
    bounds = (xb2, yb2, zb2)
    for index in range(3):
        origin = origins[index]
        slope = slopes[index]
        bound = bounds[index]
        if abs(slope) < np.float32(1.0e-8):
            if np.float32(0.0) <= origin <= bound:
                x1 = np.float32(-1.0e8)
                x2 = np.float32(1.0e8)
            else:
                x1 = np.float32(-1.0e8)
                x2 = np.float32(-1.0e8)
        else:
            x1 = np.float32(-origin / slope)
            x2 = np.float32((bound - origin) / slope)
        xmin0 = max(xmin0, min(x1, x2))
        xmax0 = min(xmax0, max(x1, x2))
    if xmax0 < xmin0:
        return 1, 0
    xmin = int(math.ceil(float(xmin0)))
    xmax = int(math.floor(float(xmax0)))
    xcoord = np.float32(o1 + np.float32(xmin) * a11)
    ycoord = np.float32(o2 + np.float32(xmin) * a21)
    zcoord = np.float32(o3 + np.float32(xmin) * a31)
    for xindex in range(xmin, xmax + 1):
        in_bounds = (
            np.float32(0.0) <= xcoord <= xb2
            and np.float32(0.0) <= ycoord <= yb2
            and np.float32(0.0) <= zcoord <= zb2
        )
        if xindex == xmin and not in_bounds:
            xmin += 1
        elif not in_bounds:
            return xmin, xindex - 1
        xcoord = np.float32(xcoord + a11)
        ycoord = np.float32(ycoord + a21)
        zcoord = np.float32(zcoord + a31)
    return xmin, xmax


@njit(cache=True, nogil=True)
def _trilinear(volume: np.ndarray, x: np.float32, y: np.float32, z: np.float32) -> np.float32:
    ix, iy, iz = int(x), int(y), int(z)
    dx = np.float32(x - np.float32(ix))
    dy = np.float32(y - np.float32(iy))
    dz = np.float32(z - np.float32(iz))
    v000 = volume[ix, iy, iz]
    v001 = volume[ix, iy, iz + 1]
    v010 = volume[ix, iy + 1, iz]
    v011 = volume[ix, iy + 1, iz + 1]
    v100 = volume[ix + 1, iy, iz]
    v101 = volume[ix + 1, iy, iz + 1]
    v110 = volume[ix + 1, iy + 1, iz]
    v111 = volume[ix + 1, iy + 1, iz + 1]
    temp1 = np.float32(np.float32(v100 - v000) * dx + v000)
    temp2 = np.float32(np.float32(v101 - v001) * dx + v001)
    temp3 = np.float32(np.float32(v110 - v010) * dx + v010)
    temp4 = np.float32(np.float32(v111 - v011) * dx + v011)
    temp5 = np.float32(np.float32(temp3 - temp1) * dy + temp1)
    temp6 = np.float32(np.float32(temp4 - temp2) * dy + temp2)
    return np.float32(np.float32(temp6 - temp5) * dz + temp5)


@njit(cache=True, nogil=True)
def _voxel_intensity_cog(volume: np.ndarray) -> np.ndarray:
    minimum = volume.min()
    limit = max(int(math.sqrt(volume.size)), 1000)
    result = np.zeros(3, dtype=np.float64)
    subtotal = np.zeros(3, dtype=np.float64)
    total = 0.0
    partial_total = 0.0
    count = 0
    for zindex in range(volume.shape[2]):
        for yindex in range(volume.shape[1]):
            for xindex in range(volume.shape[0]):
                value = float(np.float32(volume[xindex, yindex, zindex] - minimum))
                subtotal[0] += value * xindex
                subtotal[1] += value * yindex
                subtotal[2] += value * zindex
                partial_total += value
                count += 1
                if count > limit:
                    count = 0
                    total += partial_total
                    result += subtotal
                    partial_total = 0.0
                    subtotal[:] = 0.0
    total += partial_total
    result += subtotal
    if abs(total) < 1e-5:
        total = 1.0
    return result / total


def flirt_intensity_cog(volume: np.ndarray, sampling_matrix: np.ndarray) -> np.ndarray:
    """Return FSL NEWIMAGE's intensity center of gravity in scaled millimeters."""

    values = _float_volume(volume, "volume")
    sampling = _finite_4x4(sampling_matrix, "sampling_matrix")
    voxel_center = _voxel_intensity_cog(values)
    return (sampling @ np.r_[voxel_center, 1.0])[:3]


@njit(cache=True, nogil=True)
def _weighted_mi_kernel(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_weight: np.ndarray,
    moving_weight: np.ndarray,
    unit_moving_weight: bool,
    reference_first_x: np.ndarray,
    reference_last_x: np.ndarray,
    reference_bins: np.ndarray,
    voxel_mapping: np.ndarray,
    moving_min: np.float32,
    moving_max: np.float32,
    moving_voxel_sizes: np.ndarray,
    bins: int,
    smooth_size: np.float32,
    fuzzy_fraction: np.float32,
) -> np.float32:
    histogram = np.zeros((bins + 1, bins + 1), dtype=np.float32)
    reference_histogram = np.zeros(bins + 1, dtype=np.float32)
    moving_histogram = np.zeros(bins + 1, dtype=np.float32)
    xb1, yb1, zb1 = reference.shape[0] - 1, reference.shape[1] - 1, reference.shape[2] - 1
    xb2 = np.float32(moving.shape[0]) - np.float32(1.0001)
    yb2 = np.float32(moving.shape[1]) - np.float32(1.0001)
    zb2 = np.float32(moving.shape[2]) - np.float32(1.0001)
    matrix = voxel_mapping.astype(np.float32)
    a11, a12, a13, a14 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    a21, a22, a23, a24 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    a31, a32, a33, a34 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    bin_scale = np.float32(np.float32(bins) / np.float32(moving_max - moving_min))
    bin_offset = np.float32(-moving_min * bin_scale)
    smoothx = np.float32(smooth_size / moving_voxel_sizes[0])
    smoothy = np.float32(smooth_size / moving_voxel_sizes[1])
    smoothz = np.float32(smooth_size / moving_voxel_sizes[2])
    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)

    for zindex in range(zb1 + 1):
        zfloat = np.float32(zindex)
        for yindex in range(yb1 + 1):
            first_weighted_x = reference_first_x[yindex, zindex]
            last_weighted_x = reference_last_x[yindex, zindex]
            if last_weighted_x < first_weighted_x:
                continue
            yfloat = np.float32(yindex)
            o1 = np.float32(np.float32(yfloat * a12 + zfloat * a13) + a14)
            o2 = np.float32(np.float32(yfloat * a22 + zfloat * a23) + a24)
            o3 = np.float32(np.float32(yfloat * a32 + zfloat * a33) + a34)
            xmin, xmax = _find_range_x(o1, o2, o3, a11, a21, a31, xb1, xb2, yb2, zb2)
            o1 = np.float32(o1 + np.float32(xmin) * a11)
            o2 = np.float32(o2 + np.float32(xmin) * a21)
            o3 = np.float32(o3 + np.float32(xmin) * a31)
            first_x = max(xmin, first_weighted_x)
            last_x = min(xmax, last_weighted_x)
            for _ in range(xmin, first_x):
                o1 = np.float32(o1 + a11)
                o2 = np.float32(o2 + a21)
                o3 = np.float32(o3 + a31)
            for xindex in range(first_x, last_x + 1):
                ix, iy, iz = int(o1), int(o2), int(o3)
                interpolation_valid = (
                    0 <= ix < moving.shape[0] - 1
                    and 0 <= iy < moving.shape[1] - 1
                    and 0 <= iz < moving.shape[2] - 1
                )
                if (
                    ((xindex != xmin and xindex != xmax) or interpolation_valid)
                    and reference_weight[xindex, yindex, zindex] != zero
                ):
                    value = _trilinear(moving, o1, o2, o3)
                    weight_value = (
                        np.float32(1.0)
                        if unit_moving_weight
                        else _trilinear(moving_weight, o1, o2, o3)
                    )
                    geometric_weight = np.float32(weight_value * reference_weight[xindex, yindex, zindex])
                    if o1 < smoothx:
                        geometric_weight = np.float32(geometric_weight * np.float32(o1 / smoothx))
                    elif np.float32(xb2 - o1) < smoothx:
                        geometric_weight = np.float32(geometric_weight * np.float32(np.float32(xb2 - o1) / smoothx))
                    if o2 < smoothy:
                        geometric_weight = np.float32(geometric_weight * np.float32(o2 / smoothy))
                    elif np.float32(yb2 - o2) < smoothy:
                        geometric_weight = np.float32(geometric_weight * np.float32(np.float32(yb2 - o2) / smoothy))
                    if o3 < smoothz:
                        geometric_weight = np.float32(geometric_weight * np.float32(o3 / smoothz))
                    elif np.float32(zb2 - o3) < smoothz:
                        geometric_weight = np.float32(geometric_weight * np.float32(np.float32(zb2 - o3) / smoothz))
                    if geometric_weight < zero:
                        geometric_weight = zero

                    reference_bin = reference_bins[xindex, yindex, zindex]
                    moving_index = np.float32(value * bin_scale + bin_offset)
                    center = int(moving_index)
                    plus, minus = center + 1, center - 1
                    if center >= bins:
                        center, plus = bins - 1, bins - 1
                    if center < 0:
                        center, minus = 0, 0
                    if plus >= bins:
                        plus = bins - 1
                    if minus < 0:
                        minus = 0
                    fraction = np.float32(abs(np.float32(moving_index - np.float32(int(moving_index)))))
                    if fraction < fuzzy_fraction:
                        center_weight = np.float32(half + half * np.float32(fraction / fuzzy_fraction))
                        minus_weight = np.float32(one - center_weight)
                        plus_weight = zero
                    elif fraction > np.float32(one - fuzzy_fraction):
                        center_weight = np.float32(half + half * np.float32(np.float32(one - fraction) / fuzzy_fraction))
                        plus_weight = np.float32(one - center_weight)
                        minus_weight = zero
                    else:
                        center_weight, plus_weight, minus_weight = one, zero, zero
                    center_value = np.float32(geometric_weight * center_weight)
                    plus_value = np.float32(geometric_weight * plus_weight)
                    minus_value = np.float32(geometric_weight * minus_weight)
                    histogram[reference_bin, center] = np.float32(histogram[reference_bin, center] + center_value)
                    moving_histogram[center] = np.float32(moving_histogram[center] + center_value)
                    histogram[reference_bin, plus] = np.float32(histogram[reference_bin, plus] + plus_value)
                    moving_histogram[plus] = np.float32(moving_histogram[plus] + plus_value)
                    histogram[reference_bin, minus] = np.float32(histogram[reference_bin, minus] + minus_value)
                    moving_histogram[minus] = np.float32(moving_histogram[minus] + minus_value)
                    reference_histogram[reference_bin] = np.float32(
                        reference_histogram[reference_bin] + geometric_weight
                    )
                o1 = np.float32(o1 + a11)
                o2 = np.float32(o2 + a21)
                o3 = np.float32(o3 + a31)

    voxel_count = np.float32(reference.size)
    joint_entropy = np.float32(0.0)
    reference_entropy = np.float32(0.0)
    moving_entropy = np.float32(0.0)
    for first in range(bins + 1):
        for second in range(bins + 1):
            count = histogram[first, second]
            if count > zero:
                probability = np.float32(count / voxel_count)
                joint_entropy = np.float32(joint_entropy - float(probability) * math.log(float(probability)))
    for index in range(bins + 1):
        count = reference_histogram[index]
        if count > zero:
            probability = np.float32(count / voxel_count)
            reference_entropy = np.float32(reference_entropy - float(probability) * math.log(float(probability)))
    overlap = np.float32(0.0)
    for index in range(bins + 1):
        count = moving_histogram[index]
        if count > zero:
            overlap = np.float32(overlap + count)
            probability = np.float32(count / voxel_count)
            moving_entropy = np.float32(moving_entropy - float(probability) * math.log(float(probability)))
    if overlap > zero:
        ratio = np.float32(voxel_count / overlap)
        joint_entropy = np.float32(np.float32(ratio * joint_entropy) - math.log(float(ratio)))
        reference_entropy = np.float32(np.float32(ratio * reference_entropy) - math.log(float(ratio)))
        moving_entropy = np.float32(np.float32(ratio * moving_entropy) - math.log(float(ratio)))
    else:
        reference_entropy = np.float32(math.log(float(bins)))
        moving_entropy = reference_entropy
        joint_entropy = np.float32(2.0 * math.log(float(bins)))
    return np.float32(-(reference_entropy + moving_entropy - joint_entropy))


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _weighted_mi_many_kernel(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_weight: np.ndarray,
    moving_weight: np.ndarray,
    unit_moving_weight: bool,
    reference_first_x: np.ndarray,
    reference_last_x: np.ndarray,
    reference_bins: np.ndarray,
    voxel_mappings: np.ndarray,
    moving_min: np.float32,
    moving_max: np.float32,
    moving_voxel_sizes: np.ndarray,
    bins: int,
    smooth_size: np.float32,
    fuzzy_fraction: np.float32,
) -> np.ndarray:
    """Parallelize candidates while reducing each one in FSL z/y/x order."""

    results = np.empty(voxel_mappings.shape[0], dtype=np.float32)
    for index in prange(voxel_mappings.shape[0]):
        results[index] = _weighted_mi_kernel(
            reference,
            moving,
            reference_weight,
            moving_weight,
            unit_moving_weight,
            reference_first_x,
            reference_last_x,
            reference_bins,
            voxel_mappings[index],
            moving_min,
            moving_max,
            moving_voxel_sizes,
            bins,
            smooth_size,
            fuzzy_fraction,
        )
    return results


@njit(cache=True, nogil=True)
def _weighted_correlation_ratio_kernel(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_weight: np.ndarray,
    moving_weight: np.ndarray,
    unit_moving_weight: bool,
    reference_first_x: np.ndarray,
    reference_last_x: np.ndarray,
    reference_bins: np.ndarray,
    voxel_mapping: np.ndarray,
    moving_voxel_sizes: np.ndarray,
    bins: int,
    smooth_size: np.float32,
) -> np.float32:
    """Reproduce FSL's single-candidate reduction order for fully weighted correlation ratio."""

    counts = np.zeros(bins + 1, dtype=np.float32)
    sums = np.zeros(bins + 1, dtype=np.float32)
    square_sums = np.zeros(bins + 1, dtype=np.float32)
    xb1 = reference.shape[0] - 1
    yb1 = reference.shape[1] - 1
    zb1 = reference.shape[2] - 1
    xb2 = np.float32(moving.shape[0]) - np.float32(1.0001)
    yb2 = np.float32(moving.shape[1]) - np.float32(1.0001)
    zb2 = np.float32(moving.shape[2]) - np.float32(1.0001)
    matrix = voxel_mapping.astype(np.float32)
    a11, a12, a13, a14 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    a21, a22, a23, a24 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    a31, a32, a33, a34 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    smoothx = np.float32(smooth_size / moving_voxel_sizes[0])
    smoothy = np.float32(smooth_size / moving_voxel_sizes[1])
    smoothz = np.float32(smooth_size / moving_voxel_sizes[2])
    zero = np.float32(0.0)

    for zindex in range(zb1 + 1):
        zfloat = np.float32(zindex)
        for yindex in range(yb1 + 1):
            first_weighted_x = reference_first_x[yindex, zindex]
            last_weighted_x = reference_last_x[yindex, zindex]
            if last_weighted_x < first_weighted_x:
                continue
            yfloat = np.float32(yindex)
            o1 = np.float32(np.float32(yfloat * a12 + zfloat * a13) + a14)
            o2 = np.float32(np.float32(yfloat * a22 + zfloat * a23) + a24)
            o3 = np.float32(np.float32(yfloat * a32 + zfloat * a33) + a34)
            xmin, xmax = _find_range_x(o1, o2, o3, a11, a21, a31, xb1, xb2, yb2, zb2)
            o1 = np.float32(o1 + np.float32(xmin) * a11)
            o2 = np.float32(o2 + np.float32(xmin) * a21)
            o3 = np.float32(o3 + np.float32(xmin) * a31)
            first_x = max(xmin, first_weighted_x)
            last_x = min(xmax, last_weighted_x)
            for _ in range(xmin, first_x):
                o1 = np.float32(o1 + a11)
                o2 = np.float32(o2 + a21)
                o3 = np.float32(o3 + a31)
            for xindex in range(first_x, last_x + 1):
                ix, iy, iz = int(o1), int(o2), int(o3)
                interpolation_valid = (
                    0 <= ix < moving.shape[0] - 1
                    and 0 <= iy < moving.shape[1] - 1
                    and 0 <= iz < moving.shape[2] - 1
                )
                if (
                    ((xindex != xmin and xindex != xmax) or interpolation_valid)
                    and reference_weight[xindex, yindex, zindex] != zero
                ):
                    value = _trilinear(moving, o1, o2, o3)
                    moving_weight_value = (
                        np.float32(1.0)
                        if unit_moving_weight
                        else _trilinear(moving_weight, o1, o2, o3)
                    )
                    weight = np.float32(
                        moving_weight_value * reference_weight[xindex, yindex, zindex]
                    )
                    if o1 < smoothx:
                        weight = np.float32(weight * np.float32(o1 / smoothx))
                    elif np.float32(xb2 - o1) < smoothx:
                        weight = np.float32(weight * np.float32(np.float32(xb2 - o1) / smoothx))
                    if o2 < smoothy:
                        weight = np.float32(weight * np.float32(o2 / smoothy))
                    elif np.float32(yb2 - o2) < smoothy:
                        weight = np.float32(weight * np.float32(np.float32(yb2 - o2) / smoothy))
                    if o3 < smoothz:
                        weight = np.float32(weight * np.float32(o3 / smoothz))
                    elif np.float32(zb2 - o3) < smoothz:
                        weight = np.float32(weight * np.float32(np.float32(zb2 - o3) / smoothz))
                    if weight < zero:
                        weight = zero
                    bin_index = reference_bins[xindex, yindex, zindex]
                    counts[bin_index] = np.float32(counts[bin_index] + weight)
                    sums[bin_index] = np.float32(sums[bin_index] + np.float32(weight * value))
                    square_sums[bin_index] = np.float32(
                        square_sums[bin_index]
                        + np.float32(np.float32(weight * value) * value)
                    )
                o1 = np.float32(o1 + a11)
                o2 = np.float32(o2 + a21)
                o3 = np.float32(o3 + a31)

    counts[bins - 1] = np.float32(counts[bins - 1] + counts[bins])
    sums[bins - 1] = np.float32(sums[bins - 1] + sums[bins])
    square_sums[bins - 1] = np.float32(square_sums[bins - 1] + square_sums[bins])
    within_variance = np.float32(0.0)
    total_sum = np.float32(0.0)
    total_square_sum = np.float32(0.0)
    total_count = np.float32(0.0)
    variance = np.float32(0.0)
    for bin_index in range(bins):
        count = counts[bin_index]
        if count > np.float32(2.0):
            total_count = np.float32(total_count + count)
            total_sum = np.float32(total_sum + sums[bin_index])
            total_square_sum = np.float32(total_square_sum + square_sums[bin_index])
            variance = np.float32(
                np.float32(
                    square_sums[bin_index]
                    - np.float32(np.float32(sums[bin_index] * sums[bin_index]) / count)
                )
                / np.float32(count - np.float32(1.0))
            )
            within_variance = np.float32(within_variance + np.float32(variance * count))
    if total_count > zero:
        within_variance = np.float32(within_variance / total_count)
    if total_count > np.float32(1.0):
        variance = np.float32(
            np.float32(
                total_square_sum
                - np.float32(np.float32(total_sum * total_sum) / total_count)
            )
            / np.float32(total_count - np.float32(1.0))
        )
    if total_count <= np.float32(1.0) or variance <= zero:
        return np.float32(1.0)
    return np.float32(within_variance / variance)


@njit(cache=True, nogil=True, parallel=True)
def _weighted_correlation_ratio_ordered_parallel_kernel(
    reference: np.ndarray,
    moving: np.ndarray,
    moving_weight: np.ndarray,
    unit_moving_weight: bool,
    reference_first_x: np.ndarray,
    reference_last_x: np.ndarray,
    row_offsets: np.ndarray,
    active_x: np.ndarray,
    active_reference_weights: np.ndarray,
    active_reference_bins: np.ndarray,
    sampled_values: np.ndarray,
    sampled_weights: np.ndarray,
    voxel_mapping: np.ndarray,
    moving_voxel_sizes: np.ndarray,
    bins: int,
    smooth_size: np.float32,
) -> np.float32:
    """Compute interpolation in parallel, then reduce serially in strict FSL z/y/x order."""

    xb1 = reference.shape[0] - 1
    yb1 = reference.shape[1] - 1
    zb1 = reference.shape[2] - 1
    xb2 = np.float32(moving.shape[0]) - np.float32(1.0001)
    yb2 = np.float32(moving.shape[1]) - np.float32(1.0001)
    zb2 = np.float32(moving.shape[2]) - np.float32(1.0001)
    matrix = voxel_mapping.astype(np.float32)
    a11, a12, a13, a14 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    a21, a22, a23, a24 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    a31, a32, a33, a34 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    smoothx = np.float32(smooth_size / moving_voxel_sizes[0])
    smoothy = np.float32(smooth_size / moving_voxel_sizes[1])
    smoothz = np.float32(smooth_size / moving_voxel_sizes[2])
    zero = np.float32(0.0)

    row_count = (yb1 + 1) * (zb1 + 1)
    for row_index in prange(row_count):
        zindex = row_index // (yb1 + 1)
        yindex = row_index - zindex * (yb1 + 1)
        zfloat = np.float32(zindex)
        first_weighted_x = reference_first_x[yindex, zindex]
        last_weighted_x = reference_last_x[yindex, zindex]
        if last_weighted_x < first_weighted_x:
            continue
        yfloat = np.float32(yindex)
        o1 = np.float32(np.float32(yfloat * a12 + zfloat * a13) + a14)
        o2 = np.float32(np.float32(yfloat * a22 + zfloat * a23) + a24)
        o3 = np.float32(np.float32(yfloat * a32 + zfloat * a33) + a34)
        xmin, xmax = _find_range_x(o1, o2, o3, a11, a21, a31, xb1, xb2, yb2, zb2)
        o1 = np.float32(o1 + np.float32(xmin) * a11)
        o2 = np.float32(o2 + np.float32(xmin) * a21)
        o3 = np.float32(o3 + np.float32(xmin) * a31)
        first_x = max(xmin, first_weighted_x)
        last_x = min(xmax, last_weighted_x)
        previous_x = xmin
        for position in range(row_offsets[row_index], row_offsets[row_index + 1]):
            xindex = active_x[position]
            sampled_values[position] = zero
            sampled_weights[position] = zero
            if xindex < first_x:
                continue
            if xindex > last_x:
                continue
            for _ in range(previous_x, xindex):
                o1 = np.float32(o1 + a11)
                o2 = np.float32(o2 + a21)
                o3 = np.float32(o3 + a31)
            previous_x = xindex
            ix, iy, iz = int(o1), int(o2), int(o3)
            interpolation_valid = (
                0 <= ix < moving.shape[0] - 1
                and 0 <= iy < moving.shape[1] - 1
                and 0 <= iz < moving.shape[2] - 1
            )
            if (xindex != xmin and xindex != xmax) or interpolation_valid:
                value = _trilinear(moving, o1, o2, o3)
                moving_weight_value = (
                    np.float32(1.0)
                    if unit_moving_weight
                    else _trilinear(moving_weight, o1, o2, o3)
                )
                weight = np.float32(
                    moving_weight_value * active_reference_weights[position]
                )
                if o1 < smoothx:
                    weight = np.float32(weight * np.float32(o1 / smoothx))
                elif np.float32(xb2 - o1) < smoothx:
                    weight = np.float32(weight * np.float32(np.float32(xb2 - o1) / smoothx))
                if o2 < smoothy:
                    weight = np.float32(weight * np.float32(o2 / smoothy))
                elif np.float32(yb2 - o2) < smoothy:
                    weight = np.float32(weight * np.float32(np.float32(yb2 - o2) / smoothy))
                if o3 < smoothz:
                    weight = np.float32(weight * np.float32(o3 / smoothz))
                elif np.float32(zb2 - o3) < smoothz:
                    weight = np.float32(weight * np.float32(np.float32(zb2 - o3) / smoothz))
                if weight < zero:
                    weight = zero
                sampled_values[position] = value
                sampled_weights[position] = weight

    counts = np.zeros(bins + 1, dtype=np.float32)
    sums = np.zeros(bins + 1, dtype=np.float32)
    square_sums = np.zeros(bins + 1, dtype=np.float32)
    for position in range(active_x.size):
        weight = sampled_weights[position]
        value = sampled_values[position]
        bin_index = active_reference_bins[position]
        counts[bin_index] = np.float32(counts[bin_index] + weight)
        sums[bin_index] = np.float32(sums[bin_index] + np.float32(weight * value))
        square_sums[bin_index] = np.float32(
            square_sums[bin_index]
            + np.float32(np.float32(weight * value) * value)
        )

    counts[bins - 1] = np.float32(counts[bins - 1] + counts[bins])
    sums[bins - 1] = np.float32(sums[bins - 1] + sums[bins])
    square_sums[bins - 1] = np.float32(square_sums[bins - 1] + square_sums[bins])
    within_variance = np.float32(0.0)
    total_sum = np.float32(0.0)
    total_square_sum = np.float32(0.0)
    total_count = np.float32(0.0)
    variance = np.float32(0.0)
    for bin_index in range(bins):
        count = counts[bin_index]
        if count > np.float32(2.0):
            total_count = np.float32(total_count + count)
            total_sum = np.float32(total_sum + sums[bin_index])
            total_square_sum = np.float32(total_square_sum + square_sums[bin_index])
            variance = np.float32(
                np.float32(
                    square_sums[bin_index]
                    - np.float32(np.float32(sums[bin_index] * sums[bin_index]) / count)
                )
                / np.float32(count - np.float32(1.0))
            )
            within_variance = np.float32(within_variance + np.float32(variance * count))
    if total_count > zero:
        within_variance = np.float32(within_variance / total_count)
    if total_count > np.float32(1.0):
        variance = np.float32(
            np.float32(
                total_square_sum
                - np.float32(np.float32(total_sum * total_sum) / total_count)
            )
            / np.float32(total_count - np.float32(1.0))
        )
    if total_count <= np.float32(1.0) or variance <= zero:
        return np.float32(1.0)
    return np.float32(within_variance / variance)


@dataclass(frozen=True)
class FlirtWeightedMutualInformation:
    """Prepared FSL 6.0.4 fully weighted mutual-information evaluator."""

    reference: np.ndarray
    moving: np.ndarray
    reference_weight: np.ndarray
    moving_weight: np.ndarray
    reference_sampling: np.ndarray
    moving_sampling: np.ndarray
    bins: int = 256
    smooth_size: float = 1.0
    fuzzy_fraction: float = 0.5

    def __post_init__(self) -> None:
        reference = _float_volume(self.reference, "reference")
        moving = _float_volume(self.moving, "moving")
        reference_weight = _float_volume(self.reference_weight, "reference_weight")
        moving_weight = _float_volume(self.moving_weight, "moving_weight")
        if reference_weight.shape != reference.shape:
            raise ValueError("reference_weight must match reference")
        if moving_weight.shape != moving.shape:
            raise ValueError("moving_weight must match moving")
        if not isinstance(self.bins, (int, np.integer)) or self.bins < 2:
            raise ValueError("bins must be an integer of at least two")
        if not np.isfinite(self.smooth_size) or self.smooth_size <= 0:
            raise ValueError("smooth_size must be positive and finite")
        if not np.isfinite(self.fuzzy_fraction) or not 0 < self.fuzzy_fraction <= 0.5:
            raise ValueError("fuzzy_fraction must be in (0, 0.5]")
        reference_sampling = _finite_4x4(self.reference_sampling, "reference_sampling")
        moving_sampling = _finite_4x4(self.moving_sampling, "moving_sampling")
        moving_sizes = np.linalg.norm(moving_sampling[:3, :3], axis=0).astype(np.float32)
        reference_min = np.float32(reference.min())
        reference_max = np.float32(reference.max())
        if reference_max == reference_min:
            reference_max = np.float32(reference_max + np.float32(1.0))
        scale = np.float32(np.float32(self.bins) / np.float32(reference_max - reference_min))
        offset = np.float32(-reference_min * scale)
        reference_bins = np.asarray(reference * scale + offset, dtype=np.int32)
        np.clip(reference_bins, 0, self.bins - 1, out=reference_bins)
        moving_min = np.float32(moving.min())
        moving_max = np.float32(moving.max())
        if moving_max == moving_min:
            moving_max = np.float32(moving_max + np.float32(1.0))
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "moving", moving)
        object.__setattr__(self, "reference_weight", reference_weight)
        object.__setattr__(self, "moving_weight", moving_weight)
        object.__setattr__(self, "reference_sampling", reference_sampling)
        object.__setattr__(self, "moving_sampling", moving_sampling)
        object.__setattr__(self, "_inverse_moving_sampling", np.linalg.inv(moving_sampling))
        object.__setattr__(self, "_reference_bins", np.asfortranarray(reference_bins))
        object.__setattr__(self, "_moving_min", moving_min)
        object.__setattr__(self, "_moving_max", moving_max)
        object.__setattr__(self, "_moving_voxel_sizes", moving_sizes)
        first_x, last_x = _weight_x_ranges(reference_weight)
        object.__setattr__(self, "_reference_first_x", first_x)
        object.__setattr__(self, "_reference_last_x", last_x)
        object.__setattr__(
            self, "_unit_moving_weight", bool(np.all(moving_weight == np.float32(1.0)))
        )

    def __call__(self, moving_to_reference: np.ndarray) -> float:
        """Return FSL's minimized cost for one scaled-mm affine matrix."""

        transform = _finite_4x4(moving_to_reference, "moving_to_reference")
        mapping = self._inverse_moving_sampling @ np.linalg.inv(transform) @ self.reference_sampling
        return float(
            _weighted_mi_kernel(
                self.reference,
                self.moving,
                self.reference_weight,
                self.moving_weight,
                self._unit_moving_weight,
                self._reference_first_x,
                self._reference_last_x,
                self._reference_bins,
                mapping,
                self._moving_min,
                self._moving_max,
                self._moving_voxel_sizes,
                int(self.bins),
                np.float32(self.smooth_size),
                np.float32(self.fuzzy_fraction),
            )
        )

    def evaluate_many(self, transforms: np.ndarray, workers: int = 8) -> np.ndarray:
        """Evaluate candidate matrices in input order using independent workers.

        A single candidate owns its histogram and retains FSL's serial reduction
        order. Parallelism is only across candidates, which does not alter a cost
        value or the schedule's candidate ordering.
        """

        matrices = np.asarray(transforms, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] == 0:
            raise ValueError("transforms must have shape (n, 4, 4) with n greater than zero")
        if not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not np.all(np.isfinite(matrices)):
            raise ValueError("transform must be a finite 4x4 matrix")
        determinants = np.linalg.det(matrices[:, :3, :3])
        if np.any(np.abs(determinants) < 1e-12):
            raise ValueError("transform must be invertible")
        if len(matrices) == 1:
            return np.asarray([self(matrices[0])], dtype=np.float64)
        # Batched inversion and left/right multiplication only coalesce Python calls;
        # the operation order within each 4x4 matrix remains unchanged.
        mappings = (
            self._inverse_moving_sampling
            @ np.linalg.inv(matrices)
            @ self.reference_sampling
        )
        set_available_numba_threads(min(int(workers), len(matrices)))
        return _weighted_mi_many_kernel(
            self.reference,
            self.moving,
            self.reference_weight,
            self.moving_weight,
            self._unit_moving_weight,
            self._reference_first_x,
            self._reference_last_x,
            self._reference_bins,
            mappings,
            self._moving_min,
            self._moving_max,
            self._moving_voxel_sizes,
            int(self.bins),
            np.float32(self.smooth_size),
            np.float32(self.fuzzy_fraction),
        ).astype(np.float64)


@dataclass(frozen=True)
class FlirtWeightedCorrelationRatio:
    """Prepared FSL 6.0.4 fully weighted correlation-ratio evaluator."""

    reference: np.ndarray
    moving: np.ndarray
    reference_weight: np.ndarray
    moving_weight: np.ndarray
    reference_sampling: np.ndarray
    moving_sampling: np.ndarray
    bins: int = 256
    smooth_size: float = 1.0

    def __post_init__(self) -> None:
        reference = _float_volume(self.reference, "reference")
        moving = _float_volume(self.moving, "moving")
        reference_weight = _float_volume(self.reference_weight, "reference_weight")
        moving_weight = _float_volume(self.moving_weight, "moving_weight")
        if reference_weight.shape != reference.shape:
            raise ValueError("reference_weight must match reference")
        if moving_weight.shape != moving.shape:
            raise ValueError("moving_weight must match moving")
        if not isinstance(self.bins, (int, np.integer)) or self.bins < 2:
            raise ValueError("bins must be an integer of at least two")
        if not np.isfinite(self.smooth_size) or self.smooth_size <= 0:
            raise ValueError("smooth_size must be positive and finite")
        reference_sampling = _finite_4x4(self.reference_sampling, "reference_sampling")
        moving_sampling = _finite_4x4(self.moving_sampling, "moving_sampling")
        moving_sizes = np.linalg.norm(moving_sampling[:3, :3], axis=0).astype(np.float32)
        reference_min = np.float32(reference.min())
        reference_max = np.float32(reference.max())
        if reference_max == reference_min:
            reference_max = np.float32(reference_max + np.float32(1.0))
        scale = np.float32(np.float32(self.bins) / np.float32(reference_max - reference_min))
        offset = np.float32(-reference_min * scale)
        reference_bins = np.asarray(reference * scale + offset, dtype=np.int32)
        np.clip(reference_bins, 0, self.bins - 1, out=reference_bins)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "moving", moving)
        object.__setattr__(self, "reference_weight", reference_weight)
        object.__setattr__(self, "moving_weight", moving_weight)
        object.__setattr__(self, "reference_sampling", reference_sampling)
        object.__setattr__(self, "moving_sampling", moving_sampling)
        object.__setattr__(self, "_inverse_moving_sampling", np.linalg.inv(moving_sampling))
        object.__setattr__(self, "_reference_bins", np.asfortranarray(reference_bins))
        object.__setattr__(self, "_moving_voxel_sizes", moving_sizes)
        first_x, last_x = _weight_x_ranges(reference_weight)
        object.__setattr__(self, "_reference_first_x", first_x)
        object.__setattr__(self, "_reference_last_x", last_x)
        row_offsets, active_x, active_weights, active_bins = _ordered_weight_workset(
            reference_weight, reference_bins
        )
        object.__setattr__(self, "_row_offsets", row_offsets)
        object.__setattr__(self, "_active_x", active_x)
        object.__setattr__(self, "_active_reference_weights", active_weights)
        object.__setattr__(self, "_active_reference_bins", active_bins)
        # Each calling thread reuses its own workspace to avoid repeatedly allocating
        # large arrays during the final optimization stage.
        object.__setattr__(self, "_parallel_workspace", local())
        object.__setattr__(
            self, "_unit_moving_weight", bool(np.all(moving_weight == np.float32(1.0)))
        )

    def __call__(self, moving_to_reference: np.ndarray) -> float:
        """Return FSL's minimized weighted correlation-ratio cost."""

        transform = _finite_4x4(moving_to_reference, "moving_to_reference")
        mapping = (
            self._inverse_moving_sampling
            @ np.linalg.inv(transform)
            @ self.reference_sampling
        )
        return float(
            _weighted_correlation_ratio_kernel(
                self.reference,
                self.moving,
                self.reference_weight,
                self.moving_weight,
                self._unit_moving_weight,
                self._reference_first_x,
                self._reference_last_x,
                self._reference_bins,
                mapping,
                self._moving_voxel_sizes,
                int(self.bins),
                np.float32(self.smooth_size),
            )
        )

    def evaluate_ordered_parallel(self, moving_to_reference: np.ndarray) -> float:
        """Parallelize interpolation while retaining FSL's serial reduction order."""

        transform = _finite_4x4(moving_to_reference, "moving_to_reference")
        mapping = (
            self._inverse_moving_sampling
            @ np.linalg.inv(transform)
            @ self.reference_sampling
        )
        workspace = getattr(self._parallel_workspace, "arrays", None)
        if workspace is None or workspace[0].size != self._active_x.size:
            workspace = (
                np.empty(self._active_x.size, dtype=np.float32),
                np.empty(self._active_x.size, dtype=np.float32),
            )
            self._parallel_workspace.arrays = workspace
        return float(
            _weighted_correlation_ratio_ordered_parallel_kernel(
                self.reference,
                self.moving,
                self.moving_weight,
                self._unit_moving_weight,
                self._reference_first_x,
                self._reference_last_x,
                self._row_offsets,
                self._active_x,
                self._active_reference_weights,
                self._active_reference_bins,
                workspace[0],
                workspace[1],
                mapping,
                self._moving_voxel_sizes,
                int(self.bins),
                np.float32(self.smooth_size),
            )
        )

    def evaluate_many(self, transforms: np.ndarray, workers: int = 8) -> np.ndarray:
        """Evaluate candidates in input order without changing per-cost reductions."""

        matrices = np.asarray(transforms, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] == 0:
            raise ValueError("transforms must have shape (n, 4, 4) with n greater than zero")
        if not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        validated = [_finite_4x4(matrix, "transform") for matrix in matrices]
        if len(validated) == 1:
            return np.asarray([self(validated[0])], dtype=np.float64)
        with ThreadPoolExecutor(max_workers=min(int(workers), len(validated))) as executor:
            results = list(executor.map(self, validated))
        return np.asarray(results, dtype=np.float64)
