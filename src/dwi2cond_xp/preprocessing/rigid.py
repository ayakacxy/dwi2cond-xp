"""FSL MCFLIRT-compatible six-degree-of-freedom b0 registration."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit
from scipy.ndimage import affine_transform, map_coordinates

from .image_ops import select_b0_indices
from .transforms import affine_matrix, decompose_affine


ProgressCallback = Callable[[int, int], None]


def _load_selected_volumes(
    image: nib.spatialimages.SpatialImage,
    data_file: str | Path,
    indices: np.ndarray,
) -> list[np.ndarray]:
    path = Path(data_file)
    if path.name.lower().endswith(".nii.gz"):
        proxy = image.dataobj
        volume_shape = tuple(int(value) for value in image.shape[:3])
        voxel_count = int(np.prod(volume_shape))
        storage_dtype = np.dtype(proxy.dtype)
        volume_bytes = voxel_count * storage_dtype.itemsize
        volumes = []
        next_volume = 0
        with gzip.open(path, "rb") as stream:
            stream.seek(int(proxy.offset))
            for index_value in indices:
                index = int(index_value)
                stream.seek((index - next_volume) * volume_bytes, 1)
                raw = stream.read(volume_bytes)
                unscaled = np.frombuffer(
                    raw, dtype=storage_dtype, count=voxel_count
                ).reshape(volume_shape, order=proxy.order)
                scaled = nib.volumeutils.apply_read_scaling(
                    unscaled, proxy.slope, proxy.inter
                )
                volumes.append(np.asarray(scaled, dtype=np.float32))
                next_volume = index + 1
        return volumes
    return [
        np.asarray(image.dataobj[..., int(index)], dtype=np.float32)
        for index in indices
    ]


@dataclass(frozen=True)
class RigidRegistrationResult:
    """One moving-to-reference rigid registration result."""

    parameters: np.ndarray
    scaled_mm_transform: np.ndarray
    world_transform: np.ndarray
    cost: float
    evaluations: int
    success: bool
    message: str


def rigid_world_matrix(parameters: np.ndarray, center_world: np.ndarray) -> np.ndarray:
    """Compose the FSL Euler rigid transform around a physical center.

    Parameters contain ``rx, ry, rz`` in radians and ``tx, ty, tz`` in
    millimetres. FSL composes the rotations as ``Rx @ Ry @ Rz``.
    """

    values = np.asarray(parameters, dtype=np.float64)
    center = np.asarray(center_world, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Rigid parameters must be a finite six-element vector")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("The rotation center must be a finite three-element vector")
    rx, ry, rz, tx, ty, tz = values
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    rotation_y = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    rotation_z = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    rotation = rotation_x @ rotation_y @ rotation_z
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center + np.array([tx, ty, tz])
    return transform


def _rigid_parameters(transform: np.ndarray, center: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    cy = np.sqrt(rotation[0, 0] ** 2 + rotation[0, 1] ** 2)
    if abs(cy) < 1e-8:
        rx = np.arctan2(-rotation[2, 1], rotation[1, 1])
        ry = np.arctan2(-rotation[0, 2], 0.0)
        rz = 0.0
    else:
        rx = np.arctan2(rotation[1, 2] / cy, rotation[2, 2] / cy)
        ry = np.arctan2(-rotation[0, 2], cy)
        rz = np.arctan2(rotation[0, 1] / cy, rotation[0, 0] / cy)
    translation = rotation @ center + transform[:3, 3] - center
    return np.array([rx, ry, rz, *translation], dtype=np.float64)


def resample_rigid(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray,
    world_transform: np.ndarray,
    *,
    order: int = 1,
    mode: str = "constant",
) -> np.ndarray:
    """Resample a 3D moving image onto a reference NIfTI grid."""

    values = np.asarray(moving, dtype=np.float32)
    target_shape = tuple(int(value) for value in reference_shape)
    source_affine = np.asarray(moving_affine, dtype=np.float64)
    target_affine = np.asarray(reference_affine, dtype=np.float64)
    transform = np.asarray(world_transform, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("The moving image must be three-dimensional")
    if len(target_shape) != 3 or any(value <= 0 for value in target_shape):
        raise ValueError("The reference shape must contain three positive dimensions")
    if (
        source_affine.shape != (4, 4)
        or target_affine.shape != (4, 4)
        or transform.shape != (4, 4)
        or not np.all(np.isfinite(source_affine))
        or not np.all(np.isfinite(target_affine))
        or not np.all(np.isfinite(transform))
    ):
        raise ValueError("All spatial transforms must be finite 4x4 matrices")
    if order not in (0, 1, 3):
        raise ValueError("Interpolation order must be 0, 1, or 3")
    if mode not in ("constant", "nearest"):
        raise ValueError("Interpolation mode must be constant or nearest")
    output_to_input = (
        np.linalg.inv(source_affine) @ np.linalg.inv(transform) @ target_affine
    )
    return affine_transform(
        values,
        output_to_input[:3, :3],
        offset=output_to_input[:3, 3],
        output_shape=target_shape,
        order=order,
        mode=mode,
        cval=0.0,
        prefilter=order > 1,
    ).astype(np.float32, copy=False)


def normalized_correlation_cost(reference: np.ndarray, moving: np.ndarray) -> float:
    """Return ``1 - abs(r)`` for finite samples on an identical grid."""

    fixed = np.asarray(reference, dtype=np.float64)
    warped = np.asarray(moving, dtype=np.float64)
    if fixed.shape != warped.shape:
        raise ValueError("Reference and moving arrays must have identical shapes")
    valid = np.isfinite(fixed) & np.isfinite(warped)
    if np.count_nonzero(valid) < 3:
        return 1.0
    fixed_values = fixed[valid]
    moving_values = warped[valid]
    fixed_values -= np.mean(fixed_values)
    moving_values -= np.mean(moving_values)
    denominator = np.linalg.norm(fixed_values) * np.linalg.norm(moving_values)
    if denominator == 0:
        return 1.0
    correlation = float(np.dot(fixed_values, moving_values) / denominator)
    return 1.0 - abs(correlation)


def _isotropic_resample(
    values: np.ndarray, voxel_sizes: np.ndarray, spacing_mm: float
) -> np.ndarray:
    steps = spacing_mm / voxel_sizes
    output_shape = np.maximum(1, (np.asarray(values.shape) / steps).astype(int))
    coordinates = np.meshgrid(
        *(
            np.arange(size, dtype=np.float64) * step
            for size, step in zip(output_shape, steps)
        ),
        indexing="ij",
    )
    return map_coordinates(
        values.astype(np.float32, copy=False),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)


def _spacing_vector(spacing_mm: float | Sequence[float]) -> np.ndarray:
    """Normalize scalar or per-axis voxel spacing to a finite 3-vector."""

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 3)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_mm must contain three positive finite values")
    return spacing


def _intensity_center_scaled_mm(
    values: np.ndarray, spacing_mm: float | Sequence[float]
) -> np.ndarray:
    weights = values.astype(np.float64) - float(np.min(values))
    total = float(np.sum(weights))
    if abs(total) < 1e-5:
        return np.zeros(3, dtype=np.float64)
    spacing = _spacing_vector(spacing_mm)
    grid = np.indices(values.shape, dtype=np.float64)
    return np.array(
        [
            float(np.sum(grid[axis] * weights) / total) * spacing[axis]
            for axis in range(3)
        ]
    )


@njit(cache=True, nogil=True)
def _sample_smoothed_linear_kernel(
    moving: np.ndarray,
    inverse_scaled: np.ndarray,
    reference_spacing_mm: np.ndarray,
    moving_spacing_mm: np.ndarray,
    smooth_voxels: np.ndarray,
    upper: np.ndarray,
    full_weights: np.ndarray,
    full_moving: np.ndarray,
) -> int:
    """Fuse coordinate transforms, boundary weights, and trilinear sampling, releasing the GIL."""

    nx, ny, nz = full_weights.shape
    valid_count = 0
    for x in range(nx):
        x_mm = x * reference_spacing_mm[0]
        for y in range(ny):
            y_mm = y * reference_spacing_mm[1]
            for z in range(nz):
                z_mm = z * reference_spacing_mm[2]
                coordinate_x = (
                    inverse_scaled[0, 0] * x_mm
                    + inverse_scaled[0, 1] * y_mm
                    + inverse_scaled[0, 2] * z_mm
                    + inverse_scaled[0, 3]
                ) / moving_spacing_mm[0]
                coordinate_y = (
                    inverse_scaled[1, 0] * x_mm
                    + inverse_scaled[1, 1] * y_mm
                    + inverse_scaled[1, 2] * z_mm
                    + inverse_scaled[1, 3]
                ) / moving_spacing_mm[1]
                coordinate_z = (
                    inverse_scaled[2, 0] * x_mm
                    + inverse_scaled[2, 1] * y_mm
                    + inverse_scaled[2, 2] * z_mm
                    + inverse_scaled[2, 3]
                ) / moving_spacing_mm[2]
                if (
                    coordinate_x < 0.0
                    or coordinate_y < 0.0
                    or coordinate_z < 0.0
                    or coordinate_x > upper[0]
                    or coordinate_y > upper[1]
                    or coordinate_z > upper[2]
                ):
                    full_weights[x, y, z] = np.float32(0.0)
                    full_moving[x, y, z] = np.float32(0.0)
                    continue

                x0 = int(np.floor(coordinate_x))
                y0 = int(np.floor(coordinate_y))
                z0 = int(np.floor(coordinate_z))
                dx = coordinate_x - x0
                dy = coordinate_y - y0
                dz = coordinate_z - z0
                x1 = min(x0 + 1, moving.shape[0] - 1)
                y1 = min(y0 + 1, moving.shape[1] - 1)
                z1 = min(z0 + 1, moving.shape[2] - 1)
                c00 = moving[x0, y0, z0] * (1.0 - dx) + moving[x1, y0, z0] * dx
                c01 = moving[x0, y0, z1] * (1.0 - dx) + moving[x1, y0, z1] * dx
                c10 = moving[x0, y1, z0] * (1.0 - dx) + moving[x1, y1, z0] * dx
                c11 = moving[x0, y1, z1] * (1.0 - dx) + moving[x1, y1, z1] * dx
                c0 = c00 * (1.0 - dy) + c10 * dy
                c1 = c01 * (1.0 - dy) + c11 * dy
                full_moving[x, y, z] = np.float32(c0 * (1.0 - dz) + c1 * dz)

                weight = np.float32(1.0)
                coordinates = (coordinate_x, coordinate_y, coordinate_z)
                for axis in range(3):
                    coordinate = coordinates[axis]
                    if coordinate < smooth_voxels[axis]:
                        weight = np.float32(
                            weight * (coordinate / smooth_voxels[axis])
                        )
                    distance_to_high = upper[axis] - coordinate
                    if distance_to_high < smooth_voxels[axis]:
                        weight = np.float32(
                            weight * (distance_to_high / smooth_voxels[axis])
                        )
                full_weights[x, y, z] = max(weight, np.float32(0.0))
                valid_count += 1
    return valid_count


def _sample_smoothed_linear(
    moving: np.ndarray,
    inverse_scaled: np.ndarray,
    spacing_mm: float | Sequence[float],
    smooth_voxels: float | Sequence[float],
    upper: np.ndarray,
    full_weights: np.ndarray,
    full_moving: np.ndarray,
    moving_spacing_mm: float | Sequence[float] | None = None,
) -> int:
    """Sample one volume on a scalar or anisotropic physical grid."""

    spacing = _spacing_vector(spacing_mm)
    moving_spacing = (
        spacing
        if moving_spacing_mm is None
        else _spacing_vector(moving_spacing_mm)
    )
    smoothing = np.asarray(smooth_voxels, dtype=np.float64)
    if smoothing.ndim == 0:
        smoothing = np.repeat(smoothing, 3)
    if smoothing.shape != (3,) or not np.all(np.isfinite(smoothing)) or np.any(
        smoothing <= 0
    ):
        raise ValueError("smooth_voxels must contain three positive finite values")
    return _sample_smoothed_linear_kernel(
        moving,
        inverse_scaled,
        spacing,
        moving_spacing,
        smoothing,
        upper,
        full_weights,
        full_moving,
    )


@njit(cache=True, nogil=True)
def _build_normcorr_products(
    weights: np.ndarray,
    reference: np.ndarray,
    moving: np.ndarray,
    weighted_fixed: np.ndarray,
    weighted_moving: np.ndarray,
    fixed_square: np.ndarray,
    moving_square: np.ndarray,
    cross: np.ndarray,
) -> None:
    """Build five float32 product arrays in one pass, preserving per-array rounding barriers."""

    nx, ny, nz = weights.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                weighted_fixed[x, y, z] = np.float32(
                    weights[x, y, z] * reference[x, y, z]
                )
                weighted_moving[x, y, z] = np.float32(
                    weights[x, y, z] * moving[x, y, z]
                )
                fixed_square[x, y, z] = np.float32(
                    weighted_fixed[x, y, z] * reference[x, y, z]
                )
                moving_square[x, y, z] = np.float32(
                    weighted_moving[x, y, z] * moving[x, y, z]
                )
                cross[x, y, z] = np.float32(
                    weighted_fixed[x, y, z] * moving[x, y, z]
                )


@njit(cache=True, nogil=True)
def _numpy_pairwise_leaf_float32(
    values: np.ndarray, start: int, count: int
) -> np.float32:
    """Evaluate a leaf of at most 128 elements in NumPy's pairwise sum."""

    if count < 8:
        result = np.float32(-0.0)
        for index in range(count):
            result = np.float32(result + values[start + index])
        return result
    partial = np.empty(8, dtype=np.float32)
    for index in range(8):
        partial[index] = values[start + index]
    stop = count - (count % 8)
    index = 8
    while index < stop:
        for offset in range(8):
            partial[offset] = np.float32(
                partial[offset] + values[start + index + offset]
            )
        index += 8
    left = np.float32(
        np.float32(partial[0] + partial[1])
        + np.float32(partial[2] + partial[3])
    )
    right = np.float32(
        np.float32(partial[4] + partial[5])
        + np.float32(partial[6] + partial[7])
    )
    result = np.float32(left + right)
    while index < count:
        result = np.float32(result + values[start + index])
        index += 1
    return result


@njit(cache=True, nogil=True)
def _numpy_pairwise_sum_float32(values: np.ndarray, start: int, count: int) -> np.float32:
    """Reproduce NumPy's float32 pairwise sum with PW_BLOCKSIZE=128 on a contiguous axis."""

    if count <= 128:
        return _numpy_pairwise_leaf_float32(values, start, count)
    starts = np.empty(64, dtype=np.int64)
    counts = np.empty(64, dtype=np.int64)
    states = np.zeros(64, dtype=np.int8)
    left_results = np.empty(64, dtype=np.float32)
    depth = 0
    starts[0] = start
    counts[0] = count
    result = np.float32(-0.0)
    while depth >= 0:
        current_start = starts[depth]
        current_count = counts[depth]
        if current_count <= 128:
            result = _numpy_pairwise_leaf_float32(
                values, current_start, current_count
            )
            depth -= 1
        elif states[depth] == 0:
            half = current_count // 2
            half -= half % 8
            states[depth] = 1
            depth += 1
            starts[depth] = current_start
            counts[depth] = half
            states[depth] = 0
            continue
        elif states[depth] == 1:
            left_results[depth] = result
            half = current_count // 2
            half -= half % 8
            states[depth] = 2
            depth += 1
            starts[depth] = current_start + half
            counts[depth] = current_count - half
            states[depth] = 0
            continue
        else:
            result = np.float32(left_results[depth] + result)
            depth -= 1
        if depth < 0:
            return result


@njit(cache=True, nogil=True)
def _fsl_six_sums(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    fifth: np.ndarray,
    sixth: np.ndarray,
) -> tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
    """Complete six independent x/y/z float32 grouped reductions in one pass."""

    nx, ny, nz = first.shape
    z_values = np.empty((6, nz), dtype=np.float32)
    for z in range(nz):
        z0 = z1 = z2 = z3 = z4 = z5 = np.float32(0.0)
        for y in range(ny):
            x0 = x1 = x2 = x3 = x4 = x5 = np.float32(0.0)
            for x in range(nx):
                x0 = np.float32(x0 + first[x, y, z])
                x1 = np.float32(x1 + second[x, y, z])
                x2 = np.float32(x2 + third[x, y, z])
                x3 = np.float32(x3 + fourth[x, y, z])
                x4 = np.float32(x4 + fifth[x, y, z])
                x5 = np.float32(x5 + sixth[x, y, z])
            z0 = np.float32(z0 + x0)
            z1 = np.float32(z1 + x1)
            z2 = np.float32(z2 + x2)
            z3 = np.float32(z3 + x3)
            z4 = np.float32(z4 + x4)
            z5 = np.float32(z5 + x5)
        z_values[0, z] = z0
        z_values[1, z] = z1
        z_values[2, z] = z2
        z_values[3, z] = z3
        z_values[4, z] = z4
        z_values[5, z] = z5
    totals = np.empty(6, dtype=np.float32)
    for index in range(6):
        totals[index] = _numpy_pairwise_sum_float32(z_values[index], 0, nz)
    return (
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals[4],
        totals[5],
    )


class _SmoothedNormCorr:
    def __init__(
        self,
        reference: np.ndarray,
        moving: np.ndarray,
        spacing_mm: float | Sequence[float],
        center: np.ndarray,
        degrees_of_freedom: int = 6,
        smooth_mm: float = 1.0,
        moving_spacing_mm: float | Sequence[float] | None = None,
    ) -> None:
        self.reference = reference.astype(np.float32, copy=False)
        self.moving = moving.astype(np.float32, copy=False)
        self.spacing_mm = _spacing_vector(spacing_mm)
        self.moving_spacing_mm = (
            self.spacing_mm
            if moving_spacing_mm is None
            else _spacing_vector(moving_spacing_mm)
        )
        self.center = center
        self.degrees_of_freedom = degrees_of_freedom
        self.smooth_voxels = smooth_mm / self.moving_spacing_mm
        self.upper = np.asarray(self.moving.shape, dtype=np.float64) - 1.0001
        self.full_weights = np.zeros(self.reference.shape, dtype=np.float32)
        self.full_moving = np.zeros(self.reference.shape, dtype=np.float32)
        self.weighted_fixed = np.empty(self.reference.shape, dtype=np.float32)
        self.weighted_moving = np.empty(self.reference.shape, dtype=np.float32)
        self.fixed_square = np.empty(self.reference.shape, dtype=np.float32)
        self.moving_square = np.empty(self.reference.shape, dtype=np.float32)
        self.cross = np.empty(self.reference.shape, dtype=np.float32)
        self.evaluations = 0

    def __call__(self, parameters: np.ndarray) -> float:
        self.evaluations += 1
        transform = (
            rigid_world_matrix(parameters, self.center)
            if self.degrees_of_freedom == 6
            else affine_matrix(parameters, self.center)
        )
        inverse_scaled = np.linalg.inv(transform)
        valid_count = _sample_smoothed_linear(
            self.moving,
            inverse_scaled,
            self.spacing_mm,
            self.smooth_voxels,
            self.upper,
            self.full_weights,
            self.full_moving,
            self.moving_spacing_mm,
        )
        if valid_count < 3:
            return 1.0
        _build_normcorr_products(
            self.full_weights,
            self.reference,
            self.full_moving,
            self.weighted_fixed,
            self.weighted_moving,
            self.fixed_square,
            self.moving_square,
            self.cross,
        )
        (
            count_value,
            sum_fixed_value,
            sum_moving_value,
            fixed_square_value,
            moving_square_value,
            cross_value,
        ) = _fsl_six_sums(
            self.full_weights,
            self.weighted_fixed,
            self.weighted_moving,
            self.fixed_square,
            self.moving_square,
            self.cross,
        )
        count = float(count_value)
        if count <= 2.0:
            return 1.0
        sum_fixed = float(sum_fixed_value)
        sum_moving = float(sum_moving_value)
        variance_fixed = float(fixed_square_value) / (count - 1.0)
        variance_fixed -= sum_fixed * sum_fixed / (count * count)
        variance_moving = float(moving_square_value) / (count - 1.0)
        variance_moving -= sum_moving * sum_moving / (count * count)
        covariance = float(cross_value) / (count - 1.0)
        covariance -= sum_fixed * sum_moving / (count * count)
        if variance_fixed <= 0.0 or variance_moving <= 0.0:
            return 1.0
        correlation = covariance / np.sqrt(variance_fixed * variance_moving)
        return 1.0 - abs(float(correlation))


def _quadratic_minimum(
    x1: float, xm: float, x2: float, y1: float, ym: float, y2: float
) -> float | None:
    a = (xm - x2) * (ym - y1) - (xm - x1) * (ym - y2)
    b = -(xm * xm - x2 * x2) * (ym - y1) + (xm * xm - x1 * x1) * (ym - y2)
    determinant = (xm - x2) * (x2 - x1) * (x1 - xm)
    if (abs(determinant) > 1e-15 and a / determinant < 0) or abs(a) <= 1e-15:
        return None
    return -b / (2.0 * a)


def _inside_candidate(
    x1: float, xm: float, x2: float, y1: float, ym: float, y2: float
) -> float:
    candidate = _quadratic_minimum(x1, xm, x2, y1, ym, y2)
    if candidate is not None and min(x1, x2) <= candidate <= max(x1, x2):
        return candidate
    return _extrapolated_point(x1, xm, x2)


def _extrapolated_point(x1: float, xm: float, x2: float) -> float:
    ratio = 0.3819660
    endpoint = x2 if abs(x2 - xm) > abs(x1 - xm) else x1
    return ratio * endpoint + (1.0 - ratio) * xm


def _line_search(
    point: np.ndarray,
    axis: int,
    tolerances: np.ndarray,
    objective: Callable[[np.ndarray], float],
    initial_cost: float,
) -> tuple[np.ndarray, float]:
    unit_tolerance = float(tolerances[axis])

    def evaluate(distance: float) -> float:
        candidate = point.copy()
        candidate[axis] += distance
        return objective(candidate)

    xm = 0.0
    x1 = 10.0 * unit_tolerance
    ym = objective(point) if initial_cost == 0.0 else initial_cost
    y1 = evaluate(x1)
    if y1 < ym:
        x1, xm = xm, x1
        y1, ym = ym, y1
    direction = -1.0 if xm < x1 else 1.0
    x2 = xm + 1.6 * (xm - x1)
    y2 = evaluate(x2)
    bracket_iterations = 0
    while ym > y2:
        bracket_iterations += 1
        if bracket_iterations > 100:
            raise RuntimeError(
                "MCFLIRT-compatible line search failed to bracket a minimum"
            )
        maximum = xm + 3.2 * (x2 - xm)
        candidate = _quadratic_minimum(x1, xm, x2, y1, ym, y2)
        if (
            candidate is None
            or (candidate - x1) * direction < 0
            or (candidate - maximum) * direction > 0
        ):
            candidate = xm + 1.6 * (x2 - xm)
        value = evaluate(candidate)
        if (candidate - xm) * (candidate - x1) < 0:
            if value < ym:
                x2, y2 = xm, ym
                xm, ym = candidate, value
                break
            x1, y1 = candidate, value
        elif value > ym:
            x2, y2 = candidate, value
            break
        elif (candidate - x2) * direction < 0:
            x1, y1 = xm, ym
            xm, ym = candidate, value
        else:
            x1, y1 = xm, ym
            xm, ym = x2, y2
            x2, y2 = candidate, value

    minimum_distance = 0.1 * unit_tolerance
    iteration = 0
    while iteration < 100 and abs((x2 - x1) / unit_tolerance) > 1.0:
        iteration += 1
        candidate = _inside_candidate(x1, xm, x2, y1, ym, y2)
        interval_direction = 1.0 if x2 >= x1 else -1.0
        if abs(candidate - x1) < minimum_distance:
            candidate = x1 + interval_direction * minimum_distance
        if abs(candidate - x2) < minimum_distance:
            candidate = x2 - interval_direction * minimum_distance
        if abs(candidate - xm) < minimum_distance:
            candidate = _extrapolated_point(x1, xm, x2)
        if abs(xm - x1) < 0.4 * unit_tolerance:
            candidate = xm + interval_direction * 0.5 * unit_tolerance
        if abs(xm - x2) < 0.4 * unit_tolerance:
            candidate = xm - interval_direction * 0.5 * unit_tolerance
        value = evaluate(candidate)
        if (candidate - xm) * (x2 - xm) > 0:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        if value < ym:
            x2, y2 = xm, ym
            xm, ym = candidate, value
        else:
            x1, y1 = candidate, value
    updated = point.copy()
    updated[axis] += xm
    return updated, ym


def _optimize_one_stage(
    reference: np.ndarray,
    moving: np.ndarray,
    spacing_mm: float | Sequence[float],
    initial_transform: np.ndarray,
    tolerance_multiplier: float,
    center: np.ndarray | None = None,
    degrees_of_freedom: int = 6,
    parameter_axes: Sequence[int] | None = None,
    smooth_mm: float = 1.0,
    moving_spacing_mm: float | Sequence[float] | None = None,
) -> tuple[np.ndarray, float, int]:
    moving_spacing = spacing_mm if moving_spacing_mm is None else moving_spacing_mm
    center = (
        _intensity_center_scaled_mm(moving, moving_spacing)
        if center is None
        else center
    )
    parameters = (
        _rigid_parameters(initial_transform, center)
        if degrees_of_freedom == 6
        else decompose_affine(initial_transform, center)[:degrees_of_freedom]
    )
    objective = _SmoothedNormCorr(
        reference,
        moving,
        spacing_mm,
        center,
        degrees_of_freedom,
        smooth_mm=smooth_mm,
        moving_spacing_mm=moving_spacing,
    )
    tolerances = np.array(
        [0.005] * 3 + [0.2] * 3 + [0.002] * 3 + [0.001] * 3,
        dtype=np.float64,
    )[:degrees_of_freedom] * tolerance_multiplier
    cost = 0.0
    axes = (
        tuple(range(degrees_of_freedom))
        if parameter_axes is None
        else tuple(int(axis) for axis in parameter_axes)
    )
    if not axes or any(axis < 0 or axis >= degrees_of_freedom for axis in axes):
        raise ValueError("parameter_axes must select valid transform parameters")
    for axis in axes:
        parameters, cost = _line_search(parameters, axis, tolerances, objective, cost)
    transform = (
        rigid_world_matrix(parameters, center)
        if degrees_of_freedom == 6
        else affine_matrix(parameters, center)
    )
    return transform, cost, objective.evaluations


def _scaled_to_world(transform: np.ndarray, affine: np.ndarray) -> np.ndarray:
    sampling = np.diag([*nib.affines.voxel_sizes(affine), 1.0])
    return (
        affine @ np.linalg.inv(sampling) @ transform @ sampling @ np.linalg.inv(affine)
    )


def estimate_rigid_transform(
    reference: np.ndarray,
    moving: np.ndarray,
    affine: np.ndarray,
    *,
    initial_parameters: np.ndarray | None = None,
    stages_mm: Sequence[float] = (8.0, 4.0, 4.0),
    max_evaluations: int = 240,
) -> RigidRegistrationResult:
    """Estimate the MCFLIRT three-stage 6-DOF transform for one volume."""

    fixed = np.asarray(reference, dtype=np.float32)
    source = np.asarray(moving, dtype=np.float32)
    spatial_affine = np.asarray(affine, dtype=np.float64)
    if fixed.ndim != 3 or fixed.shape != source.shape:
        raise ValueError(
            "Rigid registration requires matching three-dimensional images"
        )
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(source)):
        raise ValueError("Rigid registration images must contain only finite values")
    if spatial_affine.shape != (4, 4) or not np.all(np.isfinite(spatial_affine)):
        raise ValueError("The image affine must be a finite 4x4 matrix")
    if not stages_mm or any(value <= 0 for value in stages_mm):
        raise ValueError("Registration stages must contain positive spacings")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    initial = (
        np.zeros(6, dtype=np.float64)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=np.float64)
    )
    if initial.shape != (6,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_parameters must be a finite six-element vector")

    voxel_sizes = nib.affines.voxel_sizes(spatial_affine)
    transform = rigid_world_matrix(initial, np.zeros(3))
    total_evaluations = 0
    cost = 1.0
    multipliers = (0.8, 0.8, 0.1)
    for stage_index, spacing in enumerate(stages_mm):
        fixed_coarse = _isotropic_resample(fixed, voxel_sizes, spacing)
        moving_coarse = _isotropic_resample(source, voxel_sizes, spacing)
        transform, cost, evaluations = _optimize_one_stage(
            fixed_coarse,
            moving_coarse,
            spacing,
            transform,
            multipliers[min(stage_index, 2)],
        )
        total_evaluations += evaluations
        if total_evaluations > max_evaluations:
            return RigidRegistrationResult(
                parameters=np.zeros(6),
                scaled_mm_transform=transform,
                world_transform=_scaled_to_world(transform, spatial_affine),
                cost=cost,
                evaluations=total_evaluations,
                success=False,
                message="Maximum number of function evaluations was exceeded",
            )
    original_center = _intensity_center_scaled_mm(source, float(voxel_sizes[0]))
    return RigidRegistrationResult(
        parameters=_rigid_parameters(transform, original_center),
        scaled_mm_transform=transform,
        world_transform=_scaled_to_world(transform, spatial_affine),
        cost=cost,
        evaluations=total_evaluations,
        success=True,
        message="MCFLIRT-compatible stages completed",
    )


def write_aligned_b0_mean(
    data_file: str | Path,
    bvals_file: str | Path,
    output_file: str | Path,
    *,
    b0_threshold: float = 50.0,
    stages_mm: Sequence[float] = (8.0, 4.0, 4.0),
    max_evaluations: int = 240,
    workers: int = 1,
    progress: ProgressCallback | None = None,
    qa_file: str | Path | None = None,
) -> Path:
    """Register b0 volumes to the middle b0 and write their float32 mean."""

    image = nib.load(str(data_file), mmap=True)
    if len(image.shape) != 4:
        raise ValueError("DWI data must be a four-dimensional NIfTI")
    bvals = np.asarray(np.loadtxt(bvals_file), dtype=np.float64).reshape(-1)
    if bvals.size != image.shape[3]:
        raise ValueError("The DWI fourth axis does not match bvals")
    if workers <= 0:
        raise ValueError("The worker count must be a positive integer")
    b0_indices = select_b0_indices(bvals, threshold=b0_threshold)
    reference_position = b0_indices.size // 2
    reference_index = int(b0_indices[reference_position])
    volumes = _load_selected_volumes(image, data_file, b0_indices)
    if any(not np.all(np.isfinite(volume)) for volume in volumes):
        raise ValueError("The b0 volumes contain NaN or Inf")

    reference = volumes[reference_position]
    matrices = [np.eye(4, dtype=np.float64) for _ in volumes]
    evaluations = [0 for _ in volumes]
    costs = [0.0 for _ in volumes]
    voxel_sizes = nib.affines.voxel_sizes(image.affine)
    multipliers = (0.8, 0.8, 0.1)
    progress_done = 0
    progress_total = len(stages_mm) * (len(volumes) - 1) + len(volumes)
    pyramid_cache: dict[float, list[np.ndarray]] = {}
    center_cache: dict[float, list[np.ndarray]] = {}
    for stage_index, spacing in enumerate(stages_mm):
        spacing_key = float(spacing)
        if spacing_key not in pyramid_cache:
            pyramid_cache[spacing_key] = [
                _isotropic_resample(volume, voxel_sizes, spacing) for volume in volumes
            ]
            center_cache[spacing_key] = [
                _intensity_center_scaled_mm(volume, spacing)
                for volume in pyramid_cache[spacing_key]
            ]
        coarse = pyramid_cache[spacing_key]
        stage_centers = center_cache[spacing_key]
        fixed_coarse = coarse[reference_position]
        stage_output = [matrix.copy() for matrix in matrices]
        order = list(range(reference_position + 1, len(volumes))) + list(
            range(reference_position - 1, -1, -1)
        )

        def optimize(position: int) -> tuple[int, np.ndarray, float, int]:
            transform, cost, count = _optimize_one_stage(
                fixed_coarse,
                coarse[position],
                spacing,
                matrices[position],
                multipliers[min(stage_index, 2)],
                stage_centers[position],
            )
            return position, transform, cost, count

        if stage_index == 0 or workers == 1:
            results = map(optimize, order)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            results = executor.map(optimize, order)
        try:
            for position, transform, cost, count in results:
                stage_output[position] = transform
                costs[position] = cost
                evaluations[position] += count
                progress_done += 1
                if progress is not None:
                    progress(progress_done, progress_total)
                if evaluations[position] > max_evaluations:
                    raise RuntimeError(
                        f"Rigid registration exceeded the evaluation limit for DWI volume "
                        f"{int(b0_indices[position])}"
                    )
                if stage_index == 0:
                    neighbour = position + (1 if position > reference_position else -1)
                    if 0 <= neighbour < len(volumes) - 1:
                        matrices[neighbour] = transform.copy()
        finally:
            if stage_index > 0 and workers > 1:
                executor.shutdown()
        matrices = stage_output

    aligned_sum = np.zeros(image.shape[:3], dtype=np.float64)
    records: list[dict[str, object]] = []
    final_positions = list(range(len(volumes)))

    def align_final(
        position: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        moving = volumes[position]
        transform = matrices[position]
        world_transform = _scaled_to_world(transform, image.affine)
        aligned = (
            reference
            if position == reference_position
            else resample_rigid(
                moving,
                image.affine,
                reference.shape,
                image.affine,
                world_transform,
                mode="nearest",
            )
        )
        center = _intensity_center_scaled_mm(moving, float(voxel_sizes[0]))
        return aligned, world_transform, center

    if workers == 1:
        final_results = map(align_final, final_positions)
    else:
        final_executor = ThreadPoolExecutor(max_workers=workers)
        final_results = final_executor.map(align_final, final_positions)
    try:
        ordered_results = zip(final_positions, final_results)
        for position, (aligned, world_transform, center) in ordered_results:
            volume_index = int(b0_indices[position])
            transform = matrices[position]
            aligned_sum += aligned
            records.append(
                {
                    "volume_index": volume_index,
                    "parameters_rad_mm": _rigid_parameters(transform, center).tolist(),
                    "scaled_mm_transform": transform.tolist(),
                    "world_transform": world_transform.tolist(),
                    "cost": costs[position],
                    "evaluations": evaluations[position],
                    "success": True,
                    "message": (
                        "reference volume"
                        if position == reference_position
                        else "MCFLIRT-compatible stages completed"
                    ),
                }
            )
            if progress is not None:
                progress_done += 1
                progress(progress_done, progress_total)
    finally:
        if workers > 1:
            final_executor.shutdown()

    mean = (aligned_sum / b0_indices.size).astype(np.float32)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    output_image = nib.Nifti1Image(mean, image.affine, header)
    output_image.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output_image.set_sform(image.get_sform(), int(image.header["sform_code"]))
    nib.save(output_image, str(output))

    qa_path = (
        output.with_name(f"{output.name.removesuffix('.nii.gz')}_qa.json")
        if qa_file is None
        else Path(qa_file)
    )
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "operation": "aligned_b0_mean",
                "registration": "mcflirt_6dof_compat46",
                "cost": "fsl_smoothed_normalized_correlation",
                "reference_policy": "middle_b0",
                "reference_volume_index": reference_index,
                "b0_threshold": b0_threshold,
                "b0_indices": b0_indices.tolist(),
                "stages_mm": list(stages_mm),
                "workers": workers,
                "interpolation": "trilinear",
                "input_shape": list(image.shape),
                "output_shape": list(mean.shape),
                "output_dtype": "float32",
                "volumes": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
