"""Deterministic affine and displacement-field image resampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit, prange
from scipy.ndimage import map_coordinates

from .transforms import invert_transform


_INTERPOLATION_ORDER = {"nearest": 0, "linear": 1, "spline": 3, "sinc": -1}


def _fsl_hanning_sinc_table() -> np.ndarray:
    positions = np.linspace(-3.0, 3.0, 1201, dtype=np.float32)
    sinc = np.empty(positions.shape, dtype=np.float32)
    near_zero = np.abs(positions) < np.float32(1e-7)
    sinc[near_zero] = 1.0 - np.abs(positions[near_zero])
    radians = np.float32(np.pi) * positions[~near_zero]
    sinc[~near_zero] = np.sin(radians) / radians
    window = np.float32(0.5) + np.float32(0.5) * np.cos(
        np.float32(np.pi) * positions / np.float32(3.0)
    )
    return sinc * window


_FSL_HANNING_SINC = _fsl_hanning_sinc_table()


def _fsl_sinc_kernel_values(distance: np.ndarray) -> np.ndarray:
    values = np.asarray(distance, dtype=np.float32)
    lookup = values / np.float32(3.0) * np.float32(600.0) + np.float32(601.0)
    lower = np.floor(lookup).astype(np.int64)
    fraction = lookup - lower
    valid = (lower >= 1) & (lower <= 1200)
    result = np.zeros(values.shape, dtype=np.float32)
    result[valid] = (
        _FSL_HANNING_SINC[lower[valid] - 1] * (1.0 - fraction[valid])
        + _FSL_HANNING_SINC[lower[valid]] * fraction[valid]
    )
    return result


def _fsl_sinc_sample_reference(
    source: np.ndarray, coordinates: np.ndarray, cval: float
) -> np.ndarray:
    """Run FLIRT width-7 Hanning sinc sampling with the vectorized reference kernel."""

    base = np.floor(coordinates).astype(np.int64)
    convolution = np.zeros(coordinates.shape[1], dtype=np.float32)
    kernel_sum = np.zeros(coordinates.shape[1], dtype=np.float32)
    center_valid = np.all(coordinates >= 0, axis=0) & np.all(
        coordinates <= (np.asarray(source.shape) - 1)[:, None], axis=0
    )
    axis_weights = []
    for axis in range(3):
        weights = []
        for offset in range(-3, 4):
            distance = coordinates[axis] - (base[axis] + offset)
            weights.append(_fsl_sinc_kernel_values(distance))
        axis_weights.append(weights)
    for z_offset in range(-3, 4):
        z_index = base[2] + z_offset
        for y_offset in range(-3, 4):
            y_index = base[1] + y_offset
            yz_weight = (
                axis_weights[1][y_offset + 3] * axis_weights[2][z_offset + 3]
            )
            for x_offset in range(-3, 4):
                x_index = base[0] + x_offset
                valid = (
                    center_valid
                    & (x_index >= 0)
                    & (x_index < source.shape[0])
                    & (y_index >= 0)
                    & (y_index < source.shape[1])
                    & (z_index >= 0)
                    & (z_index < source.shape[2])
                )
                weight = axis_weights[0][x_offset + 3] * yz_weight
                convolution[valid] += (
                    source[x_index[valid], y_index[valid], z_index[valid]] * weight[valid]
                )
                kernel_sum[valid] += weight[valid]
    sampled = np.full(coordinates.shape[1], cval, dtype=np.float32)
    valid_sum = np.abs(kernel_sum) > 1e-9
    sampled[valid_sum] = convolution[valid_sum] / kernel_sum[valid_sum]
    return sampled


@njit(cache=True, nogil=True, inline="always")
def _fsl_sinc_weight(distance: np.float32) -> np.float32:
    """Return one sinc weight from FSL's 1,201-point lookup table."""

    lookup = np.float32(
        distance / np.float32(3.0) * np.float32(600.0) + np.float32(601.0)
    )
    lower = int(np.floor(lookup))
    if lower < 1 or lower > 1200:
        return np.float32(0.0)
    # Subtracting an int64 index from the reference vector kernel's float32
    # lookup value promotes the result to float64.
    fraction = float(lookup) - float(lower)
    return np.float32(
        float(_FSL_HANNING_SINC[lower - 1]) * (1.0 - fraction)
        + float(_FSL_HANNING_SINC[lower]) * fraction
    )


@njit(cache=True, nogil=True, parallel=True)
def _fsl_sinc_sample_kernel(
    source: np.ndarray, coordinates: np.ndarray, cval: np.float32
) -> np.ndarray:
    """Parallelize output points while retaining the fixed z/y/x sum order per point."""

    sampled = np.full(coordinates.shape[1], cval, dtype=np.float32)
    for position in prange(coordinates.shape[1]):
        xcoord = coordinates[0, position]
        ycoord = coordinates[1, position]
        zcoord = coordinates[2, position]
        if not (
            0.0 <= xcoord <= source.shape[0] - 1
            and 0.0 <= ycoord <= source.shape[1] - 1
            and 0.0 <= zcoord <= source.shape[2] - 1
        ):
            continue
        xbase = int(np.floor(xcoord))
        ybase = int(np.floor(ycoord))
        zbase = int(np.floor(zcoord))
        convolution = np.float32(0.0)
        kernel_sum = np.float32(0.0)
        for zoffset in range(-3, 4):
            zindex = zbase + zoffset
            wz = _fsl_sinc_weight(np.float32(zcoord - zindex))
            for yoffset in range(-3, 4):
                yindex = ybase + yoffset
                wy = _fsl_sinc_weight(np.float32(ycoord - yindex))
                yz_weight = np.float32(wy * wz)
                for xoffset in range(-3, 4):
                    xindex = xbase + xoffset
                    if (
                        0 <= xindex < source.shape[0]
                        and 0 <= yindex < source.shape[1]
                        and 0 <= zindex < source.shape[2]
                    ):
                        wx = _fsl_sinc_weight(np.float32(xcoord - xindex))
                        weight = np.float32(wx * yz_weight)
                        convolution = np.float32(
                            convolution
                            + np.float32(source[xindex, yindex, zindex] * weight)
                        )
                        kernel_sum = np.float32(kernel_sum + weight)
        if abs(kernel_sum) > np.float32(1e-9):
            sampled[position] = np.float32(convolution / kernel_sum)
    return sampled


def _fsl_sinc_sample(source: np.ndarray, coordinates: np.ndarray, cval: float) -> np.ndarray:
    """Run FLIRT width-7 Hanning sinc sampling with the optimized kernel."""

    return _fsl_sinc_sample_kernel(
        np.asarray(source, dtype=np.float32),
        np.asarray(coordinates, dtype=np.float64),
        np.float32(cval),
    )


def _grid_contract(shape: Sequence[int], affine: np.ndarray, name: str) -> tuple:
    spatial_shape = tuple(int(value) for value in shape)
    matrix = np.asarray(affine, dtype=np.float64)
    if len(spatial_shape) != 3 or any(value <= 0 for value in spatial_shape):
        raise ValueError(f"{name}_shape must contain three positive dimensions")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}_affine must be a finite 4x4 matrix")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError(f"{name}_affine must be invertible")
    return spatial_shape, matrix


def output_to_input_voxel_matrix(
    moving_affine: np.ndarray,
    reference_affine: np.ndarray,
    world_transform: np.ndarray,
) -> np.ndarray:
    """Return the pull matrix from reference voxels to moving voxels."""

    _, moving = _grid_contract((1, 1, 1), moving_affine, "moving")
    _, reference = _grid_contract((1, 1, 1), reference_affine, "reference")
    return np.linalg.inv(moving) @ invert_transform(world_transform) @ reference


def resample_image(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray,
    world_transform: np.ndarray,
    *,
    interpolation: str = "linear",
    reference_to_moving_displacement: np.ndarray | None = None,
    cval: float = 0.0,
    z_chunk: int = 8,
    linear_extrapolation: str = "partial",
) -> np.ndarray:
    """Resample a 3D or channel-last 4D image in one interpolation pass.

    ``world_transform`` maps moving-world points to reference-world points.
    An optional displacement is defined on the reference grid in reference-world
    millimetres. It is added to the reference coordinate before the inverse affine
    map, matching FSL ``applywarp --premat`` pull-coordinate composition.
    """

    values = np.asarray(moving)
    target_shape, target_affine = _grid_contract(
        reference_shape, reference_affine, "reference"
    )
    _, source_affine = _grid_contract(
        values.shape[:3], moving_affine, "moving"
    )
    if values.ndim not in (3, 4):
        raise ValueError("moving must be a 3D or channel-last 4D array")
    if interpolation not in _INTERPOLATION_ORDER:
        raise ValueError("interpolation must be nearest, linear, spline, or sinc")
    if not np.isfinite(cval):
        raise ValueError("cval must be finite")
    if z_chunk <= 0:
        raise ValueError("z_chunk must be positive")
    if linear_extrapolation not in ("partial", "constant", "fsl"):
        raise ValueError("linear_extrapolation must be partial, constant, or fsl")
    displacement = reference_to_moving_displacement
    if displacement is not None:
        displacement = np.asarray(displacement, dtype=np.float64)
        if displacement.shape != target_shape + (3,) or not np.all(
            np.isfinite(displacement)
        ):
            raise ValueError(
                "reference_to_moving_displacement must be finite and match the reference grid"
            )

    inverse_world = invert_transform(world_transform)
    inverse_source = np.linalg.inv(source_affine)
    channels = 1 if values.ndim == 3 else values.shape[3]
    output_shape = target_shape if channels == 1 else target_shape + (channels,)
    output = np.empty(output_shape, dtype=np.float32)
    order = _INTERPOLATION_ORDER[interpolation]
    for z0 in range(0, target_shape[2], z_chunk):
        z1 = min(z0 + z_chunk, target_shape[2])
        grid = np.indices((target_shape[0], target_shape[1], z1 - z0), dtype=np.float64)
        grid[2] += z0
        homogeneous = np.concatenate(
            (grid.reshape(3, -1), np.ones((1, grid[0].size), dtype=np.float64)),
            axis=0,
        )
        reference_world = target_affine @ homogeneous
        if displacement is not None:
            reference_world[:3] += displacement[:, :, z0:z1].reshape(-1, 3).T
        moving_world = inverse_world @ reference_world
        coordinates = (inverse_source @ moving_world)[:3]
        for channel in range(channels):
            source = values if values.ndim == 3 else values[..., channel]
            if order == 0:
                # Match MISCMATHS::round: half values round away from zero.
                indices = np.where(
                    coordinates > 0,
                    np.floor(coordinates + 0.5),
                    np.ceil(coordinates - 0.5),
                ).astype(np.int64)
                valid = np.all(indices >= 0, axis=0) & np.all(
                    indices < np.asarray(source.shape)[:, None], axis=0
                )
                sampled_flat = np.full(indices.shape[1], cval, dtype=np.float32)
                sampled_flat[valid] = source[tuple(indices[:, valid])]
                sampled = sampled_flat.reshape(
                    target_shape[0], target_shape[1], z1 - z0
                )
            elif order > 0:
                sampled = map_coordinates(
                    source,
                    coordinates,
                    order=order,
                    mode=(
                        "grid-constant"
                        if order == 1 and linear_extrapolation == "partial"
                        else "nearest"
                        if order == 1 and linear_extrapolation == "fsl"
                        else "constant"
                    ),
                    cval=float(cval),
                    prefilter=order > 1,
                )
                if order == 1 and linear_extrapolation == "fsl":
                    outside = np.any(coordinates < 0, axis=0) | np.any(
                        coordinates > (np.asarray(source.shape) - 1)[:, None], axis=0
                    )
                    sampled[outside] = np.float32(cval)
                sampled = sampled.reshape(
                    target_shape[0], target_shape[1], z1 - z0
                )
            else:
                sampled = _fsl_sinc_sample(source, coordinates, cval).reshape(
                    target_shape[0], target_shape[1], z1 - z0
                )
            if values.ndim == 3:
                output[:, :, z0:z1] = sampled
            else:
                output[:, :, z0:z1, channel] = sampled
    return output


def resample_mask(
    mask: np.ndarray,
    moving_affine: np.ndarray,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray,
    world_transform: np.ndarray,
    *,
    reference_to_moving_displacement: np.ndarray | None = None,
    z_chunk: int = 8,
) -> np.ndarray:
    """Resample a binary mask with mandatory nearest-neighbour interpolation."""

    values = np.asarray(mask)
    if values.ndim != 3:
        raise ValueError("mask must be three-dimensional")
    sampled = resample_image(
        values.astype(np.uint8),
        moving_affine,
        reference_shape,
        reference_affine,
        world_transform,
        interpolation="nearest",
        reference_to_moving_displacement=reference_to_moving_displacement,
        z_chunk=z_chunk,
    )
    return (sampled > 0).astype(np.uint8)
