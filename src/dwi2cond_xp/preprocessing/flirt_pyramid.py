"""FSL FLIRT image and weight pyramid construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

from numba import njit, prange
import numpy as np

from .brain_mask import robust_intensity_limits


def _volume(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or min(array.shape) < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3D array of size at least two")
    # FSL uses x as the contiguous memory axis, matching this module's fixed z/y/x
    # traversal order.
    return np.asfortranarray(array)


def _sampling(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if abs(np.linalg.det(values[:3, :3])) < 1e-12:
        raise ValueError(f"{name} must be invertible")
    return values


def _background_value(volume: np.ndarray) -> np.float32:
    xsize, ysize, zsize = volume.shape
    edge_x, edge_y, edge_z = min(2, xsize - 1), min(2, ysize - 1), min(2, zsize - 1)
    values: list[np.float32] = []
    for edge in range(edge_z):
        for xindex in range(edge_x, xsize - edge_x):
            for yindex in range(edge_y, ysize - edge_y):
                values.extend((volume[xindex, yindex, edge], volume[xindex, yindex, zsize - 1 - edge]))
    for edge in range(edge_y):
        for xindex in range(edge_x, xsize - edge_x):
            for zindex in range(zsize):
                values.extend((volume[xindex, edge, zindex], volume[xindex, ysize - 1 - edge, zindex]))
    for edge in range(edge_x):
        for yindex in range(ysize):
            for zindex in range(zsize):
                values.extend((volume[edge, yindex, zindex], volume[xsize - 1 - edge, yindex, zindex]))
    ordered = np.sort(np.asarray(values, dtype=np.float32))
    return ordered[ordered.size // 10]


def _blur_kernel(final_voxel_size: float, initial_voxel_size: float) -> np.ndarray:
    ratio = np.float32(final_voxel_size / initial_voxel_size)
    if ratio < np.float32(1.1):
        return np.ones(1, dtype=np.float64)
    sigma = np.float32(0.85) * np.float32(ratio / np.float32(2.0))
    if sigma < np.float32(0.5):
        return np.ones(1, dtype=np.float64)
    size = int(np.float32(sigma - np.float32(0.001))) * 2 + 3
    middle = size // 2
    kernel = np.empty(size, dtype=np.float64)
    for index in range(size):
        distance = np.float32(index - middle)
        kernel[index] = math.exp(
            -float(np.float32(distance * distance))
            / float(np.float32(np.float32(sigma * sigma) * np.float32(4.0)))
        )
    kernel /= np.sum(kernel)
    return kernel


@njit(cache=True, nogil=True, parallel=True)
def _convolve_axis(
    source: np.ndarray, kernel: np.ndarray, axis: int, padding: np.float32
) -> np.ndarray:
    result = np.empty_like(source)
    middle = (kernel.size - 1) // 2
    for zindex in prange(source.shape[2]):
        for yindex in range(source.shape[1]):
            for xindex in range(source.shape[0]):
                value = np.float32(0.0)
                for kernel_index in range(kernel.size):
                    offset = kernel_index - middle
                    xx = xindex + (offset if axis == 0 else 0)
                    yy = yindex + (offset if axis == 1 else 0)
                    zz = zindex + (offset if axis == 2 else 0)
                    sample = padding
                    if (
                        0 <= xx < source.shape[0]
                        and 0 <= yy < source.shape[1]
                        and 0 <= zz < source.shape[2]
                    ):
                        sample = source[xx, yy, zz]
                    value = np.float32(value + float(sample) * kernel[kernel_index])
                result[xindex, yindex, zindex] = value
    return result


def flirt_blur(
    volume: np.ndarray,
    voxel_sizes: np.ndarray,
    final_voxel_size: float,
    *,
    padding: float | None = None,
) -> np.ndarray:
    """Apply FSL ``blur`` kernels with constant background extrapolation."""

    values = _volume(volume, "volume")
    sizes = np.asarray(voxel_sizes, dtype=np.float64)
    if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("voxel_sizes must contain three positive finite values")
    if not np.isfinite(final_voxel_size) or final_voxel_size <= 0:
        raise ValueError("final_voxel_size must be positive and finite")
    pad = np.float32(0.0) if padding is None else np.float32(padding)
    if not np.isfinite(pad):
        raise ValueError("padding must be finite")
    result = values
    for axis in range(3):
        result = _convolve_axis(
            result, _blur_kernel(final_voxel_size, float(sizes[axis])), axis, pad
        )
    return result


@njit(cache=True, nogil=True)
def _linear_sample(volume: np.ndarray, xvalue: np.float32, yvalue: np.float32, zvalue: np.float32, padding: np.float32) -> np.float32:
    xindex, yindex, zindex = int(xvalue), int(yvalue), int(zvalue)
    dx = np.float32(xvalue - np.float32(xindex))
    dy = np.float32(yvalue - np.float32(yindex))
    dz = np.float32(zvalue - np.float32(zindex))
    neighbours = np.empty(8, dtype=np.float32)
    position = 0
    for xoffset in range(2):
        for yoffset in range(2):
            for zoffset in range(2):
                xx, yy, zz = xindex + xoffset, yindex + yoffset, zindex + zoffset
                neighbours[position] = (
                    volume[xx, yy, zz]
                    if 0 <= xx < volume.shape[0] and 0 <= yy < volume.shape[1] and 0 <= zz < volume.shape[2]
                    else padding
                )
                position += 1
    temp1 = np.float32(np.float32(neighbours[4] - neighbours[0]) * dx + neighbours[0])
    temp2 = np.float32(np.float32(neighbours[5] - neighbours[1]) * dx + neighbours[1])
    temp3 = np.float32(np.float32(neighbours[6] - neighbours[2]) * dx + neighbours[2])
    temp4 = np.float32(np.float32(neighbours[7] - neighbours[3]) * dx + neighbours[3])
    temp5 = np.float32(np.float32(temp3 - temp1) * dy + temp1)
    temp6 = np.float32(np.float32(temp4 - temp2) * dy + temp2)
    return np.float32(np.float32(temp6 - temp5) * dz + temp5)


@njit(cache=True, nogil=True, parallel=True)
def _isotropic_kernel(
    volume: np.ndarray, steps: np.ndarray, shape: np.ndarray, padding: np.float32
) -> np.ndarray:
    result = np.empty((shape[0], shape[1], shape[2]), dtype=np.float32)
    xcoordinates = np.empty(shape[0], dtype=np.float32)
    ycoordinates = np.empty(shape[1], dtype=np.float32)
    zcoordinates = np.empty(shape[2], dtype=np.float32)
    value = np.float32(0.0)
    for xindex in range(shape[0]):
        xcoordinates[xindex] = value
        value = np.float32(value + steps[0])
    value = np.float32(0.0)
    for yindex in range(shape[1]):
        ycoordinates[yindex] = value
        value = np.float32(value + steps[1])
    value = np.float32(0.0)
    for zindex in range(shape[2]):
        zcoordinates[zindex] = value
        value = np.float32(value + steps[2])
    for zindex in prange(shape[2]):
        zvalue = zcoordinates[zindex]
        for yindex in range(shape[1]):
            yvalue = ycoordinates[yindex]
            for xindex in range(shape[0]):
                xvalue = xcoordinates[xindex]
                result[xindex, yindex, zindex] = _linear_sample(
                    volume, xvalue, yvalue, zvalue, padding
                )
    return result


def isotropic_resample(
    volume: np.ndarray,
    sampling_matrix: np.ndarray,
    scale: float,
    *,
    padding: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce NEWIMAGE ``isotropic_resample`` and its grid transform."""

    values = _volume(volume, "volume")
    sampling = _sampling(sampling_matrix, "sampling_matrix")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    sizes = np.linalg.norm(sampling[:3, :3], axis=0)
    steps = np.asarray(np.float32(scale) / sizes.astype(np.float32), dtype=np.float32)
    shape = np.maximum(1, (np.asarray(values.shape, dtype=np.float32) / steps).astype(np.int64))
    pad = np.float32(0.0) if padding is None else np.float32(padding)
    output = _isotropic_kernel(values, steps, shape, pad)
    voxel_mapping = np.diag([float(steps[0]), float(steps[1]), float(steps[2]), 1.0])
    return output, sampling @ voxel_mapping


@njit(cache=True, nogil=True, parallel=True)
def _subsample_kernel(volume: np.ndarray, padding: np.float32) -> np.ndarray:
    output = np.empty(
        (
            (volume.shape[0] + 1) // 2,
            (volume.shape[1] + 1) // 2,
            (volume.shape[2] + 1) // 2,
        ),
        dtype=np.float32,
    )
    coefficients = np.array([0.125, 0.0625, 0.0312, 0.0156], dtype=np.float32)
    for zindex in prange(output.shape[2]):
        base_z = zindex * 2
        for yindex in range(output.shape[1]):
            base_y = yindex * 2
            for xindex in range(output.shape[0]):
                base_x = xindex * 2
                value = np.float32(0.0)
                for zoffset in range(-1, 2):
                    for yoffset in range(-1, 2):
                        for xoffset in range(-1, 2):
                            distance = abs(xoffset) + abs(yoffset) + abs(zoffset)
                            sample = padding
                            xx, yy, zz = base_x + xoffset, base_y + yoffset, base_z + zoffset
                            if 0 <= xx < volume.shape[0] and 0 <= yy < volume.shape[1] and 0 <= zz < volume.shape[2]:
                                sample = volume[xx, yy, zz]
                            value = np.float32(value + np.float32(sample * coefficients[distance]))
                output[xindex, yindex, zindex] = value
    return output


def subsample_by_two(
    volume: np.ndarray, sampling_matrix: np.ndarray, *, padding: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FLIRT's centered 27-point ``subsample_by_2`` kernel."""

    values = _volume(volume, "volume")
    sampling = _sampling(sampling_matrix, "sampling_matrix")
    pad = np.float32(0.0) if padding is None else np.float32(padding)
    output = _subsample_kernel(values, pad)
    return output, sampling @ np.diag([2.0, 2.0, 2.0, 1.0])


def _filter_image(
    image: np.ndarray,
    weight: np.ndarray,
    operation,
) -> np.ndarray:
    binary = (weight > np.float32(0.01)).astype(np.float32)
    numerator = operation(np.asarray(image * binary, dtype=np.float32))
    denominator = operation(binary)
    return np.divide(
        numerator,
        denominator,
        out=np.asarray(denominator, dtype=np.float32).copy(),
        where=denominator != 0,
    ).astype(np.float32)


def _filter_weight(weight: np.ndarray, operation) -> np.ndarray:
    binary = (weight > np.float32(0.01)).astype(np.float32)
    support = operation(binary) > np.float32(0.9)
    filtered = operation(weight)
    return np.asarray(filtered * support, dtype=np.float32)


def _subsample_image_and_weight(
    image: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Share the identical binary-support convolution used by FSL image/weight filtering."""

    binary = (weight > np.float32(0.01)).astype(np.float32)
    denominator = _subsample_kernel(binary, np.float32(0.0))
    numerator = _subsample_kernel(
        np.asarray(image * binary, dtype=np.float32), np.float32(0.0)
    )
    filtered_image = np.divide(
        numerator,
        denominator,
        out=np.asarray(denominator, dtype=np.float32).copy(),
        where=denominator != 0,
    ).astype(np.float32)
    filtered_weight = (
        denominator
        if np.array_equal(weight, binary)
        else _subsample_kernel(weight, np.float32(0.0))
    )
    return filtered_image, np.asarray(
        filtered_weight * (denominator > np.float32(0.9)), dtype=np.float32
    )


@dataclass(frozen=True)
class FlirtPyramidLevel:
    """Reference/moving images, weights, sampling and histogram count at one scale."""

    requested_scale: float
    reference: np.ndarray
    moving: np.ndarray
    reference_weight: np.ndarray
    moving_weight: np.ndarray
    reference_sampling: np.ndarray
    moving_sampling: np.ndarray
    bins: int


def build_flirt_pyramid(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_weight: np.ndarray,
    moving_weight: np.ndarray,
    reference_sampling: np.ndarray,
    moving_sampling: np.ndarray,
    *,
    use_weights: bool = True,
) -> dict[int, FlirtPyramidLevel]:
    """Build the weighted or unweighted 8/4/2/minimum-scale FLIRT pyramid."""

    ref = _volume(reference, "reference")
    mov = _volume(moving, "moving")
    ref_weight = _volume(reference_weight, "reference_weight")
    mov_weight = _volume(moving_weight, "moving_weight")
    if ref_weight.shape != ref.shape or mov_weight.shape != mov.shape:
        raise ValueError("each weight must match its image")
    ref_sampling = _sampling(reference_sampling, "reference_sampling")
    mov_sampling = _sampling(moving_sampling, "moving_sampling")
    ref_min, ref_max = robust_intensity_limits(ref)
    mov_min, mov_max = robust_intensity_limits(mov)
    ref = np.clip(ref, np.float32(ref_min), np.float32(ref_max)).astype(np.float32)
    mov = np.clip(mov, np.float32(mov_min), np.float32(mov_max)).astype(np.float32)
    ref_sizes = np.linalg.norm(ref_sampling[:3, :3], axis=0)
    mov_sizes = np.linalg.norm(mov_sampling[:3, :3], axis=0)
    minimum_scale = int(math.ceil(max(float(ref_sizes.min()), float(mov_sizes.min()))))

    if use_weights:
        ref_binary = (ref_weight > np.float32(0.01)).astype(np.float32)
        ref_numerator = np.asarray(ref * ref_binary, dtype=np.float32)
        numerator_padding = 0.0
        binary_padding = 0.0
        numerator_blurred = flirt_blur(
            ref_numerator, ref_sizes, minimum_scale, padding=numerator_padding
        )
        denominator_blurred = flirt_blur(
            ref_binary, ref_sizes, minimum_scale, padding=binary_padding
        )
        ref_base, ref_base_sampling = isotropic_resample(
            numerator_blurred, ref_sampling, minimum_scale, padding=numerator_padding
        )
        denominator, _ = isotropic_resample(
            denominator_blurred, ref_sampling, minimum_scale, padding=binary_padding
        )
        ref_base = np.divide(
            ref_base,
            denominator,
            out=denominator.copy(),
            where=denominator != 0,
        ).astype(np.float32)
        weight_padding = 0.0
        blurred_support = denominator_blurred
        blurred_weight = (
            blurred_support
            if np.array_equal(ref_weight, ref_binary)
            else flirt_blur(ref_weight, ref_sizes, minimum_scale, padding=weight_padding)
        )
        ref_base_weight = (
            denominator
            if blurred_weight is blurred_support
            else isotropic_resample(
                blurred_weight, ref_sampling, minimum_scale, padding=weight_padding
            )[0]
        )
        ref_base_weight = np.asarray(
            ref_base_weight * (denominator > np.float32(0.9)), dtype=np.float32
        )
    else:
        ref_padding = float(_background_value(ref))
        ref_blurred = flirt_blur(
            ref, ref_sizes, minimum_scale, padding=ref_padding
        )
        ref_base, ref_base_sampling = isotropic_resample(
            ref_blurred, ref_sampling, minimum_scale, padding=ref_padding
        )
        ref_base_weight = np.ones(ref_base.shape, dtype=np.float32)

    ref_base = np.asfortranarray(ref_base)
    ref_base_weight = np.asfortranarray(ref_base_weight)
    reference_levels: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        1: (ref_base, ref_base_weight, ref_base_sampling)
    }
    current_image, current_weight, current_sampling = ref_base, ref_base_weight, ref_base_sampling
    for scale in (2, 4, 8):
        if minimum_scale >= scale - 0.1:
            reference_levels[scale] = (current_image, current_weight, current_sampling)
            continue
        if use_weights:
            current_image, current_weight = _subsample_image_and_weight(
                current_image, current_weight
            )
        else:
            current_image = _subsample_kernel(
                current_image, np.float32(_background_value(current_image))
            )
            current_weight = np.ones(current_image.shape, dtype=np.float32)
        current_image = np.asfortranarray(current_image)
        current_weight = np.asfortranarray(current_weight)
        current_sampling = current_sampling @ np.diag([2.0, 2.0, 2.0, 1.0])
        reference_levels[scale] = (current_image, current_weight, current_sampling)

    levels: dict[int, FlirtPyramidLevel] = {}
    moving_weight_padding = 0.0
    moving_binary = (mov_weight > np.float32(0.01)).astype(np.float32)
    moving_product = np.asarray(mov * moving_binary, dtype=np.float32)
    moving_padding = float(_background_value(mov))
    moving_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    active_scale = 8
    for requested_scale in (8, 4, 2, 1):
        if minimum_scale <= 1.25 * requested_scale:
            active_scale = requested_scale
        reference_level = reference_levels[active_scale]
        if active_scale not in moving_cache:
            if use_weights:
                moving_denominator = flirt_blur(
                    moving_binary, mov_sizes, active_scale, padding=0.0
                )
                moving_numerator = flirt_blur(
                    moving_product, mov_sizes, active_scale, padding=0.0
                )
                moving_level = np.divide(
                    moving_numerator,
                    moving_denominator,
                    out=moving_denominator.copy(),
                    where=moving_denominator != 0,
                ).astype(np.float32)
                moving_level_weight = (
                    moving_denominator
                    if np.array_equal(mov_weight, moving_binary)
                    else flirt_blur(
                        mov_weight,
                        mov_sizes,
                        active_scale,
                        padding=moving_weight_padding,
                    )
                )
                moving_level_weight = np.asarray(
                    moving_level_weight * (moving_denominator > np.float32(0.9)),
                    dtype=np.float32,
                )
            else:
                moving_level = flirt_blur(
                    mov, mov_sizes, active_scale, padding=moving_padding
                )
                moving_level_weight = np.ones(mov.shape, dtype=np.float32)
            moving_level = np.asfortranarray(moving_level)
            moving_level_weight = np.asfortranarray(moving_level_weight)
            moving_cache[active_scale] = (moving_level, moving_level_weight)
        moving_level, moving_level_weight = moving_cache[active_scale]
        levels[requested_scale] = FlirtPyramidLevel(
            requested_scale=float(requested_scale),
            reference=reference_level[0],
            moving=moving_level,
            reference_weight=reference_level[1],
            moving_weight=moving_level_weight,
            reference_sampling=reference_level[2],
            moving_sampling=mov_sampling,
            bins=256 // active_scale,
        )
    return levels
