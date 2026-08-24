"""Apply FNIRT Jacobian topology constraints in FSL ``warpfns`` order."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import fft


_GOOD_FFT_SIZES = (
    4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 60, 64, 75, 80,
    90, 98, 100, 104, 108, 112, 117, 121, 125, 128, 135, 144, 145, 150, 153,
    160, 162, 169, 175, 180, 189, 192, 196, 200, 208, 216, 225, 240, 245,
    250, 256, 270, 272, 275, 288, 289, 294, 300, 304, 315, 320, 325, 336,
    338, 343, 350, 360, 363, 375, 384, 392, 400, 405, 416, 420, 432, 441,
    448, 450, 459, 475, 480, 484, 490, 500, 504, 507, 512, 525, 550, 600,
    650, 700, 750, 800, 850, 900, 950, 1000, 1024, 1280, 1536, 1792, 2048,
    2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192,
)

_JACOBIAN_COLUMN_OFFSETS = np.asarray(
    [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [0, 0, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [1, 1, 0]],
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    ],
    dtype=np.int64,
)


def fsl_good_fft_size(size: int) -> int:
    """Return the first usable size selected by ``fnirt_CF::good_fft_size``."""

    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("FFT size must be a positive integer")
    for candidate in _GOOD_FFT_SIZES:
        if candidate >= size:
            return candidate
    return int(size)


@njit(cache=True)
def _determinant(matrix: np.ndarray) -> float:
    """Compute a third-order determinant in NEWMAT expansion order."""

    return (
        matrix[0, 0]
        * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1]
        * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2]
        * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


@njit(cache=True)
def _jacobian_check(
    warp: np.ndarray,
    voxel_sizes_mm: np.ndarray,
    minimum: float,
    maximum: float,
    store_volume: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the corner Jacobians, statistics, and traversal of ``jacobian_check``."""

    nx, ny, nz, _ = warp.shape
    jacobians = np.zeros((nx, ny, nz, 8), dtype=np.float32)
    statistics = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    matrix = np.empty((3, 3), dtype=np.float64)
    for z in range(nz - 1):
        for y in range(ny - 1):
            for x in range(nx - 1):
                for corner in range(8):
                    for column in range(3):
                        bx = x + _JACOBIAN_COLUMN_OFFSETS[corner, column, 0]
                        by = y + _JACOBIAN_COLUMN_OFFSETS[corner, column, 1]
                        bz = z + _JACOBIAN_COLUMN_OFFSETS[corner, column, 2]
                        ex, ey, ez = bx, by, bz
                        if column == 0:
                            ex += 1
                        elif column == 1:
                            ey += 1
                        else:
                            ez += 1
                        for component in range(3):
                            matrix[component, column] = float(
                                warp[ex, ey, ez, component]
                                - warp[bx, by, bz, component]
                            ) / float(voxel_sizes_mm[component])
                    determinant = _determinant(matrix)
                    if determinant < statistics[0]:
                        statistics[0] = determinant
                    if determinant > statistics[1]:
                        statistics[1] = determinant
                    if determinant < minimum:
                        statistics[2] += 1.0
                    if determinant > maximum:
                        statistics[3] += 1.0
                    if store_volume:
                        jacobians[x, y, z, corner] = np.float32(determinant)
    return jacobians, statistics


@njit(cache=True)
def _gradient_calculation(
    warp: np.ndarray, voxel_sizes_mm: np.ndarray
) -> np.ndarray:
    """Reproduce the periodic boundaries and component scaling of ``grad_calc``."""

    nx, ny, nz, _ = warp.shape
    gradient = np.empty((nx, ny, nz, 3, 3), dtype=np.float32)
    for z in range(nz):
        z2 = 0 if z + 1 == nz else z + 1
        for y in range(ny):
            y2 = 0 if y + 1 == ny else y + 1
            for x in range(nx):
                x2 = 0 if x + 1 == nx else x + 1
                for component in range(3):
                    scale = np.float32(voxel_sizes_mm[component])
                    gradient[x, y, z, component, 0] = (
                        warp[x2, y, z, component] - warp[x, y, z, component]
                    ) / scale
                    gradient[x, y, z, component, 1] = (
                        warp[x, y2, z, component] - warp[x, y, z, component]
                    ) / scale
                    gradient[x, y, z, component, 2] = (
                        warp[x, y, z2, component] - warp[x, y, z, component]
                    ) / scale
    return gradient


@njit(cache=True)
def _limit_gradient(
    gradient: np.ndarray,
    jacobians: np.ndarray,
    minimum: float,
    maximum: float,
) -> None:
    """Pull out-of-range matrices toward identity in ``limit_grad`` in-place corner order."""

    nx, ny, nz = gradient.shape[:3]
    matrix = np.empty((3, 3), dtype=np.float64)
    candidate = np.empty((3, 3), dtype=np.float64)
    for z in range(nz - 1):
        for y in range(ny - 1):
            for x in range(nx - 1):
                for corner in range(8):
                    stored_determinant = float(jacobians[x, y, z, corner])
                    if (
                        stored_determinant >= minimum
                        and stored_determinant <= maximum
                    ):
                        continue
                    for column in range(3):
                        ox = x + _JACOBIAN_COLUMN_OFFSETS[corner, column, 0]
                        oy = y + _JACOBIAN_COLUMN_OFFSETS[corner, column, 1]
                        oz = z + _JACOBIAN_COLUMN_OFFSETS[corner, column, 2]
                        for component in range(3):
                            matrix[component, column] = gradient[
                                ox, oy, oz, component, column
                            ]
                    determinant = _determinant(matrix)
                    alpha = np.float32(0.0)
                    while determinant > maximum or determinant < minimum:
                        alpha = np.float32(alpha + np.float32(0.1))
                        if alpha > 1.0:
                            alpha = np.float32(1.0)
                        for row in range(3):
                            for column in range(3):
                                identity = 1.0 if row == column else 0.0
                                candidate[row, column] = (
                                    (1.0 - float(alpha)) * matrix[row, column]
                                    + float(alpha) * identity
                                )
                        determinant = _determinant(candidate)
                    alpha = np.float32(alpha + np.float32(0.1))
                    if alpha > 1.0:
                        alpha = np.float32(1.0)
                    for column in range(3):
                        ox = x + _JACOBIAN_COLUMN_OFFSETS[corner, column, 0]
                        oy = y + _JACOBIAN_COLUMN_OFFSETS[corner, column, 1]
                        oz = z + _JACOBIAN_COLUMN_OFFSETS[corner, column, 2]
                        for row in range(3):
                            identity = 1.0 if row == column else 0.0
                            gradient[ox, oy, oz, row, column] = np.float32(
                                (1.0 - float(alpha)) * matrix[row, column]
                                + float(alpha) * identity
                            )


def _fsl_fft3(values: np.ndarray, *, inverse: bool) -> np.ndarray:
    """Transform along x/y/z, returning to FSL float storage after each direction."""

    source = np.asarray(values)
    transformed = np.asarray(source.real, dtype=np.float32).astype(np.complex128)
    transformed += 1j * np.asarray(source.imag, dtype=np.float32)
    transform = fft.ifft if inverse else fft.fft
    for axis in range(3):
        transformed = transform(transformed, axis=axis)
        real = np.asarray(transformed.real, dtype=np.float32)
        imaginary = np.asarray(transformed.imag, dtype=np.float32)
        transformed = real.astype(np.complex128) + 1j * imaginary
    return transformed


def _integrate_gradient_field(
    gradient: np.ndarray,
    voxel_sizes_mm: np.ndarray,
    means: np.ndarray,
) -> np.ndarray:
    """Reproduce FSL's frequency-domain integrable-gradient projection and mean recovery."""

    nx, ny, nz = gradient.shape[:3]
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    z = np.arange(nz, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    cosx = np.cos(two_pi * x / np.float32(nx)).astype(np.float32)
    sinx = np.sin(two_pi * x / np.float32(nx)).astype(np.float32)
    cosy = np.cos(two_pi * y / np.float32(ny)).astype(np.float32)
    siny = np.sin(two_pi * y / np.float32(ny)).astype(np.float32)
    cosz = np.cos(two_pi * z / np.float32(nz)).astype(np.float32)
    sinz = np.sin(two_pi * z / np.float32(nz)).astype(np.float32)
    norm = (
        np.float32(6.0)
        - np.float32(2.0) * cosx[:, None, None]
        - np.float32(2.0) * cosy[None, :, None]
        - np.float32(2.0) * cosz[None, None, :]
    ).astype(np.float32)
    output = np.empty(gradient.shape[:3] + (3,), dtype=np.float32)
    for component in range(3):
        spectra = tuple(
            _fsl_fft3(gradient[..., component, axis], inverse=False)
            for axis in range(3)
        )
        real = (
            spectra[0].real * (cosx[:, None, None] - np.float32(1.0))
            + spectra[0].imag * sinx[:, None, None]
        ).astype(np.float32)
        imaginary = (
            spectra[0].imag * (cosx[:, None, None] - np.float32(1.0))
            - spectra[0].real * sinx[:, None, None]
        ).astype(np.float32)
        real += spectra[1].real * (cosy[None, :, None] - np.float32(1.0))
        real += spectra[1].imag * siny[None, :, None]
        imaginary += spectra[1].imag * (cosy[None, :, None] - np.float32(1.0))
        imaginary -= spectra[1].real * siny[None, :, None]
        real += spectra[2].real * (cosz[None, None, :] - np.float32(1.0))
        real += spectra[2].imag * sinz[None, None, :]
        imaginary += spectra[2].imag * (cosz[None, None, :] - np.float32(1.0))
        imaginary -= spectra[2].real * sinz[None, None, :]
        nonzero = np.abs(norm) > np.float32(1.0e-12)
        integrated_spectrum = np.zeros(norm.shape, dtype=np.complex64)
        integrated_spectrum.real[nonzero] = real[nonzero] / norm[nonzero]
        integrated_spectrum.imag[nonzero] = imaginary[nonzero] / norm[nonzero]
        spatial = _fsl_fft3(integrated_spectrum, inverse=True).real.astype(np.float32)
        output[..., component] = (
            spatial * np.float32(voxel_sizes_mm[component])
            + np.float32(means[component])
        )
    return output


def fsl_constrain_topology(
    absolute_warp: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    *,
    minimum_jacobian: float = 0.01,
    maximum_jacobian: float = 100.0,
) -> np.ndarray:
    """Run up to nine inner projection rounds from ``warpfns::constrain_topology``."""

    warp = np.asarray(absolute_warp, dtype=np.float32)
    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float32)
    if warp.ndim != 4 or warp.shape[-1] != 3 or not np.all(np.isfinite(warp)):
        raise ValueError("absolute warp must be finite with shape (x, y, z, 3)")
    if (
        voxel_sizes.shape != (3,)
        or not np.all(np.isfinite(voxel_sizes))
        or np.any(voxel_sizes <= 0.0)
    ):
        raise ValueError("three voxel sizes must be positive and finite")
    if (
        not np.isfinite(minimum_jacobian)
        or not np.isfinite(maximum_jacobian)
        or minimum_jacobian >= maximum_jacobian
    ):
        raise ValueError("Jacobian bounds must be finite and increasing")
    result = warp.copy()
    _, statistics = _jacobian_check(
        result, voxel_sizes, minimum_jacobian, maximum_jacobian, False
    )
    iteration = 1
    while iteration < 10 and (statistics[2] > 0.5 or statistics[3] > 0.5):
        iteration += 1
        means = np.asarray(
            [np.mean(result[..., component], dtype=np.float64) for component in range(3)],
            dtype=np.float32,
        )
        gradient = _gradient_calculation(result, voxel_sizes)
        jacobians, _ = _jacobian_check(
            result, voxel_sizes, minimum_jacobian, maximum_jacobian, True
        )
        _limit_gradient(gradient, jacobians, minimum_jacobian, maximum_jacobian)
        result = _integrate_gradient_field(gradient, voxel_sizes, means)
        _, statistics = _jacobian_check(
            result, voxel_sizes, minimum_jacobian, maximum_jacobian, False
        )
    return result


def fsl_corner_jacobian_range(
    absolute_warp: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
) -> tuple[float, float]:
    """Return the minimum and maximum corner-discretized Jacobians used by topology code."""

    warp = np.asarray(absolute_warp, dtype=np.float32)
    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float32)
    if warp.ndim != 4 or warp.shape[-1] != 3:
        raise ValueError("absolute warp must have shape (x, y, z, 3)")
    _, statistics = _jacobian_check(warp, voxel_sizes, -np.inf, np.inf, False)
    return float(statistics[0]), float(statistics[1])
