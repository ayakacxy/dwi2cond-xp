"""Source-faithful fixed-subset primitives for FSL EDDY volume correction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from .brain_mask import robust_intensity_limits
from .topup import (
    _cubic_derivative_weight,
    _cubic_weight,
    _fsl_affine_inverse_4x4,
    _fsl_matrix_multiply_4x4,
    _fsl_periodic_cubic_coefficients,
    _topup_matrix_to_movement_parameters,
    _topup_movement_matrix,
)


GP_VOXEL_SELECTION_ATTEMPT_LIMIT = 100_000_000


@dataclass(frozen=True)
class EddySliceStatistics:
    """Hold the per-volume, per-slice residual statistics used by FSL EDDY."""

    mean_difference: np.ndarray
    mean_squared_difference: np.ndarray
    voxel_count: np.ndarray


@dataclass(frozen=True)
class EddyOutlierResult:
    """Hold FSL-style slice outlier flags and normalized residual statistics."""

    outlier_map: np.ndarray
    n_standard_deviations: np.ndarray
    n_squared_standard_deviations: np.ndarray


@dataclass(frozen=True)
class EddySphericalGP:
    """Hold a fitted single-shell spherical GP and its prediction weights."""

    hyperparameters: np.ndarray
    covariance: np.ndarray
    prediction_weights: np.ndarray


@dataclass(frozen=True)
class EddyHyperparameterResult:
    """Hold FSL Nelder-Mead GP hyperparameters and convergence evidence."""

    hyperparameters: np.ndarray
    cost: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class EddyTransformResult:
    """Hold one FSL model-to-scan transform and its reusable derivatives."""

    values: np.ndarray
    mask: np.ndarray
    jacobian: np.ndarray
    coordinates: np.ndarray
    spatial_gradient: np.ndarray


@dataclass(frozen=True)
class PreparedEddySusceptibilityField:
    """Hold a TOPUP field and its reusable mirror-spline coefficients."""

    values_hz: np.ndarray
    interpolation_coefficients: np.ndarray


@dataclass(frozen=True)
class EddyDerivativeResult:
    """Hold one base transform and FSL's 15 volume-to-volume derivatives."""

    base: EddyTransformResult
    derivatives: np.ndarray


@njit(cache=True, parallel=True, nogil=True)
def _coordinate_gradient_dot_fsl_order(
    coordinate_difference: np.ndarray, spatial_gradient: np.ndarray
) -> np.ndarray:
    """Apply ``ImageCoordinates::operator*`` with float z/y/x storage."""

    nx, ny, nz, _components = coordinate_difference.shape
    result = np.empty((nx, ny, nz), dtype=np.float32)
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                value = np.float32(
                    coordinate_difference[x, y, z, 0] * spatial_gradient[x, y, z, 0]
                    + coordinate_difference[x, y, z, 1] * spatial_gradient[x, y, z, 1]
                )
                result[x, y, z] = np.float32(
                    value
                    + coordinate_difference[x, y, z, 2] * spatial_gradient[x, y, z, 2]
                )
    return result


@dataclass(frozen=True)
class EddyGaussNewtonResult:
    """Hold FSL's per-volume normal equations, update, and pre-update MSS."""

    xtx: np.ndarray
    xty: np.ndarray
    update: np.ndarray
    mean_squared_error: float


@dataclass(frozen=True)
class EddyIterationResult:
    """Hold one source-ordered DWI registration iteration."""

    hyperparameters: np.ndarray
    updates: np.ndarray
    mean_squared_errors: np.ndarray
    accepted: np.ndarray
    joint_mask: np.ndarray


@dataclass(frozen=True)
class EddyDwiRegistrationResult:
    """Hold the fixed-subset DWI registration trajectory and final state."""

    movement_parameters: np.ndarray
    quadratic_ec_parameters: np.ndarray
    unwarped_scans: np.ndarray
    rotated_bvecs: np.ndarray
    iterations: tuple[EddyIterationResult, ...]
    joint_mask: np.ndarray
    outlier_map: np.ndarray | None = None
    outlier_free_scans: np.ndarray | None = None


@dataclass(frozen=True)
class EddyB0RegistrationResult:
    """Hold FSL's fixed movement-only b0 registration state."""

    movement_parameters: np.ndarray
    unwarped_scans: np.ndarray
    iterations: tuple[EddyIterationResult, ...]
    joint_mask: np.ndarray


@dataclass(frozen=True)
class EddyRunResult:
    """Hold the complete no-TOPUP SimNIBS EDDY fixed-subset result."""

    corrected_scans: np.ndarray
    movement_parameters: np.ndarray
    quadratic_ec_parameters: np.ndarray
    rotated_bvecs: np.ndarray
    outlier_map: np.ndarray
    outlier_free_scans: np.ndarray | None
    scale_factor: float
    shell_pe_translation_mm: float
    shell_alignment_parameters: np.ndarray
    b0_registration: EddyB0RegistrationResult
    dwi_registration: EddyDwiRegistrationResult


def _glibc_rand_stream(seed: int):
    """Yield the glibc ``rand`` stream used by the Linux FSL reference."""

    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must be a uint32 value")
    effective_seed = 1 if seed == 0 else seed
    state = [effective_seed]
    for _index in range(1, 31):
        state.append((16807 * state[-1]) % 2147483647)
    state.extend((state[0], state[1], state[2]))
    for index in range(34, 344):
        state.append((state[index - 31] + state[index - 3]) & 0xFFFFFFFF)
    index = 344
    while True:
        state.append((state[index - 31] + state[index - 3]) & 0xFFFFFFFF)
        yield state[index] >> 1
        index += 1


def _glibc_rand_values(seed: int, count: int) -> np.ndarray:
    """Return a finite prefix of the Linux FSL reference random stream."""

    if count < 0:
        raise ValueError("random value count must be nonnegative")
    return np.fromiter(
        islice(_glibc_rand_stream(seed), count), dtype=np.int64, count=count
    )


def select_fsl_gp_voxels(
    scans: np.ndarray,
    mask: np.ndarray,
    *,
    number_of_voxels: int = 1000,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce FSL ``DataSelector`` coordinate rejection with glibc ``rand``."""

    values = np.asarray(scans)
    mask_values = np.asarray(mask)
    if values.ndim != 4 or mask_values.shape != values.shape[:3]:
        raise ValueError("scans and mask must have shapes (X, Y, Z, N) and (X, Y, Z)")
    if number_of_voxels < 1 or np.count_nonzero(mask_values) < number_of_voxels:
        raise ValueError("number_of_voxels must fit within the nonzero mask")
    coordinates: list[tuple[int, int, int]] = []
    selected: set[tuple[int, int, int]] = set()
    random_values = _glibc_rand_stream(random_seed)
    attempts = 0
    while len(coordinates) < number_of_voxels:
        if attempts == GP_VOXEL_SELECTION_ATTEMPT_LIMIT:
            raise RuntimeError("unable to select the requested unique GP voxels")
        coordinate = tuple(
            int(next(random_values) % values.shape[axis]) for axis in range(3)
        )
        attempts += 1
        if mask_values[coordinate] != 0 and coordinate not in selected:
            selected.add(coordinate)
            coordinates.append(coordinate)
    coordinate_array = np.asarray(coordinates, dtype=np.int64)
    data = values[
        coordinate_array[:, 0], coordinate_array[:, 1], coordinate_array[:, 2], :
    ].astype(np.float64, copy=False)
    return data, coordinate_array


def spherical_gp_cv_cost(
    mean_corrected_data: np.ndarray,
    bvecs: np.ndarray,
    hyperparameters: np.ndarray,
) -> float:
    """Evaluate FSL's leave-one-out CV hyperparameter cost."""

    data = np.asarray(mean_corrected_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != np.asarray(bvecs).shape[1]:
        raise ValueError("GP data must have shape (V, N) matching bvecs")
    angles = _spherical_gp_angles_fsl_order(bvecs)
    covariance, _signal = _spherical_gp_covariance_from_angles_fsl_order(
        angles, hyperparameters
    )
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return float(np.finfo(np.float64).max)
    inverse_cholesky = np.linalg.inv(cholesky)
    inverse_covariance = inverse_cholesky.T @ inverse_cholesky
    qn = data @ inverse_covariance.T
    squared = np.sum(qn * qn, axis=0, dtype=np.float64)
    diagonal = np.diag(inverse_covariance)
    return float(
        np.sum(squared / (diagonal * diagonal), dtype=np.float64) / data.shape[1]
    )


def _fsl_zero_cost_difference(old: float, new: float, tolerance: float) -> bool:
    """Return the fractional cost convergence test from FSL ``nonlin``."""

    return 2.0 * abs(old - new) <= tolerance * (
        abs(old) + abs(new) + np.finfo(np.float64).eps
    )


def estimate_spherical_gp_hyperparameters(
    selected_shell_data: np.ndarray,
    bvecs: np.ndarray,
    *,
    error_variance_fudge_factor: float = 10.0,
    maximum_iterations: int = 500,
    fractional_cost_tolerance: float = 1.0e-8,
) -> EddyHyperparameterResult:
    """Run FSL's CV-cost, unit-simplex Nelder-Mead hyperparameter estimator."""

    data = np.asarray(selected_shell_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2 or not np.all(np.isfinite(data)):
        raise ValueError(
            "selected shell data must be a finite matrix with at least two scans"
        )
    if np.asarray(bvecs).shape != (3, data.shape[1]):
        raise ValueError("bvecs must have shape (3, N) matching selected shell data")
    if error_variance_fudge_factor < 1.0 or maximum_iterations < 1:
        raise ValueError("fudge factor and maximum iterations must be at least one")
    centered = data.copy()
    centered -= np.mean(centered, axis=1, keepdims=True)
    variance = float(np.mean(np.var(centered, axis=1, ddof=1), dtype=np.float64))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("selected shell data must have positive directional variance")
    initial = np.array((0.9 * np.log(variance / 3.0), 0.45, np.log(variance / 3.0)))
    angles = _spherical_gp_angles_fsl_order(bvecs)

    def cost(parameters: np.ndarray) -> float:
        covariance, _signal = _spherical_gp_covariance_from_angles_fsl_order(
            angles, parameters
        )
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return float(np.finfo(np.float64).max)
        inverse_cholesky = np.linalg.inv(cholesky)
        inverse_covariance = inverse_cholesky.T @ inverse_cholesky
        qn = centered @ inverse_covariance.T
        squared = np.sum(qn * qn, axis=0, dtype=np.float64)
        diagonal = np.diag(inverse_covariance)
        return float(
            np.sum(squared / (diagonal * diagonal), dtype=np.float64)
            / centered.shape[1]
        )

    dimension = initial.size
    simplex = np.repeat(initial[None, :], dimension + 1, axis=0)
    for index in range(dimension):
        simplex[index + 1, index] += 1.0
    costs = np.asarray([cost(point) for point in simplex], dtype=np.float64)
    converged = False
    iterations = 0
    for iterations in range(maximum_iterations):
        best = int(np.argmin(costs))
        worst = int(np.argmax(costs))
        remaining = [index for index in range(dimension + 1) if index != worst]
        second_worst = max(remaining, key=lambda index: costs[index])
        if _fsl_zero_cost_difference(
            float(costs[worst]), float(costs[best]), fractional_cost_tolerance
        ):
            converged = True
            break
        centroid = np.sum(simplex[remaining], axis=0, dtype=np.float64) / dimension
        reflected = 2.0 * centroid - simplex[worst]
        reflected_cost = cost(reflected)
        if reflected_cost < costs[worst]:
            simplex[worst] = reflected
            costs[worst] = reflected_cost
        if reflected_cost <= costs[best]:
            expanded = 2.0 * simplex[worst] - centroid
            expanded_cost = cost(expanded)
            if expanded_cost < costs[worst]:
                simplex[worst] = expanded
                costs[worst] = expanded_cost
        elif reflected_cost >= costs[second_worst]:
            old_worst_cost = costs[worst]
            contracted = 0.5 * (simplex[worst] + centroid)
            contracted_cost = cost(contracted)
            if contracted_cost < costs[worst]:
                simplex[worst] = contracted
                costs[worst] = contracted_cost
            if contracted_cost >= old_worst_cost:
                for index in range(dimension + 1):
                    if index != best:
                        simplex[index] = 0.5 * (simplex[index] + simplex[best])
                        costs[index] = cost(simplex[index])
    best = int(np.argmin(costs))
    result = simplex[best].copy()
    result[2] += np.log(error_variance_fudge_factor)
    return EddyHyperparameterResult(
        result, float(costs[best]), iterations + (0 if converged else 1), converged
    )


def spherical_gp_covariance(
    bvecs: np.ndarray, hyperparameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build FSL's single-shell antipodal spherical covariance and signal kernel."""

    parameters = np.asarray(hyperparameters, dtype=np.float64)
    angles = _spherical_gp_angles_fsl_order(bvecs)
    if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
        raise ValueError(
            "single-shell spherical GP requires three finite hyperparameters"
        )
    return _spherical_gp_covariance_from_angles_fsl_order(angles, parameters)


def _spherical_gp_angles_fsl_order(bvecs: np.ndarray) -> np.ndarray:
    """Build FSL's ordered antipodal angle matrix once per b-vector table."""

    vectors = np.asarray(bvecs, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != 3 or vectors.shape[1] < 2:
        raise ValueError("single-shell bvecs must have shape (3, N) with N at least 2")
    unit = np.empty_like(vectors)
    for volume in range(vectors.shape[1]):
        squared_norm = (
            vectors[0, volume] * vectors[0, volume]
            + vectors[2, volume] * vectors[2, volume]
        ) + vectors[1, volume] * vectors[1, volume]
        norm = np.sqrt(squared_norm)
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError("single-shell b-vectors must be finite and nonzero")
        unit[:, volume] = vectors[:, volume] / norm
    angles = np.empty((vectors.shape[1], vectors.shape[1]), dtype=np.float64)
    for column in range(vectors.shape[1]):
        for row in range(column, vectors.shape[1]):
            dot_product = (
                unit[0, row] * unit[0, column]
                + unit[2, row] * unit[2, column]
            ) + unit[1, row] * unit[1, column]
            angle = np.arccos(min(1.0, abs(dot_product)))
            angles[row, column] = angle
            angles[column, row] = angle
    return angles


def _spherical_gp_covariance_from_angles_fsl_order(
    angles: np.ndarray, hyperparameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate FSL's covariance while reusing one invariant angle matrix."""

    parameters = np.asarray(hyperparameters, dtype=np.float64)
    if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
        raise ValueError(
            "single-shell spherical GP requires three finite hyperparameters"
        )
    signal_variance = np.exp(parameters[0])
    angular_scale = np.exp(parameters[1])
    ratio = angles / angular_scale
    signal = np.where(
        angles < angular_scale,
        signal_variance * (1.0 - 1.5 * ratio + 0.5 * ratio**3),
        0.0,
    )
    covariance = signal.copy()
    covariance.flat[:: covariance.shape[0] + 1] += np.exp(parameters[2])
    return covariance, signal


def fit_spherical_gp_weights(
    bvecs: np.ndarray,
    hyperparameters: np.ndarray,
    *,
    exclude_target: bool = False,
) -> EddySphericalGP:
    """Compute FSL ``PredVec`` weights for a fixed single-shell GP."""

    covariance, signal = spherical_gp_covariance(bvecs, hyperparameters)
    count = covariance.shape[0]
    if not exclude_target:
        weights = np.linalg.solve(covariance, signal.T).T
    else:
        weights = np.zeros_like(covariance)
        all_indices = np.arange(count)
        for target in range(count):
            retained = all_indices != target
            weights[target, retained] = np.linalg.solve(
                covariance[np.ix_(retained, retained)], signal[target, retained]
            )
    return EddySphericalGP(
        np.asarray(hyperparameters, dtype=np.float64).copy(), covariance, weights
    )


@njit(cache=True, parallel=True)
def _spherical_gp_predict_kernel(
    scans: np.ndarray, weights: np.ndarray, exclude_target: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FSL shell mean correction and ordered double prediction products."""

    nx, ny, nz, volume_count = scans.shape
    voxel_count = nx * ny * nz
    flattened = scans.reshape((voxel_count, volume_count))
    means = np.empty(voxel_count, dtype=np.float32)
    centered = np.empty_like(flattened)
    output = np.empty_like(flattened)
    for voxel in prange(voxel_count):
        shell_mean = np.float32(0.0)
        for volume in range(volume_count):
            shell_mean = np.float32(shell_mean + flattened[voxel, volume])
        shell_mean = np.float32(shell_mean / volume_count)
        means[voxel] = shell_mean
        for volume in range(volume_count):
            centered[voxel, volume] = np.float32(flattened[voxel, volume] - shell_mean)
        for target in range(volume_count):
            prediction = 0.0
            valid = True
            for volume in range(volume_count):
                if exclude_target and volume == target:
                    continue
                value = centered[voxel, volume]
                if value == 0.0:
                    valid = False
                prediction += weights[target, volume] * float(value)
            if valid:
                output[voxel, target] = np.float32(shell_mean + np.float32(prediction))
            else:
                output[voxel, target] = np.float32(0.0)
    return output.reshape(scans.shape), means.reshape(scans.shape[:3])


def predict_spherical_gp(
    scans: np.ndarray,
    model: EddySphericalGP,
    *,
    exclude_target: bool = False,
) -> np.ndarray:
    """Predict all single-shell volumes with FSL's mean-corrected GP ordering."""

    values = np.asarray(scans)
    if values.ndim != 4 or values.shape[3] != model.prediction_weights.shape[0]:
        raise ValueError("scans must have shape (X, Y, Z, N) matching the GP")
    if values.dtype != np.float32:
        values = values.astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("GP input scans must be finite")
    predictions, _means = _spherical_gp_predict_kernel(
        np.ascontiguousarray(values),
        np.ascontiguousarray(model.prediction_weights),
        exclude_target,
    )
    return predictions


@njit(cache=True, parallel=True, nogil=True)
def _quadratic_eddy_field_fsl_order(
    shape: tuple[int, int, int],
    voxel_sizes_mm: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate and store FSL's quadratic field one float voxel at a time."""

    nx, ny, nz = shape
    result = np.empty(shape, dtype=np.float32)
    offset = np.float32(coefficients[9]) if coefficients.size == 10 else np.float32(0.0)
    for k in prange(nz):
        z = voxel_sizes_mm[2] * (k - (nz - 1) / 2.0)
        z_component = coefficients[2] * z
        z_squared_component = coefficients[5] * z * z
        for j in range(ny):
            y = voxel_sizes_mm[1] * (j - (ny - 1) / 2.0)
            y_component = coefficients[1] * y
            y_squared_component = coefficients[4] * y * y
            yz_component = coefficients[8] * y * z
            for i in range(nx):
                x = voxel_sizes_mm[0] * (i - (nx - 1) / 2.0)
                x_component = coefficients[0] * x
                x_squared_component = coefficients[3] * x * x
                xy_component = coefficients[6] * x * y
                xz_component = coefficients[7] * x * z
                total = x_component + y_component
                total += z_component
                total += x_squared_component
                total += y_squared_component
                total += z_squared_component
                total += xy_component
                total += xz_component
                total += yz_component
                result[i, j, k] = np.float32(np.float64(offset) + total)
    return result


def quadratic_eddy_field(
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate FSL's quadratic EC field in physical centered coordinates."""

    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("shape must contain three positive dimensions")
    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if (
        voxel_sizes.shape != (3,)
        or not np.all(np.isfinite(voxel_sizes))
        or np.any(voxel_sizes <= 0.0)
    ):
        raise ValueError("voxel_sizes_mm must contain three positive finite values")
    coefficients = np.asarray(parameters, dtype=np.float64)
    if coefficients.shape not in ((9,), (10,)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("quadratic EC parameters must contain 9 or 10 finite values")
    return _quadratic_eddy_field_fsl_order(shape, voxel_sizes, coefficients)


@njit(cache=True, parallel=True, nogil=True)
def _invert_displacement_lines(
    forward: np.ndarray, input_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert PE-first displacement lines with FSL's linear crossing formula."""

    line_length, second_size, third_size = forward.shape
    inverse = np.empty_like(forward)
    output_mask = np.zeros(forward.shape, dtype=np.uint8)
    sentinel = np.finfo(np.float32).max
    for line in prange(second_size * third_size):
        second = line // third_size
        third = line - second * third_size
        previous = 0
        for output_index in range(line_length):
            source_index = previous
            while (
                source_index < line_length
                and forward[source_index, second, third] + source_index < output_index
            ):
                source_index += 1
            if source_index > 0 and source_index < line_length:
                lower = forward[source_index - 1, second, third]
                upper = forward[source_index, second, third]
                inverse[output_index, second, third] = np.float32(
                    source_index
                    - output_index
                    - 1.0
                    + np.float32(output_index + 1 - source_index - lower)
                    / np.float32(upper + 1.0 - lower)
                )
                output_mask[output_index, second, third] = input_mask[
                    source_index - 1, second, third
                ]
            else:
                inverse[output_index, second, third] = sentinel
            previous = max(0, source_index - 1)
        source_index = 0
        while (
            source_index < line_length - 1
            and inverse[source_index, second, third] == sentinel
        ):
            source_index += 1
        while source_index > 0:
            inverse[source_index - 1, second, third] = inverse[
                source_index, second, third
            ]
            source_index -= 1
        source_index = line_length - 1
        while source_index > 0 and inverse[source_index, second, third] == sentinel:
            source_index -= 1
        while source_index < line_length - 1:
            inverse[source_index + 1, second, third] = inverse[
                source_index, second, third
            ]
            source_index += 1
    return inverse, output_mask


def invert_eddy_displacement(
    forward_displacement_voxels: np.ndarray,
    phase_encoding_axis: int,
    input_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce FSL ``Invert1DDisplacementField`` for one PE axis."""

    forward = np.asarray(forward_displacement_voxels, dtype=np.float32)
    mask = np.asarray(input_mask, dtype=np.uint8)
    if forward.ndim != 3 or mask.shape != forward.shape:
        raise ValueError("forward displacement and mask must share a 3D grid")
    if phase_encoding_axis not in (0, 1, 2):
        raise ValueError("phase_encoding_axis must be 0, 1, or 2")
    if not np.all(np.isfinite(forward)):
        raise ValueError("forward displacement must be finite")
    forward_pe = np.ascontiguousarray(np.moveaxis(forward, phase_encoding_axis, 0))
    mask_pe = np.ascontiguousarray(np.moveaxis(mask, phase_encoding_axis, 0))
    inverse_pe, output_mask_pe = _invert_displacement_lines(forward_pe, mask_pe)
    return (
        np.moveaxis(inverse_pe, 0, phase_encoding_axis),
        np.moveaxis(output_mask_pe, 0, phase_encoding_axis),
    )


@njit(cache=True, parallel=True, nogil=True)
def _mirror_cubic_deconvolve_lines(values: np.ndarray) -> np.ndarray:
    """Run one axis of FSL Splinterpolator's float coefficient sweep."""

    line_length, second_size, third_size = values.shape
    output = values.copy()
    if line_length == 1:
        return output
    pole = np.sqrt(3.0) - 2.0
    precision = 1.0e-8
    sweep_count = int(np.log(precision) / np.log(abs(pole)) + 1.5)
    sweep_count = min(sweep_count, line_length)
    for line_index in prange(second_size * third_size):
        second = line_index // third_size
        third = line_index - second * third_size
        coefficients = np.empty(line_length, dtype=np.float64)
        for index in range(line_length):
            coefficients[index] = values[index, second, third]
        original_last = coefficients[-1]
        initial = coefficients[0]
        power = pole
        for index in range(1, sweep_count):
            initial += power * coefficients[index]
            power *= pole
        coefficients[0] = initial
        for index in range(1, line_length):
            coefficients[index] += pole * coefficients[index - 1]
        coefficients[-1] = (
            -pole / (1.0 - pole * pole) * (2.0 * coefficients[-1] - original_last)
        )
        for index in range(line_length - 2, -1, -1):
            coefficients[index] = pole * (coefficients[index + 1] - coefficients[index])
        for index in range(line_length):
            output[index, second, third] = np.float32(coefficients[index] * 6.0)
    return output


@njit(cache=True, parallel=True, nogil=True)
def _mirror_cubic_derivative_from_coefficients(
    coefficients: np.ndarray, axis: int
) -> np.ndarray:
    """Evaluate FSL's integer-grid 3D derivative from float coefficients."""

    nx, ny, nz = coefficients.shape
    derivative = np.empty(coefficients.shape, dtype=np.float32)
    weights = (0.166666666666667, 0.666666666666667, 0.166666666666667)
    derivative_weights = (-0.5, 0.0, 0.5)
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                value = 0.0
                for offset_z in range(3):
                    index_z = z - 1 + offset_z
                    if index_z < 0:
                        index_z = 0
                    elif index_z >= nz:
                        index_z = nz - 1
                    weight_z = weights[offset_z]
                    derivative_weight_one = (
                        derivative_weights[offset_z] if axis == 2 else weight_z
                    )
                    for offset_y in range(3):
                        index_y = y - 1 + offset_y
                        if index_y < 0:
                            index_y = 0
                        elif index_y >= ny:
                            index_y = ny - 1
                        weight_y = weights[offset_y]
                        derivative_weight_y = derivative_weights[offset_y]
                        derivative_weight_two = derivative_weight_one * (
                            derivative_weight_y if axis == 1 else weight_y
                        )
                        for offset_x in range(3):
                            index_x = x - 1 + offset_x
                            if index_x < 0:
                                index_x = 0
                            elif index_x >= nx:
                                index_x = nx - 1
                            weight_x = weights[offset_x]
                            derivative_weight_x = derivative_weights[offset_x]
                            coefficient = float(coefficients[index_x, index_y, index_z])
                            if axis == 0:
                                add = (
                                    coefficient
                                    * derivative_weight_x
                                    * derivative_weight_two
                                )
                            else:
                                add = coefficient * weight_x * derivative_weight_two
                            value += add
                derivative[x, y, z] = np.float32(value)
    return derivative


def _mirror_cubic_grid_derivative(values: np.ndarray, axis: int) -> np.ndarray:
    """Evaluate FSL's mirror cubic derivative at every integer grid point."""

    coefficients = np.ascontiguousarray(values, dtype=np.float32)
    for coefficient_axis in range(3):
        moved = np.ascontiguousarray(np.moveaxis(coefficients, coefficient_axis, 0))
        coefficients = np.ascontiguousarray(
            np.moveaxis(_mirror_cubic_deconvolve_lines(moved), 0, coefficient_axis)
        )
    return _mirror_cubic_derivative_from_coefficients(coefficients, axis)


def _fsl_zeropad_cubic_coefficients(values: np.ndarray) -> np.ndarray:
    """Create NEWIMAGE cubic coefficients for the default zero extrapolation."""

    coefficients = np.ascontiguousarray(values, dtype=np.float32)
    for coefficient_axis in range(3):
        moved = np.ascontiguousarray(
            np.moveaxis(coefficients, coefficient_axis, 0)
        )
        coefficients = np.ascontiguousarray(
            np.moveaxis(
                _mirror_cubic_deconvolve_lines(moved), 0, coefficient_axis
            )
        )
    return coefficients


def prepare_eddy_susceptibility_field(
    field_hz: np.ndarray,
) -> PreparedEddySusceptibilityField:
    """Prepare one TOPUP field for repeated EDDY transforms."""

    values = np.asarray(field_hz, dtype=np.float32)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("susceptibility field must be a finite 3D array")
    values = np.ascontiguousarray(values)
    return PreparedEddySusceptibilityField(
        values, _fsl_zeropad_cubic_coefficients(values)
    )


def _prepared_eddy_susceptibility_field(
    field: np.ndarray | PreparedEddySusceptibilityField | None,
    shape: tuple[int, int, int],
) -> PreparedEddySusceptibilityField | None:
    """Validate or prepare an optional EDDY susceptibility field."""

    if field is None:
        return None
    prepared = (
        field
        if isinstance(field, PreparedEddySusceptibilityField)
        else prepare_eddy_susceptibility_field(field)
    )
    if prepared.values_hz.shape != shape:
        raise ValueError("susceptibility field must match the scan grid")
    return prepared


@njit(cache=True, parallel=True, nogil=True)
def _sample_fsl_zeropad_cubic_translation(
    coefficients: np.ndarray,
    joint_mask: np.ndarray,
    translation_voxels: float,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply NEWIMAGE spline/linear affine translation with zero extrapolation."""

    nx, ny, nz = coefficients.shape
    output = np.zeros(coefficients.shape, dtype=np.float32)
    output_mask = np.zeros(coefficients.shape, dtype=np.float32)
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                coordinate_x = float(x)
                coordinate_y = float(y)
                coordinate_z = float(z)
                if axis == 0:
                    coordinate_x -= translation_voxels
                else:
                    coordinate_y -= translation_voxels
                if (
                    coordinate_x < 0.0
                    or coordinate_x > nx - 1
                    or coordinate_y < 0.0
                    or coordinate_y > ny - 1
                    or coordinate_z < 0.0
                    or coordinate_z > nz - 1
                ):
                    continue
                rounded_x = int(coordinate_x + 0.5)
                rounded_y = int(coordinate_y + 0.5)
                rounded_z = int(coordinate_z + 0.5)
                start_x = rounded_x - (1 if rounded_x < coordinate_x else 2)
                start_y = rounded_y - (1 if rounded_y < coordinate_y else 2)
                start_z = rounded_z - (1 if rounded_z < coordinate_z else 2)
                value = 0.0
                for offset_z in range(4):
                    index_z = start_z + offset_z
                    weight_z = _cubic_weight(coordinate_z - index_z)
                    if index_z < 0:
                        index_z = -1 - index_z
                    elif index_z >= nz:
                        index_z = 2 * nz - 1 - index_z
                    for offset_y in range(4):
                        index_y = start_y + offset_y
                        weight_y = _cubic_weight(coordinate_y - index_y)
                        if index_y < 0:
                            index_y = -1 - index_y
                        elif index_y >= ny:
                            index_y = 2 * ny - 1 - index_y
                        weight_zy = weight_z * weight_y
                        for offset_x in range(4):
                            index_x = start_x + offset_x
                            weight_x = _cubic_weight(coordinate_x - index_x)
                            if index_x < 0:
                                index_x = -1 - index_x
                            elif index_x >= nx:
                                index_x = 2 * nx - 1 - index_x
                            value += (
                                coefficients[index_x, index_y, index_z]
                                * weight_x
                                * weight_zy
                            )
                output[x, y, z] = np.float32(value)
                if axis == 0:
                    low = int(np.floor(coordinate_x))
                    high = min(low + 1, nx - 1)
                    remainder = np.float32(coordinate_x - low)
                    output_mask[x, y, z] = np.float32(
                        joint_mask[low, y, z] * (np.float32(1.0) - remainder)
                        + joint_mask[high, y, z] * remainder
                    )
                else:
                    low = int(np.floor(coordinate_y))
                    high = min(low + 1, ny - 1)
                    remainder = np.float32(coordinate_y - low)
                    output_mask[x, y, z] = np.float32(
                        joint_mask[x, low, z] * (np.float32(1.0) - remainder)
                        + joint_mask[x, high, z] * remainder
                    )
    return output, output_mask


@njit(cache=True, parallel=True, nogil=True)
def _sample_fsl_zeropad_cubic_affine(
    coefficients: np.ndarray,
    joint_mask: np.ndarray,
    pull_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply NEWIMAGE spline/linear affine sampling with zero extrapolation."""

    nx, ny, nz = coefficients.shape
    output = np.zeros(coefficients.shape, dtype=np.float32)
    output_mask = np.zeros(coefficients.shape, dtype=np.float32)
    matrix = pull_matrix.astype(np.float32)
    for z in prange(nz):
        float_z = np.float32(z)
        for x in range(nx):
            float_x = np.float32(x)
            coordinate_x = np.float32(
                np.float32(matrix[0, 0] * float_x + matrix[0, 2] * float_z)
                + matrix[0, 3]
            )
            coordinate_y = np.float32(
                np.float32(matrix[1, 0] * float_x + matrix[1, 2] * float_z)
                + matrix[1, 3]
            )
            coordinate_z = np.float32(
                np.float32(matrix[2, 0] * float_x + matrix[2, 2] * float_z)
                + matrix[2, 3]
            )
            for y in range(ny):
                coordinate_valid = (
                    coordinate_x + np.float32(1.0e-8) >= 0.0
                    and coordinate_x <= np.float32(nx - 1) + np.float32(1.0e-8)
                    and coordinate_y + np.float32(1.0e-8) >= 0.0
                    and coordinate_y <= np.float32(ny - 1) + np.float32(1.0e-8)
                    and coordinate_z + np.float32(1.0e-8) >= 0.0
                    and coordinate_z <= np.float32(nz - 1) + np.float32(1.0e-8)
                )
                rounded_x = int(coordinate_x + np.float32(0.5))
                rounded_y = int(coordinate_y + np.float32(0.5))
                rounded_z = int(coordinate_z + np.float32(0.5))
                start_x = rounded_x - (1 if rounded_x < coordinate_x else 2)
                start_y = rounded_y - (1 if rounded_y < coordinate_y else 2)
                start_z = rounded_z - (1 if rounded_z < coordinate_z else 2)
                value = 0.0
                for offset_z in range(4):
                    index_z = start_z + offset_z
                    weight_z = _cubic_weight(coordinate_z - index_z)
                    index_z %= 2 * nz
                    if index_z >= nz:
                        index_z = 2 * nz - 1 - index_z
                    for offset_y in range(4):
                        index_y = start_y + offset_y
                        weight_y = _cubic_weight(coordinate_y - index_y)
                        index_y %= 2 * ny
                        if index_y >= ny:
                            index_y = 2 * ny - 1 - index_y
                        weight_zy = weight_z * weight_y
                        for offset_x in range(4):
                            index_x = start_x + offset_x
                            weight_x = _cubic_weight(coordinate_x - index_x)
                            index_x %= 2 * nx
                            if index_x >= nx:
                                index_x = 2 * nx - 1 - index_x
                            value += (
                                coefficients[index_x, index_y, index_z]
                                * weight_x
                                * weight_zy
                            )
                output[x, y, z] = np.float32(value)
                mask_coordinate_x = min(max(coordinate_x, np.float32(0.0)), nx - 1)
                mask_coordinate_y = min(max(coordinate_y, np.float32(0.0)), ny - 1)
                mask_coordinate_z = min(max(coordinate_z, np.float32(0.0)), nz - 1)
                low_x = int(np.floor(mask_coordinate_x))
                low_y = int(np.floor(mask_coordinate_y))
                low_z = int(np.floor(mask_coordinate_z))
                high_x = min(low_x + 1, nx - 1)
                high_y = min(low_y + 1, ny - 1)
                high_z = min(low_z + 1, nz - 1)
                remainder_x = np.float32(mask_coordinate_x - low_x)
                remainder_y = np.float32(mask_coordinate_y - low_y)
                remainder_z = np.float32(mask_coordinate_z - low_z)
                low_xy = np.float32(
                    joint_mask[low_x, low_y, low_z]
                    * (np.float32(1.0) - remainder_x)
                    + joint_mask[high_x, low_y, low_z] * remainder_x
                )
                high_xy = np.float32(
                    joint_mask[low_x, high_y, low_z]
                    * (np.float32(1.0) - remainder_x)
                    + joint_mask[high_x, high_y, low_z] * remainder_x
                )
                low_xyz = np.float32(
                    low_xy * (np.float32(1.0) - remainder_y)
                    + high_xy * remainder_y
                )
                low_xy = np.float32(
                    joint_mask[low_x, low_y, high_z]
                    * (np.float32(1.0) - remainder_x)
                    + joint_mask[high_x, low_y, high_z] * remainder_x
                )
                high_xy = np.float32(
                    joint_mask[low_x, high_y, high_z]
                    * (np.float32(1.0) - remainder_x)
                    + joint_mask[high_x, high_y, high_z] * remainder_x
                )
                high_xyz = np.float32(
                    low_xy * (np.float32(1.0) - remainder_y)
                    + high_xy * remainder_y
                )
                output_mask[x, y, z] = np.float32(
                    (
                        low_xyz * (np.float32(1.0) - remainder_z)
                        + high_xyz * remainder_z
                    )
                    * np.float32(coordinate_valid)
                )
                coordinate_x = np.float32(coordinate_x + matrix[0, 1])
                coordinate_y = np.float32(coordinate_y + matrix[1, 1])
                coordinate_z = np.float32(coordinate_z + matrix[2, 1])
    return output, output_mask


@njit(cache=True, parallel=True, nogil=True)
def _sample_eddy_cubic_warp_fsl_order(
    coefficients: np.ndarray,
    displacement_mm: np.ndarray,
    phase_encoding_axis: int,
    affine_to_mm: np.ndarray,
    mm_to_source_voxels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Apply ``raw_general_transform`` in its float z/y/x loop order."""

    source_nx, source_ny, source_nz = coefficients.shape
    nx, ny, nz = displacement_mm.shape
    affine = affine_to_mm.astype(np.float32)
    source_map = mm_to_source_voxels.astype(np.float32)
    values = np.empty((nx, ny, nz), dtype=np.float32)
    derivative_x = np.empty_like(values)
    derivative_y = np.empty_like(values)
    derivative_z = np.empty_like(values)
    mask = np.empty((nx, ny, nz), dtype=np.uint8)
    coordinates = np.empty((nx, ny, nz, 3), dtype=np.float32)
    for z in prange(nz):
        float_z = np.float32(z)
        for y in range(ny):
            float_y = np.float32(y)
            for x in range(nx):
                float_x = np.float32(x)
                mm_x = np.float32(
                    np.float32(
                        np.float32(
                            affine[0, 0] * float_x + affine[0, 1] * float_y
                        )
                        + affine[0, 2] * float_z
                    )
                    + affine[0, 3]
                )
                mm_y = np.float32(
                    np.float32(
                        np.float32(
                            affine[1, 0] * float_x + affine[1, 1] * float_y
                        )
                        + affine[1, 2] * float_z
                    )
                    + affine[1, 3]
                )
                mm_z = np.float32(
                    np.float32(
                        np.float32(
                            affine[2, 0] * float_x + affine[2, 1] * float_y
                        )
                        + affine[2, 2] * float_z
                    )
                    + affine[2, 3]
                )
                if phase_encoding_axis == 0:
                    mm_x = np.float32(mm_x + displacement_mm[x, y, z])
                elif phase_encoding_axis == 1:
                    mm_y = np.float32(mm_y + displacement_mm[x, y, z])
                else:
                    mm_z = np.float32(mm_z + displacement_mm[x, y, z])
                coordinate_x = np.float32(
                    np.float32(
                        np.float32(source_map[0, 0] * mm_x + source_map[0, 1] * mm_y)
                        + source_map[0, 2] * mm_z
                    )
                    + source_map[0, 3]
                )
                coordinate_y = np.float32(
                    np.float32(
                        np.float32(source_map[1, 0] * mm_x + source_map[1, 1] * mm_y)
                        + source_map[1, 2] * mm_z
                    )
                    + source_map[1, 3]
                )
                coordinate_z = np.float32(
                    np.float32(
                        np.float32(source_map[2, 0] * mm_x + source_map[2, 1] * mm_y)
                        + source_map[2, 2] * mm_z
                    )
                    + source_map[2, 3]
                )
                coordinates[x, y, z, 0] = coordinate_x
                coordinates[x, y, z, 1] = coordinate_y
                coordinates[x, y, z, 2] = coordinate_z
                valid = True
                if not 0.0 <= coordinate_x <= source_nx - 1:
                    valid = False
                if not 0.0 <= coordinate_y <= source_ny - 1:
                    valid = False
                if not 0.0 <= coordinate_z <= source_nz - 1:
                    valid = False
                rounded_x = int(coordinate_x + np.float32(0.5))
                rounded_y = int(coordinate_y + np.float32(0.5))
                rounded_z = int(coordinate_z + np.float32(0.5))
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
                            dy += coefficient * weight_x * (weight_z * dweight_y)
                            dz += coefficient * weight_x * (dweight_z * weight_y)
                values[x, y, z] = value
                derivative_x[x, y, z] = dx
                derivative_y[x, y, z] = dy
                derivative_z[x, y, z] = dz
                mask[x, y, z] = 1 if valid else 0
    return (
        values,
        derivative_x,
        derivative_y,
        derivative_z,
        mask,
        coordinates,
    )


@njit(cache=True, parallel=True, nogil=True)
def _eddy_warp_coordinates_fsl_order(
    displacement_mm: np.ndarray,
    phase_encoding_axis: int,
    affine_to_mm: np.ndarray,
    mm_to_source_voxels: np.ndarray,
) -> np.ndarray:
    """Evaluate only the FSL warp coordinates needed by finite differences."""

    nx, ny, nz = displacement_mm.shape
    affine = affine_to_mm.astype(np.float32)
    source_map = mm_to_source_voxels.astype(np.float32)
    coordinates = np.empty((nx, ny, nz, 3), dtype=np.float32)
    for z in prange(nz):
        float_z = np.float32(z)
        for y in range(ny):
            float_y = np.float32(y)
            for x in range(nx):
                float_x = np.float32(x)
                mm_x = np.float32(
                    np.float32(
                        np.float32(
                            affine[0, 0] * float_x + affine[0, 1] * float_y
                        )
                        + affine[0, 2] * float_z
                    )
                    + affine[0, 3]
                )
                mm_y = np.float32(
                    np.float32(
                        np.float32(
                            affine[1, 0] * float_x + affine[1, 1] * float_y
                        )
                        + affine[1, 2] * float_z
                    )
                    + affine[1, 3]
                )
                mm_z = np.float32(
                    np.float32(
                        np.float32(
                            affine[2, 0] * float_x + affine[2, 1] * float_y
                        )
                        + affine[2, 2] * float_z
                    )
                    + affine[2, 3]
                )
                if phase_encoding_axis == 0:
                    mm_x = np.float32(mm_x + displacement_mm[x, y, z])
                elif phase_encoding_axis == 1:
                    mm_y = np.float32(mm_y + displacement_mm[x, y, z])
                else:
                    mm_z = np.float32(mm_z + displacement_mm[x, y, z])
                coordinates[x, y, z, 0] = np.float32(
                    np.float32(
                        np.float32(source_map[0, 0] * mm_x + source_map[0, 1] * mm_y)
                        + source_map[0, 2] * mm_z
                    )
                    + source_map[0, 3]
                )
                coordinates[x, y, z, 1] = np.float32(
                    np.float32(
                        np.float32(source_map[1, 0] * mm_x + source_map[1, 1] * mm_y)
                        + source_map[1, 2] * mm_z
                    )
                    + source_map[1, 3]
                )
                coordinates[x, y, z, 2] = np.float32(
                    np.float32(
                        np.float32(source_map[2, 0] * mm_x + source_map[2, 1] * mm_y)
                        + source_map[2, 2] * mm_z
                    )
                    + source_map[2, 3]
                )
    return coordinates


@njit(cache=True, parallel=True, nogil=True)
def _sample_eddy_field_affine_fsl_order(
    coefficients: np.ndarray,
    pull_matrix: np.ndarray,
    phase_encoding_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FSL ``raw_affine_transform`` to an EC field in its loop order."""

    source_nx, source_ny, source_nz = coefficients.shape
    matrix = pull_matrix.astype(np.float32)
    values = np.empty(coefficients.shape, dtype=np.float32)
    mask = np.empty(coefficients.shape, dtype=np.uint8)
    for z in prange(source_nz):
        float_z = np.float32(z)
        for x in range(source_nx):
            float_x = np.float32(x)
            coordinate_x = np.float32(
                np.float32(
                    matrix[0, 0] * float_x + matrix[0, 2] * float_z
                )
                + matrix[0, 3]
            )
            coordinate_y = np.float32(
                np.float32(
                    matrix[1, 0] * float_x + matrix[1, 2] * float_z
                )
                + matrix[1, 3]
            )
            coordinate_z = np.float32(
                np.float32(
                    matrix[2, 0] * float_x + matrix[2, 2] * float_z
                )
                + matrix[2, 3]
            )
            for y in range(source_ny):
                valid = True
                if not 0.0 <= coordinate_x <= source_nx - 1:
                    valid = False
                if not 0.0 <= coordinate_y <= source_ny - 1:
                    valid = False
                if not 0.0 <= coordinate_z <= source_nz - 1:
                    valid = False
                rounded_x = int(coordinate_x + np.float32(0.5))
                rounded_y = int(coordinate_y + np.float32(0.5))
                rounded_z = int(coordinate_z + np.float32(0.5))
                start_x = rounded_x - (1 if rounded_x < coordinate_x else 2)
                start_y = rounded_y - (1 if rounded_y < coordinate_y else 2)
                start_z = rounded_z - (1 if rounded_z < coordinate_z else 2)
                value = 0.0
                for offset_z in range(4):
                    index_z = start_z + offset_z
                    weight_z = _cubic_weight(coordinate_z - index_z)
                    wrapped_z = index_z % source_nz
                    for offset_y in range(4):
                        index_y = start_y + offset_y
                        weight_y = _cubic_weight(coordinate_y - index_y)
                        wrapped_y = index_y % source_ny
                        weight_zy = weight_z * weight_y
                        for offset_x in range(4):
                            index_x = start_x + offset_x
                            weight_x = _cubic_weight(coordinate_x - index_x)
                            value += (
                                coefficients[index_x % source_nx, wrapped_y, wrapped_z]
                                * weight_x
                                * weight_zy
                            )
                values[x, y, z] = value
                mask[x, y, z] = 1 if valid else 0
                coordinate_x = np.float32(coordinate_x + matrix[0, 1])
                coordinate_y = np.float32(coordinate_y + matrix[1, 1])
                coordinate_z = np.float32(coordinate_z + matrix[2, 1])
    return values, mask


def transform_eddy_model_to_scan(
    prediction: np.ndarray,
    movement_parameters: np.ndarray,
    quadratic_ec_parameters: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    *,
    jacobian_modulation: bool = True,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
) -> EddyTransformResult:
    """Apply FSL's model-to-scan cubic transform with Jacobian modulation."""

    source = np.asarray(prediction, dtype=np.float32)
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("prediction must be a finite three-dimensional image")
    if phase_encoding_axis not in (0, 1, 2) or phase_encoding_sign not in (-1, 1):
        raise ValueError("phase encoding must contain one signed spatial axis")
    if not np.isfinite(readout_seconds) or readout_seconds <= 0.0:
        raise ValueError("readout_seconds must be positive and finite")
    field = np.asarray(
        quadratic_eddy_field(source.shape, voxel_sizes_mm, quadratic_ec_parameters),
        dtype=np.float32,
    )
    prepared_field = _prepared_eddy_susceptibility_field(
        susceptibility_field_hz, source.shape
    )
    sampling = np.diag((*voxel_sizes_mm, 1.0))
    movement_matrix = _topup_movement_matrix(
        np.asarray(movement_parameters, dtype=np.float64),
        source.shape,
        voxel_sizes_mm,
    )
    field_mask = np.ones(source.shape, dtype=np.uint8)
    if prepared_field is not None:
        susceptibility_pull = (
            np.linalg.inv(sampling) @ np.linalg.inv(movement_matrix) @ sampling
        )
        moved_susceptibility, susceptibility_mask = (
            _sample_fsl_zeropad_cubic_affine(
                prepared_field.interpolation_coefficients,
                np.ones(source.shape, dtype=np.float32),
                susceptibility_pull,
            )
        )
        field += moved_susceptibility
        field_mask = np.asarray(susceptibility_mask != 0.0, dtype=np.uint8)
    scale = np.float32(phase_encoding_sign * readout_seconds)
    forward_displacement = np.asarray(scale * field, dtype=np.float32)
    inverse_displacement, inverse_mask = invert_eddy_displacement(
        forward_displacement,
        phase_encoding_axis,
        field_mask,
    )
    jacobian = np.asarray(
        np.float32(1.0)
        + _mirror_cubic_grid_derivative(inverse_displacement, phase_encoding_axis),
        dtype=np.float32,
    )
    mm_to_source_voxels = np.linalg.inv(sampling) @ np.linalg.inv(movement_matrix)
    with np.errstate(over="ignore"):
        displacement_mm = np.asarray(
            inverse_displacement * np.float32(voxel_sizes_mm[phase_encoding_axis]),
            dtype=np.float32,
        )
    coefficients = _fsl_periodic_cubic_coefficients(source)
    (
        sampled,
        derivative_x,
        derivative_y,
        derivative_z,
        affine_mask,
        coordinates,
    ) = _sample_eddy_cubic_warp_fsl_order(
        coefficients,
        displacement_mm,
        phase_encoding_axis,
        sampling,
        mm_to_source_voxels,
    )
    gradient = np.stack((derivative_x, derivative_y, derivative_z), axis=3)
    combined_mask = np.asarray(inverse_mask * affine_mask, dtype=np.uint8)
    return EddyTransformResult(
        np.asarray(
            sampled * jacobian if jacobian_modulation else sampled, dtype=np.float32
        ),
        combined_mask,
        jacobian,
        coordinates,
        gradient,
    )


def _eddy_inverse_displacement_geometry(
    shape: tuple[int, int, int],
    quadratic_ec_parameters: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the movement-invariant inverse displacement and Jacobian."""

    field = np.asarray(
        quadratic_eddy_field(shape, voxel_sizes_mm, quadratic_ec_parameters),
        dtype=np.float32,
    )
    scale = np.float32(phase_encoding_sign * readout_seconds)
    forward_displacement = np.asarray(scale * field, dtype=np.float32)
    inverse_displacement, _inverse_mask = invert_eddy_displacement(
        forward_displacement,
        phase_encoding_axis,
        np.ones(shape, dtype=np.uint8),
    )
    jacobian = np.asarray(
        np.float32(1.0)
        + _mirror_cubic_grid_derivative(inverse_displacement, phase_encoding_axis),
        dtype=np.float32,
    )
    with np.errstate(over="ignore"):
        displacement_mm = np.asarray(
            inverse_displacement * np.float32(voxel_sizes_mm[phase_encoding_axis]),
            dtype=np.float32,
        )
    return displacement_mm, jacobian


def _eddy_model_to_scan_geometry(
    shape: tuple[int, int, int],
    movement_parameters: np.ndarray,
    quadratic_ec_parameters: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FSL model-to-scan coordinates and Jacobian without image sampling."""

    sampling = np.diag((*voxel_sizes_mm, 1.0))
    movement_matrix = _topup_movement_matrix(
        np.asarray(movement_parameters, dtype=np.float64), shape, voxel_sizes_mm
    )
    prepared_field = _prepared_eddy_susceptibility_field(
        susceptibility_field_hz, shape
    )
    if prepared_field is None:
        displacement_mm, jacobian = _eddy_inverse_displacement_geometry(
            shape,
            quadratic_ec_parameters,
            voxel_sizes_mm,
            phase_encoding_axis,
            phase_encoding_sign,
            readout_seconds,
        )
    else:
        field = np.asarray(
            quadratic_eddy_field(
                shape, voxel_sizes_mm, quadratic_ec_parameters
            ),
            dtype=np.float32,
        )
        susceptibility_pull = (
            np.linalg.inv(sampling) @ np.linalg.inv(movement_matrix) @ sampling
        )
        moved_susceptibility, susceptibility_mask = (
            _sample_fsl_zeropad_cubic_affine(
                prepared_field.interpolation_coefficients,
                np.ones(shape, dtype=np.float32),
                susceptibility_pull,
            )
        )
        field += moved_susceptibility
        forward_displacement = np.asarray(
            np.float32(phase_encoding_sign * readout_seconds) * field,
            dtype=np.float32,
        )
        inverse_displacement, _inverse_mask = invert_eddy_displacement(
            forward_displacement,
            phase_encoding_axis,
            np.asarray(susceptibility_mask != 0.0, dtype=np.uint8),
        )
        jacobian = np.asarray(
            np.float32(1.0)
            + _mirror_cubic_grid_derivative(
                inverse_displacement, phase_encoding_axis
            ),
            dtype=np.float32,
        )
        displacement_mm = np.asarray(
            inverse_displacement
            * np.float32(voxel_sizes_mm[phase_encoding_axis]),
            dtype=np.float32,
        )
    mm_to_source_voxels = np.linalg.inv(sampling) @ np.linalg.inv(movement_matrix)
    coordinates = _eddy_warp_coordinates_fsl_order(
        displacement_mm,
        phase_encoding_axis,
        sampling,
        mm_to_source_voxels,
    )
    return coordinates, jacobian


def transform_eddy_scan_to_model(
    observed_scan: np.ndarray,
    movement_parameters: np.ndarray,
    quadratic_ec_parameters: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    *,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
) -> EddyTransformResult:
    """Apply FSL's scan-to-model transform with Jacobian modulation."""

    source = np.asarray(observed_scan, dtype=np.float32)
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("observed_scan must be a finite three-dimensional image")
    if phase_encoding_axis not in (0, 1, 2) or phase_encoding_sign not in (-1, 1):
        raise ValueError("phase encoding must contain one signed spatial axis")
    if not np.isfinite(readout_seconds) or readout_seconds <= 0.0:
        raise ValueError("readout_seconds must be positive and finite")
    movement_matrix = _topup_movement_matrix(
        np.asarray(movement_parameters, dtype=np.float64),
        source.shape,
        voxel_sizes_mm,
    )
    sampling = np.diag((*voxel_sizes_mm, 1.0))
    field_pull = np.linalg.inv(sampling) @ movement_matrix @ sampling
    field = np.asarray(
        quadratic_eddy_field(source.shape, voxel_sizes_mm, quadratic_ec_parameters),
        dtype=np.float32,
    )
    field_coefficients = _fsl_periodic_cubic_coefficients(field)
    moved_field, field_mask = _sample_eddy_field_affine_fsl_order(
        field_coefficients,
        field_pull,
        phase_encoding_axis,
    )
    prepared_field = _prepared_eddy_susceptibility_field(
        susceptibility_field_hz, source.shape
    )
    if prepared_field is not None:
        moved_field += prepared_field.values_hz
    scale = np.float32(phase_encoding_sign * readout_seconds)
    forward_displacement = np.asarray(scale * moved_field, dtype=np.float32)
    jacobian = np.asarray(
        np.float32(1.0)
        + _mirror_cubic_grid_derivative(forward_displacement, phase_encoding_axis),
        dtype=np.float32,
    )
    displacement_mm = np.asarray(
        forward_displacement * np.float32(voxel_sizes_mm[phase_encoding_axis]),
        dtype=np.float32,
    )
    source_coefficients = _fsl_periodic_cubic_coefficients(source)
    (
        sampled,
        derivative_x,
        derivative_y,
        derivative_z,
        sample_mask,
        coordinates,
    ) = _sample_eddy_cubic_warp_fsl_order(
        source_coefficients,
        displacement_mm,
        phase_encoding_axis,
        movement_matrix @ sampling,
        np.linalg.inv(sampling),
    )
    gradient = np.stack((derivative_x, derivative_y, derivative_z), axis=3)
    combined_mask = np.asarray(field_mask * sample_mask, dtype=np.uint8)
    return EddyTransformResult(
        np.asarray(sampled * jacobian, dtype=np.float32),
        combined_mask,
        jacobian,
        coordinates,
        gradient,
    )


def eddy_parameter_derivatives(
    prediction: np.ndarray,
    movement_parameters: np.ndarray,
    quadratic_ec_parameters: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    *,
    number_of_parameters: int = 15,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
) -> EddyDerivativeResult:
    """Reproduce FSL's forward coordinate/Jacobian parameter derivatives."""

    movement = np.asarray(movement_parameters, dtype=np.float64)
    eddy = np.asarray(quadratic_ec_parameters, dtype=np.float64)
    if movement.shape != (6,) or eddy.shape not in ((9,), (10,)):
        raise ValueError(
            "EDDY derivatives require 6 movement and 9 or 10 EC parameters"
        )
    if number_of_parameters not in (6, 15, 16):
        raise ValueError("number_of_parameters must be 6, 15, or 16")
    if number_of_parameters == 16 and eddy.shape != (10,):
        raise ValueError("16-parameter EDDY derivatives require 10 EC parameters")
    base = transform_eddy_model_to_scan(
        prediction,
        movement,
        eddy,
        voxel_sizes_mm,
        phase_encoding_axis,
        phase_encoding_sign,
        readout_seconds,
        susceptibility_field_hz=susceptibility_field_hz,
    )
    scales = np.array(
        (1.0e-2,) * 3
        + (1.0e-5,) * 3
        + (1.0e-3,) * 3
        + (1.0e-5,) * 6
        + (1.0e-2,)
    )
    derivatives = np.empty(
        (*base.values.shape, number_of_parameters), dtype=np.float32
    )
    prepared_field = _prepared_eddy_susceptibility_field(
        susceptibility_field_hz, prediction.shape
    )
    if prepared_field is None:
        base_displacement_mm, base_geometry_jacobian = (
            _eddy_inverse_displacement_geometry(
                prediction.shape,
                eddy,
                voxel_sizes_mm,
                phase_encoding_axis,
                phase_encoding_sign,
                readout_seconds,
            )
        )
    sampling = np.diag((*voxel_sizes_mm, 1.0))
    inverse_sampling = np.linalg.inv(sampling)
    for parameter_index, scale in enumerate(scales[:number_of_parameters]):
        perturbed_movement = movement.copy()
        perturbed_eddy = eddy.copy()
        if parameter_index < 6:
            perturbed_movement[parameter_index] += scale
        else:
            perturbed_eddy[parameter_index - 6] += scale
        if parameter_index < 6 and prepared_field is None:
            perturbed_matrix = _topup_movement_matrix(
                perturbed_movement, prediction.shape, voxel_sizes_mm
            )
            perturbed_coordinates = _eddy_warp_coordinates_fsl_order(
                base_displacement_mm,
                phase_encoding_axis,
                sampling,
                inverse_sampling @ np.linalg.inv(perturbed_matrix),
            )
            perturbed_jacobian = base_geometry_jacobian
        else:
            perturbed_coordinates, perturbed_jacobian = (
                _eddy_model_to_scan_geometry(
                    prediction.shape,
                    perturbed_movement,
                    perturbed_eddy,
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    prepared_field,
                )
            )
        coordinate_term = _coordinate_gradient_dot_fsl_order(
            np.asarray(perturbed_coordinates - base.coordinates, dtype=np.float32),
            base.spatial_gradient,
        )
        derivatives[..., parameter_index] = np.asarray(
            coordinate_term / np.float32(scale)
            + base.values
            * (perturbed_jacobian - base.jacobian)
            / np.float32(scale),
            dtype=np.float32,
        )
    return EddyDerivativeResult(base, derivatives)


@njit(cache=True, nogil=True)
def _eddy_normal_equations_fsl_order(
    derivatives: np.ndarray, residual: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Accumulate FSL ``make_XtX``/``make_Xty`` in x-fastest image order."""

    parameter_count = derivatives.shape[3]
    xtx = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    xty = np.zeros(parameter_count, dtype=np.float64)
    squared_error = 0.0
    valid_count = 0
    for row in range(parameter_count):
        for column in range(row, parameter_count):
            value = 0.0
            for z in range(derivatives.shape[2]):
                for y in range(derivatives.shape[1]):
                    for x in range(derivatives.shape[0]):
                        if mask[x, y, z] != 0:
                            product = np.float32(
                                derivatives[x, y, z, row] * derivatives[x, y, z, column]
                            )
                            value += float(product)
            xtx[row, column] = value
            xtx[column, row] = value
        value = 0.0
        for z in range(derivatives.shape[2]):
            for y in range(derivatives.shape[1]):
                for x in range(derivatives.shape[0]):
                    if mask[x, y, z] != 0:
                        product = np.float32(
                            derivatives[x, y, z, row] * residual[x, y, z]
                        )
                        value += float(product)
        xty[row] = value
    for z in range(residual.shape[2]):
        for y in range(residual.shape[1]):
            for x in range(residual.shape[0]):
                if mask[x, y, z] != 0:
                    value = np.float32(residual[x, y, z] * mask[x, y, z])
                    squared_error += float(np.float32(value * value))
                    valid_count += 1
    return xtx, xty, squared_error, valid_count


def eddy_gauss_newton_update(
    derivatives: EddyDerivativeResult,
    observed_scan: np.ndarray,
    parameter_mask: np.ndarray | None = None,
) -> EddyGaussNewtonResult:
    """Solve FSL's unit-damped 15-parameter per-volume normal equations."""

    observed = np.asarray(observed_scan, dtype=np.float32)
    if observed.shape != derivatives.base.values.shape or not np.all(
        np.isfinite(observed)
    ):
        raise ValueError(
            "observed scan must be finite and match the transformed prediction"
        )
    mask = derivatives.base.mask
    if parameter_mask is not None:
        extra_mask = np.asarray(parameter_mask)
        if extra_mask.shape != mask.shape:
            raise ValueError("parameter_mask must match the scan grid")
        mask = np.asarray(mask * (extra_mask != 0), dtype=np.uint8)
    residual = np.asarray(derivatives.base.values - observed, dtype=np.float32)
    xtx, xty, squared_error, valid_count = _eddy_normal_equations_fsl_order(
        np.ascontiguousarray(derivatives.derivatives),
        np.ascontiguousarray(residual),
        np.ascontiguousarray(mask),
    )
    if valid_count == 0:
        raise ValueError("EDDY parameter update has no valid voxels")
    update = -np.linalg.solve(xtx + np.eye(xtx.shape[0]), xty)
    return EddyGaussNewtonResult(xtx, xty, update, squared_error / valid_count)


@njit(cache=True, nogil=True)
def _eddy_masked_mss_fsl_order(
    predicted: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Accumulate FSL's masked mean squared residual in x-fastest order."""

    squared_error = 0.0
    valid_count = 0
    for z in range(predicted.shape[2]):
        for y in range(predicted.shape[1]):
            for x in range(predicted.shape[0]):
                if mask[x, y, z] != 0:
                    residual = np.float32(predicted[x, y, z] - observed[x, y, z])
                    product = np.float32(residual * mask[x, y, z])
                    squared_error += float(np.float32(product * product))
                    valid_count += 1
    if valid_count == 0:
        return np.inf
    return squared_error / valid_count


@njit(cache=True, parallel=True, nogil=True)
def _sample_eddy_trilinear_mirror_fsl_order(
    values: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """Sample FSL's parameter mask with trilinear mirror extrapolation."""

    nx, ny, nz = values.shape
    output = np.empty(values.shape, dtype=np.float32)
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                coordinate_x = coordinates[x, y, z, 0]
                coordinate_y = coordinates[x, y, z, 1]
                coordinate_z = coordinates[x, y, z, 2]
                lower_x = int(np.floor(coordinate_x))
                lower_y = int(np.floor(coordinate_y))
                lower_z = int(np.floor(coordinate_z))
                dx = np.float32(coordinate_x - lower_x)
                dy = np.float32(coordinate_y - lower_y)
                dz = np.float32(coordinate_z - lower_z)

                indices_x = (lower_x, lower_x + 1)
                indices_y = (lower_y, lower_y + 1)
                indices_z = (lower_z, lower_z + 1)
                mirrored_x = [0, 0]
                mirrored_y = [0, 0]
                mirrored_z = [0, 0]
                for offset in range(2):
                    index_x = indices_x[offset] % (2 * nx)
                    index_y = indices_y[offset] % (2 * ny)
                    index_z = indices_z[offset] % (2 * nz)
                    mirrored_x[offset] = 2 * nx - 1 - index_x if index_x >= nx else index_x
                    mirrored_y[offset] = 2 * ny - 1 - index_y if index_y >= ny else index_y
                    mirrored_z[offset] = 2 * nz - 1 - index_z if index_z >= nz else index_z

                value_000 = values[mirrored_x[0], mirrored_y[0], mirrored_z[0]]
                value_001 = values[mirrored_x[0], mirrored_y[0], mirrored_z[1]]
                value_010 = values[mirrored_x[0], mirrored_y[1], mirrored_z[0]]
                value_011 = values[mirrored_x[0], mirrored_y[1], mirrored_z[1]]
                value_100 = values[mirrored_x[1], mirrored_y[0], mirrored_z[0]]
                value_101 = values[mirrored_x[1], mirrored_y[0], mirrored_z[1]]
                value_110 = values[mirrored_x[1], mirrored_y[1], mirrored_z[0]]
                value_111 = values[mirrored_x[1], mirrored_y[1], mirrored_z[1]]
                temporary_1 = np.float32(
                    (value_100 - value_000) * dx + value_000
                )
                temporary_2 = np.float32(
                    (value_101 - value_001) * dx + value_001
                )
                temporary_3 = np.float32(
                    (value_110 - value_010) * dx + value_010
                )
                temporary_4 = np.float32(
                    (value_111 - value_011) * dx + value_011
                )
                temporary_5 = np.float32(
                    (temporary_3 - temporary_1) * dy + temporary_1
                )
                temporary_6 = np.float32(
                    (temporary_4 - temporary_2) * dy + temporary_2
                )
                output[x, y, z] = np.float32(
                    (temporary_6 - temporary_5) * dz + temporary_5
                )
    return output


def _eddy_scan_space_parameter_mask(
    model_mask: np.ndarray,
    movement_parameters: np.ndarray,
    quadratic_ec_parameters: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
) -> np.ndarray:
    """Transform and threshold FSL's joint model-space parameter mask."""

    transformed = transform_eddy_model_to_scan(
        np.asarray(model_mask, dtype=np.float32),
        movement_parameters,
        quadratic_ec_parameters,
        voxel_sizes_mm,
        phase_encoding_axis,
        phase_encoding_sign,
        readout_seconds,
        jacobian_modulation=False,
        susceptibility_field_hz=susceptibility_field_hz,
    )
    sampled_mask = _sample_eddy_trilinear_mirror_fsl_order(
        np.ascontiguousarray(model_mask, dtype=np.float32), transformed.coordinates
    )
    return np.asarray(sampled_mask > np.float32(0.99), dtype=np.uint8)


@njit(cache=True, parallel=True, nogil=True)
def _mean_volumes_fsl_order(scans: np.ndarray) -> np.ndarray:
    """Average scans with FSL b0Predictor float accumulation order."""

    mean = np.empty(scans.shape[:3], dtype=np.float32)
    for z in prange(scans.shape[2]):
        for y in range(scans.shape[1]):
            for x in range(scans.shape[0]):
                value = np.float32(0.0)
                for volume in range(scans.shape[3]):
                    value = np.float32(value + scans[x, y, z, volume])
                mean[x, y, z] = np.float32(value / np.float32(scans.shape[3]))
    return mean


@njit(cache=True, nogil=True)
def _soft_mutual_information_fsl_order(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
    first_minimum: float,
    first_maximum: float,
    second_minimum: float,
    second_maximum: float,
) -> float:
    """Evaluate EDDY MutualInfoHelper::SoftMI in x-fastest order."""

    number_of_bins = 256
    first_histogram = np.zeros(number_of_bins, dtype=np.float64)
    second_histogram = np.zeros(number_of_bins, dtype=np.float64)
    joint_histogram = np.zeros(number_of_bins * number_of_bins, dtype=np.float64)
    voxel_weight = 0.0
    first_scale = np.float32(number_of_bins) / np.float32(
        first_maximum - first_minimum
    )
    second_scale = np.float32(number_of_bins) / np.float32(
        second_maximum - second_minimum
    )
    for z in range(first.shape[2]):
        for y in range(first.shape[1]):
            for x in range(first.shape[0]):
                weight = np.float32(weights[x, y, z])
                if weight == 0.0:
                    continue
                first_position = np.float32(
                    np.float32(first[x, y, z] - np.float32(first_minimum))
                    * first_scale
                )
                second_position = np.float32(
                    np.float32(second[x, y, z] - np.float32(second_minimum))
                    * second_scale
                )
                if first_position <= np.float32(0.5):
                    first_index = 0
                    first_remainder = np.float32(0.0)
                elif first_position >= np.float32(number_of_bins - 0.5):
                    first_index = number_of_bins - 1
                    first_remainder = np.float32(0.0)
                else:
                    first_index = int(first_position - np.float32(0.5))
                    first_remainder = np.float32(
                        first_position - np.float32(0.5) - first_index
                    )
                if second_position <= np.float32(0.5):
                    second_index = 0
                    second_remainder = np.float32(0.0)
                elif second_position >= np.float32(number_of_bins - 0.5):
                    second_index = number_of_bins - 1
                    second_remainder = np.float32(0.0)
                else:
                    second_index = int(second_position - np.float32(0.5))
                    second_remainder = np.float32(
                        second_position - np.float32(0.5) - second_index
                    )
                first_low = np.float32(weight * (np.float32(1.0) - first_remainder))
                first_high = np.float32(weight * first_remainder)
                second_low = np.float32(
                    weight * (np.float32(1.0) - second_remainder)
                )
                second_high = np.float32(weight * second_remainder)
                first_histogram[first_index] += float(first_low)
                second_histogram[second_index] += float(second_low)
                joint_histogram[second_index * number_of_bins + first_index] += float(
                    np.float32(
                        weight
                        * np.float32(np.float32(1.0) - first_remainder)
                        * np.float32(np.float32(1.0) - second_remainder)
                    )
                )
                if first_remainder != 0.0:
                    first_histogram[first_index + 1] += float(first_high)
                    joint_histogram[
                        second_index * number_of_bins + first_index + 1
                    ] += float(
                        np.float32(
                            weight
                            * first_remainder
                            * np.float32(np.float32(1.0) - second_remainder)
                        )
                    )
                if second_remainder != 0.0:
                    second_histogram[second_index + 1] += float(second_high)
                    joint_histogram[
                        (second_index + 1) * number_of_bins + first_index
                    ] += float(
                        np.float32(
                            weight
                            * np.float32(np.float32(1.0) - first_remainder)
                            * second_remainder
                        )
                    )
                if first_remainder != 0.0 and second_remainder != 0.0:
                    joint_histogram[
                        (second_index + 1) * number_of_bins + first_index + 1
                    ] += float(
                        np.float32(weight * first_remainder * second_remainder)
                    )
                voxel_weight += float(weight)
    if voxel_weight == 0.0:
        return -np.inf
    first_entropy = 0.0
    second_entropy = 0.0
    joint_entropy = 0.0
    for first_index in range(number_of_bins):
        probability = first_histogram[first_index] / voxel_weight
        if probability != 0.0:
            first_entropy -= probability * np.log(probability)
        probability = second_histogram[first_index] / voxel_weight
        if probability != 0.0:
            second_entropy -= probability * np.log(probability)
        for second_index in range(number_of_bins):
            probability = (
                joint_histogram[second_index * number_of_bins + first_index]
                / voxel_weight
            )
            if probability != 0.0:
                joint_entropy -= probability * np.log(probability)
    return first_entropy + second_entropy - joint_entropy


def estimate_eddy_shell_pe_translation(
    b0_scans: np.ndarray,
    dwi_scans: np.ndarray,
    b0_joint_mask: np.ndarray,
    dwi_joint_mask: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    *,
    maximum_iterations: int = 200,
    fractional_cost_tolerance: float = 1.0e-8,
) -> float:
    """Estimate FSL's post-EDDY shell translation along the PE axis."""

    b0_values = np.asarray(b0_scans, dtype=np.float32)
    dwi_values = np.asarray(dwi_scans, dtype=np.float32)
    b0_mask = np.asarray(b0_joint_mask, dtype=np.uint8)
    dwi_mask = np.asarray(dwi_joint_mask, dtype=np.uint8)
    if (
        b0_values.ndim != 4
        or dwi_values.ndim != 4
        or b0_values.shape[:3] != dwi_values.shape[:3]
    ):
        raise ValueError("b0 and DWI scans must share one finite 3D grid")
    if not np.all(np.isfinite(b0_values)) or not np.all(np.isfinite(dwi_values)):
        raise ValueError("b0 and DWI scans must be finite")
    if b0_mask.shape != b0_values.shape[:3] or dwi_mask.shape != b0_mask.shape:
        raise ValueError("joint masks must match the scan grid")
    if phase_encoding_axis not in (0, 1):
        raise ValueError("FSL post-EDDY shell alignment supports x or y PE")
    if maximum_iterations < 1 or fractional_cost_tolerance <= 0.0:
        raise ValueError("Nelder-Mead limits must be positive")
    b0_mean = _mean_volumes_fsl_order(b0_values)
    dwi_mean = _mean_volumes_fsl_order(dwi_values)
    joint_mask = np.asarray(b0_mask * dwi_mask, dtype=np.float32)
    b0_minimum, b0_maximum = robust_intensity_limits(b0_mean)
    dwi_minimum, dwi_maximum = robust_intensity_limits(dwi_mean)
    b0_coefficients = _fsl_zeropad_cubic_coefficients(b0_mean)
    dwi_coefficients = _fsl_zeropad_cubic_coefficients(dwi_mean)
    axis_size_mm = float(voxel_sizes_mm[phase_encoding_axis])

    def transformed(
        coefficients: np.ndarray, translation_mm: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return _sample_fsl_zeropad_cubic_translation(
            coefficients,
            joint_mask,
            translation_mm / axis_size_mm,
            phase_encoding_axis,
        )

    def cost(translation_mm: float) -> float:
        forward, forward_mask = transformed(dwi_coefficients, translation_mm)
        backward, backward_mask = transformed(b0_coefficients, -translation_mm)
        forward_mi = _soft_mutual_information_fsl_order(
            b0_mean,
            forward,
            forward_mask,
            b0_minimum,
            b0_maximum,
            dwi_minimum,
            dwi_maximum,
        )
        backward_mi = _soft_mutual_information_fsl_order(
            dwi_mean,
            backward,
            backward_mask,
            dwi_minimum,
            dwi_maximum,
            b0_minimum,
            b0_maximum,
        )
        return -0.5 * (forward_mi + backward_mi)

    simplex = np.array((0.0, 1.0), dtype=np.float64)
    costs = np.array((cost(0.0), cost(1.0)), dtype=np.float64)
    for _iteration in range(maximum_iterations):
        best = int(np.argmin(costs))
        worst = int(np.argmax(costs))
        if _fsl_zero_cost_difference(
            float(costs[worst]), float(costs[best]), fractional_cost_tolerance
        ):
            break
        centroid = simplex[best]
        reflected = 2.0 * centroid - simplex[worst]
        reflected_cost = cost(float(reflected))
        if reflected_cost < costs[worst]:
            simplex[worst] = reflected
            costs[worst] = reflected_cost
        if reflected_cost <= costs[best]:
            expanded = 2.0 * simplex[worst] - centroid
            expanded_cost = cost(float(expanded))
            if expanded_cost < costs[worst]:
                simplex[worst] = expanded
                costs[worst] = expanded_cost
        elif reflected_cost >= costs[best]:
            old_worst_cost = costs[worst]
            contracted = 0.5 * (simplex[worst] + centroid)
            contracted_cost = cost(float(contracted))
            if contracted_cost < costs[worst]:
                simplex[worst] = contracted
                costs[worst] = contracted_cost
            if contracted_cost >= old_worst_cost:
                simplex[worst] = 0.5 * (simplex[worst] + simplex[best])
                costs[worst] = cost(float(simplex[worst]))
    return float(simplex[int(np.argmin(costs))])


def estimate_eddy_shell_rigid_alignment(
    b0_scans: np.ndarray,
    dwi_scans: np.ndarray,
    b0_joint_mask: np.ndarray,
    dwi_joint_mask: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    *,
    maximum_iterations: int = 200,
    fractional_cost_tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Estimate FSL's post-EDDY six-parameter shell alignment."""

    b0_values = np.asarray(b0_scans, dtype=np.float32)
    dwi_values = np.asarray(dwi_scans, dtype=np.float32)
    b0_mask = np.asarray(b0_joint_mask, dtype=np.uint8)
    dwi_mask = np.asarray(dwi_joint_mask, dtype=np.uint8)
    if (
        b0_values.ndim != 4
        or dwi_values.ndim != 4
        or b0_values.shape[:3] != dwi_values.shape[:3]
    ):
        raise ValueError("b0 and DWI scans must share one finite 3D grid")
    if not np.all(np.isfinite(b0_values)) or not np.all(np.isfinite(dwi_values)):
        raise ValueError("b0 and DWI scans must be finite")
    if b0_mask.shape != b0_values.shape[:3] or dwi_mask.shape != b0_mask.shape:
        raise ValueError("joint masks must match the scan grid")
    if maximum_iterations < 1 or fractional_cost_tolerance <= 0.0:
        raise ValueError("Nelder-Mead limits must be positive")
    b0_mean = _mean_volumes_fsl_order(b0_values)
    dwi_mean = _mean_volumes_fsl_order(dwi_values)
    joint_mask = np.asarray(b0_mask * dwi_mask, dtype=np.float32)
    b0_minimum, b0_maximum = robust_intensity_limits(b0_mean)
    dwi_minimum, dwi_maximum = robust_intensity_limits(dwi_mean)
    b0_coefficients = _fsl_zeropad_cubic_coefficients(b0_mean)
    dwi_coefficients = _fsl_zeropad_cubic_coefficients(dwi_mean)
    sampling = np.diag((*voxel_sizes_mm, 1.0))
    inverse_sampling = np.linalg.inv(sampling)

    def cost(parameters: np.ndarray) -> float:
        movement_matrix = _topup_movement_matrix(
            parameters, b0_values.shape[:3], voxel_sizes_mm
        )
        forward_pull = inverse_sampling @ np.linalg.inv(movement_matrix) @ sampling
        backward_pull = inverse_sampling @ movement_matrix @ sampling
        forward, forward_mask = _sample_fsl_zeropad_cubic_affine(
            dwi_coefficients, joint_mask, forward_pull
        )
        backward, backward_mask = _sample_fsl_zeropad_cubic_affine(
            b0_coefficients, joint_mask, backward_pull
        )
        forward_mi = _soft_mutual_information_fsl_order(
            b0_mean,
            forward,
            forward_mask,
            b0_minimum,
            b0_maximum,
            dwi_minimum,
            dwi_maximum,
        )
        backward_mi = _soft_mutual_information_fsl_order(
            dwi_mean,
            backward,
            backward_mask,
            dwi_minimum,
            dwi_maximum,
            b0_minimum,
            b0_maximum,
        )
        return -0.5 * (forward_mi + backward_mi)

    dimension = 6
    simplex = np.zeros((dimension + 1, dimension), dtype=np.float64)
    steps = np.asarray(
        (1.0, 1.0, 1.0) + (3.1415 / 180.0,) * 3, dtype=np.float64
    )
    for index in range(dimension):
        simplex[index + 1, index] = steps[index]
    costs = np.asarray([cost(point) for point in simplex], dtype=np.float64)
    for _iteration in range(maximum_iterations):
        best = int(np.argmin(costs))
        worst = int(np.argmax(costs))
        remaining = [index for index in range(dimension + 1) if index != worst]
        second_worst = max(remaining, key=lambda index: costs[index])
        if _fsl_zero_cost_difference(
            float(costs[worst]), float(costs[best]), fractional_cost_tolerance
        ):
            break
        centroid = np.sum(simplex[remaining], axis=0, dtype=np.float64) / dimension
        reflected = 2.0 * centroid - simplex[worst]
        reflected_cost = cost(reflected)
        if reflected_cost < costs[worst]:
            simplex[worst] = reflected
            costs[worst] = reflected_cost
        if reflected_cost <= costs[best]:
            expanded = 2.0 * simplex[worst] - centroid
            expanded_cost = cost(expanded)
            if expanded_cost < costs[worst]:
                simplex[worst] = expanded
                costs[worst] = expanded_cost
        elif reflected_cost >= costs[second_worst]:
            old_worst_cost = costs[worst]
            contracted = 0.5 * (simplex[worst] + centroid)
            contracted_cost = cost(contracted)
            if contracted_cost < costs[worst]:
                simplex[worst] = contracted
                costs[worst] = contracted_cost
            if contracted_cost >= old_worst_cost:
                for index in range(dimension + 1):
                    if index != best:
                        simplex[index] = 0.5 * (simplex[index] + simplex[best])
                        costs[index] = cost(simplex[index])
    return simplex[int(np.argmin(costs))].copy()


def apply_eddy_shell_rigid_alignment(
    movement_parameters: np.ndarray,
    shell_parameters: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Apply PEASUtils::update_mov_par_estimates to one DWI shell."""

    movements = np.asarray(movement_parameters, dtype=np.float64)
    parameters = np.asarray(shell_parameters, dtype=np.float64)
    if movements.ndim != 2 or movements.shape[1] != 6:
        raise ValueError("movement_parameters must have shape (N, 6)")
    if parameters.shape != (6,) or not np.all(np.isfinite(parameters)):
        raise ValueError("shell_parameters must be a finite six-vector")
    shell_matrix = _topup_movement_matrix(parameters, shape, voxel_sizes_mm)
    result = np.empty_like(movements)
    for volume in range(movements.shape[0]):
        movement_matrix = _topup_movement_matrix(
            movements[volume], shape, voxel_sizes_mm
        )
        updated_matrix = np.linalg.inv(
            shell_matrix @ np.linalg.inv(movement_matrix)
        )
        result[volume] = _topup_matrix_to_movement_parameters(
            updated_matrix, shape, voxel_sizes_mm
        )
    return result


def apply_eddy_shell_pe_translation(
    movement_parameters: np.ndarray,
    translation_mm: float,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
) -> np.ndarray:
    """Apply PEASUtils::update_mov_par_estimates to one DWI shell."""

    movements = np.asarray(movement_parameters, dtype=np.float64)
    if movements.ndim != 2 or movements.shape[1] != 6:
        raise ValueError("movement_parameters must have shape (N, 6)")
    if phase_encoding_axis not in (0, 1):
        raise ValueError("FSL post-EDDY shell alignment supports x or y PE")
    shell_parameters = np.zeros(6, dtype=np.float64)
    shell_parameters[phase_encoding_axis] = translation_mm
    return apply_eddy_shell_rigid_alignment(
        movements, shell_parameters, shape, voxel_sizes_mm
    )


def run_eddy_b0_iterations(
    fsl_scaled_scans: np.ndarray,
    brain_mask: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    *,
    number_of_iterations: int = 5,
    workers: int = 8,
    initial_movement_parameters: np.ndarray | None = None,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
) -> EddyB0RegistrationResult:
    """Run FSL's fixed movement-only b0 registration iterations."""

    scans = np.asarray(fsl_scaled_scans, dtype=np.float32).copy()
    mask = np.asarray(brain_mask)
    if scans.ndim != 4 or scans.shape[3] < 2 or not np.all(np.isfinite(scans)):
        raise ValueError("fsl_scaled_scans must be finite with shape (X, Y, Z, N)")
    if mask.shape != scans.shape[:3] or np.count_nonzero(mask) == 0:
        raise ValueError("brain_mask must be nonempty and match the scan grid")
    if number_of_iterations < 1:
        raise ValueError("number_of_iterations must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if initial_movement_parameters is None:
        movement = np.zeros((scans.shape[3], 6), dtype=np.float64)
    else:
        movement = np.asarray(initial_movement_parameters, dtype=np.float64)
        if movement.shape != (scans.shape[3], 6) or not np.all(
            np.isfinite(movement)
        ):
            raise ValueError(
                "initial_movement_parameters must be finite with shape (N, 6)"
            )
        movement = movement.copy()
    zero_eddy = np.zeros(10, dtype=np.float64)
    prepared_field = _prepared_eddy_susceptibility_field(
        susceptibility_field_hz, scans.shape[:3]
    )
    unwarped = np.empty_like(scans)
    volume_indices = tuple(range(scans.shape[3]))
    history: list[EddyIterationResult] = []

    def configure_worker() -> None:
        set_num_threads(1)

    executor = (
        ThreadPoolExecutor(
            max_workers=min(workers, scans.shape[3]), initializer=configure_worker
        )
        if workers > 1
        else None
    )
    previous_numba_threads = get_num_threads()
    if executor is None:
        set_num_threads(1)

    def unwarp_volume(volume: int) -> EddyTransformResult:
        return transform_eddy_scan_to_model(
            scans[..., volume],
            movement[volume],
            zero_eddy,
            voxel_sizes_mm,
            phase_encoding_axis,
            phase_encoding_sign,
            readout_seconds,
            susceptibility_field_hz=prepared_field,
        )

    try:
        for _iteration in range(number_of_iterations):
            transformed_volumes = (
                list(executor.map(unwarp_volume, volume_indices))
                if executor is not None
                else list(map(unwarp_volume, volume_indices))
            )
            joint_mask = np.ones(scans.shape[:3], dtype=np.uint8)
            for volume, transformed in enumerate(transformed_volumes):
                unwarped[..., volume] = transformed.values
                joint_mask *= transformed.mask
            prediction = _mean_volumes_fsl_order(unwarped)

            def update_volume(
                volume: int,
            ) -> tuple[np.ndarray, float, np.ndarray, bool]:
                derivatives = eddy_parameter_derivatives(
                    prediction,
                    movement[volume],
                    zero_eddy,
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    number_of_parameters=6,
                    susceptibility_field_hz=prepared_field,
                )
                parameter_mask = _eddy_scan_space_parameter_mask(
                    joint_mask,
                    movement[volume],
                    zero_eddy,
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    prepared_field,
                )
                update = eddy_gauss_newton_update(
                    derivatives, scans[..., volume], parameter_mask
                )
                candidate_movement = movement[volume] + update.update
                candidate = transform_eddy_model_to_scan(
                    prediction,
                    candidate_movement,
                    zero_eddy,
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    susceptibility_field_hz=prepared_field,
                )
                candidate_mask = _eddy_scan_space_parameter_mask(
                    joint_mask,
                    candidate_movement,
                    zero_eddy,
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    prepared_field,
                )
                candidate_mask *= candidate.mask
                candidate_mss = _eddy_masked_mss_fsl_order(
                    candidate.values, scans[..., volume], candidate_mask
                )
                return (
                    update.update,
                    update.mean_squared_error,
                    candidate_movement,
                    candidate_mss <= update.mean_squared_error,
                )

            volume_results = (
                list(executor.map(update_volume, volume_indices))
                if executor is not None
                else list(map(update_volume, volume_indices))
            )
            updates = np.empty((scans.shape[3], 6), dtype=np.float64)
            mean_squared_errors = np.empty(scans.shape[3], dtype=np.float64)
            accepted = np.zeros(scans.shape[3], dtype=bool)
            for volume, volume_result in enumerate(volume_results):
                update, mss, candidate_movement, was_accepted = volume_result
                updates[volume] = update
                mean_squared_errors[volume] = mss
                if was_accepted:
                    movement[volume] = candidate_movement
                    accepted[volume] = True
            history.append(
                EddyIterationResult(
                    np.empty(0, dtype=np.float64),
                    updates,
                    mean_squared_errors,
                    accepted,
                    joint_mask.copy(),
                )
            )
        movement = _apply_eddy_dwi_location_reference(
            movement, scans.shape[:3], voxel_sizes_mm
        )
        transformed_volumes = (
            list(executor.map(unwarp_volume, volume_indices))
            if executor is not None
            else list(map(unwarp_volume, volume_indices))
        )
        final_joint_mask = np.ones(scans.shape[:3], dtype=np.uint8)
        for volume, transformed in enumerate(transformed_volumes):
            unwarped[..., volume] = transformed.values
            final_joint_mask *= transformed.mask
    finally:
        if executor is not None:
            executor.shutdown()
        else:
            set_num_threads(previous_numba_threads)
    return EddyB0RegistrationResult(
        movement.copy(), unwarped.copy(), tuple(history), final_joint_mask
    )


def _eddy_flagged_volumes(outlier_map: np.ndarray) -> tuple[int, ...]:
    """Return, in ascending order, volume indices containing at least one outlier slice."""

    return tuple(int(index) for index in np.flatnonzero(np.any(outlier_map, axis=1)))


def _apply_eddy_outlier_slices(
    scans: np.ndarray,
    original_scans: np.ndarray,
    stored_outliers: np.ndarray,
    current_outliers: np.ndarray,
    predicted_scan: np.ndarray,
    replacement_mask: np.ndarray,
    volume: int,
) -> None:
    """Restore revoked outliers and write this iteration's outlier slices in FSL order."""

    removed = stored_outliers[volume] & ~current_outliers
    for slice_index in np.flatnonzero(removed):
        scans[:, :, slice_index, volume] = original_scans[
            :, :, slice_index, volume
        ]
    for slice_index in np.flatnonzero(current_outliers):
        selected = replacement_mask[:, :, slice_index] != 0
        scans[:, :, slice_index, volume][selected] = predicted_scan[
            :, :, slice_index
        ][selected]
    stored_outliers[volume] = current_outliers


def run_eddy_dwi_iterations(
    fsl_scaled_scans: np.ndarray,
    bvecs: np.ndarray,
    brain_mask: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    *,
    number_of_iterations: int = 5,
    number_of_hyperparameter_voxels: int = 1000,
    random_seed: int,
    bvals: np.ndarray | None = None,
    workers: int = 8,
    replace_outliers: bool = False,
    initial_movement_parameters: np.ndarray | None = None,
    initial_quadratic_ec_parameters: np.ndarray | None = None,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
) -> EddyDwiRegistrationResult:
    """Run FSL's fixed no-TOPUP DWI registration and optional ``--repol``.

    Input scans must already carry EDDY's global signal scale.  This runner
    applies FSL's first-DWI location reference before final resampling; B0
    estimation remains a separate stage.
    """

    scans = np.asarray(fsl_scaled_scans, dtype=np.float32).copy()
    original_scans = scans.copy()
    vectors = np.asarray(bvecs, dtype=np.float64)
    shell_bvals = None if bvals is None else np.asarray(bvals, dtype=np.float64).reshape(-1)
    mask = np.asarray(brain_mask)
    if scans.ndim != 4 or scans.shape[3] < 2 or not np.all(np.isfinite(scans)):
        raise ValueError("fsl_scaled_scans must be finite with shape (X, Y, Z, N)")
    if vectors.shape != (3, scans.shape[3]) or not np.all(np.isfinite(vectors)):
        raise ValueError("bvecs must be finite with shape (3, N)")
    if shell_bvals is not None and (
        shell_bvals.shape != (scans.shape[3],)
        or not np.all(np.isfinite(shell_bvals))
        or np.any(shell_bvals <= 0.0)
    ):
        raise ValueError("bvals must contain one positive finite value per DWI")
    if mask.shape != scans.shape[:3] or np.count_nonzero(mask) == 0:
        raise ValueError("brain_mask must be nonempty and match the scan grid")
    if number_of_iterations < 1:
        raise ValueError("number_of_iterations must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if initial_movement_parameters is None:
        movement = np.zeros((scans.shape[3], 6), dtype=np.float64)
    else:
        movement = np.asarray(initial_movement_parameters, dtype=np.float64)
        if movement.shape != (scans.shape[3], 6) or not np.all(
            np.isfinite(movement)
        ):
            raise ValueError(
                "initial_movement_parameters must be finite with shape (N, 6)"
            )
        movement = movement.copy()
    if initial_quadratic_ec_parameters is None:
        eddy = np.zeros((scans.shape[3], 10), dtype=np.float64)
    else:
        eddy = np.asarray(initial_quadratic_ec_parameters, dtype=np.float64)
        if eddy.shape not in (
            (scans.shape[3], 9),
            (scans.shape[3], 10),
        ) or not np.all(np.isfinite(eddy)):
            raise ValueError(
                "initial_quadratic_ec_parameters must be finite with shape (N, 9 or 10)"
            )
        if eddy.shape[1] == 9:
            eddy = np.column_stack(
                (eddy, np.zeros(scans.shape[3], dtype=np.float64))
            )
        else:
            eddy = eddy.copy()
    prepared_field = _prepared_eddy_susceptibility_field(
        susceptibility_field_hz, scans.shape[:3]
    )
    parameter_count = 16 if prepared_field is not None else 15
    if prepared_field is not None and shell_bvals is None:
        raise ValueError("bvals are required when estimating TOPUP field offsets")
    history: list[EddyIterationResult] = []
    unwarped = np.empty_like(scans)
    volume_indices = tuple(range(scans.shape[3]))
    outlier_result: EddyOutlierResult | None = None
    stored_outliers = np.zeros((scans.shape[3], scans.shape[2]), dtype=bool)

    def configure_worker() -> None:
        set_num_threads(1)

    executor = (
        ThreadPoolExecutor(
            max_workers=min(workers, scans.shape[3]), initializer=configure_worker
        )
        if workers > 1
        else None
    )
    previous_numba_threads = get_num_threads()
    if executor is None:
        set_num_threads(1)

    def unwarp_volume(volume: int) -> EddyTransformResult:
        return transform_eddy_scan_to_model(
            scans[..., volume],
            movement[volume],
            eddy[volume],
            voxel_sizes_mm,
            phase_encoding_axis,
            phase_encoding_sign,
            readout_seconds,
            susceptibility_field_hz=prepared_field,
        )

    def load_prediction_state() -> tuple[
        np.ndarray,
        EddyHyperparameterResult,
        EddySphericalGP,
        np.ndarray,
    ]:
        transformed_volumes = (
            list(executor.map(unwarp_volume, volume_indices))
            if executor is not None
            else list(map(unwarp_volume, volume_indices))
        )
        joint_mask = np.ones(scans.shape[:3], dtype=np.uint8)
        for volume, transformed in enumerate(transformed_volumes):
            unwarped[..., volume] = transformed.values
            joint_mask *= transformed.mask
        selected, _coordinates = select_fsl_gp_voxels(
            unwarped,
            np.asarray(joint_mask * (mask != 0), dtype=np.uint8),
            number_of_voxels=number_of_hyperparameter_voxels,
            random_seed=random_seed,
        )
        hyperparameter_result = estimate_spherical_gp_hyperparameters(
            selected, vectors
        )
        gp = fit_spherical_gp_weights(
            vectors, hyperparameter_result.hyperparameters
        )
        predictions = predict_spherical_gp(unwarped, gp)
        return joint_mask, hyperparameter_result, gp, predictions

    def detect_outliers(
        predictions: np.ndarray, joint_mask: np.ndarray
    ) -> EddyOutlierResult:
        def scan_space_prediction(
            volume: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            transformed = transform_eddy_model_to_scan(
                predictions[..., volume],
                movement[volume],
                eddy[volume],
                voxel_sizes_mm,
                phase_encoding_axis,
                phase_encoding_sign,
                readout_seconds,
                susceptibility_field_hz=prepared_field,
            )
            detection_mask = _eddy_scan_space_parameter_mask(
                np.asarray(joint_mask * (mask != 0), dtype=np.uint8),
                movement[volume],
                eddy[volume],
                voxel_sizes_mm,
                phase_encoding_axis,
                phase_encoding_sign,
                readout_seconds,
                prepared_field,
            )
            detection_mask *= transformed.mask
            return transformed.values, detection_mask

        transformed = (
            list(executor.map(scan_space_prediction, volume_indices))
            if executor is not None
            else list(map(scan_space_prediction, volume_indices))
        )
        scan_predictions = np.stack(
            tuple(item[0] for item in transformed), axis=3
        )
        scan_masks = np.stack(tuple(item[1] for item in transformed), axis=3)
        statistics = eddy_slice_statistics(
            original_scans, scan_predictions, scan_masks
        )
        previous = None if outlier_result is None else outlier_result.outlier_map
        return detect_eddy_slice_outliers(
            statistics, previous_outlier_map=previous
        )

    def replace_detected_outliers(
        detected: EddyOutlierResult,
        joint_mask: np.ndarray,
        hyperparameters: np.ndarray,
    ) -> None:
        nonlocal stored_outliers
        flagged_volumes = _eddy_flagged_volumes(detected.outlier_map)
        if not flagged_volumes:
            return
        leave_one_out_gp = fit_spherical_gp_weights(
            vectors, hyperparameters, exclude_target=True
        )
        leave_one_out_predictions = predict_spherical_gp(
            unwarped, leave_one_out_gp, exclude_target=True
        )

        def replacement_volume(
            volume: int,
        ) -> tuple[int, np.ndarray, np.ndarray]:
            transformed = transform_eddy_model_to_scan(
                leave_one_out_predictions[..., volume],
                movement[volume],
                eddy[volume],
                voxel_sizes_mm,
                phase_encoding_axis,
                phase_encoding_sign,
                readout_seconds,
                susceptibility_field_hz=prepared_field,
            )
            sampled_mask = _sample_eddy_trilinear_mirror_fsl_order(
                np.ascontiguousarray(joint_mask, dtype=np.float32),
                transformed.coordinates,
            )
            replacement_mask = np.asarray(
                transformed.mask * (sampled_mask > np.float32(0.9)),
                dtype=np.uint8,
            )
            return volume, transformed.values, replacement_mask

        replacements = (
            list(executor.map(replacement_volume, flagged_volumes))
            if executor is not None
            else list(map(replacement_volume, flagged_volumes))
        )
        for volume, predicted_scan, replacement_mask in replacements:
            current = detected.outlier_map[volume]
            _apply_eddy_outlier_slices(
                scans,
                original_scans,
                stored_outliers,
                current,
                predicted_scan,
                replacement_mask,
                volume,
            )

    try:
        for _iteration in range(number_of_iterations):
            joint_mask, hyperparameter_result, _gp, predictions = (
                load_prediction_state()
            )
            if replace_outliers:
                outlier_result = detect_outliers(predictions, joint_mask)
                if _iteration:
                    replace_detected_outliers(
                        outlier_result,
                        joint_mask,
                        hyperparameter_result.hyperparameters,
                    )
                    joint_mask, hyperparameter_result, _gp, predictions = (
                        load_prediction_state()
                    )

            def update_volume(
                volume: int,
            ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
                derivatives = eddy_parameter_derivatives(
                    predictions[..., volume],
                    movement[volume],
                    eddy[volume],
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    number_of_parameters=parameter_count,
                    susceptibility_field_hz=prepared_field,
                )
                parameter_mask = _eddy_scan_space_parameter_mask(
                    joint_mask,
                    movement[volume],
                    eddy[volume],
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    prepared_field,
                )
                update = eddy_gauss_newton_update(
                    derivatives, scans[..., volume], parameter_mask
                )
                candidate_movement = movement[volume] + update.update[:6]
                candidate_eddy = eddy[volume].copy()
                candidate_eddy[: parameter_count - 6] += update.update[6:]
                candidate = transform_eddy_model_to_scan(
                    predictions[..., volume],
                    candidate_movement,
                    candidate_eddy,
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    susceptibility_field_hz=prepared_field,
                )
                candidate_mask = _eddy_scan_space_parameter_mask(
                    joint_mask,
                    candidate_movement,
                    candidate_eddy,
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                    prepared_field,
                )
                candidate_mask *= candidate.mask
                candidate_mss = _eddy_masked_mss_fsl_order(
                    candidate.values, scans[..., volume], candidate_mask
                )
                return (
                    update.update,
                    update.mean_squared_error,
                    candidate_movement,
                    candidate_eddy,
                    candidate_mss <= update.mean_squared_error,
                )

            volume_results = (
                list(executor.map(update_volume, volume_indices))
                if executor is not None
                else list(map(update_volume, volume_indices))
            )
            updates = np.empty(
                (scans.shape[3], parameter_count), dtype=np.float64
            )
            mean_squared_errors = np.empty(scans.shape[3], dtype=np.float64)
            accepted = np.zeros(scans.shape[3], dtype=bool)
            for volume, volume_result in enumerate(volume_results):
                update, mss, candidate_movement, candidate_eddy, was_accepted = (
                    volume_result
                )
                updates[volume] = update
                mean_squared_errors[volume] = mss
                if was_accepted:
                    movement[volume] = candidate_movement
                    eddy[volume] = candidate_eddy
                    accepted[volume] = True
            history.append(
                EddyIterationResult(
                    hyperparameter_result.hyperparameters.copy(),
                    updates,
                    mean_squared_errors,
                    accepted,
                    joint_mask.copy(),
                )
            )
            if prepared_field is not None:
                movement, eddy = _separate_eddy_field_offset_from_movement(
                    movement,
                    eddy,
                    shell_bvals,
                    vectors,
                    scans.shape[:3],
                    voxel_sizes_mm,
                    phase_encoding_axis,
                    phase_encoding_sign,
                    readout_seconds,
                )
        movement = _apply_eddy_dwi_location_reference(
            movement, scans.shape[:3], voxel_sizes_mm
        )
        if replace_outliers:
            final_joint_mask, final_hyperparameters, _gp, final_predictions = (
                load_prediction_state()
            )
            outlier_result = detect_outliers(
                final_predictions, final_joint_mask
            )
            replace_detected_outliers(
                outlier_result,
                final_joint_mask,
                final_hyperparameters.hyperparameters,
            )
        outlier_free_scans = scans.copy() if replace_outliers else None
        transformed_volumes = (
            list(executor.map(unwarp_volume, volume_indices))
            if executor is not None
            else list(map(unwarp_volume, volume_indices))
        )
        final_joint_mask = np.ones(scans.shape[:3], dtype=np.uint8)
        for volume, transformed in enumerate(transformed_volumes):
            unwarped[..., volume] = transformed.values
            final_joint_mask *= transformed.mask
    finally:
        if executor is not None:
            executor.shutdown()
        else:
            set_num_threads(previous_numba_threads)
    rotated = rotate_bvecs_eddy(
        vectors,
        np.ones(scans.shape[3], dtype=np.float64),
        movement,
        scans.shape[:3],
        voxel_sizes_mm,
    )
    return EddyDwiRegistrationResult(
        movement.copy(),
        eddy.copy(),
        unwarped.copy(),
        rotated,
        tuple(history),
        final_joint_mask,
        None if outlier_result is None else outlier_result.outlier_map.copy(),
        outlier_free_scans,
    )


def run_simnibs46_eddy(
    scans: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    brain_mask: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
    *,
    random_seed: int,
    susceptibility_field_hz: np.ndarray
    | PreparedEddySusceptibilityField
    | None = None,
    workers: int = 8,
    replace_outliers: bool = True,
    align_shells_post_eddy: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> EddyRunResult:
    """Run the complete single-shell SimNIBS 4.6 EDDY fixed subset."""

    values = np.asarray(scans, dtype=np.float32)
    shell_values = np.asarray(bvals, dtype=np.float64).reshape(-1)
    vectors = np.asarray(bvecs, dtype=np.float64)
    mask = np.asarray(brain_mask)
    if values.ndim != 4 or values.shape[3] != shell_values.size:
        raise ValueError("scans and bvals must share the fourth-axis length")
    if vectors.shape != (3, shell_values.size):
        raise ValueError("bvecs must have shape (3, N) matching bvals")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(shell_values)):
        raise ValueError("scans and bvals must be finite")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("bvecs must be finite")
    if mask.shape != values.shape[:3] or np.count_nonzero(mask) == 0:
        raise ValueError("brain_mask must be nonempty and match the scan grid")
    if workers < 1:
        raise ValueError("workers must be positive")
    b0_indices = np.flatnonzero(shell_values <= 50.0)
    dwi_indices = np.flatnonzero(shell_values > 50.0)
    if b0_indices.size == 0 or dwi_indices.size < 2:
        raise ValueError("the fixed subset requires at least one b0 and two DWI scans")
    positive_bvals = shell_values[dwi_indices]
    if np.max(positive_bvals) - np.min(positive_bvals) > 100.0:
        raise ValueError("the fixed EDDY subset supports one diffusion shell")
    first_b0 = values[..., int(b0_indices[0])]
    mask_values = mask != 0
    first_b0_mean = float(np.mean(first_b0[mask_values], dtype=np.float64))
    if first_b0_mean == 0.0:
        raise ValueError("the first b0 has zero mean inside the brain mask")
    scale_factor = 100.0 / first_b0_mean
    scaled = np.asarray(values * np.float32(scale_factor), dtype=np.float32)
    prepared_field = _prepared_eddy_susceptibility_field(
        susceptibility_field_hz, values.shape[:3]
    )

    if progress is not None:
        progress("register_b0", 0, 1)
    if b0_indices.size > 1:
        b0_result = run_eddy_b0_iterations(
            scaled[..., b0_indices],
            mask,
            voxel_sizes_mm,
            phase_encoding_axis,
            phase_encoding_sign,
            readout_seconds,
            workers=workers,
            susceptibility_field_hz=prepared_field,
        )
    else:
        identity = transform_eddy_scan_to_model(
            scaled[..., int(b0_indices[0])],
            np.zeros(6, dtype=np.float64),
            np.zeros(10, dtype=np.float64),
            voxel_sizes_mm,
            phase_encoding_axis,
            phase_encoding_sign,
            readout_seconds,
            susceptibility_field_hz=prepared_field,
        )
        b0_result = EddyB0RegistrationResult(
            np.zeros((1, 6), dtype=np.float64),
            identity.values[..., None],
            (),
            identity.mask.copy(),
        )
    if progress is not None:
        progress("register_b0", 1, 1)
        progress("register_dwi", 0, 1)
    dwi_result = run_eddy_dwi_iterations(
        scaled[..., dwi_indices],
        vectors[:, dwi_indices],
        mask,
        voxel_sizes_mm,
        phase_encoding_axis,
        phase_encoding_sign,
        readout_seconds,
        random_seed=random_seed,
        bvals=shell_values[dwi_indices],
        workers=workers,
        replace_outliers=replace_outliers,
        susceptibility_field_hz=prepared_field,
    )
    if progress is not None:
        progress("register_dwi", 1, 1)
        progress("align_shells", 0, 1)
    shell_translation = estimate_eddy_shell_pe_translation(
        b0_result.unwarped_scans,
        dwi_result.unwarped_scans,
        b0_result.joint_mask,
        dwi_result.joint_mask,
        voxel_sizes_mm,
        phase_encoding_axis,
    )
    shell_alignment = estimate_eddy_shell_rigid_alignment(
        b0_result.unwarped_scans,
        dwi_result.unwarped_scans,
        b0_result.joint_mask,
        dwi_result.joint_mask,
        voxel_sizes_mm,
    )
    dwi_movement = dwi_result.movement_parameters.copy()
    if align_shells_post_eddy:
        dwi_movement = apply_eddy_shell_rigid_alignment(
            dwi_movement,
            shell_alignment,
            values.shape[:3],
            voxel_sizes_mm,
        )
    else:
        dwi_movement = apply_eddy_shell_pe_translation(
            dwi_movement,
            shell_translation,
            values.shape[:3],
            voxel_sizes_mm,
            phase_encoding_axis,
        )
    if progress is not None:
        progress("align_shells", 1, 1)
        progress("final_resampling", 0, int(values.shape[3]))

    corrected_scaled = np.empty_like(scaled)
    corrected_scaled[..., b0_indices] = b0_result.unwarped_scans
    dwi_scan_space = (
        dwi_result.outlier_free_scans
        if dwi_result.outlier_free_scans is not None
        else scaled[..., dwi_indices]
    )

    def resample_dwi(local_index: int) -> tuple[int, np.ndarray]:
        transformed = transform_eddy_scan_to_model(
            dwi_scan_space[..., local_index],
            dwi_movement[local_index],
            dwi_result.quadratic_ec_parameters[local_index],
            voxel_sizes_mm,
            phase_encoding_axis,
            phase_encoding_sign,
            readout_seconds,
            susceptibility_field_hz=prepared_field,
        )
        return local_index, transformed.values

    dwi_local_indices = tuple(range(dwi_indices.size))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, dwi_indices.size)) as executor:
            resampled_dwi = list(executor.map(resample_dwi, dwi_local_indices))
    else:
        resampled_dwi = list(map(resample_dwi, dwi_local_indices))
    for completed, (local_index, corrected) in enumerate(resampled_dwi, start=1):
        corrected_scaled[..., int(dwi_indices[local_index])] = corrected
        if progress is not None:
            progress("final_resampling", int(b0_indices.size) + completed, values.shape[3])
    if progress is not None and b0_indices.size:
        progress("final_resampling", values.shape[3], values.shape[3])

    movement = np.zeros((values.shape[3], 6), dtype=np.float64)
    movement[b0_indices] = b0_result.movement_parameters
    movement[dwi_indices] = dwi_movement
    eddy = np.zeros((values.shape[3], 10), dtype=np.float64)
    eddy[dwi_indices] = dwi_result.quadratic_ec_parameters
    rotated = rotate_bvecs_eddy(
        vectors,
        shell_values,
        movement,
        values.shape[:3],
        voxel_sizes_mm,
    )
    outlier_map = np.zeros((values.shape[3], values.shape[2]), dtype=bool)
    if dwi_result.outlier_map is not None:
        outlier_map[dwi_indices] = dwi_result.outlier_map
    outlier_free = None
    if dwi_result.outlier_free_scans is not None:
        outlier_free_scaled = scaled.copy()
        outlier_free_scaled[..., dwi_indices] = dwi_result.outlier_free_scans
        outlier_free = np.asarray(
            outlier_free_scaled / np.float32(scale_factor), dtype=np.float32
        )
    return EddyRunResult(
        np.asarray(corrected_scaled / np.float32(scale_factor), dtype=np.float32),
        movement,
        eddy,
        rotated,
        outlier_map,
        outlier_free,
        scale_factor,
        shell_translation,
        shell_alignment,
        b0_result,
        dwi_result,
    )


def run_eddy_nifti(
    dwi_file: str | Path,
    bvals_file: str | Path,
    bvecs_file: str | Path,
    brain_mask_file: str | Path,
    output_directory: str | Path,
    *,
    readout_seconds: float,
    phase_encoding_direction: str,
    susceptibility_field_file: str | Path | None = None,
    random_seed: int = 1,
    workers: int = 8,
    replace_outliers: bool = True,
    align_shells_post_eddy: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """Run the fixed EDDY subset and write its public NIfTI artifacts."""

    import nibabel as nib

    directions = {
        "x": (0, 1),
        "x-": (0, -1),
        "y": (1, 1),
        "y-": (1, -1),
    }
    if phase_encoding_direction not in directions:
        raise ValueError("phase encoding direction must be x, x-, y, or y-")
    if not np.isfinite(readout_seconds) or readout_seconds <= 0.0:
        raise ValueError("readout seconds must be positive and finite")
    if workers < 1:
        raise ValueError("workers must be positive")
    if random_seed < 0:
        raise ValueError("random seed must be nonnegative")

    image = nib.load(str(dwi_file))
    mask_image = nib.load(str(brain_mask_file))
    scans = np.asarray(image.dataobj, dtype=np.float32)
    mask = np.asarray(mask_image.dataobj)
    if scans.ndim != 4:
        raise ValueError("DWI input must be four-dimensional")
    if mask.shape != scans.shape[:3]:
        raise ValueError("brain mask must match the DWI spatial shape")
    if not np.allclose(image.affine, mask_image.affine, rtol=0.0, atol=1.0e-6):
        raise ValueError("brain mask and DWI must share one affine")
    bvals = np.asarray(np.loadtxt(bvals_file), dtype=np.float64).reshape(-1)
    bvecs = np.atleast_2d(np.loadtxt(bvecs_file, dtype=np.float64))
    if bvals.shape != (scans.shape[3],):
        raise ValueError("b-values must contain one value per DWI volume")
    if bvecs.shape != (3, scans.shape[3]):
        raise ValueError("b-vectors must have shape (3, N) matching the DWI")

    susceptibility_field: np.ndarray | None = None
    if susceptibility_field_file is not None:
        field_image = nib.load(str(susceptibility_field_file))
        susceptibility_field = np.asarray(field_image.dataobj, dtype=np.float32)
        if susceptibility_field.shape != scans.shape[:3]:
            raise ValueError("susceptibility field must match the DWI spatial shape")
        if not np.allclose(image.affine, field_image.affine, rtol=0.0, atol=1.0e-6):
            raise ValueError("susceptibility field and DWI must share one affine")

    phase_encoding_axis, phase_encoding_sign = directions[
        phase_encoding_direction
    ]
    voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    started = perf_counter()
    result = run_simnibs46_eddy(
        scans,
        bvals,
        bvecs,
        mask,
        voxel_sizes,
        phase_encoding_axis,
        phase_encoding_sign,
        readout_seconds,
        random_seed=random_seed,
        susceptibility_field_hz=susceptibility_field,
        workers=workers,
        replace_outliers=replace_outliers,
        align_shells_post_eddy=align_shells_post_eddy,
        progress=progress,
    )
    algorithm_seconds = perf_counter() - started

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    float_header = image.header.copy()
    float_header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(result.corrected_scans, image.affine, float_header),
        output / "corrected_dwi.nii.gz",
    )
    if result.outlier_free_scans is not None:
        nib.save(
            nib.Nifti1Image(result.outlier_free_scans, image.affine, float_header),
            output / "outlier_free_data.nii.gz",
        )
    parameters = np.column_stack(
        (result.movement_parameters, result.quadratic_ec_parameters)
    )
    np.savetxt(output / "eddy_parameters.txt", parameters, fmt="%.10g")
    np.savetxt(output / "rotated_bvecs", result.rotated_bvecs, fmt="%.10g")
    np.savetxt(output / "bvals", bvals[None, :], fmt="%.10g")
    np.savetxt(
        output / "outlier_map.txt", result.outlier_map.astype(np.uint8), fmt="%d"
    )
    np.savetxt(
        output / "shell_pe_translation.txt",
        np.asarray((result.shell_pe_translation_mm,))[None, :],
        fmt="%.10g",
    )
    np.savetxt(
        output / "shell_alignment_parameters.txt",
        result.shell_alignment_parameters[None, :],
        fmt="%.10g",
    )
    if result.dwi_registration.iterations:
        np.savetxt(
            output / "dwi_hyperparameter_history.txt",
            np.stack(
                [
                    iteration.hyperparameters
                    for iteration in result.dwi_registration.iterations
                ]
            ),
            fmt="%.10g",
        )
        np.savetxt(
            output / "dwi_update_history.txt",
            np.stack(
                [
                    iteration.updates.reshape(-1)
                    for iteration in result.dwi_registration.iterations
                ]
            ),
            fmt="%.10g",
        )

    report: dict[str, object] = {
        "status": "complete",
        "algorithm": "SimNIBS-4.6-eddy-repol-fixed-subset",
        "algorithm_seconds": algorithm_seconds,
        "input_shape": list(scans.shape),
        "voxel_sizes_mm": list(voxel_sizes),
        "phase_encoding_direction": phase_encoding_direction,
        "readout_seconds": readout_seconds,
        "workers": workers,
        "random_seed": random_seed,
        "replace_outliers": replace_outliers,
        "align_shells_post_eddy": align_shells_post_eddy,
        "susceptibility_field": susceptibility_field is not None,
        "outlier_slices": int(np.count_nonzero(result.outlier_map)),
        "scale_factor": result.scale_factor,
        "shell_pe_translation_mm": result.shell_pe_translation_mm,
        "shell_alignment_parameters": result.shell_alignment_parameters.tolist(),
    }
    (output / "eddy_qa.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _separate_eddy_field_offset_from_movement(
    movement_parameters: np.ndarray,
    quadratic_ec_parameters: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    phase_encoding_axis: int,
    phase_encoding_sign: int,
    readout_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FSL's default linear field-offset separation after one iteration."""

    movement = np.asarray(movement_parameters, dtype=np.float64).copy()
    eddy = np.asarray(quadratic_ec_parameters, dtype=np.float64).copy()
    shell_bvals = np.asarray(bvals, dtype=np.float64).reshape(-1)
    vectors = np.asarray(bvecs, dtype=np.float64)
    number_of_scans = movement.shape[0]
    if movement.shape != (number_of_scans, 6):
        raise ValueError("movement_parameters must have shape (N, 6)")
    if eddy.shape != (number_of_scans, 10):
        raise ValueError("quadratic_ec_parameters must have shape (N, 10)")
    if shell_bvals.shape != (number_of_scans,) or vectors.shape != (
        3,
        number_of_scans,
    ):
        raise ValueError("bvals and bvecs must match the parameter rows")

    normalized = np.empty_like(vectors)
    for scan in range(number_of_scans):
        norm = np.sqrt(
            vectors[0, scan] * vectors[0, scan]
            + vectors[2, scan] * vectors[2, scan]
            + vectors[1, scan] * vectors[1, scan]
        )
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError("DWI b-vectors must be finite and nonzero")
        normalized[:, scan] = vectors[:, scan] / norm

    design = normalized.T * (shell_bvals / 1000.0)[:, None]
    for column in range(design.shape[1]):
        column_mean = 0.0
        for row in range(design.shape[0]):
            column_mean += design[row, column]
        column_mean /= float(design.shape[0])
        for row in range(design.shape[0]):
            design[row, column] -= column_mean

    hz_to_mm = np.float64(
        voxel_sizes_mm[phase_encoding_axis]
        * phase_encoding_sign
        * readout_seconds
    )
    inverse_hz_to_mm = np.float64(1.0) / hz_to_mm
    hz = np.asarray(
        eddy[:, 9]
        + inverse_hz_to_mm * movement[:, phase_encoding_axis],
        dtype=np.float64,
    )
    try:
        projection = design @ np.linalg.inv(design.T @ design) @ design.T
    except np.linalg.LinAlgError as error:
        raise ValueError("DWI directions cannot fit FSL's linear offset model") from error
    fitted_hz = np.asarray(projection @ hz, dtype=np.float64)
    eddy[:, 9] = fitted_hz
    movement[:, phase_encoding_axis] = (hz - fitted_hz) * hz_to_mm
    movement = _apply_eddy_dwi_location_reference(
        movement, shape, voxel_sizes_mm
    )
    return movement, eddy


def _apply_eddy_dwi_location_reference(
    movement_parameters: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
    reference: int = 0,
) -> np.ndarray:
    """Apply FSL ``ECScanManager::set_reference`` to DWI movements."""

    movements = np.asarray(movement_parameters, dtype=np.float64)
    if movements.ndim != 2 or movements.shape[1] != 6:
        raise ValueError("movement_parameters must have shape (N, 6)")
    if not np.all(np.isfinite(movements)):
        raise ValueError("movement_parameters must be finite")
    if reference < 0 or reference >= movements.shape[0]:
        raise ValueError("reference must select one movement row")

    reference_inverse = _fsl_affine_inverse_4x4(
        _topup_movement_matrix(movements[reference], shape, voxel_sizes_mm)
    )
    referenced = np.empty_like(movements)
    for index, parameters in enumerate(movements):
        relative = _fsl_matrix_multiply_4x4(
            _topup_movement_matrix(parameters, shape, voxel_sizes_mm),
            reference_inverse,
        )
        referenced[index] = _topup_matrix_to_movement_parameters(
            relative, shape, voxel_sizes_mm
        )
    return referenced


def rotate_bvecs_eddy(
    bvecs: np.ndarray,
    bvals: np.ndarray,
    movement_parameters: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes_mm: tuple[float, float, float],
) -> np.ndarray:
    """Rotate b-vectors by FSL's inverse volume movement matrices."""

    vectors = np.asarray(bvecs, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != 3:
        raise ValueError("bvecs must have shape (3, N)")
    values = np.asarray(bvals, dtype=np.float64).reshape(-1)
    movements = np.asarray(movement_parameters, dtype=np.float64)
    if values.shape != (vectors.shape[1],):
        raise ValueError("bvals must contain one value per b-vector")
    if (
        movements.ndim != 2
        or movements.shape[0] != vectors.shape[1]
        or movements.shape[1] < 6
    ):
        raise ValueError("movement_parameters must have shape (N, at least 6)")
    if (
        not np.all(np.isfinite(vectors))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(movements[:, :6]))
    ):
        raise ValueError("b-vectors, b-values, and movement parameters must be finite")
    output = np.empty_like(vectors)
    for index in range(vectors.shape[1]):
        vector = vectors[:, index]
        if values[index] != 0.0:
            norm = np.linalg.norm(vector)
            if norm == 0.0:
                raise ValueError("nonzero b-values require nonzero b-vectors")
            vector = vector / norm
        matrix = _topup_movement_matrix(movements[index, :6], shape, voxel_sizes_mm)
        output[:, index] = np.linalg.inv(matrix)[:3, :3] @ vector
        if values[index] != 0.0:
            output[:, index] /= np.linalg.norm(output[:, index])
    return output


@njit(cache=True, parallel=True, nogil=True)
def _eddy_slice_statistics_fsl_order(
    observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate FSL ``DiffStats`` with x as the innermost dimension."""

    volume_count = observed.shape[3]
    slice_count = observed.shape[2]
    means = np.zeros((volume_count, slice_count), dtype=np.float64)
    means_squared = np.zeros((volume_count, slice_count), dtype=np.float64)
    counts = np.zeros((volume_count, slice_count), dtype=np.int64)
    for item in prange(volume_count * slice_count):
        volume = item // slice_count
        slice_index = item - volume * slice_count
        mean = 0.0
        mean_squared = 0.0
        count = 0
        for y in range(observed.shape[1]):
            for x in range(observed.shape[0]):
                if mask[x, y, slice_index, volume] != 0:
                    difference = np.float32(
                        observed[x, y, slice_index, volume]
                        - predicted[x, y, slice_index, volume]
                    )
                    mean += float(difference)
                    mean_squared += float(np.float32(difference * difference))
                    count += 1
        counts[volume, slice_index] = count
        if count:
            means[volume, slice_index] = mean / count
            means_squared[volume, slice_index] = mean_squared / count
    return means, means_squared, counts


def eddy_slice_statistics(
    observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> EddySliceStatistics:
    """Compute FSL ``DiffStats`` values with x-fastest slice accumulation."""

    observed_values = np.asarray(observed)
    predicted_values = np.asarray(predicted)
    mask_values = np.asarray(mask)
    if observed_values.ndim != 4 or predicted_values.shape != observed_values.shape:
        raise ValueError("observed and predicted must have matching shape (X, Y, Z, N)")
    if mask_values.shape not in (observed_values.shape[:3], observed_values.shape):
        raise ValueError("mask must be spatial or match the complete DWI shape")
    if not np.all(np.isfinite(observed_values)) or not np.all(
        np.isfinite(predicted_values)
    ):
        raise ValueError("observed and predicted values must be finite")
    if mask_values.ndim == 3:
        mask_values = np.broadcast_to(mask_values[..., None], observed_values.shape)
    means, means_squared, counts = _eddy_slice_statistics_fsl_order(
        np.ascontiguousarray(observed_values, dtype=np.float32),
        np.ascontiguousarray(predicted_values, dtype=np.float32),
        np.ascontiguousarray(mask_values, dtype=np.uint8),
    )
    return EddySliceStatistics(means, means_squared, counts)


def detect_eddy_slice_outliers(
    statistics: EddySliceStatistics,
    *,
    threshold_standard_deviations: float = 4.0,
    minimum_voxels: int = 250,
    consider_positive: bool = False,
    consider_squared: bool = False,
    previous_outlier_map: np.ndarray | None = None,
) -> EddyOutlierResult:
    """Apply FSL's default slice-wise ``ReplacementManager`` update."""

    mean = np.asarray(statistics.mean_difference, dtype=np.float64)
    squared = np.asarray(statistics.mean_squared_difference, dtype=np.float64)
    counts = np.asarray(statistics.voxel_count, dtype=np.int64)
    if mean.ndim != 2 or squared.shape != mean.shape or counts.shape != mean.shape:
        raise ValueError("slice statistics must have matching shape (N, Z)")
    if threshold_standard_deviations <= 0.0 or minimum_voxels < 1:
        raise ValueError("outlier threshold and minimum voxel count must be positive")
    eligible = counts >= minimum_voxels
    fit_eligible = eligible.copy()
    if previous_outlier_map is not None:
        previous = np.asarray(previous_outlier_map, dtype=bool)
        if previous.shape != mean.shape:
            raise ValueError("previous_outlier_map must match the statistics shape")
        fit_eligible &= ~previous
    item_count = int(np.count_nonzero(fit_eligible))
    if item_count < 2:
        raise ValueError(
            "at least two eligible slices are required for outlier detection"
        )
    mean_center = 0.0
    squared_center = 0.0
    total_voxels = 0
    for slice_index in range(mean.shape[1]):
        for volume in range(mean.shape[0]):
            if fit_eligible[volume, slice_index]:
                voxel_count = int(counts[volume, slice_index])
                mean_center += voxel_count * mean[volume, slice_index]
                squared_center += voxel_count * squared[volume, slice_index]
                total_voxels += voxel_count
    mean_center /= total_voxels
    squared_center /= total_voxels
    mean_variance = 0.0
    squared_variance = 0.0
    for slice_index in range(mean.shape[1]):
        for volume in range(mean.shape[0]):
            if fit_eligible[volume, slice_index]:
                voxel_count = int(counts[volume, slice_index])
                mean_delta = mean[volume, slice_index] - mean_center
                squared_delta = squared[volume, slice_index] - squared_center
                mean_variance += voxel_count * mean_delta * mean_delta
                squared_variance += voxel_count * squared_delta * squared_delta
    mean_std = np.sqrt(mean_variance / (item_count - 1))
    squared_std = np.sqrt(squared_variance / (item_count - 1))
    if mean_std == 0.0 or squared_std == 0.0:
        raise ValueError("slice residual variance must be nonzero")
    normalized = np.zeros_like(mean)
    normalized_squared = np.zeros_like(mean)
    scale = np.zeros_like(mean)
    scale[eligible] = 1.0 / np.sqrt(counts[eligible])
    normalized[eligible] = (mean[eligible] - mean_center) / (scale[eligible] * mean_std)
    normalized_squared[eligible] = (squared[eligible] - squared_center) / (
        scale[eligible] * squared_std
    )
    outliers = eligible & (-normalized > threshold_standard_deviations)
    if consider_positive:
        outliers |= eligible & (normalized > threshold_standard_deviations)
    if consider_squared:
        outliers |= eligible & (normalized_squared > threshold_standard_deviations)
    return EddyOutlierResult(outliers, normalized, normalized_squared)
