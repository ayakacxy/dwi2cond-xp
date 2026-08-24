"""FSL BET 2.1-compatible brain-surface evolution and mask generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit, prange
from numba.extending import register_jitable
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from ._numba import set_available_numba_threads


@dataclass(frozen=True)
class BetResult:
    """Brain mask, final surface, and the initialization diagnostics."""

    mask: np.ndarray
    vertices_mm: np.ndarray
    faces: np.ndarray
    robust_min: float
    robust_max: float
    threshold: float
    center_mm: np.ndarray
    radius_mm: float
    median_intensity: float
    self_intersection_score: float
    passes: int


@njit(cache=True, fastmath=False)
def _sequential_mean(values: np.ndarray) -> float:
    total = 0.0
    for index in range(values.size):
        total += values[index]
    return total / values.size


@register_jitable
def _smoothing_increase(pass_number: int, iteration: int) -> float:
    increase = 10.0 ** (pass_number + 1)
    if iteration > 750:
        increase = 4.0 * (1.0 - iteration / 1000.0) * (increase - 1.0) + 1.0
    return increase


@register_jitable
def _optimized_smoothing(
    smoothing: float, normal_amount: float, pass_number: int, iteration: int
) -> float:
    if pass_number <= 0 or normal_amount <= 0.0:
        return smoothing
    return min(smoothing * _smoothing_increase(pass_number, iteration), 1.0)


@register_jitable
def _fit_force(
    inward_minimum: float, local_threshold: float, denominator: float
) -> float:
    if denominator > 0.0:
        return 2.0 * (inward_minimum - local_threshold) / denominator
    return 2.0 * (inward_minimum - local_threshold)


def _increase_outward_smoothing(
    smoothing: np.ndarray,
    normal_amount: np.ndarray,
    pass_number: int,
    iteration: int,
) -> np.ndarray:
    if pass_number <= 0:
        return smoothing
    increased = smoothing.copy()
    outward = normal_amount > 0
    increase = _smoothing_increase(pass_number, iteration)
    increased[outward] = np.minimum(increased[outward] * increase, 1.0)
    return increased


@njit(cache=True, parallel=True, fastmath=False)
def _evolve_surface_optimized(
    image: np.ndarray,
    initial: np.ndarray,
    faces: np.ndarray,
    neighbours: np.ndarray,
    neighbour_valid: np.ndarray,
    incident_faces: np.ndarray,
    incident_valid: np.ndarray,
    voxel_sizes: np.ndarray,
    threshold_2: float,
    threshold: float,
    median: float,
    main_parameter: float,
    center_z: float,
    radius: float,
    gradient_threshold: float,
    pass_number: int,
) -> np.ndarray:
    vertex_count = initial.shape[0]
    face_count = faces.shape[0]
    evolved = initial.copy()
    updated = initial.copy()
    triangle_normals = np.empty((face_count, 3), dtype=np.float64)
    normals = np.empty((vertex_count, 3), dtype=np.float64)
    differences = np.empty((vertex_count, 3), dtype=np.float64)
    mean_distances = np.empty(vertex_count, dtype=np.float64)
    mean_edge = 0.0
    curvature_center = (1.0 / 3.33 + 1.0 / 10.0) * 0.5
    curvature_scale = 6.0 / (1.0 / 3.33 - 1.0 / 10.0)
    shape_x, shape_y, shape_z = image.shape
    for iteration in range(1000):
        for face_index in prange(face_count):
            first = faces[face_index, 0]
            second = faces[face_index, 1]
            third = faces[face_index, 2]
            ax = evolved[third, 0] - evolved[first, 0]
            ay = evolved[third, 1] - evolved[first, 1]
            az = evolved[third, 2] - evolved[first, 2]
            bx = evolved[second, 0] - evolved[first, 0]
            by = evolved[second, 1] - evolved[first, 1]
            bz = evolved[second, 2] - evolved[first, 2]
            triangle_normals[face_index, 0] = ay * bz - az * by
            triangle_normals[face_index, 1] = az * bx - ax * bz
            triangle_normals[face_index, 2] = ax * by - ay * bx
        update_mean_edge = iteration == 0 or iteration == 50 or iteration % 100 == 0
        for vertex in prange(vertex_count):
            nx = 0.0
            ny = 0.0
            nz = 0.0
            for slot in range(incident_faces.shape[1]):
                if incident_valid[vertex, slot]:
                    face_index = incident_faces[vertex, slot]
                    nx += triangle_normals[face_index, 0]
                    ny += triangle_normals[face_index, 1]
                    nz += triangle_normals[face_index, 2]
            normal_length = np.sqrt(nx * nx + ny * ny + nz * nz)
            normals[vertex, 0] = nx / normal_length
            normals[vertex, 1] = ny / normal_length
            normals[vertex, 2] = nz / normal_length
            mx = 0.0
            my = 0.0
            mz = 0.0
            distance_sum = 0.0
            neighbour_count = 0
            for slot in range(neighbours.shape[1]):
                if neighbour_valid[vertex, slot]:
                    neighbour = neighbours[vertex, slot]
                    mx += evolved[neighbour, 0]
                    my += evolved[neighbour, 1]
                    mz += evolved[neighbour, 2]
                    if update_mean_edge:
                        dx = evolved[neighbour, 0] - evolved[vertex, 0]
                        dy = evolved[neighbour, 1] - evolved[vertex, 1]
                        dz = evolved[neighbour, 2] - evolved[vertex, 2]
                        distance_sum += np.sqrt(dx * dx + dy * dy + dz * dz)
                    neighbour_count += 1
            differences[vertex, 0] = mx / neighbour_count - evolved[vertex, 0]
            differences[vertex, 1] = my / neighbour_count - evolved[vertex, 1]
            differences[vertex, 2] = mz / neighbour_count - evolved[vertex, 2]
            if update_mean_edge:
                mean_distances[vertex] = distance_sum / neighbour_count
        if update_mean_edge:
            mean_edge = _sequential_mean(mean_distances)
        for vertex in prange(vertex_count):
            nx = normals[vertex, 0]
            ny = normals[vertex, 1]
            nz = normals[vertex, 2]
            dx = differences[vertex, 0]
            dy = differences[vertex, 1]
            dz = differences[vertex, 2]
            normal_amount = dx * nx + dy * ny + dz * nz
            normal_x = nx * normal_amount
            normal_y = ny * normal_amount
            normal_z = nz * normal_amount
            inverse_radius = 2.0 * abs(normal_amount) / (mean_edge * mean_edge)
            smoothing = 0.5 * (
                1.0 + np.tanh(curvature_scale * (inverse_radius - curvature_center))
            )
            smoothing = _optimized_smoothing(
                smoothing, normal_amount, pass_number, iteration
            )
            local_parameter = main_parameter
            if gradient_threshold != 0.0:
                local_parameter += (
                    gradient_threshold * (evolved[vertex, 2] - center_z) / radius
                )
                local_parameter = min(1.0, max(0.0, local_parameter))
            first_x = int((evolved[vertex, 0] - nx) / voxel_sizes[0] + 0.5)
            first_y = int((evolved[vertex, 1] - ny) / voxel_sizes[1] + 0.5)
            first_z = int((evolved[vertex, 2] - nz) / voxel_sizes[2] + 0.5)
            last_x = int((evolved[vertex, 0] - 7.0 * nx) / voxel_sizes[0] + 0.5)
            last_y = int((evolved[vertex, 1] - 7.0 * ny) / voxel_sizes[1] + 0.5)
            last_z = int((evolved[vertex, 2] - 7.0 * nz) / voxel_sizes[2] + 0.5)
            valid_path = (
                0 <= first_x < shape_x
                and 0 <= first_y < shape_y
                and 0 <= first_z < shape_z
                and 0 <= last_x < shape_x
                and 0 <= last_y < shape_y
                and 0 <= last_z < shape_z
            )
            inward_minimum = median
            inward_maximum = threshold
            if valid_path:
                for distance in range(1, 7):
                    voxel_x = int(
                        (evolved[vertex, 0] - distance * nx) / voxel_sizes[0] + 0.5
                    )
                    voxel_y = int(
                        (evolved[vertex, 1] - distance * ny) / voxel_sizes[1] + 0.5
                    )
                    voxel_z = int(
                        (evolved[vertex, 2] - distance * nz) / voxel_sizes[2] + 0.5
                    )
                    sample = image[voxel_x, voxel_y, voxel_z]
                    inward_minimum = min(inward_minimum, sample)
                    if distance <= 2:
                        inward_maximum = max(inward_maximum, sample)
                inward_minimum = max(threshold_2, inward_minimum)
                inward_maximum = min(median, inward_maximum)
            local_threshold = (
                inward_maximum - threshold_2
            ) * local_parameter + threshold_2
            denominator = inward_maximum - threshold_2
            fit = _fit_force(inward_minimum, local_threshold, denominator)
            fit *= 0.05 * mean_edge
            updated[vertex, 0] = (
                evolved[vertex, 0]
                + 0.5 * (dx - normal_x)
                + smoothing * normal_x
                + fit * nx
            )
            updated[vertex, 1] = (
                evolved[vertex, 1]
                + 0.5 * (dy - normal_y)
                + smoothing * normal_y
                + fit * ny
            )
            updated[vertex, 2] = (
                evolved[vertex, 2]
                + 0.5 * (dz - normal_z)
                + smoothing * normal_z
                + fit * nz
            )
        swap = evolved
        evolved = updated
        updated = swap
    return evolved


def robust_intensity_limits(values: np.ndarray) -> tuple[float, float]:
    """Reproduce the iterative 1,000-bin robust limits used by FSL NEWIMAGE."""

    data = np.asarray(values, dtype=np.float32)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("BET input must be a non-empty finite array")
    full_min = np.float32(np.min(data))
    full_max = np.float32(np.max(data))
    minimum = full_min
    maximum = full_max
    low_bin = 0
    high_bin = 999
    pass_number = 1
    threshold_2 = np.float32(0.0)
    threshold_98 = np.float32(0.0)
    while (
        pass_number == 1
        or float(threshold_98 - threshold_2) < float(maximum - minimum) / 10.0
    ):
        if pass_number > 1:
            low_bin = max(low_bin - 1, 0)
            high_bin = min(high_bin + 1, 999)
            old_minimum = minimum
            minimum = np.float32(
                old_minimum + (low_bin / 1000.0) * (maximum - old_minimum)
            )
            maximum = np.float32(
                old_minimum + ((high_bin + 1) / 1000.0) * (maximum - old_minimum)
            )
        if pass_number == 10 or minimum == maximum:
            minimum = full_min
            maximum = full_max
        if minimum == maximum:
            return float(minimum), float(maximum)
        scaled = (
            1000.0
            * (data.astype(np.float64) - float(minimum))
            / float(maximum - minimum)
        )
        bins = np.clip(scaled, 0.0, 999.0).astype(np.int64)
        histogram = np.bincount(bins.ravel(), minlength=1000)
        valid_size = int(data.size)
        if pass_number == 10:
            valid_size -= int(histogram[low_bin] + histogram[high_bin])
            low_bin += 1
            high_bin -= 1
        target = valid_size // 50
        count = 0
        bottom_bin = low_bin
        while count < target:
            count += int(histogram[bottom_bin])
            bottom_bin += 1
        bottom_bin -= 1
        count = 0
        top_bin = high_bin
        while count < target:
            count += int(histogram[top_bin])
            top_bin -= 1
        top_bin += 1
        bin_width = float(maximum - minimum) / 1000.0
        threshold_2 = np.float32(float(minimum) + bottom_bin * bin_width)
        threshold_98 = np.float32(float(minimum) + (top_bin + 1) * bin_width)
        if pass_number == 10:
            break
        low_bin = bottom_bin
        high_bin = top_bin
        pass_number += 1
    return float(threshold_2), float(threshold_98)


def _icosphere(order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    tau = 0.8506508084
    one = 0.5257311121
    vertices = np.array(
        [
            [tau, one, 0],
            [-tau, one, 0],
            [-tau, -one, 0],
            [tau, -one, 0],
            [one, 0, tau],
            [one, 0, -tau],
            [-one, 0, -tau],
            [-one, 0, tau],
            [0, tau, one],
            [0, -tau, one],
            [0, -tau, -one],
            [0, tau, -one],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [4, 8, 7],
            [4, 7, 9],
            [5, 6, 11],
            [5, 10, 6],
            [0, 4, 3],
            [0, 3, 5],
            [2, 7, 1],
            [2, 1, 6],
            [8, 0, 11],
            [8, 11, 1],
            [9, 10, 3],
            [9, 2, 10],
            [8, 4, 0],
            [11, 0, 5],
            [4, 9, 3],
            [5, 3, 10],
            [7, 8, 1],
            [6, 1, 11],
            [7, 2, 9],
            [6, 10, 2],
        ],
        dtype=np.int32,
    )
    faces[:, [1, 2]] = faces[:, [2, 1]]
    for _ in range(1, order):
        midpoint_indices: dict[tuple[int, int], int] = {}
        expanded = vertices.tolist()

        def midpoint(first: int, second: int) -> int:
            edge = (min(first, second), max(first, second))
            if edge not in midpoint_indices:
                midpoint_indices[edge] = len(expanded)
                expanded.append(((vertices[first] + vertices[second]) * 0.5).tolist())
            return midpoint_indices[edge]

        refined = []
        for first, second, third in faces:
            opposite_first = midpoint(int(second), int(third))
            opposite_second = midpoint(int(first), int(third))
            opposite_third = midpoint(int(first), int(second))
            refined.extend(
                [
                    [opposite_third, opposite_first, opposite_second],
                    [opposite_second, int(first), opposite_third],
                    [opposite_first, int(third), opposite_second],
                    [opposite_third, int(second), opposite_first],
                ]
            )
        vertices = np.asarray(expanded, dtype=np.float64)
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
        faces = np.asarray(refined, dtype=np.int32)
    return vertices, faces


def _mesh_arrays(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    neighbours: list[list[int]] = [[] for _ in range(vertex_count)]

    def move_to_end(vertex: int, first: int, second: int) -> None:
        items = neighbours[vertex]
        if first in items:
            items.remove(first)
        if second in items:
            items.remove(second)
        items.extend((first, second))

    for first, second, third in faces:
        move_to_end(int(first), int(second), int(third))
        move_to_end(int(second), int(third), int(first))
        move_to_end(int(third), int(first), int(second))
    width = max(len(items) for items in neighbours)
    indices = np.zeros((vertex_count, width), dtype=np.int32)
    valid = np.zeros((vertex_count, width), dtype=bool)
    for index, items in enumerate(neighbours):
        indices[index, : len(items)] = items
        valid[index, : len(items)] = True
    return indices, valid


def _incident_face_arrays(
    faces: np.ndarray, vertex_count: int
) -> tuple[np.ndarray, np.ndarray]:
    incident: list[list[int]] = [[] for _ in range(vertex_count)]
    for face_index, face in enumerate(faces):
        for vertex in face:
            incident[int(vertex)].append(face_index)
    width = max(len(items) for items in incident)
    indices = np.zeros((vertex_count, width), dtype=np.int32)
    valid = np.zeros((vertex_count, width), dtype=np.bool_)
    for vertex, items in enumerate(incident):
        indices[vertex, : len(items)] = items
        valid[vertex, : len(items)] = True
    return indices, valid


def _mesh_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_incidence: csr_matrix,
    neighbour_average: csr_matrix,
    neighbours: np.ndarray,
    neighbour_valid: np.ndarray,
    *,
    measure_distances: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_edge = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    second_edge = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    triangle_normals = np.empty_like(first_edge)
    triangle_normals[:, 0] = (
        first_edge[:, 1] * second_edge[:, 2] - first_edge[:, 2] * second_edge[:, 1]
    )
    triangle_normals[:, 1] = (
        first_edge[:, 2] * second_edge[:, 0] - first_edge[:, 0] * second_edge[:, 2]
    )
    triangle_normals[:, 2] = (
        first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    )
    normals = np.asarray(face_incidence @ triangle_normals)
    norms = np.sqrt(np.sum(normals * normals, axis=1))
    if np.any(norms == 0):
        raise RuntimeError("BET surface contains a vertex with no valid normal")
    normals /= norms[:, None]
    means = np.asarray(neighbour_average @ vertices)
    difference = means - vertices
    if measure_distances:
        selected = vertices[neighbours]
        distances = np.linalg.norm(selected - vertices[:, None, :], axis=2)
        mean_distance = np.sum(
            distances * neighbour_valid, axis=1
        ) / neighbour_valid.sum(axis=1)
    else:
        mean_distance = np.empty(0, dtype=np.float64)
    return normals, difference, mean_distance


def _sample_inward_extrema(
    image: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    voxel_sizes: np.ndarray,
    threshold_2: float,
    threshold: float,
    median: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.arange(1.0, 8.0)[:, None]
    base = vertices / voxel_sizes + 0.5
    step = normals / voxel_sizes
    voxel_x = (base[None, :, 0] - distances * step[None, :, 0]).astype(np.int64)
    voxel_y = (base[None, :, 1] - distances * step[None, :, 1]).astype(np.int64)
    voxel_z = (base[None, :, 2] - distances * step[None, :, 2]).astype(np.int64)
    bounds = np.asarray(image.shape, dtype=np.int64)
    valid_path = (
        (voxel_x[0] >= 0)
        & (voxel_x[0] < bounds[0])
        & (voxel_y[0] >= 0)
        & (voxel_y[0] < bounds[1])
        & (voxel_z[0] >= 0)
        & (voxel_z[0] < bounds[2])
        & (voxel_x[6] >= 0)
        & (voxel_x[6] < bounds[0])
        & (voxel_y[6] >= 0)
        & (voxel_y[6] < bounds[1])
        & (voxel_z[6] >= 0)
        & (voxel_z[6] < bounds[2])
    )
    samples = image[
        np.clip(voxel_x[:6], 0, bounds[0] - 1),
        np.clip(voxel_y[:6], 0, bounds[1] - 1),
        np.clip(voxel_z[:6], 0, bounds[2] - 1),
    ]
    minimum = np.minimum(median, np.min(samples, axis=0))
    maximum = np.maximum(threshold, np.max(samples[:2], axis=0))
    minimum = np.maximum(minimum, threshold_2).astype(np.float64, copy=False)
    maximum = np.minimum(maximum, median).astype(np.float64, copy=False)
    minimum[~valid_path] = median
    maximum[~valid_path] = threshold
    return minimum, maximum


def _self_intersection_score(
    vertices: np.ndarray,
    original: np.ndarray,
    mean_edge: float,
    original_mean_edge: float,
) -> float:
    pairs = cKDTree(vertices).query_pairs(mean_edge, output_type="ndarray")
    if pairs.size == 0:
        return 0.0
    current_distance = np.linalg.norm(
        vertices[pairs[:, 0]] - vertices[pairs[:, 1]], axis=1
    )
    original_distance = np.linalg.norm(
        original[pairs[:, 0]] - original[pairs[:, 1]], axis=1
    )
    return float(
        np.sum(
            (current_distance / mean_edge - original_distance / original_mean_edge) ** 2
        )
    )


def _evolve_surface(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    voxel_sizes: np.ndarray,
    threshold_2: float,
    threshold: float,
    median: float,
    fractional_threshold: float,
    center_z: float,
    radius: float,
    gradient_threshold: float,
    *,
    pass_number: int = 0,
    backend: str = "optimized",
    workers: int = 8,
) -> tuple[np.ndarray, float, float]:
    neighbours, neighbour_valid = _mesh_arrays(faces, vertices.shape[0])
    flat_faces = faces.reshape(-1)
    face_ids = np.repeat(np.arange(faces.shape[0], dtype=np.int32), 3)
    face_incidence = csr_matrix(
        (np.ones(flat_faces.size), (flat_faces, face_ids)),
        shape=(vertices.shape[0], faces.shape[0]),
    )
    neighbour_counts = neighbour_valid.sum(axis=1)
    neighbour_average = csr_matrix(
        (
            np.repeat(1.0 / neighbour_counts, neighbour_counts),
            (
                np.repeat(
                    np.arange(vertices.shape[0], dtype=np.int32), neighbour_counts
                ),
                neighbours[neighbour_valid],
            ),
        ),
        shape=(vertices.shape[0], vertices.shape[0]),
    )
    original = vertices.copy()
    _, _, original_distances = _mesh_geometry(
        original,
        faces,
        face_incidence,
        neighbour_average,
        neighbours,
        neighbour_valid,
    )
    original_mean_edge = float(np.mean(original_distances))
    if backend == "optimized":
        incident_faces, incident_valid = _incident_face_arrays(faces, vertices.shape[0])
        set_available_numba_threads(workers)
        evolved = _evolve_surface_optimized(
            image,
            original,
            faces,
            neighbours,
            neighbour_valid,
            incident_faces,
            incident_valid,
            voxel_sizes,
            threshold_2,
            threshold,
            median,
            fractional_threshold**0.275,
            center_z,
            radius,
            gradient_threshold,
            pass_number,
        )
        _, _, final_distances = _mesh_geometry(
            evolved,
            faces,
            face_incidence,
            neighbour_average,
            neighbours,
            neighbour_valid,
        )
        mean_edge = float(np.mean(final_distances))
        score = _self_intersection_score(
            evolved, original, mean_edge, original_mean_edge
        )
        return evolved, score, mean_edge
    evolved = vertices.copy()
    mean_edge = 0.0
    main_parameter = fractional_threshold**0.275
    minimum_radius = 3.33
    maximum_radius = 10.0
    curvature_center = (1.0 / minimum_radius + 1.0 / maximum_radius) * 0.5
    curvature_scale = 6.0 / (1.0 / minimum_radius - 1.0 / maximum_radius)
    for iteration in range(1000):
        update_mean_edge = iteration == 0 or iteration == 50 or iteration % 100 == 0
        normals, difference, distances = _mesh_geometry(
            evolved,
            faces,
            face_incidence,
            neighbour_average,
            neighbours,
            neighbour_valid,
            measure_distances=update_mean_edge,
        )
        if update_mean_edge:
            mean_edge = float(np.mean(distances))
        normal_amount = np.sum(difference * normals, axis=1)
        normal_difference = normals * normal_amount[:, None]
        tangential = difference - normal_difference
        inverse_radius = 2.0 * np.abs(normal_amount) / (mean_edge * mean_edge)
        smoothing = 0.5 * (
            1.0 + np.tanh(curvature_scale * (inverse_radius - curvature_center))
        )
        smoothing = _increase_outward_smoothing(
            smoothing, normal_amount, pass_number, iteration
        )
        if gradient_threshold == 0.0:
            local_parameter = main_parameter
        else:
            local_parameter = np.clip(
                main_parameter
                + gradient_threshold * (evolved[:, 2] - center_z) / radius,
                0.0,
                1.0,
            )
        inward_minimum, inward_maximum = _sample_inward_extrema(
            image,
            evolved,
            normals,
            voxel_sizes,
            threshold_2,
            threshold,
            median,
        )
        local_threshold = (inward_maximum - threshold_2) * local_parameter + threshold_2
        denominator = inward_maximum - threshold_2
        fit = np.where(
            denominator > 0,
            2.0 * (inward_minimum - local_threshold) / denominator,
            2.0 * (inward_minimum - local_threshold),
        )
        fit *= 0.05 * mean_edge
        evolved += (
            0.5 * tangential
            + smoothing[:, None] * normal_difference
            + fit[:, None] * normals
        )
    _, _, final_distances = _mesh_geometry(
        evolved,
        faces,
        face_incidence,
        neighbour_average,
        neighbours,
        neighbour_valid,
    )
    mean_edge = float(np.mean(final_distances))
    score = _self_intersection_score(evolved, original, mean_edge, original_mean_edge)
    return evolved, score, mean_edge


def _draw_surface_chunk(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes: np.ndarray,
) -> np.ndarray:
    barrier = np.zeros(shape, dtype=bool)
    step_mm = float(np.min(voxel_sizes)) * 0.5
    for face in faces:
        first, second, third = vertices_mm[face]
        edge = first - second
        edge_length = float(np.linalg.norm(edge))
        edge_direction = edge / edge_length
        for distance in np.arange(0.0, edge_length + np.finfo(float).eps, step_mm):
            edge_point = second + distance * edge_direction
            cross_edge = edge_point - third
            cross_length = float(np.linalg.norm(cross_edge))
            cross_direction = cross_edge / cross_length
            cross_distances = np.arange(
                0.0, cross_length + np.finfo(float).eps, step_mm
            )
            samples = third + cross_distances[:, None] * cross_direction
            voxels = np.floor(samples / voxel_sizes + 0.5).astype(np.int64)
            valid = np.all((voxels >= 0) & (voxels < np.asarray(shape)), axis=1)
            voxels = voxels[valid]
            barrier[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    return barrier


def _rasterize_surface(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    shape: tuple[int, int, int],
    voxel_sizes: np.ndarray,
    *,
    workers: int,
) -> np.ndarray:
    chunks = np.array_split(faces, min(workers, faces.shape[0]))
    if len(chunks) == 1:
        barrier = _draw_surface_chunk(vertices_mm, chunks[0], shape, voxel_sizes)
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    _draw_surface_chunk, vertices_mm, chunk, shape, voxel_sizes
                )
                for chunk in chunks
            ]
            barriers = [future.result() for future in futures]
        barrier = np.logical_or.reduce(barriers)
    components, _ = label(~barrier)
    center_voxel = np.floor(np.mean(vertices_mm, axis=0) / voxel_sizes).astype(int)
    center_voxel = np.clip(center_voxel, 0, np.asarray(shape) - 1)
    component = components[tuple(center_voxel)]
    if component == 0:
        raise RuntimeError("BET surface centroid lies on the rasterized boundary")
    return (barrier | (components == component)).astype(np.uint8)


def bet_brain_mask(
    values: np.ndarray,
    voxel_sizes: np.ndarray,
    *,
    fractional_threshold: float = 0.2,
    gradient_threshold: float = 0.0,
    workers: int = 8,
    backend: str = "optimized",
) -> BetResult:
    """Extract a brain mask with the BET 2.1 surface-evolution algorithm."""

    image = np.asarray(values, dtype=np.float32)
    sizes = np.asarray(voxel_sizes, dtype=np.float64)
    if image.ndim != 3 or any(size <= 0 for size in image.shape):
        raise ValueError("BET input must be a non-empty three-dimensional array")
    if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("Voxel sizes must be three finite positive values")
    if not np.all(np.isfinite(image)):
        raise ValueError("BET input must contain only finite values")
    if not 0.0 < fractional_threshold < 1.0:
        raise ValueError("The fractional threshold must be between zero and one")
    if not -1.0 <= gradient_threshold <= 1.0:
        raise ValueError("The gradient threshold must be between minus one and one")
    if workers <= 0:
        raise ValueError("The worker count must be positive")
    if backend not in ("reference", "optimized"):
        raise ValueError("The BET backend must be reference or optimized")
    threshold_2, threshold_98 = robust_intensity_limits(image)
    threshold = threshold_2 + 0.1 * (threshold_98 - threshold_2)
    weights = np.clip(
        image.astype(np.float64) - threshold_2, 0.0, threshold_98 - threshold_2
    )
    weights[image <= threshold] = 0.0
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("BET could not initialize a center from the input intensities")
    coordinates = np.indices(image.shape, dtype=np.float64)
    center = np.array(
        [
            float(np.sum(coordinates[axis] * weights) / total) * sizes[axis]
            for axis in range(3)
        ]
    )
    count = int(np.count_nonzero(image > threshold))
    radius = float((0.75 * count * float(np.prod(sizes)) / np.pi) ** (1.0 / 3.0))
    physical = coordinates * sizes[:, None, None, None]
    within = (
        np.sum((physical - center[:, None, None, None]) ** 2, axis=0) < radius * radius
    )
    median_values = image[(image > threshold_2) & (image < threshold_98) & within]
    if median_values.size < 2:
        raise ValueError("BET could not initialize the within-brain median intensity")
    median = float(
        np.partition(median_values, median_values.size // 2 - 1)[
            median_values.size // 2 - 1
        ]
    )
    vertices, faces = _icosphere(5)
    vertices = center + vertices * (radius * 0.5)
    original = vertices.copy()
    passes = 0
    while True:
        vertices, score, _ = _evolve_surface(
            image,
            original,
            faces,
            sizes,
            threshold_2,
            threshold,
            median,
            fractional_threshold,
            center[2],
            radius,
            gradient_threshold,
            pass_number=passes,
            backend=backend,
            workers=workers,
        )
        if score <= 4000.0 or passes == 10:
            break
        passes += 1
    mask = _rasterize_surface(vertices, faces, image.shape, sizes, workers=workers)
    return BetResult(
        mask=mask,
        vertices_mm=vertices,
        faces=faces,
        robust_min=threshold_2,
        robust_max=threshold_98,
        threshold=threshold,
        center_mm=center,
        radius_mm=radius,
        median_intensity=median,
        self_intersection_score=score,
        passes=passes,
    )


def write_bet_brain_mask(
    input_file: str | Path,
    output_mask_file: str | Path,
    *,
    fractional_threshold: float = 0.2,
    gradient_threshold: float = 0.0,
    workers: int = 8,
    backend: str = "optimized",
) -> BetResult:
    """Run BET on a 3D NIfTI and write a uint8 mask with matching geometry."""

    image = nib.load(str(input_file))
    if len(image.shape) != 3:
        raise ValueError("BET NIfTI input must be three-dimensional")
    values = np.asarray(image.dataobj, dtype=np.float32)
    result = bet_brain_mask(
        values,
        np.asarray(image.header.get_zooms()[:3], dtype=np.float64),
        fractional_threshold=fractional_threshold,
        gradient_threshold=gradient_threshold,
        workers=workers,
        backend=backend,
    )
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(result.mask, image.affine, header)
    output.set_qform(*image.get_qform(coded=True))
    output.set_sform(*image.get_sform(coded=True))
    nib.save(output, str(output_mask_file))
    return result
