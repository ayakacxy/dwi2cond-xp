"""Affine parameterization and FSL/NIfTI coordinate conversions."""

from __future__ import annotations

from collections.abc import Sequence
import math

import nibabel as nib
from numba import njit
import numpy as np


def _finite_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if abs(np.linalg.det(values[:3, :3])) < 1e-12:
        raise ValueError(f"{name} must be invertible")
    return values


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a finite world-coordinate transform."""

    return np.linalg.inv(_finite_matrix(transform, "transform"))


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose transforms in application order.

    ``compose_transforms(a, b)`` returns ``b @ a``: points first pass through
    ``a`` and then through ``b``.
    """

    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = _finite_matrix(transform, "transform") @ result
    return result


def rigid_matrix(parameters: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """Construct the FSL Euler rigid transform around a physical center."""

    values = np.asarray(parameters, dtype=np.float64)
    origin = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must be a finite six-element vector")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite three-element vector")
    rx, ry, rz, tx, ty, tz = values
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    rotation_y = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    rotation_z = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    rotation = rotation_x @ rotation_y @ rotation_z
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = origin - rotation @ origin + np.array([tx, ty, tz])
    return result


def affine_matrix(parameters: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """Construct an FSL-style 6-through-12-DOF affine matrix.

    The 12 parameters are rotations, translations, scales and ``xy/xz/yz``
    skews. The composition is ``rigid @ skew @ scale`` and every component is
    centered on the same physical point, matching FSL ``compose_aff``.
    """

    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or not 6 <= values.size <= 12 or not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain six through twelve finite values")
    origin = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite three-element vector")
    result = rigid_matrix(values[:6], origin)
    if values.size == 6:
        return result
    scales = np.ones(3, dtype=np.float64)
    scales[0] = values[6]
    scales[1] = values[7] if values.size >= 8 else values[6]
    scales[2] = values[8] if values.size >= 9 else values[6]
    scale = np.diag([*scales, 1.0])
    if np.any(np.abs(scales) < 1e-12):
        raise ValueError("affine scales must be nonzero")
    scale[:3, 3] = origin - scale[:3, :3] @ origin
    skew = np.eye(4, dtype=np.float64)
    if values.size >= 10:
        skew[0, 1] = values[9]
    if values.size >= 11:
        skew[0, 2] = values[10]
    if values.size >= 12:
        skew[1, 2] = values[11]
    skew[:3, 3] = origin - skew[:3, :3] @ origin
    return result @ skew @ scale


@njit(cache=True, nogil=True)
def _affine_matrices_kernel(values: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Build a batch of FSL affine matrices in a native loop using the scalar formula."""

    results = np.empty((values.shape[0], 4, 4), dtype=np.float64)
    for index in range(values.shape[0]):
        rx, ry, rz = values[index, 0], values[index, 1], values[index, 2]
        tx, ty, tz = values[index, 3], values[index, 4], values[index, 5]
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rotation_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]],
            dtype=np.float64,
        )
        rotation_y = np.array(
            [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]],
            dtype=np.float64,
        )
        rotation_z = np.array(
            [[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rotation = rotation_x @ rotation_y @ rotation_z
        rigid = np.eye(4, dtype=np.float64)
        rigid[:3, :3] = rotation
        rigid[:3, 3] = origin - rotation @ origin + np.array([tx, ty, tz])
        if values.shape[1] == 6:
            results[index] = rigid
            continue
        scales = np.ones(3, dtype=np.float64)
        scales[0] = values[index, 6]
        scales[1] = values[index, 7] if values.shape[1] >= 8 else values[index, 6]
        scales[2] = values[index, 8] if values.shape[1] >= 9 else values[index, 6]
        scale = np.eye(4, dtype=np.float64)
        scale[0, 0], scale[1, 1], scale[2, 2] = scales[0], scales[1], scales[2]
        scale[0, 3] = origin[0] - scales[0] * origin[0]
        scale[1, 3] = origin[1] - scales[1] * origin[1]
        scale[2, 3] = origin[2] - scales[2] * origin[2]
        skew = np.eye(4, dtype=np.float64)
        if values.shape[1] >= 10:
            skew[0, 1] = values[index, 9]
        if values.shape[1] >= 11:
            skew[0, 2] = values[index, 10]
        if values.shape[1] >= 12:
            skew[1, 2] = values[index, 11]
        skew_linear = np.ascontiguousarray(skew[:3, :3])
        skew[:3, 3] = origin - skew_linear @ origin
        results[index] = rigid @ skew @ scale
    return results


def affine_matrices(
    parameters: np.ndarray, center: np.ndarray | None = None
) -> np.ndarray:
    """Build FSL affine matrices in batches, preserving ``affine_matrix`` ordering."""

    values = np.asarray(parameters, dtype=np.float64)
    origin = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    if (
        values.ndim != 2
        or not 6 <= values.shape[1] <= 12
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("parameters must have shape (n, 6..12) and be finite")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite three-element vector")
    if values.shape[1] >= 7 and np.any(np.abs(values[:, 6:9]) < 1e-12):
        raise ValueError("affine scales must be nonzero")
    return _affine_matrices_kernel(values, origin)


def decompose_affine(matrix: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """Decompose an affine using FSL's rotation-skew-scale convention."""

    transform = _finite_matrix(matrix, "matrix")
    origin = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite three-element vector")
    xaxis, yaxis, zaxis = transform[:3, 0], transform[:3, 1], transform[:3, 2]
    scale_x = np.float32(np.linalg.norm(xaxis))
    scale_y = np.float32(
        math.sqrt(
            float(
                np.dot(yaxis, yaxis)
                - (np.dot(xaxis, yaxis) ** 2 / float(scale_x * scale_x))
            )
        )
    )
    skew_xy = np.float32(np.dot(xaxis, yaxis) / float(scale_x * scale_y))
    xunit = xaxis / float(scale_x)
    yunit = yaxis / float(scale_y) - float(skew_xy) * xunit
    scale_z = np.float32(
        math.sqrt(
            float(
                np.dot(zaxis, zaxis)
                - np.dot(xunit, zaxis) ** 2
                - np.dot(yunit, zaxis) ** 2
            )
        )
    )
    skew_xz = np.float32(np.dot(xunit, zaxis) / float(scale_z))
    skew_yz = np.float32(np.dot(yunit, zaxis) / float(scale_z))
    scales = np.diag([float(scale_x), float(scale_y), float(scale_z)])
    skew = np.array(
        [[1.0, float(skew_xy), float(skew_xz)], [0.0, 1.0, float(skew_yz)], [0.0, 0.0, 1.0]]
    )
    rotation = transform[:3, :3] @ np.linalg.inv(scales) @ np.linalg.inv(skew)
    cosine_y = np.float32(math.sqrt(rotation[0, 0] ** 2 + rotation[0, 1] ** 2))
    if cosine_y < np.float32(1e-4):
        angle_x = np.float32(math.atan2(-rotation[2, 1], rotation[1, 1]))
        angle_y = np.float32(math.atan2(-rotation[0, 2], 0.0))
        angle_z = np.float32(0.0)
    else:
        angle_x = np.float32(math.atan2(rotation[1, 2] / cosine_y, rotation[2, 2] / cosine_y))
        angle_y = np.float32(math.atan2(-rotation[0, 2], cosine_y))
        angle_z = np.float32(math.atan2(rotation[0, 1] / cosine_y, rotation[0, 0] / cosine_y))
    translation = transform[:3, :3] @ origin + transform[:3, 3] - origin
    return np.asarray(
        [
            angle_x,
            angle_y,
            angle_z,
            *translation,
            scale_x,
            scale_y,
            scale_z,
            skew_xy,
            skew_xz,
            skew_yz,
        ],
        dtype=np.float64,
    )


def fsl_voxel_to_scaled_mm(
    shape: Sequence[int], affine: np.ndarray
) -> np.ndarray:
    """Return FSL's voxel-to-scaled-mm matrix for a NIfTI grid.

    FSL internally uses a radiological scaled-mm grid. Neurological NIfTI
    storage therefore receives an x-axis flip about the final voxel.
    """

    spatial_shape = tuple(int(value) for value in shape)
    world = _finite_matrix(affine, "affine")
    if len(spatial_shape) != 3 or any(value <= 0 for value in spatial_shape):
        raise ValueError("shape must contain three positive dimensions")
    voxel_sizes = nib.affines.voxel_sizes(world)
    result = np.diag([*voxel_sizes, 1.0])
    if np.linalg.det(world[:3, :3]) > 0:
        result[0, 0] *= -1.0
        result[0, 3] = (spatial_shape[0] - 1) * voxel_sizes[0]
    return result


def fsl_matrix_to_world(
    fsl_matrix: np.ndarray,
    moving_shape: Sequence[int],
    moving_affine: np.ndarray,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray,
) -> np.ndarray:
    """Convert a FLIRT source-to-reference matrix to NIfTI world coordinates."""

    fsl = _finite_matrix(fsl_matrix, "fsl_matrix")
    moving_world = _finite_matrix(moving_affine, "moving_affine")
    reference_world = _finite_matrix(reference_affine, "reference_affine")
    moving_fsl = fsl_voxel_to_scaled_mm(moving_shape, moving_world)
    reference_fsl = fsl_voxel_to_scaled_mm(reference_shape, reference_world)
    return (
        reference_world
        @ np.linalg.inv(reference_fsl)
        @ fsl
        @ moving_fsl
        @ np.linalg.inv(moving_world)
    )


def world_matrix_to_fsl(
    world_matrix: np.ndarray,
    moving_shape: Sequence[int],
    moving_affine: np.ndarray,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray,
) -> np.ndarray:
    """Convert a NIfTI world transform to a FLIRT source-to-reference matrix."""

    world = _finite_matrix(world_matrix, "world_matrix")
    moving_world = _finite_matrix(moving_affine, "moving_affine")
    reference_world = _finite_matrix(reference_affine, "reference_affine")
    moving_fsl = fsl_voxel_to_scaled_mm(moving_shape, moving_world)
    reference_fsl = fsl_voxel_to_scaled_mm(reference_shape, reference_world)
    return (
        reference_fsl
        @ np.linalg.inv(reference_world)
        @ world
        @ moving_world
        @ np.linalg.inv(moving_fsl)
    )
