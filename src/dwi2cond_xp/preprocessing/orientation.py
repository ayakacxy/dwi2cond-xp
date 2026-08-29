"""Finite NIfTI geometry and orientation operations used by dwi2cond."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ..registration import matrix_to_tensor6, tensor6_to_matrix


def _f32(value: float | np.floating) -> np.float32:
    return np.float32(value)


def _f32_norm(x: np.float32, y: np.float32, z: np.float32) -> np.float32:
    squared = _f32(_f32(x * x) + _f32(y * y))
    squared = _f32(squared + _f32(z * z))
    return _f32(np.sqrt(squared))


def _f32_dot(
    x1: np.float32,
    y1: np.float32,
    z1: np.float32,
    x2: np.float32,
    y2: np.float32,
    z2: np.float32,
) -> np.float32:
    value = _f32(_f32(x1 * x2) + _f32(y1 * y2))
    return _f32(value + _f32(z1 * z2))


def _fsl_determinant(matrix: np.ndarray) -> np.float32:
    values = np.asarray(matrix, dtype=np.float32)
    r11, r12, r13 = (float(value) for value in values[0])
    r21, r22, r23 = (float(value) for value in values[1])
    r31, r32, r33 = (float(value) for value in values[2])
    return _f32(
        r11 * r22 * r33
        - r11 * r32 * r23
        - r21 * r12 * r33
        + r21 * r32 * r13
        + r31 * r12 * r23
        - r31 * r22 * r13
    )


def _fsl_axis_orientation(affine: np.ndarray) -> np.ndarray:
    q = np.asarray(affine[:3, :3], dtype=np.float32).copy()

    norm = _f32_norm(q[0, 0], q[1, 0], q[2, 0])
    if norm == 0.0:
        raise ValueError("The orientation affine has a zero-length spatial axis")
    q[:, 0] = np.asarray([_f32(value / norm) for value in q[:, 0]])

    norm = _f32_norm(q[0, 1], q[1, 1], q[2, 1])
    if norm == 0.0:
        raise ValueError("The orientation affine has a zero-length spatial axis")
    q[:, 1] = np.asarray([_f32(value / norm) for value in q[:, 1]])

    projection = _f32_dot(*q[:, 0], *q[:, 1])
    if abs(float(projection)) > 1.0e-4:
        q[:, 1] = np.asarray(
            [_f32(q[row, 1] - _f32(projection * q[row, 0])) for row in range(3)]
        )
        norm = _f32_norm(*q[:, 1])
        if norm == 0.0:
            raise ValueError("The orientation affine has parallel spatial axes")
        q[:, 1] = np.asarray([_f32(value / norm) for value in q[:, 1]])

    norm = _f32_norm(*q[:, 2])
    if norm == 0.0:
        q[:, 2] = np.asarray(
            [
                _f32(q[1, 0] * q[2, 1] - q[2, 0] * q[1, 1]),
                _f32(q[2, 0] * q[0, 1] - q[0, 0] * q[2, 1]),
                _f32(q[0, 0] * q[1, 1] - q[1, 0] * q[0, 1]),
            ],
            dtype=np.float32,
        )
    else:
        q[:, 2] = np.asarray([_f32(value / norm) for value in q[:, 2]])

    projection = _f32_dot(*q[:, 0], *q[:, 2])
    if abs(float(projection)) > 1.0e-4:
        q[:, 2] = np.asarray(
            [_f32(q[row, 2] - _f32(projection * q[row, 0])) for row in range(3)]
        )
        norm = _f32_norm(*q[:, 2])
        if norm == 0.0:
            raise ValueError("The orientation affine has dependent spatial axes")
        q[:, 2] = np.asarray([_f32(value / norm) for value in q[:, 2]])

    projection = _f32_dot(*q[:, 1], *q[:, 2])
    if abs(float(projection)) > 1.0e-4:
        q[:, 2] = np.asarray(
            [_f32(q[row, 2] - _f32(projection * q[row, 1])) for row in range(3)]
        )
        norm = _f32_norm(*q[:, 2])
        if norm == 0.0:
            raise ValueError("The orientation affine has dependent spatial axes")
        q[:, 2] = np.asarray([_f32(value / norm) for value in q[:, 2]])

    determinant = _fsl_determinant(q)
    if determinant == 0.0:
        raise ValueError("The orientation affine has singular normalized axes")

    best_trace = _f32(-666.0)
    best = (0, 1, 2, 1, 1, 1)
    for first_axis in range(3):
        for second_axis in range(3):
            if first_axis == second_axis:
                continue
            for third_axis in range(3):
                if third_axis in (first_axis, second_axis):
                    continue
                for first_sign in (-1, 1):
                    for second_sign in (-1, 1):
                        for third_sign in (-1, 1):
                            permutation = np.zeros((3, 3), dtype=np.float32)
                            permutation[0, first_axis] = first_sign
                            permutation[1, second_axis] = second_sign
                            permutation[2, third_axis] = third_sign
                            if _fsl_determinant(permutation) * determinant <= 0.0:
                                continue
                            product = np.empty((3, 3), dtype=np.float32)
                            for row in range(3):
                                for column in range(3):
                                    value = _f32(
                                        permutation[row, 0] * q[0, column]
                                        + permutation[row, 1] * q[1, column]
                                    )
                                    product[row, column] = _f32(
                                        value + permutation[row, 2] * q[2, column]
                                    )
                            trace = _f32(_f32(product[0, 0] + product[1, 1]) + product[2, 2])
                            if trace > best_trace:
                                best_trace = trace
                                best = (
                                    first_axis,
                                    second_axis,
                                    third_axis,
                                    first_sign,
                                    second_sign,
                                    third_sign,
                                )
    return np.asarray(
        [[best[0], best[3]], [best[1], best[4]], [best[2], best[5]]],
        dtype=np.float64,
    )


def fsl_canonical_orientation(affine: np.ndarray) -> np.ndarray:
    """Return the storage transform used by FSL ``fslreorient2std``.

    FSL preserves the input storage handedness: positive-determinant images
    are arranged as RAS and negative-determinant images as LAS.
    """

    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("The orientation affine must be a finite 4x4 matrix")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if abs(determinant) < 1e-12:
        raise ValueError("The orientation affine must have nonsingular spatial axes")
    current = _fsl_axis_orientation(matrix)
    target_codes = ("R", "A", "S") if determinant > 0 else ("L", "A", "S")
    target = nib.orientations.axcodes2ornt(target_codes)
    return nib.orientations.ornt_transform(current, target)


def reorient_spatial_array(values: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Permute and flip the first three axes without interpolation."""

    source = np.asarray(values)
    transform = np.asarray(orientation, dtype=np.float64)
    if source.ndim < 3:
        raise ValueError("Spatial reorientation requires at least three dimensions")
    if transform.shape != (3, 2):
        raise ValueError("The orientation transform must have shape (3, 2)")
    return np.ascontiguousarray(nib.orientations.apply_orientation(source, transform))


def voxel_basis_transform(orientation: np.ndarray) -> np.ndarray:
    """Return the signed permutation mapping old vector components to new ones."""

    transform = np.asarray(orientation, dtype=np.float64)
    if transform.shape != (3, 2):
        raise ValueError("The orientation transform must have shape (3, 2)")
    new_voxel_to_old_voxel = nib.orientations.inv_ornt_aff(
        transform, (1, 1, 1)
    )[:3, :3]
    return new_voxel_to_old_voxel.T


def reorient_bvecs_voxel(
    bvecs: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Transform row-wise b-vectors expressed in the image voxel basis."""

    vectors = np.asarray(bvecs)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("bvecs must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("bvecs contain NaN or Inf")
    basis = voxel_basis_transform(orientation)
    return np.asarray(vectors @ basis.T, dtype=vectors.dtype)


def reorient_tensor6_voxel(
    tensor: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Transform FSL-order tensor components between voxel bases."""

    values = np.asarray(tensor)
    basis = voxel_basis_transform(orientation)
    matrices = tensor6_to_matrix(values)
    transformed = np.einsum(
        "ij,...jk,lk->...il", basis, matrices, basis, optimize=True
    )
    return np.asarray(matrix_to_tensor6(transformed), dtype=values.dtype)


def write_fsl_reoriented(
    input_file: str | Path,
    output_file: str | Path,
    *,
    float32: bool = False,
    nonnegative: bool = False,
) -> Path:
    """Write the limited ``fslreorient2std`` behavior used by SimNIBS.

    This operation changes only storage order and the effective NIfTI transform;
    it performs no interpolation and does not alter vector or tensor component
    channels. ``float32`` and ``nonnegative`` fuse the official ``fslmaths`` copy
    and ``-thr 0`` into the same input read; when disabled by default, they preserve
    the original interface and numerical behavior.
    """

    source = nib.load(str(input_file))
    qform_code = int(source.header["qform_code"])
    sform_code = int(source.header["sform_code"])
    if qform_code == 0 and sform_code == 0:
        raise ValueError("NIfTI orientation requires a valid qform or sform")
    active_affine = source.get_sform() if sform_code > 0 else source.get_qform()
    orientation = fsl_canonical_orientation(active_affine)
    source_values = np.asanyarray(source.dataobj)
    if float32:
        source_values = np.asarray(source_values, dtype=np.float32)
    values = reorient_spatial_array(source_values, orientation)
    if nonnegative:
        values = np.array(values, copy=True)
        np.maximum(values, 0.0, out=values)
    new_voxel_to_old_voxel = nib.orientations.inv_ornt_aff(
        orientation, source.shape[:3]
    )
    qform = source.get_qform() @ new_voxel_to_old_voxel
    sform = source.get_sform() @ new_voxel_to_old_voxel
    new_active_affine = (
        sform if sform_code > 0 else qform
    )

    header = source.header.copy()
    header.set_data_shape(values.shape)
    header.set_data_dtype(np.float32 if float32 else source.get_data_dtype())
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(values, new_active_affine, header)
    image.set_qform(qform, qform_code)
    image.set_sform(sform, sform_code)
    nib.save(image, str(output))
    return output


def copy_nifti_geometry(
    source_file: str | Path,
    destination_file: str | Path,
    output_file: str | Path | None = None,
    *,
    copy_dimensions: bool = True,
) -> Path:
    """Copy the geometry fields used by ``fslcpgeom`` onto existing data.

    The dwi2cond call site uses matching 3D grids. Dimension-changing header
    reinterpretation is rejected because it can expose or truncate raw bytes.
    """

    source = nib.load(str(source_file))
    destination = nib.load(str(destination_file))
    if copy_dimensions and source.shape != destination.shape:
        raise ValueError("Geometry dimension copying requires matching image shapes")
    values = np.asanyarray(destination.dataobj)
    header = destination.header.copy()
    source_zooms = tuple(float(value) for value in source.header["pixdim"][1 : values.ndim + 1])
    header.set_zooms(source_zooms)
    qform_code = int(source.header["qform_code"])
    sform_code = int(source.header["sform_code"])
    active_affine = source.get_sform() if sform_code > 0 else source.get_qform()
    if qform_code == 0 and sform_code == 0:
        active_affine = destination.affine
    output = Path(destination_file) if output_file is None else Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(values, active_affine, header)
    image.set_qform(source.get_qform(), qform_code)
    image.set_sform(source.get_sform(), sform_code)
    nib.save(image, str(output))
    return output
