"""Finite NIfTI geometry and orientation operations used by dwi2cond."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ..registration import matrix_to_tensor6, tensor6_to_matrix


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
    current = nib.orientations.io_orientation(matrix)
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
