"""Cross-platform spatial mapping and reorientation of diffusion tensors."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def tensor6_to_matrix(tensor: np.ndarray) -> np.ndarray:
    """Convert the FSL six-component order to symmetric 3x3 tensors."""
    tensor = np.asarray(tensor)
    if tensor.shape[-1] != 6:
        raise ValueError("The final tensor axis must contain six components")
    matrix = np.empty(tensor.shape[:-1] + (3, 3), dtype=tensor.dtype)
    matrix[..., 0, 0] = tensor[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = tensor[..., 1]
    matrix[..., 0, 2] = matrix[..., 2, 0] = tensor[..., 2]
    matrix[..., 1, 1] = tensor[..., 3]
    matrix[..., 1, 2] = matrix[..., 2, 1] = tensor[..., 4]
    matrix[..., 2, 2] = tensor[..., 5]
    return matrix


def matrix_to_tensor6(matrix: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 tensors to the FSL six-component order."""
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("The final two matrix axes must be 3x3")
    return np.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 0, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 1],
            matrix[..., 1, 2],
            matrix[..., 2, 2],
        ),
        axis=-1,
    )


def polar_rotation(linear: np.ndarray) -> np.ndarray:
    """Extract the affine linear part's orthogonal polar factor, matching vecreg."""
    linear = np.asarray(linear, dtype=np.float64)
    if linear.shape != (3, 3) or not np.all(np.isfinite(linear)):
        raise ValueError("The affine linear part must be a finite 3x3 matrix")
    if abs(np.linalg.det(linear)) < 1e-12:
        raise ValueError("The affine linear part is singular; tensor reorientation is undefined")
    u, _, vt = np.linalg.svd(linear)
    return u @ vt


def reorient_affine_tensor(tensor: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    """Apply ``R D R^T`` using the affine rotation component."""
    transform = np.asarray(world_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_transform must be 4x4")
    rotation = polar_rotation(transform[:3, :3])
    matrix = tensor6_to_matrix(np.asarray(tensor, dtype=np.float64))
    rotated = np.einsum("ij,...jk,lk->...il", rotation, matrix, rotation, optimize=True)
    return matrix_to_tensor6(rotated)


def register_tensor_affine(
    tensor_file: str | Path,
    reference_file: str | Path,
    output_file: str | Path,
    *,
    world_transform: np.ndarray | None = None,
    source_mask_file: str | Path | None = None,
    reference_mask_file: str | Path | None = None,
    output_valid_mask_file: str | Path | None = None,
    qa_file: str | Path | None = None,
    interpolation_order: int = 1,
    progress: Callable[[int, int], None] | None = None,
    alignment_assumption: str | None = None,
) -> Path:
    """Affinely map a six-component tensor and apply finite-strain reorientation.

    ``world_transform`` maps input world coordinates to reference world
    coordinates. Without it, resampling uses only the two NIfTI affines and is
    valid only for data already aligned in physical space.
    """
    if interpolation_order not in (0, 1, 3):
        raise ValueError("interpolation_order must be 0, 1, or 3")
    tensor_img = nib.load(str(tensor_file))
    reference_img = nib.load(str(reference_file))
    if tensor_img.shape[-1:] != (6,):
        raise ValueError("The input tensor must be a four-dimensional six-component NIfTI")
    transform = np.eye(4) if world_transform is None else np.asarray(world_transform, float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world_transform must be a finite 4x4 matrix")

    # scipy affine_transform consumes the inverse map from output to input voxels.
    output_to_input = (
        np.linalg.inv(tensor_img.affine)
        @ np.linalg.inv(transform)
        @ reference_img.affine
    )
    source = np.asarray(tensor_img.dataobj, dtype=np.float32)
    target_shape = reference_img.shape[:3]
    sampled = np.empty(target_shape + (6,), dtype=np.float32)
    for component in range(6):
        sampled[..., component] = affine_transform(
            source[..., component],
            output_to_input[:3, :3],
            offset=output_to_input[:3, 3],
            output_shape=target_shape,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=interpolation_order > 1,
        )
        if progress is not None:
            progress(component + 1, 8)
    rotation = polar_rotation(transform[:3, :3])
    if not np.allclose(rotation, np.eye(3), rtol=0, atol=1e-12):
        # Whole-head 3x3 expansion needs several GiB; z-blocking bounds memory use.
        rotation32 = rotation.astype(np.float32)
        for z0 in range(0, target_shape[2], 8):
            z1 = min(z0 + 8, target_shape[2])
            matrices = tensor6_to_matrix(sampled[:, :, z0:z1])
            rotated = np.einsum(
                "ij,...jk,lk->...il",
                rotation32,
                matrices,
                rotation32,
                optimize=True,
            )
            sampled[:, :, z0:z1] = matrix_to_tensor6(rotated)
    if progress is not None:
        progress(7, 8)

    valid = np.all(np.isfinite(sampled), axis=-1) & np.any(sampled != 0, axis=-1)
    if source_mask_file is not None:
        source_mask = np.asarray(nib.load(str(source_mask_file)).dataobj) > 0
        if source_mask.shape != tensor_img.shape[:3]:
            raise ValueError("The source-mask shape does not match the tensor")
        sampled_source_mask = affine_transform(
            source_mask.astype(np.uint8),
            output_to_input[:3, :3],
            offset=output_to_input[:3, 3],
            output_shape=target_shape,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ) > 0
        valid &= sampled_source_mask
    if reference_mask_file is not None:
        reference_mask_img = nib.load(str(reference_mask_file))
        reference_mask = np.asarray(reference_mask_img.dataobj) > 0
        if reference_mask.shape != target_shape or not np.allclose(
            reference_mask_img.affine, reference_img.affine
        ):
            raise ValueError("The reference mask must match the reference grid exactly")
        valid &= reference_mask
    sampled[~valid] = 0

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    output_img = nib.Nifti1Image(sampled, reference_img.affine, header)
    output_img.set_qform(reference_img.get_qform(), int(reference_img.header["qform_code"]))
    output_img.set_sform(reference_img.get_sform(), int(reference_img.header["sform_code"]))
    nib.save(output_img, str(output_path))
    if progress is not None:
        progress(8, 8)

    name = output_path.name
    base = name[:-7] if name.endswith(".nii.gz") else output_path.stem
    valid_path = (
        Path(output_valid_mask_file)
        if output_valid_mask_file is not None
        else output_path.with_name(f"{base}_valid_mask.nii.gz")
    )
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    valid_header = reference_img.header.copy()
    valid_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(valid.astype(np.uint8), reference_img.affine, valid_header), valid_path)

    qa_path = (
        Path(qa_file)
        if qa_file is not None
        else output_path.with_name(f"{base}_registration_qa.json")
    )
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa = {
        "input_shape": list(tensor_img.shape),
        "output_shape": list(sampled.shape),
        "valid_voxels": int(np.count_nonzero(valid)),
        "world_transform": transform.tolist(),
        "polar_rotation": rotation.tolist(),
        "output_to_input_voxel": output_to_input.tolist(),
        "interpolation_order": interpolation_order,
        "tensor_component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
        "valid_mask": str(valid_path),
        "alignment_assumption": alignment_assumption,
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def make_charm_brain_mask(
    labeling_file: str | Path,
    output_file: str | Path,
    *,
    reference_file: str | Path | None = None,
) -> Path:
    """Create a CHARM brain mask from the official dwi2cond label range 1..499."""
    labeling_img = nib.load(str(labeling_file))
    labeling = np.asarray(labeling_img.dataobj)
    mask = ((labeling >= 1) & (labeling <= 499)).astype(np.uint8)
    reference_img = labeling_img
    if reference_file is not None:
        reference_img = nib.load(str(reference_file))
        if reference_img.shape[:3] != labeling_img.shape[:3] or not np.allclose(
            reference_img.affine, labeling_img.affine
        ):
            raise ValueError("CHARM labeling does not match the reference T1 grid")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(mask, reference_img.affine, header)
    image.set_qform(reference_img.get_qform(), int(reference_img.header["qform_code"]))
    image.set_sform(reference_img.get_sform(), int(reference_img.header["sform_code"]))
    nib.save(image, str(output_path))
    return output_path
