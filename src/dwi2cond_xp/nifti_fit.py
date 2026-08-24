"""Chunked NIfTI DTI fitting and output."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Callable
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from threadpoolctl import threadpool_limits

from .gradients import load_gradients, select_dti_volumes
from .preprocessing.tensor_ops import decompose_tensor6
from .tensor_fit import fit_tensor_wls


ProgressCallback = Callable[[int, int, int], None]


def _save_float_map(
    values: np.ndarray,
    path: Path,
    reference: nib.spatialimages.SpatialImage,
    *,
    vector: bool = False,
) -> None:
    """Write a float32 scalar or vector NIfTI using the reference-space contract."""
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    if vector:
        header.set_intent("vector")
    image = nib.Nifti1Image(values.astype(np.float32, copy=False), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(path))


def _save_derived_maps(
    tensor: np.ndarray,
    valid: np.ndarray,
    s0: np.ndarray,
    sse: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path,
) -> dict[str, str]:
    """Generate FSL-style DTI QA derivatives from six-component tensors."""
    name = output_path.name
    base = name[:-7] if name.endswith(".nii.gz") else output_path.stem
    outputs = decompose_tensor6(tensor, valid)
    outputs["S0"] = s0
    outputs["sse"] = sse

    paths: dict[str, str] = {}
    for suffix, values in outputs.items():
        path = output_path.with_name(f"{base}_{suffix}.nii.gz")
        _save_float_map(values, path, reference, vector=suffix.startswith("V"))
        paths[suffix] = str(path)
    return paths


def select_shell_nifti(
    data_file: str | Path,
    bvals_file: str | Path,
    bvecs_file: str | Path,
    output_data_file: str | Path,
    output_bvals_file: str | Path,
    output_bvecs_file: str | Path,
    *,
    shell: float = 1000.0,
    tolerance: float = 100.0,
    b0_threshold: float = 50.0,
) -> np.ndarray:
    """Decompress and save b=0 plus one shell for later process-shared mmap."""
    image = nib.load(str(data_file))
    if len(image.shape) != 4:
        raise ValueError("DWI data must be a four-dimensional NIfTI")
    bvals, bvecs = load_gradients(bvals_file, bvecs_file)
    if bvals.size != image.shape[3]:
        raise ValueError("The DWI fourth axis does not match bvals/bvecs")
    selected = select_dti_volumes(
        bvals,
        shell=shell,
        tolerance=tolerance,
        b0_threshold=b0_threshold,
    )
    # One sequential pass avoids every worker rescanning the same gzip stream.
    full_data = np.asanyarray(image.dataobj, dtype=np.float32)
    selected_data = np.ascontiguousarray(full_data[..., selected])
    del full_data

    output_data_path = Path(output_data_file)
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    selected_image = nib.Nifti1Image(selected_data, image.affine, header=header)
    selected_image.set_qform(image.get_qform(), int(image.header["qform_code"]))
    selected_image.set_sform(image.get_sform(), int(image.header["sform_code"]))
    nib.save(selected_image, str(output_data_path))
    np.savetxt(output_bvals_file, bvals[selected][None, :], fmt="%.10g")
    np.savetxt(output_bvecs_file, bvecs[selected].T, fmt="%.12g")
    return selected


@threadpool_limits.wrap(limits=1)
def _fit_z_block(
    data_file: str,
    mask_file: str,
    grad_dev_file: str | None,
    selected: np.ndarray,
    selected_bvals: np.ndarray,
    selected_bvecs: np.ndarray,
    z0: int,
    z1: int,
    voxel_batch: int,
) -> tuple:
    """Independent worker entry point for z-block fitting."""
    data_img = nib.load(data_file, mmap=True)
    mask_img = nib.load(mask_file, mmap=True)
    mask_chunk = np.asanyarray(mask_img.dataobj[:, :, z0:z1]) > 0
    fitted_volume = np.zeros(mask_chunk.shape + (6,), dtype=np.float32)
    s0_volume = np.zeros(mask_chunk.shape, dtype=np.float32)
    sse_volume = np.zeros(mask_chunk.shape, dtype=np.float32)
    valid_volume = np.zeros(mask_chunk.shape, dtype=bool)
    masked_count = int(np.count_nonzero(mask_chunk))
    if masked_count == 0:
        return z0, z1, fitted_volume, s0_volume, sse_volume, valid_volume, 0, 0, 0, 0

    data_chunk = np.asanyarray(data_img.dataobj[:, :, z0:z1, :], dtype=np.float32)
    signals = data_chunk[mask_chunk][:, selected]
    finite = np.all(np.isfinite(signals), axis=1)
    has_positive = np.max(np.where(np.isfinite(signals), signals, -np.inf), axis=1) > 0
    valid = finite & has_positive
    nonfinite_count = int(np.count_nonzero(~finite))
    all_nonpositive_count = int(np.count_nonzero(finite & ~has_positive))
    nonpositive_measurements = int(np.count_nonzero(np.isfinite(signals) & (signals <= 0)))
    grad_values = None
    if grad_dev_file is not None:
        grad_img = nib.load(grad_dev_file, mmap=True)
        grad_chunk = np.asanyarray(grad_img.dataobj[:, :, z0:z1, :], dtype=np.float32)
        grad_values = grad_chunk[mask_chunk]

    valid_signals = signals[valid]
    valid_grad = None if grad_values is None else grad_values[valid]
    fitted = np.zeros((masked_count, 6), dtype=np.float32)
    fitted_valid = np.empty((valid_signals.shape[0], 6), dtype=np.float32)
    s0_valid = np.empty(valid_signals.shape[0], dtype=np.float32)
    sse_valid = np.empty(valid_signals.shape[0], dtype=np.float32)
    for start in range(0, valid_signals.shape[0], voxel_batch):
        stop = min(start + voxel_batch, valid_signals.shape[0])
        batch_grad = None if valid_grad is None else valid_grad[start:stop]
        batch_tensor, batch_s0, batch_sse = fit_tensor_wls(
            valid_signals[start:stop],
            selected_bvals,
            selected_bvecs,
            batch_grad,
            return_metrics=True,
        )
        fitted_valid[start:stop] = batch_tensor.astype(np.float32)
        s0_valid[start:stop] = batch_s0.astype(np.float32)
        sse_valid[start:stop] = batch_sse.astype(np.float32)
    fitted[valid] = fitted_valid
    fitted_volume[mask_chunk] = fitted
    block_s0 = np.zeros(masked_count, dtype=np.float32)
    block_sse = np.zeros(masked_count, dtype=np.float32)
    block_s0[valid] = s0_valid
    block_sse[valid] = sse_valid
    s0_volume[mask_chunk] = block_s0
    sse_volume[mask_chunk] = block_sse
    valid_volume[mask_chunk] = valid
    return (
        z0,
        z1,
        fitted_volume,
        s0_volume,
        sse_volume,
        valid_volume,
        masked_count,
        nonfinite_count,
        all_nonpositive_count,
        nonpositive_measurements,
    )


def fit_dti_nifti(
    data_file: str | Path,
    bvals_file: str | Path,
    bvecs_file: str | Path,
    mask_file: str | Path,
    output_file: str | Path,
    *,
    grad_dev_file: str | Path | None = None,
    shell: float = 1000.0,
    tolerance: float = 100.0,
    b0_threshold: float = 50.0,
    z_chunk: int = 4,
    voxel_batch: int = 4096,
    workers: int = 1,
    progress: ProgressCallback | None = None,
    valid_mask_file: str | Path | None = None,
    qa_file: str | Path | None = None,
) -> Path:
    """Fit a NIfTI in z-blocks and write tensor, validity mask, and QA JSON.

    Masked voxels containing NaN/Inf or no positive selected measurement are
    excluded. Their tensor is set to zero and recorded in the mask and QA JSON,
    preventing FSL-style NaN/Inf output.
    """
    if z_chunk <= 0 or voxel_batch <= 0 or workers <= 0:
        raise ValueError("z_chunk, voxel_batch, and workers must be positive integers")
    data_img = nib.load(str(data_file))
    mask_img = nib.load(str(mask_file))
    if len(data_img.shape) != 4:
        raise ValueError("DWI data must be a four-dimensional NIfTI")
    if mask_img.shape != data_img.shape[:3]:
        raise ValueError("The mask shape does not match the DWI spatial shape")
    if not np.allclose(mask_img.affine, data_img.affine):
        raise ValueError("The mask affine does not match the DWI affine")

    grad_img = None
    if grad_dev_file is not None:
        grad_img = nib.load(str(grad_dev_file))
        if grad_img.shape != data_img.shape[:3] + (9,):
            raise ValueError("grad_dev shape must be the DWI spatial shape plus nine components")
        if not np.allclose(grad_img.affine, data_img.affine):
            raise ValueError("The grad_dev affine does not match the DWI affine")

    bvals, bvecs = load_gradients(bvals_file, bvecs_file)
    if bvals.size != data_img.shape[3]:
        raise ValueError("The DWI fourth axis does not match bvals/bvecs")
    selected = select_dti_volumes(
        bvals,
        shell=shell,
        tolerance=tolerance,
        b0_threshold=b0_threshold,
    )
    selected_bvals = bvals[selected]
    selected_bvecs = bvecs[selected]

    output = np.zeros(data_img.shape[:3] + (6,), dtype=np.float32)
    s0_output = np.zeros(data_img.shape[:3], dtype=np.float32)
    sse_output = np.zeros(data_img.shape[:3], dtype=np.float32)
    valid_output = np.zeros(data_img.shape[:3], dtype=np.uint8)
    total_masked = int(np.count_nonzero(np.asanyarray(mask_img.dataobj)))
    processed = 0
    nonfinite_voxels = 0
    all_nonpositive_voxels = 0
    nonpositive_measurements = 0
    if workers > 1:
        blocks = [
            (z0, min(z0 + z_chunk, data_img.shape[2]))
            for z0 in range(0, data_img.shape[2], z_chunk)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _fit_z_block,
                    str(data_file),
                    str(mask_file),
                    None if grad_dev_file is None else str(grad_dev_file),
                    selected,
                    selected_bvals,
                    selected_bvecs,
                    z0,
                    z1,
                    voxel_batch,
                ): (z0, z1)
                for z0, z1 in blocks
            }
            for future in as_completed(futures):
                (
                    z0,
                    z1,
                    fitted_volume,
                    s0_volume,
                    sse_volume,
                    valid_volume,
                    masked_count,
                    block_nonfinite,
                    block_all_nonpositive,
                    block_nonpositive_measurements,
                ) = future.result()
                output[:, :, z0:z1, :] = fitted_volume
                s0_output[:, :, z0:z1] = s0_volume
                sse_output[:, :, z0:z1] = sse_volume
                valid_output[:, :, z0:z1] = valid_volume
                processed += masked_count
                nonfinite_voxels += block_nonfinite
                all_nonpositive_voxels += block_all_nonpositive
                nonpositive_measurements += block_nonpositive_measurements
                if progress is not None:
                    progress(processed, total_masked, z1)
    else:
        (
            nonfinite_voxels,
            all_nonpositive_voxels,
            nonpositive_measurements,
        ) = _fit_dti_nifti_serial(
            data_img,
            mask_img,
            grad_img,
            selected,
            selected_bvals,
            selected_bvecs,
            output,
            total_masked,
            z_chunk,
            voxel_batch,
            progress,
            valid_output,
            s0_output,
            sse_output,
        )

    header = data_img.header.copy()
    header.set_data_dtype(np.float32)
    output_img = nib.Nifti1Image(output, data_img.affine, header=header)
    output_img.set_qform(data_img.get_qform(), int(data_img.header["qform_code"]))
    output_img.set_sform(data_img.get_sform(), int(data_img.header["sform_code"]))
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_img, str(output_path))

    name = output_path.name
    base = name[:-7] if name.endswith(".nii.gz") else output_path.stem
    if valid_mask_file is None:
        valid_mask_path = output_path.with_name(f"{base}_valid_mask.nii.gz")
    else:
        valid_mask_path = Path(valid_mask_file)
    valid_mask_path.parent.mkdir(parents=True, exist_ok=True)
    valid_img = nib.Nifti1Image(valid_output, data_img.affine, header=mask_img.header.copy())
    valid_img.set_data_dtype(np.uint8)
    valid_img.set_qform(data_img.get_qform(), int(data_img.header["qform_code"]))
    valid_img.set_sform(data_img.get_sform(), int(data_img.header["sform_code"]))
    nib.save(valid_img, str(valid_mask_path))
    derived_paths = _save_derived_maps(
        output,
        valid_output.astype(bool),
        s0_output,
        sse_output,
        data_img,
        output_path,
    )

    if qa_file is None:
        qa_path = output_path.with_name(f"{base}_qa.json")
    else:
        qa_path = Path(qa_file)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa = {
        "masked_voxels": total_masked,
        "valid_fitted_voxels": total_masked - nonfinite_voxels - all_nonpositive_voxels,
        "nonfinite_voxels": nonfinite_voxels,
        "all_nonpositive_voxels": all_nonpositive_voxels,
        "nonpositive_measurements": nonpositive_measurements,
        "invalid_tensor_value": 0.0,
        "tensor_component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
        "valid_mask": str(valid_mask_path),
        "derived_outputs": derived_paths,
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


@threadpool_limits.wrap(limits=1)
def _fit_dti_nifti_serial(
    data_img: nib.spatialimages.SpatialImage,
    mask_img: nib.spatialimages.SpatialImage,
    grad_img: nib.spatialimages.SpatialImage | None,
    selected: np.ndarray,
    selected_bvals: np.ndarray,
    selected_bvecs: np.ndarray,
    output: np.ndarray,
    total_masked: int,
    z_chunk: int,
    voxel_batch: int,
    progress: ProgressCallback | None,
    valid_output: np.ndarray,
    s0_output: np.ndarray,
    sse_output: np.ndarray,
) -> tuple[int, int, int]:
    """Retain the single-process reference path."""
    processed = 0
    nonfinite_voxels = 0
    all_nonpositive_voxels = 0
    nonpositive_measurements = 0
    for z0 in range(0, data_img.shape[2], z_chunk):
        z1 = min(z0 + z_chunk, data_img.shape[2])
        mask_chunk = np.asanyarray(mask_img.dataobj[:, :, z0:z1]) > 0
        if not np.any(mask_chunk):
            if progress is not None:
                progress(processed, total_masked, z1)
            continue

        # Arbitrary time indexing is costly for gzip NIfTI; read a z-block then select volumes.
        data_chunk = np.asanyarray(data_img.dataobj[:, :, z0:z1, :], dtype=np.float32)
        signals = data_chunk[mask_chunk][:, selected]
        finite = np.all(np.isfinite(signals), axis=1)
        has_positive = np.max(np.where(np.isfinite(signals), signals, -np.inf), axis=1) > 0
        valid = finite & has_positive
        nonfinite_voxels += int(np.count_nonzero(~finite))
        all_nonpositive_voxels += int(np.count_nonzero(finite & ~has_positive))
        nonpositive_measurements += int(np.count_nonzero(np.isfinite(signals) & (signals <= 0)))
        grad_values = None
        if grad_img is not None:
            grad_chunk = np.asanyarray(grad_img.dataobj[:, :, z0:z1, :], dtype=np.float32)
            grad_values = grad_chunk[mask_chunk]

        valid_signals = signals[valid]
        valid_grad = None if grad_values is None else grad_values[valid]
        fitted_chunk = np.zeros((signals.shape[0], 6), dtype=np.float32)
        fitted_valid = np.empty((valid_signals.shape[0], 6), dtype=np.float32)
        s0_valid = np.empty(valid_signals.shape[0], dtype=np.float32)
        sse_valid = np.empty(valid_signals.shape[0], dtype=np.float32)
        for start in range(0, valid_signals.shape[0], voxel_batch):
            stop = min(start + voxel_batch, valid_signals.shape[0])
            batch_grad = None if valid_grad is None else valid_grad[start:stop]
            batch_tensor, batch_s0, batch_sse = fit_tensor_wls(
                valid_signals[start:stop],
                selected_bvals,
                selected_bvecs,
                batch_grad,
                return_metrics=True,
            )
            fitted_valid[start:stop] = batch_tensor.astype(np.float32)
            s0_valid[start:stop] = batch_s0.astype(np.float32)
            sse_valid[start:stop] = batch_sse.astype(np.float32)
            processed += stop - start
            if progress is not None:
                progress(processed, total_masked, z1)

        # Invalid voxels are not fitted numerically but count as inspected progress.
        processed += int(np.count_nonzero(~valid))
        if progress is not None:
            progress(processed, total_masked, z1)

        fitted_chunk[valid] = fitted_valid

        target = output[:, :, z0:z1, :]
        target[mask_chunk] = fitted_chunk
        valid_target = valid_output[:, :, z0:z1]
        valid_target[mask_chunk] = valid
        block_s0 = np.zeros(signals.shape[0], dtype=np.float32)
        block_sse = np.zeros(signals.shape[0], dtype=np.float32)
        block_s0[valid] = s0_valid
        block_sse[valid] = sse_valid
        s0_target = s0_output[:, :, z0:z1]
        sse_target = sse_output[:, :, z0:z1]
        s0_target[mask_chunk] = block_s0
        sse_target[mask_chunk] = block_sse
    return nonfinite_voxels, all_nonpositive_voxels, nonpositive_measurements
