#!/usr/bin/env python3
"""Generate a deterministic public fixture for FSL EDDY and ``--repol``."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directions(count: int) -> np.ndarray:
    """Return deterministic approximately uniform unit directions."""

    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def _undistorted_series(
    shape: tuple[int, int, int], directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create an asymmetric spatial tensor phantom and its brain mask."""

    grid = np.indices(shape, dtype=np.float64)
    normalized = np.stack(
        tuple(2.0 * grid[axis] / (shape[axis] - 1.0) - 1.0 for axis in range(3)),
        axis=-1,
    )
    x, y, z = np.moveaxis(normalized, -1, 0)
    mask = ((x / 0.86) ** 2 + (y / 0.82) ** 2 + (z / 0.90) ** 2) <= 1.0
    baseline = (
        850.0
        + 180.0 * np.exp(-((x + 0.33) ** 2 + (y - 0.21) ** 2) / 0.075)
        + 120.0 * np.exp(-((x - 0.38) ** 2 + (z + 0.26) ** 2) / 0.055)
        + 45.0 * x
        - 30.0 * y
    )
    baseline *= mask

    theta = 0.65 * x - 0.35 * z + 0.18 * y
    principal = np.stack(
        (np.cos(theta), np.sin(theta), 0.25 * np.sin(np.pi * z)), axis=-1
    )
    principal /= np.linalg.norm(principal, axis=-1, keepdims=True)
    diffusivity = 0.45e-3 + 1.05e-3 * np.square(
        np.einsum("...i,ni->...n", principal, directions, optimize=True)
    )
    dwi = baseline[..., None] * np.exp(-1000.0 * diffusivity)
    series = np.concatenate((baseline[..., None], baseline[..., None], dwi), axis=3)
    return series.astype(np.float64), mask.astype(np.uint8)


def _motion_matrix(parameters: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return output-to-input voxel mapping for rigid motion parameters."""

    rx, ry, rz, tx, ty, tz = parameters
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotation = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )
    translation_voxels = np.array((tx, ty, tz), dtype=np.float64) / 2.0
    # Generate a moved image by sampling the reference through the inverse motion.
    inverse = rotation.T
    offset = center - inverse @ (center + translation_voxels)
    return inverse, offset


def _apply_ec_warp(volume: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Apply the FSL quadratic physical-coordinate basis along PE axis y."""

    shape = volume.shape
    grid = np.indices(shape, dtype=np.float64)
    x = 2.0 * (grid[0] - (shape[0] - 1.0) / 2.0)
    y = 2.0 * (grid[1] - (shape[1] - 1.0) / 2.0)
    z = 2.0 * (grid[2] - (shape[2] - 1.0) / 2.0)
    basis = (x, y, z, x * x, y * y, z * z, x * y, x * z, y * z)
    field_hz = sum(coefficient * term for coefficient, term in zip(parameters, basis))
    shift_voxels = 0.05 * field_hz
    coordinates = np.array(grid, copy=True)
    coordinates[1] += shift_voxels
    return map_coordinates(
        volume, coordinates, order=3, mode="constant", cval=0.0, prefilter=True
    )


def main() -> int:
    """Write images, acquisition tables, corruption truth, and exact hashes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)

    shape = (26, 26, 18)
    affine = np.array(
        [[-2.0, 0.0, 0.0, 25.0], [0.0, 2.0, 0.0, -25.0], [0.0, 0.0, 2.0, -17.0], [0.0, 0.0, 0.0, 1.0]]
    )
    directions = _directions(24)
    clean, mask = _undistorted_series(shape, directions)
    volume_count = clean.shape[3]
    phase = np.linspace(0.0, 2.0 * np.pi, volume_count, endpoint=False)
    motion = np.column_stack(
        (
            0.010 * np.sin(phase),
            0.008 * np.cos(1.3 * phase),
            0.007 * np.sin(0.7 * phase + 0.2),
            0.70 * np.sin(0.8 * phase),
            0.55 * np.cos(1.1 * phase + 0.1),
            0.45 * np.sin(1.4 * phase - 0.3),
        )
    )
    ec = np.zeros((volume_count, 9), dtype=np.float64)
    ec[:, 0] = 0.055 * np.sin(phase + 0.3)
    ec[:, 1] = 0.035 * np.cos(0.9 * phase)
    ec[:, 2] = 0.025 * np.sin(1.2 * phase - 0.4)
    ec[:, 3] = 0.0011 * np.cos(phase)
    ec[:, 4] = 0.0008 * np.sin(0.7 * phase)
    ec[:, 5] = 0.0006 * np.cos(1.3 * phase)
    ec[:, 6] = 0.0007 * np.sin(1.1 * phase)
    ec[:, 7] = 0.0005 * np.cos(0.8 * phase)
    ec[:, 8] = 0.0006 * np.sin(1.4 * phase)
    ec[:2] = 0.0
    center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    observed = np.empty_like(clean)
    for index in range(volume_count):
        matrix, offset = _motion_matrix(motion[index], center)
        moved = affine_transform(
            clean[..., index], matrix, offset=offset, order=3, mode="constant", cval=0.0
        )
        observed[..., index] = _apply_ec_warp(moved, ec[index])

    corruptions = ((5, 8, 0.0), (11, 9, 0.12), (18, 7, 0.0), (23, 10, 0.0))
    corrupted = observed.copy()
    for volume, slice_index, factor in corruptions:
        corrupted[:, :, slice_index, volume] *= factor

    image = nib.Nifti1Image(corrupted.astype(np.float32), affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, output / "dwi.nii")
    uncorrupted_image = nib.Nifti1Image(observed.astype(np.float32), affine)
    uncorrupted_image.set_qform(affine, 1)
    uncorrupted_image.set_sform(affine, 1)
    nib.save(uncorrupted_image, output / "truth_uncorrupted_dwi.nii")
    undistorted_image = nib.Nifti1Image(clean.astype(np.float32), affine)
    undistorted_image.set_qform(affine, 1)
    undistorted_image.set_sform(affine, 1)
    nib.save(undistorted_image, output / "truth_undistorted_dwi.nii")
    mask_image = nib.Nifti1Image(mask, affine)
    mask_image.set_qform(affine, 1)
    mask_image.set_sform(affine, 1)
    nib.save(mask_image, output / "mask.nii")
    bvals = np.concatenate((np.zeros(2), np.full(directions.shape[0], 1000.0)))
    bvecs = np.column_stack((np.zeros((3, 2)), directions.T))
    np.savetxt(output / "bvals", bvals[None, :], fmt="%.1f")
    np.savetxt(output / "bvecs", bvecs, fmt="%.12f")
    (output / "acqp.txt").write_text("0 1 0 0.05\n", encoding="utf-8")
    (output / "index.txt").write_text(
        " ".join("1" for _ in range(volume_count)) + "\n", encoding="utf-8"
    )

    input_files = [
        output / name
        for name in ("dwi.nii", "mask.nii", "bvals", "bvecs", "acqp.txt", "index.txt")
    ]
    truth_files = [
        output / "truth_uncorrupted_dwi.nii",
        output / "truth_undistorted_dwi.nii",
    ]
    manifest = {
        "fixture_id": "synthetic-eddy-v1",
        "visibility": "public-nonanatomical",
        "shape": [*shape, volume_count],
        "voxel_sizes_mm": [2.0, 2.0, 2.0],
        "phase_encoding": [0, 1, 0, 0.05],
        "b0_count": 2,
        "dwi_count": int(directions.shape[0]),
        "motion_parameters_rad_mm": motion.tolist(),
        "quadratic_ec_parameters_fsl_order": ec.tolist(),
        "ec_basis": ["x", "y", "z", "x2", "y2", "z2", "xy", "xz", "yz"],
        "slice_corruptions": [
            {"volume": volume, "slice": slice_index, "factor": factor}
            for volume, slice_index, factor in corruptions
        ],
        "sha256": {path.name: _sha256(path) for path in input_files},
        "truth_sha256": {path.name: _sha256(path) for path in truth_files},
    }
    (output / "fixture.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
