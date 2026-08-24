#!/usr/bin/env python3
"""Generate deterministic reverse-PE, fieldmap, affine, and tensor fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dwi2cond_xp.registration import matrix_to_tensor6


def _phantom(shape: tuple[int, int, int]) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64)
    values = np.zeros(shape, dtype=np.float64)
    for amplitude, center, width in (
        (900.0, (0.35, 0.43, 0.48), (0.18, 0.22, 0.20)),
        (600.0, (0.68, 0.62, 0.57), (0.13, 0.16, 0.12)),
    ):
        exponent = sum(
            ((grid[axis] / (shape[axis] - 1) - center[axis]) / width[axis]) ** 2
            for axis in range(3)
        )
        values += amplitude * np.exp(-exponent)
    values[values < 0.5] = 0.0
    return values.astype(np.float32)


def _save(path: Path, values: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Write small non-anatomical fixtures and their exact truth contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shape = (16, 14, 12)
    affine = np.array(
        [[-2.0, 0, 0, 15.0], [0, 2.2, 0, -14.3], [0, 0, 2.5, -13.75], [0, 0, 0, 1]]
    )
    base = _phantom(shape)

    reverse_pe = np.stack(
        (np.roll(base, 1, axis=1), np.roll(base, -1, axis=1)), axis=3
    )
    reverse_pe[:, 0, :, 0] = 0.0
    reverse_pe[:, -1, :, 1] = 0.0
    _save(output / "reverse_pe_b0.nii.gz", reverse_pe, affine)

    coordinates = np.indices(shape, dtype=np.float64)
    phase = (
        35.0 * np.sin(2.0 * np.pi * coordinates[0] / shape[0])
        + 10.0 * coordinates[2] / (shape[2] - 1)
    ).astype(np.float32)
    _save(output / "fieldmap_magnitude.nii.gz", base, affine)
    _save(output / "fieldmap_radians_per_second.nii.gz", phase, affine)

    angle = np.deg2rad(7.0)
    truth = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 1.5],
            [np.sin(angle), np.cos(angle), 0.0, -0.8],
            [0.0, 0.0, 1.0, 0.6],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _save(output / "affine_source.nii.gz", base, affine)
    reference_affine = truth @ affine
    _save(output / "affine_reference.nii.gz", base, reference_affine)

    theta = np.pi * coordinates[0] / (shape[0] - 1)
    principal = np.stack(
        (np.cos(theta), np.sin(theta), np.zeros(shape, dtype=np.float64)), axis=-1
    )
    secondary = np.stack(
        (-np.sin(theta), np.cos(theta), np.zeros(shape, dtype=np.float64)), axis=-1
    )
    third = np.zeros(shape + (3,), dtype=np.float64)
    third[..., 2] = 1.0
    rotation = np.stack((principal, secondary, third), axis=-1)
    eigenvalues = np.array([1.5e-3, 0.6e-3, 0.3e-3])
    matrices = np.einsum(
        "...ij,j,...kj->...ik", rotation, eigenvalues, rotation, optimize=True
    )
    tensor = matrix_to_tensor6(matrices).astype(np.float32)
    tensor[base == 0] = 0.0
    _save(output / "tensor_fsl6.nii.gz", tensor, affine)

    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "fixture_suite_id": "synthetic-preprocessing-v1",
        "visibility": "public-nonanatomical",
        "shape": list(shape),
        "affine": affine.tolist(),
        "reverse_pe": {
            "file": "reverse_pe_b0.nii.gz",
            "phase_encode_axis": 1,
            "voxel_shifts": [1, -1],
        },
        "fieldmap": {
            "magnitude": "fieldmap_magnitude.nii.gz",
            "phase": "fieldmap_radians_per_second.nii.gz",
            "phase_units": "radians_per_second",
        },
        "affine_registration": {
            "source": "affine_source.nii.gz",
            "reference": "affine_reference.nii.gz",
            "source_world_to_reference_world": truth.tolist(),
        },
        "tensor": {
            "file": "tensor_fsl6.nii.gz",
            "component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
            "eigenvalues": eigenvalues.tolist(),
        },
        "sha256": {path.name: _sha256(path) for path in files},
    }
    (output / "fixture_suite.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
