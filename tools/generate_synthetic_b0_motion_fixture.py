#!/usr/bin/env python3
"""Generate a deterministic non-anatomical b0 rigid-motion fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dwi2cond_xp.preprocessing import resample_rigid, rigid_world_matrix


def _phantom(shape: tuple[int, int, int]) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64)
    components = (
        (1.0, (0.28, 0.38, 0.42), (0.12, 0.18, 0.14)),
        (0.7, (0.70, 0.67, 0.63), (0.09, 0.13, 0.10)),
        (0.45, (0.25, 0.76, 0.72), (0.07, 0.08, 0.14)),
        (0.3, (0.78, 0.24, 0.27), (0.10, 0.06, 0.08)),
    )
    values = np.zeros(shape, dtype=np.float64)
    for amplitude, center, width in components:
        exponent = sum(
            (
                (grid[axis] - center[axis] * (shape[axis] - 1))
                / (width[axis] * shape[axis])
            )
            ** 2
            for axis in range(3)
        )
        values += amplitude * np.exp(-exponent)
    values = (values * 1000.0).astype(np.float32)
    values[values < 0.5] = 0
    return values


def main() -> int:
    """Write the 4D fixture, b-values, and moving-to-reference truth matrices."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)

    shape = (33, 31, 29)
    affine = np.array(
        [[-2.0, 0, 0, 64.0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]]
    )
    reference = _phantom(shape)
    center = (np.asarray(shape) - 1.0) * 2.0 / 2.0
    parameters = np.array(
        [
            [0.018, -0.012, 0.022, 1.2, -0.7, 0.5],
            [-0.012, 0.016, -0.014, -0.8, 0.9, -0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.015, 0.010, -0.018, 0.7, -1.0, 0.6],
            [-0.020, -0.014, 0.012, -1.1, 0.6, -0.7],
        ],
        dtype=np.float64,
    )
    transforms = [rigid_world_matrix(row, center) for row in parameters]
    sampling = np.diag([2.0, 2.0, 2.0, 1.0])
    world_transforms = [
        affine @ np.linalg.inv(sampling) @ transform @ sampling @ np.linalg.inv(affine)
        for transform in transforms
    ]
    volumes = [
        resample_rigid(reference, affine, shape, affine, np.linalg.inv(world_transform))
        for world_transform in world_transforms
    ]
    image = nib.Nifti1Image(np.stack(volumes, axis=3), affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, output / "b0_motion.nii.gz")
    np.savetxt(output / "bvals", [np.zeros(len(volumes), dtype=int)], fmt="%d")
    (output / "truth.json").write_text(
        json.dumps(
            {
                "shape": list(shape),
                "voxel_size_mm": 2.0,
                "reference_position": 2,
                "parameters_rad_mm": parameters.tolist(),
                "moving_to_reference_scaled_mm": [
                    matrix.tolist() for matrix in transforms
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
