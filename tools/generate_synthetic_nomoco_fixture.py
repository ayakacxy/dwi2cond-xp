#!/usr/bin/env python3
"""Generate a deterministic, non-anatomical single-shell DWI fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dwi2cond_xp.tensor_fit import form_design_matrix


def main() -> int:
    """Write synthetic DWI, gradients, and a public structural manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    directions = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvals = np.concatenate([np.zeros(2), np.full(directions.shape[0], 1000.0)])
    bvecs = np.vstack([np.zeros((2, 3)), directions])

    shape = (20, 20, 20)
    coordinates = np.indices(shape, dtype=np.float64)
    center = (np.asarray(shape, dtype=np.float64) - 1.0)[:, None, None, None] / 2.0
    radius = np.sqrt(np.sum((coordinates - center) ** 2, axis=0))
    support = radius <= 7.5
    s0 = np.where(support, 900.0 + 100.0 * (1.0 - radius / 7.5), 0.0)
    tensor = np.array([1.3e-3, 6e-5, -3e-5, 7e-4, 4e-5, 4e-4])
    design = form_design_matrix(bvals, bvecs)
    attenuation = np.exp(-(design[:, :6] @ tensor))
    data = s0[..., None] * attenuation

    affine = np.array(
        [[-2, 0, 0, 19], [0, 2, 0, -19], [0, 0, 2, -19], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, output / "dwi.nii.gz")
    np.savetxt(output / "bvals", bvals[None, :], fmt="%.8g")
    np.savetxt(output / "bvecs", bvecs.T, fmt="%.12g")
    (output / "fixture.json").write_text(
        json.dumps(
            {
                "fixture_id": "synthetic-nomoco-v1",
                "shape": [*shape, int(bvals.size)],
                "dtype": "float32",
                "b0_volumes": 2,
                "shell_b_value": 1000.0,
                "shell_directions": int(directions.shape[0]),
                "nonzero_spatial_voxels": int(np.count_nonzero(support)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
