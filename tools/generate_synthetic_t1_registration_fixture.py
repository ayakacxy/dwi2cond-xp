#!/usr/bin/env python3
"""Generate a deterministic non-anatomical fixture for P6 automatic T1 registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _save(path: Path, values: np.ndarray, affine: np.ndarray) -> None:
    """Write a NIfTI with fixed qform and sform."""

    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, path)


def main() -> int:
    """Generate FA, tensor, and CHARM T1 inputs with a known 2 mm translation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shape = (24, 23, 22)
    grid = np.indices(shape, dtype=np.float32)
    first = np.exp(
        -(
            (grid[0] - 7.0) ** 2 / 15.0
            + (grid[1] - 12.0) ** 2 / 35.0
            + (grid[2] - 9.0) ** 2 / 20.0
        )
    )
    second = np.exp(
        -(
            (grid[0] - 17.0) ** 2 / 10.0
            + (grid[1] - 6.0) ** 2 / 12.0
            + (grid[2] - 15.0) ** 2 / 9.0
        )
    )
    reference = np.asarray(first * 90.0 + second * 50.0, dtype=np.float32)
    moving = np.zeros_like(reference)
    moving[:-1] = reference[1:]
    affine = np.array(
        [[-2.0, 0.0, 0.0, 46.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    labels = np.zeros(shape, dtype=np.int16)
    labels[reference > np.float32(0.01)] = 2
    brain_mask = (labels > 0).astype(np.uint8)
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[..., 0] = np.float32(1.5e-3) + moving * np.float32(1.5e-7)
    tensor[..., 3] = np.float32(0.6e-3) + moving * np.float32(0.6e-7)
    tensor[..., 5] = np.float32(0.3e-3) + moving * np.float32(0.3e-7)
    sse = moving * np.float32(2.0)
    for name, values in (
        ("DTI_FA.nii.gz", moving),
        ("DTI_tensor.nii.gz", tensor),
        ("DTI_sse.nii.gz", sse),
        ("T1.nii.gz", reference),
        ("T1_bias_corrected.nii.gz", reference),
        ("T1_brain_mask.nii.gz", brain_mask),
        ("labeling.nii.gz", labels),
        ("final_tissues.nii.gz", labels),
    ):
        _save(output / name, values, affine)
    truth = np.eye(4)
    truth[0, 3] = -2.0
    expected_fsl = np.eye(4)
    expected_fsl[0, 3] = 2.0
    (output / "truth.json").write_text(
        json.dumps(
            {
                "fixture_id": "synthetic-t1-registration-v1",
                "shape": list(shape),
                "source_world_to_reference_world": truth.tolist(),
                "expected_fsl_scaled_mm_matrix": expected_fsl.tolist(),
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
