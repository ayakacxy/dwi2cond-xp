"""Compare project conductivity meshes with the SimNIBS cond_utils reference."""

from __future__ import annotations

import argparse

import nibabel as nib
import numpy as np
from simnibs.mesh_tools import mesh_io
from simnibs.utils.cond_utils import cond2elmdata, standard_cond


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor")
    parser.add_argument("source_mesh")
    parser.add_argument("candidate_mesh")
    parser.add_argument("--mode", choices=("vn", "dir", "mc"), default="vn")
    parser.add_argument("--aniso-tissue", type=int, default=3)
    args = parser.parse_args()

    image = nib.load(args.tensor)
    source = mesh_io.read_msh(args.source_mesh)
    candidate = mesh_io.read_msh(args.candidate_mesh)
    conductivity = [item.value for item in standard_cond()]
    reference = cond2elmdata(
        source,
        conductivity,
        anisotropy_volume=image.dataobj,
        affine=image.affine,
        aniso_tissues=[args.aniso_tissue],
        normalize=args.mode == "vn",
        excentricity_scaling=0.0 if args.mode == "mc" else None,
    )
    candidate_field = next(
        field for field in candidate.elmdata if field.field_name == "conductivity"
    )
    difference = np.abs(candidate_field.value - reference.value)
    print("values", difference.size)
    print("max_abs", float(np.max(difference)))
    print("mean_abs", float(np.mean(difference)))
    print("p99_abs", float(np.quantile(difference, 0.99)))


if __name__ == "__main__":
    main()
