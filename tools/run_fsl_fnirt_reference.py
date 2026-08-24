#!/usr/bin/env python3
"""Run the fixed SimNIBS 4.6 FNIRT and nonlinear VECREG reference path."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys


def _run(command: list[str], cwd: Path) -> None:
    """Run one FSL command and preserve a nonzero status."""

    subprocess.run(command, cwd=cwd, check=True)


def _simnibs_external_file(name: str) -> Path:
    """Locate one installed SimNIBS external reference file."""

    spec = find_spec("simnibs")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "SimNIBS is required unless --simnibs-t1-script is provided"
        )
    return Path(next(iter(spec.submodule_search_locations))) / "external" / name


def _worker(
    fa: Path,
    tensor: Path,
    reference: Path,
    affine: Path,
    output: Path,
) -> int:
    """Execute the exact nonlinear branch from ``dwi2cond.t1reg.source.sh``."""

    output.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "fnirt",
            f"--in={fa}",
            f"--ref={reference}",
            f"--aff={affine}",
            "--cout=FA2T1_warp",
            "--fout=FA2T1_field",
            "--jout=FA2T1_jacobian",
            "--iout=DTI_FA_nonlin",
            "--logout=fnirt.log",
            "--subsamp=8,4,2,2",
            "--verbose",
        ],
        output,
    )
    _run(
        [
            "vecreg",
            "-i",
            str(tensor),
            "-o",
            "DTI_coregT1_tensor",
            "-r",
            str(reference),
            "-w",
            "FA2T1_warp",
        ],
        output,
    )
    _run(
        ["fslmaths", "DTI_coregT1_tensor", "-tensor_decomp", "DTI_coregT1"],
        output,
    )
    return 0


def main() -> int:
    """Run the worker directly or record it through the P0 reference harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fa", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fsl-dir", type=Path, default=Path("/usr/local/fsl"))
    parser.add_argument(
        "--simnibs-t1-script",
        type=Path,
        help="Optional path to dwi2cond.t1reg.source.sh",
    )
    args = parser.parse_args()
    inputs = tuple(
        path.resolve() for path in (args.fa, args.tensor, args.reference, args.affine)
    )
    if args.worker:
        return _worker(*inputs, args.work.resolve())
    if args.manifest is None:
        parser.error("--manifest is required outside worker mode")

    from dwi2cond_xp.preprocessing import ReferenceArtifact, run_reference_command

    fsl = args.fsl_dir.resolve()
    simnibs_t1 = args.simnibs_t1_script or _simnibs_external_file(
        "dwi2cond.t1reg.source.sh"
    )
    source_paths = (
        Path(__file__).resolve(),
        simnibs_t1,
        fsl / "src/fnirt/fnirt.cpp",
        fsl / "src/fnirt/fnirtfns.cpp",
        fsl / "src/fnirt/fnirt_costfunctions.cpp",
        fsl / "src/fdt/vecreg.cc",
    )
    environment = {
        "FSLDIR": str(fsl),
        "FSLOUTPUTTYPE": "NIFTI_GZ",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": os.pathsep.join((str(fsl / "bin"), os.environ.get("PATH", ""))),
    }
    artifacts = (
        ReferenceArtifact("FA2T1_warp.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_field.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_jacobian.nii.gz", "nifti"),
        ReferenceArtifact("DTI_FA_nonlin.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_tensor.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_FA.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_V1.nii.gz", "nifti"),
        ReferenceArtifact("fnirt.log"),
    )
    manifest = run_reference_command(
        stage="simnibs46-t1reg-nonlinear",
        executable=sys.executable,
        arguments=(
            str(Path(__file__).resolve()),
            "--worker",
            "--fa",
            str(inputs[0]),
            "--tensor",
            str(inputs[1]),
            "--reference",
            str(inputs[2]),
            "--affine",
            str(inputs[3]),
            "--work",
            str(args.work.resolve()),
            "--fsl-dir",
            str(fsl),
        ),
        working_directory=args.work,
        manifest_path=args.manifest,
        artifacts=artifacts,
        environment=environment,
        reference_version="SimNIBS 4.6.0 dwi2cond 0.4 / FSL 6.0.4:ddd0a010",
        script_paths=source_paths,
        threads=1,
        timeout_seconds=600,
        include_output_digests=True,
    )
    print(json.dumps({"status": manifest["status"], "manifest": str(args.manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
