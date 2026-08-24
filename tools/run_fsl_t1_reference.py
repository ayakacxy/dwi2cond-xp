#!/usr/bin/env python3
"""Run the FSL T1 linear-registration reference required by P6 through the P0 harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def _run(command: list[str], cwd: Path) -> None:
    """Run an FSL command while preserving its failure status."""

    subprocess.run(command, cwd=cwd, check=True)


def _worker(fixture: Path, output: Path, dof: int) -> int:
    """Run the reference according to the linear branch of dwi2cond.t1reg.source.sh."""

    output.mkdir(parents=True, exist_ok=True)
    _run(
        ["fslmaths", str(fixture / "labeling.nii.gz"), "-thr", "1", "-uthr", "499", "-bin", "T1_brainmask"],
        output,
    )
    _run(
        ["fslmaths", str(fixture / "T1_bias_corrected.nii.gz"), "-mas", "T1_brainmask", "T1_brain"],
        output,
    )
    _run(
        ["fslmaths", "T1_brainmask", "-edge", "-thr", "0.3", "-bin", "T1_brainrim_QA"],
        output,
    )
    _run(
        ["fslmaths", str(fixture / "T1.nii.gz"), "-bin", "-s", "1", "T1_mask"],
        output,
    )
    _run(["fslcpgeom", "T1_brain", "T1_mask"], output)
    _run(
        [
            "flirt", "-in", str(fixture / "DTI_FA.nii.gz"), "-ref", "T1_brain",
            "-refweight", "T1_mask", "-omat", "FA2T1.mat", "-dof", str(dof),
        ],
        output,
    )
    _run(
        [
            "vecreg", "-i", str(fixture / "DTI_tensor.nii.gz"),
            "-o", "DTI_coregT1_tensor", "-r", "T1_brain", "-t", "FA2T1.mat",
        ],
        output,
    )
    _run(["fslmaths", "DTI_coregT1_tensor", "-mas", "T1_brainmask", "DTI_coregT1_tensor"], output)
    _run(["fslmaths", "DTI_coregT1_tensor", "-tensor_decomp", "DTI_coregT1"], output)
    _run(
        [
            "flirt", "-in", str(fixture / "DTI_FA.nii.gz"), "-ref", "T1_brain",
            "-refweight", "T1_mask", "-omat", "FA2T1_QA.mat", "-out", "DTI_FA_6dof_QA", "-dof", "6",
        ],
        output,
    )
    _run(["fslmaths", "DTI_FA_6dof_QA", "-mas", "T1_brainmask", "DTI_FA_6dof_QA"], output)
    _run(
        [
            "flirt", "-in", str(fixture / "DTI_sse.nii.gz"), "-ref", "T1_brain",
            "-applyxfm", "-init", "FA2T1_QA.mat", "-out", "DTI_SSE_6dof_QA",
        ],
        output,
    )
    _run(["fslmaths", "DTI_SSE_6dof_QA", "-mas", "T1_brainmask", "DTI_SSE_6dof_QA"], output)
    return 0


def main() -> int:
    """Run the worker or build the public reference manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fsl-dir", type=Path, default=Path("/usr/local/fsl"))
    parser.add_argument("--dof", type=int, choices=(6, 12), default=12)
    args = parser.parse_args()
    if args.worker:
        return _worker(args.fixture.resolve(), args.work.resolve(), args.dof)
    if args.manifest is None:
        parser.error("--manifest is required outside worker mode")
    from dwi2cond_xp.preprocessing import ReferenceArtifact, run_reference_command

    environment = {
        "FSLDIR": str(args.fsl_dir.resolve()),
        "FSLOUTPUTTYPE": "NIFTI_GZ",
        "OMP_NUM_THREADS": "1",
        "PATH": os.pathsep.join((str(args.fsl_dir.resolve() / "bin"), os.environ.get("PATH", ""))),
    }
    artifacts = (
        ReferenceArtifact("FA2T1.mat"),
        ReferenceArtifact("FA2T1_QA.mat"),
        ReferenceArtifact("T1_brain.nii.gz", "nifti"),
        ReferenceArtifact("T1_brainrim_QA.nii.gz", "nifti", mask=True),
        ReferenceArtifact("DTI_coregT1_tensor.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_FA.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_V1.nii.gz", "nifti"),
        ReferenceArtifact("DTI_FA_6dof_QA.nii.gz", "nifti"),
        ReferenceArtifact("DTI_SSE_6dof_QA.nii.gz", "nifti"),
    )
    manifest = run_reference_command(
        stage=f"simnibs46-t1reg-{args.dof}dof",
        executable=sys.executable,
        arguments=(
            str(Path(__file__).resolve()), "--worker", "--fixture", str(args.fixture.resolve()),
            "--work", str(args.work.resolve()), "--fsl-dir", str(args.fsl_dir.resolve()),
            "--dof", str(args.dof),
        ),
        working_directory=args.work,
        manifest_path=args.manifest,
        artifacts=artifacts,
        environment=environment,
        reference_version="SimNIBS 4.6.0 dwi2cond 0.4 / FSL 6.0.4:ddd0a010",
        script_paths=(Path(__file__).resolve(),),
        threads=1,
        timeout_seconds=600,
    )
    print(json.dumps({"status": manifest["status"], "manifest": str(args.manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
