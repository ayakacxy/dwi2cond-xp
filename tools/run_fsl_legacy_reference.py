#!/usr/bin/env python3
"""Run the SimNIBS 4.6 legacy-correction reference through the P0 harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dwi2cond_xp.preprocessing import ReferenceArtifact, run_reference_command


def main() -> int:
    """Run an explicit reference fixture and print the manifest path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwi2cond", type=Path, required=True)
    parser.add_argument("--fsl-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    work = args.work.resolve()
    (work / "m2m_synthetic").mkdir(parents=True, exist_ok=True)
    executable = args.dwi2cond.resolve()
    script_root = executable.parent
    fsl_dir = args.fsl_dir.resolve()
    manifest = run_reference_command(
        stage="simnibs46-legacy",
        executable=executable,
        arguments=(
            "--prepro",
            "--keepstuff",
            "synthetic",
            str(fixture / "dwi.nii.gz"),
            str(fixture / "bvals"),
            str(fixture / "bvecs"),
        ),
        working_directory=work,
        manifest_path=args.manifest,
        artifacts=(
            ReferenceArtifact("m2m_synthetic/dMRI_prep/raw/nodif.nii.gz", "nifti"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/raw/nodif_brain_mask.nii.gz", "nifti", mask=True),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/eddycorr/DWI_corr.nii.gz", "nifti"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/eddycorr/DWI_corr_mean.nii.gz", "nifti"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/dti_results_rawspace/DWIforfit.nii.gz", "nifti"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/dti_results_rawspace/DTI_tensor.nii.gz", "nifti"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/dti_results_rawspace/DTI_FA.nii.gz", "nifti"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/dti_results_rawspace/DTI_sse.nii.gz", "nifti"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/dti_results_rawspace/DWIbvecs"),
            ReferenceArtifact("m2m_synthetic/dMRI_prep/dwi2cond_log.html"),
        ),
        environment={
            "FSLDIR": str(fsl_dir),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "OMP_NUM_THREADS": "1",
            "PATH": os.pathsep.join((str(fsl_dir / "bin"), os.environ.get("PATH", ""))),
        },
        reference_version="SimNIBS 4.6.0 dwi2cond 0.4 / FSL 6.0.4:ddd0a010",
        script_paths=(
            executable,
            script_root / "dwi2cond.prepro.source.sh",
            script_root / "dwi2cond.functions.source.sh",
            script_root / "dwi2cond.t1reg.source.sh",
            script_root / "dwi2cond.check.source.sh",
        ),
        threads=1,
        timeout_seconds=1200,
    )
    print(json.dumps({"status": manifest["status"], "manifest": str(args.manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
