#!/usr/bin/env python3
"""Run the SimNIBS 4.6 FSL EDDY path for local numerical A/B testing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np


def main() -> int:
    """Run EDDY with fixed SimNIBS defaults and retain its complete artifact set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--bvecs", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--acqp", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--topup", type=Path)
    parser.add_argument(
        "--init",
        type=Path,
        help="Initialize EDDY from an existing movement/EC parameter table.",
    )
    parser.add_argument("--without-repol", action="store_true")
    parser.add_argument(
        "--dwi-only",
        action="store_true",
        help="Register only DWI scans for a stage-level parameter-trajectory oracle.",
    )
    parser.add_argument(
        "--dont-peas",
        action="store_true",
        help="Disable post-EDDY shell alignment for a stage-level trajectory oracle.",
    )
    parser.add_argument("--niter", type=int, default=5)
    parser.add_argument("--debug", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--debug-indices",
        help="Comma-separated zero-based global volume indices retained by FSL debug.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--initrand",
        type=int,
        help="Set FSL's GP voxel-selection seed for deterministic diagnostic A/B.",
    )
    parser.add_argument("--fsldir", type=Path, default=Path("/usr/local/fsl"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.niter < 1:
        parser.error("--niter must be positive")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / ("eddy_no_repol" if args.without_repol else "eddy_repol")
    executable = args.fsldir.resolve() / "bin" / "eddy_openmp"
    command = [
        str(executable),
        f"--imain={args.dwi.resolve()}",
        f"--mask={args.mask.resolve()}",
        f"--bvecs={args.bvecs.resolve()}",
        f"--bvals={args.bvals.resolve()}",
        f"--out={prefix}",
        f"--acqp={args.acqp.resolve()}",
        f"--index={args.index.resolve()}",
    ]
    if args.topup is not None:
        command.append(f"--topup={args.topup.resolve()}")
    if args.init is not None:
        command.append(f"--init={args.init.resolve()}")
    if args.dwi_only:
        command.append("--dwi_only")
    if args.dont_peas:
        command.append("--dont_peas")
    command.append(f"--niter={args.niter}")
    if args.initrand is not None:
        if args.initrand < 1:
            parser.error("--initrand must be positive when supplied")
        command.append(f"--initrand={args.initrand}")
    if args.debug:
        command.append(f"--debug={args.debug}")
        if args.debug_indices:
            command.append(f"--dbgindx={args.debug_indices}")
    if not args.without_repol:
        command.append("--repol")
    command.append("--verbose")
    environment = os.environ.copy()
    environment.update(
        {
            "FSLDIR": str(args.fsldir.resolve()),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "OMP_NUM_THREADS": str(args.workers),
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    started = perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        cwd=output,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = perf_counter() - started
    log = output / f"{prefix.name}.log"
    log.write_text(completed.stdout, encoding="utf-8")

    outlier_map = Path(f"{prefix}.eddy_outlier_map")
    outliers: list[list[int]] = []
    if outlier_map.exists():
        values = np.loadtxt(outlier_map, skiprows=1, dtype=np.int64, ndmin=2)
        outliers = np.argwhere(values != 0).astype(int).tolist()
    suffixes = (
        ".nii.gz",
        ".eddy_parameters",
        ".eddy_rotated_bvecs",
        ".eddy_movement_rms",
        ".eddy_restricted_movement_rms",
        ".eddy_outlier_map",
        ".eddy_outlier_n_stdev_map",
        ".eddy_outlier_n_sqr_stdev_map",
        ".eddy_outlier_report",
        ".eddy_post_eddy_shell_alignment_parameters",
        ".eddy_post_eddy_shell_PE_translation_parameters",
        ".eddy_command_txt",
        ".eddy_values_of_all_input_parameters",
    )
    if not args.without_repol:
        suffixes = (suffixes[0], ".eddy_outlier_free_data.nii.gz", *suffixes[1:])
    artifacts = [str(Path(f"{prefix}{suffix}")) for suffix in suffixes]
    missing = [path for path in artifacts if not Path(path).exists()]
    version = (args.fsldir / "etc" / "fslversion").read_text(encoding="utf-8").strip()
    report = {
        "status": "complete",
        "reference": "SimNIBS-4.6-APPLY_EDDY",
        "fsl_version": version,
        "workers": args.workers,
        "initrand": args.initrand,
        "repol": not args.without_repol,
        "dwi_only": args.dwi_only,
        "dont_peas": args.dont_peas,
        "niter": args.niter,
        "topup": str(args.topup.resolve()) if args.topup is not None else None,
        "init": str(args.init.resolve()) if args.init is not None else None,
        "debug": args.debug,
        "debug_indices": args.debug_indices,
        "elapsed_seconds": elapsed,
        "command": command,
        "detected_outliers_volume_slice": outliers,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "log": str(log),
    }
    (output / f"{prefix.name}_reference_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
