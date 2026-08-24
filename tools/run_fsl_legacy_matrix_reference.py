#!/usr/bin/env python3
"""Preserve the final per-volume FSL matrices from SimNIBS legacy correction."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np


def _run(command: list[str], cwd: Path) -> None:
    """Run an FSL reference command and stop immediately on failure."""

    subprocess.run(command, cwd=cwd, check=True)


def _mean_dwi(input_file: Path, bvals: np.ndarray, output: str, cwd: Path) -> None:
    """Compute the b>0 mean in the script's fslsplit/fslmerge/fslmaths order."""

    prefix = f"{output}_split"
    _run(["fslsplit", str(input_file), prefix], cwd)
    selected = [f"{prefix}{index:04d}" for index, value in enumerate(bvals) if value > 0]
    _run(["fslmerge", "-t", output, *selected], cwd)
    _run(["fslmaths", output, "-Tmean", output], cwd)


def main() -> int:
    """Generate the final matrix directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dwi", type=Path)
    parser.add_argument("bvals", type=Path)
    parser.add_argument("nodif", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dwi = args.dwi.resolve()
    nodif = args.nodif.resolve()
    bvals = np.asarray(np.loadtxt(args.bvals), dtype=np.float64).reshape(-1)
    _mean_dwi(dwi, bvals, "DWIraw_mean", output)
    _run(["mcflirt", "-in", str(dwi), "-o", "DWI_pass1_corr", "-dof", "6", "-reffile", "DWIraw_mean"], output)
    _mean_dwi(output / "DWI_pass1_corr", bvals, "DWI_pass1_mean", output)
    _run(["mcflirt", "-in", str(dwi), "-o", "DWI_corr", "-dof", "12", "-sinc_final", "-reffile", "DWI_pass1_mean", "-mats"], output)
    _mean_dwi(output / "DWI_corr", bvals, "DWI_corr_mean", output)
    _run(["flirt", "-in", "DWI_corr_mean", "-ref", str(nodif), "-nosearch", "-cost", "mutualinfo", "-interp", "sinc", "-omat", "meanDWI2nodif.mat", "-o", "meanDWI2nodif_tst"], output)
    matrix_directory = output / "DWI_corr.mat"
    for matrix in sorted(matrix_directory.glob("MAT_*")):
        _run(["convert_xfm", "-omat", str(matrix), "-concat", "meanDWI2nodif.mat", str(matrix)], output)
    _run(["mcflirt", "-in", str(dwi), "-o", "DWI_b0", "-dof", "6", "-reffile", str(nodif), "-mats"], output)
    b0_directory = output / "DWI_b0.mat"
    for index, value in enumerate(bvals):
        if value == 0:
            shutil.copyfile(b0_directory / f"MAT_{index:04d}", matrix_directory / f"MAT_{index:04d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
