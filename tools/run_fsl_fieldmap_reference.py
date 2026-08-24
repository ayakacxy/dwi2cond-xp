#!/usr/bin/env python3
"""Run the fixed SimNIBS 4.6 GRE/FUGUE commands to generate FSL A/B references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter


def _run(command: list[str], environment: dict[str, str]) -> None:
    """Run an FSL command while preserving the full argument boundary on failure."""

    subprocess.run(command, check=True, env=environment)


def main() -> int:
    """Reproduce the fixed command sequence after the rad/s input in ``APPLY_FMCORR``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("magnitude")
    parser.add_argument("field_radians_per_second")
    parser.add_argument("magnitude_mask")
    parser.add_argument("b0_brain")
    parser.add_argument("b0_mask")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--dwell-ms", type=float, required=True)
    parser.add_argument(
        "--phase-encoding-direction",
        choices=("x", "x-", "y", "y-", "z", "z-"),
        required=True,
    )
    parser.add_argument("--no-median-filter", action="store_true")
    parser.add_argument("--fsldir", type=Path, default=Path("/usr/local/fsl"))
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"FSLDIR": str(args.fsldir), "FSLOUTPUTTYPE": "NIFTI_GZ"})
    binary = args.fsldir / "bin"
    dwell_seconds = args.dwell_ms / 1000.0
    started = perf_counter()
    commands: list[list[str]] = []

    field = output / "field_radians_per_second"
    if args.no_median_filter:
        commands.append([str(binary / "fslmaths"), args.field_radians_per_second, "-mul", "1", str(field), "-odt", "float"])
    else:
        commands.append([str(binary / "fslmaths"), args.field_radians_per_second, "-fmedian", str(field)])
    for command in commands:
        _run(command, environment)
    median = subprocess.run(
        [str(binary / "fslstats"), str(field), "-k", args.magnitude_mask, "-P", "50"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    subtract = [str(binary / "fslmaths"), str(field), "-sub", median, str(field), "-odt", "float"]
    _run(subtract, environment)
    commands.append(subtract)

    distorted = output / "magnitude_brain_distorted"
    distort = [
        str(binary / "fugue"),
        "-i", args.magnitude,
        f"--loadfmap={field}",
        f"--mask={args.magnitude_mask}",
        f"--dwell={dwell_seconds:.17g}",
        "-w", str(distorted),
        "--nokspace",
        f"--unwarpdir={args.phase_encoding_direction}",
    ]
    _run(distort, environment)
    commands.append(distort)

    matrix = output / "nodif2fieldmap.mat"
    registered = output / "b0_fieldmap_space"
    register = [
        str(binary / "flirt"), "-in", args.b0_brain, "-ref", str(distorted),
        "-omat", str(matrix), "-o", str(registered), "-dof", "6",
        "-cost", "mutualinfo", "-searchcost", "mutualinfo",
    ]
    _run(register, environment)
    commands.append(register)
    inverse = output / "fieldmap2nodif.mat"
    invert = [str(binary / "convert_xfm"), "-omat", str(inverse), "-inverse", str(matrix)]
    _run(invert, environment)
    commands.append(invert)

    mapped = {
        "field_dwi_radians_per_second": (str(field), "float"),
        "magnitude_brain_dwi": (args.magnitude, "float"),
        "fieldmap_mask_dwi_float": (args.magnitude_mask, "float"),
    }
    for name, (source, output_type) in mapped.items():
        command = [
            str(binary / "flirt"), "-in", source, "-ref", args.b0_brain,
            "-applyxfm", "-init", str(inverse), "-out", str(output / name),
            "-datatype", output_type,
        ]
        _run(command, environment)
        commands.append(command)
    mask_dwi = output / "fieldmap_mask_dwi"
    threshold = [
        str(binary / "fslmaths"), str(output / "fieldmap_mask_dwi_float"),
        "-thr", "0.5", "-bin", str(mask_dwi), "-odt", "float",
    ]
    _run(threshold, environment)
    commands.append(threshold)

    corrected = output / "corrected_b0_unmasked"
    shift = output / "voxel_shift"
    unwarp = [
        str(binary / "fugue"),
        f"--loadfmap={output / 'field_dwi_radians_per_second'}",
        f"--dwell={dwell_seconds:.17g}", "-i", args.b0_brain, "-u", str(corrected),
        f"--unwarpdir={args.phase_encoding_direction}", f"--saveshift={shift}",
        f"--mask={mask_dwi}",
    ]
    _run(unwarp, environment)
    commands.append(unwarp)
    warp = output / "dwi_warp"
    convert = [
        str(binary / "convertwarp"), "-s", str(shift), "-o", str(warp),
        "-r", args.b0_brain, f"--shiftdir={args.phase_encoding_direction}",
    ]
    _run(convert, environment)
    commands.append(convert)
    unwarped_mask = output / "b0_mask_unwarped"
    apply = [
        str(binary / "applywarp"), "-i", args.b0_mask, "-r", args.b0_brain,
        "-w", str(warp), "-o", str(unwarped_mask), "--abs", "--interp=sinc",
    ]
    _run(apply, environment)
    commands.append(apply)
    mask = [
        str(binary / "fslmaths"), str(unwarped_mask), "-mul", str(mask_dwi),
        "-bin", str(output / "corrected_mask"), "-odt", "float",
    ]
    _run(mask, environment)
    commands.append(mask)
    final = [
        str(binary / "fslmaths"), str(corrected), "-mas", str(output / "corrected_mask"),
        str(output / "corrected_b0"), "-odt", "float",
    ]
    _run(final, environment)
    commands.append(final)

    version_file = args.fsldir / "etc" / "fslversion"
    version = version_file.read_text(encoding="utf-8").strip()
    report = {
        "status": "complete",
        "reference": "SimNIBS-4.6-APPLY_FMCORR-rads-branch",
        "fsl_version": version,
        "fieldmap_input_units": "radians_per_second",
        "dwell_input_units": "milliseconds",
        "dwell_seconds": dwell_seconds,
        "voxel_shift_units": "voxels",
        "phase_encoding_direction": args.phase_encoding_direction,
        "elapsed_seconds": perf_counter() - started,
        "commands": commands,
    }
    (output / "reference_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
