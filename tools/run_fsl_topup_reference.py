#!/usr/bin/env python3
"""Run the SimNIBS 4.6 TOPUP fixed path for local numerical A/B testing."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter


def _run(command: list[str], environment: dict[str, str]) -> None:
    """Run one reference command and preserve its exact argument boundary."""

    subprocess.run(command, check=True, env=environment)


def _simnibs_external_file(name: str) -> Path:
    """Locate one installed SimNIBS external reference file."""

    spec = find_spec("simnibs")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("SimNIBS is required unless --config is provided")
    return Path(next(iter(spec.submodule_search_locations))) / "external" / name


def main() -> int:
    """Reproduce ``APPLY_TOPUP`` and retain extra field/iout A/B artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward_b0")
    parser.add_argument("reverse_b0")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--phase-encoding-direction",
        choices=("x", "x-", "y", "y-", "z", "z-"),
        required=True,
    )
    parser.add_argument("--readout-seconds", type=float, required=True)
    parser.add_argument("--fsldir", type=Path, default=Path("/usr/local/fsl"))
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to the SimNIBS b02b0_nosubsamp.cnf file",
    )
    args = parser.parse_args()
    config = args.config or _simnibs_external_file("b02b0_nosubsamp.cnf")
    if not args.readout_seconds > 0.0:
        parser.error("--readout-seconds must be positive")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = args.fsldir / "bin"
    environment = os.environ.copy()
    environment.update({"FSLDIR": str(args.fsldir), "FSLOUTPUTTYPE": "NIFTI_GZ"})
    directions = {
        "x": ((1, 0, 0), (-1, 0, 0)),
        "x-": ((-1, 0, 0), (1, 0, 0)),
        "y": ((0, 1, 0), (0, -1, 0)),
        "y-": ((0, -1, 0), (0, 1, 0)),
        "z": ((0, 0, 1), (0, 0, -1)),
        "z-": ((0, 0, -1), (0, 0, 1)),
    }
    acqp = output / "acquisition_parameters.txt"
    acqp.write_text(
        "".join(
            f"{row[0]} {row[1]} {row[2]} {args.readout_seconds:.17g}\n"
            for row in directions[args.phase_encoding_direction]
        ),
        encoding="utf-8",
    )
    merged = output / "nodif_topup"
    result = output / "topup_res"
    commands = [
        [
            str(binary / "fslmerge"),
            "-t",
            str(merged),
            args.forward_b0,
            args.reverse_b0,
        ],
        [
            str(binary / "topup"),
            f"--imain={merged}",
            f"--datain={acqp}",
            f"--config={config.resolve()}",
            f"--out={result}",
            f"--iout={output / 'corrected_pair'}",
            f"--fout={output / 'field_hz'}",
            f"--logout={output / 'topup.settings'}",
        ],
        [
            str(binary / "fslroi"),
            str(output / "corrected_pair"),
            str(output / "corrected_forward_b0"),
            "0",
            "1",
        ],
        [
            str(binary / "bet"),
            str(output / "corrected_forward_b0"),
            str(output / "corrected_forward_brain"),
            "-f",
            "0.2",
            "-m",
        ],
    ]
    started = perf_counter()
    for command in commands:
        _run(command, environment)
    elapsed = perf_counter() - started
    version = (args.fsldir / "etc" / "fslversion").read_text(encoding="utf-8").strip()
    report = {
        "status": "complete",
        "reference": "SimNIBS-4.6-APPLY_TOPUP",
        "fsl_version": version,
        "config": "b02b0_nosubsamp.cnf",
        "phase_encoding_direction": args.phase_encoding_direction,
        "readout_seconds": args.readout_seconds,
        "elapsed_seconds": elapsed,
        "commands": commands,
        "outputs": {
            "field_coefficients_hz": str(output / "topup_res_fieldcoef.nii.gz"),
            "movement_parameters": str(output / "topup_res_movpar.txt"),
            "field_hz": str(output / "field_hz.nii.gz"),
            "corrected_pair": str(output / "corrected_pair.nii.gz"),
            "corrected_forward_b0": str(output / "corrected_forward_b0.nii.gz"),
            "corrected_forward_brain": str(output / "corrected_forward_brain.nii.gz"),
            "corrected_forward_mask": str(
                output / "corrected_forward_brain_mask.nii.gz"
            ),
        },
    }
    (output / "reference_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
