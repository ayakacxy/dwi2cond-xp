#!/usr/bin/env python3
"""Collect stable full-package coverage with isolated Numba caches and real synthetic E2E cases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str], *, environment: dict[str, str]) -> None:
    """Run a coverage subprocess that must succeed from the repository root."""

    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def _coverage_environment(
    base: dict[str, str], data_file: Path, cache_directory: Path
) -> dict[str, str]:
    """Build an isolated coverage and Numba execution environment."""

    environment = dict(base)
    environment["COVERAGE_FILE"] = str(data_file)
    environment["NUMBA_JIT_COVERAGE"] = "1"
    environment["NUMBA_CACHE_DIR"] = str(cache_directory)
    environment["MPLCONFIGDIR"] = str(cache_directory.parent / "matplotlib")
    source = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else os.pathsep.join((source, existing))
    return environment


def _coverage_run(
    label: str,
    arguments: list[str],
    *,
    workspace: Path,
    base_environment: dict[str, str],
    environment_updates: dict[str, str] | None = None,
) -> Path:
    """Run a coverage batch with an independent data file and cold compilation cache."""

    data_file = workspace / f"coverage-{label}"
    cache_directory = workspace / f"numba-{label}"
    environment = _coverage_environment(base_environment, data_file, cache_directory)
    if environment_updates is not None:
        environment.update(environment_updates)
    _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--source=dwi2cond_xp",
            *arguments,
        ],
        environment=environment,
    )
    if not data_file.is_file():
        raise RuntimeError(f"Coverage batch did not produce a data file: {label}")
    return data_file


def _prepare_topup_fixture(directory: Path, environment: dict[str, str]) -> tuple[Path, Path]:
    """Generate a public non-anatomical reverse-PE fixture and split its inputs."""

    _run(
        [
            sys.executable,
            "tools/generate_synthetic_preprocessing_fixtures.py",
            str(directory),
        ],
        environment=environment,
    )
    pair = nib.load(directory / "reverse_pe_b0.nii.gz")
    values = np.asarray(pair.dataobj, dtype=np.float32)
    forward = directory / "forward.nii.gz"
    reverse = directory / "reverse.nii.gz"
    for path, volume in ((forward, values[..., 0]), (reverse, values[..., 1])):
        image = nib.Nifti1Image(volume, pair.affine, pair.header.copy())
        image.set_qform(pair.get_qform(), int(pair.header["qform_code"]))
        image.set_sform(pair.get_sform(), int(pair.header["sform_code"]))
        nib.save(image, path)
    return forward, reverse


def _collect_e2e_coverage(
    workspace: Path, base_environment: dict[str, str]
) -> list[Path]:
    """Run the TOPUP, EDDY, and FNIRT real synthetic-data paths."""

    generator_environment = dict(base_environment)
    source = str(ROOT / "src")
    existing = generator_environment.get("PYTHONPATH")
    generator_environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )

    topup_fixture = workspace / "topup-fixture"
    forward, reverse = _prepare_topup_fixture(topup_fixture, generator_environment)
    topup = _coverage_run(
        "topup-e2e",
        [
            "-m",
            "dwi2cond_xp",
            "prepare-topup",
            str(forward),
            str(reverse),
            str(workspace / "topup-output"),
            "--readout-seconds",
            "0.05",
            "--phase-encoding-direction",
            "y",
            "--workers",
            "8",
            "--progress",
            "off",
        ],
        workspace=workspace,
        base_environment=base_environment,
    )

    eddy_fixture = workspace / "eddy-fixture"
    _run(
        [sys.executable, "tools/generate_synthetic_eddy_fixture.py", str(eddy_fixture)],
        environment=generator_environment,
    )
    eddy_image = nib.load(eddy_fixture / "dwi.nii")
    susceptibility = eddy_fixture / "field_hz.nii.gz"
    field_image = nib.Nifti1Image(
        np.zeros(eddy_image.shape[:3], dtype=np.float32),
        eddy_image.affine,
        eddy_image.header.copy(),
    )
    field_image.set_qform(eddy_image.get_qform(), int(eddy_image.header["qform_code"]))
    field_image.set_sform(eddy_image.get_sform(), int(eddy_image.header["sform_code"]))
    nib.save(field_image, susceptibility)
    eddy = _coverage_run(
        "eddy-e2e",
        [
            "-m",
            "dwi2cond_xp",
            "prepare-eddy",
            str(eddy_fixture / "dwi.nii"),
            str(eddy_fixture / "bvals"),
            str(eddy_fixture / "bvecs"),
            str(eddy_fixture / "mask.nii"),
            str(workspace / "eddy-output"),
            "--readout-seconds",
            "0.05",
            "--phase-encoding-direction",
            "y",
            "--susceptibility-field",
            str(susceptibility),
            "--workers",
            "8",
            "--progress",
            "off",
        ],
        workspace=workspace,
        base_environment=base_environment,
    )

    fnirt_fixture = workspace / "fnirt-fixture"
    _run(
        [
            sys.executable,
            "tools/generate_synthetic_t1_registration_fixture.py",
            str(fnirt_fixture),
        ],
        environment=generator_environment,
    )
    affine = fnirt_fixture / "affine.mat"
    np.savetxt(affine, np.eye(4), fmt="%.17g")
    fnirt = _coverage_run(
        "fnirt-e2e",
        [
            "-m",
            "dwi2cond_xp",
            "register-t1-nonlinear",
            str(fnirt_fixture / "DTI_FA.nii.gz"),
            str(fnirt_fixture / "DTI_tensor.nii.gz"),
            str(fnirt_fixture / "T1.nii.gz"),
            str(affine),
            str(workspace / "fnirt-output"),
            "--brain-mask",
            str(fnirt_fixture / "T1_brain_mask.nii.gz"),
            "--workers",
            "8",
            "--progress",
            "off",
        ],
        workspace=workspace,
        base_environment=base_environment,
    )
    return [topup, eddy, fnirt]


def main() -> int:
    """Run the stable full-package 100% statement-coverage gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=ROOT / ".coverage",
        help="Combined coverage data file; defaults to .coverage in the repository root",
    )
    parser.add_argument(
        "--keep-workspace",
        type=Path,
        help="Keep intermediate fixtures, caches, and per-batch coverage files for audit",
    )
    args = parser.parse_args()
    base_environment = os.environ.copy()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="dwi2cond-coverage-")
        workspace = Path(temporary.name)
    else:
        workspace = args.keep_workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

    try:
        unit = _coverage_run(
            "unit",
            ["-m", "pytest", "-q", "--ignore=tests/test_montage_plot.py"],
            workspace=workspace,
            base_environment=base_environment,
        )
        montage = _coverage_run(
            "montage",
            ["-m", "pytest", "-q", "tests/test_montage_plot.py"],
            workspace=workspace,
            base_environment=base_environment,
            environment_updates={"NUMBA_DISABLE_JIT": "1"},
        )
        data_files = [
            unit,
            montage,
            *_collect_e2e_coverage(workspace, base_environment),
        ]
        destination = args.data_file.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        combine_environment = dict(base_environment)
        combine_environment["COVERAGE_FILE"] = str(destination)
        # coverage combine consumes its input files, so copy them first to retain
        # keep-workspace audit evidence.
        combine_inputs = []
        for index, path in enumerate(data_files):
            copied = workspace / f"combine-{index}-{path.name}"
            shutil.copy2(path, copied)
            combine_inputs.append(copied)
        _run(
            [sys.executable, "-m", "coverage", "combine", *map(str, combine_inputs)],
            environment=combine_environment,
        )
        _run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "-m",
                "--fail-under=100",
            ],
            environment=combine_environment,
        )
        print(f"coverage data: {destination}", flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
