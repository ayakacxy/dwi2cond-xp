#!/usr/bin/env python3
"""Run the fixed nonlinear official reference benchmark with source and input fingerprints."""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from dwi2cond_xp.preprocessing import (
    ReferenceArtifact,
    run_reference_command,
    summarize_fixture_inputs,
)


def _positive_float(value: str) -> float:
    """Parse a strictly positive floating-point parameter."""

    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON atomically in the same directory to avoid leaving a partial contract after a long task."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Compute a file-content hash in fixed-size blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    """Record the current commit; return an empty value for a non-Git source package."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cpu_affinity() -> list[int] | None:
    """Read Linux process affinity; explicitly return an empty value on other platforms."""

    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return sorted(int(cpu) for cpu in getter(0))


def _physical_cores(cpus: list[int] | None) -> list[list[int]] | None:
    """Map Linux logical CPUs to package/core pairs for the eight-physical-core gate audit."""

    if cpus is None:
        return None
    cores: set[tuple[int, int]] = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError):
            return None
        cores.add((package, core))
    return [[package, core] for package, core in sorted(cores)]


def _simnibs_external_directory(explicit: Path | None) -> Path:
    """Resolve the SimNIBS external source directory without relying on the current shell's PATH."""

    if explicit is not None:
        return explicit.resolve()
    spec = find_spec("simnibs")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            "SimNIBS was not found in the current environment; use --simnibs-external "
            "to specify the SimNIBS 4.6 external directory"
        )
    return Path(next(iter(spec.submodule_search_locations))) / "external"


def _source_files(
    root: Path,
    fsl_dir: Path,
    simnibs_external: Path,
) -> tuple[Path, ...]:
    """Return the upstream and local sources that define the current nonlinear algorithm contract."""

    candidates = (
        simnibs_external / "dwi2cond",
        simnibs_external / "dwi2cond.t1reg.source.sh",
        fsl_dir / "src/fnirt/fnirt.cpp",
        fsl_dir / "src/fnirt/fnirtfns.cpp",
        fsl_dir / "src/fnirt/fnirt_costfunctions.cpp",
        fsl_dir / "src/fdt/vecreg.cc",
        fsl_dir / "src/avwutils/fslmaths.cc",
        root / "src/dwi2cond_xp/preprocessing/fnirt.py",
        root / "src/dwi2cond_xp/preprocessing/fnirt_topology.py",
        root / "src/dwi2cond_xp/preprocessing/nonlinear.py",
        root / "src/dwi2cond_xp/preprocessing/tensor_ops.py",
    )
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing nonlinear source-audit files: " + ", ".join(str(path) for path in missing)
        )
    return candidates


def _source_summary(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    """Freeze source names, sizes, and hashes without writing private absolute paths into the contract."""

    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _build_contract(args: argparse.Namespace, root: Path) -> dict[str, object]:
    """Build the same-input, same-algorithm source contract outside the timing interval."""

    affinity = _cpu_affinity()
    physical = _physical_cores(affinity)
    simnibs_external = _simnibs_external_directory(args.simnibs_external)
    source_files = _source_files(root, args.fsl_dir.resolve(), simnibs_external)
    return {
        "schema_version": 1,
        "boundary": "SimNIBS 4.6 fixed FNIRT plus FSL vecreg PPD stage",
        "implementation": args.implementation,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "input_contract": summarize_fixture_inputs(
            {
                "affine_matrix": args.affine,
                "brain_mask": args.brain_mask,
                "fa": args.fa,
                "reference": args.reference,
                "tensor": args.tensor,
            },
            nifti_aliases=("brain_mask", "fa", "reference", "tensor"),
            mask_aliases=("brain_mask",),
            include_digests=True,
        ),
        "algorithm_sources": _source_summary(source_files),
        "algorithm_flow": [
            "dwi2cond: FLIRT 12-DOF matrix",
            "FNIRT: subsamp=8,4,2,2 with unchanged defaults",
            "VECREG: nonlinear trilinear tensor sampling plus PPD",
            "fslmaths: apply T1 brain mask after vecreg",
            "fslmaths: tensor_decomp derived FA and V1",
        ],
        "runtime": {
            "git_commit": _git_commit(root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_affinity": affinity,
            "physical_core_ids": physical,
            "strict_worker_affinity": affinity is not None
            and physical is not None
            and len(affinity) == args.workers
            and len(physical) == args.workers,
        },
    }


def _validate_fresh_output(path: Path) -> None:
    """Reject reuse of an existing artifact to guarantee a fresh-output timing boundary."""

    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Benchmark output directory must not exist or must be empty: {path}")


def _python_artifacts() -> tuple[ReferenceArtifact, ...]:
    """List the required outputs at the Python nonlinear product boundary."""

    return (
        ReferenceArtifact("FA2T1_warp.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_field.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_jacobian.nii.gz", "nifti"),
        ReferenceArtifact("DTI_FA_nonlin.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_tensor.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_FA.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_V1.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_jacobian.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_valid_mask.nii.gz", "nifti", mask=True),
        ReferenceArtifact("DTI_coregT1_nonlinear_qa.json"),
        ReferenceArtifact("nonlinear_registration_qa.json"),
    )


def _fsl_artifacts() -> tuple[ReferenceArtifact, ...]:
    """List the shared scientific outputs actually written by the original FSL nonlinear branch."""

    return (
        ReferenceArtifact("FA2T1_warp.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_field.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_jacobian.nii.gz", "nifti"),
        ReferenceArtifact("DTI_FA_nonlin.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_tensor.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_FA.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_V1.nii.gz", "nifti"),
        ReferenceArtifact("fnirt.log"),
    )


def main() -> int:
    """Freeze the contract and run one fresh-output nonlinear stage through the explicit implementation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("python", "fsl"), required=True)
    parser.add_argument("--fa", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--fsl-dir", type=Path, default=Path("/usr/local/fsl"))
    parser.add_argument("--simnibs-external", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=7200.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="For preflight or smoke tests only; do not use for formal performance runs",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be greater than zero")

    root = Path(__file__).resolve().parents[1]
    contract_path = (
        args.contract.resolve()
        if args.contract is not None
        else args.manifest.resolve().with_name(f"{args.manifest.stem}.contract.json")
    )
    contract = _build_contract(args, root)
    _atomic_json(contract_path, contract)
    if args.preflight_only:
        print(
            json.dumps(
                {"status": "preflight-completed", "contract": str(contract_path)}
            )
        )
        return 0

    if not contract["runtime"]["strict_worker_affinity"] and not args.allow_unpinned:
        parser.error(
            "Formal runs require the process to be pinned to the same number of independent "
            "physical cores as --workers; bind it with taskset first, and use "
            "--allow-unpinned only for smoke tests"
        )
    work = args.work.resolve()
    _validate_fresh_output(work)
    fsl_dir = args.fsl_dir.resolve()
    environment = {
        "FSLDIR": str(fsl_dir),
        "FSLOUTPUTTYPE": "NIFTI_GZ",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": str(args.workers),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": os.pathsep.join((str(fsl_dir / "bin"), os.environ.get("PATH", ""))),
    }
    source_files = _source_files(
        root,
        fsl_dir,
        _simnibs_external_directory(args.simnibs_external),
    ) + (contract_path,)
    if args.implementation == "python":
        arguments = (
            "-m",
            "dwi2cond_xp",
            "register-t1-nonlinear",
            str(args.fa.resolve()),
            str(args.tensor.resolve()),
            str(args.reference.resolve()),
            str(args.affine.resolve()),
            str(work),
            "--brain-mask",
            str(args.brain_mask.resolve()),
            "--workers",
            str(args.workers),
            "--progress",
            "off",
        )
        artifacts = _python_artifacts()
        stage = "dwi2cond-xp-v030-fixed-nonlinear"
        version = "dwi2cond-xp current source"
        threads = args.workers
    else:
        runner = root / "tools/run_fsl_fnirt_reference.py"
        arguments = (
            str(runner),
            "--worker",
            "--fa",
            str(args.fa.resolve()),
            "--tensor",
            str(args.tensor.resolve()),
            "--reference",
            str(args.reference.resolve()),
            "--affine",
            str(args.affine.resolve()),
            "--brain-mask",
            str(args.brain_mask.resolve()),
            "--work",
            str(work),
            "--fsl-dir",
            str(fsl_dir),
        )
        artifacts = _fsl_artifacts()
        stage = "simnibs46-fsl604-v030-fixed-nonlinear"
        version = "SimNIBS 4.6.0 dwi2cond 0.4 / FSL 6.0.4:ddd0a010"
        threads = 1

    manifest = run_reference_command(
        stage=stage,
        executable=sys.executable,
        arguments=arguments,
        working_directory=work,
        manifest_path=args.manifest,
        artifacts=artifacts,
        environment=environment,
        reference_version=version,
        script_paths=source_files,
        threads=threads,
        timeout_seconds=args.timeout_seconds,
        include_output_digests=True,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "contract": str(contract_path),
                "manifest": str(args.manifest.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
