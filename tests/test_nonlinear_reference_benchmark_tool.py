from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np


def _write_nifti(path: Path, shape: tuple[int, ...]) -> None:
    values = np.zeros(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)


def _write_source_fixture(root: Path) -> tuple[Path, Path]:
    simnibs = root / "simnibs_external"
    fsl = root / "fsl"
    files = (
        simnibs / "dwi2cond",
        simnibs / "dwi2cond.t1reg.source.sh",
        fsl / "src/fnirt/fnirt.cpp",
        fsl / "src/fnirt/fnirtfns.cpp",
        fsl / "src/fnirt/fnirt_costfunctions.cpp",
        fsl / "src/fdt/vecreg.cc",
        fsl / "src/avwutils/fslmaths.cc",
    )
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    return simnibs, fsl


def test_nonlinear_preflight_freezes_inputs_sources_and_core_contract(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    fa = tmp_path / "fa.nii.gz"
    tensor = tmp_path / "tensor.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    brain_mask = tmp_path / "brain_mask.nii.gz"
    affine = tmp_path / "affine.mat"
    _write_nifti(fa, (3, 4, 5))
    _write_nifti(tensor, (3, 4, 5, 6))
    _write_nifti(reference, (4, 5, 6))
    _write_nifti(brain_mask, (4, 5, 6))
    np.savetxt(affine, np.eye(4))
    simnibs, fsl = _write_source_fixture(tmp_path)
    manifest = tmp_path / "run/manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools/run_nonlinear_reference_benchmark.py"),
            "--implementation",
            "python",
            "--fa",
            str(fa),
            "--tensor",
            str(tensor),
            "--reference",
            str(reference),
            "--affine",
            str(affine),
            "--brain-mask",
            str(brain_mask),
            "--work",
            str(tmp_path / "work"),
            "--manifest",
            str(manifest),
            "--fsl-dir",
            str(fsl),
            "--simnibs-external",
            str(simnibs),
            "--preflight-only",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    contract = json.loads(Path(report["contract"]).read_text(encoding="utf-8"))
    assert report["status"] == "preflight-completed"
    assert contract["workers"] == 8
    assert [item["alias"] for item in contract["input_contract"]] == [
        "affine_matrix",
        "brain_mask",
        "fa",
        "reference",
        "tensor",
    ]
    assert len(contract["algorithm_sources"]) == 11
    assert all(len(item["sha256"]) == 64 for item in contract["algorithm_sources"])
    assert "strict_worker_affinity" in contract["runtime"]


def test_fsl_fnirt_runner_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools/run_fsl_fnirt_reference.py"),
            "--fa",
            str(tmp_path / "fa"),
            "--tensor",
            str(tmp_path / "tensor"),
            "--reference",
            str(tmp_path / "reference"),
            "--affine",
            str(tmp_path / "affine"),
            "--brain-mask",
            str(tmp_path / "brain_mask"),
            "--work",
            str(tmp_path / "work"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--timeout-seconds",
            "0",
        ],
        check=False,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--timeout-seconds must be positive" in completed.stderr
