from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_interspersed_b0_eddy_fixture_reproduces_frozen_inputs(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    output = tmp_path / "fixture"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_synthetic_eddy_fixture.py"),
            str(output),
            "--interspersed-b0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    reference = json.loads(
        (
            root
            / "tests/fixtures/reference/synthetic_eddy_interspersed_b0_reference.json"
        ).read_text(encoding="utf-8")
    )
    generated = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
    assert generated["fixture_id"] == reference["fixture_id"]
    assert generated["b0_indices"] == [0, 13, 25]
    assert generated["slice_corruptions"] == reference["fixture"]["slice_corruptions"]
    actual_hashes = {
        name: _sha256(output / name)
        for name in reference["fixture"]["input_sha256"]
    }
    assert generated["sha256"] == actual_hashes

    # SciPy interpolation can differ by last-bit float32 values across CPU
    # architectures. Keep the Linux file hash as exact FSL provenance, while
    # requiring all non-interpolated inputs to remain byte-identical.
    expected_hashes = reference["fixture"]["input_sha256"]
    exact_names = expected_hashes.keys() - {"dwi.nii"}
    assert {name: actual_hashes[name] for name in exact_names} == {
        name: expected_hashes[name] for name in exact_names
    }

    contract = reference["fixture"]["dwi_numeric_contract"]
    dwi_image = nib.load(output / "dwi.nii")
    dwi = np.asanyarray(dwi_image.dataobj)
    assert list(dwi.shape) == contract["shape"]
    assert str(dwi.dtype) == contract["dtype"]
    np.testing.assert_allclose(dwi_image.affine, contract["affine"], rtol=0.0, atol=0.0)
    assert int(np.count_nonzero(np.isfinite(dwi))) == contract["finite_count"]
    assert int(np.count_nonzero(dwi)) == contract["nonzero_count"]
    assert float(np.sum(dwi, dtype=np.float64)) == pytest.approx(
        contract["sum"], rel=1e-8, abs=0.1
    )
    assert float(np.sum(dwi.astype(np.float64) ** 2, dtype=np.float64)) == pytest.approx(
        contract["l2_squared"], rel=1e-8, abs=10.0
    )
    assert float(np.min(dwi)) == pytest.approx(contract["min"], rel=2e-6, abs=2e-3)
    assert float(np.max(dwi)) == pytest.approx(contract["max"], rel=2e-6, abs=2e-3)
    samples = [float(dwi[13, 13, 9, index]) for index in (0, 5, 11, 13, 18, 23, 25)]
    assert samples == pytest.approx(contract["center_samples"], rel=2e-6, abs=2e-3)
    zero_corruption_counts = [
        int(np.count_nonzero(dwi[:, :, slice_index, volume]))
        for volume, slice_index in ((5, 8), (18, 7), (23, 10))
    ]
    assert zero_corruption_counts == [0, 0, 0]
    assert (
        int(np.count_nonzero(dwi[:, :, 9, 11]))
        == contract["weak_slice_nonzero_count"]
    )
