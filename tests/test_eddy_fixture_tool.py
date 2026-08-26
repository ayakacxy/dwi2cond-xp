from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


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
    assert {
        name: _sha256(output / name)
        for name in reference["fixture"]["input_sha256"]
    } == reference["fixture"]["input_sha256"]
