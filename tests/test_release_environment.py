"""Verify parsing and difference reports for the frozen SimNIBS 4.6 runtime gate."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from scripts.verify_simnibs_environment import load_pins, verify_environment


def test_frozen_runtime_constraints_match_documented_environment() -> None:
    pins = load_pins(Path("constraints-simnibs46.txt"))

    assert pins == {
        "h5py": "3.15.1",
        "matplotlib": "3.10.8",
        "mne": "1.12.1",
        "nibabel": "5.3.3",
        "numba": "0.64.0",
        "numpy": "2.3.0",
        "scipy": "1.17.1",
        "simnibs": "4.6.0",
        "threadpoolctl": "3.6.0",
        "tqdm": "4.68.4",
    }


def test_runtime_verifier_reports_missing_and_mismatched_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(name: str) -> str:
        if name == "missing":
            raise importlib.metadata.PackageNotFoundError(name)
        return "2.0"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    report = verify_environment(
        {"present": "1.0", "missing": "1.0"},
        required_python=(0, 0, 0),
    )

    assert report["status"] == "mismatched"
    assert report["mismatches"]["present"] == {
        "expected": "1.0",
        "actual": "2.0",
    }
    assert report["mismatches"]["missing"]["actual"] == "missing"
    assert "python" in report["mismatches"]


@pytest.mark.parametrize(
    "content, message",
    (
        ("numpy>=2\n", "not an exact pin"),
        ("numpy==2\nnumpy==2\n", "Duplicate runtime pin"),
        ("# only comment\n", "contain no package pins"),
    ),
)
def test_runtime_constraint_parser_rejects_ambiguous_files(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "constraints.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_pins(path)
