"""Verify the isolated runtime against the frozen SimNIBS 4.6 reference environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")


def load_pins(path: str | Path) -> dict[str, str]:
    """Read runtime constraints containing only exact ``name==version`` pins."""

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Constraint line {line_number} is not an exact pin: {line}")
        name, version = match.groups()
        normalized = name.lower().replace("_", "-")
        if normalized in pins:
            raise ValueError(f"Duplicate runtime pin: {name}")
        pins[normalized] = version
    if not pins:
        raise ValueError("Runtime constraints contain no package pins")
    return pins


def verify_environment(
    pins: dict[str, str],
    *,
    required_python: tuple[int, int, int] = (3, 11, 15),
) -> dict[str, object]:
    """Compare Python and installed distribution versions without importing SimNIBS or MPI."""

    actual_python = tuple(sys.version_info[:3])
    mismatches: dict[str, dict[str, str]] = {}
    if actual_python != required_python:
        mismatches["python"] = {
            "expected": ".".join(str(value) for value in required_python),
            "actual": ".".join(str(value) for value in actual_python),
        }
    installed: dict[str, str] = {}
    for name, expected in pins.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        installed[name] = actual
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    return {
        "status": "matched" if not mismatches else "mismatched",
        "required_python": ".".join(str(value) for value in required_python),
        "actual_python": ".".join(str(value) for value in actual_python),
        "packages": installed,
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the command-line gate and emit auditable results as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constraints",
        type=Path,
        default=ROOT / "constraints-simnibs46.txt",
    )
    parser.add_argument("--json", type=Path, help="Optional report output path")
    args = parser.parse_args(argv)
    report = verify_environment(load_pins(args.constraints))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "matched":
        raise SystemExit("SimNIBS 4.6 runtime environment does not match frozen pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
