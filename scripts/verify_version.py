"""Verify that public release metadata uses one semantic version."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _cff_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version:\s*[\"']?([^\s\"']+)", text)
    if match is None:
        raise ValueError("CITATION.cff does not contain a version")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Optional release tag, for example v0.1.0")
    args = parser.parse_args()

    with (ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    versions = {"pyproject.toml": version, "CITATION.cff": _cff_version()}
    changelog = (ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in changelog:
        raise SystemExit(f"docs/CHANGELOG.md has no section for {version}")
    if args.tag is not None and args.tag != f"v{version}":
        versions["tag"] = args.tag.removeprefix("v")
    mismatched = {name: value for name, value in versions.items() if value != version}
    if mismatched:
        raise SystemExit(f"Release version mismatch: {mismatched}; expected {version}")
    print(f"Release metadata is synchronized at {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
