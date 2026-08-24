#!/usr/bin/env python3
"""Audit public FSL reference manifests for repository-safe content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dwi2cond_xp.preprocessing import audit_public_manifest


def main() -> int:
    """Validate every JSON manifest and reject image payloads in the directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--forbid", action="append", default=[])
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifests = sorted(directory.rglob("*.json"))
    if not manifests:
        raise SystemExit("No JSON reference manifests were found")
    disallowed = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".nii", ".gz", ".img", ".hdr"}
    ]
    if disallowed:
        raise SystemExit("Public reference assets must not contain image files")
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        audit_public_manifest(payload, forbidden_terms=args.forbid)
    print(json.dumps({"status": "passed", "manifests": len(manifests)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
