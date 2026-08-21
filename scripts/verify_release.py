"""Audit repository and distribution contents for release-only contracts."""

from __future__ import annotations

import argparse
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path


FORBIDDEN_PARTS = {
    ".codex",
    ".agents",
    ".pytest_cache",
    "__pycache__",
    "data",
    "runs",
}
FORBIDDEN_SUFFIXES = {
    ".nii",
    ".nii.gz",
    ".msh",
    ".hdf5",
    ".mat",
    ".pyc",
    ".pyo",
}
PRIVATE_PATTERNS = (
    re.compile(b"/" + b"home" + rb"/[^/\s]+/"),
    re.compile(b"H" + b"CP" + rb"[_-]?\d{6}", re.IGNORECASE),
)


def _is_forbidden(name: str) -> bool:
    path = Path(name)
    lowered = {part.lower() for part in path.parts}
    if lowered & {part.lower() for part in FORBIDDEN_PARTS}:
        return True
    joined_suffixes = "".join(path.suffixes).lower()
    return any(joined_suffixes.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def _audit_bytes(name: str, content: bytes) -> list[str]:
    findings = []
    for pattern in PRIVATE_PATTERNS:
        if pattern.search(content):
            findings.append(f"private pattern in {name}: {pattern.pattern!r}")
    return findings


def audit_archive(path: Path) -> list[str]:
    findings: list[str] = []
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if _is_forbidden(name):
                    findings.append(f"forbidden archive member: {name}")
                if not name.endswith("/"):
                    findings.extend(_audit_bytes(name, archive.read(name)))
        return findings
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if _is_forbidden(member.name):
                findings.append(f"forbidden archive member: {member.name}")
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    findings.extend(_audit_bytes(member.name, handle.read()))
    return findings


def audit_repository(root: Path) -> list[str]:
    """Audit only Git-tracked files, so ignored local research data is untouched."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    findings: list[str] = []
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8")
        if _is_forbidden(name):
            findings.append(f"forbidden tracked path: {name}")
            continue
        path = root / name
        if path.is_file():
            findings.extend(_audit_bytes(name, path.read_bytes()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--dist", type=Path)
    targets.add_argument("--repository", type=Path)
    args = parser.parse_args()
    if args.repository is not None:
        findings = audit_repository(args.repository.resolve())
        if findings:
            raise SystemExit("\n".join(findings))
        print("Tracked repository privacy audit passed.")
        return 0
    assert args.dist is not None
    archives = sorted(args.dist.glob("*.whl")) + sorted(args.dist.glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"No wheel or sdist found in {args.dist}")
    findings = []
    for archive in archives:
        findings.extend(f"{archive.name}: {item}" for item in audit_archive(archive))
    if findings:
        raise SystemExit("\n".join(findings))
    print(f"Release audit passed for {len(archives)} archive(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
