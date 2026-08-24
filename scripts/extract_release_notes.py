"""Extract one tagged release section from the project changelog."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_HEADING = re.compile(r"^## \[(?P<version>[^]]+)](?:\s+-\s+.+)?$")


def extract_release_notes(changelog: str, tag: str) -> str:
    """Return the changelog body for a tag such as ``v0.2.0``."""

    version = tag.removeprefix("v")
    lines = changelog.splitlines()
    start: int | None = None
    end = len(lines)
    for index, line in enumerate(lines):
        match = VERSION_HEADING.match(line)
        if match is None:
            continue
        if start is None and match.group("version") == version:
            start = index + 1
            continue
        if start is not None:
            end = index
            break
    if start is None:
        raise ValueError(f"No changelog section found for tag {tag}")
    body = "\n".join(lines[start:end]).strip()
    if not body:
        raise ValueError(f"Changelog section for tag {tag} is empty")
    return f"## dwi2cond-xp {tag}\n\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--changelog", type=Path, default=Path("docs/CHANGELOG.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        notes = extract_release_notes(
            args.changelog.read_text(encoding="utf-8"),
            args.tag,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output.write_text(notes, encoding="utf-8")
    print(f"Wrote release notes for {args.tag} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
