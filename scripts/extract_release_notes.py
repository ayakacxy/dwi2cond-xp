"""从更新记录提取指定版本，可选生成中英双语发布正文。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_HEADING = re.compile(r"^## \[(?P<version>[^]]+)](?:\s+-\s+.+)?$")


def extract_release_notes(
    changelog: str,
    tag: str,
    *,
    chinese_changelog: str | None = None,
) -> str:
    """提取指定标签；未提供中文记录时保留原单语言格式。"""

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
    heading = f"## dwi2cond-xp {tag}"
    if chinese_changelog is None:
        return f"{heading}\n\n{body}\n"
    chinese_notes = extract_release_notes(chinese_changelog, tag)
    chinese_body = chinese_notes.split("\n\n", 1)[1].strip()
    return f"{heading}\n\n## English\n\n{body}\n\n## 简体中文\n\n{chinese_body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--changelog", type=Path, default=Path("docs/CHANGELOG.md"))
    parser.add_argument("--chinese-changelog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        notes = extract_release_notes(
            args.changelog.read_text(encoding="utf-8"),
            args.tag,
            chinese_changelog=(
                args.chinese_changelog.read_text(encoding="utf-8")
                if args.chinese_changelog is not None
                else None
            ),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output.write_text(notes, encoding="utf-8")
    print(f"Wrote release notes for {args.tag} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
