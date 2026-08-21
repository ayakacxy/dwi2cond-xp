"""Check local Markdown links without making network requests."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    return [ROOT / "README.md", ROOT / "README.zh-CN.md"] + sorted(
        (ROOT / "docs").glob("*.md")
    )


def main() -> int:
    findings: list[str] = []
    for source in _markdown_files():
        text = source.read_text(encoding="utf-8")
        for raw_target in LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (source.parent / relative).resolve().exists():
                findings.append(f"{source.relative_to(ROOT)}: missing {target}")
    if findings:
        raise SystemExit("\n".join(findings))
    print(f"Local Markdown link check passed for {len(_markdown_files())} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
