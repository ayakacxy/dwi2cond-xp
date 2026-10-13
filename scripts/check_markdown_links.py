"""检查本地文档链接、双语配对和同语言导航，不发起网络请求。"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    entries = [
        "README.md", "README.zh-CN.md", "THIRD_PARTY_NOTICES.md",
        "THIRD_PARTY_NOTICES.zh-CN.md", ".github/pull_request_template.md",
    ]
    return [ROOT / name for name in entries if (ROOT / name).is_file()] + sorted(
        (ROOT / "docs").rglob("*.md")
    )


def _language(source: Path) -> str | None:
    relative = source.relative_to(ROOT).as_posix()
    if relative.startswith("docs/en/") or relative in {
        "README.md", "THIRD_PARTY_NOTICES.md",
    }:
        return "en"
    if relative.startswith("docs/zh/") or relative in {
        "README.zh-CN.md", "THIRD_PARTY_NOTICES.zh-CN.md",
    }:
        return "zh"
    return None


def _translation(source: Path) -> Path | None:
    relative = source.relative_to(ROOT).as_posix()
    pairs = {
        "README.md": "README.zh-CN.md",
        "README.zh-CN.md": "README.md",
        "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.zh-CN.md",
        "THIRD_PARTY_NOTICES.zh-CN.md": "THIRD_PARTY_NOTICES.md",
    }
    if relative in pairs:
        return ROOT / pairs[relative]
    for current, other in (("en", "zh"), ("zh", "en")):
        prefix = f"docs/{current}/"
        if relative.startswith(prefix):
            return ROOT / f"docs/{other}" / relative.removeprefix(prefix)
    return None


def main() -> int:
    findings: list[str] = []
    for source in _markdown_files():
        text = source.read_text(encoding="utf-8")
        language = _language(source)
        translation = _translation(source)
        linked_translation = False
        for raw_target in LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if not relative:
                continue
            destination = (source.parent / relative).resolve()
            if not destination.exists():
                findings.append(f"{source.relative_to(ROOT)}: missing {target}")
            if translation is not None and destination == translation:
                linked_translation = True
                continue
            if language and destination.is_relative_to(ROOT) and destination.suffix == ".md":
                target_language = _language(destination)
                if target_language and target_language != language:
                    findings.append(f"{source.relative_to(ROOT)}: cross-language link {target}")
                if destination.parent == ROOT / "docs":
                    findings.append(f"{source.relative_to(ROOT)}: use the {language} guide for {target}")
        if translation is not None:
            if not translation.is_file():
                findings.append(f"{source.relative_to(ROOT)}: missing translation")
            if not linked_translation:
                findings.append(f"{source.relative_to(ROOT)}: missing language switch")
    if findings:
        raise SystemExit("\n".join(findings))
    print(f"Local Markdown link check passed for {len(_markdown_files())} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
