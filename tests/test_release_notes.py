"""Test deterministic changelog extraction for GitHub Releases."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_release_notes import extract_release_notes


def test_extract_release_notes_selects_only_requested_version() -> None:
    changelog = """# Changelog

## [Unreleased]

- Future change.

## [0.2.0] - 2026-08-24

### Added

- Released feature.

## [0.1.0] - 2026-08-21

- Earlier feature.
"""

    notes = extract_release_notes(changelog, "v0.2.0")

    assert notes == (
        "## dwi2cond-xp v0.2.0\n\n"
        "### Added\n\n"
        "- Released feature.\n"
    )


def test_extract_release_notes_accepts_tag_without_v_prefix() -> None:
    changelog = "## [1.0.0]\n\nRelease body.\n"

    assert extract_release_notes(changelog, "1.0.0").endswith("Release body.\n")


def test_extract_release_notes_combines_actual_v040_languages() -> None:
    """实际版本的双语正文均存在，且不包含前一版的发布章节。"""

    root = Path(__file__).resolve().parents[1]
    english = (root / "docs/en/CHANGELOG.md").read_text(encoding="utf-8")
    chinese = (root / "docs/zh/CHANGELOG.md").read_text(encoding="utf-8")
    notes = extract_release_notes(english, "v0.4.0", chinese_changelog=chinese)

    assert notes.startswith("## dwi2cond-xp v0.4.0\n\n## English\n")
    assert "## 简体中文\n\n### 概述" in notes
    assert "### Fixed\n" in notes
    assert "### 修复\n" in notes
    assert "--fnirt-device cuda" in notes
    assert "## [0.3.0]" not in notes
    assert "correctness-remediated stable release" not in notes
    assert notes.count("## dwi2cond-xp v0.4.0") == 1


@pytest.mark.parametrize(
    ("changelog", "message"),
    (
        ("## [0.1.0]\n\nOld.\n", "No changelog section"),
        ("## [0.2.0]\n\n## [0.1.0]\n\nOld.\n", "is empty"),
    ),
)
def test_extract_release_notes_rejects_missing_or_empty_sections(
    changelog: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_release_notes(changelog, "v0.2.0")
