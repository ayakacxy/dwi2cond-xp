"""Test deterministic changelog extraction for GitHub Releases."""

from __future__ import annotations

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
