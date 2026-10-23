# Release process

[English](../en/RELEASE_PROCESS.md) | [简体中文](../zh/RELEASE_PROCESS.md)

[Documentation](README.md) · [Project home](../../README.md)

Release artifacts are published on GitHub under an annotated version tag. The title is the version itself, such as `v0.4.0`. The repository does not currently publish to PyPI.

## Prepare

1. Finalize the version's entries in [the changelog](CHANGELOG.md) and synchronize version metadata.
2. Update both languages and the full legacy-path document copies. Check examples, local links, figures and formulas.
3. Run checks appropriate to the changes. Numerical changes require the complete release check; documentation-only corrections require unchanged-source verification and package checks.
4. Build and inspect the distributions:

```bash
python scripts/verify_version.py
python scripts/check_markdown_links.py
python scripts/verify_release.py --repository .
python -m build --outdir .work/package/dist
python -m twine check .work/package/dist/*
check-wheel-contents .work/package/dist/*.whl
python scripts/build_release_metadata.py --dist .work/package/dist
python scripts/verify_release.py --dist .work/package/dist
```

For ordinary code releases, the final commit must pass the Linux, macOS, Windows, package and security checks. The full numerical check is `python scripts/run_coverage.py`. Environment and dependency-audit commands are described in [Reproducibility](REPRODUCIBILITY.md).

## Publish

Create the annotated tag from the verified commit. The release workflow builds the wheel and source archive, generates a CycloneDX SBOM and checksums, attests the distributions, and publishes the matching release notes. The changelog must contain a nonempty section for the tag.

Use a coherent release body in one language, with a link to the full translation. Keep the release title to the version number. Publishing to another registry requires a separate setup.

## Correct a historical tag

Published tags normally remain fixed. An authorized correction starts by recording the old tag object, commit, Release ID and attachment hashes, and preserving a backup ref outside `refs/tags/`.

For a documentation correction, keep computational source, dependencies and tests unchanged. Build packages containing the corrected documentation and verify them before replacing assets. Update only the intended tag using its old object ID with `--force-with-lease`. Retain the original attachments and notes for rollback.

A same-version algorithm replacement needs source review, numerical regressions and full release checks. Its release notes must describe the changed behavior. It is not a documentation correction.

## Verify the public release

Download the published wheel, source archive, SBOM and checksum file. Check that:

- the tag resolves to the intended commit and the release is public;
- downloaded SHA-256 values match `SHA256SUMS` and the upload;
- package source, README and documents match that commit;
- installation, import, version and CLI help work;
- available provenance refers to those distribution hashes and the intended workflow;
- the page title and translated documentation links are correct.

Record the commit, attachment hashes, checks and rollback location in the project ledger. A rebuilt package needs new provenance; an attestation for an older hash does not verify its replacement.
