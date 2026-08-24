# Release process

Stable versions are immutable annotated tags and GitHub Releases. The `main`
branch remains the current development line; an older release receives a
temporary backport branch only when maintenance is required.

## Prepare

1. Move user-visible entries from `Unreleased` in [the changelog](CHANGELOG.md)
   into a dated version section.
2. Synchronize the version in `pyproject.toml`, `CITATION.cff`, package metadata,
   README examples, and documentation.
3. Install the release, test, and visualization extras in a clean Python 3.11
   environment. Run focused tests during development, then run the release
   gates:

   ```bash
   python -m pip install -e '.[test,release,viz]'
   ruff check src tests tools scripts
   python scripts/run_coverage.py
   python scripts/verify_version.py
   python scripts/check_markdown_links.py
   python scripts/verify_release.py --repository .
   validate-pyproject pyproject.toml
   check-manifest --verbose
   python -m pip_audit . --skip-editable --progress-spinner=off
   python -m build
   python -m twine check dist/*
   check-wheel-contents dist/*.whl
   python scripts/build_release_metadata.py --dist dist
   python scripts/verify_release.py --dist dist
   ```

4. Confirm that no human MRI, subject derivative, local path, credential, FSL
   binary, build output, or internal ledger is tracked or packaged.
5. Require the final `main` commit to pass Linux, macOS arm64, Windows, package,
   CodeQL, and OpenSSF workflows before tagging.

## Publish

Create and push an annotated `vX.Y.Z` tag that resolves to the verified `main`
commit. The tag workflow reruns the complete test and package gates, extracts
the matching version section from the changelog, builds the wheel and sdist,
creates a CycloneDX SBOM and `SHA256SUMS`, attests provenance and the SBOM, and
publishes the GitHub Release. A tag without a matching nonempty changelog section
fails before publication.

The workflow does not publish to PyPI. Adding another registry is a separate
release change that requires its own trusted-publishing and installation audit.

## Verify the public release

Download every asset from the public Release into a fresh directory, then:

- confirm that the tag dereferences to the intended commit and the Release is
  neither draft nor prerelease;
- run `sha256sum -c SHA256SUMS`;
- run Twine, wheel-content, and archive-privacy checks on the downloaded files;
- install the downloaded wheel into a clean environment and verify import,
  package version, CLI help, and dependency consistency;
- verify GitHub provenance for the wheel and sdist;
- inspect the rendered release notes and documentation links.

Record the workflow run, tag commit, asset hashes, and any intentionally skipped
external integration in the project evidence ledger.
