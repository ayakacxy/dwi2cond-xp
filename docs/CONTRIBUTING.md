# Contributing

[English](en/CONTRIBUTING.md) | [简体中文](zh/CONTRIBUTING.md)

[Documentation](en/README.md) · [Project home](../README.md)

Contributions are welcome through issues and pull requests. Discuss substantial algorithm or API changes in an issue before implementing them.

## Development setup

Follow [Reproducibility](en/REPRODUCIBILITY.md) to prepare Python 3.11 and the required dependencies, then install the development tools:

```bash
python -m pip install -e '.[test]'
```

SimNIBS integration needs version 4.6.0. FSL is an optional local numerical reference.

## Making changes

Keep each pull request focused. Update the English and Chinese guides together, including the full compatibility copies under `docs/`. Public source comments, docstrings, CLI help and error messages use English.

For numerical changes, describe the affected processing stage and compare the same inputs, parameters and solver settings. Include coordinate conventions, tensor differences and the relevant QA results. Performance reports should state hardware, thread count, cache state and timing scope. See [Methods](en/METHODS.md) and [Benchmarks](en/BENCHMARKS.md).

Keep patient data, credentials, caches and generated simulation results out of commits. Retain attribution and license notices for adapted code.

## Checks

Use checks that match the change. For documentation:

```bash
python scripts/check_markdown_links.py
python scripts/verify_version.py
git diff --check
```

For code, run Ruff and the relevant tests. Changes to shared numerical kernels also require the complete check in `python scripts/run_coverage.py`. Hardware-specific changes need a run on that hardware; record any checks you could not perform.

In the pull request, explain the problem, the resulting behavior and the checks performed. Add user-visible changes to [the changelog](en/CHANGELOG.md).
