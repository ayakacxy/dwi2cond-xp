# Reproducibility

[English](../en/REPRODUCIBILITY.md) | [简体中文](../zh/REPRODUCIBILITY.md)

[Documentation](README.md) · [Project home](../../README.md)

## Clean environment

The primary full-pipeline installation may use an existing, verified SimNIBS
4.6.0 environment:

```bash
conda activate simnibs
python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl
python -c "import simnibs, dwi2cond_xp; assert simnibs.__version__ == '4.6.0'"
```

Use the v0.4.0 wheel from the matching GitHub Release.
For editable development in that frozen environment, use
`python -m pip install --no-deps -e .` instead of the wheel command. This keeps
the existing dependency versions; confirm they match the validated versions.

To create an independent environment:

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e '.[test,release,viz]'
python -m pip check
ruff check src tests tools scripts
pytest -q tests/test_tensor_fit.py
python scripts/check_markdown_links.py
python scripts/verify_version.py
python -m build
```

Use focused tests while developing. Before a release or a change that affects
shared numerical kernels, run the complete Numba-aware check:

```bash
python scripts/run_coverage.py
```

This command is intentionally more expensive than ordinary focused tests: it
runs unit tests and real synthetic TOPUP/EDDY/FNIRT end-to-end paths in isolated
Numba caches, then combines coverage and requires 100%.

Then install the wheel into a fresh Python 3.11 environment and run:

```bash
dwi2cond-xp --help
dwi2cond-xp fit-dti --help
dwi2cond-xp register-tensor --help
dwi2cond-xp simulate-tdcs --help
```

For release-asset verification, download the wheel, sdist, SBOM, and
`SHA256SUMS` from the same GitHub Release into a fresh directory. Run
`sha256sum -c SHA256SUMS`, install the downloaded wheel into a new environment,
and verify its import and CLI. The tag workflow additionally runs Twine,
wheel-content, archive-privacy, dependency, provenance, and SBOM checks.

## CPU and optional CUDA

CPU is the default for both FNIRT Hessian assembly and PPD and does not need
CuPy or a CUDA device. The two independent options are `--fnirt-device cuda`
and `--ppd-device cuda` on the nonlinear registration and pipeline commands.
The first uses Numba CUDA; the second also needs a compatible CuPy installation.
These GPU dependencies are not installed by the CPU wheel or environment file.

The tested GPU environment was Linux, RTX 2080 Ti, CUDA 12.8, Numba 0.64.0 and
CuPy 14.2.0. This is a tested runtime, not a clean-environment installation
recipe or a GPU certification for macOS/Windows. Explicit CUDA failures stop
the run; select `cpu` to run without GPU support. See [Methods](METHODS.md)
and [v0.4.0 release notes](RELEASE_NOTES_v0.4.0.md).

## Reference tools

FSL is never downloaded by CI. Optional FSL numerical tests require the caller
to configure a local FSL 6.0.4 installation and must report a visible skip when
it is absent. SimNIBS integration tests require exactly version 4.6.0.

## Real data

Real-data reproduction requires the user to obtain diffusion and anatomical
data under their original terms, select a supported `dwi2cond-xp` preprocessing
branch, and create a SimNIBS 4.6 CHARM head model. Inputs outside the fixed
SimNIBS 4.6 subset, such as wrapped phase requiring PRELUDE, must be prepared by
an appropriate external tool. Do not upload identifiable or controlled-access
medical images to an issue.

Record software versions, DWI shape and shell, masks, tensor ordering, world
transform, interpolation order, conductivity mode, solver, CPU count, output
hashes, and QA JSON for every scientific reproduction.
