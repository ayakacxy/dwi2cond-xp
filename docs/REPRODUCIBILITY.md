# Reproducibility

[English](en/REPRODUCIBILITY.md) | [简体中文](zh/REPRODUCIBILITY.md)

[Documentation](en/README.md) · [Project home](../README.md)

## Clean environment

Use Python 3.11. The numerical stack pins NumPy 2.3.0 and Numba 0.64.0. SimNIBS-facing workflows require SimNIBS 4.6.0; the tensor-processing core can run without it.

In an existing SimNIBS environment, download the wheel from the [matching release](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.4.0) and install it without replacing the environment's dependencies:

```bash
conda activate simnibs
python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl
python -c "import simnibs, dwi2cond_xp; assert simnibs.__version__ == '4.6.0'"
python -m pip check
```

`--no-deps` assumes the required versions are already present. For a new full-workflow environment, use the repository's environment file:

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
dwi2cond-xp --help
```

For the core alone, create a Python 3.11 virtual environment and install the base package:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then run `python -m pip install .`.

### Development and builds

Install `.[test,release,viz]` for the development and release tools. During development, run Ruff and tests for the affected modules. Documentation changes need link, version and example checks. The complete numerical check is `python scripts/run_coverage.py`; it runs the unit suite plus synthetic TOPUP/EDDY/FNIRT CLI paths and enforces 100% statement coverage.

Build a wheel and source archive with:

```bash
python -m build --outdir .work/package/dist
python -m twine check .work/package/dist/*
python scripts/build_release_metadata.py --dist .work/package/dist
python scripts/verify_release.py --dist .work/package/dist
```

Use a project-local temporary directory and caches for build or benchmark work. On Linux/macOS:

```bash
mkdir -p .work/package/tmp .work/package/cache
export TMPDIR="$PWD/.work/package/tmp" TMP="$PWD/.work/package/tmp" TEMP="$PWD/.work/package/tmp"
export XDG_CACHE_HOME="$PWD/.work/package/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
```

On Windows PowerShell:

```powershell
$workDir = Join-Path (Get-Location) ".work/package"
New-Item -ItemType Directory -Force "$workDir/tmp", "$workDir/cache" | Out-Null
$env:TMPDIR = $env:TMP = $env:TEMP = "$workDir/tmp"
$env:XDG_CACHE_HOME = "$workDir/cache"
$env:PIP_CACHE_DIR = "$workDir/cache/pip"
$env:NUMBA_CACHE_DIR = "$workDir/cache/numba"
$env:MPLCONFIGDIR = "$workDir/cache/matplotlib"
```

Downloaded releases contain a wheel, source archive, CycloneDX SBOM and `SHA256SUMS`. Verify the checksums, install the wheel in a separate environment, and check import and CLI help. See [Release process](en/RELEASE_PROCESS.md) for the publishing steps.

## CPU and optional CUDA

CPU is the default for FNIRT Hessian assembly and PPD. It needs neither CuPy nor a CUDA device. `--fnirt-device cuda` requires Numba CUDA, a working NVIDIA driver and compatible CUDA toolkit. `--ppd-device cuda` additionally requires compatible CuPy. These dependencies are not installed by the base package.

The measured CUDA environment was Linux, RTX 2080 Ti, CUDA 12.8, Numba 0.64.0 and CuPy 14.2.0. These details identify the tested environment. CPU platform checks and GPU measurements are reported separately in [Validation](en/VALIDATION.md) and [Benchmarks](en/BENCHMARKS.md).

## Reference tools

Optional numerical comparisons use a locally installed FSL 6.0.4 and SimNIBS 4.6.0. FSL is not downloaded by CI or used by the runtime. Reference tests report a skip when their external tool is unavailable. The source hashes and comparison artifacts are described in [FSL reference](en/FSL_REFERENCE.md).

## Real data

Obtain diffusion and anatomical data under their original terms and create the CHARM head model required by the selected workflow. Record the input hashes, shape, shell selection, masks, gradients, coordinate options, solver, thread count, software versions, command and QA outputs.

For a timing comparison, use identical inputs and state whether process startup, JIT compilation, caches, QA, publication and FEM are included. Report repeated runs separately from first calls. Keep data and outputs from different configurations in separate directories. See [Benchmarks](en/BENCHMARKS.md) for the recorded measurements.
