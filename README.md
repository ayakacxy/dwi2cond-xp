# dwi2cond-xp

Diffusion MRI preprocessing, tensor fitting, and conductivity modelling for SimNIBS 4.6.

[![Release](https://img.shields.io/github/v/release/ayakacxy/dwi2cond-xp?display_name=tag&sort=semver)](https://github.com/ayakacxy/dwi2cond-xp/releases/latest)
[![CI](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/ayakacxy/dwi2cond-xp/badge)](https://scorecard.dev/viewer/?uri=github.com/ayakacxy/dwi2cond-xp)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](docs/en/VALIDATION.md)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF.svg)](docs/en/SIMNIBS_INTEGRATION.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[English](README.md) | [简体中文](README.zh-CN.md)

[Documentation](docs/en/README.md) · [Input files](docs/en/INPUTS.md) · [Validation](docs/en/VALIDATION.md) · [Changelog](docs/en/CHANGELOG.md)

`dwi2cond-xp` processes supported raw or preprocessed single-shell diffusion MRI,
maps diffusion tensors to a CHARM T1 grid, and runs SimNIBS conductivity and FEM
workflows. The preprocessing and tensor code is Python and does not require
FSL at runtime. FEM and lead-field commands use SimNIBS 4.6.0.

This is an independent community project. The stable release is `v0.3.0`;
the project is currently in [maintenance](docs/en/ROADMAP.md).

## Supported workflows

- Single-shell DTI fitting, with optional HCP/FSL gradient-nonlinearity correction.
- The fixed `nomoco`, legacy, GRE, TOPUP, and EDDY preprocessing used by SimNIBS 4.6.
- Affine or nonlinear DTI-to-T1 registration and tensor reorientation.
- `scalar`, `vn`, `dir`, and `mc` conductivity, fixed-montage FEM, and lead-field interfaces.

The implementation covers the settings used by SimNIBS `dwi2cond`. Arbitrary
FSL configurations, wrapped phase requiring PRELUDE, and direct single-tensor
fitting of multiple nonzero shells are outside this scope. SimNIBS-dependent
commands require a platform where SimNIBS 4.6.0 is available and validated.
See [input files and coordinates](docs/en/INPUTS.md) for the requirements of each command.

## Installation

Use Python 3.11. In an existing, compatible SimNIBS 4.6 environment, download
the wheel from [Releases](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.3.0)
and install it without changing that environment's dependencies:

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.3.0-py3-none-any.whl
dwi2cond-xp --help
```

The validated nonlinear dependency versions are NumPy 2.3.0 and Numba 0.64.0.
For a separate environment or source installation, see
[installation and reproducibility](docs/en/REPRODUCIBILITY.md).

The Python core can also be installed without SimNIBS:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

In Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
FEM and lead-field execution still require SimNIBS.

## Quick start

Given single-shell DWI, matching b-values and b-vectors, and a completed CHARM
head model, the default pipeline runs legacy preprocessing and nonlinear T1 registration:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

The final tensor is written to `m2m_subject/DTI_coregT1_tensor.nii.gz`.
Reverse-phase-encoding data and pre-fitted tensors have dedicated examples in
[the command guide](docs/en/USAGE.md).

Tensor NIfTIs use `(X,Y,Z,6)` with component order `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`.
Data from the same participant are not necessarily aligned; use automatic
registration or a known world-coordinate transform. Use `--assume-aligned`
only after alignment has been established externally.

Check a C3-to-C4 1 mA simulation without starting the solver:

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 --cpus 8 --dry-run
```

Remove `--dry-run` to solve. Each mode has a separate output directory.
Vector field NIfTIs use `Ex,Ey,Ez` and retain only WM/GM/CSF labels 1/2/3.
See [SimNIBS integration](docs/en/SIMNIBS_INTEGRATION.md) for FEM and lead-field outputs.

## Conductivity modes

| Mode | Use |
| --- | --- |
| `scalar` | A fixed conductivity per tissue, independent of DTI. |
| `vn` | Preserves direction and relative anisotropy, with local determinant normalization to the tissue reference. |
| `dir` | Preserves direction, anisotropy, and spatial magnitude, with one global intensity scale across selected tissues. |
| `mc` | Makes the `dir` result locally isotropic while preserving its determinant and spatial variation. |

The [conductivity guide](docs/en/CONDUCTIVITY.md) gives the equations, references,
safety limits, singular-tensor policy, and stable/literal eigensystem options.

## Validation and performance

The final algorithm release completed **666 tests with no skips** when all real
FSL probes were configured. A separate run covered **13,663/13,663 statements**,
including 666 main tests, 12 montage tests, and synthetic TOPUP/EDDY/FNIRT CLI
workflows. The last FEM changes were checked with real SimNIBS
`SESSION._prepare()`; the solver and full HCP raw-DWI-to-FEM chain were not
rerun after those changes.

Earlier work included a private HCP DTI comparison with FSL (tensor relative
L2 `4.18e-6`), four conductivity-mode FEM runs, and stage benchmarks. The
reported `11.09x` DTI-fitting ratio was measured on one server with 16 Python
workers: `9.76 s` versus `108.23 s` for FSL. It does not describe the speed of
the complete workflow or the current tag. The lead-field interface is unit
tested; a full-subject all-electrode run has not been completed.

See [validation](docs/en/VALIDATION.md) and [historical benchmarks](docs/en/BENCHMARKS.md)
for numerical results, input sizes, timing details, and limitations.

## Example fields

The component figure uses three rows (`Ex`, `Ey`, `Ez`) and four columns
(`scalar`, `vn`, `dir`, `mc`) with a shared symmetric scale. The slice is chosen
by maximum brain-mask area. The magnitude view uses the same fields and a shared positive scale.

![Four-mode field components](docs/images/electric_field_xyz_3x4.png)

![Four-mode field magnitude](docs/images/electric_field_magnitude_2x2.png)

These identifier-free images illustrate earlier HCP results. They are not a
new run of the final tag. No source MRI or volumetric subject derivatives are
distributed. See the [HCP acknowledgment and data-use terms](THIRD_PARTY_NOTICES.md#human-connectome-project-data).

## Contributing and citation

[Contributing](docs/en/CONTRIBUTING.md) · [Support](docs/en/SUPPORT.md) · [Security](docs/en/SECURITY.md) · [Code of conduct](docs/en/CODE_OF_CONDUCT.md)

Cite the project using [CITATION.cff](CITATION.cff), together with SimNIBS and
the diffusion-processing software used in your work. The license is
[GPL-3.0-only](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md).

The validation data were provided in part by the WU-Minn Human Connectome Project,
led by David Van Essen and Kamil Ugurbil, supported by NIH Blueprint funding and
the McDonnell Center for Systems Neuroscience at Washington University.
