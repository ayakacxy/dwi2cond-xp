<div align="center">

# dwi2cond-xp

Cross-platform, FSL-free DTI-to-conductivity workflows for SimNIBS 4.6.

[![Release: v0.1.0](https://img.shields.io/badge/release-v0.1.0-blue.svg)](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.1.0)
[![CI](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/ayakacxy/dwi2cond-xp/badge)](https://scorecard.dev/viewer/?uri=github.com/ayakacxy/dwi2cond-xp)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#-validation-at-a-glance)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF.svg)](docs/SIMNIBS_INTEGRATION.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Validation](docs/VALIDATION.md) · [Benchmarks](docs/BENCHMARKS.md) · [Release notes](docs/CHANGELOG.md)

🧠 **DTI fitting** · ⚡ **FSL-free runtime** · 🧭 **Tensor reorientation** · ⚡ **Anisotropic FEM** · 🧪 **100% statement coverage**

</div>

`dwi2cond-xp` is a cross-platform Python pipeline that converts preprocessed
diffusion MRI into conductivity tensors for SimNIBS 4.6 and orchestrates
anisotropic finite-element simulations without requiring FSL at runtime.

This is an independent community project. It is not an official SimNIBS or FSL
distribution.

> **Historical release.** This tree documents `v0.1.0` as released on
> 2026-08-21. The tag was reissued on 2026-08-29 only to correct documentation
> and release metadata; its algorithm source remains the original `v0.1.0`
> baseline. Claims below apply to this version and its stated evidence boundary.

## ⚡ Validation at a glance

| Contract | Result | Evidence boundary |
| --- | ---: | --- |
| Python test suite | **144 passed · 100.00%** | `v0.1.0` release record: 1,644/1,644 package statements, including a configured local FSL reference test |
| DTI tensor comparison | **relative L2 4.18e-6** | Historical private HCP b0+b1000 input, same WLS/gradient-nonlinearity contract versus FSL 6.0.4 |
| DTI fitting wall time | **9.76 s vs 108.23 s · 11.09x** | Historical single-server, single-input fitting/output boundary; not an end-to-end claim |
| Conductivity comparison | **max abs 0 to 2.22e-16** | Historical synthetic sphere mesh versus SimNIBS 4.6 for `vn`, `dir`, and `mc` |
| Fixed-montage FEM | **4/4 modes completed** | Historical private-subject `scalar`, `vn`, `dir`, `mc` C3→C4 runs with Pardiso |

No anatomical image, subject identifier, voxel derivative, or machine-readable
subject artifact is distributed. Full methods and evidence boundaries are in
[Validation](docs/VALIDATION.md) and [Benchmarks](docs/BENCHMARKS.md).

## 🧭 Scope

The supported post-preprocessing path is:

```text
preprocessed single-shell DWI or a six-component diffusion tensor
  -> weighted least-squares DTI fitting and QA
  -> explicit tensor mapping/reorientation to the CHARM T1 grid
  -> scalar / vn / dir / mc conductivity
  -> fixed-montage SimNIBS 4.6 FEM
  -> vector E-field NIfTI and QA manifests
```

The project deliberately does **not** perform raw-DWI motion correction,
eddy-current correction, susceptibility/topup/fieldmap correction, or automatic
6/12-DOF and nonlinear DTI-to-T1 registration. Those steps must be completed by
an external preprocessing workflow, including consistent b-vector rotation.
Nonlinear PPD tensor reorientation is not implemented.

The pure-Python DTI and tensor-mapping core uses NumPy, SciPy, NiBabel, h5py,
and tqdm. Mesh conductivity, FEM, and lead-field workflows require exactly
SimNIBS 4.6.0. Platform support for those workflows is limited to platforms on
which SimNIBS 4.6.0 itself can be installed and validated.

## 🧩 Conductivity modes

| Mode | Meaning |
| --- | --- |
| `scalar` | Fixed scalar conductivity per tissue; does not use DTI. |
| `vn` | Preserves tensor direction and anisotropy ratio, then locally normalizes the determinant to the tissue reference conductivity, subject to safety bounds. |
| `dir` | Preserves direction, anisotropy ratio, and spatial magnitude variation; its default calibration uses one global scale across all selected anisotropic tissues. |
| `mc` | Uses the same global scale as `dir`, then replaces each tensor by an isotropic tensor with the same local determinant; it is an intensity-variation control. |

Exact zero tensors in selected anisotropic tissues are replaced by the tissue's
scalar conductivity tensor before conversion and are accepted. The default
`dir`/`mc` calibration requires positive, finite aggregate determinants.
`--no-correct-intensity` bypasses that global calibration and uses the
uncalibrated per-tensor safety path. `v0.1.0` has no public
`strict-fsl`/`robust` fitting-mode switch. See [Methods](docs/METHODS.md) for the
version-specific equations and degenerate-input boundary.

## 🐍 Installation

The complete validated contract is Python 3.11 with SimNIBS 4.6.0. Installing
into an existing SimNIBS 4.6 environment is the primary path. If that environment
is intentionally frozen, install the wheel without dependency resolution:

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.1.0-py3-none-any.whl
dwi2cond-xp --help
```

For development from this checkout, replace the wheel command with
`python -m pip install --no-deps -e .`. The release wheel has been installed as
a temporary overlay on the existing reference environment and imported together
with SimNIBS 4.6.0 without modifying that environment.

To create a separate environment instead:

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
dwi2cond-xp --help
```

For the pure-Python core only:

```bash
python -m venv .venv
. .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
dwi2cond-xp --help
```

Developer installation in a non-frozen environment:

```bash
python -m pip install -e '.[test]'
pytest -q --cov=dwi2cond_xp --cov-report=term-missing --cov-fail-under=100
ruff check src tests tools scripts
```

## 📥 Input contract

A DWI input must be a preprocessed 4-D NIfTI with matching b-values,
b-vectors, and a diffusion brain mask. Use a single nonzero shell plus b=0
volumes for the DTI model. Optional gradient-nonlinearity coefficients use the
FSL/HCP nine-component `grad_dev` convention.

A tensor input must be a 4-D NIfTI whose final dimension is ordered as
`Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`. Before mapping to T1, the caller must choose exactly
one of the following:

- pass `--world-transform` with an externally estimated 4x4 input-world to
  reference-world affine; or
- pass `--assume-aligned` only when external preprocessing has already aligned
  the diffusion and T1 world coordinates.

The final SimNIBS tensor is stored at
`m2m_<subject>/DTI_coregT1_tensor.nii.gz`, matching the SimNIBS 4.6 contract.
See [Input contract](docs/INPUT_CONTRACT.md) for coordinate, mask, and failure
semantics.

## 🚀 Minimal workflow

Fit a preprocessed single-shell DWI:

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz \
  --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz \
  --qa-json tensor_fit_qa.json
```

Map the tensor to the CHARM T1 grid. The example assumes alignment only because
the alignment was established externally:

```bash
dwi2cond-xp register-tensor \
  tensor_dwi.nii.gz m2m_subject/T1.nii.gz \
  m2m_subject/DTI_coregT1_tensor.nii.gz \
  --source-mask tensor_valid_mask.nii.gz \
  --assume-aligned \
  --qa-json tensor_registration_qa.json
```

Validate a C3-to-C4 anisotropic simulation without starting the solver:

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn \
  --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 \
  --cpus 8 --dry-run
```

Remove `--dry-run` to execute SimNIBS. Run `scalar`, `vn`, `dir`, and `mc`
separately; each mode has its own output directory and manifest. An anisotropic
mode never silently falls back to `scalar` when its tensor is missing.

Each formal simulation exports one T1-grid vector E-field NIfTI with final axis
`Ex,Ey,Ez`. Magnitude is computed as `sqrt(Ex^2 + Ey^2 + Ez^2)`. Subject-volume
outputs are strictly masked to WM, GM, and CSF labels `1,2,3`; skull, scalp,
electrodes, and extracranial tissues are excluded.

## 🎨 Voxel-level comparison

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

The component figure uses three rows (`Ex`, `Ey`, `Ez`) and four columns
(`scalar`, `vn`, `dir`, `mc`) with one shared symmetric color scale. Slice
selection is based only on maximum brain-mask area, not field intensity.

![Four-mode electric-field components](docs/images/electric_field_xyz_3x4.png)

The magnitude view uses the same four vector NIfTIs and a shared positive scale:

![Four-mode electric-field magnitude](docs/images/electric_field_magnitude_2x2.png)

## ⚡ Lead-field support

`simulate-leadfield` builds SimNIBS 4.6 `TDCSLEADFIELD` configurations for all
four conductivity modes. It keeps the original HDF5 provenance and can export a
NumPy matrix shaped `(N_spatial * 3, N_electrode - 1)` plus JSON metadata. The
first cap electrode is the reference; each remaining column is a 1 A basis.
Pardiso is the default and failures are not silently rerouted to another solver.

This interface and its HDF5/NPY contracts are unit tested, but the `v0.1.0`
evidence does not include a full-subject, all-electrode lead-field run.
See [SimNIBS integration](docs/SIMNIBS_INTEGRATION.md).

## 🧪 Historical validation evidence

One private HCP subject was used for `v0.1.0` release-candidate validation. No source
image, volumetric derivative, subject identifier, or machine-readable
subject-level artifact is distributed. The two rendered field-comparison PNGs
are included as result illustrations without a subject identifier. They must
retain the HCP acknowledgment and are not a substitute for accepting the
[WU-Minn HCP Open Access Data Use Terms](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).

- The recorded b0+b1000 DTI outputs completed for 881,299 masked voxels; 881,194 were
  valid and 105 invalid voxels were explicitly zeroed and recorded.
- Against FSL 6.0.4 WLS with gradient nonlinearity, the HCP tensor comparison
  had relative L2 error `4.18e-6`; mean and p99 absolute differences were
  `4.78e-10` and `2.10e-9`.
- On the same server and output boundary, the 16-worker Python fit took 9.76 s
  versus 108.23 s for FSL 6.0.4 (`11.09x`). This is a single-system DTI-fitting
  result, not an end-to-end FEM speed claim.
- Synthetic-mesh conductivity agreed with SimNIBS 4.6 to max absolute error
  `0` for `vn` and `2.22e-16` for `dir`/`mc`.
- Real `scalar`, `vn`, `dir`, and `mc` C3-to-C4 FEM runs completed with Pardiso;
  all vector E-field NIfTIs were finite and strictly excluded tissues outside
  WM/GM/CSF.
- The recorded local release test suite completed with `144 passed` and strict
  `100.00%` statement coverage. Cross-platform CI enforces the same 100%
  threshold; the FSL comparison is skipped only where `dtifit` is unavailable.

Exact methods, timing boundaries, and limitations are in
[Validation](docs/VALIDATION.md), [Benchmarks](docs/BENCHMARKS.md), and
[Reproducibility](docs/REPRODUCIBILITY.md). The private HCP, sphere, and FEM
experiments were not rerun for the 2026-08-29 documentation refresh.

## 🗂️ Documentation and community

- [Documentation index](docs/README.md)
- [Methods](docs/METHODS.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Security and medical-data reporting](docs/SECURITY.md)
- [Support](docs/SUPPORT.md)
- [Code of conduct](docs/CODE_OF_CONDUCT.md)
- [Changelog](docs/CHANGELOG.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## 🙏 Citation and license

Please cite this software using [CITATION.cff](CITATION.cff), and cite SimNIBS
and the diffusion-processing software used upstream. This project is released
under [GPL-3.0-only](LICENSE). Third-party packages and optional reference tools
remain under their own licenses.

The validation data were provided in part by the WU-Minn Human Connectome
Project, led by David Van Essen and Kamil Ugurbil and supported by NIH Blueprint
funding and the McDonnell Center for Systems Neuroscience at Washington
University. See the official [HCP citation guidance](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations).
