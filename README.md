<div align="center">

# dwi2cond-xp

Cross-platform, FSL-free DTI-to-conductivity workflows for SimNIBS 4.6.

[![Release](https://img.shields.io/github/v/release/ayakacxy/dwi2cond-xp?display_name=tag&sort=semver)](https://github.com/ayakacxy/dwi2cond-xp/releases/latest)
[![CI](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/ayakacxy/dwi2cond-xp/badge)](https://scorecard.dev/viewer/?uri=github.com/ayakacxy/dwi2cond-xp)
[![v0.4.0 coverage: 100%](https://img.shields.io/badge/v0.4.0_coverage-100%25-brightgreen.svg)](#-validation-at-a-glance)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF.svg)](docs/en/SIMNIBS_INTEGRATION.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md) · [Documentation](docs/en/README.md) · [Validation](docs/en/VALIDATION.md) · [Roadmap](docs/en/ROADMAP.md) · [Release notes](docs/en/CHANGELOG.md)

🧠 **DTI fitting** · ⚡ **FSL-free runtime** · 🧭 **Tensor reorientation** · ⚡ **Anisotropic FEM** · 🖥️ **CPU / optional CUDA**

</div>

## 📚 Contents

- [Validation at a glance](#-validation-at-a-glance)
- [Scope](#-scope)
- [Conductivity modes](#-conductivity-modes)
- [Installation](#-installation)
- [Input files and coordinates](#-input-files-and-coordinates)
- [Minimal workflow](#-minimal-workflow)
- [Voxel-level comparison](#-voxel-level-comparison)
- [Lead-field support](#-lead-field-support)
- [Validation evidence](#-validation-evidence)
- [Roadmap](#-roadmap)
- [Documentation and community](#-documentation-and-community)
- [Citation and license](#-citation-and-license)

`dwi2cond-xp` converts single-shell diffusion MRI into conductivity tensors for SimNIBS 4.6 and provides anisotropic FEM and lead-field interfaces. It accepts raw DWI, preprocessed DWI or fitted tensors, without requiring FSL at runtime.

CPU is the default. Optional CUDA accelerates nonlinear-registration Hessian assembly and PPD. Installation and the main workflow are shown below; the [usage guide](docs/en/USAGE.md) covers all commands.

## ⚡ Validation at a glance

Measured complete runs on Linux with a Xeon Gold 6230, eight CPU cores and an RTX 2080 Ti:

| Backend | Wall time | Peak host RSS | Peak GPU allocation |
| --- | ---: | ---: | ---: |
| CPU | **205.45 s** | **6.50 GiB** | — |
| CUDA | **163.34 s** | **6.72 GiB** | **8.83 GiB** |

The input is one HCP single-shell dataset with gradient-deviation correction, `nomoco` preprocessing and nonlinear T1 registration. Timing includes startup, computation, QA and m2m tensor publication, with warm file/JIT caches and FEM disabled. Each row is one run. The default legacy preprocessing path was not timed here. Full settings, FSL comparisons and numerical results are in [Benchmarks](docs/en/BENCHMARKS.md).

The release validation completed **960 tests with no skips** and **100% statement coverage** across 15,364 statements. Cross-platform CI covers Linux, macOS and Windows. Test conditions and optional reference checks are listed in [Validation](docs/en/VALIDATION.md).

## 🧭 Scope

```text
raw or preprocessed single-shell DWI, or a fitted diffusion tensor
  -> preprocessing and two-pass WLS fitting
  -> T1 registration and tensor reorientation
  -> QA and publication to the CHARM m2m directory
  -> scalar / vn / dir / mc conductivity
  -> SimNIBS FEM or lead field
```

Supported preprocessing includes `nomoco`, two-pass legacy correction, fixed GRE fieldmap correction, reverse-PE TOPUP and single-shell EDDY. T1 alignment supports affine registration and the fixed FNIRT/PPD nonlinear path.

The tensor core uses Python, NumPy, SciPy, NiBabel, Numba and h5py. Mesh conductivity, FEM and lead fields require SimNIBS 4.6.0. FSL is used only for optional numerical comparisons. This is an independent community project.

## 🧩 Conductivity modes

| Mode | Meaning |
| --- | --- |
| `scalar` | Fixed scalar conductivity per tissue; does not use DTI. |
| `vn` | Preserves tensor direction and anisotropy ratio, then normalizes the determinant locally to the tissue reference conductivity. This is the primary anisotropic mode. |
| `dir` | Preserves direction, anisotropy ratio, and spatial intensity variation, followed by global intensity calibration. |
| `mc` | DTI-driven spatially varying mean conductivity made locally isotropic; a control for intensity variation, not directional anisotropy. |

### Implemented equations

The equations below are the mappings implemented by this project and matched
against SimNIBS 4.6. For element or voxel $i$ in anisotropic tissue $t$, write
the repaired diffusion tensor as

$$
\mathbf D_i = \mathbf V_i\mathrm{diag}
(d_{i1},d_{i2},d_{i3})\mathbf V_i^{\mathsf T},
\qquad d_{i1}\ge d_{i2}\ge d_{i3}>0,
$$

where $\mathbf V_i$ contains the principal directions, $\sigma_t$ is the
reference scalar conductivity of tissue $t$, and $w_i$ is the tetrahedron
volume (or one for an unweighted voxel calculation).

#### `vn`: volume-normalized anisotropic mapping

Define the local geometric mean $g_i=(d_{i1}d_{i2}d_{i3})^{1/3}$. The core
mapping is

$$
\boldsymbol\Sigma_i^{\mathrm{vn}}
=\sigma_t\mathbf V_i
\mathrm{diag}\left(
\frac{d_{i1}}{g_i},\frac{d_{i2}}{g_i},\frac{d_{i3}}{g_i}
\right)\mathbf V_i^{\mathsf T},
\qquad
\det\left(\boldsymbol\Sigma_i^{\mathrm{vn}}\right)^{1/3}=\sigma_t.
$$

Thus `vn` preserves eigenvectors and relative anisotropy while setting the
geometric mean conductivity locally to the tissue reference. This is the
volume-normalized construction described by Güllmar et al. and used as the
recommended anisotropic mapping in SimNIBS [3,4].

#### `dir`: directly scaled anisotropic mapping

For every anisotropic tissue, first compute the volume-weighted tensor scale

$$
m_t=
\left(
\frac{\sum_{i\in t}w_i\det(\mathbf D_i)}
     {\sum_{i\in t}w_i}
\right)^{1/3}.
$$

One global factor is then fitted jointly across the anisotropic tissues:

$$
s=\underset{a}{\mathrm{argmin}}
\sum_t(am_t-\sigma_t)^2
=\frac{\sum_t\sigma_t m_t}{\sum_t m_t^2},
\qquad
\boldsymbol\Sigma_i^{\mathrm{dir}}=s\mathbf D_i.
$$

This retains DTI-driven direction, anisotropy, and spatial magnitude. It is the
linear diffusion-to-conductivity family introduced by Tuch et al. and used in
the direct mappings described by Rullmann et al. and Opitz et al. [1,2,4].

#### `mc`: mean-conductivity control

`mc` uses the same fitted factor $s$ as `dir`, but replaces every local tensor
by an isotropic tensor with the same determinant:

$$
\boldsymbol\Sigma_i^{\mathrm{mc}}
=\det\left(\boldsymbol\Sigma_i^{\mathrm{dir}}\right)^{1/3}\mathbf I
=s\det(\mathbf D_i)^{1/3}\mathbf I.
$$

It therefore preserves the DTI-driven spatial variation in geometric-mean
conductivity while removing directional anisotropy. `mc` is a DTI-derived
control mode, not an anisotropic tensor field [4].

All three mappings apply the documented SimNIBS-compatible eigenvalue, maximum
conductivity, and anisotropy-ratio rules for their supported inputs. Degenerate
inputs follow mode-specific failure policies rather than one universal repair:
nonzero rank-deficient `vn` tensors fail by default. With the default
`correct_intensity=True`, an all-zero anisotropic `dir`/`mc` tissue is rejected
because its global scale is undefined; explicit `--no-correct-intensity` uses
the non-calibrated safety path and accepts that input. `vn` performs
normalization, safety correction, renormalization, and a second safety
correction; a bound-triggered final correction can therefore slightly perturb
the ideal determinant equality above. Tissues not selected for anisotropy use
$\boldsymbol\Sigma_i=\sigma_t\mathbf I$.

A nonzero rank-deficient tensor has no defined VN determinant normalization.
The default policy therefore raises an explicit error instead of emitting NaN
or silently substituting a scale. `tensor-to-mesh --vn-singular-policy
regularize` is an opt-in numerical-stability extension: it projects the
singular eigensystem to the configured anisotropy bound before the existing
SimNIBS safety passes and records the repaired-element count in QA. This
extension is explicit and is not presented as the literal SimNIBS 4.6
rank-deficient branch.

The converter, formal FEM, and lead-field workflows use
`--eigensystem-mode stable` by default. This
selects the symmetric `eigh` solver and keeps repeated-eigenvalue tensors on an
orthogonal, full-rank basis. `--eigensystem-mode simnibs46-literal` is an
explicit compatibility mode that reproduces SimNIBS 4.6's general `eig`
ordering, including its non-orthogonal repeated-eigenvalue edge output; the QA
JSON records the selected mode and basis-orthogonality diagnostics. Formal FEM
workflow runs use the installed SimNIBS SESSION with a project-local
conductivity hook on the selected instance; installed modules are unchanged.
The default `--basis-mode physical` normalizes spatial affine columns
separately, supporting oblique unequal-sized voxels while rejecting shear.
Explicit `--basis-mode simnibs46-literal` retains the upstream normalization
for comparison.

References:

1. Tuch DS, Wedeen VJ, Dale AM, George JS, Belliveau JW. *Conductivity tensor
   mapping of the human brain using diffusion tensor MRI*. PNAS. 2001;
   98(20):11697-11701. [doi:10.1073/pnas.171473898](https://doi.org/10.1073/pnas.171473898)
2. Rullmann M, Anwander A, Dannhauer M, Warfield SK, Duffy FH, Wolters CH.
   *EEG source analysis of epileptiform activity using a 1 mm anisotropic
   hexahedra finite element head model*. NeuroImage. 2009;44(2):399-410.
   [doi:10.1016/j.neuroimage.2008.09.009](https://doi.org/10.1016/j.neuroimage.2008.09.009)
3. Güllmar D, Haueisen J, Reichenbach JR. *Influence of anisotropic electrical
   conductivity in white matter tissue on the EEG/MEG forward and inverse
   solution. A high-resolution whole head simulation study*. NeuroImage.
   2010;51(1):145-163.
   [doi:10.1016/j.neuroimage.2010.02.014](https://doi.org/10.1016/j.neuroimage.2010.02.014)
4. Opitz A, Windhoff M, Heidemann RM, Turner R, Thielscher A. *How the brain
   tissue shapes the electric field induced by transcranial magnetic
   stimulation*. NeuroImage. 2011;58(3):849-859.
   [doi:10.1016/j.neuroimage.2011.06.069](https://doi.org/10.1016/j.neuroimage.2011.06.069)

The corresponding SimNIBS implementation-level definitions are summarized in
the official [dwi2cond documentation](https://simnibs.github.io/simnibs/build/html/documentation/command_line/dwi2cond.html).

## 🐍 Installation

Use Python 3.11. In an existing SimNIBS 4.6.0 environment, download the wheel from the [Release page](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.4.0) and install it while retaining that environment's dependencies:

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl
dwi2cond-xp --help
```

For a separate full-workflow environment:

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
```

For the tensor-processing core, install the base package in a Python 3.11 environment with `python -m pip install .`. SimNIBS is needed only for its integration workflows. CUDA dependencies are optional. Environment setup, GPU requirements and development installation are described in [Reproducibility](docs/en/REPRODUCIBILITY.md).

## 📥 Input files and coordinates

- DWI, b-values and b-vectors must describe the same ordered volumes. The tensor model uses b=0 and one nonzero shell; select a shell explicitly for multishell data.
- Masks and optional HCP/FSL nine-component `grad_dev` must match their image grids.
- A fitted tensor is a finite `(X,Y,Z,6)` NIfTI ordered `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`.
- The default gradient, storage-component, PPD and tensor-basis modes preserve physical directions. Comparison modes are explicit options.
- T1 registration uses a complete CHARM directory and writes `DTI_coregT1_tensor.nii.gz` on its T1 grid. Shared subject identity does not imply image alignment.

See [Input files and coordinates](docs/en/INPUTS.md) for phase-encoding axes, supported corrections, masks and coordinate conventions.

## 🚀 Minimal workflow

Run the default legacy correction and nonlinear T1 registration:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

For reverse-PE data, prepare TOPUP and EDDY through the same workflow:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --preprocessing-mode eddy \
  --reverse-phase-encoding reverse_pe_4d.nii.gz \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

A pre-fitted tensor can enter at the registration stage:

```bash
dwi2cond-xp run-prefit-pipeline \
  DTI_tensor.nii.gz m2m_subject workflow_outputs --workers 8
```

After QA passes, the workflow publishes the tensor and its provenance into m2m. Recoverable publication failures restore the accepted pair. To run a fixed C3-to-C4 simulation, first check the configuration:

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn \
  --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 \
  --cpus 8 --dry-run
```

Remove `--dry-run` to solve. Use separate directories for each conductivity mode. E-field output is a T1-grid vector NIfTI ordered `Ex,Ey,Ez`, masked to WM/GM/CSF labels 1, 2 and 3. Standalone preparation, fitting and registration commands are in [Usage](docs/en/USAGE.md).

## 🎨 Voxel-level comparison

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

Rows show `Ex/Ey/Ez`; columns show `scalar/vn/dir/mc` with a shared symmetric scale. Slice selection uses maximum brain-mask area.

![Four-mode electric-field components](docs/images/electric_field_xyz_3x4.png)

The magnitude view uses the same vector fields and one positive scale:

![Four-mode electric-field magnitude](docs/images/electric_field_magnitude_2x2.png)

These figures illustrate a completed four-mode simulation from the project's validation data.

## ⚡ Lead-field support

`simulate-leadfield` configures SimNIBS `TDCSLEADFIELD` for all four conductivity modes. Native HDF5 output can be exported as a NumPy matrix shaped `(N_spatial*3,N_electrode-1)` with a JSON description. The first cap electrode is the reference; each other electrode defines a 1 A basis.

The interface and formats are tested. A full-subject, all-electrode lead-field benchmark is not part of the published results. See [SimNIBS integration](docs/en/SIMNIBS_INTEGRATION.md).

## 🧪 Validation evidence

Validation covers image orientation, gradient handling, WLS, fixed GRE/TOPUP/EDDY paths, affine/FNIRT registration, PPD, conductivity mappings and SimNIBS consumers. Synthetic cases include comparisons with real FSL 6.0.4 and SimNIBS 4.6.0.

On the measured complete CPU/CUDA workflow, 61 of 64 numerical artifacts were byte-identical. Brain-tensor relative L2 was `1.8693e-9`; healthy V1 axial-angle P99 was `1.2074e-6` degrees. The [validation record](docs/en/VALIDATION.md) gives conditions, metrics and historical stage results.

Real-data validation uses private HCP data. The repository distributes the two rendered comparison figures, but no source images or subject derivatives. Data access follows the [WU-Minn HCP terms](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).

## 🛣️ Roadmap

Maintenance focuses on correctness, platform compatibility and measured runtime improvements. The supported workflows and current priorities are in [Project status](docs/en/ROADMAP.md).

## 🗂️ Documentation and community

- [Documentation index](docs/en/README.md)
- [Commands](docs/en/USAGE.md) and [input requirements](docs/en/INPUTS.md)
- [Methods](docs/en/METHODS.md) and [conductivity equations](docs/en/CONDUCTIVITY.md)
- [Validation](docs/en/VALIDATION.md), [benchmarks](docs/en/BENCHMARKS.md) and [reproducibility](docs/en/REPRODUCIBILITY.md)
- [Release notes](docs/en/RELEASE_NOTES_v0.4.0.md) and [changelog](docs/en/CHANGELOG.md)
- [Contributing](docs/en/CONTRIBUTING.md), [support](docs/en/SUPPORT.md), [security](docs/en/SECURITY.md) and [code of conduct](docs/en/CODE_OF_CONDUCT.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## 🙏 Citation and license

Please cite this software using [CITATION.cff](CITATION.cff), together with SimNIBS and the diffusion-processing software used in your workflow. The project is licensed under [GPL-3.0-only](LICENSE); dependencies retain their own licenses.

Validation data were provided in part by the WU-Minn Human Connectome Project, led by David Van Essen and Kamil Ugurbil, supported by NIH Blueprint funding and the McDonnell Center for Systems Neuroscience at Washington University. See [HCP citation guidance](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations).
