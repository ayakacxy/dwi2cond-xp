# Changelog

[English](en/CHANGELOG.md) | [简体中文](zh/CHANGELOG.md)

[Documentation](en/README.md) · [Project home](../README.md)

All notable changes follow [Keep a Changelog](https://keepachangelog.com/), and
released versions follow semantic versioning.

The `v0.1.0`--`v0.3.0` tags were republished on 2026-08-29 for documentation
and release-metadata corrections. `v0.3.0` was replaced again on 2026-08-30
after a fourth independent source audit found two algorithm defects and three
conditional compatibility/artifact defects.

## Documentation update — 2026-10-13

The documentation for `v0.1.0`, `v0.2.0`, and `v0.3.0` has been edited and
paired in English and Chinese, with links that stay in the selected language.
That documentation-only pass left algorithms, command behavior, dependency
requirements, and recorded scientific results unchanged. The later algorithm
fixes are included in v0.4.0 below.

## Release map

The commit column records the algorithm baseline for each release. A tagged
documentation or release-metadata descendant does not change that baseline or
the stated scope of the evidence.

| Release | Algorithm baseline | Release role |
| --- | --- | --- |
| `v0.1.0` | `29f98a4` | Established DTI fitting, tensor-to-conductivity conversion, and SimNIBS-facing FEM/lead-field interfaces. |
| `v0.2.0` | `20f4092` | Added the pure-Python fixed preprocessing and registration subset used by SimNIBS 4.6 `dwi2cond`. |
| `v0.3.0` | `ae5a6a7` | Previous stable release; includes the fourth independent-audit remediation for matrix, BET, eigensystem, FEM-cache, and TOPUP requirements. |
| `v0.4.0` | Tag `v0.4.0` | Coordinate/orientation fixes and CPU/CUDA workflow acceleration. |

## [Unreleased]

No additional changes are recorded after v0.4.0.

## [0.4.0] - 2026-10-14

### Summary

The release focuses on coordinate and tensor-orientation corrections, then
reduces repeated computation and large-array overhead in the same workflow.
See the [release notes](https://github.com/ayakacxy/dwi2cond-xp/blob/v0.4.0/docs/en/RELEASE_NOTES_v0.4.0.md) for migration and device choices.

### Fixed

- Rotate legacy motion-corrected gradients in the FSL basis and make
  `--bvec-mode corrected` the default. Storage reordering now transforms
  b-vectors, tensor components and column-major gradient-deviation matrices
  together (`--component-mode physical`).
- Correct GRE/TOPUP phase-encoding signs and EDDY's radiological-coordinate
  conversion for positive-determinant storage, restoring image-space outputs
  without reflecting movement parameters or FSL gradients twice.
- Use NIfTI header `pixdim` independently of the world affine for affine and
  dense-warp FSL sampling. Raw/FNIRT paths that cannot represent a mismatch
  reject it explicitly.
- Use physical target-grid derivatives for PPD and fold/near-singular QA
  (`--jacobian-mode physical`), including non-1-mm target voxels.
- Normalize each affine column separately for the physical tensor basis,
  reject shear, and apply the selected stable symmetric eigensystem and basis
  to formal SimNIBS FEM/lead-field instances without editing SimNIBS itself.
- Validate prepared external Hz fields on the working grid and complete final
  QA before publishing the m2m tensor and provenance.

### Changed

- Raw preparation and WLS reuse invariant work, stream eligible float32 gzip
  input, reduce volume/coordinate/validity copies, and retain separate raw and
  final fitting passes. T1 affine and rigid QA registration share a read-only
  pyramid; fine correlation-ratio levels use adaptive FP64 block statistics.
- FNIRT shares spline-support traversal, avoids unused basis and affine-field
  temporaries, and keeps the LM working matrix in CSR for the existing CPU
  PCG solver. Iterations, damping, tolerances and stopping rules are retained.
- PPD and tensor decomposition use bounded spatial batches, avoid unused
  analytic Jacobians and directions, and release completed fields sooner.
  QA and NIfTI writing reduce repeated reads and full-volume temporaries while
  retaining complete finite-value checks, content hashes and scientific QA.
- Add explicit CUDA backends: `--fnirt-device cuda` for FP64 spline-Hessian
  assembly and `--ppd-device cuda` for sampling, Jacobians, decomposition and
  rotation. CPU remains the default and needs no CUDA/CuPy imports. PPD retains
  float32 sampling checkpoints and FP64 matrix operations; LM/PCG stays on CPU.
  CUDA failures stop the run rather than silently changing devices.

### Validation and measured performance

- Local release checks passed 955 unique cases (942 main + 1 focused regression
  + 12 montage), with no skips and configured real FSL/SimNIBS probes enabled.
  The Numba-aware full-package gate covered 15,363/15,363 statements (100.00%)
  and ran three real synthetic TOPUP/EDDY/FNIRT CLI paths. Device-free CUDA
  math tests complement the separate real-GPU evidence; statement coverage
  is not complete branch coverage or an arbitrary-input correctness claim.
- On one Linux HCP single-shell input, eight-core warm-cache complete runs
  took **205.452 s / 6.498 GiB host RSS** on CPU and **163.342 s / 6.723 GiB
  host RSS** on GPU; GPU allocation peaked at **8.834 GiB**. FEM was off.
- The published v0.3.0 CPU run took **564.406 s / 25.469 GiB**, giving observed
  wall-time ratios of **2.747x / 3.455x** for current CPU/GPU. Official
  SimNIBS 4.6/FSL 6.0.4 subscripts with native gradient-nonlinearity correction
  took **1290.073 s / 3.794 GiB**, giving **6.279x / 7.898x**. FSL wrote fewer
  project diagnostics and often used one core; these are workflow comparisons,
  not pure hardware speedups or final FSL-equivalence claims.
- CPU/GPU agreed byte-for-byte for 61 of 64 numerical artifacts. Brain tensor
  relative L2 was `1.8693e-9`, final FA maximum absolute difference `8.7172e-7`,
  and healthy V1 axial-angle P99 `1.2074e-6` degrees. Geometry, masks, FNIRT
  fields/coefficients and recorded LM results agreed. The [benchmark page](https://github.com/ayakacxy/dwi2cond-xp/blob/v0.4.0/docs/en/BENCHMARKS.md)
  gives timing boundaries, stage times, numerical checks and limitations.

### Compatibility

The new corrected/physical defaults intentionally change affected outputs.
Explicit `--bvec-mode compat46` and `simnibs46-literal` component, Jacobian,
basis and eigensystem modes remain available for upstream comparisons; they do
not disable the GRE/TOPUP/EDDY or publication fixes. GPU evidence covers the
recorded Linux/RTX 2080 Ti case; no new other-platform GPU or full FEM
validation is claimed.

## [0.3.0] - 2026-08-30

### Summary

`v0.3.0` is the correctness-remediated stable release. It retains the fixed
SimNIBS 4.6 `dwi2cond` scope introduced in `v0.2.0`, while correcting the
data flow between processing stages, numerical edge cases, output publication transactions,
and runtime identity checks found by independent source and same-input audits.

### Key changes

- Corrected the raw-DWI and pre-fitted-tensor workflows, including EDDY/TOPUP
  artifact lineage, final mask and threshold ordering, GRE displacement
  composition, legacy MCFLIRT control flow, strict single-shell fitting, and
  affine/nonlinear DTI-to-T1 output publication.
- Restored the SimNIBS/FSL ordering for tensor decomposition, nonlinear PPD
  masking and reorientation, b-vector normalization, zero-tensor DIR/MC
  calibration, and derived FA/V1/validity outputs.
- Implemented the FSL 6.0.4/NewNifti orientation decision using its float32
  Gram--Schmidt, signed-permutation, and strict tie-breaking rules, including
  high-obliquity and all signed-permutation regression cases.
- Pinned the validated nonlinear numerical stack to NumPy 2.3.0 and Numba
  0.64.0, and expanded cache identity to cover fitting mode, scientific inputs,
  implementation modules, numerical packages, SimNIBS installation records,
  and native solver libraries.
- Made workflow and FEM output publication failure-atomic: failed retries preserve
  diagnostic manifests and do not replace the last accepted tensor, m2m
  artifact, or simulation result.
- Replaced silent VN handling of nonzero singular tensors with an explicit
  default error and an opt-in, bounded, QA-recorded regularization policy.
- Added complete raw and pre-fitted pipeline entry points, SHA-256 provenance,
  structured QA, validated cache reuse, progress/resource reporting, and
  `scalar`/`vn`/`dir`/`mc` SimNIBS orchestration.
- Closed the positive-determinant legacy matrix mismatch by converting every
  local registration producer into the FSL radiological scaled-mm convention
  before composition, saved matrices, resampling, and corrected b-vectors.
- Transcribed BET2's 1 mm and 7 mm endpoints, sub-millimetre `dscale` loop,
  near-surface maximum samples, and zero-force out-of-bounds behavior into the
  shared reference/optimized sampling rules.
- Kept stable symmetric conductivity decomposition as the default and added an
  explicit, QA-recorded `simnibs46-literal` eigensystem mode for exact
  repeated-eigenvalue compatibility testing.
- Disabled cache hits for real SimNIBS FEM execution until the complete set of dynamically loaded
  Python and native solver dependencies can be fingerprinted; deterministic dry-run
  preparation remains cacheable.
- Added official TOPUP cubic-spline intent parameters and an FSL-readable
  `topup_fieldcoef`/`topup_movpar` prefix bundle while retaining the existing
  project artifact names.

### Validation

- With all real FSL probes configured, the final repository check completed with
  `666 passed` and no skips. The independent coverage run repeated those
  `666 passed`, completed `12 passed` in the montage batch, and covered
  `13,663/13,663` production statements; real synthetic TOPUP, EDDY,
  and FNIRT CLI paths were included.
- Earlier stage references, collected before the final remediation,
  include HCP WLS tensor relative L2 `4.18e-6` against FSL 6.0.4, a local
  synthetic-sphere conductivity summary of maximum absolute error `0` for `vn`
  and `2.22e-16` for `dir`/`mc`, and a private T1/T2 CHARM-to-C3/C4 FEM run for
  all four conductivity modes with finite, strictly tissue-masked vector fields.
  They remain versioned experimental evidence, not a release full-chain rerun.
- Ubuntu, macOS, and Windows CI, package construction, dependency and metadata
  checks, CodeQL, OpenSSF Scorecard, release checksums, CycloneDX SBOM, build
  provenance, and isolated wheel installation all passed for the final tag.

### Supported scope

- This is an independent implementation of the fixed FSL behavior used by
  SimNIBS 4.6 `dwi2cond`, not a general replacement for FSL commands. FSL is an
  optional numerical reference and is not a runtime dependency.
- Strict tensor fitting is single-shell. Wrapped phase requiring PRELUDE and
  arbitrary TOPUP/EDDY/FNIRT configurations remain outside the supported
  scope; the GRE path accepts an already unwrapped radians-per-second field.
- The evidence supports the tested stage and workflow boundaries. It does not
  assert arbitrary-input bitwise identity for iterative EDDY/FNIRT, a universal
  end-to-end speed ratio, population-level generalization from one private HCP
  subject, or completed all-electrode lead-field execution.

## [0.2.0] - 2026-08-24

### Summary

`v0.2.0` expanded the initial tensor/conductivity package into a pure-Python,
FSL-free-at-runtime implementation of the fixed preprocessing and registration
subset called by SimNIBS 4.6 `dwi2cond`.

### Key changes

- Added `nomoco` and legacy raw-DWI preprocessing, weighted tensor fitting,
  gradient-nonlinearity support, aligned b0/mask generation, and explicit
  `compat46` and finite-strain-corrected b-vector modes.
- Added fixed GRE/FUGUE, TOPUP, and single-shell EDDY `--repol` paths, including
  optional TOPUP-to-EDDY field composition and structured intermediate QA.
- Added automatic 6/12-DOF affine registration, four-level FNIRT, nonlinear PPD
  tensor resampling/reorientation, and standard tensor/FA/V1 output publication.
- Added a unified workflow with input fingerprints, atomic manifests,
  validated cache reuse, progress/resource metrics, and conductivity/FEM smoke
  orchestration while preserving the `v0.1.0` interfaces.

### Validation and measured results

- The Linux release check completed with `535 passed, 7 skipped` and
  `12,443/12,443` executable statements covered. The 100% coverage threshold
  also passed on Ubuntu, macOS, and Windows; optional external-reference tests
  skipped visibly when their prerequisites were unavailable.
- On the public `26x26x18x26` injected-outlier fixture, the complete eight-thread
  EDDY CLI took `7.76 s` versus `9.77 s` for FSL 6.0.4. All four injected slices
  were detected; corrected-DWI and downstream-tensor relative L2 values were
  `0.0035352` and `0.0060714` within the mask.
- On the public `16x14x12` reverse-PE fixture, complete TOPUP CLI medians were
  `2.12 s` for dwi2cond-xp and `4.61 s` for FSL 6.0.4. These are fixture- and
  boundary-specific measurements, not a whole-workflow speed claim.

### Limitations

- The fixed subset did not cover PRELUDE-wrapped phase or arbitrary FSL command
  configurations. Strict fitting remained single-shell.
- A later source/control-flow audit found material workflow composition and
  numerical defects in `v0.2.0`; those defects motivated and were
  addressed by `v0.3.0`. New deployments should use the current stable release.

## [0.1.0] - 2026-08-21

### Summary

`v0.1.0` established the package's DTI-to-conductivity and SimNIBS integration
foundation for preprocessed single-shell DWI or a supplied six-component
diffusion tensor.

### Key changes

- Added pure-Python two-pass WLS DTI fitting with optional HCP/FSL
  gradient-nonlinearity correction, plus tensor, FA, MD, MO, eigenvalue,
  eigenvector, S0, SSE, validity-mask, and QA outputs.
- Added affine tensor resampling and finite-strain reorientation to T1, and the
  SimNIBS-compatible `vn`, `dir`, and `mc` conductivity mappings.
- Added fixed-montage SimNIBS 4.6 FEM orchestration, strict WM/GM/CSF vector
  E-field output publication, visualization, and lead-field validation/export
  interfaces.

### Validation and measured results

- The release suite completed with `144 passed` and `1,644/1,644` executable
  statements covered.
- On one private HCP b0+b1000 input, the tensor relative L2 against FSL 6.0.4
  WLS with gradient nonlinearity was `4.18e-6`. On the same host and fitting
  boundary, the 16-worker Python fit took `9.76 s` versus `108.23 s` for FSL;
  this was a DTI-fitting result, not an end-to-end workflow claim.
- Synthetic-mesh conductivity matched SimNIBS 4.6 to maximum absolute error
  `0` for `vn` and `2.22e-16` for `dir`/`mc`. Real C3-to-C4 1 mA FEM runs
  completed for `scalar`, `vn`, `dir`, and `mc` with finite, strictly
  tissue-masked vector fields.

### Limitations

- Raw-DWI motion/distortion correction and automatic affine/nonlinear
  registration were external responsibilities in this release.
- Full-subject, all-electrode lead-field execution was not part of the
  end-to-end release evidence.

[Unreleased]: https://github.com/ayakacxy/dwi2cond-xp/compare/v0.4.0...HEAD
[0.4.0]: en/RELEASE_NOTES_v0.4.0.md
[0.3.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.3.0
[0.2.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.1.0
