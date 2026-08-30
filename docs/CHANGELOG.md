# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/), and
released versions follow semantic versioning.

The `v0.1.0`--`v0.3.0` tags were republished on 2026-08-29 for documentation
and release-metadata corrections. `v0.3.0` was replaced again on 2026-08-30
after a fourth independent source audit found two algorithm defects and three
conditional compatibility/artifact defects.

## Release map

The commit column records the algorithm baseline for each release. A tagged
documentation or release-metadata descendant does not change that baseline or
the stated evidence boundary.

| Release | Algorithm baseline | Release role |
| --- | --- | --- |
| `v0.1.0` | `29f98a4` | Established DTI fitting, tensor-to-conductivity conversion, and SimNIBS-facing FEM/lead-field interfaces. |
| `v0.2.0` | `20f4092` | Added the pure-Python fixed preprocessing and registration subset used by SimNIBS 4.6 `dwi2cond`. |
| `v0.3.0` | `ae5a6a7` | Current stable release; includes the fourth independent-audit remediation for matrix, BET, eigensystem, FEM-cache, and TOPUP contracts. |

## [Unreleased]

### Changed

- Consolidated the `v0.1.0`--`v0.3.0` project history into a user-facing
  release narrative and aligned methods and theory statements with the tagged
  implementation and the experiments that were actually completed.
- Retired superseded audit and planning material from the active documentation
  set while preserving the release facts and evidence needed to interpret the
  stable version.
- Cancelled the active `v0.4.0`/`10x` performance cycle. Its incomplete planning
  samples are not a baseline, a release target, or a current development
  commitment.

## [0.3.0] - 2026-08-30

### Summary

`v0.3.0` is the correctness-remediated stable release. It retains the fixed
SimNIBS 4.6 `dwi2cond` scope introduced in `v0.2.0`, while correcting the
producer-to-consumer workflow, numerical edge cases, publication transactions,
and runtime identity checks found by independent source and same-input audits.

### Key changes

- Corrected the raw-DWI and pre-fitted-tensor workflows, including EDDY/TOPUP
  artifact lineage, final mask and threshold ordering, GRE displacement
  composition, legacy MCFLIRT control flow, strict single-shell fitting, and
  affine/nonlinear DTI-to-T1 publication.
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
- Made workflow and FEM publication failure-atomic: failed retries preserve
  diagnostic manifests and do not replace the last accepted tensor, m2m
  artifact, or simulation result.
- Replaced silent VN handling of nonzero singular tensors with an explicit
  default error and an opt-in, bounded, QA-recorded regularization policy.
- Added complete raw and pre-fitted pipeline entry points, SHA-256 provenance,
  structured QA, validated cache reuse, progress/resource reporting, and
  `scalar`/`vn`/`dir`/`mc` SimNIBS orchestration.
- Closed the positive-determinant legacy matrix mismatch by converting every
  local registration producer into the FSL radiological scaled-mm contract
  before composition, saved matrices, resampling, and corrected b-vectors.
- Transcribed BET2's 1 mm and 7 mm endpoints, sub-millimetre `dscale` loop,
  near-surface maximum samples, and zero-force out-of-bounds behavior into the
  shared reference/optimized sampling contract.
- Kept stable symmetric conductivity decomposition as the default and added an
  explicit, QA-recorded `simnibs46-literal` eigensystem mode for exact
  repeated-eigenvalue compatibility testing.
- Disabled cache hits for real SimNIBS FEM execution until the complete dynamic
  Python/native solver closure can be fingerprinted; deterministic dry-run
  preparation remains cacheable.
- Added official TOPUP cubic-spline intent parameters and an FSL-readable
  `topup_fieldcoef`/`topup_movpar` prefix bundle while retaining the existing
  project artifact names.

### Validation

- With all real FSL probes configured, the final repository gate completed with
  `666 passed` and no skips. The independent coverage run repeated those
  `666 passed`, completed `12 passed` in the montage batch, and covered
  `13,663/13,663` production statements; real synthetic TOPUP, EDDY,
  and FNIRT CLI paths were included.
- Carried-forward stage references, collected before the final remediation,
  include HCP WLS tensor relative L2 `4.18e-6` against FSL 6.0.4, a local
  synthetic-sphere conductivity summary of maximum absolute error `0` for `vn`
  and `2.22e-16` for `dir`/`mc`, and a private T1/T2 CHARM-to-C3/C4 FEM run for
  all four conductivity modes with finite, strictly tissue-masked vector fields.
  They remain versioned experimental evidence, not a final-tag full-chain rerun.
- Ubuntu, macOS, and Windows CI, package construction, dependency and metadata
  checks, CodeQL, OpenSSF Scorecard, release checksums, CycloneDX SBOM, build
  provenance, and isolated wheel installation all passed for the final tag.

### Supported scope

- This is an independent implementation of the fixed FSL behavior used by
  SimNIBS 4.6 `dwi2cond`, not a general replacement for FSL commands. FSL is an
  optional numerical oracle and is not a runtime dependency.
- Strict tensor fitting is single-shell. Wrapped phase requiring PRELUDE and
  arbitrary TOPUP/EDDY/FNIRT configurations remain outside the supported
  contract; the GRE path accepts an already unwrapped radians-per-second field.
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
  `compat46` and finite-strain-corrected b-vector contracts.
- Added fixed GRE/FUGUE, TOPUP, and single-shell EDDY `--repol` paths, including
  optional TOPUP-to-EDDY field composition and structured intermediate QA.
- Added automatic 6/12-DOF affine registration, four-level FNIRT, nonlinear PPD
  tensor resampling/reorientation, and standard tensor/FA/V1 publication.
- Added a unified workflow DAG with input fingerprints, atomic manifests,
  validated cache reuse, progress/resource metrics, and conductivity/FEM smoke
  orchestration while preserving the `v0.1.0` interfaces.

### Validation and measured results

- The Linux release gate completed with `535 passed, 7 skipped` and
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
  numerical-contract defects in `v0.2.0`; those defects motivated and were
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
  E-field publication, visualization, and lead-field validation/export
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

[Unreleased]: https://github.com/ayakacxy/dwi2cond-xp/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.3.0
[0.2.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.1.0
