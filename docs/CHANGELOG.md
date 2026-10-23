# Changelog

[English](en/CHANGELOG.md) | [简体中文](zh/CHANGELOG.md)

[Documentation](en/README.md) · [Project home](../README.md)

Changes are listed by release. Dates identify the release; detailed numerical results are kept in [Validation](en/VALIDATION.md) and [Benchmarks](en/BENCHMARKS.md).

## [Unreleased]

No additional algorithm changes are recorded.

## [0.4.0] - 2026-10-14

### Summary

Corrected gradient and tensor geometry, with CPU and optional CUDA workflow acceleration.

### Fixed

- Corrected legacy motion-gradient rotation and physical vector/tensor components during storage reordering.
- Corrected GRE/TOPUP/EDDY handedness, physical PPD derivatives and conductivity bases on oblique grids.
- Distinguished NIfTI header sampling scales from the active world affine.
- Applied the selected eigensystem and basis to formal SimNIBS FEM and lead-field consumers.
- Fixed Windows in-place and hard-link NIfTI writes.
- Checked prepared external fields and completed QA before publishing tensor/provenance outputs.

### Changed

- Reduced repeated preparation, matrix assembly, spline traversal, image reads and temporary arrays.
- Added explicit `--fnirt-device cuda` and `--ppd-device cuda` backends; CPU remains the default.
- Retained solver schedules, stopping rules and separate raw/final WLS fits.
- Reorganized the bilingual guides and synchronized packaged documentation.

[Release notes and migration](en/RELEASE_NOTES_v0.4.0.md).

## [0.3.0] - 2026-08-30

### Fixed

- Repaired workflow data flow, masks, b-vector normalization, tensor-derived maps, prefitted import and m2m publication issues identified in the v0.2 audit.
- Matched FSL/NewNifti orientation, pinned the validated NumPy/Numba stack, and corrected CHARM sentinel and resolved-mesh handling.
- Corrected legacy matrix bases, BET ray endpoints and sampling, and TOPUP coefficient intent metadata.
- Added stable and literal conductivity eigensystem choices, explicit singular-VN handling and stronger FEM execution identity; real FEM cache reuse was disabled after the fourth review.

The third and fourth audit reports and their remediation records retain the finding-level details. The scientific baseline is `ae5a6a7`; later documentation descendants keep that source unchanged.

## [0.2.0] - 2026-08-24

### Added

- Pure-Python fixed SimNIBS 4.6 preprocessing: legacy motion/eddy correction, GRE fieldmap, TOPUP and single-shell EDDY.
- Automatic affine and fixed FNIRT nonlinear registration with PPD tensor reorientation.
- Workflow records, caching and numerical reference fixtures.

Scientific baseline: `20f4092`. Findings from the subsequent [algorithm audit](en/V0.2.0_ALGORITHM_AUDIT_REPORT_2026-08-25.md) were addressed in the next release.

## [0.1.0] - 2026-08-21

### Added

- Two-pass WLS DTI fitting with optional gradient-nonlinearity correction.
- Tensor resampling, reorientation and `scalar`/`vn`/`dir`/`mc` conductivity mapping.
- SimNIBS fixed-montage and lead-field interfaces, field comparison plots and QA records.

Scientific baseline: `29f98a4`.

## Release history

The v0.1.0–v0.3.0 tags were republished on 2026-08-29 to correct documentation and metadata. The v0.3.0 source was replaced on 2026-08-30 after the fourth review. Original findings, remediation records and tagged source identities are retained for reproducibility.

[0.4.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.4.0
[0.3.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.3.0
[0.2.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.1.0
