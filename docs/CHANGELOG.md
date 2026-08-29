# Changelog

This changelog stops at the historical `v0.1.0` release. Later-version changes
belong to their corresponding tags.

## [0.1.0] - 2026-08-21

Initial public release for converting externally preprocessed diffusion MRI
into SimNIBS 4.6 scalar or anisotropic-conductivity workflows.

### Added

- Pure-Python single-shell, two-solve WLS DTI fitting with optional HCP/FSL
  gradient-nonlinearity correction and voxel-level progress.
- Tensor, FA, MD, MO, L1-L3, V1-V3, S0, SSE, validity-mask, and QA outputs.
- Caller-supplied affine tensor mapping and finite-strain reorientation to a T1
  grid.
- SimNIBS-compatible `vn`, `dir`, and `mc` conductivity conversion; `dir` and
  `mc` use one global calibration across the selected anisotropic tissues.
- SimNIBS 4.6 fixed-montage FEM orchestration with Pardiso as the default.
- Strict WM/GM/CSF subject-volume E-field masking and four-mode NIfTI figures.
- Lead-field configuration plus HDF5-to-NPY/JSON validation and export
  contracts.
- Full statement-level tests for all 1,644 executable package statements in the
  recorded release environment, including the optional local FSL reference
  comparison.

### Evidence boundary

- The retained HCP DTI, synthetic-sphere conductivity, and four-mode fixed-
  montage FEM results apply to the documented inputs and stages only.
- Raw-DWI motion/eddy-current/susceptibility correction, automatic affine or
  nonlinear registration, and nonlinear PPD tensor reorientation are not
  `v0.1.0` capabilities.
- Full-subject all-electrode lead-field execution is not part of the release
  evidence.

### Documentation maintenance - 2026-08-29

- Reissued the same `v0.1.0` tag to correct scientific wording, distinguish
  historical experiments from tag-local tests, and make Release publication a
  fail-closed upsert.
- The algorithm source remains the original `v0.1.0` baseline; this maintenance
  update adds no preprocessing stage, numerical method, or performance result.
