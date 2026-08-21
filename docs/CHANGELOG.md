# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
versions follow semantic versioning.

## [Unreleased]

## [0.1.0] - 2026-08-21

### Added

- Pure-Python single-shell two-pass WLS DTI fitting with optional HCP/FSL
  gradient-nonlinearity correction and voxel-level progress.
- FA, MD, MO, eigenvalue/eigenvector, S0, SSE, validity-mask, and QA outputs.
- Explicit affine tensor resampling and finite-strain reorientation to T1.
- SimNIBS-compatible `vn`, `dir`, and `mc` conductivity conversion.
- SimNIBS 4.6 fixed-montage FEM orchestration with Pardiso default.
- Strict WM/GM/CSF subject-volume E-field output and four-mode NIfTI figures.
- Lead-field configuration and HDF5-to-NPY/JSON validation/export contracts.
- Full statement-level tests for all 1,644 executable package statements,
  including DTI/NIfTI fitting, multiprocessing, tensor conversion and
  registration, SimNIBS adapters, simulations, lead fields, CLI dispatch,
  visualizations, error handling, and local FSL reference parity.

### Changed

- Raised CI, SimNIBS integration, and release coverage gates from 75% to 100%.
- Reworked the English and Simplified Chinese README entry pages with release,
  security, validation, performance, and scientific-boundary summaries.

### Limitations

- Raw-DWI correction, automatic affine/nonlinear registration, and nonlinear
  PPD tensor reorientation are external responsibilities.
- Full-subject all-electrode lead-field execution is not part of the release
  evidence.
