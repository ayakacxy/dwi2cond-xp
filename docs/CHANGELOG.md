# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
versions follow semantic versioning.

## [Unreleased]

No changes yet.

## [0.2.0] - 2026-08-24

### Added

- Replaced the runtime dependency on FSL preprocessing with a pure-Python
  implementation of the fixed FSL subset used by SimNIBS 4.6 `dwi2cond`.
  FSL remains optional and is used only by local numerical-reference tools.
- Added the pure-Python `preprocess-nomoco` CLI for the SimNIBS 4.6 raw-DWI
  path: input normalization, aligned b0 mean, BET-compatible mask, nonnegative
  fitting data, WLS tensor/FA/SSE, validity mask, and structured QA.
- Added the pure-Python `preprocess-legacy` CLI with the SimNIBS 4.6 two-pass
  6/12-DOF correction order, corrected-mean-to-nodif composition, direct b0
  transforms, one formal sinc resampling per volume, WLS fitting, per-volume
  matrices, and structured QA.
- Added separate `compat46` byte-preserved and finite-strain-rotated `corrected`
  b-vector contracts, plus optional precomputed field-displacement composition.
- Added `register-t1` for the SimNIBS 4.6 automatic 6/12-DOF FA-to-T1 path,
  affine tensor reorientation, FA/V1 decomposition, and 6-DOF FA/SSE QA.
- Added `prepare-fieldmap` for the fixed SimNIBS 4.6 GRE/FUGUE path from an
  already unwrapped radians-per-second fieldmap, including explicit dwell and
  phase-encoding contracts, 6-DOF MI registration, voxel shift, world-mm
  displacement, corrected b0/mask, and structured QA.
- Added `prepare-topup` for the SimNIBS 4.6 `b02b0_nosubsamp.cnf` fixed subset,
  including default source regridding, periodic cubic interpolation, rigid
  movement, nine-level LM/SCG optimization, field coefficients, corrected b0,
  joint mask, movement parameters, and structured QA.
- Added `prepare-eddy` for the SimNIBS 4.6 fixed single-shell EDDY subset,
  including five-round motion/quadratic-EC estimation, spherical-GP prediction,
  prediction-based `--repol`, rotated b-vectors, shell alignment, optional TOPUP
  field composition, 16-parameter output, iteration histories, and structured QA.
- Added fixed four-level FNIRT, analytic warp/Jacobian outputs, and nonlinear
  PPD tensor resampling/reorientation following the SimNIBS 4.6 VECREG path.
- Added a unified workflow DAG with SHA-256 input fingerprints, atomic stage
  manifests, validated cache reuse, aggregate QA, resource metrics, progress
  reporting, and `scalar`/`vn`/`dir`/`mc` FEM smoke orchestration.
- Added a Numba-aware coverage runner and isolated SimNIBS 4.6 environment
  verification used by the cross-platform CI and tag-release workflows.

### Performance

- On the public `26x26x18x26` injected-outlier fixture with a fixed seed and
  eight threads, the complete Python EDDY CLI took 7.76 seconds versus 9.77
  seconds for FSL 6.0.4 (`1.26x`). All four injected slices were detected
  exactly; mask-interior corrected-DWI and downstream tensor relative L2 were
  `0.0035352` and `0.0060714`.

- Fixed each fitting worker to one BLAS thread, preventing process/thread
  oversubscription without changing HCP tensor, FA, SSE, mask, or b0 arrays.
- Added a verified direct-mmap path for already normalized uncompressed NIfTI;
  gzip and nonconforming inputs still use one decode and one shared uncompressed
  intermediate. Both paths are bitwise identical on the HCP fixture.
- Measured the eight-worker HCP `nomoco` closure at 18.82 seconds for validated
  `.nii` mmap and 52.92 seconds for `.nii.gz`; the older local SimNIBS 4.6/FSL
  6.0.4 reference took 257.30 seconds. Its input encoding was not frozen, so
  the derived `13.67x` and `4.86x` ratios remain planning comparisons rather
  than formal same-encoding release claims.
- Replaced the width-seven Hanning-sinc sampler with an output-point-parallel
  Numba kernel that remains bitwise identical to the retained reference kernel;
  the synthetic legacy final-resampling stage fell from about 8.39 seconds to
  0.054 seconds.
- Fused the MCFLIRT normcorr sampling hot path while retaining explicit FSL
  float32 product boundaries and NumPy-compatible pairwise reduction. Linux
  uses eight processes only for independent 4 mm volume stages. Three complete
  synthetic runs took 2.75/2.71/2.87 seconds versus 6.77 seconds for FSL, with
  matrices and final Python artifacts bitwise equal to the pre-optimization path.
- Profiled and optimized the complete T1-registration closure by overlapping
  independent 6/12-DOF searches, tensor components/source mask, QA resampling,
  and tensor gzip with read-only FA/V1 decomposition. The private whole-head
  process fell from 100.90 to 53.23 seconds versus 233.18 seconds for FSL.
- Batched independent FLIRT Brent trajectories in lockstep and constructed each
  candidate round's affine matrices in native batches. The public tiny GRE
  fieldmap process fell from 6.55--6.88 seconds to 1.49--1.51 seconds while the
  matrix, seven checked output arrays, and 11,349-evaluation schedule remained
  identical to the retained Python path.
- Batched the two independent TOPUP scan samplers, used an ordered CSR view for
  repeated regularization products, and parallelized independent coefficient
  expansion voxels without changing per-voxel FSL coefficient order. The tiny
  fixture's complete CLI median is now 2.17 seconds versus 4.65 seconds for FSL;
  its resident algorithm median is 0.630 seconds, with all five artifacts
  file-bitwise equal to the previous Python output.
- Compiled the dense TOPUP PCG loop into one ordered Numba call and batched the
  five independent movement-interaction transpose columns. The retained
  reference backend is bitwise equal on the public fixture. Nine fresh CLI runs
  now have a 2.12-second median versus 4.61 seconds for FSL 6.0.4 (`2.17x`).
- Fused the four TOPUP Gauss--Newton spline Hessian products into one ordered
  Numba assembly kernel. Three independent complete CLI runs took
  4.14/4.09/4.18 seconds versus 4.78/4.75/4.66 seconds for FSL 6.0.4 on the
  public tiny reverse-PE fixture, with a 4.14 versus 4.75 second median.

### Fixed

- Restored the FSL `tensor_decomp` rule that FA/eigenvector outputs remain zero
  unless the largest tensor eigenvalue is positive; real FA support now matches
  the FSL reference exactly.
- Corrected FNIRT moving-image smoothing to use the moving DTI-FA voxel size and
  prevented parallel Hessian blocks from sharing mutable CSC structure arrays.
- Kept FNIRT progress monotonic across four levels and exposed output,
  nonlinear-PPD, and QA work after optimization completes.

### Changed

- Documented the exact `vn`, `dir`, and `mc` conductivity equations, SimNIBS
  safety projection, and primary literature references in both READMEs.
- Clarified that `mc` uses the geometric mean of the directly scaled local
  conductivity eigenvalues rather than an arithmetic mean.
- Raised the release contract from post-preprocessing tensor workflows to the
  pure-Python SimNIBS 4.6 `dwi2cond` preprocessing subset while retaining all
  `v0.1.0` tensor, conductivity, FEM, and lead-field interfaces.

### Limitations

- This release implements only the fixed FSL behavior used by SimNIBS 4.6
  `dwi2cond`; it is not a general replacement for FSL commands.
- Wrapped phase inputs that require PRELUDE remain explicitly unsupported; the
  GRE path accepts an already unwrapped radians-per-second fieldmap.
- Full-flow `10x` performance is not a `v0.2.0` release claim. Additional
  performance work is deferred to the `v0.3.0` development cycle.

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

- Final cross-platform QA/DAG and isolated-release acceptance remain in
  progress; the FSL-free replacement positioning is not yet released.
- Full-subject all-electrode lead-field execution is not part of the release
  evidence.
