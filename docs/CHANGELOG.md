# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
versions follow semantic versioning.

## [Unreleased]

## [0.3.0] - 2026-08-26

### Fixed

- Closed the final-tag independent re-audit findings (`6 P1 + 9 P2/P3`): real
  producer JSON schemas, canonical nomoco publication, reference-only MCFLIRT
  pyramids/raw-moving sampling, first-level `fix2D`, complete scientific QA
  grids and cache lineage, strict FEM inventories, and failure-atomic m2m
  tensor/provenance publication.
- Aligned standalone exact-zero QA, low-level PPD nonzero masks, FNIRT identity
  detection, and TOPUP double-precision means; made FEM dry-run/finalization,
  actual SimNIBS cache identity, ignored inputs, all-zero anisotropic tissues,
  and portable FNIRT prevalidation/attempt publication explicit and testable.
- Made the interspersed-b0 EDDY fixture gate portable across CPU architectures:
  the Linux x86_64 input hash remains frozen as FSL A/B provenance, while the
  generated DWI is also checked by affine, dtype, numerical summaries, samples,
  and injected-slice semantics instead of requiring cross-platform byte identity.
  The runtime-identity test now explicitly covers both present and absent
  threadpool-library discovery without weakening the 100% coverage gate.
- Closed the post-remediation re-audit findings `D2C-030-015..041`, including
  EDDY interspersed-b0 initialization, post-PEAS final repol, no-repol
  estimation semantics, FSL shell grouping, TOPUP common output scaling, and
  MCFLIRT thin-volume `fix2D` control flow.
- Corrected strict-FSL partial-NaN and float-mask behavior, world-to-FSL tensor
  reorientation basis, negative vecreg masks, DTIFIT metadata, and early
  rejection of non-finite tensors and gradient-deviation fields.
- Made raw QA lineage, nonlinear post-masking, simulation grid validation,
  dynamic FEM inventories, failed-stage cleanup, numerical-runtime cache
  identity, and FEM manifest transactions explicit and testable.
- Corrected GRE displacement/premat coordinate composition and made raw GRE
  registration consume the brain-masked nodif image; a non-identity real-FSL
  `applywarp` regression now locks the composition order.
- Made cache reuse verify current file and directory SHA-256 values, and made
  m2m publication require source, destination, and provenance hashes to agree.
- Made the no-TOPUP EDDY workflow construct its exact-b0 aligned nodif and BET
  0.2 mask internally while retaining an explicitly labelled external-mask
  extension.
- Restored the full FLIRT `-nosearch` default schedule, unweighted pyramid,
  correlation-ratio search cost, and FSL perturbation/range semantics; added a
  failing same-input stage/final-output FSL regression gate.
- Separated the `fslmaths -tensor_decomp` positive-eigenvalue rule from the
  `dtifit` threshold, restored strict EDDY shell distance `<100`, and aligned
  NaN, threshold, upper-median, and mask-multiply edge semantics with FSL.
- Rejected ignored reverse-PE/mask/readout combinations, enabled standalone
  EDDY z/z- phase encoding, and required the T1 brain mask for standalone
  nonlinear PPD.
- Added pre-correction raw FA/SSE fitting and 6-DOF T1 QA to every raw-DWI
  branch, corrected strict validity accounting, and declared bvals/bvecs in
  nomoco and legacy producer contracts.
- Corrected the complete EDDY DAG to fit the registered
  `corrected_dwi.nii.gz`, apply the all-volume output mask, and perform the
  official final nonnegative threshold only after geometric correction.
- Applied the T1 brain mask after nonlinear PPD reorientation and derived the
  final tensor, FA, V1, and validity mask from the masked tensor.
- Made the legacy GRE path consume the corrected fieldmap mask as an atomic
  displacement/mask contract, including an end-to-end raw rad/s fieldmap path.
- Restored the SimNIBS 4.6 DIR/MC zero-tensor ordering so zero support cannot
  distort global conductivity intensity calibration.
- Normalized nonzero b-vectors like FSL `dtifit`, separated `dtifit` and
  `fslmaths -tensor_decomp` output semantics, and added strict FSL edge-case
  behavior for nonpositive and NaN signals.
- Added canonical raw storage reorientation to EDDY, TOPUP, and fieldmap
  inputs; EDDY now accepts both b-vector layouts and x/y/z phase encoding when
  TOPUP is not used.
- Added the official reverse-PE 4-D preparation, TOPUP corrected-b0 BET mask,
  and TOPUP-to-EDDY combined workflow.
- Restored the full-volume legacy direct-b0 registration sequence, the FSL
  EDDY b0 threshold of 100, the official legacy/nonlinear defaults, and
  explicit rejection of multishell input at the strict fitting boundary.
- Added pre-fitted tensor import, atomic publication to the standard m2m tensor
  path with SHA-256 provenance, and stage-dependent CHARM/FEM input checks.
- Split strict-FSL and robust failure semantics for fitting and nonlinear PPD;
  neither mode silently changes or falls back to the other.

### Added

- Added the latest-tag independent re-audit and its issue-by-issue closure
  report, including `13370/13370` production coverage and explicit FSL 6.0.4
  reference gates.
- Added the post-remediation independent re-audit and its issue-by-issue
  closure report, including discriminative EDDY no-repol evidence and direct
  FSL gates for DTIFIT, TOPUP, vecreg, and thin-volume MCFLIRT.
- Added the independent v0.3.0 audit, its issue-by-issue remediation report,
  and a 16-threshold executable legacy/FSL stage and final-output comparison.
- Added `run-prefit-pipeline` and a complete reverse-PE `run-pipeline` branch.
- Added a versioned v0.2.0 algorithm audit and a v0.3.0 remediation/equivalence
  audit with an explicit supported-input boundary.
- Added direct local FSL 6.0.4 regression tests for standard WLS,
  gradient-nonlinearity, non-unit b-vectors, all-nonpositive signals, and NaN
  inputs.

### Changed

- Replaced the original v0.3.0 release after the independent audit while
  retaining the version number. Claims now distinguish confirmed defect
  closure and frozen synthetic tolerances from whole-optimizer or real-subject
  bitwise equivalence.
- Repositioned v0.3.0 as a correctness and official-flow remediation release.
  The previously planned acceleration cycle is deferred to v0.4.0 or later.
- Clarified that the supported GRE input is already-unwrapped radians per
  second. Wrapped phase/PRELUDE and interactive mesh QA are outside the fixed
  runtime subset and are not presented as implemented.

### Documentation

- Added clickable contents near the top of both README entry pages.
- Added a post-`v0.2.0` roadmap with explicit scientific non-goals and linked it
  from both README entry pages.
- Refreshed validation counts, release status, documentation navigation, and
  contributor/reproducibility guidance after the `v0.2.0` release audit.
- Changed future tag releases to use the matching version section of this
  changelog instead of incomplete pull-request-only generated notes.

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

- Clamped requested Numba worker counts to the runtime slots available on
  lower-core machines while preserving the public worker request and algorithm.
- Avoided concurrent Numba parallel regions when the runtime selects its
  non-thread-safe `workqueue` backend. EDDY keeps the same worker budget by
  moving parallelism inside each serial volume call on affected platforms.
- Made generated EDDY fixture text files use platform-independent LF endings
  and separated same-platform determinism from cross-platform floating-point
  tolerance checks.
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
