# Roadmap

This roadmap describes priorities after the completed `v0.2.0` release. It is
not a promise that every item will ship unchanged or meet a predetermined speed
ratio. Scientific correctness and explicit evidence remain release gates.

## Current release

Version `0.2.0` implements the pure-Python fixed FSL subset required by SimNIBS
4.6 `dwi2cond`, together with DTI fitting, tensor reorientation, anisotropic
conductivity, fixed-montage FEM, QA, caching, and cross-platform packaging. FSL
is optional and used only for local numerical-reference comparisons.

The `main` branch contains current development. Stable code remains available
through immutable tags and GitHub Releases; maintenance for an older release
starts from its tag only when a backport is needed.

## Planned `v0.3.0` priorities

1. Freeze same-input, same-output, eight-worker end-to-end benchmarks for the
   supported affine and nonlinear preprocessing branches, including explicit
   `.nii` and `.nii.gz` timing boundaries.
2. Profile before changing code, then optimize the largest measured FNIRT,
   nonlinear PPD, affine, compression, and I/O costs one hotspot at a time.
3. Reduce whole-head peak memory through equivalent chunked or fused data flow,
   with particular attention to nonlinear tensor reorientation.
4. Add regression evidence on ordinary 8--16-thread workstations and additional
   legally usable inputs without distributing human MRI data.
5. Improve full-workflow and packaging diagnostics while keeping reference and
   optimized backends explicit and independently testable.

## Non-goals

- Do not reduce image resolution, iterations, data, or convergence checks.
- Do not change stopping conditions, tensor ordering, coordinate conventions,
  floating-point contracts, or failure semantics to obtain a faster result.
- Do not silently fall back from an optimized backend or present cache hits as
  fresh computation.
- Do not claim a universal 10x end-to-end speedup without a successful,
  same-boundary measurement for each supported branch being discussed.
- Do not expand the project into a general FSL command replacement.

The quantitative constraints and current bottleneck evidence are documented in
the [end-to-end performance feasibility report](DWI2COND_10X_PERFORMANCE_FEASIBILITY_REPORT.md).
Feature proposals and performance reports should use the repository issue
templates and include a shareable reproducer, numerical A/B, timing boundary,
hardware, worker count, and peak memory.
