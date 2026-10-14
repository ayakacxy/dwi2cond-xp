# Methods

[English](../en/METHODS.md) | [简体中文](../zh/METHODS.md)

[Documentation](README.md) · [Project home](../../README.md)

This document describes the `v0.4.0` implementation, including its
algorithm fixes and performance changes. The released `v0.3.0` evidence remains
mapped separately in [Validation](VALIDATION.md); fixture-level agreement does
not establish arbitrary-input equivalence.

## Fixed preprocessing subset

The raw-DWI commands implement only the FSL behavior used by SimNIBS 4.6
`dwi2cond`. `preprocess-nomoco` aligns b0 volumes to form the fitting reference
and mask without applying motion, eddy-current, or susceptibility correction to
the DWI. `preprocess-legacy` follows the two-pass 6/12-DOF correction order and
performs one final sinc resampling per original volume.

Legacy correction now defaults to `--bvec-mode corrected`, rotating gradients
in the same FSL scaled-mm basis as the final motion transforms. Storage-axis
reordering also transforms b-vectors, tensors, and column-major gradient-
deviation matrices by default (`--component-mode physical`). The explicit
`compat46` b-vector and `simnibs46-literal` component modes preserve the old
upstream behavior for comparison. GRE, TOPUP, and EDDY convert phase-encoding
directions and spatial inputs consistently at the internal radiological-grid
boundary, including positive-determinant x phase encoding.
The PE flags refer to FSL internal axes after storage reorientation, not an
automatic remapping of original NIfTI/BIDS `i/j/k` axes.

The GRE path accepts an already unwrapped radians-per-second fieldmap and
implements the fixed FLIRT/FUGUE composition. The TOPUP path retains the fixed
`b02b0_nosubsamp.cnf` nine-level schedule. The EDDY path retains the SimNIBS 4.6
five-round single-shell motion/quadratic-eddy-current model, spherical-GP
prediction, prediction-based slice replacement, b-vector rotation, and optional
TOPUP-field composition. Unsupported inputs fail explicitly; one preprocessing
mode never silently replaces another.

## DTI fitting

By default, the fitter requires exact b=0 volumes and one nonzero diffusion
shell, validates the b-values, and uses all supplied volumes. The standalone
`fit-dti` command also supports explicit `--shell` selection and a configurable
`--b0-threshold`, independently of the compatibility mode. `run-pipeline` uses
the default single-shell settings.

The fitter constructs the standard linear tensor design matrix and solves two
weighted least-squares systems. Both solves use the squared observed signal
as the weight for positive measurements and weight 1 for nonpositive ones.
The first solve estimates S0; the second applies a signal floor of 1% of S0
before taking logarithms, while retaining the same weights. Optional gradient-
nonlinearity coefficients adjust directions and b-values voxel by voxel.

Normal matrices are assembled once with float64 batched products and reused by
both solves. Candidates whose largest absolute tensor component is at most
`1e-8 mm²/s` are recomputed with the reference contractions in both passes,
so scale-invariant FA remains stable for nearly zero tensors.

The default `strict-fsl` mode preserves the validated FSL exceptional-value
handling and may abort explicitly. The opt-in `robust` mode excludes entire
voxels containing nonfinite signals or no positive signal, writes zero tensors
for them, and records their status in the validity mask and QA JSON. Finite
nonpositive measurements in retained voxels are handled by the signal floor.
The selected mode is never changed silently. The default derivatives are
tensor, FA, MD, MO, L1-L3, V1-V3, S0, SSE, and the validity mask.

## Tensor mapping

The caller provides an affine that maps input world coordinates to reference
world coordinates, or explicitly declares prior alignment. Resampling uses the
inverse voxel mapping required by SciPy. The transform is expressed in the
source and reference FSL scaled-mm bases before extracting its orthogonal polar
factor, and tensors are reoriented by
`R D R^T` (finite-strain reorientation). Processing is z-blocked to avoid a
whole-head multi-gigabyte 3x3 expansion.

Automatic T1 registration follows the SimNIBS 4.6 6/12-DOF FLIRT schedule and
keeps an independent 6-DOF QA result. When both registrations use the same
inputs, they share one read-only image/weight pyramid within that invocation;
their searches, optimization states and QA remain independent. The two fine
levels use adaptive FP64 correlation-ratio statistics: spatial blocks for one
candidate and candidate parallelism for batches. Coordinates, interpolation
and weighted products retain their float32 checkpoints. Coarse levels keep the
ordered evaluator; bounded changes in cost and optimization trajectories are
expected. QA and stage-cache metadata record `fine-fp64-adaptive-blocks-v1`.
The nonlinear path uses the fixed
four-level FNIRT schedule, evaluates the local deformation Jacobian, and applies
preservation of principal direction (PPD) to the tensor. Near-singular or folded
regions are recorded; a failed nonlinear path does not fall back to affine or
identity alignment.

FSL sampling uses header `pixdim`, separately from the active NIfTI world
affine. The affine and dense-warp paths support differing scales; raw and
FNIRT paths that cannot yet represent both reject mismatches explicitly.
The default `--jacobian-mode physical` differentiates millimetre displacement
with respect to target scaled-mm coordinates, so PPD and fold QA share one
physical Jacobian. Explicit `simnibs46-literal` retains the old per-voxel PPD
derivative. The FNIRT analytic nonlinear Jacobian remains a separate artifact
from the complete displacement Jacobian used for tensor reorientation.

PPD processes bounded spatial batches using the selected worker count. Each
batch includes neighbouring displacement samples, so finite differences at
batch boundaries match those on the complete grid. Inversion, eigendecomposition,
and rotation use the same float64 arithmetic, and QA still includes every voxel.

The optional `--fnirt-device cuda` switch on `register-t1-nonlinear`,
`run-pipeline`, and `run-prefit-pipeline` assembles the FNIRT spline Hessian on
an NVIDIA GPU with float64 accumulation. The default is `cpu`. CUDA requires
a working Numba CUDA runtime, NVIDIA driver, and compatible CUDA toolkit;
CuPy is not required for this backend. CUDA initializes at the first Hessian,
and spline geometry stays on the device for reuse within the current level.
Registration steps outside Hessian assembly, including LM/PCG, remain on CPU.
The selected device is included in stage provenance and cache parameters;
CUDA errors stop the run instead of silently switching devices. GPU validation
currently covers the documented Linux/RTX 2080 Ti HCP case, not every platform
or acquisition path.

With `--ppd-device cuda`, CuPy runs PPD sampling, Jacobian evaluation,
eigendecomposition, and tensor rotation on the GPU in bounded spatial batches.
The source tensor stays on the device across batches; each batch uploads its
halo and mask and downloads the packed result once. The existing float32
sampling and displacement-difference checkpoints are retained, with float64
Jacobian and matrix operations. Final derived-map calculation and NIfTI output
remain on CPU. This option needs a compatible CuPy installation (validated
with CuPy 14.2.0 and CUDA 12.8), defaults to `cpu`, and never falls back silently.
The selected PPD device is recorded in QA and stage cache parameters.

Raw-DTI preparation streams already-canonical, unscaled float32 gzip input into
its uncompressed working image using bounded buffers and a NiBabel-generated
header. Other storage orientations, scalings and component conversions retain
the existing reader. This stage computes b0 intensity centers from float64 axis
projections and fuses the final same-grid affine coordinates with trilinear
sampling. Volumes with samples near the extraslice boundary, or a large estimated
coordinate round-off bound, retain the reference sampling path. Registration schedules, fitting,
brain masking and QA remain unchanged; these choices are recorded in the raw
stage cache parameters.

FNIRT Hessian assembly batches the global-parameter spline projections in one
support traversal, preserving the original multiplication and voxel-sum order.
Only fixed one-dimensional geometry is cached; masks and weights are current
at every iteration. Single-thread CPU and explicitly supplied bases retain the
reference projection path. FNIRT keeps its LM working Hessian in CSR to share
the format with the existing CPU PCG solver across damping retries. Solver
precision, preconditioner, tolerances and iteration rules remain unchanged;
the stage cache records `batched-projection-csr-lm-v1`.

Tensor decomposition constructs float64 matrices in bounded batches and writes
eigensystems directly into disjoint output slices. Calls that need only the
principal direction do not retain the other two vectors; unrequested FA
statistics are skipped. Final QA still decomposes the stored tensor independently.
For unscaled, F-order gzip DWI, QA checks and accumulates one volume at a time,
including the gzip trailer; scaled images retain the NiBabel reader. After FNIRT
field publication and after PPD sampling, arrays no longer needed by subsequent
steps are released. The coefficient float32 write/read boundary is retained.
Physical PPD expands the complete displacement without computing unused analytic
spline Jacobians; it still obtains its physical Jacobian by finite differences
of that complete field. FNIRT output and literal mode retain analytic Jacobians.
DTI fitting retains process workers. Contiguous volume selection uses views
instead of a second copy, and validity bookkeeping reuses finite-value masks
and tests for any positive sample without a floating-point maximum. FNIRT affine
field expansion fuses per-voxel arithmetic without broadcasting full temporary
fields. Final QA loads independent required images concurrently; numerical
artifact checks still scan every value in bounded blocks.
FNIRT displacement expansion and gradient projection share spline support and
weights across three independent component accumulators. Coefficient and voxel
reduction orders are retained. Nonfinite or overflowing float32 gradient
products use the original component projections.

Nonlinear estimation keeps its existing schedule. Outside that estimator,
b0 alignment reuses intensity centers, and FLIRT evaluates candidate batches
within the worker budget; T1 fine levels use the FP64 statistics described
above. Finite
floating-point tensor and gradient-deviation components use signed axis
permutations directly; other inputs keep the dense reference operation.
Pure affine linear sampling with partial extrapolation uses a composed pull
matrix without building full coordinate grids. Large NIfTI images stream
through NiBabel into bounded gzip blocks; failed writes preserve the accepted
file and restore the image's consumed header fields. Final QA still reads and
checks the stored results independently, while sharing repeated core-image
reads and accumulating DWI means without a four-dimensional index copy.
CUDA PPD initializes zero tensors with their exact eigensystem and decomposes
only finite nonzero tensors; final derived maps remain on CPU.

## Workflow and provenance

The `nomoco` workflow reuses canonical input, gradient, aligned b0, and BET
preparation from raw-DTI QA when those inputs need materialization. Producer
parameters and file hashes are checked before reuse; raw and final clamped
tensor fits and the independent final QA still run separately.

The workflow fingerprints input content with SHA-256, writes stage outputs
and manifests atomically, validates cached structure and current content hashes
before reuse, and records parameters, versions, backend, timing, resource use,
and QA. Raw-DTI QA fingerprints include the selected fitting compatibility
mode. FEM diagnostics include the SimNIBS distribution RECORD, the inspected
run/session/FEM/conductivity/mesh modules, numerical package versions, and
native solver-library hashes. Real FEM cache reuse is disabled even when
those hashes are present because the complete set of dynamically loaded solver dependencies has not
been proven; deterministic dry-run preparation remains cacheable. A cache hit
is reported as reuse rather than fresh computation.
Reference and optimized backends remain explicit so numerical A/B can be
repeated after an optimization.
Final QA is completed before publishing the accepted tensor into the m2m
directory. External Hz-field QA reads the prepared field on the working grid.

## Conductivity mapping

Diffusion tensors are sampled onto SimNIBS mesh elements. Basis correction,
positive-eigenvalue handling, maximum conductivity, and maximum anisotropy ratio
follow the SimNIBS 4.6 compatibility requirements.

- `vn` normalizes each tensor determinant to the scalar tissue conductivity.
- With the default intensity calibration, `dir` preserves local tensor
  magnitude and fits one global scale jointly across the selected anisotropic
  tissues. Explicit `--no-correct-intensity` instead uses the non-calibrated
  safety path.
- `mc` replaces each directly scaled tensor by an isotropic tensor whose
  eigenvalue is the geometric mean of the three local conductivity
  eigenvalues, preserving the determinant and DTI-driven spatial variation.

The same SimNIBS FEM assembly is used for scalar and tensor conductivity. The
standalone mesh converter defaults to a stable symmetric `eigh` basis and
offers an explicit `simnibs46-literal` `eig` mode. The latter preserves the
official repeated-eigenvalue edge behavior and reports basis non-orthogonality
instead of silently presenting it as the stable result. FEM workflow
runs and lead-field runs apply the selected eigensystem and basis modes through
a project-local hook on the installed SimNIBS instance; installed modules are
unchanged. The default physical basis normalizes spatial affine columns
separately, preserving direction for oblique unequal-sized voxels, and rejects
shear. Explicit `--basis-mode simnibs46-literal` preserves the upstream basis
normalization for comparison.
Nonzero rank-deficient VN tensors are rejected by default because determinant
normalization is undefined. The opt-in `regularize` policy first raises the
singular eigensystem to the configured anisotropy bound, then applies the
existing safety passes and records the repair count; it is a documented robust
extension rather than a literal SimNIBS 4.6 edge result. With default intensity
calibration, an all-zero anisotropic `dir`/`mc` tissue is rejected because the
global scale is undefined; explicit `--no-correct-intensity` accepts it through
the non-calibrated safety path.

For the equations and references, see [Conductivity models](CONDUCTIVITY.md).
