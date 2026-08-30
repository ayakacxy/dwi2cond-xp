# Methods

This document describes the algorithm and failure contracts implemented by the
final `v0.3.0` source. It does not turn fixture-level agreement into an
arbitrary-input equivalence claim. The corresponding source, test, numerical,
and end-to-end evidence is mapped in [Validation](VALIDATION.md).

## Fixed preprocessing subset

The raw-DWI commands implement only the FSL behavior used by SimNIBS 4.6
`dwi2cond`. `preprocess-nomoco` aligns b0 volumes to form the fitting reference
and mask without applying motion, eddy-current, or susceptibility correction to
the DWI. `preprocess-legacy` follows the two-pass 6/12-DOF correction order and
performs one formal sinc resampling per original volume.

The GRE path accepts an already unwrapped radians-per-second fieldmap and
implements the fixed FLIRT/FUGUE composition. The TOPUP path retains the fixed
`b02b0_nosubsamp.cnf` nine-level schedule. The EDDY path retains the SimNIBS 4.6
five-round single-shell motion/quadratic-eddy-current model, spherical-GP
prediction, prediction-based slice replacement, b-vector rotation, and optional
TOPUP-field composition. Unsupported inputs fail explicitly; one preprocessing
mode never silently replaces another.

## DTI fitting

The fitter selects b=0 and one diffusion shell, constructs the standard linear
tensor design matrix, and performs two-pass weighted least squares. The second
pass uses predicted signal squared as the weight, matching the validated FSL
6.0.4 reference semantics. Gradient-nonlinearity coefficients, when supplied,
modify gradient directions and b-values voxel by voxel.

Failure handling is mode-specific. The default `strict-fsl` mode preserves the
validated FSL failure and exceptional-value semantics and may abort explicitly;
the opt-in `robust` mode excludes invalid measurements or voxels, writes a zero
tensor for excluded voxels, and records their status in the validity mask and QA
JSON. The selected mode is never changed silently. The default derivatives are
tensor, FA, MD, MO, L1-L3, V1-V3, S0, SSE, and the validity mask.

## Tensor mapping

The caller provides an affine that maps input world coordinates to reference
world coordinates, or explicitly declares prior alignment. Resampling uses the
inverse voxel mapping required by SciPy. The linear part of the world transform
is reduced to its orthogonal polar factor, and tensors are reoriented by
`R D R^T` (finite-strain reorientation). Processing is z-blocked to avoid a
whole-head multi-gigabyte 3x3 expansion.

Automatic T1 registration follows the SimNIBS 4.6 6/12-DOF FLIRT schedule and
keeps an independent 6-DOF QA result. The nonlinear path uses the fixed
four-level FNIRT schedule, evaluates the local deformation Jacobian, and applies
preservation of principal direction (PPD) to the tensor. Near-singular or folded
regions are recorded; a failed nonlinear path does not fall back to affine or
identity alignment.

## Workflow and provenance

The workflow DAG fingerprints input content with SHA-256, writes stage outputs
and manifests atomically, validates cached structure and current content hashes
before reuse, and records parameters, versions, backend, timing, resource use,
and QA. Raw-DTI QA fingerprints include the selected fitting compatibility
mode. FEM diagnostics include the SimNIBS distribution RECORD, the inspected
run/session/FEM/conductivity/mesh modules, numerical package versions, and
native solver-library hashes. Real FEM cache reuse is fail-closed even when
those hashes are present because the complete dynamic solver closure has not
been proven; deterministic dry-run preparation remains cacheable. A cache hit
is reported as reuse rather than fresh computation.
Reference and optimized backends remain explicit so numerical A/B can be
repeated after an optimization.

## Conductivity mapping

Diffusion tensors are sampled onto SimNIBS mesh elements. Basis correction,
positive-eigenvalue handling, maximum conductivity, and maximum anisotropy ratio
follow the SimNIBS 4.6 compatibility contract.

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
instead of silently presenting it as the stable result. Formal FEM workflow
runs use the installed SimNIBS conductivity path.
Nonzero rank-deficient VN tensors are rejected by default because determinant
normalization is undefined. The opt-in `regularize` policy first raises the
singular eigensystem to the configured anisotropy bound, then applies the
existing safety passes and records the repair count; it is a documented robust
extension rather than a literal SimNIBS 4.6 edge result. With default intensity
calibration, an all-zero anisotropic `dir`/`mc` tissue is rejected because the
global scale is undefined; explicit `--no-correct-intensity` accepts it through
the non-calibrated safety path.
