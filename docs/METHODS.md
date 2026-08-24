# Methods

## Fixed preprocessing subset

The raw-DWI commands reproduce only the FSL behavior used by SimNIBS 4.6
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

Nonfinite measurements and voxels with no positive selected measurement are
excluded. Their tensor is set to zero and their status is recorded in a validity
mask and QA JSON. The default derivatives are tensor, FA, MD, MO, L1-L3, V1-V3,
S0, SSE, and the validity mask.

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
and manifests atomically, validates cached structure before reuse, and records
parameters, versions, backend, timing, resource use, and QA. A cache hit is
reported as reuse rather than fresh computation. Reference and optimized
backends remain explicit so numerical A/B can be repeated after an optimization.

## Conductivity mapping

Diffusion tensors are sampled onto SimNIBS mesh elements. Basis correction,
positive-eigenvalue handling, maximum conductivity, and maximum anisotropy ratio
follow the SimNIBS 4.6 compatibility contract.

- `vn` normalizes each tensor determinant to the scalar tissue conductivity.
- `dir` preserves local tensor magnitude and calibrates its tissue-level scale.
- `mc` replaces each directly scaled tensor by an isotropic tensor whose
  eigenvalue is the geometric mean of the three local conductivity
  eigenvalues, preserving the determinant and DTI-driven spatial variation.

The same SimNIBS FEM assembly is used for scalar and tensor conductivity.
