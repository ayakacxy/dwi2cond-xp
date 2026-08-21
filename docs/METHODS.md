# Methods

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
