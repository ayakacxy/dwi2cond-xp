# Methods

[English](../en/METHODS.md) | [简体中文](../zh/METHODS.md)

[Documentation](README.md) · [Project home](../../README.md)

This document describes the algorithm implemented by the historical `v0.2.0`
source. It intentionally does not import policies added by later releases.
Subsequent workflow and numerical corrections were released in
`v0.3.0`, which is recommended for new deployments. The evidence supporting
each part of this implementation is separated in [Validation](VALIDATION.md).

## Fixed preprocessing subset

The raw-DWI commands implement only the FSL behavior used by SimNIBS 4.6
`dwi2cond`. `preprocess-nomoco` aligns b0 volumes to construct the fitting
reference and mask; it does not apply motion, eddy-current, or susceptibility
correction to the diffusion-weighted volumes. `preprocess-legacy` follows the
two-pass 6/12-DOF correction order and performs one formal sinc resampling for
each original volume.

The GRE path accepts an already unwrapped radians-per-second fieldmap and
implements the fixed FLIRT/FUGUE composition. The TOPUP path uses the fixed
nine-level `b02b0_nosubsamp.cnf` schedule. The EDDY path implements the fixed
SimNIBS 4.6 single-shell motion and quadratic eddy-current model, spherical-GP
prediction, prediction-based slice replacement, b-vector rotation, and an
optional TOPUP field. These interfaces are a version-specific subset, not
general replacements for the corresponding FSL commands.

## DTI fitting

The fitter selects baseline volumes and one diffusion shell using the configured
b-value thresholds. By default, it includes volumes with b <= 50 s/mm² or
|b - 1000| <= 100 s/mm². It constructs the tensor design matrix and performs two
weighted least-squares passes. Both passes use squared measured signal as the
weight for positive measurements and weight 1 for nonpositive measurements.
The first pass estimates S0; the second applies a signal floor of 1% of S0
before taking logarithms while retaining the same weights. Optional gradient-
nonlinearity coefficients adjust gradient directions and b-values per voxel.

Voxels containing a nonfinite selected measurement or no positive selected
measurement are excluded, zero-filled, and recorded in the validity mask and
QA JSON. There is no public `strict-fsl`/`robust` mode switch in this version.
The default derivatives are tensor, FA, MD, MO,
L1--L3, V1--V3, S0, SSE, and the validity mask.

## Tensor mapping and T1 registration

For an external affine, the caller supplies the input-world to reference-world
transform or explicitly declares prior alignment. Resampling uses the inverse
voxel mapping required by SciPy. The linear part of the world transform is
reduced to its orthogonal polar factor, and tensors are reoriented by
`R D R^T` (finite-strain reorientation). Processing is z-blocked to avoid a
whole-head multi-gigabyte 3x3 expansion.

Automatic linear registration follows the SimNIBS 4.6 6/12-DOF schedule and
retains an independent 6-DOF QA result. The nonlinear path uses the fixed
four-level FNIRT schedule, evaluates the local deformation Jacobian, and
applies preservation of principal direction (PPD) to the tensor. The reported
agreement is fixture- and version-specific; coefficient or optimizer-path
bitwise identity with FSL is not guaranteed by v0.2.

## Workflow and provenance

The v0.2 workflow records parameters, versions, timing, resource use, and QA;
fingerprints inputs with SHA-256; writes stage manifests atomically; and checks
cached output structure before reuse. A cache hit is reported as reuse rather
than fresh computation. These are the v0.2 implementation semantics, not a
claim that later audit findings were already fixed in this historical source.

## Conductivity mapping

Diffusion tensors are sampled onto SimNIBS mesh elements. The implementation
applies its SimNIBS 4.6 basis correction, substitutes the tissue reference
conductivity for an all-zero sampled tensor, and enforces the configured
maximum conductivity and anisotropy ratio.

- `vn` normalizes each local tensor determinant to the scalar conductivity of
  its tissue, with the v0.2 two-pass eigenvalue safety correction.
- With intensity correction enabled, `dir` preserves local tensor magnitude
  and fits one global scale jointly across all selected anisotropic tissues.
- `mc` uses the same direct scale, then replaces each local tensor by an
  isotropic tensor whose eigenvalue is the geometric mean of its three scaled
  eigenvalues. It preserves DTI-driven spatial magnitude variation while
  removing directional anisotropy.

The full equations are given in the [project README](../../README.md). The same SimNIBS
FEM assembly is used for scalar and tensor conductivity.
