# Methods

This document describes the implementation shipped as `v0.1.0`. Later
compatibility modes and preprocessing stages are not part of this historical
version.

## DTI fitting

The fitter explicitly selects b=0 volumes and one nonzero shell. For gradient
vector $g=(g_x,g_y,g_z)$ and b-value $b$, the six tensor columns of the design
matrix are

$$
b\,[g_x^2,\,2g_xg_y,\,2g_xg_z,\,g_y^2,\,2g_yg_z,\,g_z^2],
$$

followed by the intercept column used to recover $S_0$. When a nine-component
HCP/FSL gradient-deviation field $L$ is supplied, the voxel-specific vector is
$h=(I+L)g$ and the same quadratic design is formed from $h$.

`v0.1.0` performs two linear solves. Positive measurements use $S^2$ as their
WLS weight in both solves; a nonpositive placeholder has weight 1. The first
solve estimates $S_0$; the second replaces nonpositive or very small
measurements by $0.01S_0$ before taking the logarithm.
This is the version's fixed fitting contract and was compared with FSL 6.0.4
`dtifit --wls --gradnonlin`. There is no public `strict-fsl`/`robust` fitting
mode switch in `v0.1.0`.

At the NIfTI layer, a masked voxel is excluded if any selected measurement is
nonfinite or if no selected measurement is positive. Its tensor and derived
maps are zero-filled, and the exclusion is recorded in the validity mask and QA
JSON. The written derivatives are tensor, FA, MD, MO, L1-L3, V1-V3, S0, SSE,
and the validity mask.

## Affine tensor mapping

The caller supplies a 4x4 transform from input world coordinates to reference
world coordinates, or explicitly declares that an external workflow already
established alignment. The implementation does not estimate either affine or
nonlinear registration.

Each of the six tensor components is sampled on the reference grid with the
inverse voxel map required by SciPy. If the transform's linear part has singular
value decomposition $U\Sigma V^T$, its orthogonal polar factor is $R=UV^T$.
The sampled tensor is reoriented by

$$
D_{T1}=R D_{DWI} R^T.
$$

This is affine finite-strain reorientation. Nonlinear deformation, local
Jacobians, and preservation-of-principal-direction reorientation are outside
`v0.1.0`.

## Conductivity mapping

For every selected anisotropic tissue, the sampled diffusion tensor is
diagonalized as $D_i=Q_i\operatorname{diag}(\lambda_i)Q_i^T$. Tetrahedron
volume is used as the default mesh weight. Nonselected tissues receive their
fixed scalar conductivity $\sigma_t I$.

All three anisotropic modes apply the version's eigenvalue safety rules: a
maximum eigenvalue, a maximum anisotropy ratio, and optional eccentricity
scaling. Exact zero tensors are first replaced by $\sigma_t I$ and are therefore
accepted by the calibrated and uncalibrated paths.

### `vn`

`vn` removes each tensor's local geometric-mean magnitude, applies the safety
rules, renormalizes, and multiplies by the tissue reference conductivity. Away
from an active safety bound this gives

$$
C_i=\sigma_t\,Q_i
\operatorname{diag}\!\left(
\frac{\lambda_i}{(\lambda_{i1}\lambda_{i2}\lambda_{i3})^{1/3}}
\right)Q_i^T,
\qquad \det(C_i)=\sigma_t^3.
$$

The directions and anisotropy ratio are retained subject to the configured
ratio, conductivity, and eccentricity bounds.

### `dir`

With the default intensity correction, `dir` first applies the broad
eigenvalue/ratio safety pass, then computes a single global scale across every
selected anisotropic tissue, not one scale per tissue. Let $\widetilde D_i$
denote that repaired tensor and

$$
m_t=\frac{\sum_{i\in t}w_i\det(\widetilde D_i)}{\sum_{i\in t}w_i},
\qquad r_t=m_t^{1/3}.
$$

The shared scale is

$$
a=\frac{\sum_t\sigma_t r_t}{\sum_t r_t^2},
$$

and the output is formed from the repaired eigenvalues of
$a\widetilde D_i$. Thus `dir` preserves local magnitude variation as well as
direction and anisotropy, while calibrating the complete selected anisotropic
set to the scalar tissue references. The default calibrated path requires every
aggregate determinant used by the scale to be positive and finite.

Passing `--no-correct-intensity` deliberately bypasses this global calibration.
That path still applies the per-tensor safety rules but leaves the surviving
input magnitude in its original scale. In `v0.1.0`, a nonzero
negative-semidefinite tensor can make the default aggregate scale undefined and
raise an error, whereas the uncalibrated safety path replaces it by the tissue
reference.

### `mc`

`mc` uses the same global scale $a$ as `dir`, then makes every local tensor
isotropic while preserving its post-scale determinant. If the repaired scaled
eigenvalues are $\mu_{i1},\mu_{i2},\mu_{i3}$, then

$$
C_i=(\mu_{i1}\mu_{i2}\mu_{i3})^{1/3}I.
$$

It is therefore a spatial-intensity control, not a directional anisotropy
model. `--no-correct-intensity` has the same uncalibrated meaning and safety
boundary as for `dir`.

## FEM boundary

The package supplies the resulting scalar or tensor conductivity to the same
SimNIBS 4.6 FEM assembly. Changing conductivity mode does not replace the
solver or alter the montage definition. Simulation orchestration is separate
from the upstream DWI preprocessing and registration responsibilities described
in the [input contract](INPUT_CONTRACT.md).
