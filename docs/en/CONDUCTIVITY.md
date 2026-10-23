# Conductivity models

[English](../en/CONDUCTIVITY.md) | [简体中文](../zh/CONDUCTIVITY.md)

[Documentation](README.md) · [Project home](../../README.md)

| Mode | Meaning |
| --- | --- |
| `scalar` | Fixed scalar conductivity per tissue |
| `vn` | Local determinant normalization; retains tensor direction and anisotropy |
| `dir` | One global scale across anisotropic tissues; retains local direction and intensity variation |
| `mc` | Same global scale as `dir`, with locally isotropic tensors |

The equations below use tissue reference conductivities and diffusion tensors sampled onto the target voxels or mesh elements.

## Implemented equations

The equations below are the mappings implemented by this project and matched
against SimNIBS 4.6. For element or voxel $i$ in anisotropic tissue $t$, write
the repaired diffusion tensor as

$$
\mathbf D_i = \mathbf V_i\mathrm{diag}
(d_{i1},d_{i2},d_{i3})\mathbf V_i^{\mathsf T},
\qquad d_{i1}\ge d_{i2}\ge d_{i3}>0,
$$

where $\mathbf V_i$ contains the principal directions, $\sigma_t$ is the
reference scalar conductivity of tissue $t$, and $w_i$ is the tetrahedron
volume (or one for an unweighted voxel calculation).

### `vn`: volume-normalized anisotropic mapping

Define the local geometric mean $g_i=(d_{i1}d_{i2}d_{i3})^{1/3}$. The core
mapping is

$$
\boldsymbol\Sigma_i^{\mathrm{vn}}
=\sigma_t\mathbf V_i
\mathrm{diag}\left(
\frac{d_{i1}}{g_i},\frac{d_{i2}}{g_i},\frac{d_{i3}}{g_i}
\right)\mathbf V_i^{\mathsf T},
\qquad
\det\left(\boldsymbol\Sigma_i^{\mathrm{vn}}\right)^{1/3}=\sigma_t.
$$

Thus `vn` preserves eigenvectors and relative anisotropy while setting the
geometric mean conductivity locally to the tissue reference. This is the
volume-normalized construction described by Güllmar et al. and used as the
recommended anisotropic mapping in SimNIBS [3,4].

### `dir`: directly scaled anisotropic mapping

For every anisotropic tissue, first compute the volume-weighted tensor scale

$$
m_t=
\left(
\frac{\sum_{i\in t}w_i\det(\mathbf D_i)}
     {\sum_{i\in t}w_i}
\right)^{1/3}.
$$

One global factor is then fitted jointly across the anisotropic tissues:

$$
s=\underset{a}{\mathrm{argmin}}
\sum_t(am_t-\sigma_t)^2
=\frac{\sum_t\sigma_t m_t}{\sum_t m_t^2},
\qquad
\boldsymbol\Sigma_i^{\mathrm{dir}}=s\mathbf D_i.
$$

This retains DTI-driven direction, anisotropy, and spatial magnitude. It is the
linear diffusion-to-conductivity family introduced by Tuch et al. and used in
the direct mappings described by Rullmann et al. and Opitz et al. [1,2,4].

### `mc`: mean-conductivity control

`mc` uses the same fitted factor $s$ as `dir`, but replaces every local tensor
by an isotropic tensor with the same determinant:

$$
\boldsymbol\Sigma_i^{\mathrm{mc}}
=\det\left(\boldsymbol\Sigma_i^{\mathrm{dir}}\right)^{1/3}\mathbf I
=s\det(\mathbf D_i)^{1/3}\mathbf I.
$$

It therefore preserves the DTI-driven spatial variation in geometric-mean
conductivity while removing directional anisotropy. `mc` is a DTI-derived
control mode, not an anisotropic tensor field [4].

## Safety and coordinate settings

The mappings apply the configured eigenvalue, maximum-conductivity and anisotropy-ratio rules. Non-anisotropic tissues use the scalar reference conductivity. VN normalizes, applies safety correction, renormalizes and applies safety correction again; a bound-triggered final correction can slightly change the ideal determinant equality.

A nonzero singular VN tensor has no defined determinant normalization and raises an error by default. `tensor-to-mesh --vn-singular-policy regularize` explicitly projects its spectrum to the configured anisotropy bound before the ordinary safety passes. QA records `regularized_singular_tensors`. This VN-only option is rejected for `dir` or `mc`.

With intensity calibration enabled, all-zero anisotropic `dir`/`mc` tissue raises an error because a global scale cannot be fitted. Explicit `--no-correct-intensity` uses the non-calibrated safety path.

The default `--eigensystem-mode stable` uses symmetric `eigh`. `simnibs46-literal` uses general `eig`, including the upstream repeated-eigenvalue behavior; QA records basis orthogonality. The default `--basis-mode physical` normalizes each spatial affine column separately and rejects shear. Its literal option preserves the upstream normalization for comparison.

Formal FEM and lead-field sessions use these selected settings through an instance-local adapter. The installed SimNIBS modules are unchanged. See [SimNIBS integration](SIMNIBS_INTEGRATION.md) for the consumer formats.

References:

1. Tuch DS, Wedeen VJ, Dale AM, George JS, Belliveau JW. *Conductivity tensor
   mapping of the human brain using diffusion tensor MRI*. PNAS. 2001;
   98(20):11697-11701. [doi:10.1073/pnas.171473898](https://doi.org/10.1073/pnas.171473898)
2. Rullmann M, Anwander A, Dannhauer M, Warfield SK, Duffy FH, Wolters CH.
   *EEG source analysis of epileptiform activity using a 1 mm anisotropic
   hexahedra finite element head model*. NeuroImage. 2009;44(2):399-410.
   [doi:10.1016/j.neuroimage.2008.09.009](https://doi.org/10.1016/j.neuroimage.2008.09.009)
3. Güllmar D, Haueisen J, Reichenbach JR. *Influence of anisotropic electrical
   conductivity in white matter tissue on the EEG/MEG forward and inverse
   solution. A high-resolution whole head simulation study*. NeuroImage.
   2010;51(1):145-163.
   [doi:10.1016/j.neuroimage.2010.02.014](https://doi.org/10.1016/j.neuroimage.2010.02.014)
4. Opitz A, Windhoff M, Heidemann RM, Turner R, Thielscher A. *How the brain
   tissue shapes the electric field induced by transcranial magnetic
   stimulation*. NeuroImage. 2011;58(3):849-859.
   [doi:10.1016/j.neuroimage.2011.06.069](https://doi.org/10.1016/j.neuroimage.2011.06.069)

The corresponding SimNIBS implementation-level definitions are summarized in
the official [dwi2cond documentation](https://simnibs.github.io/simnibs/build/html/documentation/command_line/dwi2cond.html).
