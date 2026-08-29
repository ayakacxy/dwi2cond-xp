# Validation

This page records the evidence assembled for the historical `v0.1.0` release.
The tag was reissued on 2026-08-29 to correct documentation and release
metadata; the algorithm source remains the original `v0.1.0` baseline. No HCP,
FSL, FEM, or performance experiment was rerun for that documentation update.

## Evidence levels

The evidence has two distinct levels:

1. Tag-local automated contracts are encoded in the `v0.1.0` test suite and can
   be rerun from the source tree. The release record was `144 passed` with
   `1,644/1,644` executable package statements covered (`100.00%`) when the
   optional local FSL test was configured.
2. HCP, synthetic-sphere, and real FEM results below are retained historical
   experiment records. Their controlled/private inputs and subject-level
   machine-readable outputs are not distributed, so the repository alone
   cannot independently replay those exact comparisons.

The historical measurements support the stated inputs and stages. They are not
evidence for raw-DWI correction, automatic registration, other acquisitions,
other solvers, or later package versions.

## Automated contracts

Synthetic tests cover gradient conventions, WLS fitting, invalid-voxel
handling, affine tensor mapping and reorientation, conductivity modes, SimNIBS
object construction, WM/GM/CSF output masking, lead-field axes, plotting, and
manifests. FSL comparisons are optional reference tests and skip explicitly
when `dtifit` is unavailable.

The Linux, macOS, and Windows workflows associated with this release enforced
the same 100% package-statement threshold. Platform CI did not convert the
optional FSL comparison into a cross-platform FSL claim.

## DTI numerical references

A synthetic DTI fixture was compared with FSL 6.0.4 `dtifit --wls
--gradnonlin`. Tensor maximum and mean absolute differences were `1.16e-10` and
`8.26e-12`.

The private HCP comparison used b=0 plus the b=1000 shell, the same WLS and
gradient-nonlinearity contract, and 5,287,164 valid tensor values. Maximum,
mean, and p99 absolute differences were `2.43e-6`, `4.78e-10`, and `2.10e-9`;
relative L2 was `4.18e-6`. Rare boundary voxels dominated the maximum, so this
is a close whole-volume comparison rather than an elementwise-exactness claim.

The fitted HCP output contained 881,299 masked voxels: 881,194 valid voxels and
105 voxels that were zero-filled and reported by the validity/QA contract.

## Conductivity reference

On the retained synthetic sphere mesh, 274,770 conductivity components were
compared with SimNIBS 4.6. Maximum absolute difference was `0` for `vn` and
`2.22e-16` for `dir` and `mc`. This comparison covers the sampled tensors,
tissue labels, scalar references, weights, and mode parameters of that fixture;
it does not prove equivalence for every degenerate tensor or user-supplied
parameter combination.

## Fixed-montage FEM evidence

One private HCP subject had a complete T1/T2 CHARM head model and a T1-grid
tensor. Real C3-to-C4, 1 mA simulations completed with Pardiso for `scalar`,
`vn`, `dir`, and `mc`. The recorded vector E-field NIfTIs were finite. After
the version's explicit subject-volume mask, WM/GM/CSF voxels were populated and
voxels outside labels 1, 2, and 3 were zero.

A scalar Pardiso/Hypre comparison on the common original mapping had maximum
absolute difference `2.98e-8 V/m` and relative L2 `6.28e-9`. This is a
single-input scalar solver comparison; it is not a four-mode solver-equivalence
claim.

The package also provided and unit-tested the all-electrode lead-field
configuration and HDF5/NPY/JSON contracts. A full-subject, all-electrode
lead-field execution was not part of the `v0.1.0` evidence.

## Capability boundary

`v0.1.0` starts from externally preprocessed DWI or an externally produced
tensor. It does not correct raw-DWI motion, eddy currents, susceptibility/EPI
distortion, or estimate affine/nonlinear DTI-to-T1 registration. It implements
only caller-supplied affine mapping and affine finite-strain reorientation;
nonlinear PPD reorientation is absent.

No HCP source data, anatomical image, volumetric derivative, subject identifier,
or machine-readable subject artifact is included in the repository or Release.
