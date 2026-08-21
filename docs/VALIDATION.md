# Validation

## Automated contracts

Synthetic tests cover gradient conventions, weighted fitting, invalid voxels,
tensor registration and reorientation, conductivity modes, SimNIBS object
construction, strict tissue masking, lead-field axes, plotting, and manifests.
FSL comparisons are optional reference tests and must skip visibly when FSL is
not configured.

The current local release suite reports `144 passed` and covers all
`1,644/1,644` executable package statements (`100.00%`) with FSL `dtifit`
configured. Linux, macOS, and Windows CI enforce the same 100% coverage gate;
the FSL reference case skips explicitly on runners without `dtifit`.

## Numerical references

A synthetic DTI fixture was compared with FSL 6.0.4 `dtifit --wls
--gradnonlin`; tensor maximum and mean absolute differences were `1.16e-10` and
`8.26e-12`. On private HCP b0+b1000 data, the tensor comparison over 5,287,164
valid values had maximum, mean, and p99 absolute differences
`2.43e-6`, `4.78e-10`, and `2.10e-9`, with relative L2 `4.18e-6`. Rare boundary
voxels dominate the maximum, so whole-head elementwise exactness is not claimed.

On a synthetic sphere mesh, 274,770 conductivity components agreed with
SimNIBS 4.6 to maximum absolute error `0` for `vn` and `2.22e-16` for `dir` and
`mc`.

## End-to-end evidence

A private HCP subject was processed through a complete T1/T2 CHARM head model,
T1-grid tensor, and real C3-to-C4 1 mA FEM for `scalar`, `vn`, `dir`, and `mc`
using Pardiso. All formal vector E-field NIfTIs were finite. Every WM/GM/CSF
voxel was populated and every voxel outside those labels was zero after the
strict mask.

The scalar Pardiso/Hypre comparison on the common original mapping had maximum
absolute difference `2.98e-8 V/m` and relative L2 `6.28e-9`.

## Evidence boundary

No HCP data or subject derivative is public. Full lead-field execution is not
part of the current end-to-end claim. Raw-DWI preprocessing and automatic
registration are not validated capabilities of this package.
