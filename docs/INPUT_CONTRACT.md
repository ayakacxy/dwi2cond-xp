# Input contract

## Preprocessed DWI

The DWI must already have motion, eddy-current, and susceptibility/EPI
distortion corrections appropriate for its acquisition. The b-vectors must have
been rotated consistently with motion correction. The DWI, brain mask, b-values,
and b-vectors must describe the same ordered volumes.

The DTI fit uses b=0 volumes and one nonzero shell. Multishell data must be
reduced explicitly with `select-shell` or an equivalent traceable step. The code
does not silently fit a single-tensor model to every nonzero shell.

The optional `grad_dev` NIfTI uses the HCP/FSL nine-component convention and
must match the DWI spatial shape and affine.

## Diffusion tensor

Tensor NIfTIs have shape `(X,Y,Z,6)` with final-axis ordering
`Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`. Values must be finite. A validity mask should be
carried forward from fitting or supplied by the external producer.

## T1 alignment

`register-tensor` applies a caller-supplied input-world to reference-world 4x4
affine and finite-strain tensor reorientation. It does not estimate that affine.
If external preprocessing already established a shared world coordinate system,
the caller may explicitly pass `--assume-aligned`.

Automatic FLIRT/FNIRT registration, nonlinear deformation, local Jacobians, and
PPD reorientation are outside the `v0.1.0` implementation. Do not use
`--assume-aligned` merely because DWI and T1 belong to the same participant;
same-subject acquisitions can still differ by motion, distortion, resolution,
orientation, and scanner coordinates.

## CHARM mask and output

The compatible brain mask follows the SimNIBS `dwi2cond` labeling interval
1 through 499. The final tensor is written to the T1 grid and placed at
`m2m_<subject>/DTI_coregT1_tensor.nii.gz` for SimNIBS integration.

Shape, affine, finite-value, valid-mask, and provenance failures are reported
explicitly. An anisotropic workflow never falls back silently to scalar FEM.
