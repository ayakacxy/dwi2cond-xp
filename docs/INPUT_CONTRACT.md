# Input contract

## Raw DWI without correction

`preprocess-nomoco` implements the SimNIBS 4.6 `nomoco` path. It reorients the
NIfTI storage with the FSL/NewNifti float32 Gram-Schmidt, signed-permutation,
and strict tie-breaking rules, registers only the exact `b=0` volumes to construct their mean
and brain mask, applies the official nonnegative threshold only at the final
fit boundary, and then runs WLS fitting. It does not apply the estimated b0 transforms to the DWI and
does not estimate motion, eddy-current, or susceptibility fields.

The DWI, b-values, and b-vectors must describe the same ordered volumes. The
strict single-tensor model accepts every volume only after validating one
nonzero shell; multishell data is never selected or fitted silently. Optional `grad_dev` must match the original DWI grid
and is storage-reoriented with the DWI before fitting.

An uncompressed `.nii` is used directly only after blockwise verification that
it is float32, in the required FSL storage orientation, finite, and nonnegative.
Otherwise, including for every `.nii.gz` input, normalization performs one
decode and writes one uncompressed `DWIforfit.nii` for process-shared mmap. The
chosen strategy and whether an intermediate was materialized are recorded in
`nomoco_qa.json`.

## Raw DWI with SimNIBS legacy correction

`preprocess-legacy` accepts the same ordered DWI, b-values, b-vectors, optional
gradient-deviation image, and explicit single-shell fitting parameters. For
SimNIBS 4.6 compatibility, b0 selection and `b>0` means use exact zero and
strictly positive b-values rather than a configurable near-zero threshold.

The pipeline runs the two MCFLIRT-compatible 6/12-DOF passes, registers the
corrected diffusion mean to nodif, replaces b0 transforms with direct 6-DOF
registrations, composes all transforms, and performs one final sinc resampling
per original volume. Temporary resampling used to construct the two reference
means is not reused as formal output. Failure aborts the legacy mode and never
falls back to `nomoco`.

`--bvec-mode compat46` is the default and byte-copies the original b-vector
file, matching the SimNIBS 4.6 legacy script. `--bvec-mode corrected` is a
separate scientific mode that rotates nonzero-shell vectors with the finite-
strain rotation of each final affine and restores unit length. These modes are
never selected implicitly.

An optional prepared `--fieldmap-displacement` must be an `(X,Y,Z,3)` finite displacement
on the DWI grid, expressed in moving-world millimetres. It is composed with the
affine pull coordinates before the single final interpolation and must be paired
with its corrected brain mask. The complete workflow can instead accept the
magnitude, already-unwrapped radians-per-second field, dwell, and PE direction
as one raw GRE bundle and prepare both artifacts in the same cached DAG.

`prepare-fieldmap <magnitude> <field_radians_per_second> <b0_brain>
<output_directory>` implements the SimNIBS 4.6 GRE/FUGUE branch for an already
scaled rad/s fieldmap. `--dwell-ms` and
`--phase-encoding-direction {x,x-,y,y-,z,z-}` are mandatory. Optional magnitude
and b0 masks must match their respective grids. The voxel-shift output is in
voxels; `displacement_world_mm.nii.gz` is the signed NIfTI-world pull field for
`preprocess-legacy --fieldmap-displacement`. Raw wrapped Siemens phase is not
accepted because PRELUDE unwrapping is outside this fixed branch.

`prepare-topup <forward_b0> <reverse_b0> <output_directory>` implements the
fixed SimNIBS 4.6 `b02b0_nosubsamp.cnf` path. Both inputs must be finite 3-D
NIfTIs with identical shape and affine. `--readout-seconds` must be positive,
and `--phase-encoding-direction` is limited to `x`, `x-`, `y`, or `y-` because
FSL 6.0.4 TOPUP rejects z phase encoding. The command writes the Hz field,
float32 spline coefficients, six movement parameters per scan, corrected pair,
joint mask, and QA JSON. Any failure aborts; no alternative field estimator is
selected silently.

`prepare-eddy <dwi> <bvals> <bvecs> <brain_mask> <output_directory>` implements
the fixed SimNIBS 4.6 single-shell EDDY path. The DWI and mask must share shape
and affine; b-values and either 3xN or Nx3 b-vectors must match the ordered volumes. The
nonzero b-values must form one shell, `--readout-seconds` must be positive, and
phase encoding may be `x`, `x-`, `y`, `y-`, `z`, or `z-`. An optional
`--susceptibility-field` is a finite Hz NIfTI on the DWI grid. The command uses
prediction-based slice replacement by default, records the deterministic random
seed, writes corrected and outlier-free DWI, 16 parameters per volume, rotated
b-vectors, outlier and shell-alignment artifacts, iteration histories, and QA.
`--no-repol` and `--no-rigid-shell-alignment` are explicit fixed-path controls;
no failed optimized path silently falls back.

When `run-pipeline` receives a reverse-PE 4-D image, it storage-reorients the
series, performs the official rigid alignment and temporal mean, estimates
TOPUP, creates the EDDY mask from TOPUP's first corrected b0 with the official
BET fraction, and passes the complete TOPUP products into EDDY. Because FSL
6.0.4 TOPUP rejects z phase encoding, this combined branch remains limited to
x/y directions even though no-TOPUP EDDY accepts z.

Without reverse PE, `run-pipeline --preprocessing-mode eddy` constructs the
canonical exact-b0 aligned nodif and BET `f=0.2` mask inside the raw-DWI DAG.
An explicitly supplied `--dwi-brain-mask` remains an extension and its source
is recorded in the stage manifest; it is never silently substituted.

## Preprocessed DWI

The DWI must already have motion, eddy-current, and susceptibility/EPI
distortion corrections appropriate for its acquisition. The b-vectors must have
been rotated consistently with motion correction. The DWI, brain mask, b-values,
and b-vectors must describe the same ordered volumes.

The strict DTI fit uses every supplied volume after validating exact b0 plus one
nonzero shell. Multishell data must be reduced explicitly with `select-shell`
or an equivalent traceable step before this boundary. The code does not silently
select b1000 or fit a single-tensor model to multiple nonzero shells.

The optional `grad_dev` NIfTI uses the HCP/FSL nine-component convention and
must match the DWI spatial shape and affine.

## Diffusion tensor

Tensor NIfTIs have shape `(X,Y,Z,6)` with final-axis ordering
`Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`. Values must be finite. A validity mask should be
carried forward from fitting or supplied by the external producer.

`run-prefit-pipeline` accepts an official-style pre-fitted tensor, performs the
storage-only reorientation and `fslmaths -tensor_decomp` semantics, then joins
the same T1 registration, publication, and QA DAG without requiring raw DWI.

## Compatibility modes

`strict-fsl` is the default for fitting and nonlinear PPD. It preserves the
FSL-compatible input handling and fails explicitly where FSL itself aborts or
where a fold/singular active-support transform has no valid tensor result.
`robust` is a separate opt-in policy that excludes invalid fit voxels and
zero-fills unsafe nonlinear locations. The selected mode is recorded in stage
parameters and is never changed through silent fallback.

## T1 alignment

`register-t1 <dti_directory> <m2m_directory> <output_directory>` automatically
reproduces the SimNIBS 4.6 linear T1-registration closure. `--mode affine`
(default) estimates the 12-DOF FA-to-T1 transform and a separate 6-DOF QA
transform; `--mode rigid` uses 6 DOF. The command requires `DTI_tensor.nii.gz`
and `DTI_FA.nii.gz`, plus CHARM `T1.nii.gz`, `segmentation/labeling.nii.gz`,
`segmentation/T1_bias_corrected.nii.gz`, and `final_tissues.nii.gz`. Optional
`DTI_sse.nii.gz` adds SSE QA. Registration failure aborts and never implies
alignment or falls back to `--assume-aligned`.

`register-tensor` applies a caller-supplied input-world to reference-world 4x4
affine and finite-strain tensor reorientation. It does not estimate that affine.
If external preprocessing already established a shared world coordinate system,
the caller may explicitly pass `--assume-aligned`.

`register-t1-nonlinear <fa> <tensor> <reference> <affine_matrix>
<output_directory>` runs the fixed SimNIBS 4.6 FNIRT subset and nonlinear PPD
tensor reorientation. `--brain-mask` is mandatory and is applied after PPD,
matching the official T1 support contract. The command writes the spline coefficients, dense deformation,
local Jacobians, registered tensor/FA/V1, validity mask, and QA. Folded or
near-singular regions are reported explicitly, and failure never falls back to
an affine result. Do not use `--assume-aligned` merely because DWI and T1 belong
to the same participant;
same-subject acquisitions can still differ by motion, distortion, resolution,
orientation, and scanner coordinates.

## CHARM mask and output

The compatible brain mask follows the SimNIBS `dwi2cond` labeling interval
1 through 499. The final tensor is written to the T1 grid and placed at
`m2m_<subject>/DTI_coregT1_tensor.nii.gz` for SimNIBS integration.

Every T1 workflow requires `final_tissues.nii.gz`, even when FEM execution is
disabled, matching the official CHARM sentinel. FEM additionally requires a
directory named `m2m_<subject>`, an EEG 10-10 cap, and a resolved head mesh.
When the standard `<subject>.msh` name is unavailable and exactly one mesh is
present, that resolved mesh is passed explicitly as `SESSION.fnamehead`; dry-run
validation and the real SimNIBS preparation path therefore consume the same
mesh.

For public `tensor-to-mesh` VN conversion, a nonzero singular tensor is rejected
by default. `--vn-singular-policy regularize` explicitly projects its
eigenvalues to the configured anisotropy ratio before the ordinary safety
passes and records `regularized_singular_tensors` in the QA JSON. Supplying this
VN-only option with `dir` or `mc` is rejected as an unused-input error.

Shape, affine, finite-value, valid-mask, and provenance failures are reported
explicitly. An anisotropic workflow never falls back silently to scalar FEM.
