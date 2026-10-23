# Input files and coordinates

[English](en/INPUTS.md) | [简体中文](zh/INPUTS.md)

[Documentation](en/README.md) · [Project home](../README.md)

The pipeline accepts raw single-shell DWI, preprocessed DWI, or a fitted six-component tensor. Use the input path that matches the processing already performed on your data.

## Required files

| Input | Requirements |
| --- | --- |
| DWI | Finite 4-D NIfTI, with b-values and b-vectors in the same volume order |
| Gradients | One b-value and one vector per volume; b-vectors may be `3×N` or `N×3` |
| Brain mask | Same spatial shape and affine as the image it masks |
| Gradient deviation | Optional HCP/FSL nine-component `grad_dev` image on the original DWI grid |
| Fitted tensor | Finite `(X,Y,Z,6)` NIfTI ordered `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz` |
| Head model | SimNIBS 4.6 CHARM `m2m_<subject>` directory with T1 and segmentation outputs |

The tensor model fits exact b=0 volumes plus one nonzero shell. For multishell data, select a shell explicitly with `select-shell` or `fit-dti --shell`. Standalone `fit-dti --b0-threshold` can include near-zero b-values; raw-DWI pipeline commands use exact b=0 selection. These choices are independent of `strict-fsl` and `robust` fitting policies.

## Raw DWI without correction

`preprocess-nomoco` constructs an aligned b0 mean and brain mask, then fits the DWI. It does not apply the b0 transforms to the diffusion volumes or correct their motion, eddy currents or susceptibility distortion. Choose this path only when that behavior is appropriate for the acquisition.

Storage reorientation follows the FSL/NewNifti float32 Gram–Schmidt and signed-permutation rules. Physical component handling also transforms b-vectors and column-major `grad_dev` matrices into the reoriented FSL basis. The nonnegative signal threshold is applied at the final fitting boundary.

A canonical, uncompressed float32 `.nii` with finite, nonnegative values can be read directly by mmap after blockwise verification. Other inputs, including `.nii.gz`, are normalized once to `DWIforfit.nii`. The chosen strategy is recorded in `nomoco_qa.json`.

## Motion and distortion correction

### Legacy correction

`preprocess-legacy` uses two 6/12-DOF correction passes, aligns the corrected diffusion mean to nodif, and substitutes direct 6-DOF b0 registrations. It composes the final transforms and performs one final sinc interpolation per original volume.

The default `--bvec-mode corrected` applies the finite-strain rotation of each final motion affine in the FSL scaled-mm basis, then normalizes the nonzero-shell vectors. `compat46` omits motion rotation. Storage-axis changes still transform vectors unless `--component-mode simnibs46-literal` is selected or the storage transform is the identity.

### GRE fieldmap

`prepare-fieldmap` accepts a magnitude image, an already unwrapped field in radians per second and a b0 brain image. Supply `--dwell-ms` and the phase-encoding direction. Optional masks must match their respective image grids. Wrapped Siemens phase requires external unwrapping before this command.

The prepared displacement has shape `(X,Y,Z,3)`, uses signed NIfTI-world millimetres and must match the DWI grid. Pass it to `preprocess-legacy --fieldmap-displacement` together with its corrected mask. The complete pipeline can prepare both outputs from the GRE bundle.

### Reverse-PE TOPUP and EDDY

`prepare-topup` takes two finite 3-D b0 images with identical shape and affine. Readout time must be positive. Supported PE directions are `x`, `x-`, `y` and `y-`, matching the fixed FSL 6.0.4 TOPUP subset.

`prepare-eddy` takes a DWI, matching gradients and a same-grid brain mask. It accepts one nonzero shell and positive readout time. Without TOPUP, EDDY also accepts `z` and `z-`. An optional susceptibility field must be finite, measured in Hz and on the DWI grid.

For a reverse-PE 4-D series, `run-pipeline` performs storage reorientation, rigid alignment and averaging, estimates TOPUP, creates the EDDY mask from the first corrected b0, and passes the TOPUP products into EDDY. This combined path supports x/y PE directions. Without reverse PE, the pipeline constructs an exact-b0 mean and BET `f=0.2` mask. An explicit `--dwi-brain-mask` overrides that mask and is recorded in the stage manifest.

PE flags name FSL internal axes after storage reorientation. Original NIfTI/BIDS `i/j/k` directions are not converted automatically. See [command examples](en/USAGE.md).

## Preprocessed DWI and fitted tensors

For `fit-dti`, supply data with the motion and distortion corrections needed for its acquisition. Its gradients must include the corresponding motion rotations. Keep the DWI, gradients and mask aligned throughout preprocessing.

`run-prefit-pipeline` imports the six-component tensor, performs storage reorientation and tensor decomposition, and joins the T1 registration and QA workflow. A fitting validity mask should accompany the tensor when available. Physical component handling preserves tensor directions during storage changes.

## Coordinates and compatibility options

| Option | Default | Comparison behavior |
| --- | --- | --- |
| `--bvec-mode` | `corrected` | `compat46` omits motion-induced rotation |
| `--component-mode` | `physical` | `simnibs46-literal` keeps storage-only component handling |
| `--jacobian-mode` | `physical` | `simnibs46-literal` uses per-voxel PPD derivatives |
| `--basis-mode` | `physical` | `simnibs46-literal` uses the upstream affine normalization |
| `--eigensystem-mode` | `stable` | `simnibs46-literal` uses the upstream general eigensolver |

These options are independent. `strict-fsl`, the default validity policy, reports invalid fits and unsafe nonlinear transforms as errors where required. Opt-in `robust` excludes invalid fit voxels and zero-fills unsafe nonlinear locations, recording them in QA. Neither policy changes the coordinate options.

## T1 alignment

`register-t1` estimates an affine from `DTI_FA.nii.gz` and maps `DTI_tensor.nii.gz` to T1. It requires CHARM `T1.nii.gz`, `segmentation/labeling.nii.gz`, `segmentation/T1_bias_corrected.nii.gz` and `final_tissues.nii.gz`. Optional `DTI_sse.nii.gz` adds SSE QA. Default affine registration uses 12 DOF and an independent 6-DOF QA transform; rigid mode uses 6 DOF.

`register-tensor` applies a supplied input-world-to-reference-world 4×4 affine; it does not estimate one. Use `--assume-aligned` only after external processing has established shared world coordinates. Images from the same subject are not necessarily aligned.

`register-t1-nonlinear` runs the fixed FNIRT schedule and PPD reorientation. A brain mask is required and applied after PPD. It writes coefficients, deformation, Jacobians, tensor/FA/V1, validity mask and QA. Folded or near-singular locations are handled by the selected validity policy.

FSL sampling uses header `pixdim`; world coordinates use the active affine. Affine and dense-warp paths support different scales in these fields. Raw preprocessing and FNIRT estimation/coefficient expansion require the scales to agree. Resample mismatched inputs to an explicit grid before using those paths.

Physical PPD uses millimetre displacement derivatives with target voxel sizes. Fold QA uses the physical Jacobian in both PPD modes. The physical conductivity basis normalizes affine columns separately and rejects shear.

## CHARM outputs and conductivity

T1 workflows require `final_tissues.nii.gz`, including runs with FEM disabled. The final tensor uses the T1 grid and is written to `m2m_<subject>/DTI_coregT1_tensor.nii.gz`. The diffusion brain mask follows the CHARM labeling interval 1–499.

FEM additionally needs an EEG cap and a resolved head mesh. If the standard subject mesh name is absent and exactly one mesh is present, the workflow passes that mesh explicitly to SimNIBS. See [SimNIBS integration](en/SIMNIBS_INTEGRATION.md).

For `tensor-to-mesh`, nonzero singular VN tensors raise an error by default. `--vn-singular-policy regularize` explicitly repairs their spectrum before the safety passes and records the count. This option applies only to VN. Conductivity policies and equations are described in [Conductivity models](en/CONDUCTIVITY.md).
