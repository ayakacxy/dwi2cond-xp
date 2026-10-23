# Command examples

[English](../en/USAGE.md) | [简体中文](../zh/USAGE.md)

[Documentation](README.md) · [Project home](../../README.md)

Commands below assume installation is complete. Replace example file names with your inputs; output directories are created by the workflow. Input and coordinate requirements are described in [Inputs](INPUTS.md).

## Complete raw-DWI workflow

The default is legacy correction followed by nonlinear T1 registration. The tensor and its provenance are written to m2m after QA succeeds.

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

## Reverse-phase-encoding data

For a reverse-PE series, the pipeline prepares TOPUP and EDDY together. Readout time is in seconds; PE directions refer to the internal FSL axes.

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --preprocessing-mode eddy \
  --reverse-phase-encoding reverse_pe_4d.nii.gz \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

## Pre-fitted tensor

Import a tensor ordered `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz` and run the same T1 mapping and QA stages.

```bash
dwi2cond-xp run-prefit-pipeline \
  DTI_tensor.nii.gz m2m_subject workflow_outputs --workers 8
```

## Nomoco preparation

Use this path when motion, eddy-current and susceptibility correction are unnecessary or have already been handled externally.

```bash
dwi2cond-xp preprocess-nomoco \
  raw_dwi.nii.gz bvals bvecs nomoco_outputs \
  --workers 8
```

Outputs include tensor, FA, SSE, brain and validity masks, and `nomoco_qa.json`. The aligned b0 mean is used to build the reference; its transforms are not applied to diffusion volumes.

## Legacy motion/eddy correction

Run the two-pass correction with motion-aware gradient rotation:

```bash
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --workers 8 --bvec-mode corrected
```

Outputs include the corrected DWI, final volume transforms, nodif and mask, tensor maps and `legacy_qa.json`. `--bvec-mode compat46` omits motion rotation; add `--component-mode simnibs46-literal` only when comparing storage-component behavior with the upstream script.

## GRE fieldmap

Prepare an already unwrapped rad/s field and use its displacement with legacy correction:

```bash
dwi2cond-xp prepare-fieldmap \
  magnitude.nii.gz field_rads.nii.gz nodif_brain.nii.gz fieldmap_out \
  --dwell-ms 0.5 --phase-encoding-direction y- --workers 8
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --fieldmap-displacement fieldmap_out/displacement_world_mm.nii.gz \
  --fieldmap-corrected-mask fieldmap_out/corrected_mask.nii.gz --workers 8
```

Dwell time is in milliseconds. The output displacement uses signed world millimetres and accompanies its corrected mask. Wrapped phase needs external unwrapping first.

## Standalone TOPUP

Estimate the fixed reverse-PE field from matching b0 images:

```bash
dwi2cond-xp prepare-topup \
  forward_b0.nii.gz reverse_b0.nii.gz topup_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

The output contains a Hz field, spline coefficients, motion parameters, corrected images, mask and QA. `topup_fieldcoef.nii.gz` and `topup_movpar.txt` also form an FSL-readable prefix bundle. Supported PE axes are x and y, with either sign.

## Standalone EDDY

Run the fixed single-shell path with an optional prepared TOPUP field:

```bash
dwi2cond-xp prepare-eddy \
  raw_dwi.nii.gz bvals bvecs brain_mask.nii.gz eddy_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8 \
  --susceptibility-field topup_out/field_hz.nii.gz
```

Omit `--susceptibility-field` to run without TOPUP. Outputs include corrected and prediction-replaced DWI, 16 parameters per volume, rotated gradients, outlier/shell-alignment files, iteration histories and QA.

## Fit preprocessed DWI

Use a same-grid mask and gradients corresponding to the processed volumes. The gradient-deviation option is optional.

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz \
  --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz \
  --qa-json tensor_fit_qa.json
```

For multishell data, select one shell explicitly. Near-zero b-values can be included with `--b0-threshold`; see [Inputs](INPUTS.md).

## Apply an existing alignment

This example assumes an external process has already aligned DWI and T1 world coordinates:

```bash
dwi2cond-xp register-tensor \
  tensor_dwi.nii.gz m2m_subject/T1.nii.gz \
  m2m_subject/DTI_coregT1_tensor.nii.gz \
  --source-mask tensor_valid_mask.nii.gz \
  --assume-aligned \
  --qa-json tensor_registration_qa.json
```

Otherwise supply the input-world-to-reference-world affine with `--world-transform`, or estimate registration using the next command.

## Automatic affine registration

Estimate FA-to-T1 alignment and produce registered tensors and QA:

```bash
dwi2cond-xp register-t1 dti_outputs m2m_subject t1_registration_outputs \
  --mode affine --workers 8
```

## Fixed-montage FEM

Prepare a C3-to-C4 simulation without starting the solver:

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn \
  --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 \
  --cpus 8 --dry-run
```

Remove `--dry-run` to solve. Run `scalar`, `vn`, `dir` and `mc` in separate output directories. Each run exports a T1-grid vector NIfTI ordered `Ex,Ey,Ez`, masked to WM/GM/CSF labels 1, 2 and 3. Magnitude is `sqrt(Ex^2 + Ey^2 + Ez^2)`.

## Compare electric fields

Draw the four vector fields with a shared component scale:

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

Rows are `Ex/Ey/Ez`, columns are `scalar/vn/dir/mc`. The selected slice has the largest brain-mask area. See [SimNIBS integration](SIMNIBS_INTEGRATION.md) for lead-field output conventions.

## CPU and CUDA

CPU is the default. To use both optional CUDA backends for nonlinear registration:

```bash
dwi2cond-xp run-pipeline raw_dwi.nii.gz bvals bvecs m2m_subject workflow_gpu --fnirt-device cuda --ppd-device cuda --workers 8
```

FNIRT needs Numba CUDA; PPD additionally needs CuPy. The same options apply to `register-t1-nonlinear` and `run-prefit-pipeline`. LM/PCG and file writing remain on CPU. See [Reproducibility](REPRODUCIBILITY.md) for dependencies.
