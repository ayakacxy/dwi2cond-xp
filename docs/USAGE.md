# Command examples

[English](en/USAGE.md) | [简体中文](zh/USAGE.md)

[Documentation](en/README.md) · [Project home](../README.md)

Run the default path used by SimNIBS 4.6 (`legacy + nonlinear`) and publish a
validated tensor/provenance pair into the m2m directory. Recoverable output publication
failures restore the previous valid pair:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

For reverse-PE acquisition, use the complete TOPUP-to-EDDY workflow rather than
manually joining standalone artifacts:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --preprocessing-mode eddy \
  --reverse-phase-encoding reverse_pe_4d.nii.gz \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

An official-style pre-fitted tensor enters the same T1, output publication, and QA workflow:

```bash
dwi2cond-xp run-prefit-pipeline \
  DTI_tensor.nii.gz m2m_subject workflow_outputs --workers 8
```

Run the FSL-free SimNIBS 4.6 `nomoco` raw-DWI path with eight workers:

```bash
dwi2cond-xp preprocess-nomoco \
  raw_dwi.nii.gz bvals bvecs nomoco_outputs \
  --workers 8
```

This writes `DTI_tensor.nii.gz`, `DTI_FA.nii.gz`, `DTI_sse.nii.gz`, the brain
and validity masks, and `nomoco_qa.json`. It never creates motion, eddy, or
field artifacts. A compatible uncompressed `.nii` is verified blockwise and
used directly by mmap; `.nii.gz` is decoded once to a shared uncompressed
intermediate. The selected strategy is recorded in QA.

Run the legacy correction path with motion-aware gradient rotation (the v0.4.0 default):

```bash
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --workers 8 --bvec-mode corrected
```

This writes the corrected DWI and mean, every final volume transform, nodif and
brain mask, tensor/FA/SSE and validity outputs, and `legacy_qa.json`.
`--bvec-mode compat46` skips motion-induced gradient rotation for upstream
comparison. The default physical component mode still transforms the gradients
when storage axes change; add `--component-mode simnibs46-literal` to retain
the old component behavior as well. See [v0.4.0 migration](en/RELEASE_NOTES_v0.4.0.md).

Prepare a fixed GRE/FUGUE correction from an already scaled rad/s fieldmap and
compose its signed world-mm displacement into the legacy command:

```bash
dwi2cond-xp prepare-fieldmap \
  magnitude.nii.gz field_rads.nii.gz nodif_brain.nii.gz fieldmap_out \
  --dwell-ms 0.5 --phase-encoding-direction y- --workers 8
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --fieldmap-displacement fieldmap_out/displacement_world_mm.nii.gz \
  --fieldmap-corrected-mask fieldmap_out/corrected_mask.nii.gz --workers 8
```

Raw wrapped Siemens phase still requires PRELUDE and is rejected by this fixed
rad/s branch.

Estimate the fixed reverse-PE TOPUP field without an FSL runtime dependency:

```bash
dwi2cond-xp prepare-topup \
  forward_b0.nii.gz reverse_b0.nii.gz topup_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

This writes the field, spline coefficients, movement parameters, corrected
pair, joint validity mask, and `topup_qa.json`. The legacy project names are
retained, and `topup_fieldcoef.nii.gz` plus `topup_movpar.txt` form an
FSL-readable `--topup=topup` bundle with the official cubic-spline intent
metadata. Only the x/y single-axis subset accepted by FSL 6.0.4 TOPUP is
supported; z phase encoding fails explicitly.

Run the fixed single-shell EDDY path, optionally composing the TOPUP field:

```bash
dwi2cond-xp prepare-eddy \
  raw_dwi.nii.gz bvals bvecs brain_mask.nii.gz eddy_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8 \
  --susceptibility-field topup_out/field_hz.nii.gz
```

The command writes corrected DWI, prediction-replaced scan-space data, 16
parameters per volume, rotated b-vectors, outlier map, shell alignment,
iteration histories, and `eddy_qa.json`. Omitting `--susceptibility-field`
selects the same fixed EDDY algorithm without a TOPUP field.

Fit a preprocessed single-shell DWI:

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz \
  --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz \
  --qa-json tensor_fit_qa.json
```

Map the tensor to the CHARM T1 grid. The example assumes alignment only because
the alignment was established externally:

```bash
dwi2cond-xp register-tensor \
  tensor_dwi.nii.gz m2m_subject/T1.nii.gz \
  m2m_subject/DTI_coregT1_tensor.nii.gz \
  --source-mask tensor_valid_mask.nii.gz \
  --assume-aligned \
  --qa-json tensor_registration_qa.json
```

Validate a C3-to-C4 anisotropic simulation without starting the solver:

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn \
  --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 \
  --cpus 8 --dry-run
```

Remove `--dry-run` to execute SimNIBS. Run `scalar`, `vn`, `dir`, and `mc`
separately; each mode has its own output directory and manifest. An anisotropic
mode never silently falls back to `scalar` when its tensor is missing.

Each simulation exports one T1-grid vector E-field NIfTI with final axis
`Ex,Ey,Ez`. Magnitude is computed as `sqrt(Ex^2 + Ey^2 + Ez^2)`. Subject-volume
outputs are strictly masked to WM, GM, and CSF labels `1,2,3`; skull, scalp,
electrodes, and extracranial tissues are excluded.

## Automatic affine registration

Estimate the FA-to-T1 affine and create registered tensors and QA outputs:

```bash
dwi2cond-xp register-t1 dti_outputs m2m_subject t1_registration_outputs \
  --mode affine --workers 8
```

## Comparing electric fields

Use the four vector fields to draw a shared-scale component figure:

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

Rows are `Ex/Ey/Ez`, columns are `scalar/vn/dir/mc`, and all panels use one
symmetric color scale. The slice is selected by maximum brain-mask area.
