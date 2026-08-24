# SimNIBS Discussion Post Draft

Recommended category: **5. Show and tell**

Suggested title:

> [Community project] dwi2cond-xp: cross-platform, FSL-free-at-runtime workflows for SimNIBS 4.6

## Post body

Hi SimNIBS maintainers and community,

I would like to share **dwi2cond-xp**, an independent community project that
implements the fixed diffusion preprocessing subset used by the SimNIBS 4.6
`dwi2cond` workflow in Python, without requiring FSL at runtime.

- Repository: <https://github.com/ayakacxy/dwi2cond-xp>
- Latest stable release: <https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.2.0>

This project is not an official SimNIBS or FSL component, and it is not intended
to be a general replacement for FSL. Its scope is deliberately limited to the
preprocessing, tensor mapping, conductivity, and simulation contracts required
by the SimNIBS 4.6 workflow.

## Motivation

The main goal is to make the supported DTI-to-conductivity workflow easier to
install and run across Linux, macOS, and Windows, particularly in environments
where adding a complete FSL installation is difficult.

Scientific compatibility is the primary objective. Performance optimizations
are accepted only when they preserve the corresponding SimNIBS 4.6/FSL 6.0.4
algorithm, image resolution, iteration schedule, stopping conditions,
coordinate conventions, tensor ordering, and output contracts.

FSL remains useful as an optional numerical reference during development and
validation, but it is not called by the released runtime workflow.

## Implemented scope

Version `0.2.0` includes:

- preprocessing of supported raw or already-preprocessed single-shell
  diffusion MRI;
- two-pass weighted least-squares DTI fitting with optional
  gradient-nonlinearity correction;
- the SimNIBS 4.6 `nomoco` path;
- the fixed legacy motion/eddy-current correction path;
- the fixed GRE fieldmap path when an already unwrapped radians-per-second
  fieldmap is supplied;
- the fixed SimNIBS TOPUP configuration;
- the fixed single-shell EDDY subset, including prediction-based `--repol` and
  optional TOPUP field support;
- automatic 6/12-DOF linear DTI-to-T1 registration;
- the fixed SimNIBS 4.6 FNIRT configuration;
- affine and nonlinear tensor reorientation, including local-Jacobian PPD;
- SimNIBS-compatible six-component tensor output;
- `scalar`, `vn`, `dir`, and `mc` conductivity modes;
- fixed-montage anisotropic FEM workflows using SimNIBS 4.6;
- lead-field configuration and export interfaces; and
- structured QA, provenance manifests, cache validation, atomic workflow
  stages, and progress reporting.

The final tensor follows the SimNIBS convention and can be written to:

`m2m_<subject>/DTI_coregT1_tensor.nii.gz`

The complete processing path is approximately:

```text
raw or preprocessed single-shell DWI, or a six-component diffusion tensor
  -> selected SimNIBS 4.6 preprocessing branch
  -> WLS DTI fitting
  -> affine or nonlinear tensor mapping to the CHARM T1 grid
  -> PPD tensor reorientation
  -> scalar / vn / dir / mc conductivity
  -> SimNIBS FEM
  -> vector E-field NIfTI and QA manifests
```

## Validation summary

The release includes unit tests, synthetic end-to-end fixtures, optional FSL
reference comparisons, and private whole-head integration evidence.

### Automated testing

- Linux release gate: `535 passed, 7 skipped`
- Statement coverage: `12,443 / 12,443` executable statements (`100.00%`)
- The same coverage threshold passes on Ubuntu 22.04, macOS 14 arm64, and
  Windows Server 2022.
- Package construction, isolated wheel installation, CLI checks, Markdown
  checks, version checks, CodeQL, OpenSSF Scorecard, dependency auditing,
  archive privacy auditing, SBOM generation, and published-asset checksum
  verification are included in the release process.

### DTI fitting

For a small synthetic fixture compared with FSL 6.0.4
`dtifit --wls --gradnonlin`:

- maximum tensor absolute difference: `1.16e-10`
- mean tensor absolute difference: `8.26e-12`

For one private HCP b0+b1000 input:

- valid tensor values compared: `5,287,164`
- relative L2 error: `4.18e-6`
- mean absolute difference: `4.78e-10`
- p99 absolute difference: `2.10e-9`
- maximum absolute difference: `2.43e-6`

Rare boundary voxels dominate the maximum difference, so whole-head
elementwise or bitwise equality is not claimed.

### TOPUP and EDDY fixtures

For the public fixed-path synthetic fixtures:

- TOPUP field relative L2 error: `0.00430`
- TOPUP corrected-pair relative L2 error: `0.00161`
- EDDY corrected-DWI relative L2 error inside the mask: `0.0035352`
- downstream EDDY tensor relative L2 error: `0.0060714`
- all four injected bad slices were detected, with no additional false
  detections in that fixture

### Nonlinear registration and PPD

For one private whole-head nonlinear comparison:

- deformation-vector mean error: `0.0130 mm`
- deformation-vector p99 error: `0.0626 mm`
- deformation-vector maximum error: `0.1828 mm`
- tensor relative L2 error: `0.00631`
- tensor-support Dice: `0.99962`
- V1 axial-angle mean: `0.343 degrees`
- V1 axial-angle p99: `3.104 degrees`

When the same FSL deformation field was supplied to both implementations, the
Python PPD tensor relative L2 error decreased to `1.51e-6`. This separates most
of the remaining endpoint difference from the tensor-reorientation
implementation.

Bitwise equality with an independently optimized FNIRT endpoint is not claimed.

### Conductivity and FEM

On a synthetic mesh containing `274,770` conductivity components, agreement
with SimNIBS 4.6 was:

- `vn`: maximum absolute error `0`
- `dir`: maximum absolute error `2.22e-16`
- `mc`: maximum absolute error `2.22e-16`

A private whole-head C3-to-C4, 1 mA FEM workflow completed successfully for
`scalar`, `vn`, `dir`, and `mc`.

All formal vector E-field NIfTIs were finite. WM, GM, and CSF voxels were
populated, while voxels outside the selected tissue contract were zeroed.

The scalar Pardiso/Hypre comparison on the common mapping had:

- maximum absolute difference: `2.98e-8 V/m`
- relative L2 error: `6.28e-9`

More detailed validation boundaries are documented in the
[validation report](https://github.com/ayakacxy/dwi2cond-xp/blob/main/docs/VALIDATION.md).

## Performance evidence

One same-server, same-input, same-output-boundary DTI fitting comparison
measured:

- dwi2cond-xp, 16 workers: `9.76 s`
- FSL 6.0.4: `108.23 s`
- observed ratio for this DTI-fitting stage: `11.09x`

This result applies only to that DTI-fitting stage and system. It is not
presented as a universal end-to-end preprocessing or FEM speedup.

The current roadmap prioritizes same-input, same-output, eight-worker
end-to-end measurements before making broader performance claims:

<https://github.com/ayakacxy/dwi2cond-xp/blob/main/docs/ROADMAP.md>

## Current limitations

The project intentionally has a narrower contract than general FSL:

- it implements only the fixed behavior required by the supported SimNIBS 4.6
  workflow;
- it is not a general implementation of `flirt`, `fnirt`, `topup`, `eddy`,
  `fugue`, or other FSL commands;
- arbitrary TOPUP, EDDY, and FNIRT configurations are outside the supported
  contract;
- the GRE path expects an already unwrapped radians-per-second fieldmap and
  does not implement PRELUDE;
- silent ordinary-DTI fitting of multishell data is not supported;
- whole-head real-data validation currently relies primarily on one private HCP
  subject, so population-level generalization is not claimed;
- no anatomical image, subject identifier, voxel derivative, or
  machine-readable subject-level artifact is distributed;
- the full-subject, all-electrode lead-field interface is tested at the API and
  data-contract level, but it is not yet included in the completed
  whole-subject evidence; and
- FEM and conductivity workflows require exactly SimNIBS 4.6.0 and remain
  limited to platforms where that version of SimNIBS can be installed and
  validated.

## Feedback requested

I would greatly appreciate feedback from the SimNIBS maintainers and community
on the following questions:

1. Have I interpreted the relevant SimNIBS 4.6 tensor ordering,
   coordinate-system, registration, PPD reorientation, and conductivity
   contracts correctly?
2. Are there additional small, legally redistributable fixtures or edge cases
   that would be particularly useful for compatibility testing?
3. Which compatibility risks should receive the highest priority when
   evaluating later SimNIBS versions?
4. Would this be useful as an optional community workflow for users who need a
   cross-platform installation without a runtime FSL dependency?
5. If the maintainers consider the project useful, would it eventually be
   appropriate to mention it in a community-resources or related-tools
   section, while keeping its independent and unsupported status explicit?

I am not asking the SimNIBS team to support or endorse the project at this
stage. The immediate goal is to invite technical review, identify incorrect
assumptions, and improve compatibility with the official SimNIBS workflow.

## Links

- Repository: <https://github.com/ayakacxy/dwi2cond-xp>
- v0.2.0 release: <https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.2.0>
- Documentation index: <https://github.com/ayakacxy/dwi2cond-xp/blob/main/docs/README.md>
- Methods: <https://github.com/ayakacxy/dwi2cond-xp/blob/main/docs/METHODS.md>
- Validation: <https://github.com/ayakacxy/dwi2cond-xp/blob/main/docs/VALIDATION.md>
- Reproducibility: <https://github.com/ayakacxy/dwi2cond-xp/blob/main/docs/REPRODUCIBILITY.md>
- Roadmap: <https://github.com/ayakacxy/dwi2cond-xp/blob/main/docs/ROADMAP.md>
- Citation metadata: <https://github.com/ayakacxy/dwi2cond-xp/blob/main/CITATION.cff>
- License: <https://github.com/ayakacxy/dwi2cond-xp/blob/main/LICENSE>

Thank you to the SimNIBS developers and community for making the software and
documentation available. I would be very interested in any technical
corrections or suggestions.
