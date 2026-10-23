# FSL reference comparisons

[English](en/FSL_REFERENCE.md) | [简体中文](zh/FSL_REFERENCE.md)

[Documentation](en/README.md) · [Project home](../README.md)

FSL 6.0.4 is the optional numerical reference for the fixed preprocessing and registration paths. It is installed separately and is not a runtime dependency of dwi2cond-xp.

## Source provenance

The reference scripts come from SimNIBS 4.6.0; its main script identifies dwi2cond version `0.4`. The recorded FSL build is `6.0.4:ddd0a010`. These source hashes identify the reference used in the retained comparisons:

| Script | SHA-256 |
| --- | --- |
| `dwi2cond` | `0978b26ab1c4d4d303b10151dbe4d0da8ed0c080c3d09af0855412c7b5c46340` |
| `dwi2cond.prepro.source.sh` | `4671e6f72b0dd0da496297e722df3e3682e659dab9c06502c83907fb287f8f78` |
| `dwi2cond.functions.source.sh` | `b133dd7e41ecac773d0af983bacb3f23b62fd2c91f84379ff81df42f89ee5932` |
| `dwi2cond.t1reg.source.sh` | `539d1951621111495792c0957705494fc592579a9c0f3dc807dd6549a904205d` |
| `dwi2cond.check.source.sh` | `29c3c907cd174bc628b3770feafb69831d4f4c76b41c91a97e1cfd938b655541` |

Use a local installation of the matching build when repeating these comparisons. Outputs from another FSL version need their own reference record.

## Stage artifact map

The harness uses the following aliases to identify inputs and results:

| Stage | Required public aliases |
| --- | --- |
| input normalization | `dwi_raw`, `bvals`, `bvecs` |
| b0 and mask | `nodif`, `nodif_brain`, `nodif_brain_mask`, `raw_fa`, `raw_sse` |
| `nomoco` handoff | `dwi_for_fit`, `fit_bvals`, `fit_bvecs`, `fit_mask` |
| tensor fit | `tensor`, `fa`, `sse` |
| T1 registration | `world_transform` or `warp`, `t1_tensor`, `registered_fa`, `registered_v1`, `registered_sse` |
| final QA | `brain_rim`, `first_eigenvector_mesh`, `run_log` |

NIfTI summaries record shape, dtype, affine, qform, sform, finite/nonzero counts and finite range. A named `grid_ref` can replace repeated identical geometry. Masks also record their voxel count.

## Execution semantics

Each command runs in its own process and working directory. Missing tools produce `status: skipped`; nonzero exit, timeout or missing required output produces `status: failed`. Manifests record tool version, source hashes, redacted command, thread count, runtime and resource use. Numerical thresholds are fixed before evaluating candidate outputs.

## Frozen public evidence

`tests/fixtures/reference/` contains synthetic definitions, thresholds and aggregate reference results:

- `synthetic_nomoco_manifest.json` and `synthetic_nomoco_reference.json`;
- `synthetic_legacy_reference.json`;
- `synthetic_b0_motion_manifest.json` and `synthetic_preprocessing_manifest.json`;
- GRE, TOPUP and EDDY reference JSON files;
- `synthetic_eddy_interspersed_b0_reference.json`;
- `synthetic_t1_registration_reference.json`;
- `private_nomoco_reference.json`, containing aggregate statistics only.

Image fixtures are generated when needed. Public records contain no raw voxel arrays, credentials or identifying subject data. The reference-asset checker can inspect a directory with:

```bash
PYTHONPATH=src python tools/audit_reference_assets.py tests/fixtures/reference
```

The measured agreement and current limitations are listed in [Validation](en/VALIDATION.md).
