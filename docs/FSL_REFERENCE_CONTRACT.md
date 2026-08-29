# SimNIBS 4.6 FSL reference contract

This document freezes the FSL reference boundary used by the final `v0.3.0`
validation. FSL remains an optional validation dependency and is not imported,
bundled, or called by the released runtime path.

## Source provenance

The compatibility source is the copy installed with SimNIBS 4.6.0. The main
script declares `dwi2cond` version `0.4`.

| Script | SHA-256 |
| --- | --- |
| `dwi2cond` | `0978b26ab1c4d4d303b10151dbe4d0da8ed0c080c3d09af0855412c7b5c46340` |
| `dwi2cond.prepro.source.sh` | `4671e6f72b0dd0da496297e722df3e3682e659dab9c06502c83907fb287f8f78` |
| `dwi2cond.functions.source.sh` | `b133dd7e41ecac773d0af983bacb3f23b62fd2c91f84379ff81df42f89ee5932` |
| `dwi2cond.t1reg.source.sh` | `539d1951621111495792c0957705494fc592579a9c0f3dc807dd6549a904205d` |
| `dwi2cond.check.source.sh` | `29c3c907cd174bc628b3770feafb69831d4f4c76b41c91a97e1cfd938b655541` |

These hashes identify the local validation source only. The files are not
copied into this repository. The first real fixture must additionally record
the exact FSL build used for its outputs; outputs from different FSL releases
must not share one numerical baseline.

The recorded release-reference installation was `/usr/local/fsl` and reported
FSL `6.0.4:ddd0a010`. This absolute path is local configuration and must be
replaced by an alias in public run manifests.

## Stage artifact map

The harness records artifacts by the public aliases below. A private manifest
may additionally contain file or voxel digests, but a public manifest must not
contain absolute paths, subject identifiers, or voxel arrays.

| Stage | Required public aliases |
| --- | --- |
| input normalization | `dwi_raw`, `bvals`, `bvecs` |
| b0 and mask | `nodif`, `nodif_brain`, `nodif_brain_mask`, `raw_fa`, `raw_sse` |
| `nomoco` handoff | `dwi_for_fit`, `fit_bvals`, `fit_bvecs`, `fit_mask` |
| tensor fit | `tensor`, `fa`, `sse` |
| T1 registration | `world_transform` or `warp`, `t1_tensor`, `registered_fa`, `registered_v1`, `registered_sse` |
| final QA | `brain_rim`, `first_eigenvector_mesh`, `run_log` |

Every NIfTI summary includes shape, dtype, affine, qform, sform, finite count,
nonzero count, and finite range. A frozen compact manifest may replace repeated
identical affine/qform/sform fields with a named `grid_ref`. Mask artifacts
additionally include mask count. A completed command with a missing required
artifact is a failed stage.

## Execution semantics

- Each reference command runs in an independent process and an explicit
  working directory.
- A missing executable produces a manifest with `status: skipped`; it is never
  counted as a passing comparison.
- Non-zero exit, timeout, missing required output, or an output escaping the
  working directory produces `status: failed` and never selects another mode.
- Manifests record reference version, script hashes, redacted command and log
  summaries, environment variable names, thread count, wall time, child CPU
  time, and peak child RSS where the platform exposes it.
- Numerical thresholds are frozen in the fixture manifest before implementing
  the corresponding Python algorithm.

The initial public threshold file is
`tests/fixtures/reference/synthetic_nomoco_manifest.json`. The threshold file
alone is a contract rather than a pass result; its companion reference summary
and the final configured real-FSL gate provide the execution evidence.

## Frozen public evidence

The public reference foundation is frozen in eleven JSON files under
`tests/fixtures/reference/`:

- `synthetic_nomoco_manifest.json` freezes comparison thresholds;
- `synthetic_nomoco_reference.json` records the completed public reference;
- `synthetic_legacy_reference.json` records the completed two-pass legacy
  output, per-volume transform, interpolation, b-vector, and timing A/B;
- `private_nomoco_reference.json` contains only the aggregate summary of the
  completed private single-shell reference;
- `synthetic_b0_motion_manifest.json` freezes rigid-motion truth and hashes;
- `synthetic_preprocessing_manifest.json` freezes reverse-PE, fieldmap, affine,
  and tensor fixture contracts and hashes;
- `synthetic_fieldmap_reference.json` records the fixed GRE/FUGUE reference;
- `synthetic_topup_reference.json` records the fixed TOPUP reference;
- `synthetic_eddy_reference.json` records the fixed EDDY `--repol` reference;
- `synthetic_eddy_interspersed_b0_reference.json` records the portable
  interspersed-b0 EDDY fixture contract and same-platform reference provenance;
- `synthetic_t1_registration_reference.json` records automatic T1 registration.

The image fixtures are generated on demand and are not stored in the public
reference directory. Audit all frozen evidence with:

```bash
PYTHONPATH=src python tools/audit_reference_assets.py \
  tests/fixtures/reference --forbid 100610 --forbid hcp --forbid cxy
```

The audit rejects image files, absolute paths, direct identifier/credential
keys, raw voxel arrays, and configured private terms.
