# v0.4.0 release notes

[English](../en/RELEASE_NOTES_v0.4.0.md) | [简体中文](../zh/RELEASE_NOTES_v0.4.0.md)

[Documentation](README.md) · [Project home](../../README.md) · [Changelog](CHANGELOG.md)

Release date: 2026-10-14.

This release corrects gradient and tensor-coordinate handling and accelerates preprocessing, registration and QA. CPU remains the default; optional CUDA backends accelerate FNIRT Hessian assembly and PPD.

## Algorithm corrections

- Legacy motion correction rotates gradients in the same FSL basis as the final motion transforms.
- Storage reordering transforms b-vectors, tensor components and column-major gradient-deviation matrices with the image axes.
- GRE, TOPUP and EDDY consistently handle positive- and negative-determinant image storage at the internal radiological-grid boundary.
- Affine and dense-warp sampling distinguish header `pixdim` from the world affine. Paths requiring matching scales report an error for a mismatch.
- PPD and fold QA use the physical Jacobian with target voxel sizes.
- Conductivity mapping normalizes affine columns separately and applies the selected basis and eigensystem to formal SimNIBS consumers.
- Windows supports writing a NIfTI back to its input file or hard link.
- External-field QA uses the prepared field, and final QA completes before publishing the m2m tensor and provenance.

## Performance changes

Raw preparation and WLS reuse invariant work and reduce large-array copies. Affine and rigid QA registration share read-only pyramids. FNIRT shares spline traversals and uses CSR for its LM working matrix, while retaining the CPU PCG solver. PPD, decomposition and QA process bounded batches.

Optional CUDA performs FP64 Hessian accumulation and batched PPD sampling, Jacobians, eigendecomposition and rotation. LM/PCG and file writing remain on CPU.

## Migration from v0.3.0

| Setting | Default | Explicit comparison option |
| --- | --- | --- |
| Motion rotation of gradients | `--bvec-mode corrected` | `--bvec-mode compat46` |
| Storage components | `--component-mode physical` | `--component-mode simnibs46-literal` |
| PPD Jacobian | `--jacobian-mode physical` | `--jacobian-mode simnibs46-literal` |
| Tensor-to-world basis | `--basis-mode physical` | `--basis-mode simnibs46-literal` |
| Conductivity eigensystem | `--eigensystem-mode stable` | `--eigensystem-mode simnibs46-literal` |
| FNIRT Hessian device | `--fnirt-device cpu` | `--fnirt-device cuda` |
| PPD device | `--ppd-device cpu` | `--ppd-device cuda` |

The stable eigensystem was already the default; this release also applies it to formal SimNIBS consumers. Corrected coordinate handling intentionally changes affected outputs.

`compat46` omits motion rotation only. Storage changes still transform gradients in physical component mode. Literal PPD retains physical fold QA. The comparison options are independent of `strict-fsl`/`robust` validity policies and do not disable the GRE/TOPUP/EDDY or publication fixes. See [Inputs](INPUTS.md) for the full conventions.

## CPU and optional GPU use

Install the wheel in Python 3.11; SimNIBS workflows require SimNIBS 4.6.0. [Installation](../../README.md#-installation) covers existing and new environments.

The default pipeline runs on CPU. To request both CUDA backends:

```bash
dwi2cond-xp run-pipeline raw_dwi.nii.gz bvals bvecs m2m_subject workflow_gpu --fnirt-device cuda --ppd-device cuda --workers 8
```

FNIRT requires Numba CUDA; PPD additionally requires CuPy. Optional GPU libraries are not installed with the base package. These device options also apply to `register-t1-nonlinear` and `run-prefit-pipeline`.

## Measured complete-run performance

On Linux with an eight-core Xeon Gold 6230 budget and RTX 2080 Ti, one warm-cache HCP single-shell run took **205.452 s on CPU** and **163.342 s on CUDA**. Host RSS peaks were **6.498 GiB** and **6.723 GiB**, with **8.834 GiB** GPU allocation for the CUDA run.

The run included gradient-deviation correction, `nomoco`, nonlinear T1 registration, QA and m2m publication, with FEM off. It did not time the default legacy preprocessing path. [Benchmarks](BENCHMARKS.md) gives the complete boundary, input, numerical differences and comparison with FSL and the previous release.

## Release validation

Linux checks completed **960 tests with no skips**, including configured real FSL and SimNIBS probes, and covered **15,364/15,364 statements**. CPU/CUDA comparison had 61 of 64 artifacts byte-identical and brain-tensor relative L2 `1.8693e-9`. Platform results and numerical reference cases are documented in [Validation](VALIDATION.md).
