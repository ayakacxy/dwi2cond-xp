# v0.4.0 release notes

[English](../en/RELEASE_NOTES_v0.4.0.md) | [简体中文](../zh/RELEASE_NOTES_v0.4.0.md)

[Documentation](README.md) · [Project home](../../README.md) · [Changelog](CHANGELOG.md)

Release date: 2026-10-14.

v0.4.0 corrects the handling of gradients, tensor components and physical
coordinates, then accelerates raw preparation, T1 registration, FNIRT, PPD
and QA. The default workflow remains usable entirely on CPU. Optional CUDA
backends accelerate FNIRT Hessian assembly and PPD without introducing FSL
as a runtime dependency.

## Algorithm corrections

- **Gradients and storage axes.** Legacy correction rotates gradients in the
  same FSL basis as the final motion transforms. Storage reordering transforms
  b-vectors, six-component tensors and column-major gradient-deviation matrices
  with the image axes, preserving their physical directions.
- **GRE, TOPUP and EDDY handedness.** Positive-determinant image storage now
  enters the required internal radiological coordinates consistently. Outputs
  return to their image storage without a second reflection of movement/EC
  parameters or FSL b-vectors.
- **Windows NIfTI writes.** Writing back to the input file or its hard link
  now works on Windows.
- **NIfTI sampling scales.** Affine and dense-warp FSL sampling use header
  `pixdim` separately from the world affine. Raw/FNIRT paths that cannot
  represent a header/affine voxel-size mismatch reject it instead of guessing.
- **Physical PPD.** Tensor reorientation and fold/near-singular QA use the same
  physical Jacobian, including the actual target voxel sizes. The former
  per-voxel derivatives could give the wrong rotation on a non-1-mm grid.
- **Conductivity directions.** The physical basis removes scale from each
  affine column separately and checks orthogonality. This avoids false
  anisotropy on oblique, unequal-voxel-size grids; shear is rejected. Formal
  SimNIBS FEM and lead-field instances now use the selected basis and stable
  symmetric eigensystem through a local adapter, without editing SimNIBS.
- **QA and publication.** Prepared external Hz fields are checked on the
  working grid. Final QA completes before the tensor and provenance are
  published into the m2m directory; a failed attempt preserves the previously
  accepted outputs.

These changes include both implementation defects and deliberate corrections
to inherited coordinate conventions. Literal comparison options are retained;
matching an older upstream output is not sufficient evidence that its physical
orientation is correct for every grid.

## Performance changes

Raw preparation streams eligible canonical float32 gzip inputs, reuses
invariant work and reduces temporary volume and coordinate arrays. WLS avoids
rebuilding shared normal matrices and retains process workers. The raw-QA fit
and formal nonnegative-threshold fit remain separate.

T1 affine and independent rigid QA registrations share a read-only image
pyramid. Fine correlation-ratio levels use adaptive FP64 block statistics;
searches, coarse-level sampling, iteration limits and stopping rules remain.

FNIRT shares spline-support traversal across components and global projections,
avoids unused dense bases and affine-field temporaries, and uses CSR for the
LM working matrix with the existing CPU PCG solver. Physical PPD avoids
analytic spline Jacobians that it does not consume, while retaining the
complete displacement and its finite-difference Jacobian. PPD and tensor
decomposition use bounded batches. QA and output writing reduce repeated reads
and full-volume temporaries, while keeping content hashes, all-value finite
checks and scientific QA.

CUDA is selected explicitly. FNIRT Hessian assembly uses FP64 accumulation;
PPD keeps source tensors on the GPU and batches sampling, Jacobians,
eigendecomposition and rotation. Existing float32 sampling and coefficient
checkpoints remain; Jacobian and matrix operations use FP64. LM and PCG stay
on CPU. Floating-point reductions may have bounded differences; scientific
logic and numerical stability are the acceptance criteria.

## Migration from v0.3.0

| Setting | v0.4.0 default | Comparison option |
| --- | --- | --- |
| Legacy motion rotation of gradients | `--bvec-mode corrected` | `--bvec-mode compat46` |
| Components during storage reordering | `--component-mode physical` | `--component-mode simnibs46-literal` |
| PPD reorientation Jacobian | `--jacobian-mode physical` | `--jacobian-mode simnibs46-literal` |
| Tensor-to-world basis | `--basis-mode physical` | `--basis-mode simnibs46-literal` |
| Conductivity eigensystem | `--eigensystem-mode stable` | `--eigensystem-mode simnibs46-literal` |
| FNIRT Hessian device | `--fnirt-device cpu` | `--fnirt-device cuda` |
| PPD device | `--ppd-device cpu` | `--ppd-device cuda` |

Affected outputs intentionally differ from v0.3.0. The stable symmetric
conductivity eigensystem was already the default in v0.3.0; v0.4.0 extends
its selected mode to formal SimNIBS consumers together with the physical basis.

`compat46` disables only motion-induced gradient rotation. With physical
components, storage-axis changes still transform gradients; byte-copying the
original b-vectors requires literal components or an identity axis transform.
Fold/near-singular QA continues to use the physical Jacobian even in literal
PPD mode. The four literal modes are independent and do not disable GRE/TOPUP/EDDY or
publication fixes. An upstream-coordinate comparison can select all five
options explicitly:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_comparison \
  --bvec-mode compat46 \
  --component-mode simnibs46-literal \
  --jacobian-mode simnibs46-literal \
  --basis-mode simnibs46-literal \
  --eigensystem-mode simnibs46-literal --workers 8
```

`strict-fsl`/`robust` fitting and invalid-position policies remain independent
of these direction modes. Phase-encoding `x/y/z` names refer to the FSL
internal axes after storage reordering; original NIfTI/BIDS `i/j/k` axes are
not converted automatically. Cache identities include the changed source and
selected policies, so older stage results are not reused as new computations.
See [Input files and coordinates](INPUTS.md) and [Methods](METHODS.md).

## CPU and optional GPU use

The complete workflow was tested with Python 3.11 and SimNIBS 4.6.0. For an
existing validated, frozen SimNIBS environment, install this checkout locally:

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps -e .
dwi2cond-xp --help
```

After release artifacts are available, the wheel installation command is
`python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl`.
`--no-deps` assumes the validated dependencies are already installed. For a
separate environment, follow [Installation](../../README.md#-installation)
and the pinned [environment file](../../environment.yml).

The default legacy + nonlinear CPU workflow is:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_cpu \
  --fnirt-device cpu --ppd-device cpu --workers 8
```

The same workflow with both CUDA backends is:

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_gpu \
  --fnirt-device cuda --ppd-device cuda --workers 8
```

CPU execution needs neither a GPU nor CuPy. The FNIRT CUDA backend requires a
working Numba CUDA runtime, NVIDIA driver and compatible CUDA toolkit; PPD
also requires compatible CuPy. The recorded GPU run used CuPy 14.2.0 with
CUDA 12.8. Optional GPU dependencies are not installed by the base package,
and CUDA failures stop the run without silently switching to CPU.
These device switches also apply to `register-t1-nonlinear` and
`run-prefit-pipeline`. Derived maps and NIfTI writing remain on CPU.

`nomoco` does not correct motion, eddy currents or susceptibility distortion;
use it only when appropriate for the acquisition. The timed HCP case used
`--preprocessing-mode nomoco --t1-mode nonlinear --grad-dev grad_dev.nii.gz`
with both fits, all project QA, m2m publication and FEM off. It does not time
the default legacy path shown above. See [Usage](USAGE.md) for acquisition-specific
preprocessing choices.

## Measured complete-run performance

One private HCP single-shell input was run on Linux with an Intel Xeon Gold
6230, an eight-physical-core budget and an NVIDIA RTX 2080 Ti. DWI was
`145x174x145x108` at 1.25 mm (18 exact b=0 plus 90 b≈1000 volumes), with
identical gradient-nonlinearity data and the same `260x311x260`, 0.7-mm T1 seed.

| Implementation | Complete wall time | Peak process-tree host RSS | Peak GPU allocation |
| --- | ---: | ---: | ---: |
| v0.4.0 CPU | 205.452 s | 6.498 GiB | none |
| v0.4.0 GPU | 163.342 s | 6.723 GiB | 8.834 GiB |
| Published v0.3.0 CPU | 564.406 s | 25.469 GiB | none |
| SimNIBS 4.6 / FSL 6.0.4 official subscripts | 1290.073 s | 3.794 GiB | none |

Relative to published v0.3.0, the observed wall-time ratios were **2.747x on
CPU** and **3.455x on GPU**, with time reduced by 63.599% and 71.060%.
Host RSS decreased by 74.487% and 73.601%. Relative to FSL, the observed ratios
were **6.279x / 7.898x**; FSL used less host memory. The v0.4.0 CPU/GPU host
RSS peaks were 1.713/1.772 times FSL's, and GPU memory is additional.

Each Python result is one new-process, fresh-output warm-cache sample with
zero JIT cache saves and no computation-stage artifact cache hits. FSL is one
completed native run. Wall time includes startup, imports, input hashes,
device initialization, computation, I/O, QA and publication; seed copying and
subsequent comparison are outside it. OS caches were not cleared. Host RSS
was sampled across the process tree and may count shared pages repeatedly;
GPU allocation includes context and allocator caches. These are sampled peaks,
not minimum memory requirements.

The FSL driver used the official subscripts with native `--gradnonlin` in both
WLS fits, omitted visual mesh export/FEM, and wrote fewer project diagnostics.
Native FSL often used one core within the same eight-core budget. Cross-version
ratios also include algorithm corrections. These workflow observations do not
establish pure hardware acceleration or final FSL numerical equivalence.
The [benchmark page](BENCHMARKS.md) preserves stage times, archived release
identity and historical measurements.

## Release validation

Local release checks passed **960 unique tests with no skips** and covered
**15,364/15,364 statements (100.00%)**, including three real synthetic
TOPUP/EDDY/FNIRT CLI runs. Test groups and numerical checks are documented
in the [validation record](VALIDATION.md).

## Numerical checks and scope

The full CPU run succeeded with `cupy`, `cupyx` and optional GPU backend
imports blocked. CPU/GPU had 61 of 64 numerical artifacts byte-identical;
geometry, masks, FNIRT coefficients/fields and recorded LM results agreed.
Brain tensor relative L2 was `1.8693e-9`, final FA maximum absolute difference
`8.7172e-7`, and healthy V1 axial-angle P99 `1.2074e-6` degrees. This direction
statistic requires both FA>0.1 and relative principal-eigenvalue gap>1e-5,
ignores eigenvector sign, and had complete healthy-voxel coverage on this input.

Algorithm regressions cover signed storage permutations, both handednesses,
non-1-mm PPD, header/affine scale mismatch, formal conductivity consumers and
publication failure. Stage-level checks include real FSL consumers.

The [validation record](VALIDATION.md) describes the scope of this evidence.
Supported preprocessing and the single-shell model are documented in
[Methods](METHODS.md) and [FSL reference](FSL_REFERENCE.md).
