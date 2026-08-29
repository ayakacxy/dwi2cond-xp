# Historical benchmarks

This page preserves stage-level measurements collected while the
`v0.1.0`--`v0.3.0` implementation was being developed. Most timings predate
the final `v0.3.0` correctness remediation and release rebuild. They were not
rerun from the final `v0.3.0` tag or release wheel, so they describe the timed
implementation snapshots, fixtures, and hosts below rather than current-tag
performance.

No benchmark was rerun for this documentation update. Two later `v0.4.0`
planning samples were collected while server load was unstable and are
excluded from all tables and comparisons on this page.

Unless stated otherwise, the reference was the SimNIBS 4.6 workflow with FSL
6.0.4 on the same host and input. Ratios compare only the paired rows in one
table. The fixtures, process boundaries, worker counts, warm-cache state, and
output sets differ between sections, so timings must not be added across
stages or treated as an end-to-end pipeline result.

## Raw-DWI preprocessing

### `nomoco` closure on HCP b0+b1000

The private input contained 108 volumes on a `145x174x145` grid. Both
implementations used the same b0-normalized b-values and b-vectors. The Python
CLI used eight worker processes with one BLAS thread per worker; the reference
used the unmodified SimNIBS 4.6 `--prepro --nomoco --keepstuff` path.

| Implementation and input | Execution | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| dwi2cond-xp, validated `.nii` mmap | 8 workers | 18.82 s | 306% | 1,770,256 KiB |
| dwi2cond-xp, `.nii.gz` single decode | 8 workers | 52.92 s | 170% | 4,791,436 KiB |
| SimNIBS 4.6 / FSL 6.0.4 | 1 process | 257.30 s | 99% | 3,167,907,840 bytes |

The observed wall-time ratios were `13.67x` for the validated uncompressed
input and `4.86x` for gzip input. The uncompressed path required a canonical,
finite, nonnegative float32 NIfTI and scanned it in z-blocks before mmap use.
The gzip path decoded once to an uncompressed fitting image and then shared
that image among workers. The two Python paths produced bitwise-identical b0,
mask, brain, tensor, FA, SSE, and validity arrays. The input encoding and
retained-intermediate difference is therefore part of the timing boundary.

Against FSL, brain-mask Dice was `0.993205660`. Within the 877,892 voxels
shared by both masks, tensor, FA, and SSE relative L2 errors were
`6.4647e-6`, `6.9847e-6`, and `3.5067e-5`. Shared-mask values are the relevant
fit comparison because zero-filled voxels at the differing BET boundary
dominate whole-grid errors.

### Legacy correction

The public nonanatomical fixture contained 14 volumes on a `20x20x20` grid.
The Python CLI used eight workers; the reference used the unmodified SimNIBS
4.6 legacy path.

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| dwi2cond-xp process, median of 3 | 2.75 s | 226,824 KiB |
| SimNIBS 4.6 / FSL 6.0.4 harness | 6.766 s | 145,436,672 bytes |

The Python process samples were `2.75/2.71/2.87 s`, giving an observed
`2.46x` median ratio on this fixture. Linux used eight processes only for the
independent 4 mm per-volume stages; the source-required first 8 mm propagation
remained sequential. The measured optimization snapshot preserved the formal
interpolation count and produced bitwise-identical matrices, corrected DWI,
mean, mask, tensor, FA, SSE, and valid mask relative to its retained Python
baseline.

Across the 14 transforms, the largest matrix maximum absolute difference from
FSL was `0.0699333`; mean and maximum moving-FOV corner displacement
differences were `0.0652414 mm` and `0.0941606 mm`. Corrected-DWI and corrected
`b>0`-mean relative L2 errors were `0.00910157` and `0.0130192`, and mask Dice
was `0.993194707`. With FSL matrices fixed, Python sinc resampling reached
`7.42962e-7` relative L2 and `0.00428009` maximum absolute error, locating the
dominant difference in the optimizer endpoint.

The sharp, noiseless fixture makes tensor and SSE ratios sensitive to small
boundary-transform changes. Within the common masks, tensor/FA/SSE relative L2
values were `0.772510/0.0373037/1.06086`; after three mask erosions, tensor and
FA values were `0.00287432/0.00248977` in the remaining 836 interior voxels.

### GRE fieldmap correction

The public nonanatomical fixture used a `16x14x12`
magnitude/fieldmap/b0 grid, an already unwrapped radians-per-second field,
`y-` phase encoding, 0.5 ms dwell time, explicit masks, no median filter, and
eight candidate workers. The complete 6-DOF mutual-information FLIRT schedule
performed 11,349 cost evaluations.

| Implementation | Process wall time | Recorded pipeline time | Peak RSS |
| --- | ---: | ---: | ---: |
| dwi2cond-xp measured snapshot | 1.49/1.51/1.51 s | 0.589/0.632/0.623 s | 204,644--205,204 KiB |
| SimNIBS 4.6 / FSL 6.0.4 harness | 0.46 s | included | not recorded |

The Python optimization snapshot retained candidate order, cost reduction,
search ranges, iteration limits, and stopping conditions. Its final matrix and
all seven checked NIfTI arrays were bitwise equal to the retained Python
baseline, with the evaluation count unchanged. The matrix maximum absolute
difference from FSL was `1.266861e-5`.

For `y-`, distorted magnitude, mapped field, voxel shift, and corrected-b0
relative L2 errors against FSL were `4.56e-8`, `6.66e-7`, `8.93e-7`, and
`7.36e-8`; mapped and corrected masks were exact. For `y`, the corresponding
values were `4.98e-8`, `2.03e-7`, `3.06e-7`, and `6.64e-8`, again with exact
masks. FSL was faster on this very small fixture, where compiled optimizer and
startup costs dominate; the measurement does not characterize anatomical GRE
runtime.

### Fixed TOPUP subset

The public reverse-PE fixture contained two volumes on a `16x14x12` grid. Both
implementations used the unchanged nine-level `b02b0_nosubsamp.cnf` schedule.
The Python runs used eight Numba workers, and the Numba disk cache was populated
before the independent-process measurements.

| Implementation | Complete-process wall time | Median peak RSS |
| --- | ---: | ---: |
| dwi2cond-xp `prepare-topup` CLI | 2.14/2.06/2.12/2.18/2.34/2.04/2.03/2.04/2.26 s | about 225 MiB |
| FSL 6.0.4 `topup` | 4.77/4.59/4.61 s | about 14 MiB |

The medians were `2.12 s` and `4.61 s`, an observed `2.17x` wall-time ratio on
this fixture. The timed Python snapshot retained the nine-level schedule,
iteration counts, stopping conditions, and ordered accumulation of nonzero
terms. Its five public artifacts were file-bitwise equal to the retained
post-source-correction Python baseline. First-time JIT compilation was not part
of this steady-state comparison.

Against FSL, final field relative L2 and maximum absolute differences were
`0.00429657` and `0.451101 Hz`. Inside the joint validity mask, corrected-pair
relative L2 and maximum absolute differences were `0.00161346` and `1.41406`;
pair consistency was `0.00922993`, compared with FSL's `0.00892792`. The
largest movement-parameter absolute difference was `0.00241634`, while the
movement matrix matched a direct FSL C++ probe within `2.78e-17`. The remaining
output difference reflects optimizer-trajectory amplification. Because this
was a tiny synthetic input, it does not establish anatomical-size TOPUP
performance.

### EDDY `--repol`

The public deterministic fixture contained 26 volumes on a `26x26x18` grid:
two b0 volumes, one 24-direction shell, known volume motion and quadratic EC,
and four injected bad slices. Both implementations used seed 1 and eight
threads on the same host. The complete boundary included process startup,
NIfTI read/write, five registration rounds, prediction replacement, rotated
b-vectors, shell alignment, and structured outputs.

| Implementation | Wall time | User time | Peak RSS |
| --- | ---: | ---: | ---: |
| dwi2cond-xp `prepare-eddy` | 7.76 s | 39.68 s | 238,076 KiB |
| FSL 6.0.4 `eddy_openmp` | 9.77 s | 37.93 s | 36,680 KiB |

The observed complete-process ratio was `1.26x`; Python's recorded algorithm
boundary was `6.896 s`. All four injected outliers, at volume/slice pairs
`(5,8)`, `(11,9)`, `(18,7)`, and `(23,10)`, matched exactly. Repeated Python
runs produced byte-identical parameter and outlier-map files.

Within the 3,512-voxel brain mask, corrected-DWI relative L2 was `0.0035352`.
Outlier-free scan-space data and rotated b-vectors had relative L2 values of
`0.0031508` and `0.0014983`. Applying the same project WLS implementation to
each corrected result gave tensor/FA/SSE relative L2 values of
`0.0060714/0.0035526/0.0539494`; SSE mean/P99/maximum absolute differences were
`0.001279/0.01844/0.07959`. The relative SSE value is amplified by a near-zero
residual denominator, so the absolute statistics are reported alongside it.

## DTI fitting

The private HCP b0+b1000 input contained 108 volumes on a `145x174x145` grid
with 881,299 masked voxels. The output boundary included tensor, FA, MD, MO,
L1--L3, V1--V3, S0, SSE, validity mask, and QA JSON.

| Implementation | Workers | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| dwi2cond-xp Python | 16 | 9.76 s | 670% | 767,656 KiB |
| FSL 6.0.4 | 1 process | 108.23 s | 99% | 2,023,996 KiB |

The same server, DWI, shell selection, WLS/gradient-nonlinearity semantics, and
output set were used. The observed wall-time ratio was `11.09x`; Python peak
RSS was approximately 37.9% of FSL's. This is a stage result for one input and
one server, whose exact CPU model was not captured in the frozen measurement
record.

## DTI-to-T1 registration

### Automatic affine registration

The private whole-head comparison used the same fitted tensor/FA/SSE and CHARM
T1, labeling, and bias-corrected T1. The Python boundary included the complete
12-DOF registration, an independent 6-DOF QA registration, affine tensor
interpolation and rotation, FA/V1 decomposition, masks, gzip outputs, and
FA/SSE QA. Eight candidate workers were requested; lower-level runtimes were
not additionally thread-limited.

| Implementation | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: |
| dwi2cond-xp measured snapshot | 53.23 s | 2351% | 3,103,744 KiB |
| SimNIBS 4.6 / FSL 6.0.4 | 233.18 s | 99% | 2,142,568 KiB |

The observed same-machine wall-time ratio was `4.38x`. Profiling attributed
about 5 seconds to T1 preparation, 17--20 seconds to the overlapping 12/6-DOF
registrations, and about 29 seconds to the tensor and derived-output critical
path.

Against FSL, the 12-DOF matrix maximum absolute difference was `0.00661515`;
mean and maximum moving-FOV corner displacement differences were
`0.00385192/0.00771821 mm`. Registered tensor and FA relative L2 errors were
`2.37187e-4` and `3.23951e-4`, with exactly 3,947,315 FA-support voxels in both.
V1 axial-angle mean/P99 were `0.0156600/0.155766` degrees. The maximum was
`84.2445` degrees at a low-anisotropy or boundary voxel. The independent 6-DOF
FA and SSE QA relative L2 errors were `1.23549e-4` and `2.22144e-4`; output
grids, T1 brain, and brain rim matched exactly.

The optimization snapshot kept matrices, registered tensor, valid mask, and QA
arrays bitwise equal to its retained Python baseline. FA/V1 changed at three
voxels when the source rule was restored that tensor-decomposition outputs stay
zero unless the largest eigenvalue is positive.

### Nonlinear registration

The public `24x23x22` synthetic fixture used the same FA, tensor, T1 reference,
and affine initialization for both implementations. The FSL boundary ran the
fixed `fnirt --subsamp=8,4,2,2`, `vecreg`, and tensor-decomposition sequence.
The Python boundary additionally wrote a validity mask, two Jacobian products,
and structured QA.

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| dwi2cond-xp complete CLI, 8 workers | 9.74 s | 289,984 KiB |
| SimNIBS 4.6 / FSL 6.0.4 harness | 9.203 s | 118,861,824 bytes |

Python was about 5.8% slower on this small complete boundary while producing
the additional outputs. Its internal recorded wall time was `8.98 s`.

For the timed snapshot, warped-FA relative L2 was `0.00309871`; deformation
vector error had mean/P99/maximum values of `0.204/0.603/1.147 mm`, and
Jacobian relative L2 was `0.0253746`. Common tensor support had relative L2
`0.0445701`. FA-support Dice was `0.996765`, common-support FA relative L2 was
`6.90e-8`, and sign-invariant V1 angle mean/P99/maximum was
`2.533/8.131/11.352` degrees.

The sparse PCG system is sufficiently ill-conditioned that BLAS
reduction-level differences can change the accepted optimizer endpoint.
Coefficient or field bitwise identity with FSL is therefore not part of this
historical result; the reported image, field, tensor, and direction metrics are
the relevant A/B evidence for the timed snapshot.
