# Historical benchmarks

This page preserves stage-level measurements collected during the v0.1 and
v0.2 development lines. No benchmark was rerun for the 2026-08-29
documentation and release-metadata republication. The measurements describe
the named implementation snapshot, fixture, host, worker count, cache state,
and output boundary; they are not new results from the republished tag.

A later audit identified workflow and numerical-contract defects in v0.2 that
were corrected in `v0.3.0`. These historical timings remain useful for the
named stages, but they do not establish the correctness or speed of a new
deployment. Use v0.3 for current deployments and its own validation evidence.

Unless stated otherwise, the reference was SimNIBS 4.6 with FSL 6.0.4 on the
same host and input. Ratios compare only paired rows in one table. Fixtures,
process boundaries, worker counts, warm-cache states, and output sets differ
between sections, so timings must not be added into an end-to-end result.

## SimNIBS 4.6 GRE fieldmap path

The public nonanatomical fixture used a `16x14x12` magnitude/fieldmap/b0 grid,
an already unwrapped radians-per-second field, `y-` phase encoding, 0.5 ms
dwell time, explicit masks, no median filter, and eight candidate workers. The
Python process retained the complete 6-DOF mutual-information FLIRT schedule and
performed 11,349 cost evaluations.

| Implementation | Process wall time | Pipeline time | Peak RSS |
| --- | ---: | ---: | ---: |
| dwi2cond-xp, final batched path | 1.49/1.51/1.51 s | 0.589/0.632/0.623 s | 204,644--205,204 KiB |
| dwi2cond-xp, retained pre-batch path | 6.55/6.81/6.88 s | 5.80--6.08 s | about 220 MiB |
| SimNIBS 4.6 / FSL 6.0.4 harness | 0.46 s | included | not recorded |

The final optimization locksteps independent Brent trajectories and evaluates
each round as one Numba candidate batch. Affine matrices are also constructed
in batches. It does not alter candidate order within a trajectory, cost
reduction, search ranges, iteration limits, or stopping conditions. The final
matrix and all seven checked NIfTI arrays were bitwise equal to the retained
Python path, and the evaluation count remained 11,349. The final path also
stacks matrix validation/inversion and vectorizes all 11-cubed coarse-grid
interpolation and affine construction. The matrix maximum
absolute difference from FSL remained `1.266861e-5`.

For `y-`, distorted magnitude, mapped field, voxel shift, and corrected-b0
relative L2 errors against FSL were `4.56e-8`, `6.66e-7`, `8.93e-7`, and
`7.36e-8`; mapped and corrected masks were exact. For `y`, the corresponding
values were `4.98e-8`, `2.03e-7`, `3.06e-7`, and `6.64e-8`, again with exact
masks. FSL remains faster on this extremely small input because its optimizer
and startup path are compiled C++; no claim is made that Python is faster here
or that this timing predicts anatomical GRE performance.

After all imports and Numba kernels were resident, four same-process runs took
`0.585/0.358/0.413/0.400 s` at the pipeline boundary. This warm-process series
is reported only to separate computation from startup; it is not compared as a
speed ratio against the one-process FSL harness.

## Automatic affine DTI-to-T1 registration

The private whole-head comparison used the same fitted tensor/FA/SSE and CHARM
T1, labeling, and bias-corrected T1. The Python path ran the complete 12-DOF
registration, independent 6-DOF QA registration, affine tensor interpolation and
rotation, FA/V1 decomposition, masks, all gzip outputs, and FA/SSE QA. Eight
candidate workers were requested; lower-level runtimes were not additionally
thread-limited.

| Implementation | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: |
| dwi2cond-xp final process | 53.23 s | 2351% | 3,103,744 KiB |
| dwi2cond-xp pre-optimization process | 100.90 s | 1607% | 3,410,112 KiB |
| SimNIBS 4.6 / FSL 6.0.4 | 233.18 s | 99% | 2,142,568 KiB |

The final same-machine wall ratio was `4.38x` over FSL and `1.90x` over the
retained Python baseline. Profiling separated about 5 seconds of T1 preparation,
17--20 seconds for overlapped 12/6-DOF FLIRT, and about 29 seconds for the tensor
and derived-output critical path. The retained optimizations overlap independent
registrations, tensor components and source-mask sampling, and tensor gzip with
read-only FA/V1 decomposition. An attempted background T1-gzip branch was
rejected because memory-bandwidth contention slowed FLIRT.

Against FSL, the 12-DOF matrix maximum absolute difference was `0.00661515` and
the mean/maximum moving-FOV corner displacement differences were
`0.00385192/0.00771821 mm`. Registered tensor and FA relative L2 errors were
`2.37187e-4` and `3.23951e-4`; FA support was exactly 3,947,315 voxels in both.
V1 axial-angle mean/P99 were `0.0156600/0.155766` degrees. The rare maximum of
`84.2445` degrees is retained rather than hidden and occurs at a low-anisotropy
or boundary case. The 6-DOF FA and SSE QA relative L2 errors were `1.23549e-4`
and `2.22144e-4`. Output grids, T1 brain, and brain rim matched exactly.

The optimized matrices, registered tensor, valid mask, and QA arrays remained
bitwise equal to the pre-optimization Python path. FA/V1 intentionally changed
at three voxels after restoring FSL's source rule that tensor-decomposition
outputs remain zero unless the largest eigenvalue is positive.

## SimNIBS 4.6 fixed TOPUP subset

The public nonanatomical reverse-PE fixture contains two volumes on a
`16x14x12` grid. Both implementations used the unchanged nine-level
`b02b0_nosubsamp.cnf` schedule. The Python runs used eight Numba workers.

| Implementation | Measured wall boundary | Median peak RSS |
| --- | ---: | ---: |
| dwi2cond-xp complete `prepare-topup` CLI | 2.14/2.06/2.12/2.18/2.34/2.04/2.03/2.04/2.26 s | about 225 MiB |
| dwi2cond-xp same-process resident algorithm | 0.681/0.630/0.609/0.648/0.709/0.610/0.592 s | included above |
| FSL 6.0.4 `topup` | 4.77/4.59/4.61 s | about 14 MiB |

The Numba disk cache was populated before the independent-process measurements.
The latest complete-CLI medians are 2.12 and 4.61 seconds, an observed `2.17x`
wall-time ratio on this tiny fixture. The first and
previous optimized Python medians were 4.14, 3.31, and 2.91 seconds. The latest
pass lazily imports CLI routes and preprocessing exports, caches identical rigid
pull matrices and their inverses, skips spline-coefficient pairs and voxels with
provably disjoint support, stores the small LM systems densely while retaining
ascending-column PCG accumulation, and reuses identical regularization products.
It does not change the schedule, iteration counts, stopping conditions, or the
z/y/x accumulation order of any nonzero term. All five numerical artifacts are
file-bitwise equal to the previous Python outputs. A same-process kernel-resident
median was about 1.12 seconds; it is not compared with a fresh
FSL process. First-time JIT compilation remains a separate one-off cost.

The fifth optimization pass flattened only independent, potentially nonzero
field-Hessian elements across the eight workers. Each element retains the same
FSL z/y/x accumulation order. It also caches the ordered coefficient-pair
worksets and repeated one-dimensional bending-kernel overlaps. The five public
artifacts remain file-bitwise equal to the post-source-correction baseline. A
post-edit run that still had to populate missing Numba signatures took 5.30
seconds; a fully empty cache has previously taken about 15 seconds and is not
mixed into the steady-state comparison.

The sixth optimization pass keeps every Hessian element's FSL z/y/x reduction
unchanged but orders independent elements by their actual overlapping support,
then stripes them into equal worker chunks. This removes the previous 17--47%
static-work imbalance across the nine coefficient grids. It also evaluates the
two independent spline-transpose accumulators in one voxel traversal while
preserving each accumulator's original order. The five public artifacts remain
file-bitwise equal to the fifth-pass baseline; one-worker and eight-worker
outputs are also file-bitwise equal. After one resident warm-up, six algorithm
samples had a median of 0.823 seconds. This resident number describes the long
pipeline's kernel-resident boundary and is not compared with a fresh FSL
process. A candidate that combined dense field and derivative expansion was
bitwise equal but slower and was removed.

The seventh optimization pass batches the two independent periodic-cubic scan
samplers into one Numba dispatch, keeps a CSR view solely for repeated ordered
regularization products, and expands independent field voxels in parallel while
retaining each voxel's coefficient z/y/x accumulation order. The original CSC
Hessian and coefficient-major expansion remain available as numerical
references. All five artifacts are file-bitwise equal to the sixth-pass output.
The final one-worker and eight-worker artifacts are also file-bitwise equal.
The seven-run complete-CLI median is 2.17 seconds; after one resident warm-up,
the seven-run algorithm median is 0.630 seconds. Relative to the sixth pass,
this reduces the resident algorithm boundary by about 23.5%. A dense precomputed
3D-basis Hessian was bitwise equal but raised peak RSS to about 312 MiB and had
no wall-time benefit, so it was removed.

The eighth optimization pass compiles the complete dense-PCG iteration loop in
one Numba call while retaining an explicit Python reference backend and the
same ascending-column matrix-vector accumulation. It also computes the five
independent movement-interaction transpose columns in one coefficient/voxel
traversal; every column retains its original z/y/x reduction order. The nine
fresh-process CLI samples in the primary table have a 2.12-second median, and
their internal algorithm samples have a 1.497-second median. All five numerical
artifacts are file-bitwise equal to the seventh-pass baseline. Batched regrid,
axis-rotation caching, cross-level objective reuse, phase-specialized Hessian,
and paired direct/derivative movement candidates were removed because they did
not improve the complete wall boundary.

A fourth startup-focused pass also made the package-root tensor APIs and the
disabled progress-bar path lazy. On a later same-host load sample, the
pre-change Python CLI ran in `3.73/3.61/3.84 s` (median `3.73 s`) and the final
path ran in `3.47/3.49/3.52 s` (median `3.49 s`). A contemporaneous FSL TOPUP
rerun took `4.90/4.72/4.72 s` (median `4.72 s`), so Python remained `1.35x`
faster under that noisier system load. The five output artifacts were again
file-bitwise identical. These later timings are a paired optimization sample,
not a replacement for the quieter primary table above. In the same process,
with all kernels resident, the sparse-reference path had a ten-run median of
`0.751 s`.

Two additional candidates were rejected. Mirroring one triangular half of the
field Hessian would change real last-bit differences between the two halves
(maximum observed asymmetry `6.82e-13`). Replacing the cached sparse
regularization product with the ordered dense kernel was bitwise identical but
slower (`0.806 s` versus `0.751 s` median in alternating same-process trials).

After restoring FSL 6.0.4's float-trigonometry movement-matrix semantics, the
final field relative L2 and maximum absolute differences were `0.00429657` and
`0.451101 Hz`. Inside the joint validity mask, the corrected-pair relative L2
and maximum absolute differences were `0.00161346` and `1.41406`; pair
consistency was `0.00922993`, compared with FSL's `0.00892792`. The largest
movement-parameter absolute difference was `0.00241634`. The movement matrix
itself matched a direct FSL C++ probe within `2.78e-17`; the remaining output
difference is optimizer-trajectory amplification. These results are not
bitwise equivalence claims and must not be extrapolated to anatomical image
sizes before a same-shape benchmark is run.

## SimNIBS 4.6 EDDY `--repol`

The public deterministic fixture contains 26 volumes on a `26x26x18` grid: two
b0 volumes, one 24-direction shell, known volume motion/quadratic EC, and four
injected bad slices. Both implementations used seed 1 and eight threads on the
same host. The complete boundary includes process startup, NIfTI read/write,
five registration rounds, prediction replacement, rotated b-vectors, shell
alignment, and structured outputs.

| Implementation | Wall time | User time | Peak RSS |
| --- | ---: | ---: | ---: |
| dwi2cond-xp `prepare-eddy` | 7.76 s | 39.68 s | 238,076 KiB |
| FSL 6.0.4 `eddy_openmp` | 9.77 s | 37.93 s | 36,680 KiB |

The observed complete-process speedup is `1.26x`; Python's recorded algorithm
boundary was `6.896 s`. All four injected outliers, at volume/slice pairs
`(5,8)`, `(11,9)`, `(18,7)`, and `(23,10)`, matched exactly. Repeated Python
runs produced byte-identical parameter and outlier-map files.

Within the 3,512-voxel brain mask, corrected-DWI relative L2 was `0.0035352`.
Outlier-free scan-space data and rotated b-vectors had relative L2 values
`0.0031508` and `0.0014983`. Applying the same project WLS implementation to
each corrected result gave tensor/FA/SSE relative L2 values
`0.0060714/0.0035526/0.0539494`; SSE mean/P99/maximum absolute differences were
`0.001279/0.01844/0.07959`. The larger SSE relative value reflects a small
near-zero residual denominator and is accompanied by its absolute statistics.
An optional TOPUP-Hz-field complete CLI smoke run also wrote a
`26x26x18x26` corrected DWI and `26x16` parameter table successfully.

The release wheel was also installed without dependency downloads into an
isolated overlay that reads the unchanged SimNIBS 4.6.0 environment. `pip
check` reported no broken requirements. On the same fixture, that installed
wheel took `37.78 s` on its first process including Numba compilation and
`9.34 s` in the next independent process with cached kernels. Both runs found
the same four outliers, and the parameter and outlier-map files were byte
identical. The cached wheel smoke is a packaging compatibility check; the
strict source/FSL comparison in the table remains the primary performance
result.

## SimNIBS 4.6 nonlinear T1 registration

The public `24x23x22` synthetic fixture used the same FA, tensor, T1 reference,
and affine initialization for both implementations. The FSL boundary ran the
fixed `fnirt --subsamp=8,4,2,2`, `vecreg`, and tensor-decomposition sequence;
the Python boundary additionally wrote a validity mask, two Jacobian products,
and structured QA.

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| dwi2cond-xp complete CLI, 8 workers | 9.74 s | 289,984 KiB |
| SimNIBS 4.6 / FSL 6.0.4 harness | 9.203 s | 118,861,824 bytes |

Python was about 5.8% slower on this small complete boundary while producing
the additional outputs above. Its internal recorded wall time was 8.98 seconds.
The warped-FA relative L2 was `0.00309871`; deformation-vector error had
mean/P99/max `0.204/0.603/1.147 mm`, and Jacobian relative L2 was `0.0253746`.
The common tensor support had relative L2 `0.0445701`. FA support Dice was
`0.996765`, common-support FA relative L2 was `6.90e-8`, and sign-invariant V1
angle mean/P99/max was `2.533/8.131/11.352 degrees`.

The sparse PCG system is ill-conditioned enough that BLAS reduction-level
differences alter the accepted optimizer endpoint. Accordingly, coefficient
or field bitwise identity is not claimed. Two fused PCG experiments were
rejected: one preserved the prior Python outputs bitwise but was slower, while
the old-NEWMAT serial reduction order reduced FSL agreement. Neither is enabled.

### Installed-wheel DAG and persistent kernel cache

The pre-release candidate wheel, which was still versioned `0.1.0` at the time,
was installed with `--no-deps` into a
`--system-site-packages` overlay of the frozen SimNIBS 4.6 environment. The
public nomoco plus nonlinear DAG used the same input, eight workers, and output
contract for all measurements.

| Boundary | Wall time | Peak RSS |
| --- | ---: | ---: |
| Source tree, empty Numba cache | 58.55 s | 556,600 KiB |
| Installed wheel, empty Numba cache | 58.37 s | 553,328 KiB |
| Installed wheel, populated kernel cache and new output | 20.58 s | 317,664 KiB |
| Installed wheel, complete DAG artifact-cache hit | 0.77 s | 144,776 KiB |

The source and wheel runs produced 40 bitwise-identical NIfTI arrays and
affines. The cold and warm wheel runs were also bitwise identical. Numba's disk
cache is reused across subjects because ordinary image-shape changes do not
change a compiled function signature; a dtype, dimensionality/layout, CPU,
Numba/Python version, or source-code change may create a new specialization.
These host-specific `.nbi`/`.nbc` machine-code caches are intentionally not
shipped in the platform-independent wheel. Coverage CI alone forces a fresh
cache so that compiled lines are measured; production runs retain their local
persistent cache.

## SimNIBS 4.6 legacy correction

The public nonanatomical fixture contained 14 volumes on a `20x20x20` grid.
The Python CLI used eight workers and the FSL reference used the unmodified
SimNIBS 4.6 legacy path.

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| dwi2cond-xp process, median of 3 | 2.75 s | 226,824 KiB |
| dwi2cond-xp pipeline boundary, median of 3 | 1.951 s | included above |
| SimNIBS 4.6 / FSL 6.0.4 harness | 6.766 s | 145,436,672 bytes |

The Python process wall times were `2.75/2.71/2.87 s`, or a `2.46x` median
speedup over the FSL harness. Linux uses eight processes only for the independent
4 mm per-volume stages; the source-required first 8 mm propagation remains
sequential. Windows and macOS retain the same algorithm through a thread backend.
No HCP legacy-correction speed claim has been measured, and the P4 `nomoco`
result is not extrapolated to this mode.

The normcorr hot path fuses coordinate generation, edge weighting and linear
sampling without fast-math, materializes all FSL float32 product boundaries,
and reproduces NumPy/FSL x-then-y-then-z pairwise reduction. All 14 matrices and
the corrected DWI, mean, mask, tensor, FA, SSE and valid mask were bitwise equal
to the retained pre-optimization Python outputs in three independent runs.

Across all 14 final transforms, the largest matrix max-absolute difference was
`0.0699333`; the mean and maximum moving-FOV corner displacement differences
were `0.0652414 mm` and `0.0941606 mm`. Corrected-DWI and corrected `b>0`-mean
relative L2 errors were `0.00910157` and `0.0130192`; mask Dice was
`0.993194707`. With the FSL matrices fixed, Python sinc resampling reached
`7.42962e-7` relative L2 and `0.00428009` maximum absolute error, localizing the
dominant difference to the optimizer endpoint rather than coordinate conversion
or final interpolation.

The optimized width-seven Hanning-sinc kernel is bitwise identical to its
retained vectorized reference kernel. On this fixture it reduced the final
formal resampling stage from about 8.39 seconds to 0.054 seconds. Each formal
output volume still undergoes exactly one interpolation.

The synthetic input is sharp and noiseless, so small boundary-transform changes
make tensor and SSE relative errors ill-conditioned. Within the common masks,
tensor/FA/SSE relative L2 values were `0.772510`, `0.0373037`, and `1.06086`;
after three mask erosions, tensor and FA were `0.00287432` and `0.00248977` in
the remaining 836 interior voxels. Both figures are reported; no whole-mask
tensor-equivalence claim is made.

## SimNIBS 4.6 `nomoco` raw-DWI closure

The private HCP b0+b1000 input contained 108 volumes on a `145x174x145` grid.
Both runs used the same b0-normalized b-values and b-vectors. The Python CLI used
eight worker processes with one BLAS thread per worker; the FSL reference used
the unmodified SimNIBS 4.6 `--prepro --nomoco --keepstuff` path.

| Implementation and input | Execution | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| dwi2cond-xp, validated `.nii` mmap | 8 workers | 18.82 s | 306% | 1,770,256 KiB |
| dwi2cond-xp, `.nii.gz` single decode | 8 workers | 52.92 s | 170% | 4,791,436 KiB |
| SimNIBS 4.6 / FSL 6.0.4 | 1 process | 257.30 s | 99% | 3,167,907,840 bytes |

The observed same-machine wall-time ratios were `13.67x` for validated mmap and
`4.86x` for gzip input. A compatible uncompressed float32, canonical, finite,
nonnegative NIfTI is scanned in z-blocks and used directly. A `.nii.gz` input is
decoded exactly once to an uncompressed `DWIforfit.nii`, after which all workers
share it by mmap. The two Python paths produced bitwise-identical b0, mask,
brain, tensor, FA, SSE, and validity arrays. File encoding, retained intermediate,
and memory differences must remain explicit when quoting either number.

The brain-mask Dice was `0.993205660`. Within the 877,892 voxels shared by both
masks, tensor, FA, and SSE relative L2 errors were `6.4647e-6`, `6.9847e-6`, and
`3.5067e-5`. Whole-grid relative errors are not used for fitting parity because
zero-filled values in the non-overlapping BET boundary dominate them.

## DTI fitting

The private HCP b0+b1000 input contained 108 volumes on a `145x174x145` grid
with 881,299 masked voxels. The output boundary included tensor, FA, MD, MO,
L1-L3, V1-V3, S0, SSE, validity mask, and QA JSON.

| Implementation | Workers | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| dwi2cond-xp Python | 16 | 9.76 s | 670% | 767,656 KiB |
| FSL 6.0.4 | 1 process | 108.23 s | 99% | 2,023,996 KiB |

The same server, DWI, shell selection, WLS/gradient-nonlinearity semantics, and
output set were used. The observed wall-time ratio was `11.09x`; Python peak RSS
was approximately 37.9% of FSL's. This is one system and one DTI-fitting input.
It must not be extrapolated to raw preprocessing, registration, meshing, FEM, or
other hardware.

The run used a many-core server, but the frozen evidence record did not capture
the exact CPU model. Consequently no model-specific performance claim is made.
