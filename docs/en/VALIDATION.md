# Validation

[English](../en/VALIDATION.md) | [简体中文](../zh/VALIDATION.md)

[Documentation](README.md) · [Project home](../../README.md)

## v0.4.0 validation — 2026-10-14

The local release gate passed with **955 unique tests and no skips**: the
main batch passed 942 tests, a separately executed ordered-evaluator regression
added one passing case, and the montage batch passed 12. The focused case was
included in the completed unit coverage data before aggregation and remains in
the test tree for normal CI collection. Configured real FSL and SimNIBS probes
were enabled. The main batch reported 12 fixture-related SimNIBS `UserWarning`
messages; this is not a warning-free result.

`python scripts/run_coverage.py` completed successfully with **15,363/15,363
executable statements covered (100.00%)**. Coverage used isolated cold Numba
compilation caches with `NUMBA_JIT_COVERAGE=1`, combined with three real
synthetic TOPUP, EDDY and FNIRT CLI end-to-end runs. The 100% statement threshold
and source scope were retained; statement coverage does not establish complete
branch coverage or correctness for arbitrary inputs.

The 36 device-free CUDA tests exercise backend mathematics and error handling;
they do not substitute for execution on a CUDA device. The separately completed
Linux/RTX 2080 Ti CPU/GPU end-to-end comparison is documented in
[Benchmarks](BENCHMARKS.md); it validates the recorded input and processing boundary.
No other-platform GPU certification or new complete real-subject FEM run is
inferred from these checks.

The release checks also passed Ruff, version synchronization, Markdown links
and source-manifest validation. The runtime dependency audit inspected 19
packages and found no known vulnerabilities in the audit database on this date.
Remote platform/security CI and public distribution checks are separate release
steps; the local gate does not replace them.

## Historical v0.3.0 validation

The full record below retains the v0.3.0 results and their earlier experimental
boundaries; those counts and numerical values are not new v0.4.0 measurements.

The revised `v0.3.0` release was validated against algorithm baseline commit
`ae5a6a73f9f475cb51355af085f7832771cf8d77`. A tagged documentation or
release-metadata descendant does not change that scientific baseline. With
every configured real-FSL probe enabled, the clean main batch completed with
`666 passed` and no skips. The independent coverage check repeated those `666`
tests, completed `12 passed` in the montage batch, exercised all three real
synthetic TOPUP/EDDY/FNIRT CLI paths, and covered `13,663/13,663` executable
statements (`100.00%`).

### Evidence levels

The project uses three evidence levels so a valid older experiment is not
mistaken for a release rerun:

- **Release evidence** is produced by, or explicitly repeated against, the
  final `v0.3.0` source and frozen dependency stack.
- **Earlier experimental evidence** is a real or synthetic result from
  an earlier implementation. It remains useful for the named stage, but is not a
  release end-to-end acceptance result.
- **Historical performance evidence** records an earlier implementation and
  machine boundary. It supports engineering history, not a current release-wide
  speed claim.

### Method-to-evidence map

| Method | Strongest evidence | Result supported | Important boundary |
| --- | --- | --- | --- |
| FSL/NewNifti storage orientation | Release real-FSL probes and discriminative tests | The high-obliquity counterexample and all 48 signed permutations match `fslreorient2std` data and geometry. | This validates the implemented discrete orientation decision, not arbitrary downstream optimizer equality. |
| Two-pass WLS DTI and gradient nonlinearity | Release source/tests plus real-FSL synthetic probes; earlier HCP stage A/B | The fitting equations, tensor layout, derivatives, strict/robust policies, and tested FSL edge semantics are supported. | The HCP `4.18e-6` tensor relative-L2 result predates the final tag and is not a final raw-DWI pipeline rerun. |
| GRE, TOPUP, and EDDY fixed subsets | Release tests and configured real-FSL synthetic CLI probes | The documented fixed schedules, artifact lineage, masks, and failure semantics pass the release checks. | Results apply to the fixed SimNIBS 4.6 subsets and frozen fixtures, not arbitrary FSL configurations. |
| Linear registration, FNIRT, and PPD | Release source/tests and synthetic FSL checks; earlier whole-head A/B | The fixed schedules, transforms, tensor reorientation, masks, and no-fallback behavior are supported. | Independent optimizer trajectories retain nonzero residuals; the whole-head nonlinear numbers below are from an earlier implementation. |
| `vn`, `dir`, and `mc` conductivity | Release implementation and discriminative tests, including zero, singular, weighted, and multi-tissue cases; earlier local SimNIBS sphere A/B | The equations and current failure policies are supported. | The historical `0` to `2.22e-16` sphere summary has no retained machine-readable comparator artifact and is not presented as a release rerun. |
| Fixed-montage FEM composition | Release construction tests and real SimNIBS 4.6 `SESSION._prepare()`; earlier four-mode solve | Explicit mesh, m2m, solver, cache identity, output, and strict-mask behavior are supported. | The final remediation did not rerun the complete HCP raw-DWI-to-FEM chain or the solver. |
| Lead-field interface | Release unit and interface tests | HDF5 axes, reference-electrode convention, NumPy export, and manifest fields are supported. | No completed full-subject all-electrode run is claimed. |

### Automated and release checks

Tests cover gradient conventions, weighted fitting, invalid-input policies, the
fixed legacy/GRE/TOPUP/EDDY branches, linear and nonlinear registration, tensor
reorientation, conductivity modes, SimNIBS object construction, strict tissue
masking, lead-field axes, plotting, QA, manifests, and cache validation. The
coverage runner executes real synthetic TOPUP/EDDY/FNIRT end-to-end paths in
isolated Numba caches.

The final tagged source also passed package construction and installation,
Markdown and version checks, CodeQL, OpenSSF Scorecard, tracked-file and archive
privacy audits, wheel/sdist metadata checks, and dependency auditing. The
published wheel, sdist, CycloneDX SBOM, and checksum file were downloaded again;
their SHA-256 values, isolated wheel import/CLI, and GitHub build provenance were
verified independently.

### Current numerical references

The [v0.2.0 algorithm audit](V0.2.0_ALGORITHM_AUDIT_REPORT_2026-08-25.md)
records the workflow and numerical defects that motivated the correctness
release. The
[fourth independent v0.3.0 re-audit](V0.3.0_FOURTH_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REPORT_2026-08-30.md)
and [its remediation report](V0.3.0_FOURTH_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REMEDIATION_2026-08-30.md)
record the current matrix, BET, conductivity eigensystem, FEM-cache, and TOPUP
corrections. The earlier
[third re-audit](V0.3.0_THIRD_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REPORT_2026-08-29.md)
remains the evidence for orientation, dependency, CHARM/FEM composition, and VN
singular-tensor findings that were not superseded.

The final dependency stack retains a nonzero FNIRT residual against FSL 6.0.4:
corrected-image relative L2 `0.001298916`, dense-field relative L2 `0.175488519`,
and maximum field-component error `0.994548967 mm`. These values are release
checks, not bitwise-equivalence claims. They also explain why the documentation
states that the fixed FNIRT subset is validated on discriminative fixtures while
arbitrary-input optimizer identity remains out of scope.

A non-identity 90-degree premat plus one-voxel GRE shift was compared with real
FSL `convertwarp` and `applywarp`; the corrected Python trilinear result is
array-exact on the `9x9x3` fixture. A synthetic DTI fixture compared with FSL
6.0.4 `dtifit --wls --gradnonlin` had tensor maximum and mean absolute
differences `1.16e-10` and `8.26e-12`.

For the public fixed-path fixtures, TOPUP field and corrected-pair relative L2
errors were `0.00430` and `0.00161`. EDDY detected all four injected bad slices
and no others; mask-interior corrected-DWI and downstream-tensor relative L2
errors were `0.0035352` and `0.0060714`. These fixture results support the fixed
branches exercised by the final check.

### Earlier real-data evidence

Earlier validation used one private HCP subject. Of 881,299 masked voxels,
881,194 were valid; 105 invalid voxels were zeroed and recorded. On the selected b0+b1000 DTI
stage, the comparison over 5,287,164 valid values had maximum, mean, and p99
absolute tensor differences `2.43e-6`, `4.78e-10`, and `2.10e-9`, with relative
L2 `4.18e-6`. This remains useful evidence for the WLS stage, but it is not a
release raw-preprocessing rerun.

The 2026-08-23 whole-head nonlinear comparison was produced by an implementation
then reporting version `0.1.0`. Deformation-vector error had mean/P99/maximum
`0.0130/0.0626/0.1828 mm`, tensor relative L2 was `0.00631`, support Dice was
`0.99962`, and V1 axial-angle mean/P99 was `0.343/3.104` degrees. Applying the
same FSL warp reduced Python PPD tensor relative L2 to `1.51e-6`, which localized
most of the difference to the independently optimized FNIRT endpoint. These are
historical localization results, not final-`v0.3.0` acceptance values.

The same subject also completed a T1/T2 CHARM head model and real C3-to-C4 1 mA
Pardiso FEM for `scalar`, `vn`, `dir`, and `mc`. All vector E-field NIfTIs
were finite, every WM/GM/CSF voxel was populated, and voxels outside those labels
were zero after the strict mask. The scalar Pardiso/Hypre comparison on the
common original mapping had maximum absolute difference `2.98e-8 V/m` and
relative L2 `6.28e-9`. This establishes that the four-mode workflow has completed
on a real subject; it does not assert that the final tag reran the solver after
the last FEM composition and cache-identity changes.

### Scope of the evidence

No HCP data or subject derivative is public. Real-data evidence is dominated by
one private subject, so population-level generalization is not claimed. The
final remediation did not repeat a complete HCP raw-DWI-to-TOPUP/EDDY-to-FNIRT-
to-FEM run; release evidence for the last FEM changes stops at real
`SESSION._prepare()` plus discriminative construction and transaction tests.

The project implements only the fixed FSL behavior required by SimNIBS 4.6
`dwi2cond`; it is not a general FSL command replacement. Wrapped phase requiring
PRELUDE, arbitrary TOPUP/EDDY/FNIRT configurations, and silent multishell
single-tensor fitting remain outside the supported scope. Stage timings and
earlier experiments do not establish a universal full-workflow speed or
equivalence claim.
