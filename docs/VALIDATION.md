# Validation

The final `v0.3.0` correctness gate is tied to algorithm baseline commit
`04644866ce9ce3ca3cf6fa88ecab356391c3c1f6`. A later documentation and
release-metadata correction does not change that scientific baseline. With
every configured real-FSL probe enabled, it completed with `671 passed` and no
skips. The independent clean
coverage gate completed with `659 passed` in the main batch, `12 passed` in the
montage batch, all three real synthetic TOPUP/EDDY/FNIRT CLI paths, and
`13,607/13,607` executable statements covered (`100.00%`).

## Evidence levels

The project uses three evidence levels so a valid older experiment is not
mistaken for a final-tag rerun:

- **Final-tag evidence** is produced by, or explicitly repeated against, the
  final `v0.3.0` source and frozen dependency stack.
- **Carried-forward experimental evidence** is a real or synthetic result from
  an earlier implementation. It remains useful for the named stage, but is not a
  final-tag end-to-end acceptance result.
- **Historical performance evidence** records an earlier implementation and
  machine boundary. It supports engineering history, not a current release-wide
  speed claim.

## Method-to-evidence map

| Contract | Strongest evidence | Result supported | Important boundary |
| --- | --- | --- | --- |
| FSL/NewNifti storage orientation | Final-tag real-FSL probes and discriminative tests | The high-obliquity counterexample and all 48 signed permutations match `fslreorient2std` data and geometry. | This validates the implemented discrete orientation decision, not arbitrary downstream optimizer equality. |
| Two-pass WLS DTI and gradient nonlinearity | Final-tag source/tests plus real-FSL synthetic probes; carried-forward HCP stage A/B | The fitting equations, tensor layout, derivatives, strict/robust policies, and tested FSL edge semantics are supported. | The HCP `4.18e-6` tensor relative-L2 result predates the final tag and is not a final raw-DWI pipeline rerun. |
| GRE, TOPUP, and EDDY fixed subsets | Final-tag tests and configured real-FSL synthetic CLI probes | The documented fixed schedules, artifact lineage, masks, and failure semantics pass the release gates. | Results apply to the fixed SimNIBS 4.6 subsets and frozen fixtures, not arbitrary FSL configurations. |
| Linear registration, FNIRT, and PPD | Final-tag source/tests and synthetic FSL gates; carried-forward whole-head A/B | The fixed schedules, transforms, tensor reorientation, masks, and no-fallback behavior are supported. | Independent optimizer trajectories retain nonzero residuals; the whole-head nonlinear numbers below are from an earlier implementation. |
| `vn`, `dir`, and `mc` conductivity | Final-tag implementation and discriminative tests, including zero, singular, weighted, and multi-tissue cases; carried-forward local SimNIBS sphere A/B | The equations and current failure policies are supported. | The historical `0` to `2.22e-16` sphere summary has no retained machine-readable comparator artifact and is not presented as a final-tag rerun. |
| Fixed-montage FEM composition | Final-tag construction tests and real SimNIBS 4.6 `SESSION._prepare()`; carried-forward four-mode solve | Explicit mesh, m2m, solver, cache identity, output, and strict-mask contracts are supported. | The final remediation did not rerun the complete HCP raw-DWI-to-FEM chain or the solver. |
| Lead-field interface | Final-tag unit and contract tests | HDF5 axes, reference-electrode convention, NumPy export, and manifest fields are supported. | No completed full-subject all-electrode run is claimed. |

## Automated and release contracts

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

## Current numerical references

The [v0.2.0 algorithm audit](V0.2.0_ALGORITHM_AUDIT_REPORT_2026-08-25.md)
records the workflow and numerical defects that motivated the correctness
release. The final
[v0.3.0 independent re-audit](V0.3.0_THIRD_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REPORT_2026-08-29.md)
and [remediation report](V0.3.0_THIRD_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REMEDIATION_2026-08-29.md)
record the final orientation, dependency, cache, CHARM/FEM composition, and VN
singular-tensor closure. Intermediate audit snapshots remain recoverable from
algorithm baseline commit `0464486` but are not current documentation entry
points.

The final dependency stack retains a nonzero FNIRT residual against FSL 6.0.4:
corrected-image relative L2 `0.001298916`, dense-field relative L2 `0.175488519`,
and maximum field-component error `0.994548967 mm`. These values are release
gates, not bitwise-equivalence claims. They also explain why the documentation
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
branches exercised by the final gate.

## Carried-forward real-data evidence

Earlier validation used one private HCP subject. On the selected b0+b1000 DTI
stage, the comparison over 5,287,164 valid values had maximum, mean, and p99
absolute tensor differences `2.43e-6`, `4.78e-10`, and `2.10e-9`, with relative
L2 `4.18e-6`. This remains useful evidence for the WLS stage, but it is not a
final-tag raw-preprocessing rerun.

The 2026-08-23 whole-head nonlinear comparison was produced by an implementation
then reporting version `0.1.0`. Deformation-vector error had mean/P99/maximum
`0.0130/0.0626/0.1828 mm`, tensor relative L2 was `0.00631`, support Dice was
`0.99962`, and V1 axial-angle mean/P99 was `0.343/3.104` degrees. Applying the
same FSL warp reduced Python PPD tensor relative L2 to `1.51e-6`, which localized
most of the difference to the independently optimized FNIRT endpoint. These are
historical localization results, not final-`v0.3.0` acceptance values.

The same subject also completed a T1/T2 CHARM head model and real C3-to-C4 1 mA
Pardiso FEM for `scalar`, `vn`, `dir`, and `mc`. All formal vector E-field NIfTIs
were finite, every WM/GM/CSF voxel was populated, and voxels outside those labels
were zero after the strict mask. The scalar Pardiso/Hypre comparison on the
common original mapping had maximum absolute difference `2.98e-8 V/m` and
relative L2 `6.28e-9`. This establishes that the four-mode workflow has completed
on a real subject; it does not assert that the final tag reran the solver after
the last FEM composition and cache-identity changes.

## Evidence boundary

No HCP data or subject derivative is public. Real-data evidence is dominated by
one private subject, so population-level generalization is not claimed. The
final remediation did not repeat a complete HCP raw-DWI-to-TOPUP/EDDY-to-FNIRT-
to-FEM run; final-tag evidence for the last FEM changes stops at real
`SESSION._prepare()` plus discriminative construction and transaction tests.

The project implements only the fixed FSL behavior required by SimNIBS 4.6
`dwi2cond`; it is not a general FSL command replacement. Wrapped phase requiring
PRELUDE, arbitrary TOPUP/EDDY/FNIRT configurations, and silent multishell
single-tensor fitting remain outside the supported contract. Stage timings and
carried-forward experiments do not establish a universal full-workflow speed or
equivalence claim.
