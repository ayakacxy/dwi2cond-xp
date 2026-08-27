# Validation

The final `v0.3.0` correctness gate completed with `617 passed, 13 skipped` in
the ordinary local batch and `13,370/13,370` executable statements covered
(`100.00%`). With the available FSL 6.0.4 references configured, the main batch
completed with `627 passed, 3 skipped`; all `630/630` available reference probes
passed after the separate FSLMATHS checks. Cross-platform CI enforces the same
source, coverage, documentation, CLI, and package gates. A replacement tag is
published only after main CI succeeds; the tag-triggered Release workflow then
repeats the release gate before publishing assets, SBOM, and attestations.

## Automated contracts

Tests cover gradient conventions, weighted fitting, invalid voxels, the fixed
legacy/GRE/TOPUP/EDDY preprocessing branches, linear and nonlinear
registration, tensor reorientation, conductivity modes, SimNIBS object
construction, strict tissue masking, lead-field axes, plotting, QA, manifests,
and cache validation. The coverage runner also executes real synthetic
TOPUP/EDDY/FNIRT end-to-end paths in isolated Numba caches.

The `v0.2.0` Linux release gate completed with `535 passed, 7 skipped` and
`12,443/12,443` executable statements covered (`100.00%`). The same strict
coverage threshold passed on Ubuntu 22.04, macOS 14 arm64, and Windows Server
2022. Optional FSL reference or SimNIBS integration tests skip visibly when
their external prerequisites are unavailable; the threshold is not lowered.

The final tagged source also passed package construction and installation,
Markdown and version checks, CodeQL, OpenSSF Scorecard, tracked-file and archive
privacy audits, wheel/sdist metadata checks, and dependency auditing. The
published wheel, sdist, CycloneDX SBOM, and checksum file were downloaded again;
their SHA-256 values, isolated wheel import/CLI, and GitHub build provenance were
verified independently.

## Numerical references

The final-tag independent re-audit identified `6 P1 + 9 P2/P3` finding clusters
across workflow composition, legacy MCFLIRT control flow, scientific QA/cache
lineage, FEM transactions, and public API boundaries. The
[independent re-audit](V0.3.0_LATEST_TAG_INDEPENDENT_REAUDIT_REPORT_2026-08-26.md)
records the discriminative counterexamples, and the
[final remediation report](V0.3.0_LATEST_TAG_REMEDIATION_REPORT_2026-08-27.md)
records their fixes and acceptance gates. The final release retains explicit
limits: EDDY/FNIRT optimizer trajectories are not claimed bitwise-identical for
arbitrary inputs, and no new full HCP TOPUP-to-EDDY-to-FNIRT-to-FEM run is
inferred from synthetic stage evidence.

A non-identity 90-degree premat plus one-voxel GRE shift was also compared with
real FSL `convertwarp` and `applywarp`; the corrected Python trilinear result is
array-exact on the `9x9x3` fixture.

A synthetic DTI fixture was compared with FSL 6.0.4 `dtifit --wls
--gradnonlin`; tensor maximum and mean absolute differences were `1.16e-10` and
`8.26e-12`. On private HCP b0+b1000 data, the tensor comparison over 5,287,164
valid values had maximum, mean, and p99 absolute differences
`2.43e-6`, `4.78e-10`, and `2.10e-9`, with relative L2 `4.18e-6`. Rare boundary
voxels dominate the maximum, so whole-head elementwise exactness is not claimed.

On a synthetic sphere mesh, 274,770 conductivity components agreed with
SimNIBS 4.6 to maximum absolute error `0` for `vn` and `2.22e-16` for `dir` and
`mc`.

For the public fixed-path preprocessing fixtures, TOPUP field and corrected-pair
relative L2 errors were `0.00430` and `0.00161`. EDDY detected all four injected
bad slices and no others; mask-interior corrected-DWI and downstream-tensor
relative L2 errors were `0.0035352` and `0.0060714`.

On the private whole-head nonlinear comparison, deformation-vector error had
mean/P99/maximum `0.0130/0.0626/0.1828 mm`, tensor relative L2 was `0.00631`,
support Dice was `0.99962`, and V1 axial-angle mean/P99 was `0.343/3.104`
degrees. Applying the same FSL warp reduced Python PPD tensor relative L2 to
`1.51e-6`, separating reorientation error from the iterative FNIRT endpoint.
Bitwise equality with the independent FNIRT optimization is not claimed.

## End-to-end evidence

A private HCP subject was processed through a complete T1/T2 CHARM head model,
T1-grid tensor, and real C3-to-C4 1 mA FEM for `scalar`, `vn`, `dir`, and `mc`
using Pardiso. All formal vector E-field NIfTIs were finite. Every WM/GM/CSF
voxel was populated and every voxel outside those labels was zero after the
strict mask.

The scalar Pardiso/Hypre comparison on the common original mapping had maximum
absolute difference `2.98e-8 V/m` and relative L2 `6.28e-9`.

## Evidence boundary

No HCP data or subject derivative is public. Real-data evidence is dominated by
one private HCP subject, so population-level generalization is not claimed. The
full-subject all-electrode lead-field interface is tested but is not part of the
current completed end-to-end evidence.

The project implements only the fixed FSL behavior required by SimNIBS 4.6
`dwi2cond`; it is not a general FSL command replacement. Wrapped phase requiring
PRELUDE, arbitrary TOPUP/EDDY/FNIRT configurations, and silent multishell
single-tensor fitting remain outside the supported contract. Stage-level timing
results do not establish a universal full-workflow speed ratio, and no HCP
nonlinear Python/FSL speed claim is published without a successful same-boundary
FSL completion.
