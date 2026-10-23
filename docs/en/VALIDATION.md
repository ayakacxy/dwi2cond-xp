# Validation

[English](../en/VALIDATION.md) | [简体中文](../zh/VALIDATION.md)

[Documentation](README.md) · [Project home](../../README.md)

The results below identify the release, configuration and processing stage that were tested. Benchmark timings are reported separately in [Benchmarks](BENCHMARKS.md).

## v0.4.0 validation — 2026-10-14

| Check | Result | Configuration |
| --- | --- | --- |
| Linux release suite | 948 main + 12 montage tests passed; no skips | Configured real FSL and SimNIBS probes enabled |
| Statement coverage | 15,364/15,364 statements, 100% | Isolated cold Numba caches and synthetic TOPUP/EDDY/FNIRT CLI runs |
| Linux/macOS/Windows CI | 902 main + 12 montage tests passed per platform | 46 optional FSL/SimNIBS probes skipped without those installations |
| Conductivity consumer adapter | 30 passed with SimNIBS; 18 passed and 12 skipped without it | Includes four independent mathematical reference cases |
| CUDA backend mathematics and errors | 36 device-free tests passed | Separate from the actual GPU run below |

The release also passed package construction, installation, source/version checks, CodeQL and dependency checks. The published wheel and source archive were downloaded and checked against their source and SHA-256 values. Statement coverage measures executed statements, not all possible input behavior.

## CPU and CUDA comparison

The complete Linux/RTX 2080 Ti run used one HCP single-shell input, `nomoco`, gradient-deviation correction, nonlinear registration, QA and m2m publication, with FEM off. The CPU run completed with optional GPU imports blocked.

Of 64 numerical artifacts, 61 were byte-identical between CPU and CUDA. Masks, FNIRT coefficients, dense fields and recorded LM results agreed. Brain-tensor relative L2 was `1.8693e-9`; final FA maximum absolute difference was `8.7172e-7`.

Healthy V1 axial-angle P99 was `1.2074e-6` degrees, ignoring eigenvector sign. This metric requires FA>0.1 in both outputs and relative principal-eigenvalue gap>1e-5; all healthy voxels were covered. Details and timing scope are in [Benchmarks](BENCHMARKS.md).

## Numerical reference cases

The fixed preprocessing and registration paths have synthetic comparisons with real FSL 6.0.4. The following retained results describe their stated fixtures:

| Case | Recorded result |
| --- | --- |
| Storage orientation | High-obliquity counterexample and all 48 signed permutations match the FSL discrete orientation decision |
| WLS with gradient nonlinearity | Synthetic tensor max/mean absolute differences `1.16e-10 / 8.26e-12` |
| GRE transform composition | Nonidentity 90° premat plus one-voxel shift is array-exact on the `9×9×3` trilinear fixture |
| TOPUP | Field and corrected-pair relative L2 `0.00430 / 0.00161` |
| EDDY | All four injected bad slices detected, no additional outliers; interior DWI/tensor relative L2 `0.0035352 / 0.0060714` |
| FNIRT | Corrected-image/field relative L2 `0.001298916 / 0.175488519`; maximum field-component difference `0.994548967 mm` |

The FNIRT result retains a nonzero field difference from the independent FSL optimizer. Coordinate regressions additionally cover storage permutations, both handednesses, non-1-mm PPD, header/affine sampling mismatch, conductivity consumers and failed publication. The unconstrained WLS model permits negative eigenvalues and FA>1.

## FEM and lead-field checks

An earlier private-subject study completed CHARM and real C3-to-C4, 1 mA Pardiso solves for `scalar`, `vn`, `dir` and `mc`. Vector fields were finite and masked to WM/GM/CSF. Scalar Pardiso/Hypre differences on the common mapping were max-absolute `2.98e-8 V/m` and relative L2 `6.28e-9`.

Subsequent FEM composition fixes were checked with real SimNIBS 4.6 `SESSION._prepare()` and construction, adapter and publication tests. The complete subject solve was not repeated after those changes. Lead-field checks cover configuration, reference-electrode and HDF5/NumPy conventions; a complete all-electrode subject benchmark is not included.

## Historical records

The v0.3.0 scientific baseline `ae5a6a7` completed 666 main tests, 12 montage tests and synthetic TOPUP/EDDY/FNIRT CLI paths with 13,663/13,663 statements covered. These counts belong to that source snapshot.

Earlier HCP WLS validation used 881,299 masked voxels: 881,194 valid and 105 zeroed as invalid. Tensor relative L2 was `4.18e-6`; max/mean/P99 absolute differences were `2.43e-6 / 4.78e-10 / 2.10e-9` over 5,287,164 valid values.

The 2026-08-23 nonlinear study recorded field-vector mean/P99/max errors `0.0130 / 0.0626 / 0.1828 mm`, tensor relative L2 `0.00631`, support Dice `0.99962` and V1 mean/P99 angles `0.343 / 3.104°`. Using the same FSL warp reduced PPD tensor relative L2 to `1.51e-6`, localizing most differences to the independently optimized warp.

The dated [v0.2 audit](V0.2.0_ALGORITHM_AUDIT_REPORT_2026-08-25.md), [third v0.3 review](V0.3.0_THIRD_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REPORT_2026-08-29.md) and [fourth review](V0.3.0_FOURTH_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REPORT_2026-08-30.md), together with their remediation records, preserve finding-level evidence. They describe their reviewed source snapshots.

## Scope

Real-data results are dominated by one private subject. The fixed SimNIBS 4.6 subsets, rather than arbitrary FSL command configurations, were tested. No source images or subject derivatives are distributed. Current input restrictions and coordinate choices are listed in [Inputs](INPUTS.md).
