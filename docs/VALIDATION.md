# Validation

This page preserves the evidence available for the historical `v0.2.0`
release line. No HCP, FSL, FEM, or performance experiment was rerun for the
2026-08-29 documentation and release-metadata republication.

## Evidence levels

The results below are separated so that an earlier valid experiment is not
mistaken for a rerun of the republished tag:

- **v0.2 release-line evidence** was produced by the final v0.2 source/test
  line or its public synthetic fixtures.
- **Carried-forward experimental evidence** predates the final v0.2 tag but
  remains relevant to the named DTI, conductivity, or FEM stage.
- **Historical performance evidence** is tied to its recorded implementation,
  input, host, worker count, cache state, and output boundary; it is summarized
  separately in [Benchmarks](BENCHMARKS.md).

## Automated and release-line contracts

Tests cover gradient conventions, weighted fitting, invalid voxels, the fixed
legacy/GRE/TOPUP/EDDY preprocessing branches, linear and nonlinear
registration, tensor reorientation, conductivity modes, SimNIBS object
construction, tissue masking, lead-field axes, plotting, QA, manifests, and
cache validation. The coverage runner also executes the public synthetic
TOPUP/EDDY/FNIRT command-line paths in isolated Numba caches.

The recorded final v0.2 code-freeze gate ran on commit `90a0553`. GitHub
Actions run `32683104733` completed on Linux, macOS, Windows, and in the
package job; Linux reported `535 passed, 7 skipped`, and the merged coverage
gate covered `12,443/12,443` executable statements (`100.00%`). Optional FSL
reference and SimNIBS integration tests skip visibly when their external
prerequisites are unavailable; the coverage threshold is not lowered.

The original tag commit followed that gate with documentation-only corrections.
The 2026-08-29 republication likewise changes documentation and release
metadata, not `src/`, tests, the package version, or the v0.2 algorithm. The
older `144 passed` and `1,644 statements` figures described the v0.1-scale
suite and are not used as v0.2 tag evidence.

The recorded v0.2 publication also passed package construction and
installation, Markdown and version checks, CodeQL, OpenSSF Scorecard,
tracked-file and archive privacy audits, wheel/sdist metadata checks, and
dependency auditing. Its wheel, sdist, CycloneDX SBOM, and checksum file were
downloaded and verified after publication.

## Public synthetic numerical references

The version-specific FSL reference foundation contains ten JSON contracts and
summaries listed in [FSL reference contract](FSL_REFERENCE_CONTRACT.md). A
synthetic DTI fixture compared with FSL 6.0.4 `dtifit --wls --gradnonlin` had
tensor maximum and mean absolute differences of `1.16e-10` and `8.26e-12`.

For the fixed preprocessing fixtures, TOPUP field and corrected-pair relative
L2 errors were `0.00430` and `0.00161`. EDDY detected all four injected bad
slices and no others; mask-interior corrected-DWI and downstream-tensor
relative L2 errors were `0.0035352` and `0.0060714`. These values support the
named SimNIBS 4.6 subsets and frozen inputs, not arbitrary TOPUP or EDDY
configurations.

The public nonlinear fixture recorded warped-FA relative L2 `0.00309871`,
deformation-vector mean/P99/maximum error `0.204/0.603/1.147 mm`, Jacobian
relative L2 `0.0253746`, and common tensor-support relative L2 `0.0445701`.
These are endpoint metrics; coefficient and optimizer-trajectory bitwise
identity with FSL was not claimed.

## Carried-forward real-data and conductivity evidence

On one private HCP b0+b1000 input, the DTI comparison over 5,287,164 valid
values had maximum, mean, and p99 absolute tensor differences `2.43e-6`,
`4.78e-10`, and `2.10e-9`, with relative L2 `4.18e-6`. Rare boundary voxels
dominated the maximum. This supports the recorded WLS stage; it is not a
2026-08-29 rerun of the complete raw-DWI workflow.

On a synthetic sphere mesh, 274,770 conductivity components agreed with
SimNIBS 4.6 to maximum absolute error `0` for `vn` and `2.22e-16` for `dir`
and `mc`. The retained documentation contains the aggregate comparison rather
than a machine-readable comparator manifest, so the result is carried forward
at the named stage boundary.

The same historical subject completed a T1/T2 CHARM head model and real
C3-to-C4 1 mA Pardiso FEM for `scalar`, `vn`, `dir`, and `mc`. All formal
vector E-field NIfTIs were finite; WM/GM/CSF voxels were populated and voxels
outside those labels were zero after the documented mask. On the common
mapping, scalar Pardiso/Hypre maximum absolute difference was `2.98e-8 V/m`
and relative L2 was `6.28e-9`. This establishes a completed historical
four-mode run, not an exact-tag raw-DWI-to-FEM rerun.

## Evidence boundary and successor release

No HCP data or subject derivative is public. Real-data evidence is dominated
by one private subject, and full-subject all-electrode lead-field execution is
not part of the completed evidence. Stage results are not a universal
full-workflow equivalence or speed claim.

The project implements only the fixed FSL behavior required by SimNIBS 4.6
`dwi2cond`; it is not a general FSL replacement. Subsequent audit identified
workflow and numerical-contract defects in v0.2 that were corrected in
`v0.3.0`. The historical measurements remain useful within their stated
boundaries, while new deployments should use `v0.3.0`.
