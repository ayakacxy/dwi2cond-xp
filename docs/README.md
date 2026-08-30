# Documentation

## Start here

- [Input contract](INPUT_CONTRACT.md): supported DWI, field, tensor, coordinate,
  and failure contracts.
- [Methods](METHODS.md): preprocessing, registration, DTI fitting, tensor
  mapping, and conductivity methods.
- [SimNIBS integration](SIMNIBS_INTEGRATION.md): CHARM, FEM, output, and
  lead-field contracts.
- [Reproducibility](REPRODUCIBILITY.md): clean installation and validation
  procedure.

## Evidence

- [Validation](VALIDATION.md): numerical, cross-platform, packaging, and
  end-to-end evidence with explicit limits.
- [Historical benchmarks](BENCHMARKS.md): version-bound timing and numerical
  measurements that are not a final-tag end-to-end baseline.
- [FSL reference contract](FSL_REFERENCE_CONTRACT.md): frozen source
  provenance, stage artifacts, manifest fields, and failure semantics.

## Project status

- [Project status](ROADMAP.md): current `v0.3.0` maintenance state, evidence
  requirements for future changes, and explicit non-goals.
- [Changelog and version map](CHANGELOG.md): concise roles, original algorithm
  baselines, and user-visible changes for `v0.1.0`, `v0.2.0`, and `v0.3.0`.
- [v0.2.0 algorithm audit](V0.2.0_ALGORITHM_AUDIT_REPORT_2026-08-25.md):
  workflow and numerical defects that motivated the v0.3.0 correctness release.
- [v0.3.0 fourth independent algorithm re-audit](V0.3.0_FOURTH_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REPORT_2026-08-30.md):
  final released-tree review of matrix handedness, BET sampling, conductivity
  eigensystems, FEM cache closure, and TOPUP coefficient artifacts.
- [v0.3.0 fourth remediation](V0.3.0_FOURTH_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REMEDIATION_2026-08-30.md):
  issue-by-issue closure and the real-FSL, regression, package, and publication
  gates for the replacement release.
- [Release process](RELEASE_PROCESS.md): version, tag, asset, provenance, and
  post-publication verification gates.

Superseded intermediate v0.3.0 audit snapshots are intentionally absent from
the current documentation set. Their exact contents remain recoverable from the
original algorithm baseline commit `0464486` and Git history.

## Community

- [Contributing](CONTRIBUTING.md)
- [Security and medical-data reporting](SECURITY.md)
- [Support](SUPPORT.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
