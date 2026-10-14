# Project Status

[English](../en/ROADMAP.md) | [简体中文](../zh/ROADMAP.md)

[Documentation](README.md) · [Project home](../../README.md)

`v0.4.0` is the stable release dated 2026-10-14. It includes
coordinate/orientation repairs and component optimizations, with complete
CPU/GPU comparisons against the published v0.3.0 wheel and official FSL subscripts. See [v0.4.0 release notes](RELEASE_NOTES_v0.4.0.md)
for migration and [Benchmarks](BENCHMARKS.md) for the measured scope.

## Stable release line

The supported line is the fixed, pure-Python subset used by SimNIBS 4.6
`dwi2cond`: raw or pre-fitted single-shell tensor workflows, fixed legacy/GRE/
TOPUP/EDDY preprocessing, affine and nonlinear DTI-to-T1 registration, PPD
tensor reorientation, `scalar`/`vn`/`dir`/`mc` conductivity, and SimNIBS-facing
FEM and lead-field interfaces. FSL remains optional for numerical reference
comparisons and is not a runtime dependency.

Maintenance work may include confirmed correctness fixes, dependency and
platform compatibility, packaging/security upkeep, and documentation that
keeps public claims aligned with tagged evidence. Stable tags are normally
immutable. An explicitly authorized historical-documentation correction must
preserve the original tag object and commit IDs in a backup ref and project
ledger, keep algorithm source unchanged, rebuild the release assets, and
publish a verifiable rollback map.

## Evidence policy

Reorganizing documentation does not require repeating completed experiments.
A historical result remains usable when it is tied to its tagged source,
fixture, environment, metric, and timing or numerical boundary. It should be
rerun only when a new claim depends on a changed implementation or environment,
when the original evidence cannot be reconstructed, or when an inconsistency
affects the conclusion.

The current release evidence therefore remains the basis for the `v0.1.0` to
`v0.3.0` history. Detailed fixture-specific numerical and timing results remain
in [Benchmarks](BENCHMARKS.md); they should be read as measured results for
their stated boundaries, not as universal hardware or input guarantees.

## v0.4.0 completed work

Completed changes cover legacy gradients, storage-axis components,
GRE/TOPUP/EDDY handedness, physical PPD derivatives, conductivity bases and
QA-before-publication. Optimizations reduce repeated work and large-array
overhead in raw preparation, T1 registration, FNIRT, PPD, QA and output writing.
CPU is the default; explicit CUDA backends accelerate FP64 Hessian assembly
and PPD. LM/PCG remains on CPU.

The completed same-input comparison measured 205.452 s on CPU and 163.342 s
on GPU, against 564.406 s for published v0.3.0 and 1290.073 s for FSL.
These are single warm-cache observations on one Linux input with FEM off,
qualified by differing output sets and core use. Stage-level numerical checks
and CPU/GPU end-to-end artifact checks are recorded separately; no repeated-run
dispersion or universal speed guarantee is inferred from the four-way table.

Further memory optimization was stopped at the user's request. The final source
retains the earlier verified speed and numerical behavior. Historical
v0.3.0 test counts and coverage remain attached to that version; v0.4.0
validation is recorded separately in the release notes.

## Scope boundaries

- The project remains an independent implementation of the fixed SimNIBS 4.6
  `dwi2cond` FSL subset, not a general FSL command replacement.
- Strict tensor fitting remains single-shell; wrapped phase requiring PRELUDE
  and arbitrary TOPUP/EDDY/FNIRT configurations are outside the supported
  behavior.
- No universal end-to-end speedup is claimed. Performance statements remain
  attached to the exact fixture and measurement boundary that produced them.
