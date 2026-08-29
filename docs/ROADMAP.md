# Maintenance Status

`v0.3.0` is the current stable release. The project is in maintenance mode:
there is no active `v0.4.0` development cycle, no active `10x` target, and no
scheduled performance release. The earlier performance cycle was cancelled
before a reliable new baseline was completed; its planning samples do not
define a product claim or future obligation.

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

## If development is explicitly restarted

Any future feature or performance cycle would begin with a new, scoped
decision. Before publishing a numerical or speed claim, it would need to:

1. Freeze the exact reference tag or wheel and a legal, reproducible input.
2. Establish a reliable baseline on a stable host under the same input, output,
   resource, cache, and process boundary used for the candidate.
3. Use repeated paired measurements with dispersion, rather than a fastest
   sample or an interrupted run.
4. Pass stage-level and end-to-end numerical A/B gates before interpreting
   performance.
5. Change one measurable hotspot at a time, retain the reference path, and keep
   optimized-backend failures explicit.
6. Support any GPU or cross-platform claim with runs on the relevant real
   hardware and platform.

These are scientific entry criteria, not a `v0.4.0` plan, release promise,
schedule, or predetermined speed target.

## Scope boundaries

- The project remains an independent implementation of the fixed SimNIBS 4.6
  `dwi2cond` FSL subset, not a general FSL command replacement.
- Strict tensor fitting remains single-shell; wrapped phase requiring PRELUDE
  and arbitrary TOPUP/EDDY/FNIRT configurations are outside the supported
  contract.
- No universal end-to-end speedup is claimed. Performance statements remain
  attached to the exact fixture and measurement boundary that produced them.
