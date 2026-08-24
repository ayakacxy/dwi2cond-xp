# Documentation

- [Input contract](INPUT_CONTRACT.md): required preprocessing, coordinates, and tensor layout.
- [Methods](METHODS.md): DTI fitting, tensor mapping, and conductivity modes.
- [SimNIBS integration](SIMNIBS_INTEGRATION.md): CHARM, FEM, output, and lead-field contracts.
- [Validation](VALIDATION.md): numerical and end-to-end evidence with explicit limits.
- [Benchmarks](BENCHMARKS.md): timing boundaries and hardware-specific results.
- [Overall completion and performance report](DWI2COND_COMPLETION_AND_PERFORMANCE_REPORT.md): current P0--P11 status, numerical fidelity, performance boundaries, hotspots, and `v0.2.0` release gates.
- [End-to-end 10x FSL feasibility report](DWI2COND_10X_PERFORMANCE_FEASIBILITY_REPORT.md): deferred `v0.3.0` performance roadmap with branch-specific Amdahl budgets, nonlinear bottlenecks, and a fair benchmark contract; it is not a `v0.2.0` release gate.
- [Reproducibility](REPRODUCIBILITY.md): clean-environment and validation procedure.
- [FSL subset reimplementation plan](FSL_SUBSET_REIMPLEMENTATION_PLAN.md): staged contracts for replacing only the FSL behavior used by SimNIBS 4.6 `dwi2cond`.
- [FSL reference contract](FSL_REFERENCE_CONTRACT.md): frozen script provenance, stage artifacts, manifest fields, and failure semantics.
- [Contributing](CONTRIBUTING.md), [security](SECURITY.md), [support](SUPPORT.md), and [code of conduct](CODE_OF_CONDUCT.md).
- [Changelog](CHANGELOG.md).
