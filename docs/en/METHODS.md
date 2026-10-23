# Methods

[English](../en/METHODS.md) | [简体中文](../zh/METHODS.md)

[Documentation](README.md) · [Project home](../../README.md)

The pipeline follows the preprocessing and tensor-to-conductivity steps used by SimNIBS 4.6. Image processing, fitting and registration are implemented in Python; FEM uses the installed SimNIBS solver. [Inputs](INPUTS.md) defines supported data and coordinates, and [Usage](USAGE.md) provides commands.

## Fixed preprocessing subset

| Path | Processing |
| --- | --- |
| `nomoco` | Align b0 volumes to make the reference and mask; fit the diffusion volumes without applying motion or distortion correction |
| `legacy` | Two-pass 6/12-DOF motion/eddy correction, transform composition and one final sinc interpolation per original volume |
| GRE | Fixed FLIRT/FUGUE composition using an unwrapped rad/s fieldmap |
| TOPUP | Fixed nine-level `b02b0_nosubsamp.cnf` schedule |
| EDDY | Fixed five-round single-shell motion/quadratic-eddy model, spherical-GP prediction, slice replacement, gradient rotation and optional TOPUP field |

Storage reorientation transforms image axes and, by default, b-vectors, packed tensors and column-major gradient-deviation matrices together. Legacy motion rotation uses the final transforms in the FSL scaled-mm basis. GRE, TOPUP and EDDY use internal radiological coordinates consistently across image handedness. PE options refer to those internal axes.

## DTI fitting

The model uses b=0 plus one nonzero shell. Standalone fitting supports explicit shell selection and a near-zero b-value threshold; raw pipelines use exact b=0.

Two weighted least-squares solves use squared observed signal as the weight for positive measurements, and weight 1 for nonpositive measurements. The first estimates S0. Before the second logarithmic solve, signals are floored at 1% of S0 while the weights remain unchanged. Optional gradient-nonlinearity data adjust directions and b-values voxel by voxel.

Shared float64 normal matrices are reused across both solves. Tensors with largest absolute component at most `1e-8 mm²/s` are recomputed with reference contractions to keep scale-invariant FA stable near zero. Fitting produces tensor, FA, MD, MO, L1–L3, V1–V3, S0, SSE and validity outputs.

`strict-fsl` uses the validated FSL exceptional-value handling. `robust` excludes voxels containing nonfinite signals or no positive signal, writes zero tensors and records them in QA. The unconstrained fit can produce negative eigenvalues or FA above one; it does not impose a positive-semidefinite tensor model.

## Tensor mapping

Affine mapping uses an input-world-to-reference-world transform and inverse voxel sampling. Its orthogonal polar factor is extracted in the source and reference FSL scaled-mm bases, and tensors are reoriented as `R D R^T`. Automatic T1 registration uses the fixed 6/12-DOF FLIRT schedule with an independent rigid QA registration.

Nonlinear registration uses the fixed four-level FNIRT schedule and preservation of principal direction (PPD). Physical PPD differentiates millimetre displacements using target voxel sizes. Tensor rotation and fold QA use the same physical Jacobian. The analytic FNIRT nonlinear Jacobian is retained separately where requested.

Sampling uses header `pixdim` independently of the world affine. Affine and dense-warp paths support different scales; raw and FNIRT paths require matching scales. The physical conductivity basis removes affine-column scales separately and checks orthogonality, supporting oblique unequal-sized voxels and rejecting shear.

### CPU execution and memory

Preparation streams eligible float32 gzip inputs into a shared uncompressed working image. WLS reuses invariant matrices; T1 affine and rigid QA share a read-only pyramid while maintaining independent optimizers. Fine correlation-ratio levels use adaptive FP64 block statistics.

FNIRT shares spline-support traversal and global projections and uses CSR for the LM working Hessian with the existing CPU PCG solver. Iteration limits, damping and stopping rules are retained. PPD and tensor decomposition operate in bounded batches, including neighboring displacement samples at batch boundaries. QA independently checks stored results in blocks and avoids repeated large-array reads. Output writing preserves the last accepted file if a write fails.

### Optional CUDA

`--fnirt-device cuda` assembles spline Hessians with FP64 accumulation using Numba CUDA. Spline geometry is reused within each level; LM and PCG remain on CPU. This backend does not require CuPy.

`--ppd-device cuda` uses CuPy for tensor sampling, Jacobians, eigendecomposition and rotation in bounded batches. Source tensors remain on the device across batches. Float32 sampling and displacement-difference checkpoints are retained, with float64 Jacobian and matrix operations. Derived maps and NIfTI writing remain on CPU.

Both options default to `cpu`. The measured CUDA environment is Linux with an RTX 2080 Ti, CuPy 14.2.0 and CUDA 12.8. Device choices are recorded in stage parameters and QA; a requested CUDA backend reports an error if it cannot run. See [Benchmarks](BENCHMARKS.md) for the actual measurements.

## Workflow and provenance

Stages record parameters, software versions, input hashes, timing, resource use and QA. Cached outputs are checked for structure and content before reuse. Cache hits are reported as reuse. The raw-QA and final nonnegative-threshold fits remain separate, as does the final QA of stored outputs.

The final tensor and its provenance are published to m2m only after QA succeeds. A recoverable publication error restores the accepted pair. External-field QA checks the prepared field on the working grid.

Real FEM executions are not reused from cache because the complete dynamically loaded solver dependency set is not captured. Deterministic dry-run preparation can be cached. Solver records include the inspected SimNIBS modules, numerical dependency versions and native-library hashes.

## Conductivity mapping

Tensor samples are mapped to mesh elements using the selected physical basis and eigensystem. `vn` normalizes the local determinant; `dir` fits one global scale across anisotropic tissues; `mc` uses the same scale and makes each local tensor isotropic. All modes apply their configured eigenvalue, conductivity and anisotropy safety rules.

The default symmetric `eigh` eigensystem provides an orthogonal basis. Literal `eig` behavior is available for upstream comparisons. Formal FEM and lead-field instances receive the selected settings through an instance-local conductivity adapter, leaving the installed SimNIBS modules unchanged.

Nonzero singular VN tensors raise an error unless explicit regularization is requested. All-zero anisotropic `dir`/`mc` tissue cannot define a global calibration scale; the non-calibrated option has its own safety handling. Equations and references are given in [Conductivity models](CONDUCTIVITY.md).
