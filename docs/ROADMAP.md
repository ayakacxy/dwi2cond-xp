# Project Status

[English](en/ROADMAP.md) | [简体中文](zh/ROADMAP.md)

[Documentation](en/README.md) · [Project home](../README.md)

The current release provides the single-shell diffusion-to-conductivity workflow used with SimNIBS 4.6: raw DWI preparation or tensor import, T1 registration, tensor reorientation, conductivity mapping and simulation interfaces.

## Maintenance priorities

- Correctness of gradient, tensor and image-coordinate handling.
- Dependency and platform compatibility.
- Runtime and memory improvements that preserve the numerical method.
- Clear installation, input and output documentation.

CPU is the default. Optional CUDA accelerates FNIRT Hessian assembly and PPD; LM/PCG stays on CPU. The [benchmark report](en/BENCHMARKS.md) records complete-run times and memory use. Further memory optimization is not part of the current work plan.

## Supported scope

The project implements the fixed preprocessing and registration subset used by SimNIBS 4.6. It accepts one nonzero diffusion shell per tensor fit. Wrapped phase unwrapping and arbitrary TOPUP, EDDY or FNIRT configurations require other tools.

Feature proposals should describe the intended acquisition or analysis, expected outputs and a reproducible validation case. Use [GitHub issues](https://github.com/ayakacxy/dwi2cond-xp/issues) to discuss additions. Completed changes are listed in [the changelog](en/CHANGELOG.md).
