# SimNIBS 4.6 integration

[English](../en/SIMNIBS_INTEGRATION.md) | [简体中文](../zh/SIMNIBS_INTEGRATION.md)

[Documentation](README.md) · [Project home](../../README.md)

Simulation commands use a complete SimNIBS 4.6 CHARM head model in `m2m_<subject>`. The tensor workflow itself can run without SimNIBS; mesh conductivity, FEM and lead fields require exactly SimNIBS 4.6.0.

## Head model

The simulation inputs are the resolved head mesh, T1, `final_tissues.nii.gz` and an EEG cap. Anisotropic modes also need `DTI_coregT1_tensor.nii.gz` on the T1 grid. If the standard mesh name is absent and exactly one mesh exists, it is supplied explicitly to the SimNIBS session.

## Fixed montage

`simulate-tdcs` creates a SimNIBS `SESSION`. Pardiso is the default solver, and each conductivity mode receives its own output directory and manifest. The selected eigensystem and tensor basis are applied through an instance-local adapter; the installed SimNIBS modules are not modified.

The T1-grid vector E-field NIfTI uses final-axis order `Ex,Ey,Ez`. Mapping and the final mask use WM, GM and CSF labels 1, 2 and 3; values outside these tissues are zero. See [Usage](USAGE.md) for a dry-run and solver example.

## Lead field

`simulate-leadfield` configures `TDCSLEADFIELD` for `scalar`, `vn`, `dir` or `mc`. The native HDF5 shape is `(N_basis,N_spatial,3)`. The first cap electrode is the reference; each remaining electrode supplies a 1 A basis.

Optional NumPy export reshapes the result to `(N_spatial*3,N_basis)`. The JSON manifest records electrode order, reference, units, spatial sampling, component order, tensor, mode and solver. ROI and avoid masks are separate Boolean NPY files; mesh labels are preserved.

Interface and format checks cover this path. The published validation does not include a full-subject all-electrode lead-field benchmark. Numerical and FEM results are listed in [Validation](VALIDATION.md).
