# SimNIBS 4.6 integration

## Head model

The simulation commands consume a complete `m2m_<subject>` directory produced
by SimNIBS 4.6 CHARM. They validate the subject mesh, T1, `final_tissues`, EEG
cap, and, for anisotropic modes, the six-component T1-grid tensor.

## Fixed montage

`simulate-tdcs` constructs one SimNIBS `SESSION`. Pardiso is the default solver.
Each conductivity mode receives a separate output directory and JSON manifest.
The formal voxel output is one vector E-field NIfTI with final axis `Ex,Ey,Ez`.

Subject-volume mapping uses labels 1, 2, and 3 (WM, GM, CSF). A strict
post-mapping mask zeros every voxel outside those labels, excluding skull,
scalp, electrodes, and extracranial tissue.

## Lead field

`simulate-leadfield` configures SimNIBS `TDCSLEADFIELD` with the same scalar,
`vn`, `dir`, or `mc` contract. The native HDF5 dataset is validated as
`(N_basis,N_spatial,3)`. The first cap electrode is the reference and the other
electrodes define 1 A bases.

The optional NumPy export has shape `(N_spatial*3,N_basis)`. Its JSON manifest
records the reference, active-electrode order, unit, spatial grain, component
order, tensor, mode, and solver. ROI and avoid masks are separate Boolean NPY
files; the original mesh labels are not modified.

`v0.1.0` supports and unit-tests this interface but does not claim a completed
full-subject all-electrode lead-field benchmark.
