# Third-party notices

## SimNIBS

[SimNIBS 4.6.0](https://github.com/simnibs/simnibs/releases/tag/v4.6.0) is the
validated head-model, conductivity, FEM, and lead-field integration target.
SimNIBS 4.6.0 identifies its license as GPL-3.0-only. This repository does not
redistribute SimNIBS binaries, resources, atlases, or source code. Users install
SimNIBS separately through the documented environment.

The conductivity conversion and integration code in this project was developed
to reproduce the documented behavior and numerical contracts of SimNIBS 4.6.0.
The project is therefore distributed under GPL-3.0-only.

## FSL

[FSL 6.0.4](https://fsl.fmrib.ox.ac.uk/fsl/docs/) was used only as an optional,
local numerical reference for DTI fitting and tensor conventions. FSL is not a
runtime dependency and no FSL source code, binary, model, template, or other
redistributable artifact is included. FSL has separate license terms, including
restrictions that users must review independently.

## Human Connectome Project data

Human Connectome Project data were used for private end-to-end validation. No
source image, volumetric derivative, subject identifier, metadata, or access
credential is distributed in the Python packages or release assets. Two
rendered, identifier-free field-comparison PNGs are included in the repository
as result illustrations. They must retain the HCP acknowledgment. Users who
obtain or redistribute HCP data must accept and follow the
[WU-Minn HCP Open Access Data Use Terms](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).

The validation data were provided in part by the WU-Minn Human Connectome
Project, led by David Van Essen and Kamil Ugurbil and supported by NIH Blueprint
funding and the McDonnell Center for Systems Neuroscience at Washington
University. See the official [HCP citation guidance](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations).

## Python dependencies

NumPy, SciPy, NiBabel, h5py, Matplotlib, tqdm, and optional MNE are external
dependencies and remain under their respective licenses. Their source code and
license texts are available from their upstream projects and package metadata.
