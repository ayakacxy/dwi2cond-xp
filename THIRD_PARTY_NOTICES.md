# Third-party notices

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

[Documentation](docs/en/README.md) · [Project home](README.md)

## SimNIBS

[SimNIBS 4.6.0](https://github.com/simnibs/simnibs/releases/tag/v4.6.0) is the conductivity, FEM and lead-field integration target. It is licensed under GPL-3.0-only. This project's conductivity conversion and integration follow its public behavior and numerical methods and are distributed under GPL-3.0-only.

Users install SimNIBS separately. This package does not redistribute its binaries, resources, atlases or source.

## FSL

[FSL 6.0.4](https://fsl.fmrib.ox.ac.uk/fsl/docs/) is used for optional local numerical comparisons of the fixed DTI, preprocessing, registration and PPD paths. It is not a runtime dependency. No FSL source, binary, model or template is distributed in this package. FSL's own license terms apply to the user's reference installation.

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

NumPy, SciPy, NiBabel, Numba, threadpoolctl, h5py, Matplotlib, tqdm and optional MNE are installed as external dependencies and retain their respective licenses. Their source and license texts are available from the upstream projects and package metadata.
