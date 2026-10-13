# 第三方声明

[English](THIRD_PARTY_NOTICES.md) · [项目首页](README.zh-CN.md)

## SimNIBS

[SimNIBS 4.6.0](https://github.com/simnibs/simnibs/releases/tag/v4.6.0) 是本项目已验证的头模型、电导率、有限元及先导场集成版本，其声明的许可证为 GPL-3.0-only。本仓库不重新分发 SimNIBS 的二进制程序、资源、图谱或源码；用户需按照环境说明单独安装 SimNIBS。

本项目的电导率转换和集成代码旨在复现 SimNIBS 4.6.0 文档描述的行为及数值方法，因此本项目采用 GPL-3.0-only 许可证。

## FSL

[FSL 6.0.4](https://fsl.fmrib.ox.ac.uk/fsl/docs/) 仅作为可选的本机 DTI 拟合及张量约定数值参考。FSL 不是运行时依赖；仓库不包含 FSL 源码、二进制程序、模型、模板或其他可分发产物。FSL 有独立的许可条款及限制，用户应自行阅读。

## Human Connectome Project 数据

Human Connectome Project 数据曾用于私有的完整流程验证。Python 包及发布资产不分发原始影像、体数据派生物、被试标识、元数据或访问凭据。仓库仅包含两张已渲染、无被试标识的电场对比 PNG，用作结果展示；这些图片必须保留 HCP 致谢。获取或重新分发 HCP 数据的用户必须接受并遵守 [WU-Minn HCP 开放数据使用条款](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)。

部分验证数据由 WU-Minn Human Connectome Project 提供。该项目由 David Van Essen 和 Kamil Ugurbil 领导，获得 NIH Blueprint 资助及华盛顿大学 McDonnell Center for Systems Neuroscience 支持。引用要求见 [HCP 官方引用说明](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)。

## Python 依赖

NumPy、SciPy、NiBabel、h5py、Matplotlib、tqdm 及可选 MNE 均为外部依赖，分别遵循各自的许可证。其源码及许可证文本可从上游项目和软件包元数据中获取。
