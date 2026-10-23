# 第三方声明

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

[文档目录](docs/zh/README.md) · [项目首页](README.zh-CN.md)

## SimNIBS

[SimNIBS 4.6.0](https://github.com/simnibs/simnibs/releases/tag/v4.6.0) 是本项目的电导率、FEM 和先导场集成版本，采用 GPL-3.0-only。项目的电导率转换与集成实现依据其公开行为和数值方法，使用 GPL-3.0-only 许可。

SimNIBS 由用户另行安装，本软件包不分发其二进制、资源、图谱或源码。

## FSL

[FSL 6.0.4](https://fsl.fmrib.ox.ac.uk/fsl/docs/) 用于可选的本地数值对照，覆盖固定的 DTI 拟合、预处理、配准和 PPD 配置。它不是运行依赖，本软件包不分发 FSL 源码、二进制、模型或模板。FSL 的独立许可条款适用于用户自行安装的参考工具。

## Human Connectome Project 数据

Human Connectome Project 数据用于私有的完整流程验证。Python 包和发布文件不分发
原始影像、体数据衍生结果、被试标识、元数据或访问凭据。仓库保留两张已渲染、无被试
标识的电场对比 PNG，作为结果示意；使用时须保留 HCP 致谢。获得或再分发 HCP 数据的
用户须接受并遵守
[WU-Minn HCP 开放访问数据使用条款](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)。

验证数据部分由 WU-Minn Human Connectome Project 提供，该项目由 David Van Essen
和 Kamil Ugurbil 领导，获 NIH Blueprint 及华盛顿大学 McDonnell Center for Systems
Neuroscience 支持。另见[HCP 官方引用说明](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)。

## Python 依赖

NumPy、SciPy、NiBabel、Numba、threadpoolctl、h5py、Matplotlib、tqdm 和可选 MNE 均作为外部依赖安装，保留各自许可。源码和许可文本可从上游项目及各包元数据获取。
