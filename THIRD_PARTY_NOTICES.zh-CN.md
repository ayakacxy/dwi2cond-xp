# 第三方声明

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

[文档目录](docs/zh/README.md) · [项目首页](README.zh-CN.md)

## SimNIBS

[SimNIBS 4.6.0](https://github.com/simnibs/simnibs/releases/tag/v4.6.0) 是已验证的头模型、
电导率、FEM 和先导场集成版本，声明许可证为 GPL-3.0-only。本仓库不分发 SimNIBS
二进制文件、资源、图谱或源码；用户需按环境说明另行安装。

本项目的电导率转换和集成代码旨在复现 SimNIBS 4.6.0 文档所述的行为与数值方法，
因此项目使用 GPL-3.0-only 许可证。

## FSL

[FSL 6.0.4](https://fsl.fmrib.ox.ac.uk/fsl/docs/) 仅作为可选的本地数值参考，用于
SimNIBS 4.6 固定预处理子集的 DTI 拟合、图像操作、运动/涡流校正、GRE/TOPUP/EDDY、
线性/FNIRT 配准、Jacobian 和 PPD 张量重定向。FSL 不是运行依赖，仓库不包含其源码、
二进制文件、模型、模板或其他分发材料。FSL 有独立的许可条款及限制，用户应自行查阅。

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

NumPy、SciPy、NiBabel、Numba、threadpoolctl、h5py、Matplotlib、tqdm 和可选 MNE
均为外部依赖，保留各自许可证。源码和许可文本可从上游项目及包元数据获取。
