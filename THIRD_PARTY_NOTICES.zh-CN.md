# 第三方说明

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES.zh-CN.md) · [项目首页](README.zh-CN.md)

## SimNIBS

[SimNIBS 4.6.0](https://github.com/simnibs/simnibs/releases/tag/v4.6.0) 是本项目验证过的头模型、电导率、FEM 和先导场集成版本。SimNIBS 4.6.0 声明其许可证为 GPL-3.0-only。本仓库不再分发 SimNIBS 二进制文件、资源、图谱或源码；用户应按安装说明单独安装。

本项目的电导率转换和集成代码以复现 SimNIBS 4.6.0 文档所述行为和数值结果为目标，因此本项目按 GPL-3.0-only 分发。

## FSL

[FSL 6.0.4](https://fsl.fmrib.ox.ac.uk/fsl/docs/) 仅用作可选本地数值参考，覆盖 SimNIBS 4.6 使用的固定预处理功能：DTI 拟合、图像操作、运动/涡流校正、GRE/TOPUP/EDDY、线性/FNIRT 配准、Jacobian 和 PPD 张量重定向。FSL 不是运行依赖；本仓库不包含其源码、二进制、模型、模板或其他可再分发文件。FSL 有独立许可条款，用户须自行阅读其中的限制。

## Human Connectome Project 数据

Human Connectome Project 数据用于私有完整流程验证。Python 包和发布附件不分发原始影像、体数据派生物、被试标识、元数据或访问凭据。仓库中的两张电场比较 PNG 是已渲染、无标识符的结果示意图，使用时须保留 HCP 致谢。获取或再分发 HCP 数据的用户须接受并遵守 [WU-Minn HCP 开放数据使用条款](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)。

部分验证数据来自 WU-Minn Human Connectome Project，由 David Van Essen 和 Kamil Ugurbil 领导，得到 NIH Blueprint 经费及华盛顿大学 McDonnell Center for Systems Neuroscience 支持。引用方式见 [HCP 官方引用指南](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)。

## Python 依赖

NumPy、SciPy、NiBabel、Numba、threadpoolctl、h5py、Matplotlib、tqdm 及可选 MNE 是外部依赖，分别遵循各自许可证。其源码和许可文本可在上游项目及包元数据中获取。
