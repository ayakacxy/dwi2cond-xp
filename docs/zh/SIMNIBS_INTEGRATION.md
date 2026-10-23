# 与 SimNIBS 4.6 配合使用

[English](../en/SIMNIBS_INTEGRATION.md) | [简体中文](../zh/SIMNIBS_INTEGRATION.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

仿真命令使用 `m2m_<subject>` 中完整的 SimNIBS 4.6 CHARM 头模型。张量流程本身可在未安装 SimNIBS 时运行；网格电导率、FEM 和先导场要求 SimNIBS 4.6.0。

## 头模型

仿真输入包含明确的头网格、T1、`final_tissues.nii.gz` 和 EEG 帽文件。各向异性模式还需要 T1 网格上的 `DTI_coregT1_tensor.nii.gz`。如果标准网格名不存在且目录中只有一份网格，流程会将其显式传给 SimNIBS session。

## 固定电极配置

`simulate-tdcs` 创建 SimNIBS `SESSION`，默认求解器为 Pardiso，各电导率模式分别使用独立输出目录和记录文件。所选特征系统与张量基底通过实例局部的适配器应用，不修改已安装的 SimNIBS 模块。

T1 网格向量电场 NIfTI 的末轴顺序为 `Ex,Ey,Ez`。映射和最终掩膜使用 WM、GM、CSF 标签 1、2、3，这些组织以外的值为零。Dry-run 和求解示例见[使用说明](USAGE.md)。

## 先导场

`simulate-leadfield` 为 `scalar`、`vn`、`dir` 或 `mc` 配置 `TDCSLEADFIELD`。原始 HDF5 形状为 `(N_basis,N_spatial,3)`，帽文件的第一个电极作为参考，其余每个电极对应 1 A 基底。

可选 NumPy 导出将结果重排为 `(N_spatial*3,N_basis)`。JSON 记录电极顺序、参考电极、单位、空间采样、分量顺序、张量、模式和求解器。ROI 与避让掩膜单独保存为布尔 NPY，网格标签保持不变。

该路径已有接口和格式检查，公开验证记录尚不包含全被试、全电极先导场性能测试。数值与 FEM 结果见[验证记录](VALIDATION.md)。
