# SimNIBS 4.6 集成

[English](../en/SIMNIBS_INTEGRATION.md) | [简体中文](../zh/SIMNIBS_INTEGRATION.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 头模型

仿真命令使用 SimNIBS 4.6 CHARM 生成的完整 `m2m_<subject>` 目录，检查被试网格、T1、`final_tissues` 和 EEG 电极帽文件。各向异性模式还要求 T1 网格上的六分量张量。

## 固定电极配置

`simulate-tdcs` 创建一个 SimNIBS `SESSION`，默认求解器为 Pardiso。每种电导率模式有独立输出目录和 JSON 运行记录。体素输出为一个末维依次是 `Ex,Ey,Ez` 的向量电场 NIfTI。

被试体积映射使用标签 1、2、3（白质、灰质、脑脊液）。映射后将这些标签之外的体素严格置零，排除颅骨、头皮、电极和颅外组织。

## 先导场

`simulate-leadfield` 为标量、`vn`、`dir` 或 `mc` 模式配置 SimNIBS `TDCSLEADFIELD`。原生 HDF5 数据集按 `(N_basis,N_spatial,3)` 检查。电极帽中的第一个电极作为参考，其余电极各定义一个 1 A 基函数。

可选 NumPy 导出形状为 `(N_spatial*3,N_basis)`。JSON 记录参考电极、激活电极顺序、单位、空间采样方式、分量顺序、张量、模式和求解器。ROI 与避让掩膜各自保存为布尔 NPY 文件，不改动原始网格标签。

本版本已实现并单元测试该接口，但验证记录中没有完整被试的全电极先导场基准。
