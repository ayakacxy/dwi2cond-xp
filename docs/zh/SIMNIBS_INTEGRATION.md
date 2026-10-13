# SimNIBS 4.6 集成

[English](../en/SIMNIBS_INTEGRATION.md) | [简体中文](../zh/SIMNIBS_INTEGRATION.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 头模型

仿真命令读取由 SimNIBS 4.6 CHARM 生成的完整 `m2m_<subject>` 目录，并检查被试网格、T1、`final_tissues` 和 EEG 电极帽文件。各向异性模式还需要 T1 网格上的六分量张量。

## 固定电极配置

`simulate-tdcs` 构建一个 SimNIBS `SESSION`，默认使用 Pardiso。每种电导率模式使用单独的输出目录和 JSON 清单。正式体素输出是一个向量电场 NIfTI，最后一维为 `Ex,Ey,Ez`。

体数据映射使用白质、灰质和脑脊液标签 1、2、3；映射后再次施加严格掩膜，将这些标签以外的体素归零，排除颅骨、头皮、电极和颅外组织。

## 先导场

`simulate-leadfield` 使用相同的 `scalar`、`vn`、`dir`、`mc` 电导率设置来配置 SimNIBS `TDCSLEADFIELD`。原生 HDF5 数据集必须符合 `(N_basis,N_spatial,3)` 排列。电极帽中的第一个电极为参考电极，其余电极各对应一个 1 A 基。

可选 NumPy 导出的形状为 `(N_spatial*3,N_basis)`。JSON 清单记录参考电极、活动电极顺序、单位、空间数据类型（网格单元或节点）、分量顺序、张量、模式和求解器。ROI 与避让区掩膜分别保存为布尔 NPY 文件，不修改原始网格标签。

`v0.1.0` 已提供此接口并通过单元测试，但没有完成真实被试的完整全电极先导场性能验证。
