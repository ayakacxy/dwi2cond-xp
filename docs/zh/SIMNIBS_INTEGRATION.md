# 与 SimNIBS 4.6 配合使用

[English](../en/SIMNIBS_INTEGRATION.md) | [简体中文](../zh/SIMNIBS_INTEGRATION.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 头模型

仿真命令读取 SimNIBS 4.6 CHARM 生成的完整 `m2m_<subject>` 目录，并检查头网格、
T1、`final_tissues`、EEG 电极帽。各向异性模式还要求 T1 网格上的六分量张量。

## 固定电极组合

`simulate-tdcs` 创建一个 SimNIBS `SESSION`，默认使用 Pardiso。每种电导率模式
分别保存输出目录和 JSON 清单。体素输出为一个末维顺序是 `Ex,Ey,Ez` 的向量电场 NIfTI。

体素映射仅保留 WM、GM、CSF（标签 1、2、3）。映射后的严格掩膜将其余体素置零，
排除颅骨、头皮、电极和颅外组织。

## 先导场

`simulate-leadfield` 以相同的 `scalar`、`vn`、`dir` 或 `mc` 设置配置 SimNIBS
`TDCSLEADFIELD`，并检查原生 HDF5 数据集形状为 `(N_basis,N_spatial,3)`。电极帽中
第一个电极作为参考，其余电极定义 1 A 基底。

可选 NumPy 导出形状为 `(N_spatial*3,N_basis)`。JSON 清单记录参考电极、活动电极
顺序、单位、空间采样方式、分量顺序、张量、模式和求解器。ROI 与避让区掩膜分别保存
为布尔 NPY 文件，不修改原网格标签。

当前版本已实现并单元测试此接口，尚无真实被试全部电极完整先导场的基准运行记录。
