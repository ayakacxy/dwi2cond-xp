# 更新记录

[English](../en/CHANGELOG.md) | [简体中文](../zh/CHANGELOG.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

按版本记录主要变化。日期标明发布版本，详细数值结果见[验证记录](VALIDATION.md)和[性能报告](BENCHMARKS.md)。

## [Unreleased]

暂无额外算法变化。

## [0.4.0] - 2026-10-14

### 概述

修正梯度与张量几何，提供 CPU 和可选 CUDA 流程加速。

### 修复

- 修正 legacy 运动梯度旋转，以及存储重排中的物理向量/张量分量。
- 修正 GRE/TOPUP/EDDY 手性、物理 PPD 导数和斜置网格的电导率基底。
- 区分 NIfTI 头字段采样尺度与有效世界仿射。
- 将所选特征系统和基底应用到正式 SimNIBS FEM 与先导场消费端。
- 修复 Windows 原地及硬链接 NIfTI 写回。
- 检查准备后的外部场，完成 QA 后再发布张量与来源记录。

### 调整

- 减少重复准备、矩阵装配、样条遍历、图像读取和临时数组。
- 新增显式 `--fnirt-device cuda` 和 `--ppd-device cuda` 后端，CPU 保持默认。
- 保留求解器调度、停止规则及独立的原始/最终 WLS 拟合。
- 整理双语指南，同步安装包内文档。

[版本说明与迁移](RELEASE_NOTES_v0.4.0.md)。

## [0.3.0] - 2026-08-30

### 修复

- 修正 v0.2 审计发现的流程数据流、掩膜、b 向量归一化、张量派生图、预拟合导入和 m2m 发布问题。
- 对齐 FSL/NewNifti 方向规则，固定已验证的 NumPy/Numba 栈，修正 CHARM 标记文件和已解析网格传递。
- 修正 legacy 矩阵基底、BET 射线端点与采样，以及 TOPUP 系数 intent 元数据。
- 新增稳定与字面电导率特征系统、显式奇异 VN 处理及更完整的 FEM 执行标识；第四轮复核后禁用真实 FEM 缓存复用。

第三、第四轮审计及整改报告保留逐项细节。科学源码基线为 `ae5a6a7`，后续文档提交保持该源码不变。

## [0.2.0] - 2026-08-24

### 新增

- 纯 Python 的固定 SimNIBS 4.6 预处理：legacy 运动/涡流校正、GRE 场图、TOPUP 和单壳层 EDDY。
- 自动仿射与固定 FNIRT 非线性配准，以及 PPD 张量重定向。
- 流程记录、缓存和数值参考示例。

科学源码基线为 `20f4092`，随后[算法审计](V0.2.0_ALGORITHM_AUDIT_REPORT_2026-08-25.md)中的问题在下一版处理。

## [0.1.0] - 2026-08-21

### 新增

- 两次 WLS DTI 拟合与可选梯度非线性校正。
- 张量重采样、重定向和 `scalar`/`vn`/`dir`/`mc` 电导率映射。
- SimNIBS 固定电极及先导场接口、电场比较图和 QA 记录。

科学源码基线为 `29f98a4`。

## 发布历史

v0.1.0–v0.3.0 标签于 2026-08-29 重新发布，以修正文档和元数据。第四轮复核后，v0.3.0 源码于 2026-08-30 被替换。原始问题、整改记录及标签源码标识保留用于复现。

[0.4.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.4.0
[0.3.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.3.0
[0.2.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.2.0
[0.1.0]: https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.1.0
