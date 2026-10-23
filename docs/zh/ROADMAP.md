# 项目状态

[English](../en/ROADMAP.md) | [简体中文](../zh/ROADMAP.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

当前版本提供适用于 SimNIBS 4.6 的单壳层扩散数据到电导率流程：原始 DWI 处理或张量导入、T1 配准、张量重定向、电导率映射和仿真接口。

## 维护重点

- 梯度、张量和图像坐标处理的正确性。
- 依赖与平台兼容。
- 保持数值方法的运行时间和内存优化。
- 清晰的安装、输入和输出说明。

默认使用 CPU。可选 CUDA 加速 FNIRT Hessian 装配和 PPD；LM/PCG 仍在 CPU 执行。[性能报告](BENCHMARKS.md)记录完整流程的耗时与内存。当前工作计划不包含进一步的内存优化。

## 支持范围

项目实现 SimNIBS 4.6 使用的固定预处理与配准配置，每次张量拟合接受一个非零扩散壳层。包裹相位解缠，以及任意 TOPUP、EDDY 或 FNIRT 配置，需要其他工具处理。

功能建议应说明采集或分析场景、预期输出和可复现的验证示例，可通过 [GitHub issues](https://github.com/ayakacxy/dwi2cond-xp/issues) 讨论。已完成的变化见[更新记录](CHANGELOG.md)。
