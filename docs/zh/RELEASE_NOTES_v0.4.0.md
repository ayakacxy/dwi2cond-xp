# v0.4.0 发布说明

[English](../en/RELEASE_NOTES_v0.4.0.md) | [简体中文](../zh/RELEASE_NOTES_v0.4.0.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md) · [更新记录](CHANGELOG.md)

发布日期：2026-10-14。

本版修正梯度和张量坐标处理，并加速预处理、配准和 QA。默认使用 CPU，可选 CUDA 后端加速 FNIRT Hessian 装配和 PPD。

## 算法修正

- Legacy 运动校正在最终运动变换使用的 FSL 基底中旋转梯度。
- 存储重排同时变换图像轴、b 向量、张量分量和列优先梯度偏差矩阵。
- GRE、TOPUP 和 EDDY 在内部放射学网格边界统一处理正负行列式图像存储。
- 仿射和稠密形变采样区分头字段 `pixdim` 与世界仿射；要求尺度一致的路径会对不一致输入报错。
- PPD 和折叠 QA 使用包含目标体素尺寸的物理 Jacobian。
- 电导率映射分别归一化仿射列，并将所选基底和特征系统应用到正式 SimNIBS 消费端。
- Windows 支持将 NIfTI 写回输入文件或其硬链接。
- 外部场 QA 使用准备后的场，最终 QA 完成后再发布 m2m 张量和来源记录。

## 性能变化

原始数据准备和 WLS 复用不变计算，减少大数组复制；仿射与刚体 QA 配准共享只读金字塔。FNIRT 共享样条遍历，以 CSR 保存 LM 工作矩阵，沿用 CPU PCG。PPD、分解和 QA 按有限大小的批次处理。

可选 CUDA 执行 FP64 Hessian 累加，以及分批 PPD 采样、Jacobian、特征分解和旋转。LM/PCG 和文件写出仍在 CPU 执行。

## 从 v0.3.0 迁移

| 设置 | 默认值 | 显式对照选项 |
| --- | --- | --- |
| 梯度运动旋转 | `--bvec-mode corrected` | `--bvec-mode compat46` |
| 存储分量 | `--component-mode physical` | `--component-mode simnibs46-literal` |
| PPD Jacobian | `--jacobian-mode physical` | `--jacobian-mode simnibs46-literal` |
| 张量到世界坐标基底 | `--basis-mode physical` | `--basis-mode simnibs46-literal` |
| 电导率特征系统 | `--eigensystem-mode stable` | `--eigensystem-mode simnibs46-literal` |
| FNIRT Hessian 设备 | `--fnirt-device cpu` | `--fnirt-device cuda` |
| PPD 设备 | `--ppd-device cpu` | `--ppd-device cuda` |

稳定特征系统此前已是默认值，本版将其应用到正式 SimNIBS 消费端。坐标处理修正会改变受影响的输出。

`compat46` 只跳过运动旋转；物理分量模式下，存储变化仍会变换梯度。字面 PPD 保留物理折叠 QA。各对照选项独立于 `strict-fsl`/`robust` 有效性策略，也不会关闭 GRE/TOPUP/EDDY 或发布修正。完整约定见[输入说明](INPUTS.md)。

## CPU 与可选 GPU

在 Python 3.11 中安装 wheel；SimNIBS 流程要求 4.6.0。[安装说明](../../README.zh-CN.md#-安装)涵盖已有和新建环境。

默认流程使用 CPU，启用两个 CUDA 后端的命令为：

```bash
dwi2cond-xp run-pipeline raw_dwi.nii.gz bvals bvecs m2m_subject workflow_gpu --fnirt-device cuda --ppd-device cuda --workers 8
```

FNIRT 需要 Numba CUDA，PPD 还需要 CuPy。基础包不安装可选 GPU 库；设备选项也适用于 `register-t1-nonlinear` 和 `run-prefit-pipeline`。

## 完整流程性能

在 Linux、8 核 Xeon Gold 6230 资源预算和 RTX 2080 Ti 上，一次已有缓存的 HCP 单壳层运行耗时为 **CPU 205.452 秒**、**CUDA 163.342 秒**。主机 RSS 峰值分别为 **6.498 GiB**、**6.723 GiB**，CUDA 运行的 GPU 分配峰值为 **8.834 GiB**。

运行包含梯度偏差校正、`nomoco`、非线性 T1 配准、QA 和 m2m 发布，关闭 FEM；未计时默认 legacy 预处理路径。[性能报告](BENCHMARKS.md)提供完整计时范围、输入、数值差异及 FSL 和前版对照。

## 发布验证

Linux 检查完成 **960 项测试、无跳过**，包含配置好的真实 FSL 和 SimNIBS 探针，覆盖 **15,364/15,364 条语句**。CPU/CUDA 对照的 64 项产物有 61 项逐字节一致，脑内张量相对 L2 为 `1.8693e-9`。平台结果和数值参考示例见[验证记录](VALIDATION.md)。
