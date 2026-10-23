# 输入文件与坐标

[English](../en/INPUTS.md) | [简体中文](../zh/INPUTS.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

流程接受原始单壳层 DWI、预处理后的 DWI，或已拟合的六分量张量。请根据数据已经完成的处理选择入口。

## 所需文件

| 输入 | 要求 |
| --- | --- |
| DWI | 有限值四维 NIfTI；b 值、b 向量与图像体积顺序一致 |
| 梯度 | 每个体积对应一个 b 值和一个向量；b 向量支持 `3×N` 或 `N×3` |
| 脑掩膜 | 与被掩膜图像的空间形状和仿射矩阵一致 |
| 梯度偏差 | 可选 HCP/FSL 九分量 `grad_dev`，位于原始 DWI 网格 |
| 拟合张量 | 有限值 `(X,Y,Z,6)` NIfTI，分量为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz` |
| 头模型 | SimNIBS 4.6 CHARM 生成的 `m2m_<subject>` 目录，包含 T1 和分割结果 |

张量模型拟合精确 b=0 体积及一个非零壳层。多壳层数据请通过 `select-shell` 或 `fit-dti --shell` 显式选壳。独立 `fit-dti --b0-threshold` 可纳入近零 b 值；原始 DWI 流程按精确 b=0 选择。这些设置独立于 `strict-fsl` 和 `robust` 拟合策略。

## 不进行校正的原始 DWI

`preprocess-nomoco` 构建对齐的 b0 均值和脑掩膜后拟合 DWI，不把 b0 变换应用到扩散体积，也不校正其运动、涡流或磁敏感畸变。仅在这种处理方式适合采集数据时选择此入口。

存储重定向采用 FSL/NewNifti 的 float32 Gram–Schmidt 和带符号轴置换规则。物理分量处理同时把 b 向量和列优先的 `grad_dev` 矩阵变换到重定向后的 FSL 基底。非负信号阈值在最终拟合前应用。

方向已规范、值有限且非负的未压缩 float32 `.nii`，经过分块检查后可直接 mmap 读取。其他输入（包括 `.nii.gz`）会一次性规范化为 `DWIforfit.nii`，所选方式记录在 `nomoco_qa.json`。

## 运动和畸变校正

### Legacy 校正

`preprocess-legacy` 执行两轮 6/12 自由度校正，将校正后的扩散均值对齐到 nodif，并使用直接的 6 自由度 b0 配准。最后组合变换，对每个原始体积执行一次 sinc 插值。

默认 `--bvec-mode corrected` 在 FSL scaled-mm 基底中应用最终运动仿射的有限应变旋转，再归一化非零壳层向量。`compat46` 跳过运动旋转；存储轴变化仍会变换向量，除非选择 `--component-mode simnibs46-literal`，或存储变换本身为恒等。

### GRE 场图

`prepare-fieldmap` 接受幅度图、已解缠且单位为弧度/秒的场图，以及 b0 脑图像。需指定 `--dwell-ms` 和相位编码方向，可选掩膜应与各自图像网格一致。包裹的 Siemens 相位需先由外部工具解缠。

生成的位移形状为 `(X,Y,Z,3)`，单位为带符号的 NIfTI 世界坐标毫米，网格必须与 DWI 一致。通过 `preprocess-legacy --fieldmap-displacement` 传入时，应同时提供其校正掩膜；完整流程可直接从 GRE 输入准备这两项输出。

### 反向相位编码 TOPUP 和 EDDY

`prepare-topup` 接受形状和仿射矩阵一致的两份有限值三维 b0 图像，读出时间必须为正。支持 `x`、`x-`、`y`、`y-`，对应固定的 FSL 6.0.4 TOPUP 配置。

`prepare-eddy` 接受 DWI、对应梯度及同网格脑掩膜，要求一个非零壳层和正读出时间。不使用 TOPUP 时，EDDY 也支持 `z`、`z-`。可选磁敏感场应位于 DWI 网格、单位为 Hz，且所有值有限。

对于反向相位编码四维序列，`run-pipeline` 完成存储重定向、刚体对齐和平均，估计 TOPUP，从第一份校正 b0 生成 EDDY 掩膜，再把 TOPUP 结果传入 EDDY。此组合流程支持 x/y 方向。没有反向序列时，流程构建精确 b0 均值和 BET `f=0.2` 掩膜；显式 `--dwi-brain-mask` 可覆盖该掩膜，来源会写入阶段记录。

PE 标记指存储重定向后的 FSL 内部轴，不会自动转换原始 NIfTI/BIDS 的 `i/j/k` 方向。命令见[使用示例](USAGE.md)。

## 预处理 DWI 和拟合张量

向 `fit-dti` 提供的数据应已完成采集所需的运动和畸变校正，梯度应包含相应运动旋转。整个预处理中，DWI、梯度和掩膜应保持对应。

`run-prefit-pipeline` 导入六分量张量，完成存储重定向和张量分解，再进入 T1 配准与 QA。若拟合器提供有效性掩膜，应一并保留。物理分量处理会在存储轴变化时保持张量方向。

## 坐标和兼容选项

| 选项 | 默认值 | 对照行为 |
| --- | --- | --- |
| `--bvec-mode` | `corrected` | `compat46` 跳过运动引起的旋转 |
| `--component-mode` | `physical` | `simnibs46-literal` 仅重排存储、不变换分量 |
| `--jacobian-mode` | `physical` | `simnibs46-literal` 使用逐体素 PPD 导数 |
| `--basis-mode` | `physical` | `simnibs46-literal` 使用上游仿射归一化 |
| `--eigensystem-mode` | `stable` | `simnibs46-literal` 使用上游一般特征分解 |

这些选项相互独立。默认有效性策略 `strict-fsl` 会在规定情况下对无效拟合或不安全非线性变换报错。可选 `robust` 排除无效拟合体素，对不安全非线性位置填零，并记录到 QA。两种策略均不会改变坐标选项。

## T1 对齐

`register-t1` 从 `DTI_FA.nii.gz` 估计仿射并将 `DTI_tensor.nii.gz` 映射到 T1，需要 CHARM 的 `T1.nii.gz`、`segmentation/labeling.nii.gz`、`segmentation/T1_bias_corrected.nii.gz` 和 `final_tissues.nii.gz`。可选 `DTI_sse.nii.gz` 用于 SSE QA。默认仿射配准使用 12 自由度，并独立估计 6 自由度 QA 变换；刚体模式使用 6 自由度。

`register-tensor` 应用外部提供的输入世界坐标到参考世界坐标的 4×4 仿射，不负责估计变换。只有外部处理已经建立共同世界坐标时才能使用 `--assume-aligned`；同一被试的图像不一定已经对齐。

`register-t1-nonlinear` 执行固定 FNIRT 调度和 PPD 重定向。必须提供脑掩膜，并在 PPD 后应用。输出包含系数、形变、Jacobian、张量/FA/V1、有效性掩膜和 QA；折叠或近奇异位置按所选有效性策略处理。

FSL 采样使用头字段 `pixdim`，世界坐标使用有效仿射。仿射和稠密形变路径支持两者尺度不同；原始预处理以及 FNIRT 估计/系数展开要求尺度一致。使用这些路径前，应将不一致的输入重采样到明确网格。

物理 PPD 按目标体素尺寸计算毫米位移导数。两种 PPD 模式的折叠 QA 都使用物理 Jacobian。物理电导率基底分别归一化仿射列，并拒绝剪切。

## CHARM 输出和电导率

T1 流程均要求 `final_tissues.nii.gz`，包括关闭 FEM 的运行。最终张量位于 T1 网格，写入 `m2m_<subject>/DTI_coregT1_tensor.nii.gz`。扩散脑掩膜采用 CHARM 标签区间 1–499。

FEM 还需要 EEG 帽文件和明确的头网格。当标准被试网格名不存在且目录中只有一份网格时，流程会将其显式传给 SimNIBS，见[SimNIBS 集成](SIMNIBS_INTEGRATION.md)。

`tensor-to-mesh` 默认对非零奇异 VN 张量报错；`--vn-singular-policy regularize` 可显式修复其谱，再执行安全处理并记录数量。此选项只适用于 VN。各模式的策略和公式见[电导率模型](CONDUCTIVITY.md)。
