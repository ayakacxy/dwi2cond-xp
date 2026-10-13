# 输入要求

[English](../en/INPUTS.md) | [简体中文](../zh/INPUTS.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 不校正 DWI 的原始数据路径

`preprocess-nomoco` 对应 SimNIBS 4.6 的 `nomoco` 流程：调整 NIfTI 存储方向，将拟合数据转为非负 float32，只配准 b0 体积以计算均值和脑掩膜，然后进行 WLS 拟合。估计出的 b0 变换不会应用到 DWI，也不会估计运动、涡流或磁敏感畸变场。

DWI、b 值和 b 向量必须对应同一组体积，且顺序一致。单张量模型只使用 b=0 和明确选定的一个非零壳层，不会自动将多壳数据一起拟合。可选 `grad_dev` 必须与原始 DWI 的网格一致，拟合前随 DWI 一同调整存储方向。

未压缩 `.nii` 只有在分块检查确认其为 float32、满足 FSL 所需存储方向、数值有限且非负时才直接使用。其他输入（包括所有 `.nii.gz`）会解码一次，写入未压缩的 `DWIforfit.nii`，供各进程通过 mmap 共享。所用方法及是否生成中间文件记录在 `nomoco_qa.json` 中。

## 使用 SimNIBS legacy 校正的原始数据路径

`preprocess-legacy` 接受相同顺序的 DWI、b 值、b 向量、可选梯度偏差图和明确指定的单壳拟合参数。为兼容 SimNIBS 4.6，b0 选择使用严格的零值，`b>0` 均值使用严格大于零的 b 值，不采用可配置的近零阈值。

流程依次执行与 MCFLIRT 对应的两轮 6/12 自由度校正，将校正后的扩散加权均值配准到 nodif，将 b0 变换替换为直接的 6 自由度配准结果，再组合所有变换，对每个原始体积仅做一次最终 sinc 重采样。计算两轮参考均值时的临时重采样不会作为正式输出。该模式失败时直接终止，不切换到 `nomoco`。

默认 `--bvec-mode compat46` 逐字节复制原始 b 向量文件，与 SimNIBS 4.6 legacy 脚本一致。`--bvec-mode corrected` 则根据各最终仿射矩阵的有限应变旋转，对非零壳层 b 向量做旋转并恢复单位长度。这两种模式的科学含义不同，必须明确选择。

可选 `--fieldmap-displacement` 必须是 DWI 网格上形状为 `(X,Y,Z,3)` 的有限位移场，单位为移动图像世界坐标中的毫米。它在唯一一次最终插值前与仿射逆向采样坐标组合。

`prepare-fieldmap <magnitude> <field_radians_per_second> <b0_brain> <output_directory>` 接受已换算为 rad/s 的场图，实现 SimNIBS 4.6 GRE/FUGUE 分支。必须提供 `--dwell-ms` 和 `--phase-encoding-direction {x,x-,y,y-,z,z-}`。可选幅度图与 b0 掩膜必须分别匹配各自图像网格。体素位移输出以体素为单位；`displacement_world_mm.nii.gz` 是带符号的 NIfTI 世界坐标逆向位移场，供 `preprocess-legacy --fieldmap-displacement` 使用。原始 Siemens 缠绕相位不被接受，因为此分支不包含 PRELUDE 相位解缠。

`prepare-topup <forward_b0> <reverse_b0> <output_directory>` 实现 SimNIBS 4.6 固定的 `b02b0_nosubsamp.cnf` 流程。两份输入必须是形状和仿射矩阵相同、数值有限的三维 NIfTI。`--readout-seconds` 必须为正；相位编码方向仅支持 `x`、`x-`、`y`、`y-`，因为 FSL 6.0.4 TOPUP 不支持 z 方向。输出包括 Hz 场、float32 样条系数、每个扫描的六个运动参数、校正图像对、联合掩膜和 QA JSON。失败时终止，不自动换用其他场估计方法。

`prepare-eddy <dwi> <bvals> <bvecs> <brain_mask> <output_directory>` 实现 SimNIBS 4.6 固定的单壳 EDDY 流程。DWI 和掩膜的空间形状及仿射矩阵必须一致；b 值和 3×N b 向量必须与体积顺序一致。非零 b 值必须属于一个壳层，`--readout-seconds` 必须为正，相位编码方向仅支持 `x`、`x-`、`y`、`y-`。可选 `--susceptibility-field` 是 DWI 网格上以 Hz 为单位、数值有限的 NIfTI。默认按预测信号替换异常切片，并记录确定性随机种子。输出校正 DWI、移除异常值的 DWI、每体积 16 个参数、旋转后的 b 向量、异常切片与壳层对齐结果、迭代记录和 QA。`--no-repol` 与 `--no-rigid-shell-alignment` 用于明确关闭对应步骤；计算失败不会触发隐式回退。

## 已预处理 DWI

输入应已完成与采集方式相适应的运动、涡流和磁敏感/EPI 畸变校正。b 向量必须随运动校正一致地旋转。DWI、脑掩膜、b 值和 b 向量必须匹配，体积顺序必须一致。

v0.2 拟合按阈值选取基线图像和一个非零壳层，默认使用 b <= 50 s/mm² 或 |b - 1000| <= 100 s/mm² 的体积，不会把全部非零壳层一起拟合。拟合前请检查所选体积；准备多壳数据时可用 `select-shell` 或其他可追溯步骤明确选择。

可选 `grad_dev` NIfTI 采用 HCP/FSL 九分量约定，空间形状和仿射矩阵必须与 DWI 一致。

## 扩散张量

张量 NIfTI 的形状为 `(X,Y,Z,6)`，末维依次为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`，数值必须有限。有效性掩膜应从拟合步骤传递，或由外部张量生成程序提供。

## 与 T1 对齐

`register-t1 <dti_directory> <m2m_directory> <output_directory>` 自动执行 SimNIBS 4.6 线性 T1 配准流程。默认 `--mode affine` 估计 12 自由度 FA→T1 变换，另做一次独立的 6 自由度配准用于 QA；`--mode rigid` 使用 6 自由度。所需文件是 `DTI_tensor.nii.gz`、`DTI_FA.nii.gz`，以及 CHARM 的 `T1.nii.gz`、`segmentation/labeling.nii.gz`、`segmentation/T1_bias_corrected.nii.gz` 和 `final_tissues.nii.gz`。可选 `DTI_sse.nii.gz` 用于 SSE 检查。配准失败即终止，不会假定输入已经对齐或自动使用 `--assume-aligned`。

`register-tensor` 应用用户提供的“输入世界坐标→参考世界坐标”4×4 仿射矩阵，并进行有限应变张量重定向；它不会估计该矩阵。只有外部预处理已建立共用世界坐标系时，才可明确使用 `--assume-aligned`。

`register-t1-nonlinear <fa> <tensor> <reference> <affine_matrix> <output_directory>` 执行 SimNIBS 4.6 固定 FNIRT 流程和非线性主方向保持（PPD）张量重定向。输出样条系数、稠密形变场、局部 Jacobian、配准后的张量/FA/V1、有效性掩膜和 QA。折叠或近奇异区域会单独报告，失败时不会改用仿射结果。DWI 和 T1 来自同一被试并不代表已对齐：运动、畸变、分辨率、方向和扫描仪坐标仍可能不同。

## CHARM 掩膜与输出

脑掩膜采用 SimNIBS `dwi2cond` 的标签区间 1–499。最终张量位于 T1 网格，写入 `m2m_<subject>/DTI_coregT1_tensor.nii.gz`，供 SimNIBS 使用。

形状、仿射矩阵、有限值、有效性掩膜或数据来源检查不通过时会明确报错。各向异性流程不会在失败后自动改用标量 FEM。
