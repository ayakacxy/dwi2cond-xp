# 命令示例

[English](../en/USAGE.md) | [简体中文](../zh/USAGE.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

以下命令假设已完成安装。将示例文件名替换为实际输入，流程会创建输出目录。文件与坐标要求见[输入说明](INPUTS.md)。

## 原始 DWI 完整流程

默认执行 legacy 校正和非线性 T1 配准，QA 通过后将张量及来源记录写入 m2m。

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

## 反向相位编码数据

有反向相位编码序列时，流程会一起准备 TOPUP 和 EDDY。读出时间单位为秒，PE 方向指 FSL 内部轴。

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --preprocessing-mode eddy \
  --reverse-phase-encoding reverse_pe_4d.nii.gz \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

## 已拟合张量

导入分量顺序为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz` 的张量，进入相同的 T1 映射和 QA 阶段。

```bash
dwi2cond-xp run-prefit-pipeline \
  DTI_tensor.nii.gz m2m_subject workflow_outputs --workers 8
```

## Nomoco 准备

仅在不需要运动、涡流和磁敏感校正，或已由外部处理完成这些校正时使用。

```bash
dwi2cond-xp preprocess-nomoco \
  raw_dwi.nii.gz bvals bvecs nomoco_outputs \
  --workers 8
```

输出包括张量、FA、SSE、脑掩膜、有效性掩膜和 `nomoco_qa.json`。对齐后的 b0 均值用于生成参考，其变换不会应用到扩散体积。

## Legacy 运动与涡流校正

运行两轮校正并按运动旋转梯度：

```bash
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --workers 8 --bvec-mode corrected
```

输出包括校正 DWI、最终逐体积变换、nodif 和掩膜、张量图及 `legacy_qa.json`。`--bvec-mode compat46` 跳过运动旋转；仅在对照上游脚本的存储分量行为时，再添加 `--component-mode simnibs46-literal`。

## GRE 场图

准备已解缠的 rad/s 场图，并将位移传给 legacy 校正：

```bash
dwi2cond-xp prepare-fieldmap \
  magnitude.nii.gz field_rads.nii.gz nodif_brain.nii.gz fieldmap_out \
  --dwell-ms 0.5 --phase-encoding-direction y- --workers 8
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --fieldmap-displacement fieldmap_out/displacement_world_mm.nii.gz \
  --fieldmap-corrected-mask fieldmap_out/corrected_mask.nii.gz --workers 8
```

Dwell 时间单位为毫秒。输出位移使用带符号的世界坐标毫米，并配有校正掩膜。包裹相位需先在外部解缠。

## 独立 TOPUP

从网格一致的 b0 图像估计固定反向相位编码场：

```bash
dwi2cond-xp prepare-topup \
  forward_b0.nii.gz reverse_b0.nii.gz topup_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

输出包含 Hz 场、样条系数、运动参数、校正图像、掩膜和 QA。`topup_fieldcoef.nii.gz` 与 `topup_movpar.txt` 还组成 FSL 可读取的前缀文件组。支持 x、y 两个 PE 轴及其正负方向。

## 独立 EDDY

使用可选的已准备 TOPUP 场运行固定单壳层流程：

```bash
dwi2cond-xp prepare-eddy \
  raw_dwi.nii.gz bvals bvecs brain_mask.nii.gz eddy_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8 \
  --susceptibility-field topup_out/field_hz.nii.gz
```

不使用 TOPUP 时省略 `--susceptibility-field`。输出包括校正及预测替换后的 DWI、每体积 16 个参数、旋转梯度、异常切片与壳层对齐文件、迭代记录和 QA。

## 拟合预处理 DWI

提供同网格掩膜及与处理后体积对应的梯度，梯度偏差选项为可选项。

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz \
  --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz \
  --qa-json tensor_fit_qa.json
```

多壳层数据需显式选择一个壳层；近零 b 值可通过 `--b0-threshold` 纳入，见[输入说明](INPUTS.md)。

## 应用已有对齐

此示例假设外部处理已经对齐 DWI 和 T1 世界坐标：

```bash
dwi2cond-xp register-tensor \
  tensor_dwi.nii.gz m2m_subject/T1.nii.gz \
  m2m_subject/DTI_coregT1_tensor.nii.gz \
  --source-mask tensor_valid_mask.nii.gz \
  --assume-aligned \
  --qa-json tensor_registration_qa.json
```

否则请通过 `--world-transform` 提供输入世界坐标到参考世界坐标的仿射，或使用下一条命令估计配准。

## 自动仿射配准

估计 FA 到 T1 的对齐，并生成配准张量和 QA：

```bash
dwi2cond-xp register-t1 dti_outputs m2m_subject t1_registration_outputs \
  --mode affine --workers 8
```

## 固定电极配置 FEM

准备 C3 到 C4 仿真，此命令不启动求解器：

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn \
  --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 \
  --cpus 8 --dry-run
```

移除 `--dry-run` 后执行求解。`scalar`、`vn`、`dir` 和 `mc` 分别使用独立输出目录。每次运行导出 T1 网格的 `Ex,Ey,Ez` 向量 NIfTI，掩膜限定为 WM/GM/CSF 标签 1、2、3。幅值为 `sqrt(Ex^2 + Ey^2 + Ez^2)`。

## 比较电场

以共同的分量色阶绘制四份向量场：

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

行对应 `Ex/Ey/Ez`，列对应 `scalar/vn/dir/mc`，切片按脑掩膜面积最大的位置选择。先导场输出约定见[SimNIBS 集成](SIMNIBS_INTEGRATION.md)。

## CPU 和 CUDA

默认使用 CPU。非线性配准启用两个可选 CUDA 后端的示例：

```bash
dwi2cond-xp run-pipeline raw_dwi.nii.gz bvals bvecs m2m_subject workflow_gpu --fnirt-device cuda --ppd-device cuda --workers 8
```

FNIRT 需要 Numba CUDA，PPD 还需要 CuPy。相同选项适用于 `register-t1-nonlinear` 和 `run-prefit-pipeline`；LM/PCG 和文件写出仍在 CPU 执行。依赖说明见[复现指南](REPRODUCIBILITY.md)。
