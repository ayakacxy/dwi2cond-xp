# 命令示例

[English](../en/USAGE.md) | [简体中文](../zh/USAGE.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

运行 SimNIBS 4.6 默认的 `legacy + nonlinear` 路径，将通过验证的张量及其来源记录一起写入 m2m 目录。若输出发布过程中发生可恢复的错误，会还原上一组有效文件：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

具有反向相位编码采集时，使用完整的 TOPUP→EDDY 工作流，避免手动拼接各独立命令的产物：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --preprocessing-mode eddy \
  --reverse-phase-encoding reverse_pe_4d.nii.gz \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

符合官方格式的预拟合张量也可进入相同的 T1 映射、输出发布和质量检查流程：

```bash
dwi2cond-xp run-prefit-pipeline \
  DTI_tensor.nii.gz m2m_subject workflow_outputs --workers 8
```

使用八个 worker 运行无需 FSL 的 SimNIBS 4.6 `nomoco` 原始 DWI 路径：

```bash
dwi2cond-xp preprocess-nomoco \
  raw_dwi.nii.gz bvals bvecs nomoco_outputs \
  --workers 8
```

该命令输出 `DTI_tensor.nii.gz`、`DTI_FA.nii.gz`、`DTI_sse.nii.gz`、脑掩膜、有效掩膜及 `nomoco_qa.json`，不生成运动、涡流或场校正产物。对于兼容的未压缩 `.nii`，程序逐块验证后直接使用 mmap 读取；`.nii.gz` 则只解压一次，生成共享的未压缩中间文件。实际读取方式会记录在 QA 中。

运行 legacy 校正路径，并旋转梯度以跟随运动校正（v0.4.0 默认）：

```bash
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --workers 8 --bvec-mode corrected
```

该命令输出校正后的 DWI 及均值图、每个体积的最终变换、nodif 与脑掩膜、张量/FA/SSE 和有效性输出，以及 `legacy_qa.json`。
`--bvec-mode compat46` 为上游对照跳过运动引起的梯度旋转；默认的 physical 分量模式
仍会在存储轴改变时转换梯度。若同时需要保留旧分量行为，添加
`--component-mode simnibs46-literal`。迁移细节见[v0.4.0 发布说明](RELEASE_NOTES_v0.4.0.md)。

使用已转换为 rad/s 的场图准备固定的 GRE/FUGUE 校正，并将其带方向符号的世界坐标毫米位移组合到旧版命令中：

```bash
dwi2cond-xp prepare-fieldmap \
  magnitude.nii.gz field_rads.nii.gz nodif_brain.nii.gz fieldmap_out \
  --dwell-ms 0.5 --phase-encoding-direction y- --workers 8
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --fieldmap-displacement fieldmap_out/displacement_world_mm.nii.gz \
  --fieldmap-corrected-mask fieldmap_out/corrected_mask.nii.gz --workers 8
```

原始 Siemens 缠绕相位仍需 PRELUDE 处理，这一固定 rad/s 路径会拒绝该类输入。

在无需 FSL 运行时依赖的情况下，估计固定反向相位编码 TOPUP 场：

```bash
dwi2cond-xp prepare-topup \
  forward_b0.nii.gz reverse_b0.nii.gz topup_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

该命令输出场图、样条系数、运动参数、校正后的成对影像、联合有效掩膜及 `topup_qa.json`。项目原有文件名保持不变；`topup_fieldcoef.nii.gz` 与 `topup_movpar.txt` 组成 FSL 可读取的 `--topup=topup` 文件组，包含官方三次样条 intent 元数据。仅支持 FSL 6.0.4 TOPUP 接受的 x/y 单轴子集；z 方向相位编码会明确报错。

运行固定单壳 EDDY 路径，可选合入 TOPUP 场：

```bash
dwi2cond-xp prepare-eddy \
  raw_dwi.nii.gz bvals bvecs brain_mask.nii.gz eddy_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8 \
  --susceptibility-field topup_out/field_hz.nii.gz
```

该命令输出校正后的 DWI、用预测值替换后的扫描空间数据、每个体积的 16 个参数、旋转后的 b 向量、异常值图、壳层对齐结果、迭代历史及 `eddy_qa.json`。省略 `--susceptibility-field` 时，仍运行相同的固定 EDDY 算法，只是不使用 TOPUP 场。

拟合已预处理的单壳 DWI：

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz \
  --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz \
  --qa-json tensor_fit_qa.json
```

将张量映射到 CHARM T1 网格。下面示例使用 `--assume-aligned` 的前提是外部流程已经完成对齐：

```bash
dwi2cond-xp register-tensor \
  tensor_dwi.nii.gz m2m_subject/T1.nii.gz \
  m2m_subject/DTI_coregT1_tensor.nii.gz \
  --source-mask tensor_valid_mask.nii.gz \
  --assume-aligned \
  --qa-json tensor_registration_qa.json
```

检查 C3→C4 各向异性仿真配置，暂不启动求解器：

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn \
  --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 \
  --cpus 8 --dry-run
```

移除 `--dry-run` 后执行 SimNIBS。分别运行 `scalar`、`vn`、`dir`、`mc`，每种模式使用独立输出目录和清单。各向异性模式缺少张量时会报错，不会自动改用 `scalar`。

每次仿真在 T1 网格上输出一个向量电场 NIfTI，最后一维为 `Ex,Ey,Ez`，模长通过 `sqrt(Ex^2 + Ey^2 + Ez^2)` 计算。体数据严格限于白质、灰质和脑脊液标签 `1,2,3`，排除颅骨、头皮、电极及颅外组织。

## 自动仿射配准

估计 FA→T1 仿射矩阵，并生成配准后的张量与 QA 输出：

```bash
dwi2cond-xp register-t1 dti_outputs m2m_subject t1_registration_outputs \
  --mode affine --workers 8
```

## 比较电场

用四种模式的向量场绘制共用色标的分量图：

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

三行对应 `Ex/Ey/Ez`，四列对应 `scalar/vn/dir/mc`，所有面板共用对称色标；切片按脑
掩膜最大面积选择。
