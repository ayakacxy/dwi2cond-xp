# 输入文件与坐标

[English](../en/INPUTS.md) | [简体中文](../zh/INPUTS.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 不校正 DWI 的原始数据流程

`preprocess-nomoco` 对应 SimNIBS 4.6 的 `nomoco` 流程。它按 FSL/NewNifti 的
float32 Gram–Schmidt、带符号轴置换和严格相等时的取舍规则调整 NIfTI 存储方向；
只配准精确 `b=0` 的体积，用于生成均值图和脑掩膜。在拟合前才按上游规则将信号限制为
非负，然后执行 WLS。估计出的 b0 变换不会应用到 DWI；此流程不校正 DWI 的运动、涡流
或磁敏感畸变。

DWI、b 值和 b 向量必须对应同一组、同一顺序的体积。严格拟合在确认只有一个非零壳层后
使用全部输入体积，不会自动从多壳数据中挑选壳层。可选 `grad_dev` 必须与原 DWI 的网格
一致，并随 DWI 一起调整存储方向。

只有逐块确认未压缩 `.nii` 为 float32、符合所需 FSL 存储方向、所有值有限且非负时，
才直接通过内存映射读取。其他输入（包括所有 `.nii.gz`）会解码一次，写成供多个进程
共享的未压缩 `DWIforfit.nii`。读取方式及是否生成中间文件记入 `nomoco_qa.json`。

## SimNIBS legacy 校正

`preprocess-legacy` 接受同样按体积对应的 DWI、b 值、b 向量、可选梯度偏差图，以及
明确的单壳拟合参数。为匹配 SimNIBS 4.6，b0 用精确零值筛选，扩散加权均值用严格
`b>0` 筛选，不使用可配置的近零阈值。

流程执行两轮兼容 MCFLIRT 的 6/12 自由度校正，将校正后的扩散均值图配准到 nodif，
将 b0 变换替换为直接估计的 6 自由度变换，再组合全部变换。每个原始体积只执行一次
最终 sinc 重采样。生成两轮参考均值时的临时重采样结果不作为最终输出。失败时流程
终止，不自动改用 `nomoco`。

默认 `--bvec-mode compat46` 逐字节复制原 b 向量文件，与 SimNIBS 4.6 脚本一致。
`--bvec-mode corrected` 则用各最终仿射矩阵的有限应变旋转更新非零壳层 b 向量，
并恢复单位长度。这是需主动选择的独立模式。

可选 `--fieldmap-displacement` 必须是 DWI 网格上形状为 `(X,Y,Z,3)` 的有限位移场，
单位为移动图像世界坐标中的毫米。它与仿射反向采样坐标组合后，一起用于最终插值；
同时必须提供对应的已校正脑掩膜。完整流程也可接收幅度图、已解缠且单位为 rad/s 的场图、
dwell time 和相位编码方向，并在流程内生成位移场与掩膜。

### GRE 场图

`prepare-fieldmap <magnitude> <field_radians_per_second> <b0_brain> <output_directory>`
实现 SimNIBS 4.6 中接收已缩放 rad/s 场图的 GRE/FUGUE 流程。必须提供 `--dwell-ms`
和 `--phase-encoding-direction {x,x-,y,y-,z,z-}`。可选幅度图掩膜与 b0 掩膜各自必须
匹配对应图像的网格。voxel-shift 输出以体素为单位；`displacement_world_mm.nii.gz`
是带符号的 NIfTI 世界坐标反向采样位移，供 `preprocess-legacy --fieldmap-displacement`
使用。此流程不包含 PRELUDE，因此拒绝原始缠绕的 Siemens 相位图。

### TOPUP

`prepare-topup <forward_b0> <reverse_b0> <output_directory>` 使用 SimNIBS 4.6 的固定
`b02b0_nosubsamp.cnf` 配置。两幅输入必须是数值有限、形状和仿射矩阵一致的三维 NIfTI。
`--readout-seconds` 必须为正；相位编码方向仅接受 `x`、`x-`、`y`、`y-`，因为 FSL
6.0.4 TOPUP 不支持 z 方向。

输出包括 Hz 场图、float32 样条系数、每次扫描的六个运动参数、校正后的图像对、联合
掩膜与 QA JSON。保留 `field_coefficients.nii.gz` 和 `movement_parameters.txt`，
另生成 FSL 可读取的 `topup_fieldcoef.nii.gz`/`topup_movpar.txt`。系数 NIfTI 使用
三次 TOPUP 的 intent code `2016`，并在 `intent_p1..p3` 中记录原始体素尺寸；该组合
已通过真实 FSL 6.0.4 `applytopup` 读取检查。计算失败时终止，不自动换用其他场估计器。

### EDDY

`prepare-eddy <dwi> <bvals> <bvecs> <brain_mask> <output_directory>` 使用 SimNIBS 4.6
固定的单壳 EDDY 流程。DWI 与掩膜的空间形状和仿射矩阵必须相同；b 值和 `3xN` 或
`Nx3` b 向量必须按顺序对应 DWI 体积。非零 b 值只能构成一个壳层。
`--readout-seconds` 必须为正，相位编码方向可为 `x`、`x-`、`y`、`y-`、`z`、`z-`。
可选 `--susceptibility-field` 是 DWI 网格上数值有限、单位为 Hz 的 NIfTI。

命令默认使用预测信号替换异常切片，并记录确定性的随机种子。输出包括校正后的 DWI、
替换异常值后的数据、每体积 16 个参数、旋转后的 b 向量、异常值与壳层对齐结果、迭代
记录和 QA。`--no-repol` 和 `--no-rigid-shell-alignment` 可关闭对应步骤；优化实现
失败时不会自动切换到其他实现。

`run-pipeline` 收到四维反向相位编码图像时，会调整存储方向，按上游顺序执行刚体配准
和时间平均，再估计 TOPUP；它用 TOPUP 第一幅校正 b0 和上游 BET 参数生成 EDDY 掩膜，
并将完整 TOPUP 输出交给 EDDY。由于 TOPUP 的限制，此组合流程仅接受 x/y 方向。

没有反向相位编码输入时，`run-pipeline --preprocessing-mode eddy` 在流程内用精确
b0 生成方向标准化、配准后的 nodif 和 BET `f=0.2` 掩膜。也可明确提供
`--dwi-brain-mask`；其来源会记入阶段清单，不会在用户不知情时替换掩膜。

## 已预处理 DWI

DWI 应已完成该采集所需的运动、涡流和磁敏感/EPI 畸变校正，b 向量应随运动校正一致地
旋转。DWI、脑掩膜、b 值和 b 向量必须相互对应，体积顺序一致。

默认 DTI 拟合在确认精确 b0 加一个非零壳层后使用全部输入体积。多壳数据应通过
`select-shell`、`fit-dti --shell` 或等效且可追溯的操作明确选壳。独立 `fit-dti` 还
支持用 `--b0-threshold` 处理近零 b 值；这两个选项与 `strict-fsl`/`robust` 无关。
完整原始 DWI 流程保留默认的精确 b0、单壳设置。程序不会自动选取 b1000，也不会
将多个非零壳层直接用于单张量拟合。

可选 `grad_dev` NIfTI 使用 HCP/FSL 九分量约定，空间形状与仿射矩阵须匹配 DWI。

## 扩散张量

张量 NIfTI 的形状为 `(X,Y,Z,6)`，末维依次为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`，所有值必须
有限。应保留拟合产生的有效性掩膜，或由外部张量生成工具提供该掩膜。

`run-prefit-pipeline` 接受符合上游格式的预拟合张量，执行仅改变存储方向的重排和
`fslmaths -tensor_decomp` 对应分解，随后进入与原始 DWI 相同的 T1 配准、输出发布和
QA 步骤，无需原始 DWI。

## 兼容模式

拟合和非线性 PPD 默认使用 `strict-fsl`，保留兼容 FSL 的输入处理；在 FSL 本身会
终止，或有效区域中的折叠/奇异变换无法得到合法张量时，明确报错。可主动选择 `robust`
来排除无效拟合体素，并将不安全的非线性位置填零。所选模式记入阶段参数，不会自动改变。

## 配准到 T1

`register-t1 <dti_directory> <m2m_directory> <output_directory>` 自动执行 SimNIBS 4.6
的线性 T1 配准步骤。默认 `--mode affine` 估计 12 自由度 FA→T1 变换，并单独生成
6 自由度 QA 变换；`--mode rigid` 使用 6 自由度。输入需含 `DTI_tensor.nii.gz` 和
`DTI_FA.nii.gz`，CHARM 目录需含 `T1.nii.gz`、`segmentation/labeling.nii.gz`、
`segmentation/T1_bias_corrected.nii.gz` 和 `final_tissues.nii.gz`。可选
`DTI_sse.nii.gz` 用于 SSE QA。配准失败时终止，不假定图像已对齐。

`register-tensor` 应用调用者提供的 4x4 仿射矩阵，将输入世界坐标映射到参考世界坐标，
并执行有限应变张量重定向；它不估计配准矩阵。只有外部预处理已建立共同世界坐标时，
才可使用 `--assume-aligned`。

`register-t1-nonlinear <fa> <tensor> <reference> <affine_matrix> <output_directory>`
运行固定的 FNIRT 子集和非线性 PPD。必须提供 `--brain-mask`，并按上游顺序在 PPD
之后应用它。输出包括样条系数、稠密形变场、局部 Jacobian、配准后的张量/FA/V1、
有效性掩膜和 QA。折叠或近奇异区域会明确记录；失败时不自动返回仿射结果。

DWI 与 T1 来自同一被试并不代表已经对齐：运动、畸变、分辨率、方向和扫描仪坐标都
可能不同，不能仅据此使用 `--assume-aligned`。

## CHARM 掩膜与最终输出

脑掩膜使用 SimNIBS `dwi2cond` 的标签区间 1–499。最终张量位于 T1 网格，保存为
`m2m_<subject>/DTI_coregT1_tensor.nii.gz`。

即使不运行 FEM，每个 T1 流程也必须有 `final_tissues.nii.gz`。FEM 还要求目录名为
`m2m_<subject>`，并含 EEG 10-10 电极帽和可确定的头网格。若标准 `<subject>.msh`
不存在但目录中只有一个网格，则将该网格明确传入 `SESSION.fnamehead`，确保 dry-run
与真实 SimNIBS 准备步骤使用同一文件。

`tensor-to-mesh` 的 VN 转换默认拒绝非零奇异张量。显式选择
`--vn-singular-policy regularize` 时，先按配置的各向异性比调整特征值，再执行常规
安全修正，并在 QA JSON 的 `regularized_singular_tensors` 中记录修复数量。此选项
只适用于 VN，若与 `dir` 或 `mc` 一起提供会报错。

默认 `--eigensystem-mode stable` 用 `eigh` 获得正交的对称张量基底。显式
`simnibs46-literal` 模式按 SimNIBS 4.6 的 `eig`、取实部、降序排序与重建方式运行，
也保留重复特征值可能产生非正交基底的行为。QA 记录所选模式、最大正交误差和受影响
基底数量。FEM 流程调用已安装的 SimNIBS SESSION，不会替换为此独立转换命令。

形状、仿射矩阵、有限值、有效性掩膜和来源记录检查失败时均明确报错；各向异性流程
不会自动改成标量 FEM。
