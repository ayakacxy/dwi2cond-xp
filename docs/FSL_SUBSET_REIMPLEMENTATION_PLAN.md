# SimNIBS 4.6 `dwi2cond` 所需 FSL 子集复现计划

## 阶段进度总览

> **总体进度：11 / 12 个阶段已完成（92%）**
> **当前状态：P0–P10 已完成；P9/M9 已封板，不再追加施工范围**
> **进行中：P11 / M11 — QA、DAG 与发布验收**
> 最后更新：2026-08-24

| 阶段 | 对应里程碑 | 主要内容 | 当前状态 |
|---|---|---|---|
| P0 | M0 | reference harness 与 fixture manifest | ✅ 已完成 |
| P1 | M1 | 有限 NIfTI 运算与方向契约 | ✅ 已完成 |
| P2 | M2 | 变换、重采样与自动线性配准 | ✅ 已完成 |
| P3 | M3 | b0 参考图与脑掩膜 | ✅ 已完成 |
| P4 | M4 | nomoco raw-DWI 闭环 | ✅ 已完成 |
| P5 | M5 | SimNIBS legacy correction | ✅ 已完成 |
| P6 | M6 | 自动 T1 刚体/仿射配准与张量仿射重定向 | ✅ 已完成 |
| P7 | M7 | GRE fieldmap 固定路径 | ✅ 已完成 |
| P8 | M8 | TOPUP 子集 | ✅ 已完成 |
| P9 | M9 | EDDY `--repol` 子集 | ✅ 已完成 |
| P10 | M10 | 非线性 T1 配准与张量重定向 | ✅ 已完成 |
| P11 | M11 | QA、DAG 与发布验收 | 🚧 进行中 |

只有代码、测试、FSL A/B、文档和项目台账均达到对应里程碑要求后，阶段状态才更新为“已完成”。

## 1. 目标

本计划只复现 SimNIBS 4.6.0 `dwi2cond` 实际调用的 FSL 行为，使
`dwi2cond-xp` 能从原始单壳 DWI 生成
`m2m_<subject>/DTI_coregT1_tensor.nii.gz`，并继续进入现有的
`scalar/vn/dir/mc` 电导率和 SimNIBS FEM 流程。

本计划不是通用 FSL 替代项目，不实现与该目标无关的 fMRI、统计、分割、纤维追踪、
GUI 或完整命令行兼容层。

当前已经完成并保留的能力包括：

- 单壳选择、FSL 6.0.4 语义的 WLS DTI fitting 和 gradient nonlinearity；
- tensor、FA/MD/MO、L1--L3、V1--V3、S0、SSE、valid mask 和 QA；
- 应用调用者提供的 world affine、tensor 插值和 affine finite-strain 重定向；
- CHARM mask、`scalar/vn/dir/mc`、固定 montage FEM、lead field 和 E-field 输出。

因此新增工作只覆盖原始 DWI 预处理、自动 DWI/DTI 到 T1 配准、非线性 tensor
重定向和相应 QA。

## 2. Reference 范围

唯一兼容 reference 固定为 SimNIBS 4.6.0 随附的下列脚本：

- `simnibs/external/dwi2cond`；
- `dwi2cond.prepro.source.sh`；
- `dwi2cond.functions.source.sh`；
- `dwi2cond.t1reg.source.sh`；
- `dwi2cond.check.source.sh`。

FSL reference 版本在首个 fixture manifest 中固定。在版本冻结前，不把来自不同 FSL
release 的输出混入同一数值结论。

公开仓库不分发 FSL 源码、二进制、模型或受限数据。每个准备参考 FSL 源码的模块在
实现前单独登记源文件、版本和许可证结论；许可证未确认时，FSL 只作为本地 reference，
不能把相应源码或衍生文件加入公开包。

## 3. 支持模式

### 3.1 预处理模式

| 模式 | 输入 | 行为 | 必要输出 |
| --- | --- | --- | --- |
| `tensor` | 已拟合六分量 tensor | 跳过 raw-DWI correction 和 fitting | raw-space tensor及派生图 |
| `nomoco` | DWI、bvals、bvecs | b0 reference、brain mask、无 motion/eddy correction、WLS fitting | tensor、FA、SSE、mask |
| `legacy` | DWI、bvals、bvecs | SimNIBS 两轮 MCFLIRT/FLIRT 等价流程 | corrected DWI、变换、tensor |
| `eddy` | DWI、bvals、bvecs、PE/readout | 固定参数 `eddy --repol` 子集 | corrected DWI、rotated bvecs、outlier QA |

### 3.2 susceptibility 模式

| 模式 | 输入 | 行为 |
| --- | --- | --- |
| `none` | 无附加输入 | 不做 susceptibility correction |
| `fieldmap` | magnitude、phase、TE difference、dwell、PE方向 | SimNIBS GRE fieldmap/FUGUE 固定路径 |
| `topup` | reverse-PE b0、readout、PE方向 | `b02b0_nosubsamp.cnf` 对应的 TOPUP 固定路径 |

`fieldmap` 只与 `legacy` 组合；`topup` 只与 `eddy` 组合。非法组合必须显式报错。

### 3.3 T1 配准模式

| 模式 | 行为 |
| --- | --- |
| `rigid` | 自动估计 DTI-FA 到 T1 brain 的6 DOF变换 |
| `affine` | 自动估计12 DOF变换 |
| `nonlinear` | affine初始化后执行固定 FNIRT 子集，并做局部 tensor 重定向 |
| `external` | 保留当前调用者提供4x4 world transform的路径 |
| `assume-aligned` | 保留当前显式已对齐路径，不自动推断 |

## 4. 兼容性与科学改进的隔离

所有可能改变 SimNIBS 4.6 行为的改进必须与兼容路径分开：

- `compat46`：以 SimNIBS 4.6 脚本的参数、步骤和输出为 reference；
- `corrected`：允许修正已确认的问题，但必须使用独立开关、输出目录和 manifest；
- 任何后端不可用或失败时都不得静默切换模式；
- reference、optimized 和 corrected 输出不得共用 cache key。

已知需要显式决策的第一项是 legacy correction 的 bvec 行为：SimNIBS 脚本在该分支
复制原始 bvec，未使用类似 `eddy_rotated_bvecs` 的产物。`compat46` 保持 reference
行为；若实现逐 volume bvec rotation，只能进入 `corrected` 模式并单独验证。

## 5. 拟实现的 Python 结构

```text
src/dwi2cond_xp/preprocessing/
├── image_ops.py          # dwi2cond 所需有限 NIfTI 运算
├── orientation.py        # voxel/world/bvec/tensor 方向合同
├── transforms.py         # rigid/affine/warp 表示、求逆与组合
├── resample.py           # 3D/4D 标量、mask、vector、tensor 重采样
├── registration.py       # 多分辨率6/12 DOF估计
├── brain_mask.py         # b0 reference和DWI brain mask
├── legacy.py             # SimNIBS legacy 两轮校正流程
├── fieldmap.py           # GRE fieldmap/FUGUE 固定路径
├── topup.py              # reverse-PE field估计固定路径
├── eddy.py               # 单壳 eddy --repol 固定子集
├── nonlinear.py          # FNIRT固定子集和warp Jacobian
├── qa.py                 # motion、field、outlier、配准和tensor QA
└── pipeline.py           # 显式DAG、manifest和失败语义
```

初期保持纯 Python/NumPy/SciPy reference。只有 profile 证明热点后，才增加显式
optimized 后端；不得在 reference 尚未通过 fixture A/B 时提前替换算法。

## 6. 模块合同

### P0：Reference harness 和 fixture manifest

功能：

- 以独立进程运行本地 FSL/SimNIBS reference；
- 固定命令、环境、版本、线程、输入结构元数据和输出清单；
- 保存阶段计时、返回码、stdout/stderr摘要和结构化 QA；
- 对敏感真实数据只保存允许公开的聚合指标，不复制体数据。

完成条件：

- 至少有 deterministic synthetic fixture；
- 至少有一个私有真实单壳 DWI fixture；
- 每个 reference 输出登记 shape、dtype、affine/qform/sform、finite count、mask count；
- 数值数组的 reference 摘要与私有 artifact 分离；
- reference 未配置时显式 skip，不能把缺失 reference 记为通过。

当前阶段证据（2026-08-22）：

- deterministic synthetic single-shell DWI及b0 motion fixture已冻结；新增同一非解剖
  fixture工厂覆盖reverse-PE、fieldmap、已知world affine和FSL六分量tensor，连续两次
  生成的文件SHA-256逐项一致；
- synthetic `20x20x20x14`完整`dwi2cond --prepro --nomoco --keepstuff`已通过reference
  harness重跑，wall 3.283秒、child CPU 3.76秒、peak RSS 75,640,832 bytes，10个required
  artifact全部登记；
- 私有真实单壳`145x174x145x108`完整reference保持既有257.30秒实测，不重新运行；公开
  manifest仅登记匿名fixture ID、输入结构、计时和10个输出的shape/dtype/grid、finite/
  nonzero/range及mask count，不包含源路径、被试标识、体数据或artifact digest；
- reference `.nii.gz`摘要器已由逐x-slice随机读取改为单次顺序解压。同一533,407,034-byte
  HCP输出的结构摘要由重复解压改为36.20秒一次扫描（user 32.72秒、system 3.46秒、peak
  RSS 3,152,332 KiB）；摘要数值合同不变；
- 自动公开审计覆盖5份JSON，显式禁止MRI/NIfTI文件、绝对路径、直接identifier/credential
  key、raw voxel arrays和配置的私有词；当前审计通过；
- P0完成条件已全部满足。显式FSL全仓库回归为187 passed、2552/2552 statement、100%；
  未配置FSL时为183 passed、4 skipped、2552/2552 statement、100%。

### P1：有限 NIfTI 操作和方向合同

只实现脚本实际使用的行为：

- copy/format conversion、threshold、binary mask、multiply、subtract；
- time mean、median filter、edge、Gaussian smooth；
- split/merge/ROI、masked percentile、dimension读取；
- geometry复制和 tensor decomposition；
- canonical reorientation及其对 bvec/tensor basis 的影响。

完成条件：

- 不含插值的标量操作在相同 dtype 下逐数组一致；
- header、qform、sform和输出 dtype 有独立测试；
- axis permutation/flip fixture 同时验证 image、bvec和tensor方向；
- 没有把仅修改 header 误当作数据重排。

当前阶段证据（2026-08-22）：

- 已从SimNIBS 4.6脚本和本机FSL 6.0.4 `fslmaths.cc/newimage.cc/fslstats.cc`
  冻结实际语义，新增有限数组操作：`-thr/-uthr/-bin/-mas/-mul/-sub/-Tmean`、
  `fslmerge -t`、`fslroi`、`fslstats -P`和`fslval dim1..4`对应能力；
- `-thr/-uthr`保留等于边界的值，`-bin/-mas`使用严格正值；3D operand可按FSL
  合同作用于4D各timepoint；`fslroi`选择单个timepoint时返回3D；
- `fslstats -k mask -P p`使用mask `>=0.5`且排除值为0的voxel，排序后取
  `floor(N*p/100)`位置，不使用NumPy默认线性百分位插值；
- deterministic 3D/4D fixture已与本机FSL真实命令逐数组A/B，阈值、binary、mask、
  image/scalar运算、time mean、ROI和time merge均array-exact，masked percentile
  数值一致；显式FSL回归为9 passed、150/150 statement、100%；
- default 3x3x3 median在边界只使用图像内voxel并取FSL upper-rank median；edge保留
  FOV边界原值并使用物理voxel size归一化；Gaussian按4 sigma截断、逐轴卷积和每轴
  边界重归一化。真实FSL A/B中median/edge array-exact，Gaussian max absolute和
  relative L2分别为4.76837e-7和2.18973e-8；
- 已实现`fslmaths`空操作对应的float32 copy/format conversion，以及限定为dwi2cond
  实际同网格调用的`fslcpgeom` geometry copy；独立测试覆盖输出dtype、pixdim、qform、
  sform、code和destination数据保持；
- 已实现canonical storage reorientation：按FSL手性合同，正行列式排列为RAS，负行列式
  排列为LAS，只做axis permutation/flip而不插值；qform和sform使用同一个new-voxel到
  old-voxel变换更新，不能把header修改冒充数据重排；
- 已将存储重排与分量basis变换拆开：`fslreorient2std`兼容写出保持bvec/tensor分量通道
  不变，另提供显式voxel-basis signed-permutation API用于需要改变分量basis的调用者；
  synthetic fixture同时验证image、bvec和六分量tensor，张量特征值在变换前后不变；
- 已实现FSL六分量顺序的tensor decomposition并复用到现有DTI派生图写出。真实FSL A/B
  中，正/负手性reorientation的数据逐数组一致，qform/sform最大差均为0；geometry copy
  的数据、qform/sform和zooms一致；tensor scalar maps最大绝对差为1.78814e-7，三个
  特征向量的最大axial dot误差为5.96046e-8；
- P1完成条件已全部满足。显式FSL全仓库回归为182 passed、2527/2527 statement、100%；
  未配置FSL时为178 passed、4 skipped，覆盖率仍为2527/2527、100%。

### P2：变换、重采样和自动线性配准

功能：

- 6 DOF rigid和12 DOF affine参数化；
- voxel/world矩阵转换、求逆和组合；
- 默认weighted correlation ratio、reference weight和固定搜索策略；weighted MI作为
  源码对照与后续显式cost选项保留；
- 多分辨率图像金字塔；
- nearest、linear、spline和兼容所需的高质量最终插值；
- affine和displacement field一次性组合后重采样。

完成条件：

- 已知真值的 synthetic rigid/affine fixture 恢复到冻结阈值；
- transform round-trip和matrix composition通过；
- mask仅使用nearest插值；
- 相同输入重复运行的变换和输出确定性一致；
- FSL FLIRT A/B 同时报告矩阵差、角点位移、图像误差和相似度，不只比较一个矩阵元素。

当前阶段证据（2026-08-22）：

- 已新增FSL `compose_aff`一致的6/12 DOF参数化、world transform求逆/按应用顺序组合、
  neurological/radiological两种手性的FSL scaled-mm矩阵与NIfTI world矩阵双向转换；
- 已新增3D和channel-last 4D的分块重采样，支持nearest、linear、三阶spline和FSL
  width-7 Hanning-windowed sinc；mask只有
  独立nearest入口。affine逆映射与reference-grid上的moving-world位移在生成采样坐标后
  只插值一次，未发生affine一次、field再一次的重复模糊；
- 非半体素synthetic
  FLIRT `applyxfm` A/B中nearest逐体素bitwise一致，linear最大绝对差为
  `3.5762787e-6`、相对L2为`2.0227977e-7`；按源码补齐1201点Hanning-sinc查表和
  float32固定累加顺序后，sinc最大绝对差为`6.6757202e-6`、相对L2为
  `3.3851068e-7`；
- 此处“P2仍未完成”的历史判断已由下方封板证据取代。Generic Powell原型即使角点
  接近也不满足“同一算法”，已从主线删除；正式入口只使用FSL default schedule的
  weighted correlation ratio、refweight、搜索、扰动和阶段优化顺序。
- 删除generic optimizer后，显式FSL全仓库回归为218 passed，3223/3223 statements、
  100%。当前P2主线不存在用近似优化器冒充FLIRT的公开入口。
- 已按FSL 6.0.4 `Costfn::calc_fully_weighted_entropy`新增纯Python/Numba weighted-MI
  cost层：保持reference z/y/x遍历、`findrangex`边界修正、float32三线性插值、边缘
  smoothing、reference/test weight相乘、fuzzy binning、直方图累加和entropy归约顺序；
  未使用SciPy通用MI或自造代价函数。临时`/tmp` C++探针只调用本机FSL作为开发期oracle，
  不在仓库、包或运行依赖中。随机加权fixture四个矩阵的cost对FSL绝对差为
  `0`至`1.4305e-6`；focused测试为18 passed，连同transform/resampling为41 passed。
- 8-worker优化只并行彼此独立的候选矩阵，每个候选独占直方图并保持串行归约；输出按
  输入顺序重组。`80x80x64`、64 bins、24个固定候选的预热内存态A/B中，1 worker两次
  为`0.5886/0.5817 s`，8 workers为`0.1470/0.1452 s`，约`4.01x`，两路cost逐元素
  bitwise identical。该结果是candidate-cost微基准，不能外推为完整FLIRT或FA→T1速度。
- 已继续从FSL `MISCMATHS::optimise/optimise1d`移植默认coordinate-wise Brent，保留
  float cost、初始bracket、二次估计、golden fallback、minimum-distance修正、boundguess
  序列、average-tolerance停止条件和evaluation顺序；没有调用SciPy优化器。独立解析
  objective的FSL oracle与Python最终3参数在打印精度内一致，cost为
  `-0.0972308591008`，且均为23次一维迭代、52次cost evaluation。另已补齐FSL 7/8/9
  DOF的共享/部分scale语义、moving intensity COG及单stage tolerance缩放入口。
- 当前新增MI/optimizer/transform/resampling focused回归为60 passed；排除本机旧MNE与
  Numba 0.64导入冲突的无关`test_montage_plot.py`后，全仓库为238 passed、5 skipped。
  该环境收集错误不是本轮数值代码失败；尚未在正式test extra环境重跑100% coverage。
- 已按FSL `flirt.cc`自动`basescale`、`measurecost1`和默认schedule补齐8/4/2/1
  pyramid、weighted correlation ratio、64点粗搜索、11³细搜索、局部极小值、4 mm
  多起点及10组扰动、2 mm 7/9/12 DOF和末级qform竞争。默认`flirt`未指定`-cost`时使用
  weighted correlation ratio，不再误用weighted MI；MI实现继续作为独立源码oracle保留。
- pyramid的reference/moving及两侧weight在8/4/2/1四层共16个数组与优化前已通过FSL
  oracle的快照逐数组bitwise一致。相关ratio串行和有序体素并行在identity及非整数平移
  上bitwise一致；FSL打印的四层identity cost与Python分别为
  `0.0772655/0.077265680`、`0.120464/0.120463774`、
  `0.195185/0.195185244`、`0.194873/0.194872856`。
- 只做等价执行优化：体积采用与FSL x连续扫描一致的Fortran-order；8个独立候选同时
  调度；单候选先并行插值、再按原z/y/x次序串行归约；非零refweight工作集、每线程
  scratch、pyramid数组和moving COG均复用。未使用fastmath、低精度、降采样、减迭代、
  修改停止条件或浮点归约重排。
- 同一HCP FA、T1 brain和T1 refweight、矩阵估计边界的正式8-worker A/B：6 DOF
  Python算法/整进程为`10.398/11.66 s`，FSL 6.0.4为`61.97 s`，加速`5.31x`；12 DOF
  为`17.173/18.48 s`，FSL为`85.37 s`，加速`4.62x`。Python峰值RSS约
  `1,099,896 KiB`，FSL约`556,700 KiB`；用户已明确内存充足。计时均为独占运行，
  FSL约99% CPU，Python固定8 workers且约393--412% CPU。
- 相同HCP输出的6 DOF矩阵max-abs差`3.94333e-5`，移动FOV角点平均/最大位移
  `4.89364e-5/6.24869e-5 mm`，注册FA relative L2 `1.48225e-5`、Pearson
  `0.999999999868`；12 DOF对应为`0.00173711`、`0.00376072/0.00515443 mm`、
  `4.16271e-4`和`0.999999895265`。Python最终cost分别比FSL矩阵低
  `3.12924e-7/1.78814e-6`。
- 自动schedule测试覆盖6/12 DOF的已知2 mm synthetic平移、相同输入确定性、进度、
  参数验证和防御分支；四个新增FLIRT模块均达到100% statement coverage，没有omit或
  pragma。全仓库主批次272 passed、5 skipped，既有MNE/Numba导入冲突的montage测试
  以`NUMBA_DISABLE_JIT=1`单独追加12 passed；合并coverage为4572/4572 statements、
  100%。P2的变换、重采样、自动rigid/affine和真实FA→T1 A/B完成，P2/M2封板；将其
  接入raw-DWI公开CLI与完整T1 registration artifact合同仍属于P4/P6，不混作P2完成证据。

### P3：b0 reference 和 brain mask

功能：

- 按 compatibility 设置选择 b0；
- b0间6 DOF对齐；
- mean b0；
- DWI brain mask；
- 可选使用映射后的 CHARM/T1 mask作QA交叉检查。

完成条件：

- synthetic motion真值通过；
- mask非空、单一主连通域、无 NaN/Inf；
- 与reference比较 Dice、边界距离、体积比；
- mask差异对最终 tensor valid voxel和SSE的影响被量化。

当前阶段证据（2026-08-22）：

- 已实现显式 `b0_threshold=50` 的 b0 选择和分块直接均值，不再把4D DWI拆成逐
  volume临时NIfTI，也不复制完整4D DWI；
- HCP 100610单壳fixture `145x174x145x108` 上，未配准b0均值墙钟为4.20秒，
  user/system为2.76/1.44秒，峰值RSS为365976 KiB；输出为
  `145x174x145 float32`，finite且affine与官方`nodif`一致；
- 已按本机FSL 6.0.4 `mcflirt`源码冻结默认6 DOF行为：middle-volume reference、
  `normcorr`、8/4/4 mm三阶段、FSL Euler组合和逐轴bracket/golden搜索，并显式完成
  FSL scaled-mm矩阵到NIfTI world矩阵的转换；reference命令仍只用于A/B，不是运行依赖；
- HCP 100610的18个b0上，优化前单worker刚体配准加均值墙钟为26.59秒，峰值RSS为
  542224 KiB；独立FSL `mcflirt -dof 6`为12.80秒、565432 KiB；
- 固定8 workers后，第一阶段的邻近传播仍严格串行，第二/三阶段按volume并行；最终
  重采样并行计算，但均值仍按原volume顺序串行累加。再加上代价函数固定坐标和工作
  缓冲区复用、重复4 mm金字塔及COG缓存后，同一HCP输入三次墙钟为
  6.97/6.98/8.59秒，中位数6.98秒，相对FSL为1.83x；中位数运行峰值RSS为
  1553616 KiB，速度提升以更高内存和多核占用为代价；
- 优化前后18个scaled-mm矩阵、每volume evaluation count和最终float32 NIfTI均
  bitwise identical；没有减少迭代、修改停止条件、改变插值或重排最终均值归约；
- 对完整`145x174x145x288`压缩DWI新增单次顺序gzip读取，避免NiBabel逐b0随机访问
  导致重复解压；从原始`.nii.gz`直接完成b0选择、刚体配准和均值为19.38秒，输出矩阵
  和图像与预提取未压缩fixture bitwise identical。该19.38秒包含压缩输入读取，不能
  与只接收已提取b0的FSL 12.80秒直接计算加速比；
- 自有变换与FSL变换在图像八角点的最大/平均位移差为0.335776/0.153796 mm；
  对齐后均值在官方brain mask内相对L2为0.00627127，即0.627%。该图像尺度误差
  作为当前工程门禁可接受，但不表述为MCFLIRT优化轨迹逐点一致；
- 固定使用FSL矩阵时，自有重采样与FSL均值的相对L2为1.45701e-5，说明当前主要
  差异来自优化器最终落点，而不是矩阵坐标转换或最终重采样；
- 已按BET 2.1 `-f 0.2 -m`实现robust 2/98 limits、强度加权COG/半径、5阶细分二十
  面体（2562 vertices/5120 faces）、1000次同步表面演化、自交评分和闭合网格栅格化；
  reference为NumPy/SciPy，optimized为`fastmath=False`的Numba并行实现，均不调用FSL；
- 用FSL最终mesh输入自有栅格化器时，synthetic mask逐体素bitwise一致。自有完整演化
  对synthetic reference Dice为0.994级；真实HCP nodif对FSL mask为883146/883571
  voxel、Dice 0.993691689、体积比0.999518997、单一连通域，双向边界平均/P95/最大
  距离为0.315235/1.25/4.145781 mm；
- HCP初始化的`t2/t98/threshold/COG/radius/median`与FSL verbose逐项一致。optimized
  连续5次最终vertex SHA-256前16位均为`d51121b46ec7fd04`，且与NumPy reference的
  HCP mask和边界指标一致；
- 固定8 workers且JIT预热后，HCP `nodif`内存态BET约0.78--0.92秒，读取NIfTI到写出
  `.nii.gz`约0.96秒；同输入FSL BET进程为1.59秒，因此常驻Python流水线的同阶段端到端
  约1.66x。独立冷启动Python进程约2.12秒、首次Numba编译约4.8秒，均不混入预热加速比；
- 用同一未压缩HCP `DWIforfit`、同一Python WLS拟合器和固定8 workers分别输入FSL与
  自有mask，墙钟为24.806/24.342秒。FSL/self mask得到883293/882859个valid voxel，
  自有路径仅少434个（`-0.0491343%`），valid-mask Dice为`0.993719680`；
- 两组共同的877530个valid voxel中，六分量tensor和SSE均逐数组bitwise一致、最大绝对
  差为0。全valid区域SSE总和相对变化为`-0.0815186%`；差异只来自mask不重合边界带，
  因而P3的下游影响量化完成。不得把BET或6.98秒MCFLIRT局部结果外推为完整`nomoco`
  的257.30秒加速比。

### P4：`nomoco` raw-DWI 闭环

功能：

- 输入标准化、b0/mask、非负截断；
- 调用现有 WLS fitting；
- 生成raw-space tensor、FA、SSE和QA。

完成条件：

- synthetic raw DWI从CLI完整运行到tensor；
- 相同预处理输入下与当前 `fit-dti` 输出一致；
- 对本地FSL reference报告tensor、FA、SSE和valid mask A/B；
- 无 correction 时不得生成伪造的motion/field结果。

状态：已于2026-08-22完成。

- 新增纯Python `preprocess-nomoco` CLI，按SimNIBS 4.6顺序执行float32/storage
  标准化、零下截断、b0刚体配准均值、BET `-f 0.2 -m`、既有WLS fitting和QA；默认
  固定8 workers，不对DWI volume应用motion、eddy-current或susceptibility correction；
- 已经是未压缩float32、FSL标准方向、有限且非负的`.nii`输入按z-block验证后直接mmap，
  不再重复写出1.58 GB `DWIforfit`；`.nii.gz`及其他不满足条件的输入仍将标准化、重排和
  `-thr 0`融合为一次解码，并只物化一个未压缩`DWIforfit.nii`供全部worker共享；
- 每个拟合worker的BLAS固定单线程，避免8进程内部过度订阅。限线程和mmap快路径前后
  HCP nodif、mask、tensor、FA、SSE和valid mask逐数组bitwise一致；QA显式记录
  `validated_input_mmap`或`single_decode_materialization`，不静默混淆计时边界；
- 公开`20x20x20x14` synthetic真实CLI整进程1.72秒；同机SimNIBS 4.6/FSL 6.0.4
  reference为3.267秒。mask Dice为0.993194707，tensor/FA/SSE relative L2分别为
  `3.7433e-15`、`5.8697e-8`、`5.2993e-8`，均通过预先冻结门槛；
- 私有HCP b0+b1000 `145x174x145x108`、官方归一化b0和相同bvec输入的8-worker完整
  CLI：规范未压缩`.nii`快路径墙钟18.82秒、user/system 48.70/9.00秒、峰值RSS
  1770256 KiB；同数据`.nii.gz`单次解码路径墙钟52.92秒、user/system 71.71/18.67秒、
  峰值RSS 4791436 KiB。官方完整`nomoco`为257.30秒、CPU 99%、峰值RSS
  3167907840 bytes，因此同机墙钟分别约13.67x和4.86x；
- HCP自有/FSL brain mask为884224/883571 voxel，Dice 0.993205660。共同877892个
  mask voxel内tensor、FA、SSE relative L2分别为`6.4647e-6`、`6.9847e-6`、
  `3.5067e-5`；全域差异由两种BET mask的不重合边界主导，不能用全域零填充值衡量拟合器；
- 相同自有预处理输入再次调用现有`fit_dti_nifti`时tensor、FA和SSE逐数组bitwise一致；
  `.nii`与`.nii.gz`路径的nodif、mask、brain、tensor、FA、SSE和valid mask也逐数组
  bitwise一致。输出目录不生成任何motion、eddy或field伪artifact。最终主批次284
  passed，montage隔离批次12 passed，合并为4717/4717 statements、100%。

### P5：SimNIBS legacy correction

固定步骤：

1. 计算 `b>0` mean；
2. 第一轮6 DOF逐volume配准；
3. 更新mean；
4. 第二轮12 DOF并保留每volume矩阵；
5. corrected mean到nodif的配准；
6. 矩阵组合；
7. b0 volume使用直接到nodif的6 DOF矩阵；
8. 可选组合fieldmap warp；
9. 每个volume只做最终一次重采样；
10. 合并4D DWI并进入现有 WLS fitting。

完成条件：

- 每个volume的变换、最终DWI、mean、FA、SSE和tensor均有reference A/B；
- 记录插值次数并证明正式输出只经过一次最终重采样；
- `compat46` 与 `corrected` bvec合同有独立测试；
- 失败不得回退到 `nomoco`。

状态：已于2026-08-22完成。

- 新增纯Python `preprocess-legacy` CLI和`preprocessing.legacy`，按上游脚本顺序执行
  `b>0` mean、第一轮6 DOF逐volume MCFLIRT、更新mean、第二轮12 DOF、corrected
  mean到nodif的`flirt -nosearch -cost mutualinfo`、矩阵左乘、b0直接到nodif的6 DOF
  覆盖、一次正式sinc重采样以及既有WLS fitting；默认8 workers，任何阶段失败均直接报错；
- 第一/二轮仅为更新reference mean的临时重采样不会进入正式DWI；QA固定记录每个正式
  volume只有一次最终插值。可选fieldmap位移与affine pull坐标先组合再采样，不增加第二次
  正式插值；fieldmap估计本身仍属于P7；
- `compat46`逐字节复制原bvec，复现SimNIBS 4.6 legacy行为；`corrected`以独立开关使用
  最终affine有限应变旋转并重新单位化，两个模式不共享隐式语义；
- width-7 Hanning-sinc新增Numba输出点并行核；每个点内部仍按z/y/x和7×7×7固定顺序
  累加，随机坐标对旧向量化reference核逐值bitwise一致。synthetic正式最终重采样阶段
  由约8.39秒降至0.054秒；
- normcorr热核融合坐标、边界权重和三线性采样，显式物化FSL float32乘法舍入点，并
  复现NumPy/FSL先x、再y、最后z的`PW_BLOCKSIZE=128` pairwise归约；未启用fast-math。
  Linux仅把没有跨volume依赖的4 mm阶段交给8个进程，首个8 mm邻近传播仍严格串行；
  Windows/macOS使用同算法线程路径；
- 公开`20x20x20x14` synthetic完整CLI三次独立进程墙钟为2.75/2.71/2.87秒，
  中位数2.75秒、峰值RSS中位数226824 KiB；同机SimNIBS/FSL reference harness为
  6.7657秒、峰值145436672 bytes，故当前同边界墙钟加速2.46x。三次运行的14个矩阵、
  corrected DWI、mean、mask、tensor、FA、SSE和valid mask均与优化前Python输出bitwise一致；
- 14个最终矩阵对FSL的max-abs最大值为0.0699333，移动FOV八角点平均/最大位移差为
  0.0652414/0.0941606 mm；corrected DWI和`b>0` mean relative L2为0.00910157和
  0.0130192，mask Dice为0.993194707。固定使用FSL最终矩阵时，自有sinc重采样对FSL
  corrected DWI的max-abs为0.00428009、relative L2为7.42962e-7，说明主要差异来自
  MCFLIRT优化落点，而不是矩阵坐标或最终插值；
- sharp/noiseless synthetic在BET边界对tensor/SSE极敏感；共同mask内tensor/FA/SSE
  relative L2分别为0.772510、0.0373037和1.06086，不能据此宣称tensor逐值等价。
  三次mask erosion后的内部836 voxel中tensor和FA relative L2降至0.00287432和
  0.00248977。该边界事实已保留，未用全域零填充或只报内部值掩盖；
- 主测试批次297 passed、4 skipped，MNE隔离批次12 passed；Numba热核另以
  `NUMBA_DISABLE_JIT=1`执行28个逐行测试，合并5199/5199 statements、100%。ruff、
  `git diff --check`和6份公开reference manifest审计均通过。

### P6：自动 T1 rigid/affine 和 affine tensor 重定向

状态：已于2026-08-22完成。

功能：

- 从 CHARM labeling生成brain mask和T1 brain；
- DTI-FA到T1的6/12 DOF估计；
- 调用现有tensor affine resampling/reorientation；
- 生成registered FA、V1、SSE和brain-rim QA。

完成条件：

- synthetic transform真值、真实FA/T1 reference A/B通过；
- 最终tensor严格匹配T1 grid、qform/sform和CHARM mask；
- tensor有限、对称，valid voxel中的特征值/方向统计可解释；
- affine失败不得接受 `assume-aligned` 作为隐式回退。

当前阶段证据（2026-08-22）：

- 新增纯Python `register-t1` CLI，按SimNIBS 4.6脚本生成CHARM `1..499` brain mask、
  T1 brain、brain-rim QA和Gaussian reference weight；默认12 DOF主配准，同时保留独立
  6 DOF QA配准，rigid模式只执行6 DOF。失败直接报错，不回退`assume-aligned`；
- affine tensor路径按FSL `vecreg`使用六分量
  `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`、前三分量构造source mask、trilinear插值、FLIRT
  scaled-mm矩阵的polar rotation和reference brain mask；输出tensor、valid mask、FA、V1、
  6 DOF FA/SSE QA及结构化阶段计时；
- `fslmaths -tensor_decomp`源码复核补齐`L1 > 0`才写出派生量的条件。真实全头A/B中
  FA非零support由原先多3 voxel修正为双方均3,947,315 voxel；FA relative L2为
  `3.23951e-4`，最大绝对差`0.00575823`；V1共同support 3,947,315 voxel，轴向夹角
  mean/P99/max为`0.0156600/0.155766/84.2445`度，最大值来自极少数低各向异性或边界点；
- 同一真实输入的12 DOF矩阵max-abs差`0.00661515`，移动FOV角点平均/最大位移差
  `0.00385192/0.00771821 mm`；registered tensor relative L2为`2.37187e-4`，T1 brain、
  brain rim和全部输出网格一致；6 DOF FA QA与SSE QA relative L2分别为
  `1.23549e-4`和`2.22144e-4`；
- 完整同边界端到端计时：Python 8 workers、底层runtime不额外限线程，wall
  `53.23 s`、CPU `2351%`、peak RSS `3,103,744 KiB`；FSL/SimNIBS reference wall
  `233.18 s`，同机墙钟约`4.38x`。Python初版为`100.90 s`，本轮严格等价调度和数据
  生命周期优化约`1.90x`；未把局部FLIRT微基准外推为完整流程；
- profile显示最终运行约由T1准备5秒、并行双FLIRT 17--20秒、tensor/派生量关键路径
  约29秒构成。保留的优化包括6/12 DOF并行、六tensor分量与source-mask独立任务并行、
  tensor/valid gzip与同一只读内存数组上的FA/V1分解重叠、FA/V1和FA/SSE成对压缩写出。
  T1四文件后台gzip因使FLIRT变慢而撤回；没有修改schedule、cost、迭代、停止条件或浮点
  归约顺序；
- 公开`24x23x22`非解剖fixture的6/12 DOF矩阵、角点、tensor、FA和V1门禁已写入第7份
  reference manifest。优化前后矩阵、registered tensor、valid mask和QA数组bitwise一致；
  只有按FSL源码修正的3个非正L1体素发生预期FA/V1变化；最终主批次306 passed、
  5 skipped，Numba热核禁用JIT逐行批次22 passed，MNE隔离批次12 passed，合并
  5464/5464 statements、100%。ruff、`git diff --check`和7份manifest审计均通过。

### P7：GRE fieldmap 固定路径

功能：

- magnitude第一volume和brain mask；
- Siemens phase difference到rad/s，或接受已为rad/s的fieldmap；
- median filter和mask中位数offset移除；
- b0到distorted magnitude的6 DOF配准；
- fieldmap映射到DWI空间；
- 根据dwell和PE方向生成voxel-shift/displacement field；
- mask、b0和每volume的warp组合。

完成条件：

- rad/s、秒、毫米和voxel单位在manifest中明确；
- 正负PE synthetic phantom恢复已知位移；
- FSL FUGUE路径比较field、shift、corrected b0和最终tensor；
- 禁止通过视觉结果猜测PE符号。

完成记录（2026-08-22）：

- 冻结并实现SimNIBS 4.6 `APPLY_FMCORR` 的已为rad/s输入分支：magnitude首volume、
  BET默认`f=0.5`、可选3x3x3 median、mask中位数offset、FUGUE默认hole fill/
  rigid extension、`--nokspace` forward warp、6 DOF mutual-information FLIRT、
  applyxfm边界、voxel shift、world-mm pull displacement、b0/mask unwarp；legacy正式
  volume继续只做一次sinc插值；
- CLI为`prepare-fieldmap`，默认8 workers，输出manifest显式记录rad/s、毫秒/秒、voxel、
  NIfTI world-mm和PE方向。`x/x-/y/y-/z/z-`均由解析函数确定轴与符号，负号只在
  displacement方向生效，不通过图像外观猜符号；
- 公开`16x14x12`非解剖fixture同时验证`y`和`y-`。两方向mapped mask与corrected mask
  均逐体素一致；`y-`的distorted magnitude、mapped field、voxel shift和corrected b0
  relative L2依次为`4.56e-8`、`6.66e-7`、`8.93e-7`、`7.36e-8`；`y`依次为
  `4.98e-8`、`2.03e-7`、`3.06e-7`、`6.64e-8`；
- fixed-FSL-shift组合门禁中，每个legacy volume仍只有一次正式插值；corrected DWI和
  同一fitter得到的tensor relative L2分别为`0.0167622`和`0.0155070`。该尖锐无噪声
  小fixture对边界插值敏感，因此不把tensor误差外推到解剖数据；
- profile确认完整MI FLIRT是热点。先把1331个MI候选改为Numba候选间并行，再将64个
  coarse、筛选后的refine和多起点Brent轨迹按轮锁步批评估，并批量构造每轮仿射矩阵；
  每条轨迹的候选/分支顺序、histogram的FSL z/y/x归约、搜索范围、迭代、停止条件和
  11,349次evaluation均不变；后续又整批执行矩阵校验/求逆、11³网格插值和仿射构造。
  进程wall由`6.88/6.55/6.81 s`进一步降至`1.49/1.51/1.51 s`，pipeline内部为
  `0.589/0.632/0.623 s`；矩阵及7个关键NIfTI数组与批处理前
  Python路径bitwise一致，对FSL矩阵max-abs仍为`1.266861e-5`。同一极小fixture的FSL
  harness约`0.46 s`，其C++全优化器和更低启动开销仍占优；不宣称P7快于FSL，也不
  外推真实GRE性能。相同进程内全部内核常驻后的后3次pipeline为
  `0.358/0.413/0.400 s`，仅用于分离启动成本，不与FSL单进程harness计算加速比；
- 本阶段选择计划允许的“接受已为rad/s fieldmap”固定分支。原始Siemens wrapped phase
  仍需PRELUDE解缠；在完成PRELUDE同算法复现前显式不接受该输入，绝不以`numpy.unwrap`
  等近似替代。

### P8：TOPUP 固定子集

范围只覆盖 `b02b0_nosubsamp.cnf` 对应行为：

- 正反PE b0联合forward model；
- PE单轴 susceptibility field；
- B-spline basis和固定正则化；
- 图像间motion；
- 固定多分辨率优化；
- field coefficients、movement parameters和corrected b0。

完成条件：

- synthetic reverse-PE phantom恢复field方向和幅度；
- 输出field、movement、corrected b0和mask均与reference分阶段比较；
- corrected pair的一致性必须优于uncorrected pair；
- TOPUP失败不得继续启动EDDY。

完成证据（2026-08-23）：

- 已逐项冻结SimNIBS随附`b02b0_nosubsamp.cnf`的九级schedule：全程不下采样，前五级
  LM并估计相对运动，后四级SCG且固定运动；三次B-spline、double Hessian、逐图强度
  缩放、SSD加权bending-energy正则和spline图像插值均保持原配置；
- 已实现FSL 6.0.4 `basisfield`同定义的knot spacing、向后兼容coefficient shape、
  三次B-spline及一/二阶导数、FSL系数/体素循环顺序的dense field展开。使用FSL写出的
  `11x10x9` float32系数重建`16x14x12` Hz field，relative L2为`2.98345e-8`、
  max-abs为`1.46583e-6 Hz`；差异边界来自`--fout`使用优化器内部double系数，而系数
  NIfTI写为float32；
- 新增本地reference runner并保留field、coefficients、movement、corrected pair和BET
  mask。FSL同fixture的TOPUP本体wall约`4.63 s`、完整runner约`5.55 s`，peak RSS
  `14552 KiB`；corrected pair relative L2由`0.677007`降至`0.00892792`；
- FSL 6.0.4 `TopupScan`源码明确拒绝PE向量第三分量非零，因此兼容运行路径显式拒绝
  `z/-z`，不把SimNIBS shell层虽能写出的无效组合静默替换成另一算法；
- 已完成FSL `regrid=1`源网格、周期三次图像spline及导数、field→voxel displacement、
  spline Jacobian、图像间刚体运动导数、FSL顺序pair SSD、bending energy、对角PCG、
  LM和Moller SCG。首层零场/8 mm平滑SSD为`6981.48122`，与FSL日志`6981.48`对齐；
  九级schedule、前五级运动估计、后四级固定运动和每级迭代上限均未改变；
- 通用SciPy稀疏Hessian三次乘法已替换为保持四个Gauss--Newton项及z/y/x累加顺序的
  单个Numba装配内核。首层field Hessian对FSL debug relative L2为`9.33e-8`；同一进程
  完整九级热跑约`2.42 s`，只用于说明内核常驻收益；
- Numba磁盘cache已由一次显式预热建立后，三次独立新进程、8个Numba worker、同一
  `16x14x12x2`公开fixture和相同四类TOPUP
  主要输出下，初版Python完整CLI wall为`4.14/4.09/4.18 s`（中位数`4.14 s`），算法
  内部为`3.259/3.218/3.298 s`；FSL 6.0.4 TOPUP为`4.78/4.75/4.66 s`（中位数
  `4.75 s`）。随后profile驱动的等价优化缓存重复spline basis/坐标网格，以保持原循环
  顺序的Numba内核装配稀疏bending Hessian和scan-major SSD，并在每个独立coefficient
  内保持FSL z/y/x累加顺序并行计算spline transpose；没有改变九级schedule、迭代、
  停止条件或浮点归约顺序。首轮优化后三次独立进程为`3.45/3.31/3.18 s`（中位数
  `3.31 s`）。第二轮进一步跳过支撑不相交的Hessian系数对/voxel、缓存重复刚体矩阵、
  复用相同正则项乘积，并把仅64--185维的LM系统改为稠密存储；PCG每一行仍按列递增
  累加。第三轮把CLI route和preprocessing public export改为按需加载，并缓存固定运动、
  rejected field state重复使用的pull matrix及同一刚体逆矩阵；没有改变TOPUP目标函数或
  浮点运算。最终三次独立进程为`2.75/2.70/2.71 s`（中位数`2.71 s`），算法内部为
  `2.079/2.040/2.051 s`（中位数`2.051 s`）。相对初版完整CLI约快`1.53x`，相对FSL
  完整边界约快`1.75x`；peak RSS约`199 MiB`，FSL约`14 MiB`。首次建立JIT cache的
  一次性编译成本仍与稳态结果分开报告。融合PCG循环和缩短spline transpose支撑区间
  两个候选虽保持artifact bitwise一致但没有稳定收益，均已撤回；
- 第四轮只优化启动边界：包根目录的tensor/gradient API、版本元数据和关闭状态的
  progress bar均改为按需加载，不触碰TOPUP数值函数。同机后续负载样本中，修改前
  `3.73/3.61/3.84 s`（中位数`3.73 s`），修改后`3.47/3.49/3.52 s`（中位数
  `3.49 s`），约再快`6.4%`；同期FSL TOPUP为`4.90/4.72/4.72 s`（中位数
  `4.72 s`），Python仍快约`1.35x`。这组负载较高的配对数据不覆盖上方安静系统的
  主基准。五类artifact再次逐文件bitwise一致；
- 进一步候选审计表明，field Hessian上下三角存在最大`6.82e-13`的真实末位非对称，
  因此拒绝只算半边再镜像；有序稠密regularization matvec虽bitwise一致，但交错十轮
  常驻内核中位数为`0.806 s`，慢于稀疏reference的`0.751 s`，同样撤回；
- 第五轮将彼此独立且支撑可能重叠的field Hessian元素展开为细粒度workset交给8个
  worker；每个元素内部仍保持FSL z/y/x累加顺序。同时按几何缓存有序系数对、spline
  支撑和bending regularizer重复使用的一维核重叠。优化后四次独立稳态CLI为
  `2.38/2.42/2.46/2.40 s`（中位数`2.41 s`），算法内部为
  `1.735/1.769/1.819/1.763 s`（中位数`1.766 s`）；同期FSL 6.0.4 TOPUP为
  `4.65/4.66/4.64 s`（中位数`4.65 s`），Python约快`1.93x`。五类artifact与
  FSL float三角函数语义修正后的优化前基线逐文件bitwise一致；16 workers在交错样本中
  慢于8 workers，因此保持8-worker默认值。预计算Hessian坐标的候选没有稳定收益且增加
  约4 MiB内存，已撤回；
- 第六轮继续保持每个Hessian元素内部的FSL z/y/x归约不变，只按实际相交支撑体素数对
  独立元素做稳定降序并条带分配到8个等长worker chunk。九级网格原有连续切块的最大
  工作量不均衡约为`17%--47%`；重排不改变任一矩阵元素的输入、算术或写入位置。同时把
  field gradient的两个独立spline transpose累加器合并到一次体素遍历中，两个累加器各自
  的系数和z/y/x顺序保持不变。五类artifact与第五轮基线逐文件bitwise一致，1 worker和
  8 workers输出也逐文件bitwise一致。最新五次独立CLI为
  `2.26/2.21/2.33/2.44/2.51 s`（中位数`2.33 s`），同期FSL 6.0.4 TOPUP为
  `4.74/4.65/4.65 s`（中位数`4.65 s`），完整边界约快`2.00x`；进程内预热一次后的
  六次算法样本为`0.780/0.748/0.867/0.874/0.918/0.741 s`（中位数`0.823 s`），该
  常驻数字只用于长流水线，不与新进程FSL比较。合并dense field/derivative展开的候选虽
  bitwise一致但实测更慢，已撤回；
- 第七轮将同一objective内两幅独立scan的周期三次插值合并为一次Numba并行派发，每幅
  scan内部仍保持逐voxel和4x4x4支撑的原累加顺序；正则Hessian保留原CSC公开合同，另缓存
  CSR视图用于重复matvec，行内列顺序与原CSC逐列访问一致；coefficient展开改为voxel间
  并行，而每个voxel内部仍按FSL coefficient z/y/x顺序累加。五类artifact与第六轮逐文件
  bitwise一致，最终1/8 workers输出也逐文件bitwise一致。最终七次独立CLI为
  `2.17/2.25/2.26/2.31/2.07/2.09/2.13 s`（中位数
  `2.17 s`），算法内部中位数`1.541 s`，相对FSL `4.65 s`约快`2.14x`；一次预热后的
  七次常驻算法中位数为`0.630 s`，较第六轮`0.823 s`减少约`23.5%`。预计算完整3D
  spline basis的Hessian候选虽bitwise一致，但完整CLI中位数`2.17 s`没有收益且peak RSS
  升至约`312 MiB`，已撤回。focused回归为`26 passed`；排除当前环境两个MNE绘图文件的
  全仓回归为`367 passed, 4 skipped`，ruff、compileall和diff门禁通过；
- 第八轮把64--185维dense PCG的完整迭代收进单个Numba调用，逐行矩阵乘法仍按列升序
  累加，并保留显式`reference`后端；同时把五个彼此独立的movement-interaction
  transpose列放进一次coefficient/voxel遍历，每列内部仍按FSL z/y/x顺序累加。九次
  独立新进程CLI为`2.14/2.06/2.12/2.18/2.34/2.04/2.03/2.04/2.26 s`（中位数
  `2.12 s`），算法内部中位数`1.497 s`；同期FSL 6.0.4为`4.77/4.59/4.61 s`
  （中位数`4.61 s`），Python约快`2.17x`。五类数值artifact与第七轮逐文件bitwise
  一致；focused为`28 passed`，排除两个MNE绘图文件的全仓回归为
  `370 passed, 5 skipped`，ruff、compileall和diff门禁通过。批量regrid、轴旋转缓存、
  跨级objective复用、PE特化Hessian和movement direct/derivative配对候选均因完整边界
  无稳定收益而撤回；
- 两轮优化前后field、coefficients、movement、corrected pair和joint mask的最终文件
  SHA-256逐项一致；旧CSC/SciPy与新稠密有序PCG路径的五类数组也逐数组bitwise一致；
  七套bending Hessian的`data/indices/indptr`也逐数组bitwise一致。完整显式FSL回归更新为
  `387 passed, 4 skipped`，ruff、py_compile和diff门禁通过；
- 恢复FSL 6.0.4 `construct_rotmat_euler`的float三角函数语义后，最终Hz field对FSL
  relative L2/max-abs/mean-abs刷新为
  `0.00429657 / 0.451101 Hz / 0.0501639 Hz`；joint mask内corrected pair对FSL
  relative L2为`0.00161346`、max-abs为`1.41406`，pair一致性relative L2为
  `0.00922993`（FSL `0.00892792`），远优于未校正输入；movement最大参数绝对差为
  `0.00241634`（旋转单位rad、平移单位mm）。field轨迹误差仍低于用户已接受的千分之六
  量级，且movement matrix本身对FSL C++探针max-abs为`2.78e-17`；剩余差异来自LM对
  极小运动导数差的轨迹放大，不宣称bitwise exact；
- 已新增`prepare-topup`纯Python CLI和NIfTI闭环，固定写出Hz field、float32 spline
  coefficients、6参数movement、corrected pair、joint mask及QA JSON；z向PE显式拒绝，
  失败直接抛错。第五轮优化后的主批次为`383 passed, 4 skipped`，focused批次为
  `39 passed`；当前可视化环境缺少可导入的MNE，montage隔离文件为`1 skipped`，
  ruff、compileall和diff门禁通过。P8完成条件已全部满足。

### P9：EDDY `--repol` 固定子集

范围：单壳、volume-to-volume motion、SimNIBS生成的acqp/index、可选TOPUP、
slice outlier replacement和rotated bvecs。不实现未被SimNIBS调用的完整EDDY CLI。

分阶段实现：

1. 无TOPUP、无outlier replacement；
2. motion/eddy参数和rotated bvecs；
3. 组合TOPUP field；
4. slice outlier detection；
5. prediction-based replacement；
6. 完整固定参数CLI闭环。

完成条件：

- 每个volume motion、eddy参数和bvec旋转有结构化输出；
- bvec保持单位长度，b0向量合同明确；
- synthetic injected motion/outlier有已知真值；
- 同时比较corrected DWI、outlier map、rotated bvecs、FA、SSE和tensor；
- `--repol` 不能退化为只删除异常slice或简单邻slice平均。

完成证据（2026-08-23）：

- 已按FSL 6.0.4源码完成固定五轮volume-to-volume runner、quadratic EC、spherical GP、
  scan/model双向重采样、Jacobian、accept/reject、b0/DWI location reference、rotated bvec、
  shell PE/rigid alignment及prediction-based `--repol`；可选TOPUP场包含第十个field-offset
  参数及每轮field-offset/movement separation，不实现SimNIBS未调用的高级EDDY选项；
- 同状态FSL debug oracle中，prediction/derivative/Gauss--Newton update最大差异约
  `1.14e-13`；4x4 Armadillo tiny inverse、NEWMAT小矩阵累加和标量`atan2f`路径均按源码
  复现。固定seed用于正式A/B，避免把FSL默认时间随机初始化误当数值差异；
- 新增`prepare-eddy`纯Python CLI和`run_eddy_nifti`，支持`.nii`/`.nii.gz`、8 workers、
  可选susceptibility field、`--repol`及结构化corrected DWI、outlier-free data、16参数、
  rotated bvec、outlier map、shell alignment、迭代history和QA JSON；TOPUP field完整CLI
  smoke输出`26x26x18x26` corrected DWI及`26x16`参数表；
- 同一公开`26x26x18x26`注入fixture、FSL/Python均固定seed 1时，四个outlier
  `(5,8)/(11,9)/(18,7)/(23,10)`全部且仅有它们被检测。brain-mask内corrected DWI
  relative L2为`0.0035352`，outlier-free data为`0.0031508`，rotated bvec为
  `0.0014983`；下游WLS tensor/FA/SSE分别为`0.0060714/0.0035526/0.0539494`，其中
  SSE绝对误差mean/P99/max为`0.001279/0.01844/0.07959`；
- 严格8线程、同机、同输入、完整CLI端到端：Python `7.76 s`、FSL 6.0.4
  `9.77 s`，Python约快`1.26x`；Python算法内部`6.90 s`、peak RSS约`232 MiB`，FSL
  peak RSS约`36 MiB`。重复Python运行的参数表和outlier map逐文件bitwise一致；
- wheel已在基于现有SimNIBS 4.6.0只读依赖的隔离overlay中无依赖联网安装，`pip check`
  无缺失；公开fixture完整CLI冷启动（含首次Numba编译）`37.78 s`，缓存后独立进程
  `9.34 s`，输出形状、16列参数、四个outlier及重复运行bitwise门禁均通过。参考环境
  未被修改；Conda完整clone仅因本机包缓存清单缺失若干`.pyc`而未作为验收路径；
- focused EDDY/CLI测试为`48 passed`，ruff通过；P9/M9完成条件全部满足，阶段complete。

### P10：非线性 T1 配准和 tensor 重定向

范围只覆盖 SimNIBS 的固定调用：affine初始化、`subsamp=8,4,2,2`、输出warp，随后
执行nonlinear tensor resampling和局部重定向。

功能：

- 固定FNIRT多分辨率deformation和正则化；
- warp Jacobian；
- finite-strain/PPD兼容行为；
- 低FA、重复特征值和近奇异Jacobian处理；
- registered tensor、FA、V1和QA。

完成条件：

- synthetic deformation的Jacobian和tensor方向真值通过；
- deformation、Jacobian determinant、registered FA/V1/tensor均有reference A/B；
- fold、近奇异区域和无效tensor必须显式记录；
- nonlinear失败不得回退到affine输出并沿用nonlinear名称。

完成证据（2026-08-23）：

- 已按SimNIBS 4.6固定调用和FSL 6.0.4源码实现四级FNIRT：affine初始化、
  `subsamp=8,4,2,2`、`miter=5,5,5,5`、输入/参考平滑、bending energy、
  spatially varying intensity mapping、稀疏对角PCG、LM接受/拒绝、cubic B-spline
  coefficient扩展及`.nii`/`.nii.gz` warp输出；未减少迭代、修改停止条件或启用
  fast-math；
- 已实现解析warp Jacobian及FSL `vecreg` preservation-of-principal-direction tensor
  重采样/重定向，输出registered tensor、FA、V1、valid mask、Jacobian和结构化QA；
  fold、近奇异Jacobian、低FA、重复特征值和无效tensor均显式计数，失败不回退；
- synthetic affine/nonlinear deformation、解析Jacobian、PPD方向、fold和失败语义的
  focused回归通过。同一公开`24x23x22` fixture相对FSL：warped FA relative L2
  `0.00309871`；形变向量误差mean/P99/max为`0.204/0.603/1.147 mm`，Jacobian
  relative L2 `0.0253746`；tensor公共support relative L2 `0.0445701`；FA support
  Dice `0.996765`，公共support内FA relative L2 `6.90e-8`；V1按轴向符号不变角度
  mean/P99/max为`2.533/8.131/11.352°`。非线性病态PCG的末位归约会放大到optimizer
  端点，因此不宣称coefficient或field bitwise一致；
- 同机完整边界、FSL/Python均使用固定fixture：FSL FNIRT+VECREG+tensor decomposition
  `9.203 s`，Python完整CLI（额外写valid mask、两个Jacobian和QA）`9.74 s`，仅慢约
  `5.8%`；Python算法内部约`8.98 s`，peak RSS约`283 MiB`，FSL约`113 MiB`。
  稀疏PCG融合候选虽可保持Python既有五类输出bitwise，但稳态变慢约`0.7 s`；另一种
  旧NEWMAT串行归约会降低FSL数值一致性，两者均未进入默认路径；
- focused optimizer/nonlinear回归为`25 passed, 1 skipped`；排除参考环境既有
  MNE/Numba缓存导入问题后的全仓回归为`419 passed, 6 skipped`，compileall和
  diff检查通过；P10/M10完成条件满足，阶段complete。下一阶段只剩P11 QA、DAG和
  隔离发布验收。

### P11：QA、DAG和发布验收

正式QA至少包括：

- b0和brain mask；
- raw/corrected mean DWI、FA和SSE；
- per-volume motion/eddy参数；
- original/rotated bvec角度；
- fieldmap/TOPUP field、voxel shift和Jacobian；
- DTI-FA/T1 overlay和V1方向；
- tensor特征值、FA、有效体素和异常计数；
- 最终 `scalar/vn/dir/mc` FEM smoke。

完成条件：

- 每个阶段有独立manifest、输入fingerprint、参数、后端、版本、wall/CPU/RSS；
- 长任务显示阶段、已完成量和真实进度；
- cache命中仍进行结构检查，最终验收另做数值验证；
- Linux/macOS/Windows纯Python测试通过；
- SimNIBS 4.6集成平台完成至少一个真实完整subject；
- 只有全部正式门禁通过后才把公开定位从“post-preprocessing FSL-free”改为
  “SimNIBS dwi2cond FSL-free replacement”。

当前阶段证据（2026-08-23，进行中）：

- 2026-08-24 已完成 `v0.2.0` 本地发布候选收口：精确 SimNIBS 4.6 环境在
  `NUMBA_NUM_THREADS=3` 的低核复核中为 `530 passed, 7 skipped`，TOPUP、EDDY、
  FNIRT/nonlinear 三个真实合成 E2E 均完成；合并 coverage 为
  `12444/12444 statements`、100%。请求 8 workers 会在低核设备安全收敛到 Numba
  可用槽位，不改变算法、迭代或归约合同；
- `v0.2.0` wheel/sdist/SBOM/SHA256 已重建并通过版本、Markdown、reference assets、
  tracked-file/archive 隐私、`validate-pyproject`、`check-manifest`、Twine、wheel
  contents、隔离 overlay 安装、CLI、`pip check` 和 `pip-audit`。当前只剩冻结提交的
  Linux/macOS/Windows/package 与 SimNIBS 4.6 self-hosted 远端记录，因此 P11/M11
  暂保持 `in_progress`；远端全绿后关闭并打 `v0.2.0` tag；

- 已实现完整 workflow DAG、强输入 SHA-256 fingerprint、原子 stage manifest、结构
  cache 校验、最终数值复核、wall/CPU/RSS 和阶段进度；nomoco、legacy、eddy、affine、
  nonlinear 及 `scalar/vn/dir/mc` FEM smoke 均已接入，失败不静默回退；
- P11 新增 `pipeline.py`、`qa.py`、`workflow.py` 已达到 `645/645 statements`、100%；
  新增 `scripts/run_coverage.py` 将普通测试、TOPUP/EDDY/FNIRT 真实合成 E2E 放入互相
  独立的冷 Numba cache 后再合并 coverage，正式运行的持久缓存不受影响；该统一命令在
  精确 SimNIBS 4.6 overlay 中为 `507 passed, 6 skipped`，三个真实 E2E 均完成；
- 当前源码全包为 `11645/11645 statements`、100%，未降低阈值，未使用 omit/pragma；
  TOPUP 为 `1802/1802`、EDDY 为 `1764/1764`、FNIRT+nonlinear 为 `1205/1205`。
  Linux/macOS/Windows CI 和 tag release 已统一改用该 Numba-aware coverage 入口；
- 最终 wheel SHA-256 为
  `bb40cb4bd5cad83cdbfe070ba0cc83c64b52f1a619ecca9e5ad1af341ce78dd6`，已在新建的
  `--system-site-packages` overlay 中 `--no-deps` 安装；Python `3.11.15`、SimNIBS
  `4.6.0` 及十个关键依赖逐项匹配，`pip check` 无破损依赖；
- 最终 wheel 的公开 nomoco+nonlinear 完整 DAG：冷启动 `58.37 s`、跨输出热启动
  `20.58 s`、结构 cache 命中 `0.77 s`，
  峰值 RSS 分别约 `540/310/141 MiB`；同一 SimNIBS 环境的源树冷启动 `58.55 s`，
  wheel 与源树 40 个 NIfTI 数组及 affine 全部 bitwise 相等，wheel 冷/热输出也全部
  bitwise 相等；
- 私有 HCP whole-head 探针显示旧 FNIRT Level 3 单次 Hessian 约 `80 s`，该 level 总计约
  `7 min`；运行按用户要求在 Level 4 开始时中止，因此这不是完成的 HCP 端到端结果。
  本轮保持 FSL 三次 B-spline 局部支撑、下三角装配、逐元素体素累加和 LM/PCG 合同，
  只把通用 Kronecker/SciPy 路径替换为专用 Numba 稀疏内核：六个位移块共享遍历、交叉
  块按真实稀疏工作量分配、固定结构编译缓存，并直接构建当前 mask 的 CSR；没有降低
  分辨率、减少迭代、修改停止条件或启用 fast-math；
- 同一 HCP Level 3 输入、Numba cache 已预热时，8 workers 的首次 Hessian 为
  `11.28 s`，相对旧约 `80 s` 快约 `7.1x`，峰值 RSS 约 `5.9 GiB`；16 workers 为
  `8.51 s`。Level 1/2 的 8-worker 首次 Hessian 分别为 `0.37/1.60 s`，16-worker 为
  `0.41/1.34 s`。8 workers 继续作为 reference A/B 默认，16 workers 只作为普通设备
  可选加速档；不依据本机 40 个物理核扩大公开默认；
- 同一真实 Level 3 的单次 LM 微剖析（只把 `maximum_iterations` 设为1用于计时，不作
  正式算法输出）为 `15.75 s`：Hessian `11.25 s`、两次 cost `2.42 s`、gradient
  `1.20 s`、34步 PCG `0.76 s`。因此剩余主热点仍是 Hessian，而不是 PCG。该证据是
  阶段微基准，不能外推为四级 FNIRT、完整 DAG 或相对 FSL 的端到端加速比；
- 逐阶段 HCP/FSL A/B 定位出两个此前被小型 fixture 和全局 Frobenius 指标掩盖的问题：
  moving DTI-FA 平滑错误使用 T1 的 `0.7 mm` voxel size，而不是自身 `1.25 mm` voxel
  size；六个并行 Hessian CSC block 又共享了会被 `eliminate_zeros()` 原地修改的结构数组。
  修复后初始 warped moving 与 FSL bitwise 相等，初始 total cost 为
  `1308.814429567`（FSL verbose 为 `1308.82`）；8-worker 首次 LM candidate cost 为
  `505.868724685`，与1-worker `505.868724680`及FSL `505.869`一致。六个位移子块逐块
  对1-worker的相对误差为`5.0e-17`至`1.42e-16`；测试现逐块检查且使用非立方 shape，
  不再允许全局大对角块掩盖交叉块错误；
- 固定同一 FSL warp 后，Python PPD tensor公共support Dice为`1.0`，tensor relative L2
  `1.51e-6`，V1轴向角mean/P99为`0.000133/0.001044°`，证明残余端到端误差不来自
  PPD；直接用同一FSL tensor做派生分解时，V1 mean/P99/max均约`2.15e-7/1.21e-6/
  1.21e-6°`，FA只有极少数FSL边界/tie异常值；
- 修复后的完整HCP四级、固定8-worker nonlinear边界墙钟为`360.63 s`，算法QA记录
  FNIRT estimation/output write/tensor PPD为`186.64/25.11/147.71 s`，peak RSS约
  `26.1 GiB`。最终相对FSL的形变向量误差mean/P99/max由修复前
  `0.614/2.048/3.045 mm`降至`0.0130/0.0626/0.1828 mm`；registered FA relative L2
  由`0.215`降至`0.00839`；tensor relative L2由`0.182`降至`0.00631`；support Dice
  由`0.98563`升至`0.99962`；V1轴向角mean/P99由`12.58/79.69°`降至
  `0.343/3.104°`。Level 1末端仍有`0.0534 mm`平均场差，符合迭代中末位差异累积，
  因而不宣称bitwise一致；
- 进度输出已改为每个FNIRT level只保留一个持续更新的细节条，phase/iteration/PCG/
  topology显示在postfix，完成四级后继续明确显示warp写出、tensor PPD和QA阶段，不再
  为每个phase新建嵌套条或提前显示100%。本轮focused nonlinear为
  `37 passed, 2 skipped`，CLI为`22 passed`，ruff check/format通过；按用户要求未重复
  运行全仓coverage/full suite；
- EDDY 最后一次冗余局部赋值内联前后的 10 个数值/文本 artifact 逐文件 bitwise 相等；
  该改动没有改变循环、浮点运算或归约顺序。当前仍缺 Linux/macOS/Windows 远端实际
  green 记录和 SimNIBS 4.6 集成平台完整真实 subject 记录，因此 M11 保持
  `in_progress`。

## 7. 统一验证协议

每个里程碑依次经过四级验证：

| 级别 | 目的 | 最低证据 |
| --- | --- | --- |
| V0 | 静态和结构合同 | CLI、shape、dtype、affine、失败语义、确定性 |
| V1 | 已知真值 synthetic | 变换、field、outlier、tensor方向或数组真值 |
| V2 | 同输入 FSL 阶段 A/B | 中间参数、数组误差、mask、图像和日志 |
| V3 | 下游科学合同 | 最终tensor、conductivity、FEM和QA |

数值阈值不得在看到候选结果后为“刚好通过”而设置。每个模块的正式阈值在实现优化前，
根据 synthetic 真值精度、reference重复性和下游敏感性写入 fixture manifest。报告至少
包含：

- `max_abs`、`mean_abs`、`p99_abs`和relative L2；
- mask Dice、体积比和边界距离；
- transform角点位移及rotation/translation差；
- bvec最大/平均角度差；
- field/warp方向、幅度、Jacobian determinant和fold count；
- tensor有效体素、FA/SSE、特征值和主方向角度；
- 计时边界、线程、硬件、预热、同步和峰值内存。

局部微基准不得外推为raw-DWI到FEM端到端性能。

## 8. 实施顺序与状态

| 里程碑 | 内容 | 依赖 | 当前状态 |
| --- | --- | --- | --- |
| M0 | P0 reference harness、版本和fixture manifest | 无 | complete |
| M1 | P1有限图像操作与方向合同 | M0 | complete |
| M2 | P2 rigid/affine、transform和resampling | M1 | complete |
| M3 | P3 b0 reference和brain mask | M2 | complete |
| M4 | P4 `nomoco` raw-DWI闭环 | M3 | complete |
| M5 | P5 legacy correction | M4 | complete |
| M6 | P6自动T1 rigid/affine闭环 | M2、M4 | complete |
| M7 | P7 GRE fieldmap | M5 | complete |
| M8 | P8 TOPUP固定子集 | M3、M2 | complete |
| M9 | P9 EDDY固定子集 | M8、M4 | complete |
| M10 | P10 nonlinear T1和tensor重定向 | M6 | complete |
| M11 | P11全部模式E2E、CI和发布验收 | M7、M9、M10 | in_progress |

每次只允许一个里程碑处于 `in_progress`。里程碑只有在代码、测试、reference A/B、
文档和台账证据同时完成后才能标记 `complete`。

## 9. 首个施工任务：M0

M0 不实现新算法，只建立后续不会反复变化的验收地基：

1. 固定本地 SimNIBS 4.6和FSL reference版本、命令及环境；
2. 建立公开 synthetic DWI、b0 motion、reverse-PE、fieldmap、affine和tensor fixtures；
3. 为私有真实fixture建立不含被试标识和体数据的manifest模板；
4. 将官方脚本的每个阶段输出映射成结构化artifact列表；
5. 写reference runner，缺少FSL时显式skip；
6. 运行官方 `nomoco` 最短闭环并冻结第一份阶段基线；
7. 独立检查公开artifact不含FSL文件、MRI数据、绝对路径或凭据。

M0已于2026-08-22完成。后续逐模块复现时，仍禁止跨过尚未冻结的上游合同直接实现
EDDY、TOPUP或FNIRT。

## 10. 明确不做

- 不实现通用 FSL 命令替代器；
- 不实现多壳扩散模型、bedpostx、probtrackx或TBSS；
- 不实现EDDY未被SimNIBS调用的全部高级选项；
- 不以降低分辨率、减少迭代、改变停止条件或删除outlier步骤换取速度；
- 不把“可以运行”当作“已与reference等价”；
- 不把私有MRI、FSL源码/二进制或subject derivative加入公开仓库；
- 不静默回退到无校正、affine、scalar conductivity或其他后端。
