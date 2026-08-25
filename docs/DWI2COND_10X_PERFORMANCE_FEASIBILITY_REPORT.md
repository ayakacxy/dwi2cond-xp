# dwi2cond-xp 全流程 10× FSL 可行性与瓶颈分析

> **后续版本决策（2026-08-26）：** 本文是历史性能规划，不再定义 v0.3.0。
> v0.3.0 已改为算法流程与计算逻辑修复版；本文候选性能工作顺延至
> v0.4.0 或更高版本，且仍须先满足同算法数值 A/B 门禁。

> 数值与性能证据截至：2026-08-23
> 分析日期：2026-08-24
> 优化约束：保持 SimNIBS 4.6 / FSL 6.0.4 算法、迭代、停止条件和数值合同
> 正式 Python 配置：8 workers；artifact-cache hit 不计入计算加速比
> 路线图状态：10× 不作为 `v0.2.0` 发布门禁，进一步性能工作延期到 `v0.3.0`

## 技术结论：affine 比 nonlinear 更接近可优化区间，但两者都没有正式 10× 证据

现有证据不能支持“完整 dwi2cond 已经比 FSL 快 10×”。更准确的判断是：

- **HCP `nomoco + affine` 路径呈现了比 nonlinear 更小的规划差距。** 使用未压缩
  `.nii` 的两个独立阶段相加，Python 为 `72.05 s`、FSL 为 `490.48 s`，规划性比值
  为 `6.81×`；达到
  `10×` 还需把 Python 阶段和降到 `49.05 s`，即再加速 `1.47×`。若正式输入为
  `.nii.gz`，Python 阶段和为 `106.15 s`，则还需 `2.16×`，必须同时优化压缩输入
  和 affine，不能只改其中一个。由于这些是独立阶段相加，且 affine 实测曾达到
  `2351%` CPU，它们不是严格 8-core、压缩输入、完整 DAG 下“接近 10×”的实测证据。
- **HCP `nomoco + nonlinear` 不能在当前证据下确认 10×。** 当前 DAG 的 `.nii`
  阶段和为 `432.68 s`，`.nii.gz` 为 `466.78 s`；FSL nonlinear 正式 harness 在
  `600.103 s` 超时且没有成功输出，因此完整 FSL denominator 仍未知。
- **在 `N≈600–1200 s` 的规划场景中，nonlinear 的 10× 依赖组合优化。** Python 必须
  同时压缩 FNIRT estimation、PPD 数据流、nonlinear 前置 affine 和 gzip/I/O；仅优化
  一个热点不可能达到目标。如果最终 native FSL 远慢于该区间，所需优化幅度会相应下降，
  所以不能在取得成功 denominator 前把组合优化写成无条件必要条件。
- **在三个独立阶段的 `.nii` 内部 QA wall 规划分解中，77.778% 集中在 FNIRT
  estimation 与 tensor nonlinear/PPD。** 这不是正式 `.nii.gz` 完整 DAG 的实测占比。
  PPD 又在 `260×311×260` 网格上为约 2,102 万体素构造多个全尺寸坐标、Jacobian、
  eigensystem 和 rotation 数组，而最终有效 tensor 只有约 516 万体素。这是最大的
  等价数据流优化机会，也是 `26.1 GiB` 峰值内存的主要嫌疑边界。
- **FNIRT 的下一优先级不是继续盲挖 PCG。** 当前 moving pull 仍是串行 Numba 三重循环，
  且在不需要 derivatives 的调用中仍物化完整 coordinates、零 derivative 和 mask；
  这是 Hessian 已经大幅优化后新增的高可信候选。
- **10× 不应定义成所有模式的统一承诺。** `nomoco`、`legacy`、`eddy+topup` 和 GRE
  是互斥路径；当前真实 HCP 证据只足以规划 `nomoco + affine/nonlinear`。TOPUP、EDDY
  和 GRE 的小 fixture 结果不能外推成全模式 10×。

因此，`v0.3.0` 性能周期最合理的目标不是直接承诺 10×，而是先取得一对成功且产品边界
已冻结的完整 DAG，
随后按“去掉 nonlinear DAG 的无用 affine tensor 工作 → PPD 分块融合 → FNIRT 完整
level profile 与 Hessian 优化”的顺序推进。若第一轮组合优化后完整比值仍低于 `5×`，
应把近期目标改为稳定 `5×+`；只有 pilot 达到 `9.5×` 以上才值得进入 10× 重复确认。

本报告使用精确表格而没有绘制综合柱状图，因为 FSL nonlinear 是右删失的超时值，
完整 DAG 尚未实测，且 `.nii`/`.nii.gz`、线程和输出合同不同；图形会把规划场景误画成
已经测得的性能。

## “整体流程”必须先冻结成一个具体分支

### 正式性能目标应使用 native nonlinear 产品边界

建议把 `v0.3.0` 性能工作的主要验收边界冻结为：

```text
同一 HCP b0+b1000 单壳 raw DWI (.nii.gz)
  -> nomoco + FSL-semantics WLS
  -> T1 preparation + 12-DOF affine matrix + 6-DOF QA
  -> fixed four-level FNIRT (subsamp=8,4,2,2; miter=5,5,5,5)
  -> nonlinear VECREG PPD tensor reorientation
  -> tensor + FA/V1 + warp/field/Jacobian + mandatory product QA/manifest
```

这一路径从 CLI 进程启动前开始计时，到 fresh output directory 中全部合同输出关闭并完成
原子 manifest 为止。数值 A/B 的重新读取放在计时外。FEM、lead field 和完整 artifact
cache hit 不在被替代的 FSL 预处理边界内。

两边原生产品合同并不完全相同。FSL 写它原生 nonlinear 分支的输出；Python 计时包含
自身公开合同强制要求的 valid mask、两套 Jacobian、结构化 QA 与原子 manifest，因此
主指标是 **native FSL 产品边界对 Python 必需输出超集**，对 Python 更保守，但不是
严格 same-output benchmark。基准 manifest 必须分别冻结两边输出清单，并对 tensor、
FA/V1、warp/field/Jacobian 等共同科学数组使用相同 `.nii.gz` 压缩边界。另行记录只覆盖
共同 artifact 交集的阶段诊断值仅用于定位开销，不用于对外发布完整流程加速比。

### 当前 DAG 比上游 native nonlinear 分支多做了一套 affine tensor 输出

当前代码的 `register_nonlinear` 依赖完整 `register_t1`。后者不仅估计 affine matrix 和
生成 T1/QA，还先完成 affine tensor resampling、reorientation、FA/V1 decomposition 和
gzip 输出；随后 nonlinear 又从原始 DTI tensor 重新生成最终 nonlinear tensor。

上游 SimNIBS 4.6 的 nonlinear 分支是：

```text
12-DOF FLIRT matrix -> FNIRT -> nonlinear vecreg -> tensor decomposition
```

它不会先写一套最终不被 nonlinear 消费的 affine tensor。因此：

- `nomoco + 完整 affine + nonlinear` 可以描述当前 DAG 的阶段和，但不是 native 上游
  nonlinear 的精确边界；
- 直接用 `nomoco + nonlinear` 又会漏掉必需的 T1 preparation、affine matrix 和 QA；
- `v0.3.0` 可为 nonlinear DAG 增加“matrix/T1/QA-only”前置路径，跳过无用 affine tensor
  产物；这更接近上游源码逻辑，不是删减 FNIRT、PPD 或最终输出算法。

现有 affine profile 中，tensor/派生量后处理 critical path 为 `29.046 s`。跳过该分支
的实际端到端收益会受 6-DOF QA 并行重叠影响，不能预先把全部 `29.046 s` 当作必然节省，
但这是高可信、低科学风险的第一项工程改造。

## 已有数据只能形成阶段规划模型，不能替代完整 DAG 实测

| 阶段 | Python wall | FSL wall | 比值或状态 | 证据边界 |
| --- | ---: | ---: | ---: | --- |
| HCP `nomoco`，`.nii` mmap | `18.82 s` | `257.30 s` | 规划 `13.67×` | 同数值 fixture/shape；FSL manifest 未冻结输入编码；Python 8 workers。 |
| HCP `nomoco`，`.nii.gz` | `52.92 s` | `257.30 s` | 规划 `4.86×` | 同数值 fixture/shape，FSL 输入编码未冻结；Python wall 含解码与临时 `.nii` 物化。 |
| HCP 完整 affine T1 registration | `53.23 s` | `233.18 s` | `4.38×` | Python 请求 8 candidate workers，但底层达到 `2351%` CPU。 |
| 修复后 HCP FNIRT+PPD | `360.63 s` | `>600.103 s` | 不发布倍数 | Python 完成；FSL 1-thread harness 超时且 `outputs=[]`。 |

来源与数值误差见[整体完成度与性能报告](DWI2COND_COMPLETION_AND_PERFORMANCE_REPORT.md)、
[基准记录](BENCHMARKS.md)和[实施计划](FSL_SUBSET_REIMPLEMENTATION_PLAN.md)。这些阶段
来自同一服务器和同一 HCP subject，但不是同一趟修复后的完整 DAG；`257.30 s` 的 FSL
manifest 也没有冻结输入是 `.nii` 还是 `.nii.gz`。任何相加或由此得到的比值都必须标为
“规划推导”，不能视为同编码输入的正式 A/B。

### affine 阶段和展示了较小差距，但尚未形成正式 10× 基准

未压缩路径的阶段相加为：

```text
Python: 18.82 + 53.23 = 72.05 s
FSL:   257.30 + 233.18 = 490.48 s
ratio: 490.48 / 72.05 = 6.8075x
10x Python budget: 490.48 / 10 = 49.048 s
```

这意味着在该规划模型中还需节省 `23.002 s`、降低 `31.93%` wall，或在当前基础上再快
`1.469×`。
如果 nomoco 保持 `18.82 s`，affine 必须从 `53.23 s` 降到 `30.23 s` 以下，即
`1.761×`；只优化 nomoco 不可能达到目标，因为 affine 自身已经超过 `49.048 s`。

对于正式 `.nii.gz` 输入：

```text
Python: 52.92 + 53.23 = 106.15 s
current ratio: 490.48 / 106.15 = 4.621x
required further speedup: 106.15 / 49.048 = 2.164x
```

两个 Python 阶段都单独超过或接近总预算，因此压缩输入与 affine 必须共同优化。若 affine
达到 `2×`，`.nii.gz` nomoco 仍需降到 `22.43 s` 以下。这里只能说明存在明确优化空间，
不是已经满足或接近 10×；正式结果还需严格 8-core 总预算重测，因为现有 affine 使用了超过
8 个逻辑执行单元。

### nonlinear 路径的可行性取决于尚未知的 FSL 完成时间

当前 DAG 的外层阶段和为：

```text
Python .nii:    18.82 + 53.23 + 360.63 = 432.68 s
Python .nii.gz: 52.92 + 53.23 + 360.63 = 466.78 s
```

为了做规划，可把完整 affine reference 与 nonlinear 成功时间 `N` 相加：

```text
FSL expanded stage-sum planning model = 257.30 + 233.18 + N = 490.48 + N
known only: N > 600.103 s
10x budget B10(N) = (490.48 + N) / 10
```

这个模型给双方都增加了 native nonlinear 不需要的完整 affine tensor 边界，因此不能
冒充上游完整 CLI 实测；它的用途只是显示 10× 对未知 `N` 有多敏感。

| 假设成功 FSL nonlinear `N` | FSL 阶段和模型 | 10× Python budget | 当前 `.nii` 还需加速 | 当前 `.nii.gz` 还需加速 |
| ---: | ---: | ---: | ---: | ---: |
| `600.103 s`，仅为超时下界极限 | `1090.583 s` | `109.058 s` | `3.97×` | `4.28×` |
| `900 s`，情景假设 | `1390.48 s` | `139.048 s` | `3.11×` | `3.36×` |
| `1200 s`，情景假设 | `1690.48 s` | `169.048 s` | `2.56×` | `2.76×` |
| `1800 s`，情景假设 | `2290.48 s` | `229.048 s` | `1.89×` | `2.04×` |

除第一行的 `>600.103 s` 外，其余 `N` 都是敏感性场景，不是 FSL 测量。当前 `.nii`
阶段和若已经达到 10×，需要 `N≥3836.32 s`；`.nii.gz` 则需要
`N≥4177.32 s`，约 `69.6 min`。现有超时证据远不足以判断 FSL 是否真的需要这么久。

## `.nii` 内部 QA 规划分解中 77.778% 集中在两个 nonlinear 热点

用三次独立 Python 阶段运行各自记录的内部 QA wall 做拼接，总计为 `429.880 s`：
`nomoco=18.012 s`、affine registration=`52.403 s`、nonlinear=`359.465 s`。它不是
正式 `.nii.gz` 完整 DAG，也不是一次进程内的完整 wall；以下百分比只用于 Amdahl 规划：

| 当前组成 | 时间 | 占内部阶段和 | 结论 |
| --- | ---: | ---: | --- |
| `nomoco` | `18.012 s` | `4.190%` | `.nii` 已很快；正式 `.nii.gz` 外层 wall 仍有约 `34.1 s` 输入路径增量。 |
| 完整 affine | `52.403 s` | `12.190%` | nonlinear DAG 存在未被消费的 affine tensor 后处理。 |
| FNIRT optimization + first full-resolution expansion | `186.642 s` | `43.417%` | 最大单一边界；计时不只是四级 optimizer，返回前还包含第一次完整 coefficient expansion。 |
| FNIRT output materialization/FA warp/write | `25.111 s` | `5.841%` | 不是纯 I/O；包含 registered-FA warp 与多个输出物化。 |
| tensor nonlinear/PPD | `147.712 s` | `34.361%` | 最大的数据流、内存和向量化机会。 |

FNIRT estimation 与 tensor nonlinear/PPD 合计 `334.354 s`，占 `77.778%`。只优化
其中一个有明确的 Amdahl 上限：即使 FNIRT estimation 变成零，仍约 `243.238 s`；即使
PPD 整段变成零，仍约 `282.168 s`，都无法达到 `N≈600 s` 时的保守规划预算。

| 对 FNIRT estimation + PPD 两项做相同倍数优化 | 推导后 Python 时间 | 对 `1090.583 s` 规划下界的比值 |
| ---: | ---: | ---: |
| 当前 `1×` | `429.880 s` | `2.537×` |
| `2×` | `262.703 s` | `4.151×` |
| `4×` | `179.115 s` | `6.089×` |
| `6×` | `151.252 s` | `7.210×` |
| `10×` | `128.961 s` | `8.457×` |
| 两项理论归零 | `95.526 s` | `11.417×` |

这张表不是性能预测。它说明即使两个最大热点都快 `10×`，其余 affine、输出和 nomoco
仍使内部规划时间约 `129.0 s`；要在 `N≈600 s` 的保守规划下达到 10×，前置与输出
数据流也必须同步收缩。

## PPD 的主要问题是全网格物化，而不是 eigendecomposition 本身

tensor nonlinear 内部 `147.411 s` 的 QA 分解为：

| 子阶段 | 时间 | 占该阶段 | 代码结构观察 |
| --- | ---: | ---: | --- |
| tensor/coefficient load + float32 coefficient expansion | `11.903 s` | `8.08%` | 同时包含六分量 tensor 解码和必须保留的第二次 expansion；约 `0.84 MB` coefficient 文件重读只是很小一部分。 |
| warp resample + PPD | `117.076 s` | `79.42%` | 六个 tensor component 分别调用一次 `map_coordinates`，随后构造全网格 Jacobian/eigensystem/rotation。 |
| tensor decomposition | `8.694 s` | `5.90%` | 已不是第一优先级。 |
| parallel gzip output | `9.738 s` | `6.61%` | 多文件并行，但仍受压缩与内存带宽影响。 |

输出网格有 `21,023,600` 个体素，最终 valid tensor 只有 `5,156,328` 个，占
`24.53%`。当前实现仍会在全网格上构造或处理多类大型数组：

- 一个 `3×N` float64 coordinate/grid 数组约 `481 MiB`；计算过程中同时存在多个；
- 一个 `N×3×3` float64 Jacobian/rotation 数组约 `1.41 GiB`；
- 一个 `N×6` float32 tensor 数组约 `481 MiB`；
- backward/forward Jacobian、matrices、eigenvectors、rotation 和 rotated tensor 会形成
  多个 GiB 级重叠生命周期。

这些结构与约 `26.1 GiB` 实测 peak RSS 一致。最有价值的等价改造顺序是：

1. 原始 displacement gradient 只计算一次，分别派生 PPD Jacobian 与物理 voxel-size
   determinant。当前第二次 Jacobian 调用的 backward/forward 会被丢弃，却仍对约
   2,102 万个 3×3 矩阵执行 inverse，应首先删除这项确定冗余；
2. preliminary candidate 只能按现有 target/source/support、fold/near-singular 和 finite
   tensor 条件形成，再在其中执行允许延后的 inverse、eigh 和 PPD。low-FA 与
   repeated-eigenvalue 必须在 eigendecomposition 后按现有公式计算，并继续只作为诊断，
   不能新增为 valid 或 PPD gate；
3. 第一版为 pipeline 增加私有 chunk fast path，同时长期保留返回完整、逐数组相同
   backward/forward Jacobian 与诊断的 `NonlinearTensorResult` reference API。私有路径
   不物化 pipeline 未消费的 full-grid backward/forward 矩阵，但必须逐数组保持实际落盘
   的 determinant、masks、最终 tensor 以及全部 QA 计数。target-grid z chunk 为
   displacement 有限差分带一层 halo；source tensor sampling 不能假设使用同一 z slab，
   必须保留完整 source，或按该 target chunk 的实际 source-coordinate min/max 构造带
   插值 halo 的 bounding box。六个 tensor component 在同一 target chunk 上采样，并将
   tensor、masks 与 determinant artifacts 写入最终数组或 memmap，缩短 coordinates、
   matrices、eigenvectors 和 rotation 生命周期；
4. 在内存中保留与落盘完全相同的 `coefficients32` 可省去约 `0.84 MB` 文件重读，但
   **不能**省掉由 float32 coefficients 进行的第二次 expansion，也不能复用第一次由
   float64 coefficients 生成的 expansion；单独收益预计很低，只有与 chunk/lifecycle
   改造结合后才值得 profile；
5. 只有 SciPy chunk 版本通过并完成 profile 后，才评估将六次 sampling 融为保持边界、
   component 和每体素运算顺序的 Numba kernel；不能假设自制线性插值天然 bitwise。

这类优化不改变 warp、插值阶数、PPD 数学或输出分辨率，但会改变浮点执行组织，风险高于
简单删除重复 I/O。固定 FSL warp 下当前 tensor relative L2 `1.514e-6`、support Dice
`1.0` 和 V1 mean/P99 `0.000133/0.001044°` 是不可放宽的验收基线；纯数据复用候选优先
要求对当前 Python 输出逐数组 bitwise。

## FNIRT 仍是最大单一热点，但继续加线程不会自动解决

当前 `186.642 s` 边界包含四级 FNIRT 与返回前的第一次 full-resolution expansion。
真实 Level 3 单次 LM 微剖析为 `15.75 s`：
Hessian `11.25 s`、两次 cost `2.42 s`、gradient `1.20 s`、34 步 PCG `0.76 s`。
因此 PCG 不是优先级；但在继续改 Hessian 前，还应先 profile 和优化每次 cost/gradient
都会经过的 moving pull。

当前 `_warp_fnirt_fsl_order` 是串行 Numba 三重循环。即使
`calculate_derivatives=False`，它仍返回完整 coordinates、三分量零 derivative 和多套
mask。每个 pull voxel 可以独立计算，所以可增加 8-worker voxel-parallel kernel，并为
不需要 derivative 的内部 cost/final-output 调用提供 lean result；每体素 float32 表达式
必须原样保留，最终 SSD 等归约仍按 FSL z/y/x 顺序。这一候选应排在新一轮 Hessian
微优化之前。

Hessian 已经从旧 Level 3 单次约 `80 s` 优化到 8-worker `11.28 s`。16 workers 只有
`8.51 s`，相对 8 workers 仅 `1.33×`，只足以证明线程扩展正在趋于饱和；内存带宽、
稀疏结构构造和串行边界是待子阶段 profile 验证的可能原因。不能把“再开更多 workers”
当作达到 10× 的主方案。

仍值得验证的严格等价候选包括：

- 将 selected spline basis 支撑生成与六个 Hessian block 累加进一步融合，减少每次
  LM 的 CSR/CSC 物化、权重数组和 SciPy object 拼接；
- 将三路 partial 一次性生成 `(6, n_selected)` weight matrix，避免六次 full-grid
  multiply 和随后 `vstack` copy；
- 在 mask 完全相同时复用 selected structure；mask 变化时必须重新生成，不允许近似；
- 直接为六块分配独立结构缓冲，避免 `copy → eliminate_zeros → bmat` 的重复内存流量，
  同时保持已修复的 CSC ownership；
- 分别 profile 四个完整 level 的 cost、gradient、Hessian、topology 和 spline zoom，
  避免把 Level 3 单次微基准外推成完整 `186.642 s`；
- 只对可独立 coefficient/block 改变调度，块内继续保持 FSL z/y/x 累加顺序。

现有专用六块 Numba Hessian 已经很强，下一轮合理的首批目标只是再降低约 `5–15%`，
而不是假设还能复现此前从 `80 s` 到 `11.28 s` 的数量级收益。这一部分的收益不确定性
高于 PPD。FNIRT 是带接受/拒绝的迭代优化，末位 Hessian 差异会
改变后续轨迹；每个候选必须先通过逐 block、一步 LM 和完整 level trace，再允许跑一次
昂贵的四级 HCP。

## 其余性能债务必须配合解决，但不是第一热点

| 优化候选 | 当前证据 | 机会等级 | 主要风险或边界 |
| --- | --- | --- | --- |
| nonlinear 前置 registration 只生成 matrix/T1/QA | affine 后处理 critical path `29.046 s`；估计净收益约 `17–20 s`；native 上游不写 affine tensor | 高 | 只改 nonlinear DAG；独立 `register-t1` CLI 合同保持不变。 |
| `.nii.gz` 流式物化到共享 `.nii` | 相对 mmap 多约 `34.1 s`，peak RSS约 `4.57 GiB` | 高（正式压缩输入） | gzip 解码仍可能单线程；非标准 dtype/orientation 必须保留完整 fallback。 |
| 合并 PPD 两次 Jacobian/inverse | 第二次调用仍求 full-grid inverse，但只消费 determinant/masks | 高 | 保持 float32 gradient 后两套 float64 scaling 顺序。 |
| FNIRT moving pull 8-worker + lean result | 当前 pull 为串行 Numba，且返回部分未使用的大数组 | 高 | 每 voxel 表达式不变；SSD/mask 归约顺序不变。 |
| 复用 `coefficients32`，不复用 float64 expansion | coefficient 文件约 `0.84 MB`；`11.903 s` 还含 tensor 解码与必要 expansion | 低；与 chunk/lifecycle 融合后待 profile | 必须保留 coefficient 落盘舍入边界、第二次 expansion 与只读 ownership。 |
| 输出压缩级别、写出重叠和 pipeline scratch | nonlinear 两段输出合计占数十秒 | 中 | 不得改变 NIfTI 数组/header；并行压缩可能争抢内存带宽。 |
| 生成输出时同步计算 SHA-256 | DAG 当前可能在下游首次使用时重新读取内部 artifact | 中低 | 必须保留强内容 fingerprint，不能退化成 mtime/size。 |
| pipeline QA resolved-path 去重 | nomoco raw/corrected、最终 FA/registered FA 可能指向同一路径；旧 QA stage `16.23 s` | 中低 | 保留两套公开 QA artifact；只复用同一路径 load/mean。 |
| 预加载 Numba kernel | 冷启动对小 fixture 明显，对 HCP 数百秒流程占比小 | 低 | 只能改善启动，不等于算法提速。 |
| 增加到 16+ workers | Level 3 Hessian 8→16 仅 `1.33×` | 低 | 普通设备可移植性、带宽饱和和过度订阅。 |

`.nii.gz` 的第一步不应直接引入平台相关二进制。对于 float32、orientation identity 的
常见输入，可以先把 gzip 流顺序物化为合法 `.nii`，随后 mmap 分块验证和原地执行
nonnegative 合同，避免当前整幅解码、contiguous reorientation 和额外 copy 的重叠。
其他 dtype/orientation 继续走现有通用路径。任何替代 gzip backend 都必须先证明 Linux、
macOS、Windows wheel 可用且解码字节一致。

EDDY 路径还有独立问题：大型 `outlier_free_data.nii.gz` 目前会被多进程 z-block fitting
反复打开并读取不同 block，未来应先 staging 为一次性共享 `.nii`。但当前没有 HCP EDDY
完整计时，所以该候选不进入本报告的 10× 数值预算。

## 10× 的现实可行性分为三档

| 目标 | 当前判断 | 达成条件 |
| --- | --- | --- |
| HCP `.nii` `nomoco+affine` 达到 `10×` | **中等、待严格重测** | 阶段规划还需 `1.47×`；必须优化 affine，并取得严格 8-core 完整 DAG 实测。 |
| HCP `.nii.gz` `nomoco+affine` 达到 `10×` | **中低、仍需组合优化** | 正式输入的阶段规划还需 `2.16×`；gzip 与 affine 必须同时改善。 |
| HCP `.nii.gz` native `nomoco+nonlinear` 达到 `10×` | **条件可行、风险较高** | 成功 FSL denominator、Python PPD 大幅下降、FNIRT 至少再取得显著完整流程收益、去掉无用 affine 工作。 |
| 所有 `legacy/eddy/topup/GRE/nonlinear` 模式统一达到 `10×` | **当前不支持** | 需要各分支真实同输入完整 A/B；现有小 fixture 倍数差异过大。 |

对 nonlinear 最有希望的条件场景是：成功 FSL nonlinear 最终落在约
`900–1200 s` 或更高，同时 Python native `.nii.gz` 完整 DAG 经组合优化降到约
`140–170 s`。这时 10× 才可能进入重复确认。这里的时间范围是由预算反推的设计目标，
不是对尚未实现优化的性能预测。

若要求在 `N≈600 s` 的最严格规划下仍确保 10×，Python 总预算约 `109 s`。在保留
相同算法、8 workers、`.nii.gz` 输出和 QA 的条件下，这要求多个大阶段同时获得接近极限
的收益，属于 stretch goal，不应在完成首轮原型前作为版本承诺。

## `v0.3.0` 按最小实验矩阵先确定 go/no-go

### 第一步：取得真正可比较的 baseline

| ID | 实验 | 重复 | 用途 |
| --- | --- | ---: | --- |
| A0 | 冻结输入 SHA-256、shape/affine、参数、输出清单和 8-core cpuset | 1 | 不计时预检。 |
| A1 | native FSL 完整 `nomoco+nonlinear`，提高 timeout，fresh output | 1 | 取得成功 denominator；失败则暂不判断 10×。 |
| A2 | 当前 Python 8-worker，空独立 Numba cache，fresh output | 1 | cold-JIT 完整边界。 |
| A3 | 当前 Python 8-worker，新进程、已有 kernel cache、fresh output | 1 | 生产主指标 pilot。 |
| A4 | Python 1-worker、warm kernel cache、fresh output | 1 | 区分实现效率与 8-worker 并行收益。 |
| A5 | artifact-cache hit | 1 | 只报告重复调用 UX，不进入计算倍数。 |

双方使用同一个 `.nii.gz`、共同科学数组使用同一 `.nii.gz` 格式，并固定各自原生输出
manifest；Python 的必需输出超集全部计时。Python 总进程树限制在 8 个物理核预算，每个
process 的 BLAS/OpenMP 为 1；FSL 使用未修改 reference，如果
实际仍约单核，只能表述为“Python 8-worker production 对未修改、观测为近单核的 FSL”，
不能说成 8-thread 对 8-thread 的效率比较。

OS page cache 不应通过破坏性系统操作强行清空。双方采用相同 page-cache warm 条件并
交替顺序；如果 pilot 接近目标，再按 `FSL→Python、Python→FSL、FSL→Python` 增加到
至少三对 fresh-output 运行。记录 wall、user/system、整个进程树 CPU、peak RSS、I/O
bytes、swap、输出大小和后台负载。

### 第二步：按收益与风险排序开发，不重复跑全量

1. **只增加细粒度计时。** 分开记录 FNIRT 各 level 的 pull/basis/weights/JtJ/assembly/
   final expansion，以及 PPD 的 coordinates/sample/Jacobian/inverse/eigh/rotation；不改输出。
2. **nonlinear matrix-only registration。** 先在小 fixture 和保存的 HCP registration
   输入上验证，最终 affine matrix、T1/QA 与当前路径一致；不跑 FNIRT。
3. **固定 warp PPD benchmark。** 复用现有 HCP tensor、reference 和 FSL warp，只测
   tensor nonlinear；先删第二次 inverse/Jacobian 冗余，再做 float32 coefficient 直通、
   mask-aware SciPy chunk，最后才评估融合插值。
4. **FNIRT pull 与 Level 3 一步 LM。** 先验证 voxel-parallel/lean pull，再用保存的
   objective 分别 profile basis、weights、六块
   Hessian、sparse assembly 和复制；候选通过后再做一个完整 level。
5. **`.nii.gz` 单次流式物化。** 只测输入规范化与下游 mmap bitwise，不重复跑完整
   registration。
6. **QA 同路径 load 去重与可选 gzip 策略。** 单独验证公开 QA 数值和 NIfTI 数组；
   不把压缩文件 SHA 改变误判成科学数值改变。
7. **里程碑才跑一次完整 DAG。** 只有上述候选各自通过数值门禁后才组合，避免每次改动
   都等待完整 HCP。

### 第三步：用明确条件决定是否继续追 10×

第一对成功 pilot 的比值定义为：

```text
R = FSL complete wall / Python warm-kernel fresh-output complete wall
```

- `R < 5`：停止把“完整 nonlinear 10×”作为该性能周期目标，先以稳定 `5×+` 和降低
  RSS 为目标；
- `5 ≤ R < 9.5`：继续热点优化，但不能进入 10× 确认或公开措辞；
- `R ≥ 9.5`：完成至少三对交替运行；内部通过要求中位数 `≥10.0×`、每一对
  `≥9.5×`；
- 对外声称“稳定 10×”建议至少五对，公开全部样本，并要求 bootstrap 95% 下界不低于
  `10×`；否则只能说“同机观察到中位数约 10×”。

## 精度与内存门禁不因 10× 目标而放宽

性能候选首先对冻结的当前 Python 输出做回归：

- 只改变复用、调度或 I/O 且可保持顺序的 artifact，优先要求逐数组 bitwise；
- PPD 融合必须保持固定 warp tensor relative L2 `1.514e-6`、support Dice `1.0`、
  V1 mean/P99 `0.000133/0.001044°`，且不得改变 mask、affine 或 tensor 分量顺序；
- FNIRT 必须比较每 level attempts、accepted iterations、cost trace、Jacobian range、
  六个 Hessian block 和一步 LM；不能只看最终全局 relative L2；
- 当前完整轨迹 attempts 为 `[8,6,7,9]`，四级各有 5 次成功迭代；调度或内核候选不得
  静默改变 accept/reject、damping 或 PCG 轨迹；
- 最终 FSL A/B 不得差于 field mean/P99/max
  `0.013025/0.062561/0.182772 mm`、registered FA relative L2 `0.0083907`、tensor
  relative L2 `0.0063056`、support Dice `0.9996219` 和 V1 mean/P99
  `0.34296/3.10381°`；
- `fold_voxels=0`、`near_singular_voxels=0`、无 NaN/Inf、shape/affine 和输出集合不变；
- 不降低分辨率、不减少迭代、不修改停止条件、不改变 tie-breaking/归约顺序，不使用
  `-ffast-math`、autocast 或 silent fallback。

内存使用同一种整个进程树采样方法，禁止 swap/OOM。当前 baseline 约 `26.1 GiB`；默认
候选不得超过其 `5%`，即约 `27.4 GiB`，除非单独批准。PPD chunking 应把 RSS 明显降低，
但在实测前不写具体倍数。FSL timeout manifest 中的 RSS 不是完成态峰值，不能据此声称
Python 比 FSL 更省内存。

## 当前绝对不能对外声称的结论

- “完整 dwi2cond 已经比 FSL 快 10×”；
- “所有预处理模式都能达到 10×”；
- “HCP nonlinear 已经比 FSL 快多少倍”；
- “8 threads 对 8 threads 快 10×”或“单核实现效率快 10×”；
- 把 DTI fitting 的 `11.09×` 外推成完整流程；
- 在 FSL 输入编码尚未冻结时，把 Python `.nii` 或 `.nii.gz` 阶段规划比写成正式 A/B；
- 把 `0.77 s` artifact-cache hit 当成重新计算被试；
- 只测 warm kernel cache 后声称“冷启动也达到 10×”；
- 把不同 fixture、不同 workers 或阶段倍数相加成正式 E2E speedup；
- “FNIRT 与 FSL bitwise 一致”或“为了 10× 可以接受更低精度”；
- “当前实现已获得跨平台性能验证”或“Python 比 FSL 更省内存”；
- 从单个 HCP subject 外推成一般被试或普通 8--16 线程设备上稳定达到 10×。

## `v0.3.0` 性能工作开始前仍需回答的问题

- 成功完成的 native FSL HCP nonlinear 和完整 `nomoco+nonlinear` 到底需要多久？
- 当前修复后、固定 8-worker、`.nii.gz`、fresh-output 完整 Python DAG 的 outer wall、
  CPU 和 peak RSS 是多少？
- nonlinear matrix-only registration 能实际节省多少 critical path，而不是理论最多
  `29.046 s`？
- PPD 的 `117.076 s` 中，六次 interpolation、两次 Jacobian、全网格 inverse/eigh、
  coordinate construction 各占多少？
- mask-aware subset 调用 SciPy interpolation 能否保持当前输出 bitwise；若不能，误差边界
  出现在坐标、权重、float32 舍入还是边界处理？
- FNIRT 四级总 `186.642 s` 中各 level 与 cost/gradient/Hessian/topology 的完整分布
  是什么？
- 严格 8 个物理核心后，现有 `2351%` CPU 的 affine wall 会变成多少？
- 第二个真实 subject 是否维持相同有效 support 比例、内存压力和 10× 可行性？

在这些问题中，前两项决定 10× 是否是现实目标；第三到第六项决定优化顺序；最后两项
决定结果能否从当前服务器推广到普通 8–16 线程设备。
