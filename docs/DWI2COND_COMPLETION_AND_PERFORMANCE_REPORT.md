# dwi2cond-xp 整体完成度与性能分析

> 数值与性能证据截至：2026-08-23
> 工作树与发布审计截至：2026-08-24
> 目标版本：`v0.2.0`
> 参考实现：SimNIBS 4.6.0 / FSL 6.0.4 `ddd0a010`
> 正式默认并行度：8 workers

> 发布决策更新（2026-08-24）：`v0.2.0` 以纯 Python 替代 SimNIBS 4.6 `dwi2cond`
> 所需 FSL 运行依赖为核心，不再以全流程 10× 或新增性能优化作为发布门禁。进一步
> FNIRT/PPD 性能工作延期到 `v0.3.0`；本报告中的发布前状态仍保留为审计快照。

> 发布门禁更新（2026-08-24）：精确 SimNIBS 4.6 本地低核复核为
> `530 passed, 7 skipped`；冻结提交 `90a0553` 的 Linux/macOS/Windows/package
> GitHub Actions run `32683104733` 全绿，Linux 为 `535 passed, 7 skipped`，合并
> coverage `12443/12443 statements`、100%。这些证据取代下文历史候选状态。

## 技术结论：核心 FSL 子集与 v0.2.0 发布门禁已经完成

`dwi2cond-xp` 已经完成 P0–P11，共 12/12 个阶段；SimNIBS 4.6
`dwi2cond` 实际使用的 DTI fitting、MCFLIRT/FLIRT、BET、GRE/FUGUE 固定路径、
TOPUP、单壳 EDDY `--repol`、FNIRT、VECREG PPD 以及张量派生量均已有纯 Python
实现。运行时不需要 FSL，FSL 只保留为开发和验收 reference。

最新私有 HCP whole-head fixture、固定 8-worker nonlinear A/B 中，最终形变向量误差
mean/P99/max 为 `0.0130/0.0626/0.1828 mm`，公共 support 内 tensor relative L2
为 `0.00631`，support Dice 为 `0.99962`，V1 轴向角 mean/P99 为
`0.343/3.104°`。固定使用同一 FSL warp 后，Python PPD tensor relative L2 进一步
降到 `1.51e-6`，说明最终剩余误差主要来自迭代 FNIRT 终点，而不是 tensor PPD
重定向。

性能不是单一结论：在当前已做的同边界 A/B 中，HCP DTI、HCP `nomoco`、HCP affine
T1 registration、公开 legacy、TOPUP 和 EDDY fixture 均快于 FSL；其中 DTI 使用
16 workers，affine 底层 runtime 达到 `2351%` CPU，不能把它们包装成固定 8 个总线程
的结果。极小 GRE fieldmap fixture 仍由 FSL 明显领先，旧的小型 FNIRT+PPD 完整边界中
Python 慢 `5.8%`。修复后 HCP nonlinear 的 Python 外层墙钟为 `360.63 s`，但正式 FSL
1-thread harness 在 600 秒超时，因此当前既没有成功的 FSL 计时，也不能发布 HCP
nonlinear 的精确 Python/FSL 加速比。

当前最准确的项目定位是：

- **算法完成度：** P0–P11 complete，核心子集实现及同输入 A/B 已完成；
- **工程完成度：** DAG、QA、cache、进度、wheel overlay 和 100% coverage 已完成；
- **发布完成度：** Linux/macOS/Windows/package、CodeQL 和 OpenSSF 已对冻结提交通过，
  `v0.2.0` 正在执行 tag/Release 资产发布；
- **SimNIBS 集成：** 未修改 SimNIBS 4.6.0 环境、真实 HCP subject 和隔离 wheel overlay
  已通过；仓库没有注册 self-hosted runner，因此不把不会启动的远端队列伪装成门禁；
- **性能声明仍需：** 只有要发布 HCP nonlinear Python/FSL 加速比时，才必须补一份成功
  结束的同边界 FSL manifest；它不是不声明该加速比时的版本硬门禁。

这里的“替代 FSL”只指
[SimNIBS 4.6 `dwi2cond` 实际调用的固定子集](FSL_SUBSET_REIMPLEMENTATION_PLAN.md)，
不代表通用 FSL 命令、fMRI、统计、纤维追踪或任意 EDDY/FNIRT 配置的替代。

## 比较口径：不同 fixture、workers 和缓存状态不能混算

本报告使用以下定义：

- `relative L2 = ||Python - FSL||₂ / ||FSL||₂`；数值越小越接近 reference。
- `Dice = 2|A∩B| / (|A|+|B|)`；用于 mask 或有效 support，越接近 1 越好。
- `V1 axial angle` 把主方向视为无向轴，`v` 与 `-v` 视为相同方向。
- 性能比默认写成 `FSL wall / Python wall`；大于 1 表示 Python 更快。
- `process/CLI` 包含进程启动和文档声明的 I/O；`pipeline` 是同进程算法边界；
  `resident` 表示 import 与 JIT kernel 已常驻；`artifact-cache hit` 只复核既有产物。
- `cold Numba cache`、`populated kernel cache` 和 `artifact cache` 是三种不同状态，
  不能互换，也不能把 cache hit 描述成重新计算一个被试。

性能表使用精确查数表而不是综合柱状图，因为各阶段的输入形状、线程数、输出集合、
缓存状态和 FSL 并行方式不同。把这些数字画成同一“总加速比”会制造不存在的可比性。

主要证据分为三类：

1. **真实 HCP：** 单壳 `145×174×145×108` DWI、`145×174×145×6` tensor，
   nonlinear 输出为 `260×311×260×6`；
2. **公开非解剖 fixture：** 用于可重复的 legacy、fieldmap、TOPUP、EDDY 和
   nonlinear FSL A/B；
3. **微基准/缓存分析：** 只用于定位热点或启动成本，不外推为完整流程性能。

更完整的输入合同见 [INPUT_CONTRACT.md](INPUT_CONTRACT.md)，参考环境与失败语义见
[FSL_REFERENCE_CONTRACT.md](FSL_REFERENCE_CONTRACT.md)。

## 12/12 阶段完成：P11 已完成最终验收

| 阶段 | 内容 | 状态 | 当前最重要证据或边界 |
| --- | --- | --- | --- |
| P0 | Reference harness 与 fixture manifest | complete | FSL/SimNIBS 独立进程、版本、命令、结构摘要、计时和失败语义已冻结。 |
| P1 | 有限 NIfTI 运算、方向和 tensor decomposition | complete | 基础算子多项 array-exact；Gaussian relative L2 `2.19e-8`。 |
| P2 | 变换、重采样和 FLIRT 核心 | complete | 按 FSL default schedule 实现 6/12 DOF、MI、Brent 和 scaled-mm/world 合同。 |
| P3 | b0 reference、MCFLIRT 固定路径和 BET | complete | HCP BET Dice `0.99369`；固定 FSL 矩阵时重采样误差已单独定位。 |
| P4 | `nomoco` raw-DWI 闭环 | complete | HCP 公共 mask 内 tensor/FA/SSE relative L2 `6.46e-6/6.98e-6/3.51e-5`。 |
| P5 | SimNIBS legacy correction | complete | 固定 FSL 矩阵后 sinc resampling relative L2 `7.43e-7`；优化终点差异单独公开。 |
| P6 | 自动 T1 rigid/affine 与 finite-strain tensor 重定向 | complete | HCP tensor relative L2 `2.37e-4`，V1 mean/P99 `0.0157/0.1558°`。 |
| P7 | GRE fieldmap/FUGUE 固定路径 | complete | 已解缠 rad/s 输入的 corrected b0 relative L2 约 `6.64e-8–7.36e-8`；不含 PRELUDE。 |
| P8 | `b02b0_nosubsamp.cnf` TOPUP 子集 | complete | 最新 field relative L2 `0.00430`，corrected pair `0.00161`。 |
| P9 | 单壳 EDDY `--repol` 子集 | complete | 四个注入坏 slice 全部且仅有它们被检出；tensor relative L2 `0.00607`。 |
| P10 | 四级 FNIRT、Jacobian 和 PPD | complete | 原小 fixture 封板证据已被 P11 发现的两处修复和最新 HCP A/B 取代。 |
| P11 | QA、DAG、cache、打包和发布验收 | complete | 本地精确 SimNIBS 4.6 集成与冻结提交三平台/打包/安全门禁均通过。 |

P11 的关闭不扩大算法范围：公开定位仍只覆盖 SimNIBS 4.6 `dwi2cond` 使用的固定 FSL
子集，进一步性能优化归入 `v0.3.0`。

## 当前固定范围内的运行路径已经形成产品矩阵

| 维度 | 已实现路径 | 明确限制 |
| --- | --- | --- |
| 输入 | 已预处理单壳 DWI、六分量 tensor、raw-DWI 固定预处理入口 | multi-shell 不会被静默普通 DTI 拟合；必须显式选壳或采用独立模型。 |
| 预处理 | `nomoco`、SimNIBS legacy、EDDY `--repol` | EDDY 是 SimNIBS 4.6 使用的固定单壳子集，不是完整 EDDY CLI。 |
| susceptibility | `none`、已解缠 GRE rad/s fieldmap、固定 TOPUP | wrapped Siemens phase/PRELUDE 未实现；TOPUP/EDDY 仅支持 x/y 单轴 PE。 |
| T1 配准 | external、assume-aligned、rigid、affine、nonlinear | nonlinear 固定为 SimNIBS 4.6 的 `subsamp=8,4,2,2` 与 `miter=5,5,5,5`。 |
| tensor | FSL 六分量顺序、affine finite strain、nonlinear PPD、FA/V1/特征值 QA | 不承诺 FNIRT coefficient 或完整 field bitwise 相等。 |
| 电导率/FEM | `scalar`、`vn`、`dir`、`mc`，固定 montage FEM 和 lead-field 接口 | 真实完整全电极 lead field 尚不属于当前发布证据。 |
| 工程 | `.nii`/`.nii.gz`、8 workers、进度、DAG、SHA-256 fingerprint、原子 manifest、cache | optimized 失败不静默回退；reference 后端长期保留。 |

主数据流为：

```text
single-shell DWI
  -> nomoco / legacy / EDDY (+ fieldmap or TOPUP)
  -> FSL-semantics WLS tensor fitting
  -> rigid / affine / FNIRT T1 registration
  -> finite-strain or PPD tensor reorientation
  -> tensor / FA / V1 / field / motion / outlier QA
  -> scalar / vn / dir / mc conductivity
  -> SimNIBS FEM or lead-field entry
```

## 最新 HCP nonlinear 已把误差定位到 FNIRT 迭代终点

逐阶段 tensor A/B 是本轮最重要的诊断结果。它先固定 FSL warp，再依次比较 PPD、
tensor decomposition 和最终派生量，从而排除了“每个子算子都很准但最后 tensor 很差”
这一表面矛盾。

| 边界 | 指标 | 结果 | 解释 |
| --- | --- | ---: | --- |
| 固定 FSL warp 后的 support | Dice | `1.000000` | Python 与 FSL 处理完全相同的有效体素集合。 |
| 固定 FSL warp 后的 tensor PPD | relative L2 | `1.514e-6` | PPD 重采样与重定向不是最终主要误差源。 |
| 固定 FSL warp 后的 V1 | mean / P99 axial angle | `0.000133/0.001044°` | 主方向几乎重合。 |
| 同一 FSL tensor 的 V1 decomposition | mean / P99 / max | `2.15e-7/1.21e-6/1.21e-6°` | tensor decomposition 本身正确。 |
| 最终 Python/FSL field | mean / P99 / max vector error | `0.0130/0.0626/0.1828 mm` | 剩余差异主要由多轮 FNIRT 轨迹累积产生。 |
| 最终 warped moving FA | relative L2 | `0.00839` | 对 moving FA 做最终 warp 的结果；不是从最终 tensor 再分解得到的 FA。 |
| 最终 tensor 公共 support | relative L2 | `0.00631` | 当前最终 tensor 主指标。 |
| 最终 tensor-derived FA | relative L2；mean/P99/max absolute error | `0.02407`；`0.000895/0.00749/1.22177` | 整体误差小，但边界或特征值并列附近仍有稀有长尾。 |
| 最终 tensor support | Dice | `0.99962` | support 高度一致。 |
| 最终 V1 | mean / P99 / max axial angle | `0.343/3.104/89.757°` | max 来自少量低各向异性、边界或方向并列点，不能只用 max 代表主体分布。 |

本轮定位并修复了两处真实问题：

1. moving DTI-FA 平滑错误使用了 T1 的 `0.7 mm` voxel size，而不是 moving 自身的
   `1.25 mm`；
2. 六个并行 Hessian CSC block 共享会被 `eliminate_zeros()` 原地修改的结构数组，
   导致部分交叉块结构被后一个 block 继承。

修复后，初始 warped moving 与 FSL bitwise 相等；首次 8-worker LM candidate cost
为 `505.868724685`，1-worker 为 `505.868724680`，FSL 日志为 `505.869`。六个位移
Hessian 子块逐块对 1-worker 的相对误差为 `5.0e-17–1.42e-16`。测试也改为逐子块、
非立方 shape 检查，避免全局 Frobenius 指标再次掩盖交叉块错误。

这组结果支持“相同算法逻辑下达到很小最终误差”，但不支持 bitwise FNIRT 声明。
FNIRT 是带接受/拒绝的病态迭代优化，末位 gradient/Hessian 差异会改变后续轨迹；
报告最终输出误差比声称 FP64 理论 bitwise 更准确。

上述最新 HCP 结果的逐阶段 JSON 和 nonlinear QA 保留在 ignored `runs/` 本地工作区，
使用私有 fixture alias，不作为公开包内容分发。QA JSON 持久化的是内部 wall
`359.465 s`；外层 `/usr/bin/time` 的 `360.63 s` 与约 `26.1 GiB` peak RSS 记录在项目
台账，因此后二者是本地运行台账证据，而不是 QA JSON 字段。FSL 正式 1-thread harness
的 manifest 为 `status=failed`、`outputs=[]`；超时进程留下的输出文件可以用于诊断 A/B，
但不能冒充一份成功的 reference 运行记录或正式计时。

## 多数同边界阶段快于 FSL，但不存在一个统一的全流程倍数

下表只比较列中明确给出的边界。`FSL/Python` 大于 1 表示 Python 更快。

| 阶段与输入 | Python | FSL | FSL/Python | Peak RSS 与解释 |
| --- | ---: | ---: | ---: | --- |
| DTI fitting，HCP `145×174×145×108`，Python **16 workers** | `9.76 s` | `108.23 s` | `11.09×` | Python `767,656 KiB`，FSL `2,023,996 KiB`；不是默认 8-worker 结果。 |
| `nomoco`，HCP，兼容未压缩 `.nii` mmap，8 workers | `18.82 s` | `257.30 s` | `13.67×` | Python约 `1.69 GiB`，FSL约 `2.95 GiB`。 |
| `nomoco`，同一 HCP，`.nii.gz` 单次解码，8 workers | `52.92 s` | `257.30 s` | `4.86×` | Python约 `4.57 GiB`；必须保留压缩输入成本。 |
| b0 MCFLIRT 固定阶段，HCP 18 b0，8 workers | median `6.98 s` | `12.80 s` | `1.83×` | FSL 只接收已提取 b0；原始 gzip 读取时间不计入该比值。 |
| BET，HCP nodif，8 workers，JIT 预热 | 约 `0.96 s` | `1.59 s` | `1.66×` | 冷进程约 `2.12 s`，首次编译约 `4.8 s`。 |
| legacy，公开 `20³×14`，8 workers | median `2.75 s` | `6.766 s` | `2.46×` | Python约 `222 MiB`，FSL约 `139 MiB`；没有 HCP legacy 性能结论。 |
| FLIRT 6/12 DOF 矩阵估计，HCP，8 candidate workers | `11.66/18.48 s` | `61.97/85.37 s` | `5.31×/4.62×` | 只表示矩阵估计边界。 |
| 完整 affine T1 registration，HCP | `53.23 s` | `233.18 s` | `4.38×` | Python约 `2.96 GiB`；底层 runtime 未限为总计 8 线程，CPU达 `2351%`。 |
| GRE fieldmap，公开 `16×14×12`，8 workers | median `1.51 s` | `0.46 s` | `0.30×` | FSL 在极小输入约快 `3.3×`；不外推真实 GRE。 |
| TOPUP，公开 `16×14×12×2`，8 workers，JIT cache 已存在 | median `2.12 s` | `4.61 s` | `2.17×` | Python约 `225 MiB`，FSL约 `14 MiB`；首次空 JIT cache 约 `15 s`。 |
| EDDY，公开 `26×26×18×26`，seed 1，8 threads | `7.76 s` | `9.77 s` | `1.26×` | Python约 `232 MiB`，FSL约 `36 MiB`；首次 wheel JIT 为 `37.78 s`。 |
| FNIRT+PPD，旧公开 `24×23×22`，8 workers | `9.74 s` | `9.203 s` | `0.945×` | Python多写 valid mask、两个 Jacobian 和 QA；该结果应在最终代码重测。 |
| 修复后 FNIRT+PPD，HCP，8 workers | outer `360.63 s` | 正式 1-thread harness 超时 | — | Python peak RSS约 `26.1 GiB`；当前不得发布 HCP speedup。 |

详细证据分布在 [BENCHMARKS.md](BENCHMARKS.md) 与
[实施计划](FSL_SUBSET_REIMPLEMENTATION_PLAN.md)；两者在已记录处给出原始边界、
user/system CPU 或重复样本，并非每项同时具备这三类信息。

### 缓存能显著改善长流水线，但不能代替计算性能

公开 `20³×14` nomoco 到 `24×23×22` nonlinear 的安装 wheel DAG 给出。这里是最终
HCP FNIRT 修复前的历史 source/candidate-wheel 快照，只用于说明启动与缓存层级，不能
当作当前 `v0.2.0` 工作树的重新计时：

| 状态 | 墙钟 | Peak RSS | 合法解释 |
| --- | ---: | ---: | --- |
| source tree，空 Numba cache | `58.55 s` | `556,600 KiB` | 源树冷启动。 |
| installed wheel，空 Numba cache | `58.37 s` | `553,328 KiB` | 包含首次 kernel 编译。 |
| installed wheel，新进程、已有磁盘 kernel cache、新输出 | `20.58 s` | `317,664 KiB` | 跨被试可复用的已编译 kernel 路径。 |
| complete artifact-cache hit | `0.77 s` | `144,776 KiB` | 结构复核已有输出，不是重新计算。 |

40 个 NIfTI 数组和 affine 在 source/wheel、cold/warm 之间 bitwise 相等。Numba 的
机器码 cache 不进入跨平台 wheel；CPU、Python/Numba版本、dtype、维度或布局变化都可能
产生新 specialization。

## 当前热点已经从“Python 循环”转为大体积 FNIRT 与 PPD 数据流

修复后 HCP nonlinear 的内部 wall 为 `359.465 s`，外层 `/usr/bin/time` 为
`360.63 s`。两者口径不同但一致：

| 阶段 | 时间 | 占内部 wall | 后续含义 |
| --- | ---: | ---: | --- |
| FNIRT estimation | `186.64 s` | `51.9%` | Hessian 仍是重要热点，但必须以完整 level 和轨迹 A/B 验收。 |
| FNIRT output materialization + registered-FA warp/write | `25.11 s` | `7.0%` | 包含 coefficient、field、Jacobian 和 FA 的计算、物化与写出，不能解释成纯磁盘 I/O。 |
| tensor nonlinear pipeline（resample + PPD + decomposition/output） | `147.71 s` | `41.1%` | 其中 warp resampling 为 `117.08 s`；以该阶段内部 QA wall `147.4105 s` 为分母，占约 `79.4%`。 |

tensor decomposition 仅约 `8.69 s`，因此下一轮不应优先“手搓 eigendecomposition”。
真正值得 profile 的是大体积 warp sampling、临时坐标/矩阵数组、chunk 生命周期和输出
并行。任何优化仍必须保持 FSL PPD 每体素合同和固定 warp A/B `1.51e-6` 水平。

约 `26.1 GiB` 峰值 RSS 是当前最明显的工程风险。用户当前机器内存足够，但普通
8–16 线程设备未必有相同余量；在不改变算法的前提下降低临时数组同时也可能改善内存
带宽和墙钟。

其他阶段的热点判断也需要更新：

- HCP `nomoco` 内部 QA 中 BET 为 `1.752/18.012 s = 9.73%`，继续极限优化 BET 对
  全流程收益有限；
- 完整 affine 内部 QA 中，tensor/派生量后处理为 `29.046/52.403 s = 55.43%`，双
  FLIRT 为 `17.726/52.403 s = 33.83%`，
  因此“热点全部是 FLIRT”已经不准确；
- `.nii.gz` HCP `nomoco` 比直接 mmap 多约 `34.1 s`；这是压缩输入解码与临时 `.nii`
  物化路径的合计增量，尚未单独隔离纯 DEFLATE 解码时间；
- 极小 fieldmap 的主要劣势是 Python/Numba 启动与完整 11,349 次 schedule 固定成本，
  不应通过减少搜索或改停止条件解决。

## 性能收益来自等价执行优化，不是删减算法

当前保留的优化主要属于以下几类：

- 兼容未压缩 NIfTI 的验证后 mmap，以及 `.nii.gz` 只解码一次再供 workers 共享；
- 每个拟合 worker 固定一个 BLAS thread，避免进程与 BLAS 线程过度订阅；
- 对独立 volume、候选、tensor 分量、Hessian block 和输出做显式并行；
- 用 Numba 编译固定循环，但保留 FSL 的 z/y/x、列索引和 float32 舍入边界；
- 缓存只依赖几何/结构的 spline basis、稀疏结构、pull matrix 和正则项；
- 重叠相互独立的 6/12 DOF registration、tensor decomposition 与 gzip 写出；
- 保留 reference 后端、逐阶段 QA 和失败不回退，方便每次优化做同输入 A/B。

明确未采用的手段包括降分辨率、减少迭代、修改停止条件、合并类别、近似替代、
`-ffast-math`、隐式 autocast 和 silent fallback。性能结果因此可以解释为同算法工程
优化，而不是降低计算要求。

## 当前 v0.2.0 候选已通过本地与远端发布门禁

2026-08-24 在最终生产代码树上完成的本地证据为：

- 精确 SimNIBS 4.6 环境的低核完整单元测试为 `530 passed, 7 skipped`；TOPUP、EDDY、
  FNIRT/nonlinear 三个真实合成 E2E 均完成；
- 完整运行与三个真实合成 E2E 合并后为 `12443/12443 statements`、100%；运行时固定
  `NUMBA_NUM_THREADS=3`，同时验证请求 8 workers 在低核设备上会安全收敛到可用槽位；
- 最后 focused 回归为 `92 passed, 2 skipped`，用于关闭进度条、nonlinear sparse
  Hessian/JtJ 和 TOPUP optimizer 的剩余分支；
- Ruff、compileall、版本一致性、Markdown 链接、10 份 reference manifest、Git staged
  diff 和 tracked-file 隐私审计均通过；
- 精确 SimNIBS 4.6 环境版本合同全部匹配；候选 wheel 在临时 overlay 中完成
  `--no-deps` 安装、版本读取、CLI help 和 `pip check`；
- wheel、sdist、CycloneDX SBOM 与 `SHA256SUMS` 已重建；`validate-pyproject`、
  `check-manifest`、Twine、wheel contents、release archive 隐私审计及 `pip-audit`
  全部通过。manifest 审计曾发现 10 个 JSON 未进入 sdist，补充规则并重建后已关闭。

冻结提交 `90a0553` 的 CI run `32683104733` 已在 macOS 14 arm64、Ubuntu 22.04、
Windows Server 2022 与 package job 全绿；CodeQL `32683104727` 和 OpenSSF
`32683104735` 同样完成。source/wheel 与 wheel cold/warm 的 40 个 NIfTI 数组和
affine bitwise 相等仍属于此前
已记录的候选快照，用于证明安装形式不改变数值；它没有被包装为本轮重新计时结果。

## 当前是已封板的 v0.2.0 候选，进入 tag/Release 流程

公开仓库已有 `v0.1.0`。当前 FSL 子集实现已整理为 `v0.2.0` staged candidate，
最终提交与远端 CI 已完成；在 tag Release workflow 和资产回下载审计完成前仍称为
“已封板候选”。版本策略固定为：
`main` 保持最新代码，稳定版本由不可变 tag/Release 保留；只有需要维护旧版补丁时才
创建临时 backport 分支，不为 `0.1`、`0.2` 长期各维护一条开发分支。

当前发布门禁状态如下：

| 门禁 | 当前状态 | 封板动作 |
| --- | --- | --- |
| P0–P10 算法与阶段 A/B | 通过 | 保留最终聚合指标和失败边界。 |
| 最终工作树 full suite + 100% coverage | 通过 | 本地低核 `530 passed, 7 skipped`；远端 Linux `535 passed, 7 skipped`，`12443/12443`。 |
| 修复后固定 8-worker 完整真实 subject 与 SimNIBS 合同 | 通过 | 完整四级 nonlinear、逐阶段 FSL A/B、最终 tensor/QA、四模式 FEM 与隔离 wheel overlay 均已验证。 |
| HCP FSL nonlinear 成功 manifest | 性能声明门禁 | 仅在发布 HCP nonlinear 加速比前，提高 timeout 后正式重跑并登记 outputs 与完整计时。 |
| 最终冻结并提交后的 Linux/macOS/Windows green | 通过 | GitHub Actions run `32683104733` 四个 job 全绿。 |
| `v0.2.0` 版本、Changelog、README 与依赖声明 | 本地通过 | 已统一版本、能力、测试、FSL reference 与性能边界。 |
| 最终 wheel/sdist/SBOM/SHA256 与隔离安装 | 本地通过 | 最终 commit 后由 Release workflow 再生成正式资产。 |
| staged/tracked 文件隐私与 provenance 审计 | 本地通过 | 未分发私有 MRI、FSL 源码/二进制或 subject derivative。 |

当前适合称为“算法、科学 A/B、跨平台和发布产物门禁已完成，正在创建 GitHub Release
的 `v0.2.0` 封板候选”。现有 workflow 只创建 GitHub Release，没有 PyPI publish；
若计划发布到 PyPI，还需另行增加并验证发布步骤。

## 已知限制决定了公开声明的边界

下列限制会实质影响读者如何解释结果：

- 性能数据来自一台服务器；部分历史记录没有冻结精确 CPU 型号。
- DTI `11.09×` 使用 16 workers；affine 虽请求 8 candidate workers，但底层 runtime
  未限线程并达到 `2351%` CPU。两者不能包装成“全项目固定八总线程”。
- HCP evidence 主要来自一个 subject；需要额外被试验证泛化和边界稳定性。
- public tiny fixture、HCP、冷 JIT、热 kernel 和 artifact cache 数字不能相加或互换。
- legacy 的尖锐无噪声 fixture 在边界处会使 tensor/SSE relative L2 病态；报告同时
  保留 whole common-mask tensor/FA/SSE relative L2
  `0.772510/0.0373037/1.06086`，以及三次 erosion 后 tensor/FA relative L2
  `0.00287432/0.00248977`，不选择性隐藏不利数字。
- TOPUP、EDDY 和 FNIRT 是 SimNIBS 4.6 固定调用子集，不是通用命令兼容层。
- nonlinear 最终结果非常接近，但不是 bitwise FSL；迭代路径差异必须作为已知边界。
- 当前没有 GPU 运行路径或 CUDA 性能声明。
- FEM 与 lead field 仍依赖 SimNIBS 4.6；被替代的是 dwi2cond 的 FSL 运行依赖。
- 发布仓库不能包含私有 MRI、FSL 源码/二进制、subject derivative 或绝对本地路径。

## v0.2.0 只剩 tag 资产复核，性能工作进入 v0.3.0

建议按以下顺序推进：

1. **创建 `v0.2.0` tag。** P11、跨平台与安全门禁已关闭。
2. **复核 Release 资产。** 等 tag Release workflow 成功后重新下载 wheel、
   sdist、SBOM 和校验和，复核 tag、资产哈希、provenance 与 latest 状态。
3. **在 `v0.3.0` 再继续性能优化。** 优先 profile PPD warp resampling、FNIRT完整
   estimation 和 26.1 GiB临时数组；每次只改一个热点，并用固定 warp、轨迹和最终
   tensor A/B 验收。

HCP nonlinear 的正式 FSL 成功计时只约束是否发布该阶段加速比，不阻塞 v0.2.0。
新的性能优化明确归入 `v0.3.0`，不再推迟已经完成本地门禁的 v0.2.0 功能封板。

## 仍需回答的关键问题

- 能否在不改变每体素 PPD 顺序的情况下，对坐标、Jacobian 和 tensor 临时数组分块，
  同时降低 26.1 GiB RSS 和内存带宽成本？
- 修复后的 current-code 小型 nonlinear 基准是否仍是 Python 比 FSL 慢 `5.8%`？
- 在 8–16 线程普通工作站而不是多核服务器上，nomoco、affine、TOPUP 和 EDDY 的实际
  性能排序是否保持？
- 真实解剖 GRE fieldmap 输入上，Python 的完整 11,349-evaluation FLIRT 固定开销是否
  仍会被更大体积摊薄？
- 第二个和第三个真实 subject 是否会维持当前私有 HCP fixture 的 mask、field、tensor 和 V1
  误差分布？
- Windows/macOS 上的线程后端是否与 Linux 多进程路径保持相同输出和可接受性能？

在这些问题中，前两个直接决定下一轮性能工作；其余问题决定 `v0.2.0` 发布声明可以
覆盖多广。
