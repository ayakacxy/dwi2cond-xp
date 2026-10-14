# v0.4.0 发布说明

[English](../en/RELEASE_NOTES_v0.4.0.md) | [简体中文](../zh/RELEASE_NOTES_v0.4.0.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md) · [更新记录](CHANGELOG.md)

发布日期：2026-10-14。

v0.4.0 修复梯度、张量分量和物理坐标处理，再加速原始准备、T1 配准、FNIRT、PPD
及 QA。默认工作流仍可完全使用 CPU；可选 CUDA 后端加速 FNIRT Hessian 组装和 PPD，
运行时无需 FSL。

## 算法修复

- **梯度与存储轴。** legacy 在最终运动变换所属的 FSL 基底中旋转梯度。存储轴重排
  同步转换 b 向量、六分量张量和列主序梯度偏差矩阵，保持它们的物理方向。
- **GRE、TOPUP 和 EDDY 手性。** 正行列式存储的影像一致地进入所需的内部放射学
  坐标；输出恢复图像存储坐标，避免对运动/涡流参数或 FSL b 向量重复反射。
- **NIfTI 采样尺度。** 仿射和稠密位移的 FSL 采样独立使用 header `pixdim`，与世界
  affine 区分。不能同时表达 header/affine 体素尺度的 raw/FNIRT 路径明确拒绝
  尺度不一致，不猜测两者的关系。
- **物理 PPD。** 张量重定向与折叠/近奇异 QA 使用同一物理 Jacobian，包含目标网格
  的实际体素尺寸。此前按体素求导的方式可能在非 1 mm 网格上产生错误旋转。
- **电导率方向。** 物理基底分别去除 affine 每一列的尺度，并检查正交性，避免倾斜
  非等距网格引入假各向异性；剪切被明确拒绝。正式 SimNIBS FEM 和先导场实例通过
  项目内适配使用所选基底与稳定对称特征分解，不修改 SimNIBS。
- **QA 与发布。** 已准备的外部 Hz 场图在工作网格上检查；最终 QA 完成后才向 m2m
  发布张量及来源记录，失败的尝试保留此前已通过检查的输出。

这些变化既包括本项目的实现缺陷，也包括对继承坐标约定的有意修正。旧行为对照选项
仍然保留；复现旧上游输出本身，并不足以证明它在所有网格上的物理方向都正确。

## 性能优化

原始准备对满足条件的规范 float32 gzip 输入流式解压，复用不变的计算，减少体积与
坐标临时数组。WLS 避免重建共享正规矩阵，仍保留进程 worker。原始 QA 拟合和正式
非负阈值处理后的拟合保持独立。

T1 主仿射和独立刚性 QA 配准共享只读图像金字塔；细层相关比使用自适应 FP64 分块
统计，搜索、粗层采样、迭代上限和停止规则保持原有逻辑。

FNIRT 的多分量和全局投影共享样条支持域遍历，省去未消费的稠密基底及仿射场临时数组；
LM 工作矩阵使用 CSR，保留原 CPU PCG 求解器。physical PPD 不再计算未使用的解析
样条 Jacobian，保留完整位移及其有限差分 Jacobian。PPD 和张量分解采用有界分块。
QA 和写出减少重复读取与全体积临时数组，保留内容哈希、全部值的有限性检查和科学 QA。

CUDA 需要显式选择。FNIRT Hessian 组装使用 FP64 累加；PPD 将源张量保留在 GPU，
按块执行采样、Jacobian、特征分解和旋转。原有 float32 采样及系数检查点保留，
Jacobian 和矩阵运算使用 FP64；LM 和 PCG 仍在 CPU。浮点规约允许有界差异，以
算法逻辑与数值稳定性作为验收依据。

## 从 v0.3.0 迁移

| 设置 | v0.4.0 默认值 | 对照选项 |
| --- | --- | --- |
| legacy 运动变换的梯度旋转 | `--bvec-mode corrected` | `--bvec-mode compat46` |
| 存储轴重排时的分量变换 | `--component-mode physical` | `--component-mode simnibs46-literal` |
| PPD 重定向的 Jacobian | `--jacobian-mode physical` | `--jacobian-mode simnibs46-literal` |
| 张量到世界坐标的基底 | `--basis-mode physical` | `--basis-mode simnibs46-literal` |
| 电导率特征分解 | `--eigensystem-mode stable` | `--eigensystem-mode simnibs46-literal` |
| FNIRT Hessian 设备 | `--fnirt-device cpu` | `--fnirt-device cuda` |
| PPD 设备 | `--ppd-device cpu` | `--ppd-device cuda` |

受影响的输出会有意不同于 v0.3.0。稳定对称电导率特征分解在 v0.3.0 已是默认值；
v0.4.0 将所选模式和物理基底一并应用到正式 SimNIBS 消费入口。

`compat46` 只关闭运动造成的梯度旋转。使用 physical 分量模式时，存储轴变化仍会
变换梯度；原 b 向量的逐字节复制需要 literal 分量模式或恒等轴变换。即使选择 literal
PPD，折叠/近奇异 QA 仍使用物理 Jacobian。四个 literal
模式相互独立，不会关闭 GRE/TOPUP/EDDY 和发布修复。需要上游坐标行为对照时，可
显式选择全部五项：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_comparison \
  --bvec-mode compat46 \
  --component-mode simnibs46-literal \
  --jacobian-mode simnibs46-literal \
  --basis-mode simnibs46-literal \
  --eigensystem-mode simnibs46-literal --workers 8
```

`strict-fsl`/`robust` 控制拟合与无效位置处理，与这些方向模式独立。相位编码的
`x/y/z` 指存储重排后的 FSL 内部轴，不会自动转换原 NIfTI/BIDS 的 `i/j/k`。
缓存身份包含已更改的源码及所选模式，旧阶段结果不会作为新版计算复用，应重新计算
受影响的输出。参见[输入文件与坐标](INPUTS.md)及[计算方法](METHODS.md)。

## CPU 与可选 GPU 用法

完整工作流已在 Python 3.11、SimNIBS 4.6.0 环境运行。对已验证并冻结依赖的现有
SimNIBS 环境，可从本地工作树安装：

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps -e .
dwi2cond-xp --help
```

发布产物可用后，wheel 安装命令为
`python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl`。
`--no-deps` 假定已经安装经过验证的依赖。单独创建环境时，请按
[安装说明](../../README.zh-CN.md#-安装)及固定版本的[环境文件](../../environment.yml)操作。

默认 legacy + nonlinear CPU 工作流：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_cpu \
  --fnirt-device cpu --ppd-device cpu --workers 8
```

同一工作流启用两个 CUDA 后端：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_gpu \
  --fnirt-device cuda --ppd-device cuda --workers 8
```

CPU 不需要 GPU 或 CuPy。FNIRT CUDA 后端需要可用的 Numba CUDA 运行环境、NVIDIA
驱动及兼容的 CUDA toolkit；PPD 还需要兼容的 CuPy。实测 GPU 环境使用 CuPy 14.2.0
和 CUDA 12.8。基础软件包不会安装可选 GPU 依赖；CUDA 失败会终止运行，不静默切换
为 CPU。这两个设备选项也适用于 `register-t1-nonlinear` 和 `run-prefit-pipeline`。
派生图和 NIfTI 写出仍在 CPU。

`nomoco` 不校正运动、涡流或磁敏感畸变，只应在采集条件适用时使用。实测 HCP 流程
使用 `--preprocessing-mode nomoco --t1-mode nonlinear --grad-dev grad_dev.nii.gz`，
保留两次拟合、全部项目 QA 和 m2m 发布，FEM 关闭；计时对应这一流程，不是上面的
默认 legacy 路径。按采集选择预处理的说明见[命令示例](USAGE.md)。

## 完整流程性能实测

同一私有 HCP 单壳输入在 Linux 上运行，主机为 Intel Xeon Gold 6230，使用八个
物理核心，GPU 为 NVIDIA RTX 2080 Ti。DWI 为 `145×174×145×108`、1.25 mm，
包括 18 个精确 b=0 与 90 个 b≈1000 体积；梯度非线性数据相同，T1 seed 相同，
为 `260×311×260`、0.7 mm。

| 实现 | 完整耗时 | 峰值主机进程树 RSS | 峰值 GPU 分配 |
| --- | ---: | ---: | ---: |
| v0.4.0 CPU | 205.452 s | 6.498 GiB | 不使用 |
| v0.4.0 GPU | 163.342 s | 6.723 GiB | 8.834 GiB |
| 公开 v0.3.0 CPU | 564.406 s | 25.469 GiB | 不使用 |
| SimNIBS 4.6 / FSL 6.0.4 官方子脚本 | 1290.073 s | 3.794 GiB | 不使用 |

相对公开 v0.3.0，CPU/GPU 观察耗时比分别为 **2.747× / 3.455×**，时间减少
63.599% / 71.060%，主机 RSS 减少 74.487% / 73.601%。相对 FSL，观察耗时比为
**6.279× / 7.898×**，但 FSL 使用的主机内存更少；当前 CPU/GPU 的主机 RSS 为
FSL 的 1.713 / 1.772 倍，GPU 显存另计。

各 Python 结果均为一次独立进程、全新输出目录的热缓存样本，无 JIT 缓存写入及计算
阶段产物缓存命中；FSL 是一次已完成的原生运行。计时包含启动、导入、输入哈希、设备
初始化、计算、读写、QA 和发布，seed 复制及后续核验在计时外；未清除操作系统缓存。
主机 RSS 按进程树采样，可能重复计入共享页；GPU 分配包含上下文和分配器缓存。这些
是采样峰值，不是最低内存需求。

FSL 使用官方子脚本，两次 WLS 均传入原生 `--gradnonlin`，跳过可视化网格导出和 FEM，
写出的项目诊断较少；在相同八核预算内，原生 FSL 经常只使用一个核心。跨版本耗时比
还包含算法修复，因此不证明纯硬件加速或最终 FSL 数值等价。[基准记录](BENCHMARKS.md)
保留阶段耗时、留档发布版身份和历史测量。

## 发布验证

本地 **955 个独立测试通过，无跳过**（主批次 942 + 独立回归 1 + 电极配置 12），
已启用配置的真实 FSL/SimNIBS 探针。Numba-aware 全包门槛覆盖
**15,363/15,363 条语句（100.00%）**，并运行真实合成 TOPUP/EDDY/FNIRT 三条 CLI。
主批次有 12 条与测试输入有关的 SimNIBS 警告。无设备 CUDA 测试与独立真实 GPU
证据各有作用；语句覆盖率不代表完整分支覆盖或任意输入正确性。
[验证记录](VALIDATION.md)说明各批次与适用范围。

## 数值检查与适用范围

完整 CPU 流程在禁止 `cupy`、`cupyx` 和可选 GPU 后端导入时成功完成。CPU/GPU 的
64 份数值产物中，61 份逐字节相同；几何、掩膜、FNIRT 系数/位移场及 LM 记录一致。
脑内张量相对 L2 为 `1.8693e-9`，最终 FA 最大绝对差为 `8.7172e-7`，健康谱 V1
轴向夹角 P99 为 `1.2074e-6` 度。这一方向统计要求双方 FA>0.1、相对主特征值间隔
>1e-5，忽略特征向量符号；本输入健康体素覆盖完整。

算法回归覆盖带符号存储轴置换、两种手性、非 1 mm PPD、header/affine 尺度不一致、
正式电导率消费入口与发布失败；阶段检查包括真实 FSL 消费。无半正定约束的 WLS
仍可能产生负特征值和 FA>1，有限输出并不能证明每个张量半正定。

以上证据限于所述 Linux 输入和处理范围。本次没有新增完整真实被试 raw-DWI→四模式
FEM、其他采集研究、其他平台 GPU 认证或冷启动性能保证。固定预处理子集、单壳模型，
以及不支持缠绕相位/PRELUDE 和任意 FSL 配置的范围，仍以[计算方法](METHODS.md)及
[FSL 参考比较](FSL_REFERENCE.md)为准。
