<div align="center">

# dwi2cond-xp

面向 SimNIBS 4.6 的跨平台、运行时无 FSL 的 DTI→电导率流程。

[![Release](https://img.shields.io/github/v/release/ayakacxy/dwi2cond-xp?display_name=tag&sort=semver)](https://github.com/ayakacxy/dwi2cond-xp/releases/latest)
[![CI](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/ayakacxy/dwi2cond-xp/badge)](https://scorecard.dev/viewer/?uri=github.com/ayakacxy/dwi2cond-xp)
[![v0.4.0 coverage: 100%](https://img.shields.io/badge/v0.4.0_coverage-100%25-brightgreen.svg)](#-验证概览)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF.svg)](docs/zh/SIMNIBS_INTEGRATION.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md) · [文档](docs/zh/README.md) · [验证](docs/zh/VALIDATION.md) · [路线图](docs/zh/ROADMAP.md) · [更新记录](docs/zh/CHANGELOG.md)

🧠 **DTI 拟合** · ⚡ **运行时无 FSL** · 🧭 **张量重定向** · ⚡ **各向异性 FEM** · 🖥️ **CPU / 可选 CUDA**

</div>

## 📚 目录

- [验证概览](#-验证概览)
- [支持范围](#-支持范围)
- [电导率模式](#-电导率模式)
- [安装](#-安装)
- [输入文件与坐标](#-输入文件与坐标)
- [最小流程](#-最小流程)
- [四模式图](#-四模式图)
- [先导场支持](#-先导场支持)
- [验证结果](#-验证结果)
- [后续路线图](#-后续路线图)
- [文档与社区](#-文档与社区)
- [引用与许可证](#-引用与许可证)

`dwi2cond-xp` 是一个跨平台 Python 流程：对支持的原始或已预处理单壳扩散 MRI
执行预处理，生成 SimNIBS 4.6 电导率张量，并组织各向异性有限元仿真。运行时
不依赖 FSL。

这是独立社区项目，不是 SimNIBS 或 FSL 官方发行版。中英文文档提供相同的安装、
使用和验证信息，可通过页首链接切换。

`v0.1.0`–`v0.3.0` 三个标签于 2026-08-29 因文档和发布元数据修正而重新发布；
第四次独立源码审计发现正确性缺陷后，`v0.3.0` 又于 2026-08-30 被同版本修正版替换。
替换后的算法基线和验证范围见更新记录。

`v0.4.0` 修复了梯度旋转、存储轴重排时的向量/张量分量转换、PPD 的物理导数，以及
倾斜网格上的张量基底。同时减少重复计算和大数组开销，加入可选 FP64 CUDA 加速。
CPU 是默认路径，无 GPU 或 CuPy 也能运行。发布日期为 2026-10-14。
具体变化和旧坐标行为的迁移方式见
[v0.4.0 发布说明](docs/zh/RELEASE_NOTES_v0.4.0.md)。

## ⚡ 验证概览

`v0.4.0` 实现已与公开 `v0.3.0` wheel 和官方 FSL 流程完成同输入的完整流程对照：

| 实现 | 完整耗时 | 主机峰值 RSS | GPU 分配峰值 |
| --- | ---: | ---: | ---: |
| v0.4.0 CPU | **205.45 秒** | **6.50 GiB** | 无 |
| v0.4.0 GPU | **163.34 秒** | **6.72 GiB** | **8.83 GiB** |
| 已发布 v0.3.0 CPU | 564.41 秒 | 25.47 GiB | 无 |
| SimNIBS 4.6 / FSL 6.0.4 子脚本 | 1290.07 秒 | 3.79 GiB | 无 |

此次测量中，CPU/GPU 相对 v0.3.0 **加速 2.75 倍 / 3.46 倍**。使用同一 HCP 单壳输入，
含梯度非线性校正、`nomoco` 预处理及非线性配准，Linux、八个 CPU 核心和 RTX 2080 Ti，
文件/JIT 缓存预热，包含全部项目 QA 和 m2m 张量发布，FEM 关闭。此次没有计时默认
legacy 预处理路径。FSL 的输出范围和核心使用方式有所不同。这些是实测值，不是最低
内存需求或通用速度保证；计时范围和数值对照见[性能测试](docs/zh/BENCHMARKS.md)。

v0.4.0 的本地发布检查于 2026-10-14 完成：

| 检查项目 | 结果 | 测试条件 |
| --- | ---: | --- |
| v0.4.0 正确性检查 | **955 passed · 0 skipped** | Linux 主批次 942 + 独立回归 1 + 电极配置 12 项，启用已配置的真实 FSL/SimNIBS 探针 |
| 全包覆盖率门槛 | **100.00% 语句覆盖率** | 15,363/15,363 条语句；Numba 独立冷缓存和 JIT coverage，真实合成 TOPUP/EDDY/FNIRT CLI |

语句覆盖率不代表完整分支覆盖或任意输入正确性。CUDA 数学检查与先前真实 GPU
完整流程证据分别记录于[验证](docs/zh/VALIDATION.md)。

以下历史检查对应已发布的 `v0.3.0` 标签；其测试数和覆盖率不是 v0.4.0 的新测量。

| 检查项目 | 结果 | 测试条件 |
| --- | ---: | --- |
| SimNIBS 4.6 预处理子集 | **纯 Python · 运行时无需 FSL** | `nomoco`、旧版校正、固定 GRE/TOPUP/EDDY、线性/FNIRT 配准和 PPD 张量重定向 |
| 最终标签正确性检查 | **666 passed · 0 skipped** | 固定依赖版本，并启用全部已配置的真实 FSL 探针 |
| 独立覆盖率检查 | **100.00% 语句覆盖率** | 主批次 666 + 电极配置批次 12 项测试、真实合成 TOPUP/EDDY/FNIRT 命令，以及 13,663/13,663 条可执行语句 |
| 最终标签 SimNIBS 准备 | **`SESSION._prepare()` 完成** | 真实 SimNIBS 4.6 使用显式指定的网格完成准备；最终修复后未重跑 FEM 求解器 |

仓库不分发解剖影像、被试标识、体数据派生物或被试级机器可读产物。方法及验证细节
见[验证](docs/zh/VALIDATION.md)和[性能测试](docs/zh/BENCHMARKS.md)。

## 🧭 支持范围

预处理与电导率流程为：

```text
原始或已预处理单壳 DWI，或六分量扩散张量
  -> 选择 SimNIBS 4.6 预处理分支并执行两遍 WLS DTI 拟合
  -> 质量检查、输出记录和缓存验证
  -> 显式映射并重定向至 CHARM T1 网格
  -> scalar / vn / dir / mc 电导率
  -> SimNIBS 4.6 固定电极配置 FEM
  -> 向量电场 NIfTI 与质量检查报告
```

`v0.3.0` 已实现 SimNIBS 4.6 旧版运动/涡流校正、接受已换算 rad/s 场图的固定
GRE/FUGUE 分支、固定 TOPUP 分支，以及支持可选 TOPUP 场的单壳 EDDY `--repol` 子集。
同时支持自动 6/12 自由度线性 DTI→T1 配准，以及 SimNIBS 4.6 固定 FNIRT 与非线性 PPD
张量重定向。流程包含质量检查、原子写入的步骤记录、缓存验证和进度显示，并修复了
v0.2.0 审计发现的数据流向、掩膜、拟合行为、TOPUP→EDDY 流程、官方默认值、预拟合
张量导入和 m2m 输出写入问题。FSL 仅作为可选的本机数值参考，不会被发布版本的运行
流程调用。
最终 v0.3.0 修复还处理了方向判定、依赖、缓存、CHARM/FEM 集成及奇异 VN 张量问题。
这些检查不代表所有 FSL 优化器都能逐位一致，也没有重新完成真实被试与官方流程的
完整对照。具体数值结果与限制见
[v0.3.0 修复报告](docs/zh/V0.3.0_FOURTH_FRESH_INDEPENDENT_ALGORITHM_REAUDIT_REMEDIATION_2026-08-30.md)。

纯 Python DTI 与张量映射部分依赖 NumPy、SciPy、NiBabel、h5py 和 tqdm；已验证的
非线性运行环境固定为 NumPy 2.3.0 与 Numba 0.64.0。网格电导率、FEM 与先导场
固定要求 SimNIBS 4.6.0；这些功能仅适用于能够安装并验证该 SimNIBS 版本的平台。

## 🧩 电导率模式

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每种组织使用固定标量电导率，不使用 DTI。 |
| `vn` | 保留主方向和各向异性比例，并按组织参考电导率逐位置归一化行列式；这是主要的各向异性模式。 |
| `dir` | 保留方向、比例及局部强度变化，再进行总体强度校准。 |
| `mc` | 保留 DTI 驱动的空间平均电导率变化，但局部各向同性；它是强度变化对照。 |

### 实现公式

下面是本项目实现并与 SimNIBS 4.6 对照的电导率映射公式。对于各向异性组织 $t$
中的单元或体素 $i$，把修复后的扩散张量写成

$$
\mathbf D_i = \mathbf V_i\mathrm{diag}
(d_{i1},d_{i2},d_{i3})\mathbf V_i^{\mathsf T},
\qquad d_{i1}\ge d_{i2}\ge d_{i3}>0,
$$

其中 $\mathbf V_i$ 的列是主方向，$\sigma_t$ 是组织 $t$ 的参考标量电导率，$w_i$
是四面体体积；在不加权的体素计算中取 $w_i=1$。

#### `vn`：体积归一化各向异性映射

定义局部几何均值 $g_i=(d_{i1}d_{i2}d_{i3})^{1/3}$，核心映射为

$$
\boldsymbol\Sigma_i^{\mathrm{vn}}
=\sigma_t\mathbf V_i
\mathrm{diag}\left(
\frac{d_{i1}}{g_i},\frac{d_{i2}}{g_i},\frac{d_{i3}}{g_i}
\right)\mathbf V_i^{\mathsf T},
\qquad
\det\left(\boldsymbol\Sigma_i^{\mathrm{vn}}\right)^{1/3}=\sigma_t.
$$

因此，`vn` 保留特征向量和相对各向异性，同时把每个位置的几何平均电导率设为对应
组织的参考值。这是 Güllmar 等提出并被 SimNIBS 推荐使用的体积归一化映射 [3,4]。

#### `dir`：直接缩放各向异性映射

先对每种各向异性组织计算体积加权张量尺度

$$
m_t=
\left(
\frac{\sum_{i\in t}w_i\det(\mathbf D_i)}
     {\sum_{i\in t}w_i}
\right)^{1/3}.
$$

然后在全部各向异性组织之间联合拟合一个全局缩放因子：

$$
s=\underset{a}{\mathrm{argmin}}
\sum_t(am_t-\sigma_t)^2
=\frac{\sum_t\sigma_t m_t}{\sum_t m_t^2},
\qquad
\boldsymbol\Sigma_i^{\mathrm{dir}}=s\mathbf D_i.
$$

该映射保留 DTI 给出的方向、各向异性和空间强度变化。它属于 Tuch 等提出的线性
扩散率到电导率映射，并对应 Rullmann 等和 Opitz 等使用的直接映射 [1,2,4]。

#### `mc`：平均电导率对照

`mc` 使用与 `dir` 相同的全局因子 $s$，但把每个局部张量替换为具有相同行列式的
各向同性张量：

$$
\boldsymbol\Sigma_i^{\mathrm{mc}}
=\det\left(\boldsymbol\Sigma_i^{\mathrm{dir}}\right)^{1/3}\mathbf I
=s\det(\mathbf D_i)^{1/3}\mathbf I.
$$

所以它保留 DTI 驱动的几何平均电导率空间变化，却移除了方向各向异性。`mc` 是
DTI 派生的对照模式，而不是各向异性张量场 [4]。

三种映射均对支持的输入应用与 SimNIBS 兼容的特征值、最大电导率及各向异性比限制。
退化输入按各模式的规则处理：非零秩亏 `vn` 张量默认报错；
默认 `correct_intensity=True` 时，全零各向异性 `dir/mc` 组织因全局尺度无定义而被拒绝；
显式指定 `--no-correct-intensity` 时改走未校准的保护处理，并接受该输入。`vn` 实际执行
“归一化→保护规则修正→再次归一化→再次保护规则修正”；如果最后一次修正触及保护阈值，上式中
理想的行列式等式可能出现轻微偏移。不参与各向异性的组织统一使用
$\boldsymbol\Sigma_i=\sigma_t\mathbf I$。

非零秩亏张量无法进行 VN 行列式归一化，因此默认明确报错，避免输出 NaN 或
静默替换缩放系数。`tensor-to-mesh --vn-singular-policy regularize` 是显式的
数值稳定扩展：先把奇异特征系统投影到配置的最大各向异性比，再执行既有 SimNIBS
保护规则修正，并在 QA 中记录修复单元数。这是用户显式选择的数值扩展，与 SimNIBS 4.6 原有的秩亏处理不同。

转换命令、正式 FEM 和先导场默认使用 `--eigensystem-mode stable`，即对称 `eigh` 求解器；
重复特征值张量仍保持正交、满秩基底。显式指定
`--eigensystem-mode simnibs46-literal` 时，使用 SimNIBS 4.6 的通用 `eig`
排序方式，包括重复特征值下可能产生非正交基底的情况；QA JSON 会记录模式和基底
正交性诊断。正式 FEM 流程仍调用已安装的 SimNIBS SESSION，通过所选实例上的项目内
电导率适配接入，不修改安装模块。默认 `--basis-mode physical` 分别归一化 affine
空间列，支持倾斜非等距体素，并拒绝剪切；显式 `simnibs46-literal` 保留上游归一化
方式供对照。

参考文献：

1. Tuch DS, Wedeen VJ, Dale AM, George JS, Belliveau JW. *Conductivity tensor
   mapping of the human brain using diffusion tensor MRI*. PNAS. 2001;
   98(20):11697-11701. [doi:10.1073/pnas.171473898](https://doi.org/10.1073/pnas.171473898)
2. Rullmann M, Anwander A, Dannhauer M, Warfield SK, Duffy FH, Wolters CH.
   *EEG source analysis of epileptiform activity using a 1 mm anisotropic
   hexahedra finite element head model*. NeuroImage. 2009;44(2):399-410.
   [doi:10.1016/j.neuroimage.2008.09.009](https://doi.org/10.1016/j.neuroimage.2008.09.009)
3. Güllmar D, Haueisen J, Reichenbach JR. *Influence of anisotropic electrical
   conductivity in white matter tissue on the EEG/MEG forward and inverse
   solution. A high-resolution whole head simulation study*. NeuroImage.
   2010;51(1):145-163.
   [doi:10.1016/j.neuroimage.2010.02.014](https://doi.org/10.1016/j.neuroimage.2010.02.014)
4. Opitz A, Windhoff M, Heidemann RM, Turner R, Thielscher A. *How the brain
   tissue shapes the electric field induced by transcranial magnetic
   stimulation*. NeuroImage. 2011;58(3):849-859.
   [doi:10.1016/j.neuroimage.2011.06.069](https://doi.org/10.1016/j.neuroimage.2011.06.069)

对应的 SimNIBS 实现定义见官方
[dwi2cond 文档](https://simnibs.github.io/simnibs/build/html/documentation/command_line/dwi2cond.html)。

## 🐍 安装

完整流程已在 Python 3.11 与 SimNIBS 4.6.0 环境中验证。推荐安装到现有 SimNIBS 4.6
环境；若需保持其依赖不变，可使用 `--no-deps` 安装 wheel：

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl
dwi2cond-xp --help
```

从 [GitHub Release](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.4.0) 下载上述 v0.4.0 wheel。
从当前源码开发时，将 wheel 安装命令替换为 `python -m pip install --no-deps -e .`。
发布 wheel 曾以临时附加安装的方式与参考环境中的 SimNIBS 4.6.0 一起导入验证，
该验证没有修改参考环境。

也可以另建独立环境：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
dwi2cond-xp --help
```

仅安装纯 Python 核心：

```bash
python -m venv .venv
. .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
dwi2cond-xp --help
```

允许更新依赖的开发环境可执行：

```bash
python -m pip install -e '.[test,viz]'
ruff check src tests tools scripts
pytest -q tests/test_cli.py
python scripts/check_markdown_links.py
```

开发过程中运行与修改相关的测试。完整发布检查使用 `python scripts/run_coverage.py`；
该脚本在隔离的 Numba 缓存中执行单元测试及真实合成 TOPUP/EDDY/FNIRT 端到端命令，
要求 100% 语句覆盖率。

## 📥 输入文件与坐标

`fit-dti` 接受已预处理的四维 NIfTI，以及匹配的 b 值、b 向量和扩散脑掩膜。
`preprocess-nomoco` 接受原始单壳 DWI，并生成 b0 参考与脑掩膜，但不对 DWI 体积
执行运动、涡流或磁敏感性校正；只有在这种 SimNIBS 4.6 `nomoco` 处理方式适合数据时
才使用。DTI 模型使用 b=0 加一个非零壳层，不会静默合并全部壳层拟合。可选的梯度
非线性系数 `grad_dev` 使用 FSL/HCP 九分量排列。

`preprocess-legacy` 按 SimNIBS 4.6 的顺序执行两遍 6/12 自由度校正，每个体积仅做
一次最终 sinc 重采样。默认 `corrected` 在 FSL 基底中按最终变换旋转 b 向量；显式
`compat46` 不做运动旋转，但仍应用 physical 存储基底变换；只有分量模式为 literal
或存储轴未改变时才复制原文件。

存储轴重排默认 `--component-mode physical`，随图像轴转换 b 向量、张量和
`grad_dev` 分量；非线性 PPD 默认 `--jacobian-mode physical`，张量旋转与折叠 QA
共用按毫米求导的 Jacobian。显式 `simnibs46-literal` 选项保留上游行为供对照。
header 采样及网格限制见[输入文件与坐标](docs/zh/INPUTS.md)。

张量输入为四维 NIfTI，最后一维依次为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`。可使用自动
`register-t1` 或完整工作流。使用通用 `register-tensor` 时，需选择以下一种方式：

- 通过 `--world-transform` 提供外部估计的 4×4 仿射矩阵，从输入世界坐标映射到参考世界坐标；
- 仅在外部预处理已对齐扩散和 T1 世界坐标时，显式指定 `--assume-aligned`。

最终张量按 SimNIBS 4.6 的要求保存到 `m2m_<subject>/DTI_coregT1_tensor.nii.gz`。
同一被试的数据并不意味着已经对齐。坐标、掩膜及输入检查见
[输入文件与坐标](docs/zh/INPUTS.md)。

自动估计 SimNIBS 4.6 仿射 T1 配准，并生成张量与 QA 文件：

```bash
dwi2cond-xp register-t1 dti_outputs m2m_subject t1_registration_outputs \
  --mode affine --workers 8
```

## 🚀 最小流程

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

该命令输出 `DTI_tensor.nii.gz`、`DTI_FA.nii.gz`、`DTI_sse.nii.gz`、脑掩膜、有效掩膜及 `nomoco_qa.json`，不生成运动、涡流或场校正产物。对于满足 float32、FSL 标准方向、有限值和非负值要求的未压缩 `.nii`，程序逐块验证后直接使用 mmap 读取；`.nii.gz` 则只解压一次，生成共享的未压缩中间文件。实际读取方式会记录在 QA 中。

运行旧版校正路径，并保留 SimNIBS 4.6 的 b 向量处理方式：

```bash
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --workers 8 --bvec-mode corrected
```

该命令输出校正后的 DWI 及均值图、每个体积的最终变换、nodif 与脑掩膜、张量/FA/SSE 和有效性输出，以及 `legacy_qa.json`。需要与 4.6 脚本的原始 b 向量处理对照时，使用 `--bvec-mode compat46 --component-mode simnibs46-literal`。

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

该命令输出 rad/s 场图、体素位移、带符号的 NIfTI 世界坐标毫米 pull 位移、校正后的 b0/掩膜，以及记录单位的清单。原始 Siemens 缠绕相位仍需 PRELUDE 处理，这一固定 rad/s 路径会拒绝该类输入，不使用近似解缠。

在无需 FSL 运行时依赖的情况下，估计固定反向相位编码 TOPUP 场：

```bash
dwi2cond-xp prepare-topup \
  forward_b0.nii.gz reverse_b0.nii.gz topup_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

SimNIBS 4.6 固定九级配置输出 Hz 场图、样条系数、运动参数、校正后的成对影像、联合有效掩膜及 `topup_qa.json`。项目原有文件名保持不变；`topup_fieldcoef.nii.gz` 与 `topup_movpar.txt` 组成 FSL 可读取的 `--topup=topup` 文件组，包含官方三次样条 intent 元数据。仅支持 FSL 6.0.4 TOPUP 接受的 x/y 单轴子集；z 方向相位编码会明确报错。

运行固定单壳 EDDY 路径，可选合入 TOPUP 场：

```bash
dwi2cond-xp prepare-eddy \
  raw_dwi.nii.gz bvals bvecs brain_mask.nii.gz eddy_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8 \
  --susceptibility-field topup_out/field_hz.nii.gz
```

固定五轮的 SimNIBS 4.6 单壳 EDDY 流程包含运动与二次涡流校正、球面高斯过程、基于预测的 `--repol`、b 向量旋转和壳层对齐。该命令输出校正后的 DWI、用预测值替换后的扫描空间数据、每个体积的 16 个参数、旋转后的 b 向量、异常值图、壳层对齐结果、迭代历史及 `eddy_qa.json`。省略 `--susceptibility-field` 时，仍运行相同的固定 EDDY 算法，只是不使用 TOPUP 场。

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

## 🎨 四模式图

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

分量图按三行（`Ex`、`Ey`、`Ez`）、四列（`scalar`、`vn`、`dir`、`mc`）排列，
共用一个对称色标。切片根据脑掩膜面积最大的位置选择，不根据场强选取。

![四模式电场分量](docs/images/electric_field_xyz_3x4.png)

模长图使用相同的四份向量 NIfTI，并共用一个正值色标：

![四模式电场模长](docs/images/electric_field_magnitude_2x2.png)

这两张图沿用已有实验结果，不代表最终 `v0.3.0` 标签重新运行了完整被试流程。

## ⚡ 先导场支持

`simulate-leadfield` 为四种电导率模式构建 SimNIBS 4.6 `TDCSLEADFIELD` 配置，
保留原始 HDF5 来源信息，并可导出形状为 `(N_spatial * 3, N_electrode - 1)` 的
NumPy 矩阵及 JSON 元数据。电极帽中的第一个电极为参考电极，其余每列对应一个
1 A 基。默认使用 Pardiso，失败时不会自动改用其他求解器。

该接口及 HDF5/NPY 格式已通过单元测试，但当前版本的验证不包含真实被试的完整
全电极先导场计算。详见 [SimNIBS 集成](docs/zh/SIMNIBS_INTEGRATION.md)。

## 🧪 验证结果

项目验证使用过一名私有 HCP 被试。仓库不分发原始影像、体数据派生物、被试标识或
被试级机器可读产物，仅保留两张已渲染且无被试标识的历史电场对比 PNG。
使用这些图片需保留 HCP 致谢；图片的公开展示不能替代接受
[WU-Minn HCP 开放数据使用条款](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)。

- 已记录的 b0+b1000 DTI 拟合包含 881,299 个掩膜内体素，881,194 个有效，
  105 个无效体素被明确填零并记录。
- 在该阶段性实验中，与 FSL 6.0.4 的 WLS 加梯度非线性校正比较，张量相对 L2 误差
  为 `4.18e-6`，平均和 p99 绝对差分别为 `4.78e-10`、`2.10e-9`。
- 同一服务器、相同输出范围下，16 个 worker 的 Python 拟合耗时 9.76 s，
  FSL 6.0.4 为 108.23 s，耗时比 `11.09x`。这是单机 DTI 拟合结果，
  不代表完整 FEM 流程的加速。
- 历史本地合成网格比较记录的最大绝对差，`vn` 为 `0`、`dir`/`mc` 为 `2.22e-16`。
  当时机器可读的对照产物未保留，因此最终标签的电导率结论以当前有区分力的测试
  为依据，不能只依靠这一历史摘要。
- 最终修复之前，真实 `scalar`、`vn`、`dir`、`mc` C3→C4 FEM 均由 Pardiso 完成；
  所有向量电场 NIfTI 数值有限，并严格排除白质、灰质、脑脊液以外的组织。
- 最终标签单独完成了真实 SimNIBS 4.6 `SESSION._prepare()`，使用解析后显式指定的
  网格。在最后一次集成及缓存识别信息修正后，没有重跑完整 HCP 原始 DWI→FEM
  流程或求解器。
- 配置全部真实 FSL 探针后，最终 v0.3.0 检查为 `666 passed`，无跳过项。
  独立覆盖率检查重复执行主批次 `666` 项测试，电极配置批次为 `12 passed`，
  真实合成 TOPUP/EDDY/FNIRT 命令均完成，全部 `13,663/13,663` 条可执行语句达到
  `100.00%` 覆盖率。跨平台 CI 要求相同覆盖率；只有缺少外部前置条件时，
  才跳过相应的可选参考或集成测试。

方法、计时范围及限制见[验证](docs/zh/VALIDATION.md)、
[性能测试](docs/zh/BENCHMARKS.md)和[复现](docs/zh/REPRODUCIBILITY.md)。

## 🛣️ 后续路线图

v0.4.0 稳定版包含算法修复、组件提速及 CPU/GPU 完整流程对照。进一步内存优化已暂停，
优先保留经过验证的
速度和数值行为。当前状态见[项目状态](docs/zh/ROADMAP.md)，迁移说明见
[v0.4.0 发布说明](docs/zh/RELEASE_NOTES_v0.4.0.md)，实测见[性能测试](docs/zh/BENCHMARKS.md)。

## 🗂️ 文档与社区

- [文档目录](docs/zh/README.md)
- [v0.4.0 发布与迁移说明](docs/zh/RELEASE_NOTES_v0.4.0.md)
- [输入文件与坐标](docs/zh/INPUTS.md)
- [方法](docs/zh/METHODS.md)
- [验证](docs/zh/VALIDATION.md)
- [复现](docs/zh/REPRODUCIBILITY.md)
- [贡献指南](docs/zh/CONTRIBUTING.md)
- [安全与医学数据报告](docs/zh/SECURITY.md)
- [支持](docs/zh/SUPPORT.md)
- [行为准则](docs/zh/CODE_OF_CONDUCT.md)
- [更新记录](docs/zh/CHANGELOG.md)
- [第三方声明](THIRD_PARTY_NOTICES.zh-CN.md)

## 🙏 引用与许可证

请按照 [CITATION.cff](CITATION.cff) 引用本软件，并同时引用 SimNIBS 及上游使用的
扩散处理软件。本项目采用 [GPL-3.0-only](LICENSE)；第三方软件包与可选参考工具
保留各自许可证。

部分验证数据由 WU-Minn Human Connectome Project 提供。该项目由 David Van Essen
和 Kamil Ugurbil 领导，获得 NIH Blueprint 资助及华盛顿大学 McDonnell Center for
Systems Neuroscience 支持。引用要求见
[HCP 官方引用说明](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)。
