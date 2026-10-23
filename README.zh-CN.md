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

`dwi2cond-xp` 将单壳层扩散 MRI 转为 SimNIBS 4.6 可用的电导率张量，并提供各向异性 FEM 和先导场接口。原始 DWI、已预处理 DWI 和拟合张量均有对应入口，运行时不需要 FSL。

CPU 为默认后端，可选 CUDA 加速非线性配准的 Hessian 装配和 PPD。安装和基本流程见下文，完整命令见[使用说明](docs/zh/USAGE.md)。

## ⚡ 验证概览

在 Linux、Xeon Gold 6230、8 个 CPU 核心和 RTX 2080 Ti 上，完整运行结果如下：

| 后端 | 总耗时 | 主机 RSS 峰值 | GPU 分配峰值 |
| --- | ---: | ---: | ---: |
| CPU | **205.45 秒** | **6.50 GiB** | — |
| CUDA | **163.34 秒** | **6.72 GiB** | **8.83 GiB** |

输入为一组 HCP 单壳层数据，包含梯度偏差校正、`nomoco` 预处理和非线性 T1 配准。计时包含启动、计算、QA 和 m2m 张量发布，文件/JIT 缓存已建立，关闭 FEM。每行对应一次运行；此处未计时默认 legacy 预处理路径。完整设置、FSL 对照和数值结果见[性能报告](docs/zh/BENCHMARKS.md)。

发布验证完成 **960 项测试、无跳过**，覆盖 15,364 条语句，**语句覆盖率 100%**。跨平台 CI 覆盖 Linux、macOS 和 Windows；测试条件和可选参考检查见[验证记录](docs/zh/VALIDATION.md)。

## 🧭 支持范围

```text
原始或预处理的单壳层 DWI，或已拟合扩散张量
  -> 预处理和两次加权最小二乘拟合
  -> T1 配准和张量重定向
  -> QA 及写入 CHARM m2m 目录
  -> scalar / vn / dir / mc 电导率
  -> SimNIBS FEM 或先导场
```

支持 `nomoco`、两轮 legacy 校正、固定 GRE 场图校正、反向相位编码 TOPUP 和单壳层 EDDY。T1 对齐支持仿射配准和固定 FNIRT/PPD 非线性路径。

张量核心使用 Python、NumPy、SciPy、NiBabel、Numba 和 h5py。网格电导率、FEM 和先导场需要 SimNIBS 4.6.0。FSL 仅用于可选数值对照。本项目由社区独立维护。

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

使用 Python 3.11。在已有 SimNIBS 4.6.0 环境中，从[发布页面](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.4.0)下载 wheel 后，保留现有依赖安装：

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl
dwi2cond-xp --help
```

创建独立的完整流程环境：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
```

仅使用张量处理核心时，可在 Python 3.11 环境中执行 `python -m pip install .`。SimNIBS 仅用于其集成流程，CUDA 依赖为可选。环境配置、GPU 要求和开发安装见[复现说明](docs/zh/REPRODUCIBILITY.md)。

## 📥 输入文件与坐标

- DWI、b 值和 b 向量必须与体积顺序对应。张量模型使用 b=0 和一个非零壳层，多壳层数据需显式选壳。
- 掩膜和可选 HCP/FSL 九分量 `grad_dev` 应与各自图像网格一致。
- 拟合张量为有限值 `(X,Y,Z,6)` NIfTI，分量为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`。
- 默认梯度、存储分量、PPD 和张量基底模式保持物理方向，对照模式需显式选择。
- T1 配准使用完整 CHARM 目录，将 `DTI_coregT1_tensor.nii.gz` 写到 T1 网格。同一被试的图像不一定已经对齐。

相位编码轴、支持的校正、掩膜和坐标约定见[输入文件与坐标](docs/zh/INPUTS.md)。

## 🚀 最小流程

默认执行 legacy 校正和非线性 T1 配准：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

反向相位编码数据通过同一流程准备 TOPUP 和 EDDY：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --preprocessing-mode eddy \
  --reverse-phase-encoding reverse_pe_4d.nii.gz \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

已拟合张量可从配准阶段进入：

```bash
dwi2cond-xp run-prefit-pipeline \
  DTI_tensor.nii.gz m2m_subject workflow_outputs --workers 8
```

QA 通过后，流程向 m2m 发布张量及来源记录；可恢复的发布错误会还原已接受的文件对。固定 C3 到 C4 仿真可先检查配置：

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn \
  --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 \
  --cpus 8 --dry-run
```

移除 `--dry-run` 后执行求解，各电导率模式使用独立目录。电场输出为 T1 网格的 `Ex,Ey,Ez` 向量 NIfTI，掩膜限定为 WM/GM/CSF 标签 1、2、3。独立准备、拟合和配准命令见[使用说明](docs/zh/USAGE.md)。

## 🎨 四模式图

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

行对应 `Ex/Ey/Ez`，列对应 `scalar/vn/dir/mc`，采用同一对称色阶；切片按脑掩膜面积最大的位置选择。

![四模式电场分量](docs/images/electric_field_xyz_3x4.png)

幅值图使用相同向量场和同一正值色阶：

![四模式电场幅值](docs/images/electric_field_magnitude_2x2.png)

两图展示项目验证数据中已完成的四模式仿真。

## ⚡ 先导场支持

`simulate-leadfield` 为四种电导率模式配置 SimNIBS `TDCSLEADFIELD`。原始 HDF5 可导出为 `(N_spatial*3,N_electrode-1)` 的 NumPy 矩阵及 JSON 说明。帽文件的第一个电极作为参考，其余每个电极对应 1 A 基底。

接口和格式已经测试；公开结果尚不包含全被试、全电极先导场性能测试。具体约定见[SimNIBS 集成](docs/zh/SIMNIBS_INTEGRATION.md)。

## 🧪 验证结果

验证覆盖图像方向、梯度处理、WLS、固定 GRE/TOPUP/EDDY 路径、仿射/FNIRT 配准、PPD、电导率映射和 SimNIBS 消费端。合成示例包含真实 FSL 6.0.4 与 SimNIBS 4.6.0 对照。

完整 CPU/CUDA 对照中，64 项数值产物有 61 项逐字节一致，脑内张量相对 L2 为 `1.8693e-9`，健康 V1 轴向角 P99 为 `1.2074e-6` 度。条件、指标和历史阶段结果见[验证记录](docs/zh/VALIDATION.md)。

真实数据验证使用私有 HCP 数据。仓库提供两份渲染后的比较图，不分发原始影像或被试衍生数据。数据访问须遵守 [WU-Minn HCP 条款](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)。

## 🛣️ 后续路线图

维护工作关注正确性、平台兼容和有测量依据的运行效率改进。支持的流程与当前重点见[项目状态](docs/zh/ROADMAP.md)。

## 🗂️ 文档与社区

- [文档目录](docs/zh/README.md)
- [命令示例](docs/zh/USAGE.md)与[输入要求](docs/zh/INPUTS.md)
- [算法说明](docs/zh/METHODS.md)与[电导率公式](docs/zh/CONDUCTIVITY.md)
- [验证记录](docs/zh/VALIDATION.md)、[性能报告](docs/zh/BENCHMARKS.md)与[复现方法](docs/zh/REPRODUCIBILITY.md)
- [版本说明](docs/zh/RELEASE_NOTES_v0.4.0.md)与[更新记录](docs/zh/CHANGELOG.md)
- [贡献指南](docs/zh/CONTRIBUTING.md)、[使用支持](docs/zh/SUPPORT.md)、[安全政策](docs/zh/SECURITY.md)与[社区行为准则](docs/zh/CODE_OF_CONDUCT.md)
- [第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)

## 🙏 引用与许可证

引用本软件时请使用 [CITATION.cff](CITATION.cff)，同时引用 SimNIBS 和流程中使用的扩散处理软件。项目采用 [GPL-3.0-only](LICENSE) 许可，依赖保留各自许可。

验证数据部分由 WU-Minn Human Connectome Project 提供，由 David Van Essen 和 Kamil Ugurbil 领导，获 NIH Blueprint 资金及华盛顿大学 McDonnell Center for Systems Neuroscience 支持。详见 [HCP 引用指南](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)。
