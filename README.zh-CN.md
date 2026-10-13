<div align="center">

# dwi2cond-xp

面向 SimNIBS 4.6 的跨平台 DTI 电导率计算流程，运行时不依赖 FSL。

[![Release](https://img.shields.io/github/v/release/ayakacxy/dwi2cond-xp?display_name=tag&sort=semver)](https://github.com/ayakacxy/dwi2cond-xp/releases/latest)
[![CI](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/ayakacxy/dwi2cond-xp/badge)](https://scorecard.dev/viewer/?uri=github.com/ayakacxy/dwi2cond-xp)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#-验证概览)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF.svg)](docs/zh/SIMNIBS_INTEGRATION.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

[English](README.md) · [文档](docs/zh/README.md) · [验证结果](docs/zh/VALIDATION.md) · [历史基准](docs/zh/BENCHMARKS.md) · [更新记录](docs/zh/CHANGELOG.md)

🧠 **DTI 拟合** · ⚡ **运行时不依赖 FSL** · 🧭 **张量重定向** · ⚡ **各向异性 FEM** · 🧪 **100% 语句覆盖率**

</div>

> **历史版本说明：** `v0.2.0` 是最初实现不依赖 FSL 的预处理版本。后续审计发现的流程和数值问题在 `v0.3.0` 中作了修复，新部署请使用后续版本并阅读其验证说明。2026-08-29 的重新发布只调整文档和发布元数据，算法源码仍为原始 `v0.2.0` 版本。

`dwi2cond-xp` 是跨平台 Python 工具，可处理受支持的原始或已预处理单壳扩散 MRI，生成 SimNIBS 4.6 电导率张量，并运行各向异性有限元仿真。计算流程不依赖 FSL。

本项目由社区独立维护，不是 SimNIBS 或 FSL 官方发行版。

## ⚡ 验证概览

| 检查项目 | 结果 | 测试条件与适用范围 |
| --- | ---: | --- |
| SimNIBS 4.6 部分预处理功能 | **纯 Python，运行时不依赖 FSL** | `nomoco`、legacy、固定 GRE/TOPUP/EDDY、线性/FNIRT 配准及 PPD 张量重定向 |
| Python 测试 | **100.00% 语句覆盖率** | 最终 v0.2 发布系列检查覆盖 12,443/12,443 条可执行语句 |
| DTI 张量一致性 | **相对 L2 4.18e-6** | 沿用早期 HCP 阶段结果：相同输入，WLS 加梯度非线性校正，对照 FSL 6.0.4 |
| DTI 拟合耗时 | **9.76 s 对 108.23 s，11.09x** | 同一服务器、输入和输出集；Python 使用 16 个工作进程，FSL 为单进程；不代表完整 FEM 加速 |
| 电导率一致性 | **最大绝对差 0 至 2.22e-16** | 沿用合成网格比较，`vn`、`dir`、`mc` 对照 SimNIBS 4.6 |
| 固定电极配置 FEM | **4/4 模式完成** | 历史 C3→C4 Pardiso 运行：`scalar`、`vn`、`dir`、`mc` |

仓库不分发解剖影像、被试标识、体数据派生物或机器可读的被试结果。完整方法和适用范围见[验证结果](docs/zh/VALIDATION.md)与[历史基准](docs/zh/BENCHMARKS.md)。

## 🧭 支持范围

`v0.2.0` 的预处理和电导率流程如下：

```text
原始或已预处理的单壳 DWI，或六分量扩散张量
  -> 选定 SimNIBS 4.6 预处理分支并进行 WLS DTI 拟合
  -> QA、运行记录和缓存检查
  -> 张量映射和重定向到 CHARM T1 网格
  -> scalar / vn / dir / mc 电导率
  -> SimNIBS 4.6 固定电极配置 FEM
  -> 向量电场 NIfTI 和 QA 记录
```

`v0.2.0` 实现了 SimNIBS 4.6 legacy 运动/涡流校正、固定 GRE 场图与 TOPUP 分支，以及可组合 TOPUP 场的固定单壳 EDDY `--repol` 流程。配准包括自动 6/12 自由度 DTI→T1 线性配准、固定 FNIRT 和非线性 PPD 张量重定向。同时提供 QA、原子写入的阶段记录、缓存检查和进度显示。FSL 只作为可选本地数值参考工具，正式预处理运行不会调用它。

纯 Python DTI 与张量映射核心使用 NumPy、SciPy、NiBabel、h5py 和 tqdm。网格电导率、FEM 和先导场流程要求 SimNIBS 4.6.0；这些功能的平台范围取决于 SimNIBS 4.6.0 能否安装并通过验证。

## 🧩 电导率模式

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每种组织使用固定标量电导率，不使用 DTI。 |
| `vn` | 保留张量方向和各向异性比例，按位置将行列式归一化到组织参考电导率，是主要的各向异性模式。 |
| `dir` | 保留方向、各向异性比例和空间强度变化，再进行全局强度校准。 |
| `mc` | 保留 DTI 驱动的空间平均电导率变化，将局部张量变为各向同性，用作强度变化对照。 |

### 实现公式

以下公式对应本项目实现并与 SimNIBS 4.6 比较过的映射。对于各向异性组织 $t$
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

三种映射都遵循 SimNIBS 的数值保护措施：修复无效张量、保持电导率张量正定、默认把
特征值限制在 2 S/m 以内，并把最大/最小特征值之比限制为 10。`vn` 实际执行
“归一化→安全修正→再次归一化→再次安全修正”；如果最后的边界修正被触发，上式中
理想的行列式等式可能出现轻微偏移。不参与各向异性的组织统一使用
$\boldsymbol\Sigma_i=\sigma_t\mathbf I$。

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

SimNIBS 中对应的定义见官方
[dwi2cond 文档](https://simnibs.github.io/simnibs/build/html/documentation/command_line/dwi2cond.html)。

## 🐍 安装

完整流程验证使用 Python 3.11 和 SimNIBS 4.6.0，建议安装到现有 SimNIBS 4.6 环境。若需保持该环境的依赖版本，可用 `--no-deps` 安装 wheel：

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.2.0-py3-none-any.whl
dwi2cond-xp --help
```

从源码开发时，将 wheel 安装命令换为 `python -m pip install --no-deps -e .`。发布 wheel 曾在读取现有参考环境的临时 overlay 中通过与 SimNIBS 4.6.0 一同导入的检查，该测试未改动参考环境。

也可以创建独立环境：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
dwi2cond-xp --help
```

仅使用纯 Python 核心时：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
dwi2cond-xp --help
```

Windows PowerShell 的环境激活命令为 `.venv\Scripts\Activate.ps1`。

在允许调整依赖的开发环境中：

```bash
python -m pip install -e '.[test]'
pytest -q --cov=dwi2cond_xp --cov-report=term-missing --cov-fail-under=100
ruff check src tests tools scripts
```

## 📥 输入要求

`fit-dti` 接受已预处理的四维 NIfTI，以及匹配的 b 值、b 向量和扩散脑掩膜。`preprocess-nomoco` 接受原始单壳 DWI，生成 b0 参考和脑掩膜，但不对 DWI 体积做运动、涡流或磁敏感畸变校正。只有这种处理方式适合数据时才使用它。可选梯度非线性系数 `grad_dev` 采用 FSL/HCP 九分量约定。

`preprocess-legacy` 按 SimNIBS 4.6 顺序执行两轮 6/12 自由度校正，每体积仅做一次最终 sinc 重采样。默认 `compat46` 逐字节保留原始 b 向量文件，与上游脚本一致；明确选择 `corrected` 时才根据最终变换旋转 b 向量。

张量输入必须是末维依次为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz` 的四维 NIfTI。使用 `register-tensor` 映射到 T1 时，须选择一种方式：

- 通过 `--world-transform` 提供外部估计的“输入世界坐标→参考世界坐标”4×4 仿射矩阵；
- 仅在外部预处理已对齐 DWI 与 T1 世界坐标时，使用 `--assume-aligned`。

最终张量按 SimNIBS 4.6 所需路径保存在 `m2m_<subject>/DTI_coregT1_tensor.nii.gz`。坐标、掩膜和错误处理详见[输入要求](docs/zh/INPUTS.md)。

也可以自动估计 SimNIBS 4.6 仿射 T1 配准，并生成张量和 QA 输出：

```bash
dwi2cond-xp register-t1 dti_outputs m2m_subject t1_registration_outputs \
  --mode affine --workers 8
```

## 🚀 最小流程

使用八个工作进程运行不依赖 FSL 的 SimNIBS 4.6 `nomoco` 原始 DWI 路径：

```bash
dwi2cond-xp preprocess-nomoco \
  raw_dwi.nii.gz bvals bvecs nomoco_outputs \
  --workers 8
```

输出 `DTI_tensor.nii.gz`、`DTI_FA.nii.gz`、`DTI_sse.nii.gz`、脑掩膜、有效性掩膜和 `nomoco_qa.json`，不生成运动、涡流或畸变场结果。符合要求的未压缩 `.nii` 分块检查后直接通过 mmap 使用；`.nii.gz` 只解码一次到共享的未压缩中间文件。所用方法记录在 QA 中。

运行 legacy 校正，并保留 SimNIBS 4.6 的 b 向量处理方式：

```bash
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --workers 8 --bvec-mode compat46
```

输出校正 DWI 及均值、每体积的最终变换、nodif、脑掩膜、张量/FA/SSE、有效性掩膜和 `legacy_qa.json`。`--bvec-mode corrected` 会改变相对 4.6 脚本的 b 向量处理，应明确按科学分析需要选择。

从已换算为 rad/s 的场图准备固定 GRE/FUGUE 校正，再将带符号的世界坐标毫米位移组合到 legacy 命令：

```bash
dwi2cond-xp prepare-fieldmap \
  magnitude.nii.gz field_rads.nii.gz nodif_brain.nii.gz fieldmap_out \
  --dwell-ms 0.5 --phase-encoding-direction y- --workers 8
dwi2cond-xp preprocess-legacy \
  raw_dwi.nii.gz bvals bvecs legacy_outputs \
  --fieldmap-displacement fieldmap_out/displacement_world_mm.nii.gz --workers 8
```

原始 Siemens 缠绕相位仍需 PRELUDE；本固定 rad/s 分支会拒绝该输入。

估计固定反向相位编码 TOPUP 场，无需 FSL 运行依赖：

```bash
dwi2cond-xp prepare-topup \
  forward_b0.nii.gz reverse_b0.nii.gz topup_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8
```

输出场、样条系数、运动参数、校正图像对、联合有效性掩膜和 `topup_qa.json`。只支持 FSL 6.0.4 TOPUP 接受的 x/y 单轴设置，z 相位编码会报错。

运行固定单壳 EDDY 流程，可选择组合 TOPUP 场：

```bash
dwi2cond-xp prepare-eddy \
  raw_dwi.nii.gz bvals bvecs brain_mask.nii.gz eddy_out \
  --readout-seconds 0.05 --phase-encoding-direction y --workers 8 \
  --susceptibility-field topup_out/field_hz.nii.gz
```

输出校正 DWI、经过预测替换的扫描空间数据、每体积 16 个参数、旋转后 b 向量、异常切片图、壳层对齐结果、迭代记录和 `eddy_qa.json`。不提供 `--susceptibility-field` 时，同一固定 EDDY 算法在不使用 TOPUP 场的情况下运行。

拟合已预处理的单壳 DWI：

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz \
  --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz \
  --qa-json tensor_fit_qa.json
```

将张量映射到 CHARM T1 网格。以下示例仅适用于已由外部流程完成对齐的情况：

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

移除 `--dry-run` 后执行 SimNIBS。`scalar`、`vn`、`dir`、`mc` 分别运行，各有独立输出目录和运行记录。各向异性张量缺失时会报错，不自动改用 `scalar`。

每次仿真输出一个 T1 网格上的向量电场 NIfTI，末维为 `Ex,Ey,Ez`。模长由 `sqrt(Ex^2 + Ey^2 + Ez^2)` 计算。被试空间体数据严格限制在白质、灰质、脑脊液（标签 `1,2,3`）内，排除颅骨、头皮、电极和颅外组织。

## 🎨 体素电场比较

```bash
dwi2cond-xp compare-fields \
  scalar_E.nii.gz vn_E.nii.gz dir_E.nii.gz mc_E.nii.gz \
  m2m_subject/T1.nii.gz m2m_subject/final_tissues.nii.gz \
  electric_field_xyz_3x4.png \
  --view components --plane axial --mask-labels 1 2 3
```

分量图以 `Ex`、`Ey`、`Ez` 为三行，以 `scalar`、`vn`、`dir`、`mc` 为四列，共用一个对称色标。切片仅按脑掩膜面积最大处选择，不根据场强挑选。

![四模式电场分量](docs/images/electric_field_xyz_3x4.png)

模长图使用相同四份向量 NIfTI，共用正值色标：

![四模式电场模长](docs/images/electric_field_magnitude_2x2.png)

## ⚡ 先导场

`simulate-leadfield` 可为四种电导率模式生成 SimNIBS 4.6 `TDCSLEADFIELD` 配置，保留原始 HDF5 来源信息，并导出形状为 `(N_spatial * 3, N_electrode - 1)` 的 NumPy 矩阵及 JSON 元数据。电极帽中第一个电极是参考，每个其余列对应一个 1 A 基函数。默认使用 Pardiso，失败时不自动换求解器。

该接口及 HDF5/NPY 格式已通过单元测试，但本版本没有完整被试的全电极先导场运行记录。详见 [SimNIBS 集成](docs/zh/SIMNIBS_INTEGRATION.md)。

## 🧪 验证结果

历史发布系列验证使用一个私有 HCP 被试。仓库不分发原始影像、体数据派生物、被试标识或机器可读的被试级结果。两张已渲染电场比较 PNG 不含被试标识，仅作结果示意；使用时须保留 HCP 致谢，也不能因此免除接受 [WU-Minn HCP 开放数据使用条款](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)的要求。

- 完成 b0+b1000 DTI 输出，掩膜内 881,299 个体素，其中 881,194 个有效；105 个无效体素已置零并记录。
- 对照 FSL 6.0.4 WLS 加梯度非线性校正，HCP 张量相对 L2 为 `4.18e-6`，平均和 P99 绝对差为 `4.78e-10`、`2.10e-9`。
- 同一服务器、相同输出集下，16 工作进程 Python 拟合耗时 9.76 秒，FSL 6.0.4 为 108.23 秒（`11.09x`）。这是单机 DTI 拟合结果，不代表完整 FEM 加速。
- 合成网格电导率与 SimNIBS 4.6 比较，`vn` 最大绝对差为 `0`，`dir`/`mc` 为 `2.22e-16`。
- 真实 `scalar`、`vn`、`dir`、`mc` C3→C4 Pardiso FEM 均已完成，全部向量电场 NIfTI 数值有限，严格排除白质/灰质/脑脊液外组织。
- 最终 v0.2 发布系列检查对应代码冻结提交 `90a0553`：Linux 为 `535 passed, 7 skipped`，合并覆盖率为 `12,443/12,443` 条可执行语句（`100.00%`）。随后的原始 tag 提交和 2026-08-29 重新发布只调整文档或发布元数据，没有改动 v0.2 算法源码和测试。

具体方法、计时范围和局限见[验证结果](docs/zh/VALIDATION.md)、[历史基准](docs/zh/BENCHMARKS.md)和[复现说明](docs/zh/REPRODUCIBILITY.md)。

## 🗂️ 文档与社区

- [文档目录](docs/zh/README.md)
- [计算方法](docs/zh/METHODS.md)
- [参与贡献](docs/zh/CONTRIBUTING.md)
- [安全与医学数据报告](docs/zh/SECURITY.md)
- [获取帮助](docs/zh/SUPPORT.md)
- [行为准则](docs/zh/CODE_OF_CONDUCT.md)
- [更新记录](docs/zh/CHANGELOG.md)
- [第三方说明](THIRD_PARTY_NOTICES.zh-CN.md)

## 🙏 引用与许可

引用本软件请使用 [CITATION.cff](CITATION.cff)，同时引用 SimNIBS 及上游扩散处理软件。本项目采用 [GPL-3.0-only](LICENSE)，第三方依赖和可选参考工具各自遵循其原有许可证。

部分验证数据来自 WU-Minn Human Connectome Project，由 David Van Essen 和 Kamil Ugurbil 领导，得到 NIH Blueprint 经费及华盛顿大学 McDonnell Center for Systems Neuroscience 支持。引用方式见 [HCP 官方引用指南](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)。
