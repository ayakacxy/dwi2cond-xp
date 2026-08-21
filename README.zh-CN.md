<div align="center">

# dwi2cond-xp

面向 SimNIBS 4.6 的跨平台、运行时无 FSL 的 DTI→电导率流程。

[![Release](https://img.shields.io/github/v/release/ayakacxy/dwi2cond-xp?display_name=tag&sort=semver)](https://github.com/ayakacxy/dwi2cond-xp/releases/latest)
[![CI](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#-验证概览)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF.svg)](docs/SIMNIBS_INTEGRATION.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

[English](README.md) · [文档](docs/README.md) · [验证](docs/VALIDATION.md) · [基准](docs/BENCHMARKS.md) · [更新记录](docs/CHANGELOG.md)

🧠 **DTI 拟合** · ⚡ **运行时无 FSL** · 🧭 **张量重定向** · ⚡ **各向异性 FEM** · 🧪 **100% 语句覆盖**

</div>

`dwi2cond-xp` 是一个跨平台 Python 流程：从已预处理的 diffusion MRI 或六分量
diffusion tensor 出发，生成 SimNIBS 4.6 电导率张量并运行经过验证的各向异性有限元
仿真。主运行路径不依赖 FSL。

这是独立社区项目，不是 SimNIBS 或 FSL 官方发行物。英文 [README](README.md) 是默认
发布入口，本文件同步最重要的安装、输入和科学边界。

## ⚡ 验证概览

| 合同 | 结果 | 证据边界 |
| --- | ---: | --- |
| Python 测试 | **144 passed · 100.00%** | 覆盖 1,644/1,644 条可执行语句，包含本机 FSL reference |
| DTI tensor 一致性 | **relative L2 4.18e-6** | 同一 HCP 输入、WLS 与 gradient-nonlinearity 合同，对照 FSL 6.0.4 |
| DTI 拟合时间 | **9.76 s vs 108.23 s · 11.09x** | 同服务器、输入与输出边界，不代表完整 FEM 加速 |
| 电导率一致性 | **max abs 0 至 2.22e-16** | synthetic mesh 对照 SimNIBS 4.6 的 `vn/dir/mc` |
| 特定 montage FEM | **4/4 模式完成** | Pardiso 完成真实 `scalar/vn/dir/mc` C3→C4 仿真 |

仓库不分发任何解剖影像、被试标识、体数据派生物或机器可读被试产物。完整方法与证据
边界见 [验证](docs/VALIDATION.md) 和 [基准](docs/BENCHMARKS.md)。

## 🧭 支持范围

```text
已预处理单壳 DWI 或六分量 diffusion tensor
  -> 两遍加权最小二乘 DTI 拟合与 QA
  -> 显式映射/重定向到 CHARM T1 网格
  -> scalar / vn / dir / mc 电导率
  -> SimNIBS 4.6 固定 montage FEM
  -> 三分量 E-field NIfTI 与 QA manifest
```

本项目不负责原始 DWI 的 motion、eddy-current、susceptibility/topup/fieldmap 校正，
也不自动估计 6/12 DOF affine 或非线性 DTI→T1 配准。上述步骤及一致的 b-vector
rotation 必须由外部预处理完成。当前未实现 nonlinear PPD tensor reorientation。

纯 Python DTI/tensor 映射核心依赖 NumPy、SciPy、NiBabel、h5py 和 tqdm。Mesh
电导率、FEM 与 lead field 固定要求 `SimNIBS 4.6.0 + Python 3.11`；完整流程的平台
范围受 SimNIBS 4.6.0 可安装和已验证平台限制。

## 🧩 电导率模式

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每种组织使用固定标量电导率，不使用 DTI。 |
| `vn` | 保留主方向和各向异性比例，并逐位置归一化行列式；这是主各向异性模式。 |
| `dir` | 保留方向、比例及局部强度变化，再进行总体强度校准。 |
| `mc` | 保留 DTI 驱动的空间平均电导率变化，但局部各向同性；它是强度变化对照。 |

### 实现公式

下面的公式就是本项目实现并与 SimNIBS 4.6 对照过的映射。对于各向异性组织 $t$
中的单元或体素 $i$，把修复后的扩散张量写成

$$
\mathbf D_i = \mathbf V_i\,\operatorname{diag}
(d_{i1},d_{i2},d_{i3})\,\mathbf V_i^{\mathsf T},
\qquad d_{i1}\ge d_{i2}\ge d_{i3}>0,
$$

其中 $\mathbf V_i$ 的列是主方向，$\sigma_t$ 是组织 $t$ 的参考标量电导率，$w_i$
是四面体体积；在不加权的 voxel 计算中取 $w_i=1$。

#### `vn`：体积归一化各向异性映射

定义局部几何均值 $g_i=(d_{i1}d_{i2}d_{i3})^{1/3}$，核心映射为

$$
\boldsymbol\Sigma_i^{\mathrm{vn}}
=\sigma_t\,\mathbf V_i\,
\operatorname{diag}\!\left(
\frac{d_{i1}}{g_i},\frac{d_{i2}}{g_i},\frac{d_{i3}}{g_i}
\right)\mathbf V_i^{\mathsf T},
\qquad
\det\!\left(\boldsymbol\Sigma_i^{\mathrm{vn}}\right)^{1/3}=\sigma_t.
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
s=\underset{a}{\operatorname{argmin}}
\sum_t(am_t-\sigma_t)^2
=\frac{\sum_t\sigma_t m_t}{\sum_t m_t^2},
\qquad
\boldsymbol\Sigma_i^{\mathrm{dir}}=s\mathbf D_i.
$$

该映射保留 DTI 给出的方向、各向异性和空间强度变化。它属于 Tuch 等提出的线性
扩散率到电导率映射，并对应 Rullmann 等和 Opitz 等使用的 direct mapping [1,2,4]。

#### `mc`：平均电导率对照

`mc` 使用与 `dir` 相同的全局因子 $s$，但把每个局部张量替换为具有相同行列式的
各向同性张量：

$$
\boldsymbol\Sigma_i^{\mathrm{mc}}
=\det\!\left(\boldsymbol\Sigma_i^{\mathrm{dir}}\right)^{1/3}\mathbf I
=s\det(\mathbf D_i)^{1/3}\mathbf I.
$$

所以它保留 DTI 驱动的几何平均电导率空间变化，却移除了方向各向异性。`mc` 是
DTI 派生的对照模式，而不是各向异性张量场 [4]。

三种映射都遵循 SimNIBS 的安全合同：修复无效张量、保持电导率张量正定、默认把
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

对应的 SimNIBS 实现级定义见官方
[dwi2cond 文档](https://simnibs.github.io/simnibs/build/html/documentation/command_line/dwi2cond.html)。

## 🐍 安装

首选方式就是直接基于已有 SimNIBS 4.6 环境安装。若该环境需要保持冻结，可用
`--no-deps` 安装 wheel，避免 pip 改动其依赖：

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.1.0-py3-none-any.whl
dwi2cond-xp --help
```

从源码开发时把 wheel 命令换为 `python -m pip install --no-deps -e .`。当前 wheel
已经通过临时 overlay 与现有 SimNIBS 4.6.0 同时导入验证，该验证没有修改参考环境。

如果希望另建独立环境：

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
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## 📥 输入合同

DWI 必须是已预处理的四维 NIfTI，bvals、bvecs、brain mask 必须与其体积顺序一致。
DTI 模型显式使用 b=0 加一个非零壳层，不会静默拟合全部多壳数据。可选 `grad_dev`
使用 HCP/FSL 九分量约定。

Tensor NIfTI 的末维固定为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`。映射到 T1 前，调用者必须
二选一：提供外部估计的 input-world→reference-world 4×4 affine，或在已有外部对齐
证据时显式使用 `--assume-aligned`。同一被试并不自动意味着 DWI 与 T1 已经对齐。

## 🚀 最小流程

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz --qa-json tensor_fit_qa.json

dwi2cond-xp register-tensor \
  tensor_dwi.nii.gz m2m_subject/T1.nii.gz \
  m2m_subject/DTI_coregT1_tensor.nii.gz \
  --source-mask tensor_valid_mask.nii.gz --assume-aligned \
  --qa-json tensor_registration_qa.json

dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 --cpus 8 --dry-run
```

移除 `--dry-run` 后才执行求解。`scalar/vn/dir/mc` 分目录保存；各向异性 tensor 缺失
时直接报错，不静默回退 scalar。正式体数据只保留 WM/GM/CSF（标签 1/2/3），严格
排除颅骨、头皮、电极和颅外组织。每种模式只保存一个末维为 `Ex/Ey/Ez` 的向量
E-field NIfTI，模长现场计算。

## 🎨 四模式图

分量主图是三行 `Ex/Ey/Ez` × 四列 `scalar/vn/dir/mc` 的 3×4 axial 布局，12 个
panel 共用对称色标；切片只由 brain mask 最大面积决定，不根据场强挑选。

![四模式电场分量](docs/images/electric_field_xyz_3x4.png)

![四模式电场模长](docs/images/electric_field_magnitude_2x2.png)

## 🧪 证据边界

私有 HCP 数据已用于 DTI、CHARM 和真实四模式 FEM 验收。原始影像、体数据派生物、
被试标识和机器可读的被试级产物都不进入仓库或 Release；README 只保留两张无被试
标识的结果示意 PNG，并附 HCP 致谢与数据使用条款链接。16 worker DTI 拟合在同一服务器、同一输出
边界下相对 FSL 6.0.4 实测为 `11.09x`；这只是单机 DTI fitting 结果，不能外推到
预处理、配准、建模或 FEM。全电极 lead-field 接口和数据合同已支持并测试，但当前
发布证据不包含真实被试的全电极完整运行。

本机 release 测试为 `144 passed`、严格 `100.00%` 语句覆盖；跨平台 CI 同样强制
100% 门槛。没有 `dtifit` 的平台只跳过 FSL 对照测试，不降低其余覆盖要求。

详细方法、数值误差、复现步骤、贡献和安全规范见 [文档目录](docs/README.md)。项目采用
[GPL-3.0-only](LICENSE)，第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
