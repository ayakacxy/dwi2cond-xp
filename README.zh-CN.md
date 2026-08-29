<div align="center">

# dwi2cond-xp

面向 SimNIBS 4.6 的跨平台、运行时无 FSL 的 DTI→电导率流程。

[![Release: v0.1.0](https://img.shields.io/badge/release-v0.1.0-blue.svg)](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.1.0)
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
diffusion tensor 出发，生成 SimNIBS 4.6 电导率张量并组织各向异性有限元仿真。
主运行路径不依赖 FSL。

这是独立社区项目，不是 SimNIBS 或 FSL 官方发行物。英文 [README](README.md) 是默认
发布入口，本文件同步最重要的安装、输入和科学边界。

> **历史版本说明：** 本树对应 2026-08-21 发布的 `v0.1.0`。该 tag 于
> 2026-08-29 仅因科学表述和发布元数据修正而重发；算法源码仍是原始 `v0.1.0`
> baseline。下列能力和实验结论均限定于本版本及其明确输入边界。

## ⚡ 验证概览

| 合同 | 结果 | 证据边界 |
| --- | ---: | --- |
| Python 测试 | **144 passed · 100.00%** | `v0.1.0` 发布记录：覆盖 1,644/1,644 条 package 语句，包含已配置的本机 FSL reference |
| DTI tensor 对照 | **relative L2 4.18e-6** | 历史私有 HCP b0+b1000 输入，同一 WLS/gradient-nonlinearity 合同，对照 FSL 6.0.4 |
| DTI 拟合时间 | **9.76 s vs 108.23 s · 11.09x** | 历史单服务器、单输入拟合与输出边界，不代表端到端加速 |
| 电导率对照 | **max abs 0 至 2.22e-16** | 历史 synthetic sphere mesh，对照 SimNIBS 4.6 的 `vn/dir/mc` |
| 特定 montage FEM | **4/4 模式完成** | 历史私有被试上由 Pardiso 完成 `scalar/vn/dir/mc` C3→C4 仿真 |

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
rotation 必须由外部预处理完成。`v0.1.0` 未实现 nonlinear PPD tensor reorientation。

纯 Python DTI/tensor 映射核心依赖 NumPy、SciPy、NiBabel、h5py 和 tqdm。Mesh
电导率、FEM 与 lead field 固定要求 `SimNIBS 4.6.0 + Python 3.11`；完整流程的平台
范围受 SimNIBS 4.6.0 可安装和已验证平台限制。

## 🧩 电导率模式

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每种组织使用固定标量电导率，不使用 DTI。 |
| `vn` | 保留方向和各向异性比例，并逐位置把行列式归一化到组织参考电导率，受 safety bounds 约束。 |
| `dir` | 保留方向、比例和局部幅值变化；默认使用一个跨全部所选各向异性组织的全局标度，而非逐组织单独标定。 |
| `mc` | 使用与 `dir` 相同的全局标度，再把每个 tensor 变为保持局部行列式的各向同性 tensor；它是强度变化对照。 |

所选各向异性组织中的精确全零 tensor 会先替换为该组织的标量电导率 tensor，因此可以
进入默认或非校准路径。默认 `dir/mc` 全局校准要求 aggregate determinant 为正且有限；
`--no-correct-intensity` 会明确跳过全局校准，仅执行逐 tensor safety path。`v0.1.0`
不存在公开的 `strict-fsl/robust` 拟合模式切换。版本专属公式与退化输入边界见
[方法](docs/METHODS.md)。

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

## 🧪 历史证据边界

私有 HCP 数据曾用于 `v0.1.0` 的 DTI、CHARM 和真实四模式 FEM 验收。原始影像、体数据派生物、
被试标识和机器可读的被试级产物都不进入仓库或 Release；README 只保留两张无被试
标识的结果示意 PNG，并附 HCP 致谢与数据使用条款链接。16 worker DTI 拟合在同一服务器、同一输出
边界下相对 FSL 6.0.4 实测为 `11.09x`；这只是单机 DTI fitting 结果，不能外推到
预处理、配准、建模或 FEM。全电极 lead-field 接口和数据合同已支持并测试，但当前
发布证据不包含真实被试的全电极完整运行。

发布记录中的本机测试为 `144 passed`、严格 `100.00%` 语句覆盖；跨平台 CI 同样强制
100% 门槛。没有 `dtifit` 的平台只跳过 FSL 对照测试，不降低其余覆盖要求。

2026-08-29 的文档修正没有重跑私有 HCP、sphere、FEM 或性能实验。详细方法、数值误差、
复现步骤、贡献和安全规范见 [文档目录](docs/README.md)。项目采用
[GPL-3.0-only](LICENSE)，第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
