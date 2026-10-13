# dwi2cond-xp

面向 SimNIBS 4.6 的扩散 MRI 预处理、张量拟合与电导率建模。

[![Release](https://img.shields.io/github/v/release/ayakacxy/dwi2cond-xp?display_name=tag&sort=semver)](https://github.com/ayakacxy/dwi2cond-xp/releases/latest)
[![CI](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml/badge.svg)](https://github.com/ayakacxy/dwi2cond-xp/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/ayakacxy/dwi2cond-xp/badge)](https://scorecard.dev/viewer/?uri=github.com/ayakacxy/dwi2cond-xp)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](docs/zh/VALIDATION.md)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF.svg)](docs/zh/SIMNIBS_INTEGRATION.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[English](README.md) | [简体中文](README.zh-CN.md)

[文档目录](docs/zh/README.md) · [输入文件](docs/zh/INPUTS.md) · [验证记录](docs/zh/VALIDATION.md) · [更新记录](docs/zh/CHANGELOG.md)

`dwi2cond-xp` 处理受支持的原始或已预处理单壳扩散 MRI，将扩散张量映射到 CHARM T1
网格，并运行 SimNIBS 电导率与有限元流程。预处理和张量计算采用 Python 实现，运行时
不需要 FSL；FEM 与先导场命令使用 SimNIBS 4.6.0。

本项目由社区独立维护。当前稳定版本为 `v0.3.0`，处于[维护阶段](docs/zh/ROADMAP.md)。

## 支持的流程

- 单壳 DTI 拟合，可使用 HCP/FSL 梯度非线性校正。
- SimNIBS 4.6 使用的固定 `nomoco`、legacy、GRE、TOPUP 和 EDDY 预处理。
- 仿射或非线性 DTI→T1 配准及张量重定向。
- `scalar`、`vn`、`dir`、`mc` 电导率，固定电极 FEM 和先导场接口。

实现范围限于 SimNIBS `dwi2cond` 使用的设置，不包含任意 FSL 配置、需要 PRELUDE 的
缠绕相位，也不直接将多个非零壳层用于单张量拟合。依赖 SimNIBS 的命令只能在已安装并
验证 SimNIBS 4.6.0 的平台运行。各命令的具体要求见[输入文件与坐标](docs/zh/INPUTS.md)。

## 安装

使用 Python 3.11。已有兼容的 SimNIBS 4.6 环境时，从
[Releases](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.3.0) 下载 wheel，
通过以下方式安装，保留该环境现有依赖：

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.3.0-py3-none-any.whl
dwi2cond-xp --help
```

非线性流程已验证的依赖版本为 NumPy 2.3.0 和 Numba 0.64.0。新建环境或从源码安装
请参阅[安装与结果复现](docs/zh/REPRODUCIBILITY.md)。

不使用 SimNIBS 时，也可只安装 Python 核心：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Windows PowerShell 改用 `.venv\Scripts\Activate.ps1` 激活。运行 FEM 和先导场仍需 SimNIBS。

## 快速开始

准备好单壳 DWI、对应的 b 值和 b 向量，以及完整 CHARM 头模型后，默认流程执行 legacy
预处理和非线性 T1 配准：

```bash
dwi2cond-xp run-pipeline \
  raw_dwi.nii.gz bvals bvecs m2m_subject workflow_outputs \
  --workers 8
```

最终张量保存为 `m2m_subject/DTI_coregT1_tensor.nii.gz`。反向相位编码数据和预拟合
张量的用法见[命令示例](docs/zh/USAGE.md)。

张量 NIfTI 形状为 `(X,Y,Z,6)`，分量顺序为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`。同一被试的
影像不一定已经对齐，应使用自动配准或已知世界坐标变换；只有外部处理已确认对齐时，
才使用 `--assume-aligned`。

检查 C3→C4、1 mA 的仿真设置而不启动求解器：

```bash
dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 --cpus 8 --dry-run
```

去掉 `--dry-run` 后执行求解。各模式使用独立输出目录，向量电场 NIfTI 的分量顺序为
`Ex,Ey,Ez`，仅保留 WM/GM/CSF 标签 1/2/3。FEM 与先导场输出说明见
[SimNIBS 集成](docs/zh/SIMNIBS_INTEGRATION.md)。

## 电导率模式

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每种组织使用固定标量电导率，不依赖 DTI。 |
| `vn` | 保留方向与相对各向异性，逐张量归一化行列式，使其对应组织参考值。 |
| `dir` | 保留方向、各向异性和空间幅度，对选定组织共同拟合一个全局强度比例。 |
| `mc` | 将 `dir` 结果局部各向同性化，同时保留行列式和空间变化。 |

[电导率模型](docs/zh/CONDUCTIVITY.md)给出公式、参考文献、安全限制、奇异张量策略，
以及稳定/逐句兼容两种特征系统选项。

## 验证与性能

最终算法版本在配置全部真实 FSL 比较后完成 **666 项测试，无跳过**。独立覆盖率运行
覆盖 **13,663/13,663 条语句**，包括 666 项主测试、12 项电极组合测试和合成
TOPUP/EDDY/FNIRT CLI 流程。最后的 FEM 修改通过真实 SimNIBS `SESSION._prepare()`
检查；修改后没有重跑求解器或完整 HCP 原始 DWI→FEM 流程。

早期工作包含私有 HCP DTI 与 FSL 比较（张量相对 L2 `4.18e-6`）、四种电导率 FEM 和
各阶段基准。`11.09x` 的 DTI 拟合耗时比来自单台服务器、16 个 Python worker：
`9.76 s` 对 FSL 的 `108.23 s`，不代表完整流程或当前 tag 的速度。先导场接口已通过
单元测试，但尚未完成真实被试全部电极的完整运行。

详细数值、输入规模、计时方式和限制见[验证记录](docs/zh/VALIDATION.md)与
[历史基准](docs/zh/BENCHMARKS.md)。

## 电场示例

分量图为三行 `Ex/Ey/Ez`、四列 `scalar/vn/dir/mc`，共用对称色标。切片按脑掩膜
最大面积选择。模长图使用相同电场和共用正值色标。

![四种模式的电场分量](docs/images/electric_field_xyz_3x4.png)

![四种模式的电场模长](docs/images/electric_field_magnitude_2x2.png)

这两张去标识图片展示早期 HCP 结果，不是最终 tag 的新运行。仓库不分发原始 MRI 或
被试体数据衍生结果。使用时请保留[HCP 致谢并遵守数据条款](THIRD_PARTY_NOTICES.zh-CN.md#human-connectome-project-数据)。

## 贡献与引用

[贡献指南](docs/zh/CONTRIBUTING.md) · [获取帮助](docs/zh/SUPPORT.md) · [安全报告](docs/zh/SECURITY.md) · [行为准则](docs/zh/CODE_OF_CONDUCT.md)

请使用 [CITATION.cff](CITATION.cff) 引用项目，并引用实际使用的 SimNIBS 和扩散处理
软件。许可证为 [GPL-3.0-only](LICENSE)，另见[第三方声明](THIRD_PARTY_NOTICES.zh-CN.md)。

验证数据部分来自 WU-Minn Human Connectome Project，由 David Van Essen 和
Kamil Ugurbil 领导，获 NIH Blueprint 及华盛顿大学 McDonnell Center for Systems
Neuroscience 支持。
