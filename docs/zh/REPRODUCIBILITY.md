# 安装与结果复现

[English](../en/REPRODUCIBILITY.md) | [简体中文](../zh/REPRODUCIBILITY.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 干净环境

使用 Python 3.11，数值栈固定为 NumPy 2.3.0 和 Numba 0.64.0。SimNIBS 集成流程要求 SimNIBS 4.6.0，张量处理核心可独立运行。

在已有 SimNIBS 环境中，从[对应发布页面](https://github.com/ayakacxy/dwi2cond-xp/releases/tag/v0.4.0)下载 wheel，保留现有依赖安装：

```bash
conda activate simnibs
python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl
python -c "import simnibs, dwi2cond_xp; assert simnibs.__version__ == '4.6.0'"
python -m pip check
```

`--no-deps` 假设环境已具备所需依赖版本。新建完整流程环境时使用仓库提供的环境文件：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
dwi2cond-xp --help
```

仅使用核心时，可以创建 Python 3.11 虚拟环境并安装基础包：

```bash
python -m venv .venv
```

Linux/macOS 使用 `source .venv/bin/activate` 激活，Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，然后执行 `python -m pip install .`。

### 开发与构建

开发及发布工具通过 `.[test,release,viz]` 安装。开发时运行 Ruff 和受影响模块的测试；文档修改检查链接、版本与示例。完整数值检查为 `python scripts/run_coverage.py`，包含单元测试及合成 TOPUP/EDDY/FNIRT CLI 流程，并要求 100% 语句覆盖率。

构建 wheel 和源码包：

```bash
python -m build --outdir .work/package/dist
python -m twine check .work/package/dist/*
python scripts/build_release_metadata.py --dist .work/package/dist
python scripts/verify_release.py --dist .work/package/dist
```

构建或性能测试可使用项目内的临时目录和缓存。Linux/macOS 设置示例：

```bash
mkdir -p .work/package/tmp .work/package/cache
export TMPDIR="$PWD/.work/package/tmp" TMP="$PWD/.work/package/tmp" TEMP="$PWD/.work/package/tmp"
export XDG_CACHE_HOME="$PWD/.work/package/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
```

Windows PowerShell：

```powershell
$workDir = Join-Path (Get-Location) ".work/package"
New-Item -ItemType Directory -Force "$workDir/tmp", "$workDir/cache" | Out-Null
$env:TMPDIR = $env:TMP = $env:TEMP = "$workDir/tmp"
$env:XDG_CACHE_HOME = "$workDir/cache"
$env:PIP_CACHE_DIR = "$workDir/cache/pip"
$env:NUMBA_CACHE_DIR = "$workDir/cache/numba"
$env:MPLCONFIGDIR = "$workDir/cache/matplotlib"
```

发布附件包含 wheel、源码包、CycloneDX SBOM 和 `SHA256SUMS`。下载后核对摘要，在独立环境中安装 wheel，检查导入与 CLI 帮助。发布步骤见[发布流程](RELEASE_PROCESS.md)。

## CPU 与可选 CUDA

FNIRT Hessian 装配和 PPD 默认使用 CPU，不需要 CuPy 或 CUDA 设备。`--fnirt-device cuda` 需要 Numba CUDA、可用 NVIDIA 驱动和兼容的 CUDA 工具包；`--ppd-device cuda` 还需要兼容的 CuPy。这些依赖不随基础包安装。

已有 CUDA 测量环境为 Linux、RTX 2080 Ti、CUDA 12.8、Numba 0.64.0 和 CuPy 14.2.0。这些信息用于标明测试环境；CPU 平台检查与 GPU 测量分别列在[验证记录](VALIDATION.md)和[性能报告](BENCHMARKS.md)。

## 参考工具

可选数值对照使用本地安装的 FSL 6.0.4 和 SimNIBS 4.6.0。CI 不下载 FSL，运行流程也不调用 FSL。外部工具不可用时，参考测试会报告跳过。源码摘要和对照产物见[FSL 参考说明](FSL_REFERENCE.md)。

## 真实数据

按原始条款获取扩散与结构数据，准备所选流程需要的 CHARM 头模型。记录输入摘要、形状、壳层选择、掩膜、梯度、坐标选项、求解器、线程数、软件版本、命令及 QA 输出。

性能对照应使用相同输入，并说明是否计入进程启动、JIT 编译、缓存、QA、发布和 FEM。重复运行与首次调用分别报告，不同配置的数据和输出分别保存。既有测量见[性能报告](BENCHMARKS.md)。
