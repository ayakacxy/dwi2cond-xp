# 安装与结果复现

[English](../en/REPRODUCIBILITY.md) | [简体中文](../zh/REPRODUCIBILITY.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 环境准备

完整流程可安装到已经验证的 SimNIBS 4.6.0 环境中：

```bash
conda activate simnibs
python -m pip install --no-deps dwi2cond_xp-0.4.0-py3-none-any.whl
python -c "import simnibs, dwi2cond_xp; assert simnibs.__version__ == '4.6.0'"
```

使用对应 GitHub Release 中的 v0.4.0 wheel。
在该冻结环境中进行源码开发，可将 wheel 安装命令改为
`python -m pip install --no-deps -e .`。该方式保留既有依赖版本，请确认其与已验证版本一致。

也可创建独立环境：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e '.[test,release,viz]'
python -m pip check
ruff check src tests tools scripts
pytest -q tests/test_tensor_fit.py
python scripts/check_markdown_links.py
python scripts/verify_version.py
python -m build
```

开发中优先运行与修改有关的测试。发布前，或修改共用数值计算内核后，运行包含 Numba
代码覆盖率的完整检查：

```bash
python scripts/run_coverage.py
```

该命令在隔离的 Numba 缓存中运行单元测试和真实合成 TOPUP/EDDY/FNIRT 完整流程，合并
覆盖率并要求达到 100%，因此比局部测试耗时更长。

将 wheel 安装到全新的 Python 3.11 环境后检查：

```bash
dwi2cond-xp --help
dwi2cond-xp fit-dti --help
dwi2cond-xp register-tensor --help
dwi2cond-xp simulate-tdcs --help
```

核验发布文件时，从同一 GitHub Release 下载 wheel、源码包、SBOM 和 `SHA256SUMS`
到新目录。运行 `sha256sum -c SHA256SUMS`，在新环境安装下载的 wheel，并检查导入与
命令行。tag 工作流还检查 Twine、wheel 内容、归档中的隐私数据、依赖、构建来源和 SBOM。

## CPU 与可选 CUDA

FNIRT Hessian 组装和 PPD 均默认使用 CPU，不需要 CuPy 或 CUDA 设备。
非线性配准及 pipeline 命令提供两个独立选项：`--fnirt-device cuda` 和
`--ppd-device cuda`。前者使用 Numba CUDA，后者还需要兼容的 CuPy。
CPU wheel 和环境文件不会安装这些 GPU 依赖。

实际验证环境为 Linux、RTX 2080 Ti、CUDA 12.8、Numba 0.64.0 和 CuPy 14.2.0。
这是已测运行环境，不是干净环境安装步骤或 macOS/Windows GPU 认证。
显式 CUDA 执行失败会终止；无 GPU 时选择 `cpu`。细节见[方法](METHODS.md)和
[v0.4.0 发布说明](RELEASE_NOTES_v0.4.0.md)。

## 参考工具

CI 不下载 FSL。可选 FSL 数值比较要求调用者配置本地 FSL 6.0.4；缺失时必须明确报告
跳过。SimNIBS 集成测试要求版本精确为 4.6.0。

## 真实数据

复现真实数据结果需按原数据条款获得扩散与解剖影像，选择受支持的预处理方式，并用
SimNIBS 4.6 CHARM 生成头模型。固定子集以外的输入（如需 PRELUDE 的缠绕相位）应先
用合适的外部工具处理。不要向 issue 上传可识别个人身份或受控访问的医学影像。

每次科学复现应记录软件版本、DWI 形状与壳层、掩膜、张量顺序、世界坐标变换、插值阶数、
电导率模式、求解器、CPU 数、输出哈希和 QA JSON。
