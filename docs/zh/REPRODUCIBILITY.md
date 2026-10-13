# 复现说明

[English](../en/REPRODUCIBILITY.md) | [简体中文](../zh/REPRODUCIBILITY.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 安装环境

完整流程可以安装到已核验的 SimNIBS 4.6.0 环境：

```bash
conda activate simnibs
python -m pip install --no-deps dwi2cond_xp-0.2.0-py3-none-any.whl
python -c "import simnibs, dwi2cond_xp; assert simnibs.__version__ == '4.6.0'"
```

也可以创建独立环境：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e '.[test]'
python -m pip check
ruff check src tests tools scripts
pytest -q
python -m build
```

随后在新的 Python 3.11 环境安装 wheel，并运行：

```bash
dwi2cond-xp --help
dwi2cond-xp fit-dti --help
dwi2cond-xp register-tensor --help
dwi2cond-xp simulate-tdcs --help
```

## 参考工具

CI 不会下载 FSL。可选 FSL 数值测试要求用户自行配置本地 FSL 6.0.4；未安装时须明确显示跳过。SimNIBS 集成测试要求版本恰为 4.6.0。

## 真实数据

复现真实数据结果时，用户须按数据原有条款获取扩散和解剖影像，选择受支持的 `dwi2cond-xp` 预处理分支，并生成 SimNIBS 4.6 CHARM 头模型。超出本项目固定功能范围的输入，例如需要 PRELUDE 解缠的相位图，应先用合适的外部工具处理。不要将可识别身份或受控访问的医学影像上传到 issue。

每次科学复现应记录软件版本、DWI 形状和壳层、掩膜、张量分量顺序、世界坐标变换、插值阶数、电导率模式、求解器、CPU 数量、输出哈希和 QA JSON。
