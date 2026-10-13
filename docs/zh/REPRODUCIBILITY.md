# 复现

[English](../en/REPRODUCIBILITY.md) | [简体中文](../zh/REPRODUCIBILITY.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

## 准备环境

完整流程可安装到已经确认版本为 SimNIBS 4.6.0 的现有环境：

```bash
conda activate simnibs
python -m pip install --no-deps dwi2cond_xp-0.1.0-py3-none-any.whl
python -c "import simnibs, dwi2cond_xp; assert simnibs.__version__ == '4.6.0'"
```

若需新建独立环境：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e '.[test]'
python -m pip check
ruff check src tests tools scripts
pytest -q
python -m build
```

然后在新的 Python 3.11 环境中安装生成的 wheel，并检查命令入口：

```bash
dwi2cond-xp --help
dwi2cond-xp fit-dti --help
dwi2cond-xp register-tensor --help
dwi2cond-xp simulate-tdcs --help
```

## 参考工具

CI 不下载 FSL。可选的 FSL 数值对照需要用户在本机配置 FSL 6.0.4；缺少 FSL 时必须明确显示跳过。SimNIBS 集成测试固定要求 4.6.0。

## 真实数据

复现真实数据实验时，用户需依照原始条款自行获取扩散及解剖影像，完成本项目不包含的预处理，并建立 SimNIBS 4.6 CHARM 头模型。请勿向 issue 上传可识别个人身份或受控访问的医学影像。

每次实验应记录软件版本、DWI 尺寸与壳层、掩膜、张量分量顺序、世界坐标变换、插值阶数、电导率模式、求解器、CPU 数量、输出文件哈希和 QA JSON。
