# 贡献指南

[English](../en/CONTRIBUTING.md) | [简体中文](../zh/CONTRIBUTING.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

欢迎通过 issue 和 Pull Request 参与项目。较大的算法或接口修改，请先在 issue 中讨论实现方案。

## 开发环境

按[复现说明](REPRODUCIBILITY.md)配置 Python 3.11 和所需依赖，再安装开发工具：

```bash
python -m pip install -e '.[test]'
```

SimNIBS 集成需要 4.6.0；FSL 仅用于可选的本地数值对照。

## 修改约定

每个 Pull Request 聚焦一个改动。英文和中文指南同步更新，`docs/` 下保留的完整兼容副本也要同步。公开源码注释、docstring、CLI 帮助和错误信息使用英文。

数值修改应说明影响哪个处理阶段，使用相同输入、参数和求解器设置进行比较，报告坐标约定、张量差异及相关 QA 结果。性能报告需说明硬件、线程数、缓存状态和计时范围，具体方法见[算法说明](METHODS.md)和[性能记录](BENCHMARKS.md)。

不要提交患者数据、凭据、缓存或生成的仿真结果。改编代码应保留来源与许可说明。

## 检查方法

按改动范围选择检查。文档修改运行：

```bash
python scripts/check_markdown_links.py
python scripts/verify_version.py
git diff --check
```

代码修改运行 Ruff 和相关测试；共享数值内核的修改还需要执行 `python scripts/run_coverage.py`。硬件相关修改应在对应设备上运行，无法执行的检查请在提交中注明。

Pull Request 应说明问题、修改后的行为和检查方法，并在[更新记录](CHANGELOG.md)中补充用户可见的变化。
