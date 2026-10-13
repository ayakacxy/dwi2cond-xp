# 贡献指南

[English](../en/CONTRIBUTING.md) | [简体中文](../zh/CONTRIBUTING.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

欢迎提交范围明确的 issue 和 pull request。

1. 不要提交人类 MRI、被试标识、凭据、私有绝对路径或生成的 FEM 结果。
2. 保留纯 Python 参考实现。新增加速应可显式选择，请求的后端不能被自动替换。
3. 不得通过降低分辨率、减少数据或迭代、删除科学检查来获得性能提升。
4. 新增注释和 docstring 使用中文；命令行帮助与错误信息保留英文。
5. 运行 `ruff check src tests tools scripts` 及相关测试；发布前或修改共用数值内核、
   覆盖率行为时，运行 `python scripts/run_coverage.py`。
6. 对比参考实现时，说明数值误差和计时所包含的步骤。
7. 修改文档后运行 `python scripts/check_markdown_links.py`、
   `python scripts/verify_version.py` 和 `git diff --check`。
8. 用户可见的变化加入 `docs/CHANGELOG.md` 的 `Unreleased` 节，并同步中英文版本。
   tag 的发布说明从该更新记录生成。

较大改动请先在 issue 中说明要保留或改变的科学行为、可合法共享的输入和验证方法。
