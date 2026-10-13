# 参与贡献

[English](../en/CONTRIBUTING.md) | [简体中文](../zh/CONTRIBUTING.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

欢迎通过主题明确的 issue 和 pull request 参与项目。

1. 不提交人体 MRI、被试标识、凭据、本地绝对路径或生成的 FEM 结果。
2. 保留纯 Python 参考路径。新增加速方式必须明确选择，不得在未告知用户时替换指定后端或改用其他后端。
3. 性能改进不得以降低分辨率、减少数据或迭代、删减科学检查为代价。
4. 新增注释和 docstring 使用中文；CLI 帮助和报错保持英文。
5. 运行 `ruff check src tests tools scripts` 和 `pytest -q`。
6. 与参考实现比较时，同时报告数值误差，以及计时包含和排除的步骤。

较大的修改请先开 issue，说明要解决的科学问题、可合法分享的输入数据和验证方案。
