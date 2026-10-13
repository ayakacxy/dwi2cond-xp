# 参与贡献

[English](../en/CONTRIBUTING.md) | [简体中文](../zh/CONTRIBUTING.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

欢迎通过具体的 issue 和 pull request 参与项目。

1. 请勿提交人体 MRI、被试标识、凭据、本机绝对路径或生成的 FEM 输出。
2. 保留纯 Python 参考实现。新增加速方式应由用户明确选择，不应擅自替换指定后端，或在失败时静默切换后端。
3. 性能比较不得通过降低分辨率、减少数据或迭代次数、跳过科学检查来获得加速。
4. 新增代码注释和 docstring 使用中文；命令行帮助和错误信息保留英文。
5. 运行 `ruff check src tests tools scripts` 和 `pytest -q`。
6. 与参考实现比较时，报告数值误差，并说明计时包含哪些步骤。

较大的改动应先创建 issue，说明拟采用的方法、预期输入与输出、可合法共享的输入数据，以及验证计划。
