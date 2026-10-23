# 安全问题报告

[English](../en/SECURITY.md) | [简体中文](../zh/SECURITY.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

安全修复面向最新发布版本和 `main` 分支。

## 报告漏洞

安全缺陷请通过 [GitHub 私密漏洞报告](https://github.com/ayakacxy/dwi2cond-xp/security/advisories/new)提交，说明受影响版本、复现步骤、影响和建议的缓解措施。尚未修复的漏洞应避免公开披露。

复现示例请使用合成输入。附件中应移除凭据、身份信息和受访问限制的医学数据。

## 范围

软件包读取扩散影像、张量、网格和配置文件，并可调用已安装的 SimNIBS 求解器。项目不上传输入数据，也不收集遥测信息。数值结果异常通常应作为软件缺陷报告；由安全漏洞导致的问题除外。具体渠道见[使用支持](SUPPORT.md)。
