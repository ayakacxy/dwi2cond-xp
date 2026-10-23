# 发布流程

[English](../en/RELEASE_PROCESS.md) | [简体中文](../zh/RELEASE_PROCESS.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

发布附件通过 GitHub 的 annotated 版本标签提供，标题直接使用版本号，如 `v0.4.0`。目前不发布到 PyPI。

## 准备

1. 整理[更新记录](CHANGELOG.md)中对应版本的内容，同步版本元数据。
2. 同步中英文和旧路径下的完整文档副本，检查示例、本地链接、图片和公式。
3. 按改动范围执行检查。数值修改需要完整发布检查；纯文档修正检查源码不变性和安装包。
4. 构建并检查分发文件：

```bash
python scripts/verify_version.py
python scripts/check_markdown_links.py
python scripts/verify_release.py --repository .
python -m build --outdir .work/package/dist
python -m twine check .work/package/dist/*
check-wheel-contents .work/package/dist/*.whl
python scripts/build_release_metadata.py --dist .work/package/dist
python scripts/verify_release.py --dist .work/package/dist
```

正常代码发布的最终提交应通过 Linux、macOS、Windows、打包和安全检查。完整数值检查为 `python scripts/run_coverage.py`，环境和依赖审计命令见[复现说明](REPRODUCIBILITY.md)。

## 发布

从已验证提交创建 annotated 标签。发布工作流构建 wheel 和源码包，生成 CycloneDX SBOM 与校验文件，签发分发文件证明，并发布对应版本说明。更新记录必须包含该标签的非空章节。

Release 正文使用一种语言，并链接到完整译文；标题使用版本号。发布到其他注册表需要单独配置。

## 修正历史标签

已发布标签通常保持固定。经授权修正时，先记录旧标签对象、提交、Release ID 和附件摘要，并在 `refs/tags/` 以外保留备份引用。

文档修正应保持计算源码、依赖和测试不变，构建包含修正文档的包并验证后再替换附件。使用旧对象 ID 和 `--force-with-lease` 只更新指定标签，保留原附件及说明以便回退。

同版本算法替换需要源码复核、数值回归和完整发布检查，版本说明应描述行为变化，不能作为纯文档修正处理。

## 核对公开版本

下载发布后的 wheel、源码包、SBOM 和校验文件，检查：

- 标签指向预期提交，Release 已公开；
- 下载文件的 SHA-256 与 `SHA256SUMS` 及上传文件一致；
- 包内源码、README 和文档与该提交对应；
- 安装、导入、版本和 CLI 帮助正常；
- 可用来源证明对应这些文件摘要及预期工作流；
- 页面标题及翻译链接正确。

在项目台账记录提交、附件摘要、检查结果和回退位置。重建安装包后需要新的来源证明，旧摘要的证明不适用于替换文件。
