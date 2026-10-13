# 发布流程

[English](../en/RELEASE_PROCESS.md) | [简体中文](../zh/RELEASE_PROCESS.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

稳定版本使用附注标签（annotated tag）和 GitHub Release。正常开发中不修改已有标签。只有在获得明确授权、且不改变算法源码的文档或发布元数据修正中，才可重新发布历史标签；同时必须在备份引用和项目台账中保存原始标签对象 ID 与最终指向的提交 ID，并重新构建全部发布资产。`main` 保持为当前开发分支；旧版本只有在需要维护时才创建临时回移分支。

明确授权的同版本正确性修复属于另一种例外：只有完成源码重新审计、有区分力的数值测试、完整发布检查、标签和 Release 的精确回滚记录，以及最终远端 CI 与安全验证后，才可修改算法源码。更新记录和公开说明必须明确写明该版本已被替换，不能称为仅修改元数据。

## 准备

1. 将[更新记录](CHANGELOG.md)中 `Unreleased` 下用户可见的变化移入带日期的版本章节。
2. 同步 `pyproject.toml`、`CITATION.cff`、软件包元数据、README 示例和文档中的版本号。
3. 在干净的 Python 3.11 环境中安装发布、测试和可视化额外依赖。开发过程中运行针对性测试，发布前运行以下检查：

   ```bash
   python -m pip install -e '.[test,release,viz]'
   ruff check src tests tools scripts
   python scripts/run_coverage.py
   python scripts/verify_version.py
   python scripts/check_markdown_links.py
   python scripts/verify_release.py --repository .
   validate-pyproject pyproject.toml
   check-manifest --verbose
   python -m pip_audit . --skip-editable --progress-spinner=off
   python -m build
   python -m twine check dist/*
   check-wheel-contents dist/*.whl
   python scripts/build_release_metadata.py --dist dist
   python scripts/verify_release.py --dist dist
   ```

4. 确认 Git 跟踪文件和发布包中不含人体 MRI、被试派生数据、本机路径、凭据、FSL 二进制程序、构建输出或内部台账。
5. 打标签前，要求最终 `main` 提交通过 Linux、macOS arm64、Windows、软件包、CodeQL 和 OpenSSF 工作流。

## 发布

创建并推送指向已验证 `main` 提交的附注标签 `vX.Y.Z`。标签工作流会重新运行完整测试及软件包检查，从更新记录提取对应版本章节，构建 wheel 与 sdist，生成 CycloneDX SBOM 和 `SHA256SUMS`，为构建来源与 SBOM 生成证明，并发布 GitHub Release。若找不到匹配且非空的更新记录章节，工作流会在发布产物前失败。

当前工作流不发布到 PyPI。接入其他软件包仓库属于单独的发布改动，需另行审计可信发布配置和安装过程。

## 修正历史标签

移动已有标签之前，在项目台账记录其标签对象 ID、最终指向的提交 ID、Release ID、资产名称和资产哈希。在 `refs/tags/` 之外创建本地备份引用，然后从原始提交出发，只做针对该版本所需的最小改动。不要将后续版本的算法或功能描述复制到旧版本。

对每份修正后的工作树运行版本、链接、打包、归档隐私及针对性测试检查。重新创建附注标签，仅强制更新对应的远端标签，并由标签工作流重新构建 wheel、sdist、SBOM、校验和文件及相关证明。如果 GitHub Release 已存在，工作流原位替换同名资产，并保留人工整理的发布说明；除非已另行明确授权修正发布说明。最后，下载替换后的资产，执行与新版本发布相同的验证。只有新对象 ID 与回滚对象 ID 均被记录后，修正才算完成。

对于已授权的同版本正确性修复，采用相同的精确备份和 `force-with-lease` 方式，但从当前 `main` 分支创建工作分支。针对修改后的源码运行完整跨平台检查和科学验证，在版本章节记录修正的算法及保留实验结果的适用范围，并确保替换标签指向新验证过的 `main` 提交。

## 验证公开发布

将公开 Release 的全部资产下载到新目录，然后：

- 确认标签最终指向预期提交，Release 既不是草稿，也不是预发布版本；
- 运行 `sha256sum -c SHA256SUMS`；
- 对下载文件运行 Twine、wheel 内容和归档隐私检查；
- 将下载的 wheel 安装到干净环境，检查导入、软件包版本、命令行帮助及依赖一致性；
- 验证 wheel 与 sdist 的 GitHub 构建来源证明；
- 检查渲染后的发布说明及文档链接。

在项目验证台账中记录工作流运行、标签提交、资产哈希，以及任何有意跳过的外部集成检查。
