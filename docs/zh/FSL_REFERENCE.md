# SimNIBS 4.6 FSL 参考实现

[English](../en/FSL_REFERENCE.md) | [简体中文](../zh/FSL_REFERENCE.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

本页记录 `v0.2.0` 原始 DWI 开发时采用的本地参考实现。FSL 是可选验证工具，发布的运行流程不会导入、打包或调用 FSL。这里的清单只对应历史 v0.2 源码；后续版本有各自的参考数据和流程修正。

## 脚本来源

兼容性参照为 SimNIBS 4.6.0 安装目录中的脚本，主脚本声明的 `dwi2cond` 版本是 `0.4`。

| 脚本 | SHA-256 |
| --- | --- |
| `dwi2cond` | `0978b26ab1c4d4d303b10151dbe4d0da8ed0c080c3d09af0855412c7b5c46340` |
| `dwi2cond.prepro.source.sh` | `4671e6f72b0dd0da496297e722df3e3682e659dab9c06502c83907fb287f8f78` |
| `dwi2cond.functions.source.sh` | `b133dd7e41ecac773d0af983bacb3f23b62fd2c91f84379ff81df42f89ee5932` |
| `dwi2cond.t1reg.source.sh` | `539d1951621111495792c0957705494fc592579a9c0f3dc807dd6549a904205d` |
| `dwi2cond.check.source.sh` | `29c3c907cd174bc628b3770feafb69831d4f4c76b41c91a97e1cfd938b655541` |

哈希仅用于标识本地验证来源，脚本没有复制到本仓库。首次真实参考实验还须记录生成输出所用的准确 FSL 构建版本；不同 FSL 版本的输出不可混作同一数值基线。

当时核验的本地可执行文件是 `/usr/local/fsl/bin/dtifit`，安装版本报告为 FSL `6.0.4:ddd0a010`。这个绝对路径仅是本地配置，公开运行记录中应以别名替代。

## 各阶段输出

比较工具使用以下公开别名记录输出。私有记录可以额外保存文件或体素摘要；公开记录不能包含绝对路径、被试标识或体素数组。

| 阶段 | 必需的公开别名 |
| --- | --- |
| 输入规范化 | `dwi_raw`, `bvals`, `bvecs` |
| b0 与掩膜 | `nodif`, `nodif_brain`, `nodif_brain_mask`, `raw_fa`, `raw_sse` |
| `nomoco` 拟合输入 | `dwi_for_fit`, `fit_bvals`, `fit_bvecs`, `fit_mask` |
| 张量拟合 | `tensor`, `fa`, `sse` |
| T1 配准 | `world_transform` 或 `warp`, `t1_tensor`, `registered_fa`, `registered_v1`, `registered_sse` |
| 最终 QA | `brain_rim`, `first_eigenvector_mesh`, `run_log` |

每个 NIfTI 摘要记录形状、数据类型、affine、qform、sform、有限值数量、非零值数量和有限值范围。固定的精简记录可用命名 `grid_ref` 代替重复且相同的 affine/qform/sform。掩膜另记录体素数。即使命令已结束，缺少必需输出也判定该阶段失败。

## 运行与失败处理

- 每条参考命令在独立进程和明确指定的工作目录中运行。
- 缺少可执行文件时写入 `status: skipped`，不能视为比较通过。
- 非零退出码、超时、缺少输出或输出写到工作目录之外时，写入 `status: failed`，不自动选择其他模式。
- 运行记录包含参考版本、脚本哈希、脱敏命令与日志摘要、环境变量名称、线程数、实际耗时、子进程 CPU 时间，以及平台支持时的子进程峰值 RSS。
- 开始实现对应 Python 算法前，先固定样例的数值阈值。

初始公开阈值文件是 `tests/fixtures/reference/synthetic_nomoco_manifest.json`。它定义比较标准，不表示完整 SimNIBS `nomoco` 流程已经通过验证。

## 固定的公开参考材料

v0.2 的参考材料为 `tests/fixtures/reference/` 下的十份 JSON：

- `synthetic_nomoco_manifest.json`：比较阈值。
- `synthetic_nomoco_reference.json`：已完成的公开参考运行。
- `synthetic_legacy_reference.json`：已完成的两轮 legacy 输出、逐体积变换、插值、b 向量和耗时比较。
- `private_nomoco_reference.json`：已完成的私有单壳参考运行，仅含汇总统计。
- `synthetic_b0_motion_manifest.json`：刚体运动真值和哈希。
- `synthetic_preprocessing_manifest.json`：反向相位编码、场图、仿射及张量样例定义和哈希。
- `synthetic_fieldmap_reference.json`：固定 GRE/FUGUE 参考结果。
- `synthetic_topup_reference.json`：固定 TOPUP 参考结果。
- `synthetic_eddy_reference.json`：固定 EDDY `--repol` 参考结果。
- `synthetic_t1_registration_reference.json`：自动 T1 配准参考结果。

这十份文件只支持所列 v0.2 样例和阈值，不能据此推断任意 FSL 配置均已验证，也不能说明 `v0.3.0` 才引入的流程或数值修正已存在于本版本。

影像样例按需生成，不存放在公开参考目录中。可用以下命令审计全部固定材料：

```bash
PYTHONPATH=src python tools/audit_reference_assets.py \
  tests/fixtures/reference --forbid 100610 --forbid hcp --forbid cxy
```

审计会拒绝影像文件、绝对路径、直接标识符或凭据字段、原始体素数组，以及配置的私有关键词。
