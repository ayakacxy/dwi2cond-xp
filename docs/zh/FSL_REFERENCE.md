# FSL 参考比较

[English](../en/FSL_REFERENCE.md) | [简体中文](../zh/FSL_REFERENCE.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

本文记录最终 `v0.3.0` 验证使用的 FSL 来源、比较范围和输出要求。FSL 仅是可选验证
依赖，发布软件运行时不导入、不附带，也不调用 FSL。

## 参考源码

兼容性依据是 SimNIBS 4.6.0 安装中的脚本；主脚本声明 `dwi2cond` 版本为 `0.4`。

| 脚本 | SHA-256 |
| --- | --- |
| `dwi2cond` | `0978b26ab1c4d4d303b10151dbe4d0da8ed0c080c3d09af0855412c7b5c46340` |
| `dwi2cond.prepro.source.sh` | `4671e6f72b0dd0da496297e722df3e3682e659dab9c06502c83907fb287f8f78` |
| `dwi2cond.functions.source.sh` | `b133dd7e41ecac773d0af983bacb3f23b62fd2c91f84379ff81df42f89ee5932` |
| `dwi2cond.t1reg.source.sh` | `539d1951621111495792c0957705494fc592579a9c0f3dc807dd6549a904205d` |
| `dwi2cond.check.source.sh` | `29c3c907cd174bc628b3770feafb69831d4f4c76b41c91a97e1cfd938b655541` |

哈希只用于标识本地验证源码，仓库不复制这些文件。首次真实比较还必须记录所用 FSL
的精确构建版本；不同 FSL 版本的输出不能合用一个数值基线。

发布参考环境的安装位置为 `/usr/local/fsl`，版本为 `6.0.4:ddd0a010`。该绝对路径
只是本地配置，公开运行清单应以别名替代。

## 各阶段输出

比较程序使用下列公开别名记录输出。私有清单可另含文件或体素摘要；公开清单不得含
绝对路径、被试标识或体素数组。

| 阶段 | 必需的公开别名 |
| --- | --- |
| 输入标准化 | `dwi_raw`, `bvals`, `bvecs` |
| b0 与掩膜 | `nodif`, `nodif_brain`, `nodif_brain_mask`, `raw_fa`, `raw_sse` |
| `nomoco` 到拟合的数据传递 | `dwi_for_fit`, `fit_bvals`, `fit_bvecs`, `fit_mask` |
| 张量拟合 | `tensor`, `fa`, `sse` |
| T1 配准 | `world_transform` 或 `warp`, `t1_tensor`, `registered_fa`, `registered_v1`, `registered_sse` |
| 最终 QA | `brain_rim`, `first_eigenvector_mesh`, `run_log` |

每个 NIfTI 摘要含形状、dtype、affine、qform、sform、有限值数量、非零数量和有限值
范围。固定的精简清单可用命名 `grid_ref` 代替重复的相同 affine/qform/sform 字段。
掩膜还需记录体素数。即使命令正常结束，缺少必需输出也判为该阶段失败。

## 执行与失败处理

- 每条参考命令在独立进程和明确工作目录中运行。
- 找不到可执行文件时，清单写 `status: skipped`，不计为比较通过。
- 非零退出、超时、缺失输出或输出越出工作目录时，写 `status: failed`，不改用其他模式。
- 清单记录参考版本、脚本哈希、脱敏命令与日志摘要、环境变量名、线程数、实际耗时、
  子进程 CPU 时间，以及平台可提供的子进程峰值 RSS。
- 实现对应 Python 算法前，先在输入清单中固定数值阈值。

初始公开阈值文件为 `tests/fixtures/reference/synthetic_nomoco_manifest.json`。
阈值文件只说明检查要求，不代表已通过；配套参考摘要和最终配置后的真实 FSL 检查才
提供执行证据。

## 已保留的公开记录

`tests/fixtures/reference/` 下保留十一份 JSON：

- `synthetic_nomoco_manifest.json`：比较阈值。
- `synthetic_nomoco_reference.json`：已完成的公开参考结果。
- `synthetic_legacy_reference.json`：已完成的两轮 legacy 输出、逐体积变换、插值、
  b 向量及耗时比较。
- `private_nomoco_reference.json`：私有单壳参考的汇总，不含原数据。
- `synthetic_b0_motion_manifest.json`：刚体运动真值与哈希。
- `synthetic_preprocessing_manifest.json`：反向相位编码、场图、仿射和张量测试定义与哈希。
- `synthetic_fieldmap_reference.json`：固定 GRE/FUGUE 参考。
- `synthetic_topup_reference.json`：固定 TOPUP 参考。
- `synthetic_eddy_reference.json`：固定 EDDY `--repol` 参考。
- `synthetic_eddy_interspersed_b0_reference.json`：可跨平台使用的交错 b0 EDDY 测试定义，
  以及同平台参考的来源。
- `synthetic_t1_registration_reference.json`：自动 T1 配准参考。

测试影像按需生成，不放在公开参考目录。检查这些记录可运行：

```bash
PYTHONPATH=src python tools/audit_reference_assets.py \
  tests/fixtures/reference --forbid 100610 --forbid hcp --forbid cxy
```

检查会拒绝图像文件、绝对路径、直接身份/凭据字段、原始体素数组及配置的私有关键词。
