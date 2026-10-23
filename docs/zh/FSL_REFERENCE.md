# FSL 参考比较

[English](../en/FSL_REFERENCE.md) | [简体中文](../zh/FSL_REFERENCE.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

FSL 6.0.4 是固定预处理和配准路径的可选数值参考，需单独安装，不是 dwi2cond-xp 的运行依赖。

## 源码来源

参考脚本来自 SimNIBS 4.6.0，其主脚本声明 dwi2cond 版本为 `0.4`；记录的 FSL 构建为 `6.0.4:ddd0a010`。下列摘要标识既有对照使用的参考源码：

| 脚本 | SHA-256 |
| --- | --- |
| `dwi2cond` | `0978b26ab1c4d4d303b10151dbe4d0da8ed0c080c3d09af0855412c7b5c46340` |
| `dwi2cond.prepro.source.sh` | `4671e6f72b0dd0da496297e722df3e3682e659dab9c06502c83907fb287f8f78` |
| `dwi2cond.functions.source.sh` | `b133dd7e41ecac773d0af983bacb3f23b62fd2c91f84379ff81df42f89ee5932` |
| `dwi2cond.t1reg.source.sh` | `539d1951621111495792c0957705494fc592579a9c0f3dc807dd6549a904205d` |
| `dwi2cond.check.source.sh` | `29c3c907cd174bc628b3770feafb69831d4f4c76b41c91a97e1cfd938b655541` |

复现时使用匹配构建的本地安装，其他 FSL 版本的输出应建立独立参考记录。

## 阶段产物

对照工具通过以下别名标识输入和结果：

| 阶段 | 公开别名 |
| --- | --- |
| 输入规范化 | `dwi_raw`, `bvals`, `bvecs` |
| b0 与掩膜 | `nodif`, `nodif_brain`, `nodif_brain_mask`, `raw_fa`, `raw_sse` |
| `nomoco` 拟合输入 | `dwi_for_fit`, `fit_bvals`, `fit_bvecs`, `fit_mask` |
| 张量拟合 | `tensor`, `fa`, `sse` |
| T1 配准 | `world_transform` 或 `warp`, `t1_tensor`, `registered_fa`, `registered_v1`, `registered_sse` |
| 最终 QA | `brain_rim`, `first_eigenvector_mesh`, `run_log` |

NIfTI 摘要记录形状、dtype、affine、qform、sform、有限/非零值数量和有限值范围。重复的相同几何可用命名 `grid_ref` 代替；掩膜另记体素数量。

## 执行与记录

每条命令在独立进程和工作目录运行。工具缺失记为 `status: skipped`；非零退出、超时或缺少必要输出记为 `status: failed`。记录包含工具版本、源码摘要、脱敏命令、线程数、耗时和资源使用。数值阈值在评估候选输出前确定。

## 公开参考记录

`tests/fixtures/reference/` 保存合成定义、阈值和汇总参考结果：

- `synthetic_nomoco_manifest.json` 和 `synthetic_nomoco_reference.json`；
- `synthetic_legacy_reference.json`；
- `synthetic_b0_motion_manifest.json` 和 `synthetic_preprocessing_manifest.json`；
- GRE、TOPUP 和 EDDY 的参考 JSON；
- `synthetic_eddy_interspersed_b0_reference.json`；
- `synthetic_t1_registration_reference.json`；
- 仅含汇总统计的 `private_nomoco_reference.json`。

图像示例按需生成。公开记录不包含原始体素数组、凭据或可识别被试数据。可通过以下命令检查参考目录：

```bash
PYTHONPATH=src python tools/audit_reference_assets.py tests/fixtures/reference
```

实测一致性与当前限制见[验证记录](VALIDATION.md)。
