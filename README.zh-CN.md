# dwi2cond-xp

`dwi2cond-xp` 是一个跨平台 Python 流程：从已预处理的 diffusion MRI 或六分量
diffusion tensor 出发，生成 SimNIBS 4.6 电导率张量并运行经过验证的各向异性有限元
仿真。主运行路径不依赖 FSL。

这是独立社区项目，不是 SimNIBS 或 FSL 官方发行物。英文 [README](README.md) 是默认
发布入口，本文件同步最重要的安装、输入和科学边界。

## 支持范围

```text
已预处理单壳 DWI 或六分量 diffusion tensor
  -> 两遍加权最小二乘 DTI 拟合与 QA
  -> 显式映射/重定向到 CHARM T1 网格
  -> scalar / vn / dir / mc 电导率
  -> SimNIBS 4.6 固定 montage FEM
  -> 三分量 E-field NIfTI 与 QA manifest
```

本项目不负责原始 DWI 的 motion、eddy-current、susceptibility/topup/fieldmap 校正，
也不自动估计 6/12 DOF affine 或非线性 DTI→T1 配准。上述步骤及一致的 b-vector
rotation 必须由外部预处理完成。当前未实现 nonlinear PPD tensor reorientation。

纯 Python DTI/tensor 映射核心依赖 NumPy、SciPy、NiBabel、h5py 和 tqdm。Mesh
电导率、FEM 与 lead field 固定要求 `SimNIBS 4.6.0 + Python 3.11`；完整流程的平台
范围受 SimNIBS 4.6.0 可安装和已验证平台限制。

## 电导率模式

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每种组织使用固定标量电导率，不使用 DTI。 |
| `vn` | 保留主方向和各向异性比例，并逐位置归一化行列式；这是主各向异性模式。 |
| `dir` | 保留方向、比例及局部强度变化，再进行总体强度校准。 |
| `mc` | 保留 DTI 驱动的空间平均电导率变化，但局部各向同性；它是强度变化对照。 |

## 安装

首选方式就是直接基于已有 SimNIBS 4.6 环境安装。若该环境需要保持冻结，可用
`--no-deps` 安装 wheel，避免 pip 改动其依赖：

```bash
conda activate simnibs
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
python -m pip install --no-deps dwi2cond_xp-0.1.0-py3-none-any.whl
dwi2cond-xp --help
```

从源码开发时把 wheel 命令换为 `python -m pip install --no-deps -e .`。当前 wheel
已经通过临时 overlay 与现有 SimNIBS 4.6.0 同时导入验证，该验证没有修改参考环境。

如果希望另建独立环境：

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e .
python -c "import simnibs; assert simnibs.__version__ == '4.6.0'"
dwi2cond-xp --help
```

仅安装纯 Python 核心：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## 输入合同

DWI 必须是已预处理的四维 NIfTI，bvals、bvecs、brain mask 必须与其体积顺序一致。
DTI 模型显式使用 b=0 加一个非零壳层，不会静默拟合全部多壳数据。可选 `grad_dev`
使用 HCP/FSL 九分量约定。

Tensor NIfTI 的末维固定为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`。映射到 T1 前，调用者必须
二选一：提供外部估计的 input-world→reference-world 4×4 affine，或在已有外部对齐
证据时显式使用 `--assume-aligned`。同一被试并不自动意味着 DWI 与 T1 已经对齐。

## 最小流程

```bash
dwi2cond-xp fit-dti \
  preprocessed_dwi.nii.gz bvals bvecs brain_mask.nii.gz tensor_dwi.nii.gz \
  --grad-dev grad_dev.nii.gz --workers 8 \
  --valid-mask-out tensor_valid_mask.nii.gz --qa-json tensor_fit_qa.json

dwi2cond-xp register-tensor \
  tensor_dwi.nii.gz m2m_subject/T1.nii.gz \
  m2m_subject/DTI_coregT1_tensor.nii.gz \
  --source-mask tensor_valid_mask.nii.gz --assume-aligned \
  --qa-json tensor_registration_qa.json

dwi2cond-xp simulate-tdcs m2m_subject simulation_outputs \
  --mode vn --anode C3 --cathode C4 --current-ma 1 \
  --shape rect --dimensions 50 50 --thickness 4 \
  --solver pardiso --volume-tissues 1 2 3 --cpus 8 --dry-run
```

移除 `--dry-run` 后才执行求解。`scalar/vn/dir/mc` 分目录保存；各向异性 tensor 缺失
时直接报错，不静默回退 scalar。正式体数据只保留 WM/GM/CSF（标签 1/2/3），严格
排除颅骨、头皮、电极和颅外组织。每种模式只保存一个末维为 `Ex/Ey/Ez` 的向量
E-field NIfTI，模长现场计算。

## 四模式图

分量主图是三行 `Ex/Ey/Ez` × 四列 `scalar/vn/dir/mc` 的 3×4 axial 布局，12 个
panel 共用对称色标；切片只由 brain mask 最大面积决定，不根据场强挑选。

![四模式电场分量](docs/images/electric_field_xyz_3x4.png)

![四模式电场模长](docs/images/electric_field_magnitude_2x2.png)

## 证据边界

私有 HCP 数据已用于 DTI、CHARM 和真实四模式 FEM 验收。原始影像、体数据派生物、
被试标识和机器可读的被试级产物都不进入仓库或 Release；README 只保留两张无被试
标识的结果示意 PNG，并附 HCP 致谢与数据使用条款链接。16 worker DTI 拟合在同一服务器、同一输出
边界下相对 FSL 6.0.4 实测为 `11.09x`；这只是单机 DTI fitting 结果，不能外推到
预处理、配准、建模或 FEM。全电极 lead-field 接口和数据合同已支持并测试，但当前
发布证据不包含真实被试的全电极完整运行。

详细方法、数值误差、复现步骤、贡献和安全规范见 [文档目录](docs/README.md)。项目采用
[GPL-3.0-only](LICENSE)，第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
