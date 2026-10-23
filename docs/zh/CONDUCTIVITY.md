# 电导率模型

[English](../en/CONDUCTIVITY.md) | [简体中文](../zh/CONDUCTIVITY.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每类组织使用固定标量电导率 |
| `vn` | 逐点归一化行列式，保留张量方向和各向异性 |
| `dir` | 在各向异性组织间拟合一个全局尺度，保留局部方向和强度变化 |
| `mc` | 使用与 `dir` 相同的全局尺度，局部张量为各向同性 |

下列公式使用组织参考电导率，以及采样到目标体素或网格单元的扩散张量。

## 实现公式

下面的公式就是本项目实现并与 SimNIBS 4.6 对照过的映射。对于各向异性组织 $`t`$
中的单元或体素 $`i`$，把修复后的扩散张量写成

$$
\mathbf D_i = \mathbf V_i\mathrm{diag}
(d_{i1},d_{i2},d_{i3})\mathbf V_i^{\mathsf T},
\qquad d_{i1}\ge d_{i2}\ge d_{i3}>0,
$$

其中 $`\mathbf V_i`$ 的列是主方向，$`\sigma_t`$ 是组织 $`t`$ 的参考标量电导率，$`w_i`$
是四面体体积；在不加权的体素计算中取 $`w_i=1`$。

### `vn`：体积归一化各向异性映射

定义局部几何均值 $`g_i=(d_{i1}d_{i2}d_{i3})^{1/3}`$，核心映射为

$$
\boldsymbol\Sigma_i^{\mathrm{vn}}
=\sigma_t\mathbf V_i
\mathrm{diag}\left(
\frac{d_{i1}}{g_i},\frac{d_{i2}}{g_i},\frac{d_{i3}}{g_i}
\right)\mathbf V_i^{\mathsf T},
\qquad
\det\left(\boldsymbol\Sigma_i^{\mathrm{vn}}\right)^{1/3}=\sigma_t.
$$

因此，`vn` 保留特征向量和相对各向异性，同时把每个位置的几何平均电导率设为对应
组织的参考值。这是 Güllmar 等提出并被 SimNIBS 推荐使用的体积归一化映射 [3,4]。

### `dir`：直接缩放各向异性映射

先对每种各向异性组织计算体积加权张量尺度

$$
m_t=
\left(
\frac{\sum_{i\in t}w_i\det(\mathbf D_i)}
     {\sum_{i\in t}w_i}
\right)^{1/3}.
$$

然后在全部各向异性组织之间联合拟合一个全局缩放因子：

$$
s=\underset{a}{\mathrm{argmin}}
\sum_t(am_t-\sigma_t)^2
=\frac{\sum_t\sigma_t m_t}{\sum_t m_t^2},
\qquad
\boldsymbol\Sigma_i^{\mathrm{dir}}=s\mathbf D_i.
$$

该映射保留 DTI 给出的方向、各向异性和空间强度变化。它属于 Tuch 等提出的线性
扩散率到电导率映射，并对应 Rullmann 等和 Opitz 等使用的直接映射 [1,2,4]。

### `mc`：平均电导率对照

`mc` 使用与 `dir` 相同的全局因子 $`s`$，但把每个局部张量替换为具有相同行列式的
各向同性张量：

$$
\boldsymbol\Sigma_i^{\mathrm{mc}}
=\det\left(\boldsymbol\Sigma_i^{\mathrm{dir}}\right)^{1/3}\mathbf I
=s\det(\mathbf D_i)^{1/3}\mathbf I.
$$

所以它保留 DTI 驱动的几何平均电导率空间变化，却移除了方向各向异性。`mc` 是
DTI 派生的对照模式，而不是各向异性张量场 [4]。

## 安全处理与坐标设置

映射应用所配置的特征值、最大电导率和各向异性比例规则。非各向异性组织使用标量参考电导率。VN 依次执行归一化、安全修正、再次归一化和再次安全修正，最后一次触发边界修正时，理想行列式等式可能略有变化。

非零奇异 VN 张量无法定义行列式归一化，默认报错。`tensor-to-mesh --vn-singular-policy regularize` 可显式将其谱投影到配置的各向异性范围，再执行常规安全处理，QA 记录 `regularized_singular_tensors`。此选项只适用于 VN，`dir` 或 `mc` 会拒绝该参数。

启用强度标定时，全零的各向异性 `dir`/`mc` 组织无法拟合全局尺度，会报错。显式 `--no-correct-intensity` 使用不标定的安全路径。

默认 `--eigensystem-mode stable` 使用对称 `eigh`；`simnibs46-literal` 使用一般 `eig`，包含上游重复特征值行为，QA 会记录基底正交性。默认 `--basis-mode physical` 分别归一化空间仿射列并拒绝剪切，其字面对照选项保留上游归一化。

正式 FEM 和先导场 session 通过实例局部适配器应用所选设置，不修改已安装的 SimNIBS 模块。消费格式见[SimNIBS 集成](SIMNIBS_INTEGRATION.md)。

参考文献：

1. Tuch DS, Wedeen VJ, Dale AM, George JS, Belliveau JW. *Conductivity tensor
   mapping of the human brain using diffusion tensor MRI*. PNAS. 2001;
   98(20):11697-11701. [doi:10.1073/pnas.171473898](https://doi.org/10.1073/pnas.171473898)
2. Rullmann M, Anwander A, Dannhauer M, Warfield SK, Duffy FH, Wolters CH.
   *EEG source analysis of epileptiform activity using a 1 mm anisotropic
   hexahedra finite element head model*. NeuroImage. 2009;44(2):399-410.
   [doi:10.1016/j.neuroimage.2008.09.009](https://doi.org/10.1016/j.neuroimage.2008.09.009)
3. Güllmar D, Haueisen J, Reichenbach JR. *Influence of anisotropic electrical
   conductivity in white matter tissue on the EEG/MEG forward and inverse
   solution. A high-resolution whole head simulation study*. NeuroImage.
   2010;51(1):145-163.
   [doi:10.1016/j.neuroimage.2010.02.014](https://doi.org/10.1016/j.neuroimage.2010.02.014)
4. Opitz A, Windhoff M, Heidemann RM, Turner R, Thielscher A. *How the brain
   tissue shapes the electric field induced by transcranial magnetic
   stimulation*. NeuroImage. 2011;58(3):849-859.
   [doi:10.1016/j.neuroimage.2011.06.069](https://doi.org/10.1016/j.neuroimage.2011.06.069)

对应的 SimNIBS 实现级定义见官方
[dwi2cond 文档](https://simnibs.github.io/simnibs/build/html/documentation/command_line/dwi2cond.html)。
