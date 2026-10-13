# 电导率模型

[English](../en/CONDUCTIVITY.md) | [简体中文](../zh/CONDUCTIVITY.md)

[文档目录](README.md) · [项目首页](../../README.zh-CN.md)

| 模式 | 含义 |
| --- | --- |
| `scalar` | 每种组织使用固定标量电导率，不使用 DTI。 |
| `vn` | 保留主方向和各向异性比例，并逐位置归一化行列式；这是主各向异性模式。 |
| `dir` | 保留方向、比例及局部强度变化，再进行总体强度校准。 |
| `mc` | 保留 DTI 驱动的空间平均电导率变化，但局部各向同性；它是强度变化对照。 |

## 实现公式

下面的公式就是本项目实现并与 SimNIBS 4.6 对照过的映射。对于各向异性组织 $t$
中的单元或体素 $i$，把修复后的扩散张量写成

$$
\mathbf D_i = \mathbf V_i\mathrm{diag}
(d_{i1},d_{i2},d_{i3})\mathbf V_i^{\mathsf T},
\qquad d_{i1}\ge d_{i2}\ge d_{i3}>0,
$$

其中 $\mathbf V_i$ 的列是主方向，$\sigma_t$ 是组织 $t$ 的参考标量电导率，$w_i$
是四面体体积；在不加权的体素计算中取 $w_i=1$。

### `vn`：体积归一化各向异性映射

定义局部几何均值 $g_i=(d_{i1}d_{i2}d_{i3})^{1/3}$，核心映射为

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

`mc` 使用与 `dir` 相同的全局因子 $s$，但把每个局部张量替换为具有相同行列式的
各向同性张量：

$$
\boldsymbol\Sigma_i^{\mathrm{mc}}
=\det\left(\boldsymbol\Sigma_i^{\mathrm{dir}}\right)^{1/3}\mathbf I
=s\det(\mathbf D_i)^{1/3}\mathbf I.
$$

所以它保留 DTI 驱动的几何平均电导率空间变化，却移除了方向各向异性。`mc` 是
DTI 派生的对照模式，而不是各向异性张量场 [4]。

三种映射会对其支持的输入执行已有 SimNIBS 兼容特征值、最大电导率和各向异性比规则，
但退化输入按模式分别处理，而不是统一“自动修复”：非零秩亏 `vn` 张量默认报错；
默认 `correct_intensity=True` 时，全零各向异性 `dir/mc` 组织因全局尺度无定义而被拒绝；
显式指定 `--no-correct-intensity` 时改走非校准安全路径，并接受该输入。`vn` 实际执行
“归一化→安全修正→再次归一化→再次安全修正”；如果最后的边界修正被触发，上式中
理想的行列式等式可能出现轻微偏移。不参与各向异性的组织统一使用
$\boldsymbol\Sigma_i=\sigma_t\mathbf I$。

非零秩亏张量没有定义良好的 VN 行列式归一化。默认策略会明确报错，不再产生 NaN 或
静默把缩放因子替换成 1。`tensor-to-mesh --vn-singular-policy regularize` 是显式的
数值稳定扩展：先把奇异特征系统投影到配置的最大各向异性比，再执行既有 SimNIBS
安全修正，并在 QA 中记录修复单元数。该扩展不会被描述成 SimNIBS 4.6 秩亏分支的逐句
等价实现。

独立转换命令默认使用 `--eigensystem-mode stable`，即对称 `eigh` 求解器；
重复特征值张量仍保持正交、满秩基底。显式指定
`--eigensystem-mode simnibs46-literal` 时，才逐句复现 SimNIBS 4.6 的通用 `eig`
排序及重复特征值下可能产生非正交基底的行为；QA JSON 会记录模式和基底正交性诊断。FEM
流程仍把电导率构造交给已安装的 SimNIBS SESSION，不会自动改用独立转换命令。

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
