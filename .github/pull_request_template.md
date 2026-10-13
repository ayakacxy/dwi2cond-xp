## Summary / 改动说明

Describe the change users will see and its effect on the scientific method.
说明用户可见的变化及其对科学方法的影响。

## Validation / 验证

- [ ] `ruff check src tests tools scripts`
- [ ] `pytest -q --cov=dwi2cond_xp --cov-report=term-missing --cov-fail-under=100`
- [ ] Add or update numerical reference comparisons where needed. / 必要时新增或更新数值参考对照。
- [ ] Record hardware and measured steps for performance claims. / 性能结论注明硬件及计时包含的步骤。

## Data and privacy / 数据与隐私

- [ ] Exclude human MRI, subject identifiers, credentials, private paths, and generated subject outputs. / 不包含人体 MRI、被试标识、凭据、私有路径或生成的被试输出。
- [ ] Document the origin and license of copied or derived material. / 说明复制或衍生材料的来源和许可证。

## Methods and numerical behavior / 方法与数值行为

State whether the change affects resolution, iterations, stopping conditions,
tensor ordering, coordinate transforms, floating-point behavior, or error handling.
说明是否改变分辨率、迭代次数、停止条件、张量排列、坐标变换、浮点计算或错误处理。
