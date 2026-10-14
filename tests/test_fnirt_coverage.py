"""FNIRT 稀疏备用路径与输入边界，使用独立稠密数学参考。"""

import builtins

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix, kron

from dwi2cond_xp.preprocessing import fnirt


def _basis(shape, spacing):
    """用 Kronecker 乘积独立构造体素最快的完整样条基底。"""

    axes = [csr_matrix(fnirt.spline_design_matrix(size, step))
            for size, step in zip(shape, spacing, strict=True)]
    return kron(axes[2], kron(axes[1], axes[0], format="csr"), format="csr")


def test_masked_sparse_jtj_uses_fortran_voxel_order():
    shape = (2, 3, 2)
    rng = np.random.default_rng(604)
    left, right = rng.normal(size=(12, 3)), rng.normal(size=(12, 2))
    weights = np.arange(12, dtype=float).reshape(shape)
    mask = np.indices(shape).sum(axis=0) % 2 == 0
    flattened = mask.reshape(-1, order="F")
    expected = left[flattened].T @ (
        weights.reshape(-1, order="F")[flattened, None] * right[flattened])
    actual = fnirt._masked_sparse_jtj(csr_matrix(left), csr_matrix(right), weights, mask)
    np.testing.assert_allclose(actual.toarray(), expected, atol=1e-13)


def test_parallel_sparse_cross_blocks_preserve_weighted_products(monkeypatch):
    monkeypatch.setattr(fnirt, "_sparse_cross_column_costs",
                        fnirt._sparse_cross_column_costs.py_func)
    left = np.array([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.], [0., 6., 7.]])
    right = np.array([[0., 2., 1., 0.], [3., 0., 0., 4.],
                      [0., 5., 6., 0.], [7., 0., 0., 8.]])
    weights = (np.array([.2, -.5, .8, 1.2]), np.array([2., 1., 0., -1.]))
    sparse_left, sparse_right = csr_matrix(left), csr_matrix(right)
    assert fnirt._selected_sparse_jtj_blocks(sparse_left, sparse_right, (), 2) == ()
    for workers in (1, 3):
        actual = fnirt._selected_sparse_jtj_blocks(sparse_left, sparse_right, weights, workers)
        for weight, block in zip(weights, actual, strict=True):
            np.testing.assert_allclose(block.toarray(), left.T @ (weight[:, None] * right), atol=1e-14)
    with pytest.raises(ValueError, match="basis rows"):
        fnirt._selected_sparse_jtj(sparse_left, sparse_right, np.ones(3))
    empty = fnirt._selected_sparse_jtj_blocks(
        csr_matrix((4, 3)), csr_matrix((4, 4)), weights, 3)
    assert all(block.nnz == 0 for block in empty)


@pytest.mark.parametrize("count", [1, 2, 6])
def test_symmetric_sparse_support_kernel_matches_dense_products(monkeypatch, count):
    monkeypatch.setattr(fnirt, "_symmetric_spline_jtj_values_fsl_order",
                        fnirt._symmetric_spline_jtj_values_fsl_order.py_func)
    monkeypatch.setattr(fnirt, "_spline_support_lower_bound",
                        fnirt._spline_support_lower_bound.py_func)
    # 列支持域含空洞和交错首位置，检查二分定位及逐体素交集推进。
    dense = np.array([[0., 2., 0.], [1., 0., 0.], [0., 0., 3.],
                      [4., 5., 0.], [0., 6., 7.], [8., 0., 9.]])
    weights = tuple(np.linspace(-.3 + index, 1.1 + index, dense.shape[0])
                    for index in range(count))
    actual = fnirt._symmetric_spline_jtj_blocks(csr_matrix(dense), weights, (3, 1, 1), 2)
    for weight, block in zip(weights, actual, strict=True):
        expected = dense.T @ (np.asarray(weight, np.float32)[:, None] * dense)
        np.testing.assert_allclose(block.toarray(), expected, atol=1e-13, rtol=1e-14)
        np.testing.assert_array_equal(block.toarray(), block.T.toarray())
    # 空列和完整零基底都必须输出精确零，无支持域越界读。
    empty = fnirt._symmetric_spline_jtj_blocks(csr_matrix(dense.shape), weights, (3, 1, 1), 2)
    assert all(block.nnz == 0 for block in empty)


@pytest.mark.parametrize("options,message", [
    ({"workers": 0}, "workers"),
    ({"workers": 1.5}, "workers"),
    ({"symmetric": True, "right_spacing": (1, 2, 2)}, "matching knot spacing"),
    ({"selected": np.ones(7, bool)}, "selected mask"),
    ({"voxel_weights": (np.ones(7),)}, "basis rows"),
])
def test_direct_spline_products_validate_selected_geometry(options, message):
    arguments = dict(field_shape=(2, 2, 2), left_spacing=(2, 2, 2),
                     right_spacing=(2, 2, 2), selected=np.ones(8, bool),
                     voxel_weights=(np.ones(8),), workers=2)
    arguments.update(options)
    with pytest.raises(ValueError, match=message):
        fnirt._direct_spline_jtj_blocks(**arguments)
    arguments.update(voxel_weights=())
    assert fnirt._direct_spline_jtj_blocks(**arguments) == ()


@pytest.mark.parametrize("bad_selected,bad_values", [(True, False), (False, True)])
def test_spline_projection_rejects_inconsistent_fields(bad_selected, bad_values):
    shape = (2, 3, 2)
    selected = np.ones(11 if bad_selected else 12, bool)
    values = (np.ones((2, 2, 2)),) if bad_values else (np.ones(shape),)
    with pytest.raises(ValueError, match="match"):
        fnirt._spline_projection_many(values, selected, shape, (2, 2, 2), workers=2)
    with pytest.raises(ValueError, match="field shape"):
        fnirt._spline_projection_many((), np.ones(12, bool), shape, (2, 2, 2), workers=2)


def test_hessian_single_worker_and_explicit_basis_match_independent_products(monkeypatch):
    monkeypatch.setattr(fnirt, "_symmetric_spline_jtj_values_fsl_order",
                        fnirt._symmetric_spline_jtj_values_fsl_order.py_func)
    monkeypatch.setattr(fnirt, "_spline_support_lower_bound",
                        fnirt._spline_support_lower_bound.py_func)
    shape, spacing = (3, 2, 2), (2, 2, 2)
    reference = np.ones(shape, np.float32)
    mask = np.ones(shape, bool)
    mask[0, 0, 0] = False
    level_images = fnirt.FnirtLevelImages(reference, reference, mask, mask, (1., 1., 1.))
    derivatives = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0) + .25
    warped = fnirt.FnirtWarpedMoving(reference, np.indices(shape), derivatives, mask, mask, mask)
    cost = fnirt.FnirtCostEvaluation(0., 0., 0., 0., int(mask.sum()), warped, reference)
    gradient = fnirt.FnirtGradientEvaluation(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), cost)
    level = fnirt.FnirtLevel(1, 1, 0., 0., 0., False)
    basis = _basis(shape, spacing)
    coefficient_shape = fnirt.fsl_coefficient_shape(shape, spacing)
    mapping = fnirt.FnirtIntensityMapping(np.array([0., 1.]), np.zeros(coefficient_shape),
                                         reference, spacing)
    selected = mask.reshape(-1, order="F")
    selected_basis = basis.toarray()[selected]
    blocks = []
    for first in range(3):
        row = []
        for second in range(3):
            weight = np.multiply(derivatives[..., first], derivatives[..., second], dtype=np.float32)
            row.append(2 / mask.sum() * selected_basis.T @ (
                weight.reshape(-1, order="F")[selected, None] * selected_basis))
        blocks.append(row)
    expected = np.block(blocks)
    builder_calls = []

    def reference_builder(field_shape, left_spacing, right_spacing, selection, weights,
                          workers, *, symmetric=False):
        assert field_shape == shape and left_spacing == right_spacing == spacing
        assert symmetric and workers == 2
        np.testing.assert_array_equal(selection, selected)
        builder_calls.append(len(weights))
        # 外部后端接口仍执行真实稀疏核，结果由上方独立稠密公式核验。
        return fnirt._symmetric_spline_jtj_blocks(basis[selection], weights,
                                                 coefficient_shape, workers)

    for workers, builder in ((1, None), (2, None), (2, reference_builder)):
        backend = ({"_displacement_basis": basis} if builder is None
                   else {"_hessian_builder": builder})
        actual = fnirt.evaluate_fnirt_hessian(
            level_images, np.eye(4), np.eye(4), np.eye(4), mapping, level,
            workers=workers, displacement_knot_spacing=spacing,
            _gradient_evaluation=gradient,
            _displacement_regularization=csc_matrix((basis.shape[1],) * 2), **backend)
        np.testing.assert_allclose(actual.hessian.toarray(), expected, atol=2e-15, rtol=2e-14)
        assert actual.intensity_hessian.shape == (0, 0)
        assert actual.gradient is gradient
    assert builder_calls == [6]


def test_fnirt_warp_without_derivatives_preserves_identity_values(monkeypatch):
    monkeypatch.setattr(fnirt, "_sample_trilinear_value_fsl", fnirt._sample_trilinear_value_fsl.py_func)
    values = np.arange(24, dtype=np.float32).reshape((4, 3, 2))
    moving_mask = np.ones(values.shape, np.float32)
    reference_mask = np.ones(values.shape, bool)
    reference_mask[1, 1, 1] = False
    actual = fnirt._warp_fnirt_fsl_order.py_func(
        values, moving_mask, reference_mask, np.eye(4, dtype=np.float32),
        np.eye(4, dtype=np.float32), np.zeros(values.shape + (3,), np.float32), False)
    np.testing.assert_array_equal(actual[0], values)
    np.testing.assert_array_equal(actual[1], np.indices(values.shape))
    assert not np.any(actual[2])
    assert np.all(actual[3]) and np.all(actual[4])
    np.testing.assert_array_equal(actual[5], reference_mask)


def test_lazy_cuda_dependency_failure_is_explicit(monkeypatch):
    original_import = builtins.__import__

    def without_cuda(name, *arguments, **options):
        if name.endswith("fnirt_cuda"):
            raise ImportError("CUDA dependency is missing")
        return original_import(name, *arguments, **options)

    monkeypatch.setattr(builtins, "__import__", without_cuda)
    builder = fnirt._make_cuda_hessian_builder()
    with pytest.raises(RuntimeError, match="dependencies are unavailable; no CPU fallback") as error:
        builder()
    assert isinstance(error.value.__cause__, ImportError)


def test_internal_fnirt_runner_rejects_invalid_worker_budget():
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        fnirt._run_simnibs46_fnirt(np.ones((2, 2, 2)), np.ones((2, 2, 2)),
                                  np.eye(4), np.eye(4), np.eye(4), workers=0)
