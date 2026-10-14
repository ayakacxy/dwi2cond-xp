"""无设备 CI 执行真实 GPU 宿主和内核数学；真实 CUDA 验收另有整链证据。"""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix, kron

from dwi2cond_xp.preprocessing import fnirt, nonlinear


@pytest.fixture
def cpu_cuda(monkeypatch):
    """仅模拟设备搬运和线程编号，执行原内核 Python 函数。"""

    from dwi2cond_xp.preprocessing import fnirt_cuda

    function = fnirt_cuda.direct_fp64_kernel.py_func
    state = SimpleNamespace(uploads=[], downloads=0, synchronizations=0, entry=0)

    class DeviceArray:
        def __init__(self, array):
            self.array = np.array(array, copy=True)

        def copy_to_host(self):
            state.downloads += 1
            return self.array.copy()

    def upload(array):
        state.uploads.append(np.array(array, copy=True))
        return DeviceArray(array)

    def synchronize():
        state.synchronizations += 1

    class Kernel:
        def __getitem__(self, launch):
            def execute(*arguments):
                arrays = tuple(
                    value.array if isinstance(value, DeviceArray) else value
                    for value in arguments
                )
                # 多执行两个越界线程，覆盖原内核的启动尾部保护。
                for entry in range(arrays[15].size + 2):
                    state.entry = entry
                    function(*arrays)

            return execute

    api = SimpleNamespace(
        grid=lambda dimensions: state.entry,
        is_available=lambda: True,
        to_device=upload,
        device_array=lambda shape, dtype: DeviceArray(np.empty(shape, dtype=dtype)),
        synchronize=synchronize,
    )
    monkeypatch.setattr(fnirt_cuda, "cuda", api)
    monkeypatch.setattr(fnirt_cuda, "direct_fp64_kernel", Kernel())
    monkeypatch.setattr(fnirt_cuda, "libdevice", SimpleNamespace(
        dmul_rn=lambda first, second: np.float64(first) * np.float64(second),
        dadd_rn=lambda first, second: np.float64(first) + np.float64(second),
    ))
    return fnirt_cuda, state


def _independent_basis(shape, spacing):
    """用 Kronecker 乘积独立组装完整基底，避免复用直接支持域内核。"""

    axes = [csr_matrix(fnirt.spline_design_matrix(size, step))
            for size, step in zip(shape, spacing, strict=True)]
    return kron(axes[2], kron(axes[1], axes[0], format="csr"), format="csr")


@pytest.mark.parametrize("blocks,symmetric", [(1, False), (3, True), (6, True)])
def test_cuda_hessian_real_kernel_matches_independent_basis(cpu_cuda, blocks, symmetric):
    module, state = cpu_cuda
    shape = (3, 3, 3)
    left, right = (2, 2, 2), (2, 2, 2) if symmetric else (1, 2, 3)
    selected = np.arange(np.prod(shape)) % 3 != 0
    weights = tuple(np.linspace(-.25 + block, 1.3 + block, selected.sum())
                    for block in range(blocks))
    backend = module.CudaHessianBackend()
    actual = backend(shape, left, right, selected, weights, 1, symmetric=symmetric)
    basis_left = _independent_basis(shape, left)[selected]
    basis_right = _independent_basis(shape, right)[selected]
    dtype = np.float32 if symmetric else np.float64
    for weight, matrix in zip(weights, actual, strict=True):
        expected = basis_left.T @ basis_right.multiply(np.asarray(weight, dtype=dtype)[:, None])
        np.testing.assert_allclose(matrix.toarray(), expected.toarray(), atol=2e-15, rtol=2e-14)
        assert matrix.format == "csc"
    assert len(state.uploads) == 17
    assert state.downloads == 1 and state.synchronizations == 2
    first_geometry = next(iter(backend._workspaces.values()))[0]
    # 掩膜和权重必须更新，几何应保留；全空选择使完整结果精确为零。
    empty = backend(shape, left, right, np.zeros(selected.shape, bool),
                    tuple(np.zeros(0) for _ in weights), 1, symmetric=symmetric)
    assert all(matrix.nnz == 0 for matrix in empty)
    assert next(iter(backend._workspaces.values()))[0] is first_geometry
    assert len(state.uploads) == 20 and state.downloads == 2


def test_cuda_hessian_workspace_eviction_and_layer_change(cpu_cuda):
    module, state = cpu_cuda
    backend = module.CudaHessianBackend()
    shape = (2, 2, 2)
    selected = np.ones(8, bool)
    for spacing in (1, 2, 3, 4):
        result = backend(shape, (spacing,) * 3, (spacing,) * 3,
                         selected, (np.ones(8),), 1)
        assert result[0].nnz > 0
    assert len(backend._workspaces) == 1
    previous = next(iter(backend._workspaces))
    shape = (3, 2, 2)
    result = backend(shape, (2,) * 3, (2,) * 3, np.ones(12, bool), (np.ones(12),), 1)
    assert result[0].nnz > 0 and previous not in backend._workspaces
    assert len(backend._workspaces) == 1 and state.downloads == 5


def test_cuda_hessian_kernel_disjoint_support_is_zero(cpu_cuda):
    module, _ = cpu_cuda
    axis = np.ones((1, 2))
    left_bound, right_bound = np.array([[0, 1]]), np.array([[1, 2]])
    unit_bound = np.array([[0, 1]])
    selection = np.ones(2, bool)
    integral = np.zeros((3, 2, 2), np.int64)
    integral[1:, 1:, 1:] = 1
    for dimension in range(3):
        np.cumsum(integral, axis=dimension, out=integral)
    output = np.full((1, 1), np.nan)
    module.direct_fp64_kernel[1, 1](
        axis, axis, axis, axis, axis, axis,
        left_bound, unit_bound, unit_bound, right_bound, unit_bound, unit_bound,
        selection, integral, np.ones((1, 2)), np.zeros(1, np.int32),
        np.zeros(1, np.int32), 1, 1, 1, 1, 2, 1, output,
    )
    # 两个单体素支持域不交，独立稠密内积亦为零。
    expected = np.array([1., 0.]) @ np.array([0., 1.])
    np.testing.assert_array_equal(output, [[expected]])


@pytest.mark.parametrize("update,message", [
    ({"workers": 0}, "workers"),
    ({"workers": 1.5}, "workers"),
    ({"voxel_weights": ()}, "1, 3 or 6"),
    ({"field_shape": (1, 2)}, "field_shape"),
    ({"left_spacing": (0, 1, 1)}, "left_spacing"),
    ({"right_spacing": (1, 1, 1.5)}, "right_spacing"),
])
def test_cuda_hessian_rejects_invalid_geometry_before_upload(cpu_cuda, update, message):
    module, state = cpu_cuda
    arguments = dict(field_shape=(2, 2, 2), left_spacing=(1,) * 3,
                     right_spacing=(1,) * 3, selected=np.ones(8, bool),
                     voxel_weights=(np.ones(8),), workers=1)
    arguments.update(update)
    with pytest.raises(ValueError, match=message):
        module.CudaHessianBackend()(**arguments)
    assert not state.uploads


def test_cuda_hessian_execution_error_is_not_silently_retried(cpu_cuda, monkeypatch):
    module, state = cpu_cuda

    def fail():
        raise RuntimeError("device execution failed")

    monkeypatch.setattr(module.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="no CPU fallback") as error:
        module.CudaHessianBackend()((2,) * 3, (1,) * 3, (1,) * 3,
                                    np.ones(8, bool), (np.ones(8),), 1)
    assert str(error.value.__cause__) == "device execution failed"
    assert state.downloads == 0


@pytest.fixture
def numpy_ppd(monkeypatch):
    """用 NumPy/SciPy 执行原 PPD 链，并记录可选 API 的上传和回传。"""

    state = SimpleNamespace(uploads=[], downloads=[], synchronizations=0)
    cupy = ModuleType("cupy")
    cupy.__getattr__ = lambda name: getattr(np, name)

    def upload(array, *arguments, **options):
        state.uploads.append(np.asarray(array).copy())
        return np.asarray(array, *arguments, **options)

    def download(array):
        state.downloads.append(np.asarray(array).copy())
        return np.asarray(array).copy()

    def synchronize():
        state.synchronizations += 1

    cupy.asarray = upload
    cupy.asnumpy = download
    cupy.cuda = SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=synchronize))
    cupyx = ModuleType("cupyx")
    cupyx.errstate = lambda **options: nullcontext()
    scipy_module, ndimage = ModuleType("cupyx.scipy"), ModuleType("cupyx.scipy.ndimage")
    ndimage.map_coordinates = map_coordinates
    for name, module in (("cupy", cupy), ("cupyx", cupyx),
                         ("cupyx.scipy", scipy_module), ("cupyx.scipy.ndimage", ndimage)):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(nonlinear.__file__).with_name("ppd_cuda.py")
    specification = importlib.util.spec_from_file_location(
        "dwi2cond_xp.preprocessing.ppd_cuda", path)
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, specification.name, module)
    specification.loader.exec_module(module)
    return module, state


def _ppd_inputs():
    shape = (7, 4, 3)
    tensor = np.broadcast_to([3., .15, .08, 2., .1, 1.], shape + (6,)).astype(np.float32).copy()
    tensor[2] = [1., 0., 0., 1., 0., 1.]
    affine = np.diag([-1.2, .8, 1.6, 1.])
    grid = np.indices(shape, dtype=np.float32)
    warp = np.stack((.04 * np.sin(grid[0]), .03 * np.cos(grid[2]), .02 * grid[1]), axis=-1)
    return tensor, affine, shape, affine, warp


@pytest.mark.parametrize("mode", ["physical", "simnibs46-literal"])
def test_cupy_ppd_full_chain_matches_cpu_with_halo_and_packed_transfer(numpy_ppd, mode):
    module, state = numpy_ppd
    tensor, affine, shape, _, warp = _ppd_inputs()
    source_mask = np.ones(shape, bool)
    source_mask[0, 0] = False
    reference_mask = np.ones(shape, bool)
    reference_mask[-1] = False
    workspace = module.CudaPpdWorkspace(tensor, affine, shape, affine, source_mask=source_mask)
    assert len(state.uploads) == 7
    assert workspace.source_upload_bytes == tensor.nbytes + source_mask.nbytes + 2 * affine.nbytes + 9 * 8 + 6 * 8
    expected = nonlinear.resample_tensor_ppd_fsl(
        tensor, affine, shape, affine, warp, source_mask=source_mask,
        reference_mask=reference_mask, jacobian_mode=mode)
    for start, stop in ((0, 2), (2, 4), (4, 7)):
        actual, profile = workspace.slab(warp, start, stop, reference_mask=reference_mask,
                                         jacobian_mode=mode)
        for name in nonlinear.NonlinearTensorResult.__dataclass_fields__:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name)[start:stop],
                                       rtol=2e-6, atol=2e-7, equal_nan=True)
        expected_halo = warp[max(0, start - 1):min(shape[0], stop + 1)]
        assert profile["slab_upload_bytes"] == expected_halo.nbytes + reference_mask[start:stop].nbytes
        assert profile["numerical_download_calls"] == 1
        assert profile["download_bytes"] == sum(getattr(actual, name).nbytes
                                                  for name in nonlinear.NonlinearTensorResult.__dataclass_fields__) + 1
        assert all(profile[name] >= 0 for name in ("slab_upload_seconds", "compute_seconds", "pack_seconds", "download_seconds", "chain_seconds"))
    assert len(state.downloads) == 3
    assert not any(array.shape == tensor.shape for array in state.uploads[7:])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_cupy_ppd_nonfinite_propagation_and_strict_error(numpy_ppd, bad_value):
    module, state = numpy_ppd
    tensor, affine, shape, _, _ = _ppd_inputs()
    tensor[3, 1, 1, 0] = bad_value
    warp = np.zeros(shape + (3,), np.float32)
    workspace = module.CudaPpdWorkspace(tensor, affine, shape, affine,
                                       source_mask=np.ones(shape, bool))
    with np.errstate(invalid="ignore"):
        actual, profile = workspace.slab(warp, 0, shape[0], compatibility_mode="robust")
        expected = nonlinear.resample_tensor_ppd_fsl(
            tensor, affine, shape, affine, warp, source_mask=np.ones(shape, bool),
            compatibility_mode="robust")
        for name in nonlinear.NonlinearTensorResult.__dataclass_fields__:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                       rtol=2e-6, atol=2e-7, equal_nan=True)
        with pytest.raises(ValueError, match="strict-fsl nonlinear PPD"):
            workspace.slab(warp, 0, shape[0])
    assert np.any(actual.invalid_tensor_mask)
    assert profile["numerical_download_calls"] == 1 and len(state.downloads) == 2


def test_cupy_ppd_zero_spectrum_singular_and_fold_qa(numpy_ppd):
    module, _ = numpy_ppd
    tensor, affine, shape, _, warp = _ppd_inputs()
    tensor[...] = 0
    workspace = module.CudaPpdWorkspace(tensor, affine, shape, affine)
    actual, _ = workspace.slab(np.zeros_like(warp), 0, shape[0])
    assert not np.any(actual.valid_mask) and not np.any(actual.tensor)
    assert np.all(actual.low_fa_mask) and np.all(actual.repeated_eigenvalue_mask)
    tensor[...] = [3., 0., 0., 2., 0., 1.]
    # 单位网格使解析线性位移构造精确奇异和负行列式场。
    unit_affine = np.diag([-1., 1., 1., 1.])
    workspace = module.CudaPpdWorkspace(tensor, unit_affine, shape, unit_affine)
    grid = np.indices(shape, dtype=np.float32)
    for slope in (-1., -2.):
        warp[...] = 0
        warp[..., 0] = slope * grid[0]
        actual, _ = workspace.slab(warp, 0, shape[0], compatibility_mode="robust")
        assert np.all(actual.fold_mask)
        assert np.all(actual.near_singular_mask) == (slope == -1.)
        assert not np.any(actual.tensor) and not np.any(actual.valid_mask)


@pytest.mark.parametrize("case,message", [
    ("tensor", "tensor must have shape"),
    ("grid", "at least two voxels"),
    ("mask", "source mask must match"),
])
def test_cupy_ppd_initial_validation_before_device_upload(numpy_ppd, case, message):
    module, state = numpy_ppd
    tensor, affine, shape, _, _ = _ppd_inputs()
    if case == "tensor":
        tensor = tensor[..., :5]
    elif case == "grid":
        shape = (1, 4, 3)
    options = dict(source_mask=np.zeros((1, 1, 1))) if case == "mask" else {}
    with pytest.raises(ValueError, match=message):
        module.CudaPpdWorkspace(tensor, affine, shape, affine, **options)
    assert not state.uploads


@pytest.mark.parametrize("options,message", [
    ({"compatibility_mode": "bad"}, "Unsupported"),
    ({"jacobian_mode": "bad"}, "Unsupported"),
    ({"start": -1}, "outside"),
    ({"stop": 8}, "outside"),
    ({"singular_tolerance": 0}, "Tolerances"),
    ({"repeated_eigenvalue_tolerance": np.nan}, "Tolerances"),
    ({"low_fa_threshold": -1}, "FA threshold"),
    ({"low_fa_threshold": np.inf}, "FA threshold"),
    ({"bad_warp": "shape"}, "shape mismatch"),
    ({"bad_warp": "nan"}, "must be finite"),
])
def test_cupy_ppd_slab_validation_before_dynamic_upload(numpy_ppd, options, message):
    module, state = numpy_ppd
    tensor, affine, shape, _, warp = _ppd_inputs()
    workspace = module.CudaPpdWorkspace(tensor, affine, shape, affine)
    options = dict(options)
    bad_warp = options.pop("bad_warp", None)
    if bad_warp == "shape":
        warp = warp[:-1]
    elif bad_warp == "nan":
        warp[1, 1, 1] = np.nan
    arguments = dict(start=0, stop=shape[0])
    arguments.update(options)
    with pytest.raises(ValueError, match=message):
        workspace.slab(warp, **arguments)
    assert len(state.uploads) == 7 and not state.downloads


def test_cupy_ppd_solver_failure_propagates_without_download(numpy_ppd, monkeypatch):
    module, state = numpy_ppd
    tensor, affine, shape, _, warp = _ppd_inputs()
    workspace = module.CudaPpdWorkspace(tensor, affine, shape, affine)

    def fail(values):
        raise np.linalg.LinAlgError("device eigensolver failed")

    # 替换 API 入口，不影响独立 NumPy 参考及其他测试的线性代数。
    monkeypatch.setattr(module.cp, "linalg", SimpleNamespace(
        det=np.linalg.det, inv=np.linalg.inv, norm=np.linalg.norm, eigh=fail,
    ), raising=False)
    with pytest.raises(np.linalg.LinAlgError, match="device eigensolver failed"):
        workspace.slab(warp, 0, shape[0], compatibility_mode="robust")
    assert not state.downloads
