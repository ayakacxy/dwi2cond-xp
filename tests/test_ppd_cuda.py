"""可选 PPD 后端的导入与失败边界；真实 GPU 数值比较另有实测记录。"""

import builtins
import sys
from types import ModuleType

import numpy as np
import pytest

from dwi2cond_xp.preprocessing import nonlinear


def _inputs():
    shape = (3, 4, 5)
    tensor = np.broadcast_to([3, 0, 0, 2, 0, 1], shape + (6,)).astype(np.float32)
    affine = np.diag([-1.2, 0.8, 1.6, 1.0])
    return tensor, affine, shape, affine, np.zeros(shape + (3,), np.float32)


def test_ppd_cpu_does_not_import_cuda_and_cuda_dependency_error_is_explicit(monkeypatch):
    original = builtins.__import__

    def without_cuda(name, *args, **kwargs):
        if "ppd_cuda" in name:
            raise ImportError("CuPy is absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cuda)
    arguments = _inputs()
    result = nonlinear.resample_tensor_ppd_fsl(*arguments)
    np.testing.assert_array_equal(result.tensor, arguments[0])
    assert np.all(result.valid_mask)
    with pytest.raises(RuntimeError, match="requires CuPy; no CPU fallback"):
        nonlinear.resample_tensor_ppd_fsl(*arguments, ppd_device="cuda")
    with pytest.raises(ValueError, match="ppd_device"):
        nonlinear.resample_tensor_ppd_fsl(*arguments, ppd_device="unknown")


def test_ppd_cuda_execution_failure_does_not_use_cpu(monkeypatch):
    module = ModuleType("dwi2cond_xp.preprocessing.ppd_cuda")

    class FailingWorkspace:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("CUDA allocation failed")

    module.CudaPpdWorkspace = FailingWorkspace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="CUDA allocation failed"):
        nonlinear.resample_tensor_ppd_fsl(*_inputs(), ppd_device="cuda")
