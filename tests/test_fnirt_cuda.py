"""无 GPU 的三个关键后端边界检查。"""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from dwi2cond_xp.preprocessing import fnirt as fnirt_module


CUDA_MODULE = "dwi2cond_xp.preprocessing.fnirt_cuda"


def test_cpu_default_never_imports_cuda(monkeypatch):
    """CPU 默认请求不导入可选模块，且保留单线程后端标记。"""

    captured = []
    original_import = builtins.__import__

    def import_without_cuda(name, *args, **kwargs):
        if "fnirt_cuda" in name:
            raise AssertionError("CPU requested CUDA import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cuda)
    monkeypatch.setattr(
        fnirt_module, "_run_simnibs46_fnirt",
        lambda *args, **kwargs: captured.append(kwargs) or "result",
    )
    arguments = (np.ones((1, 1, 1)),) * 2 + (np.eye(4),) * 3
    assert fnirt_module.run_simnibs46_fnirt(*arguments, workers=1) == "result"
    assert captured[0]["workers"] == 1
    assert captured[0]["_hessian_builder"] is None
    with pytest.raises(ValueError, match="hessian_device"):
        fnirt_module.run_simnibs46_fnirt(*arguments, hessian_device="gpu")


def test_cuda_is_lazy_and_instances_are_per_run(monkeypatch):
    """不同运行不共享实例，同一运行的重复 Hessian 复用实例。"""

    instances = []
    captured = []

    class FakeBackend:
        """用计数器检查构造与调用，不访问 CUDA。"""

        def __init__(self):
            instances.append(self)

        def __call__(self, *args, **kwargs):
            return self, args, kwargs

    fake_module = types.ModuleType(CUDA_MODULE)
    fake_module.CudaHessianBackend = FakeBackend
    monkeypatch.setitem(sys.modules, CUDA_MODULE, fake_module)
    monkeypatch.setattr(
        fnirt_module, "_run_simnibs46_fnirt",
        lambda *args, **kwargs: captured.append(kwargs),
    )
    arguments = (np.ones((1, 1, 1)),) * 2 + (np.eye(4),) * 3
    for workers in (1, 2):
        fnirt_module.run_simnibs46_fnirt(
            *arguments, workers=workers, hessian_device="cuda"
        )
    assert not instances
    first, second = (entry["_hessian_builder"] for entry in captured)
    assert first is not second
    assert first(1, symmetric=True)[1:] == ((1,), {"symmetric": True})
    assert first(2)[0] is instances[0]
    assert len(instances) == 1
    assert second(3)[0] is instances[1]
    assert len(instances) == 2
    with pytest.raises(ValueError, match="explicitly supplied"):
        fnirt_module.evaluate_fnirt_hessian(
            None, None, None, None, None, None,
            _hessian_builder=first, _displacement_basis=object(),
        )


def test_cuda_invalid_inputs_and_runtime_errors_are_explicit(monkeypatch):
    """检查宿主验证与 CUDA 不可用、显存分配失败，禁止回退 CPU。"""

    from dwi2cond_xp.preprocessing import fnirt_cuda

    availability_calls = []
    monkeypatch.setattr(
        fnirt_cuda.cuda, "is_available",
        lambda: availability_calls.append(True) or False,
    )
    monkeypatch.setattr(
        fnirt_module, "_spline_axis_support",
        lambda *args: (np.ones((1, 1)), np.zeros((1, 2))),
    )
    monkeypatch.setattr(
        fnirt_module, "_spline_support_output_pattern",
        lambda *args: (
            np.array([0, 1]), np.array([0]), np.array([0]), (1, 1)
        ),
    )
    backend = fnirt_cuda.CudaHessianBackend()
    assert not availability_calls
    arguments = ((1, 1, 1), (1, 1, 1), (1, 1, 1), np.ones(1, dtype=bool), (np.ones(1),), 1)
    with pytest.raises(ValueError, match="matching knot spacing"):
        backend(arguments[0], arguments[1], (2, 2, 2), *arguments[3:], symmetric=True)
    with pytest.raises(ValueError, match="selected mask"):
        backend(*arguments[:3], np.ones(2, dtype=bool), *arguments[4:])
    with pytest.raises(ValueError, match="selected voxel weights"):
        backend(*arguments[:4], (np.ones(2),), 1)
    assert not availability_calls
    with pytest.raises(RuntimeError, match="no CPU fallback") as unavailable:
        backend(*arguments)
    assert "CUDA is unavailable" in str(unavailable.value.__cause__)
    assert not backend._workspaces
    monkeypatch.setattr(fnirt_cuda.cuda, "is_available", lambda: True)

    def fail_allocation(array):
        raise MemoryError("simulated device allocation failure")

    monkeypatch.setattr(fnirt_cuda.cuda, "to_device", fail_allocation)
    with pytest.raises(RuntimeError, match="no CPU fallback") as failed:
        backend(*arguments)
    assert isinstance(failed.value.__cause__, MemoryError)
    assert not backend._workspaces
