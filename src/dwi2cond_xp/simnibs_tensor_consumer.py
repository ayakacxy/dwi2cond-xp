"""仅为本次 SimNIBS 实例适配张量方向和对称特征值分解。"""

from __future__ import annotations

import functools
import hashlib
import inspect
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np

from .conductivity import _sorted_eigensystem, fsl_tensor_basis


def _source_sha256(path: Path) -> str:
    """计算文件校验值，供消费方法和原始张量的来源记录使用。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_simnibs_tensor_consumer(
    tensor_file: str | Path, *, basis_mode: str = "physical",
    eigensystem_mode: str = "stable",
) -> dict[str, Any]:
    """记录原张量和显式算法设置；不重写张量文件。"""

    if eigensystem_mode not in {"stable", "simnibs46-literal"}:
        raise ValueError("eigensystem_mode must be stable or simnibs46-literal")
    source = Path(tensor_file).resolve()
    image = nib.load(str(source))
    if len(image.shape) != 4 or image.shape[-1] != 6:
        raise ValueError("SimNIBS tensor input must be a 4-D six-component NIfTI")
    selected = fsl_tensor_basis(image.affine, basis_mode=basis_mode)
    literal = fsl_tensor_basis(image.affine, basis_mode="simnibs46-literal")
    return {
        "basis_mode": basis_mode,
        "eigensystem_mode": eigensystem_mode,
        "purpose": "simnibs46-conductivity-consumer-only",
        "original_tensor": str(source),
        "original_sha256": _source_sha256(source),
        "adapter_module_sha256": _source_sha256(Path(__file__)),
        "consumer_tensor": str(source),
        "tensor_file_rewritten": False,
        "adapter": "instance-scoped-method" if (
            basis_mode != "simnibs46-literal" or eigensystem_mode != "simnibs46-literal"
        ) else "upstream-unmodified",
        "selected_basis": selected.tolist(),
        "upstream_literal_basis": literal.tolist(),
        "formula": "D_world = B_selected @ D_FSL @ B_selected.T",
        "basis_transform_stage": "sampled-tissue-tensors-before-eigensystem",
        "component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
    }


def _clone_function(function, globals_replacements):
    """复制单个函数的名字空间，不改第三方模块的全局变量。"""

    namespace = dict(function.__globals__)
    namespace.update(globals_replacements)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return functools.update_wrapper(clone, function)


def bind_simnibs_tensor_consumer(
    consumer, tensor_file: str | Path, *, basis_mode: str = "physical",
    eigensystem_mode: str = "stable", leadfield: bool = False,
) -> dict[str, Any]:
    """绑定当前实例的方法，保留官方采样、组织缩放、FEM 和输出逻辑。

    TDCSLEADFIELD 使用静态 SimuList 调用，故仅复制其 run 的全局名字空间，
    将那一次调用指向当前实例的消费方法。原类、模块和其他实例均不变。
    """

    import simnibs
    from simnibs import sim_struct
    from simnibs.utils import cond_utils

    if simnibs.__version__ != "4.6.0":
        raise RuntimeError("Tensor consumer adaptation requires SimNIBS 4.6.0")
    report = describe_simnibs_tensor_consumer(
        tensor_file, basis_mode=basis_mode, eigensystem_mode=eigensystem_mode
    )
    if report["adapter"] == "upstream-unmodified":
        return report
    upstream_cond = cond_utils.cond2elmdata
    upstream_method = sim_struct.SimuList.cond2elmdata
    report["upstream_cond2elmdata_source_sha256"] = hashlib.sha256(
        inspect.getsource(upstream_cond).encode("utf-8")
    ).hexdigest()
    if ("_get_sorted_eigenv" not in upstream_cond.__code__.co_names or
        upstream_method.__globals__.get("simnibs") is not simnibs):
        raise RuntimeError("SimNIBS 4.6 conductivity hook differs from the expected implementation")
    def selected_cond(*args, **kwargs):
        """只变换已采样的组织张量，避免分配完整头部的张量副本。"""

        basis = None
        volume = kwargs.get("anisotropy_volume")
        if volume is not None and basis_mode == "physical":
            basis = fsl_tensor_basis(kwargs["affine"])
            kwargs["correct_FSL"] = False
        local_cond = upstream_cond
        if basis is not None or eigensystem_mode == "stable":
            def selected_eigensystem(tensors):
                """每个组织在上游分解入口只作一次物理方向变换。"""

                if basis is not None:
                    tensors = np.einsum(
                        "ij,njk,lk->nil", basis, tensors, basis, optimize=True
                    )
                return _sorted_eigensystem(tensors, eigensystem_mode)

            local_cond = _clone_function(
                upstream_cond, {"_get_sorted_eigenv": selected_eigensystem}
            )
        return local_cond(*args, **kwargs)

    # 此代理仅属于复制的方法，第三方模块的 simnibs 引用保持不变。
    scoped_simnibs = SimpleNamespace(utils=SimpleNamespace(
        cond_utils=SimpleNamespace(cond2elmdata=selected_cond)
    ))
    local_method = _clone_function(upstream_method, {"simnibs": scoped_simnibs})
    run_clone = None
    if leadfield:
        upstream_run = type(consumer).run
        report["upstream_leadfield_run_source_sha256"] = hashlib.sha256(
            inspect.getsource(upstream_run).encode("utf-8")
        ).hexdigest()
        if ("SimuList" not in upstream_run.__code__.co_names or
            "SimuList.cond2elmdata(self, w_elec)" not in inspect.getsource(upstream_run)):
            raise RuntimeError("SimNIBS 4.6 lead-field static conductivity call differs from the expected implementation")

        class ScopedSimuList:
            """只为本实例的 run 提供静态消费入口。"""

            @staticmethod
            def cond2elmdata(instance, *args, **kwargs):
                return instance.cond2elmdata(*args, **kwargs)

        run_clone = _clone_function(upstream_run, {"SimuList": ScopedSimuList})
    consumer.cond2elmdata = MethodType(local_method, consumer)
    if run_clone is not None:
        consumer.run = MethodType(run_clone, consumer)
    report["leadfield_static_call_adapted"] = run_clone is not None
    return report
