"""FNIRT 直接支持域 Hessian 的可选 CUDA FP64 后端。"""

from __future__ import annotations

import numpy as np
from numba import cuda, float64
from numba.cuda import libdevice
from scipy.sparse import csc_matrix, tril

from . import fnirt


@cuda.jit(cache=True, fastmath=False)
def direct_fp64_kernel(
    left_x, left_y, left_z, right_x, right_y, right_z,
    left_bounds_x, left_bounds_y, left_bounds_z,
    right_bounds_x, right_bounds_y, right_bounds_z,
    selected, integral, weights, rows, columns,
    lx, ly, rx, ry, nx, ny, output,
):
    """每线程独立系数对；显式 RN double 乘加保留原规约顺序。"""

    entry = cuda.grid(1)
    if entry >= rows.size:
        return
    count = weights.shape[0]
    for block in range(count):
        output[block, entry] = 0.0
    row = rows[entry]
    column = columns[entry]
    ax, ay, az = row % lx, (row // lx) % ly, row // (lx * ly)
    bx, by, bz = column % rx, (column // rx) % ry, column // (rx * ry)
    x0 = max(left_bounds_x[ax, 0], right_bounds_x[bx, 0])
    x1 = min(left_bounds_x[ax, 1], right_bounds_x[bx, 1])
    y0 = max(left_bounds_y[ay, 0], right_bounds_y[by, 0])
    y1 = min(left_bounds_y[ay, 1], right_bounds_y[by, 1])
    z0 = max(left_bounds_z[az, 0], right_bounds_z[bz, 0])
    z1 = min(left_bounds_z[az, 1], right_bounds_z[bz, 1])
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return
    occupied = (
        integral[x1, y1, z1] - integral[x0, y1, z1]
        - integral[x1, y0, z1] - integral[x1, y1, z0]
        + integral[x0, y0, z1] + integral[x0, y1, z0]
        + integral[x1, y0, z0] - integral[x0, y0, z0]
    )
    if occupied == 0:
        return
    v0, v1, v2, v3, v4, v5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for z in range(z0, z1):
        lz, rz = left_z[az, z], right_z[bz, z]
        for y in range(y0, y1):
            lyv, ryv = left_y[ay, y], right_y[by, y]
            offset = (z * ny + y) * nx
            for x in range(x0, x1):
                voxel = offset + x
                if not selected[voxel]:
                    continue
                row_basis = libdevice.dmul_rn(lz, libdevice.dmul_rn(lyv, left_x[ax, x]))
                column_basis = libdevice.dmul_rn(rz, libdevice.dmul_rn(ryv, right_x[bx, x]))
                v0 = libdevice.dadd_rn(v0, libdevice.dmul_rn(
                    libdevice.dmul_rn(column_basis, float64(weights[0, voxel])), row_basis))
                if count >= 3:
                    v1 = libdevice.dadd_rn(v1, libdevice.dmul_rn(
                        libdevice.dmul_rn(column_basis, float64(weights[1, voxel])), row_basis))
                    v2 = libdevice.dadd_rn(v2, libdevice.dmul_rn(
                        libdevice.dmul_rn(column_basis, float64(weights[2, voxel])), row_basis))
                if count == 6:
                    v3 = libdevice.dadd_rn(v3, libdevice.dmul_rn(
                        libdevice.dmul_rn(column_basis, float64(weights[3, voxel])), row_basis))
                    v4 = libdevice.dadd_rn(v4, libdevice.dmul_rn(
                        libdevice.dmul_rn(column_basis, float64(weights[4, voxel])), row_basis))
                    v5 = libdevice.dadd_rn(v5, libdevice.dmul_rn(
                        libdevice.dmul_rn(column_basis, float64(weights[5, voxel])), row_basis))
    output[0, entry] = v0
    if count >= 3:
        output[1, entry], output[2, entry] = v1, v2
    if count == 6:
        output[3, entry], output[4, entry], output[5, entry] = v3, v4, v5



class CudaHessianBackend:
    """每次 FNIRT 运行独立保存几何，不初始化 CUDA 或修改全局函数。"""

    def __init__(self) -> None:
        self._workspaces: dict = {}
        self._field_shape: tuple[int, int, int] | None = None

    def __call__(
        self,
        field_shape: tuple[int, int, int],
        left_spacing: tuple[int, int, int],
        right_spacing: tuple[int, int, int],
        selected: np.ndarray,
        voxel_weights: tuple[np.ndarray, ...],
        workers: int,
        *,
        symmetric: bool = False,
    ) -> tuple[csc_matrix, ...]:
        """按原始支持域和乘加顺序计算，只在明确请求时使用 GPU。"""

        if not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if len(voxel_weights) not in (1, 3, 6):
            raise ValueError("CUDA Hessian supports 1, 3 or 6 weight blocks")
        field_shape = tuple(field_shape)
        left_spacing = tuple(left_spacing)
        right_spacing = tuple(right_spacing)
        for name, values in (
            ("field_shape", field_shape),
            ("left_spacing", left_spacing),
            ("right_spacing", right_spacing),
        ):
            if len(values) != 3 or any(
                not isinstance(value, (int, np.integer)) or value < 1
                for value in values
            ):
                raise ValueError(f"{name} must contain three positive integers")
        if symmetric and left_spacing != right_spacing:
            raise ValueError("symmetric spline products require matching knot spacing")
        selection = np.asarray(selected, dtype=bool)
        voxel_count = int(np.prod(field_shape))
        if selection.shape != (voxel_count,):
            raise ValueError("selected mask must match the flattened field")
        selected_count = int(np.count_nonzero(selection))
        dtype = np.float32 if symmetric else np.float64
        weights = np.zeros((len(voxel_weights), voxel_count), dtype=dtype)
        for block, weight in enumerate(voxel_weights):
            values = np.asarray(weight, dtype=dtype)
            if values.shape != (selected_count,):
                raise ValueError("selected voxel weights must match the basis rows")
            weights[block, selection] = values
        integral = np.zeros(tuple(size + 1 for size in field_shape), dtype=np.int64)
        integral[1:, 1:, 1:] = selection.reshape(field_shape, order="F")
        for axis in range(3):
            np.cumsum(integral, axis=axis, out=integral)
        key = (field_shape, left_spacing, right_spacing, symmetric)
        if field_shape != self._field_shape:
            self._workspaces.clear()
            self._field_shape = field_shape
        try:
            if key not in self._workspaces:
                if not cuda.is_available():
                    raise RuntimeError("CUDA is unavailable")
                # 每层仅需位移、偏置和交叉三套几何，限制额外调用的驻留空间。
                if len(self._workspaces) >= 3:
                    self._workspaces.clear()
                left = tuple(
                    fnirt._spline_axis_support(size, spacing)
                    for size, spacing in zip(field_shape, left_spacing, strict=True)
                )
                right = tuple(
                    fnirt._spline_axis_support(size, spacing)
                    for size, spacing in zip(field_shape, right_spacing, strict=True)
                )
                indptr, indices, columns, matrix_shape = fnirt._spline_support_output_pattern(
                    field_shape, left_spacing, right_spacing, symmetric
                )
                geometry = tuple(cuda.to_device(array) for array in (
                    *(axis[0] for axis in left), *(axis[0] for axis in right),
                    *(axis[1] for axis in left), *(axis[1] for axis in right),
                    indices, columns,
                ))
                self._workspaces[key] = (geometry, indptr, indices, matrix_shape)
            geometry, indptr, indices, matrix_shape = self._workspaces[key]
            device_selection = cuda.to_device(selection)
            device_integral = cuda.to_device(integral)
            device_weights = cuda.to_device(weights)
            device_output = cuda.device_array(
                (len(voxel_weights), indices.size), dtype=np.float64
            )
            left_shape = fnirt.fsl_coefficient_shape(field_shape, left_spacing)
            right_shape = fnirt.fsl_coefficient_shape(field_shape, right_spacing)
            threads = 128
            direct_fp64_kernel[(indices.size + threads - 1) // threads, threads](
                *geometry[:12], device_selection, device_integral, device_weights,
                geometry[12], geometry[13], left_shape[0], left_shape[1],
                right_shape[0], right_shape[1], field_shape[0], field_shape[1], device_output,
            )
            cuda.synchronize()
            block_values = device_output.copy_to_host()
            cuda.synchronize()
        except Exception as error:
            raise RuntimeError(
                "CUDA Hessian initialization, allocation or execution failed; no CPU fallback"
            ) from error
        blocks = []
        for value in block_values:
            matrix = csc_matrix((value.copy(), indices.copy(), indptr.copy()), shape=matrix_shape)
            matrix.eliminate_zeros()
            if symmetric:
                matrix = (matrix + tril(matrix, k=-1).T).tocsc()
            blocks.append(matrix)
        return tuple(blocks)
