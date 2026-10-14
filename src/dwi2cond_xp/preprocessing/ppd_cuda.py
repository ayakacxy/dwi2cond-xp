"""可选 CuPy PPD 后端：连续运算留在设备，末尾一次回传结果。"""

from __future__ import annotations

from time import perf_counter

import cupy as cp
import cupyx
from cupyx.scipy.ndimage import map_coordinates
import numpy as np

from .nonlinear import NonlinearTensorResult
from .transforms import fsl_voxel_to_scaled_mm


def synchronize():
    cp.cuda.get_current_stream().synchronize()


def tensor_matrix(values):
    output = cp.empty(values.shape[:-1] + (3, 3), dtype=cp.float64)
    output[..., 0, 0] = values[..., 0]
    output[..., 0, 1] = output[..., 1, 0] = values[..., 1]
    output[..., 0, 2] = output[..., 2, 0] = values[..., 2]
    output[..., 1, 1] = values[..., 3]
    output[..., 1, 2] = output[..., 2, 1] = values[..., 4]
    output[..., 2, 2] = values[..., 5]
    return output


def gradient_float32(values):
    """沿用当前体素差分的 float32 检查点及全局单边边界。"""

    result = cp.empty(values.shape[:3] + (3, 3), dtype=cp.float32)
    for component in range(3):
        field = values[..., component]
        for axis in range(3):
            inside = [slice(None)] * 3
            lower, upper = inside.copy(), inside.copy()
            inside[axis], lower[axis], upper[axis] = slice(1, -1), slice(0, -2), slice(2, None)
            result[..., component, axis][tuple(inside)] = (
                field[tuple(upper)] - field[tuple(lower)]
            ) / cp.float32(2)
            first, second = [slice(None)] * 3, [slice(None)] * 3
            first[axis], second[axis] = 0, 1
            result[..., component, axis][tuple(first)] = field[tuple(second)] - field[tuple(first)]
            first[axis], second[axis] = -1, -2
            result[..., component, axis][tuple(first)] = field[tuple(first)] - field[tuple(second)]
    return result


def jacobians(gradient, directions, dimensions, tolerance):
    backward = gradient * (directions / dimensions).reshape((1, 1, 1, 1, 3))
    for axis in range(3):
        backward[..., axis, axis] += cp.float64(1)
    determinant = cp.linalg.det(backward)
    fold = determinant <= 0
    near = cp.abs(determinant) <= tolerance
    forward = cp.full(backward.shape, cp.nan, dtype=cp.float64)
    selected = ~near & cp.isfinite(determinant)
    forward[selected] = cp.linalg.inv(backward[selected])
    return backward, forward, determinant, fold, near


def ppd_rotation(forward, eigenvectors, tolerance):
    first, second = eigenvectors[..., :, 2], eigenvectors[..., :, 1]
    mapped_first = cp.einsum('...ij,...j->...i', forward, first)
    mapped_second = cp.einsum('...ij,...j->...i', forward, second)
    first_norm = cp.linalg.norm(mapped_first, axis=-1)
    second_norm = cp.linalg.norm(mapped_second, axis=-1)
    stable = cp.isfinite(first_norm) & cp.isfinite(second_norm) & (first_norm > tolerance) & (second_norm > tolerance)
    target_first = cp.where(stable[..., None], mapped_first / cp.where(stable, first_norm, 1)[..., None], 0)
    target_second = cp.where(stable[..., None], mapped_second / cp.where(stable, second_norm, 1)[..., None], 0)
    projection = target_second - cp.sum(target_first * target_second, axis=-1)[..., None] * target_first
    projection_norm = cp.linalg.norm(projection, axis=-1)
    stable &= cp.isfinite(projection_norm) & (projection_norm > tolerance)
    projection = cp.where(stable[..., None], projection / cp.where(stable, projection_norm, 1)[..., None], projection)
    source_frame = cp.stack((first, second, cp.cross(first, second)), axis=-1)
    target_frame = cp.stack((target_first, projection, cp.cross(target_first, projection)), axis=-1)
    rotation = cp.einsum('...ij,...kj->...ik', target_frame, source_frame, optimize=True)
    return cp.where(stable[..., None, None], rotation, cp.nan), stable


class CudaPpdWorkspace:
    """源张量和采样矩阵只上传一次；每个连续分块保留 halo 与全部 QA。"""

    def __init__(self, tensor, source_affine, reference_shape, reference_affine,
                 *, source_mask=None, source_voxel_sizes=None, reference_voxel_sizes=None):
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.ndim != 4 or tensor.shape[-1] != 6:
            raise ValueError('tensor must have shape (x,y,z,6)')
        self.shape = tuple(int(v) for v in reference_shape)
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError('Reference grid axes require at least two voxels')
        sampling = fsl_voxel_to_scaled_mm(self.shape, reference_affine, reference_voxel_sizes)
        source_sampling = fsl_voxel_to_scaled_mm(tensor.shape[:3], source_affine, source_voxel_sizes)
        source_valid = np.any(tensor[..., :3] != 0, axis=-1) if source_mask is None else np.asarray(source_mask) != 0
        if source_valid.shape != tensor.shape[:3]:
            raise ValueError('source mask must match tensor')
        self.has_nonfinite = not bool(np.all(np.isfinite(tensor)))
        synchronize()
        start = perf_counter()
        self.values = cp.asarray(tensor)
        self.source_valid = cp.asarray(source_valid)
        self.target_sampling = cp.asarray(sampling)
        self.source_sampling = cp.asarray(source_sampling)
        self.source_inverse = cp.asarray(np.linalg.inv(source_sampling[:3, :3]))
        self.directions = cp.asarray(np.sign(np.diag(sampling)[:3]))
        self.dimensions = cp.asarray(np.abs(np.diag(sampling)[:3]))
        synchronize()
        self.source_upload_seconds = perf_counter() - start
        self.source_upload_bytes = tensor.nbytes + source_valid.nbytes + 2 * sampling.nbytes + 9 * 8 + 6 * 8

    def slab(self, displacement, start, stop, *, reference_mask=None,
             compatibility_mode='strict-fsl', jacobian_mode='physical',
             singular_tolerance=1e-8, repeated_eigenvalue_tolerance=1e-8, low_fa_threshold=.05):
        """上传 halo 后完成采样到 QA；数值数组只在末尾统一回传。"""

        if compatibility_mode not in ('strict-fsl', 'robust') or jacobian_mode not in ('physical', 'simnibs46-literal'):
            raise ValueError('Unsupported compatibility/Jacobian mode')
        if not 0 <= start < stop <= self.shape[0]:
            raise ValueError('Slab is outside reference grid')
        for tolerance in (singular_tolerance, repeated_eigenvalue_tolerance):
            if not np.isfinite(tolerance) or tolerance <= 0:
                raise ValueError('Tolerances must be finite and positive')
        if not np.isfinite(low_fa_threshold) or low_fa_threshold < 0:
            raise ValueError('FA threshold must be finite and nonnegative')
        displacement = np.asarray(displacement, dtype=np.float32)
        if displacement.shape != self.shape + (3,):
            raise ValueError('Displacement shape mismatch')
        halo_start, halo_stop = max(0, start - 1), min(self.shape[0], stop + 1)
        halo = displacement[halo_start:halo_stop]
        if not np.all(np.isfinite(halo)):
            raise ValueError('Displacement must be finite')
        slab_shape = (stop - start,) + self.shape[1:]
        reference_valid = np.ones(slab_shape, bool) if reference_mask is None else np.asarray(reference_mask[start:stop]) != 0
        synchronize()
        total_start = perf_counter()
        warp = cp.asarray(halo)
        target_valid = cp.asarray(reference_valid)
        synchronize()
        uploaded = perf_counter()
        with cupyx.errstate(linalg='raise'):
            grid = cp.indices(slab_shape, dtype=cp.float64).reshape(3, -1)
            grid[0] += start
            target_mm = self.target_sampling[:3, :3] @ grid + self.target_sampling[:3, 3, None]
            core = warp[start - halo_start:stop - halo_start]
            coordinates = self.source_inverse @ (target_mm + core.reshape(-1, 3).T - self.source_sampling[:3, 3, None])
            rounded = cp.where(coordinates >= 0, cp.floor(coordinates + .5), cp.ceil(coordinates - .5)).astype(cp.int64)
            bounds = cp.asarray(self.values.shape[:3])[:, None]
            in_bounds = cp.all(rounded >= 0, axis=0) & cp.all(rounded < bounds, axis=0)
            clipped = cp.minimum(cp.maximum(rounded, 0), bounds - 1)
            sampled_source = self.source_valid[tuple(clipped)] & in_bounds
            sampled = cp.empty(slab_shape + (6,), dtype=cp.float32)
            for component in range(6):
                # 与 SciPy 一样先进行双精度插值，再写回 float32 采样检查点。
                sampled[..., component] = map_coordinates(
                    self.values[..., component], coordinates, output=cp.float64,
                    order=1, mode='grid-constant', cval=0, prefilter=False,
                ).reshape(slab_shape).astype(cp.float32)
            if self.has_nonfinite:
                # SciPy 保留 0×NaN/Inf 的传播；CuPy 插值会跳过零权重角。
                lower = cp.floor(coordinates).astype(cp.int64)
                fraction = coordinates - lower
                tainted = cp.zeros((coordinates.shape[1], 6), dtype=bool)
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            corner = lower + cp.asarray((dx, dy, dz))[:, None]
                            inside = cp.all(corner >= 0, axis=0) & cp.all(corner < bounds, axis=0)
                            clipped_corner = cp.minimum(cp.maximum(corner, 0), bounds - 1)
                            pixel = self.values[tuple(clipped_corner)]
                            weight = (fraction[0] if dx else 1 - fraction[0]) * (fraction[1] if dy else 1 - fraction[1]) * (fraction[2] if dz else 1 - fraction[2])
                            bad = cp.isnan(pixel) | (cp.isinf(pixel) & (weight[:, None] == 0))
                            tainted |= inside[:, None] & bad
                sampled = cp.where(tainted.reshape(sampled.shape), cp.float32(cp.nan), sampled)
            gradient = gradient_float32(warp)[start - halo_start:stop - halo_start].astype(cp.float64)
            backward, forward, determinant, fold, near = jacobians(gradient, self.directions, self.dimensions, singular_tolerance)
            ppd_fold, ppd_near = fold, near
            if jacobian_mode == 'simnibs46-literal':
                backward, forward, _, ppd_fold, ppd_near = jacobians(gradient, self.directions, cp.ones(3, cp.float64), singular_tolerance)
            matrices = tensor_matrix(sampled)
            finite = cp.all(cp.isfinite(matrices), axis=(-2, -1))
            zero = finite & cp.all(sampled == 0, axis=-1)
            # 零张量直接给出精确零谱与单位基；只分解非零有限矩阵。
            eigenvalues = cp.zeros(matrices.shape[:-1], dtype=cp.float64)
            eigenvectors = cp.broadcast_to(
                cp.eye(3, dtype=cp.float64), matrices.shape,
            ).copy()
            selected_eigen = finite & ~zero
            solver_input = matrices[selected_eigen]
            if solver_input.shape[0]:
                selected_values, selected_vectors = cp.linalg.eigh(solver_input)
                eigenvalues[selected_eigen] = selected_values
                eigenvectors[selected_eigen] = selected_vectors
            eigenvalues = cp.where(finite[..., None], eigenvalues, cp.nan)
            eigenvectors = cp.where(finite[..., None, None], eigenvectors, cp.nan)
            scale = cp.maximum(cp.max(cp.abs(eigenvalues), axis=-1), 1)
            repeated = cp.minimum(cp.abs(eigenvalues[..., 2] - eigenvalues[..., 1]), cp.abs(eigenvalues[..., 1] - eigenvalues[..., 0])) <= repeated_eigenvalue_tolerance * scale
            mean = cp.mean(eigenvalues, axis=-1)
            numerator = cp.sum((eigenvalues - mean[..., None]) ** 2, axis=-1)
            denominator = cp.sum(eigenvalues ** 2, axis=-1)
            fa = cp.where(denominator > 0, cp.sqrt(1.5 * numerator / cp.where(denominator > 0, denominator, 1)), 0)
            low_fa = fa < low_fa_threshold
            active = target_valid & sampled_source.reshape(slab_shape)
            rotation, stable = ppd_rotation(forward, eigenvectors, singular_tolerance)
            stable &= active
            unsafe = active & (~finite | fold | near | ppd_fold | ppd_near | ~stable)
            valid = active & finite & ~fold & ~near & ~ppd_fold & ~ppd_near & stable
            # 只让有效矩阵参与乘法，避免 0×NaN 向零填充值扩散。
            safe_rotation = cp.where(valid[..., None, None], rotation, 0)
            safe_matrix = cp.where(valid[..., None, None], matrices, 0)
            rotated = cp.einsum('...ij,...jk,...lk->...il', safe_rotation, safe_matrix, safe_rotation, optimize=True)
            tensor = cp.stack((rotated[..., 0, 0], rotated[..., 0, 1], rotated[..., 0, 2], rotated[..., 1, 1], rotated[..., 1, 2], rotated[..., 2, 2]), axis=-1).astype(cp.float32)
            invalid = ~finite | ~cp.all(cp.isfinite(eigenvalues), axis=-1)
            arrays = dict(tensor=tensor, valid_mask=valid, backward_jacobian=backward,
                forward_jacobian=forward, backward_jacobian_determinant=determinant,
                fold_mask=fold, near_singular_mask=near, low_fa_mask=low_fa,
                repeated_eigenvalue_mask=repeated, invalid_tensor_mask=invalid)
            unsafe_byte = cp.any(unsafe).astype(cp.uint8).reshape(1)
            synchronize()
            computed = perf_counter()
            # 保留每种 dtype 的原始字节，最后仅一次大数据 D2H。
            metadata = [(name, value.shape, value.dtype, value.nbytes) for name, value in arrays.items()]
            packed = cp.concatenate([cp.ascontiguousarray(value).view(cp.uint8).reshape(-1) for value in arrays.values()] + [unsafe_byte])
            synchronize()
            packed_at = perf_counter()
            host = cp.asnumpy(packed)
            synchronize()
            downloaded = perf_counter()
        fields = {}
        offset = 0
        for name, shape, dtype, size in metadata:
            fields[name] = host[offset:offset + size].view(dtype).reshape(shape)
            offset += size
        if compatibility_mode == 'strict-fsl' and host[offset]:
            raise ValueError('strict-fsl nonlinear PPD encountered an invalid tensor or Jacobian; use robust mode only when explicit zero-filling is intended')
        return NonlinearTensorResult(**fields), dict(
            slab_upload_seconds=uploaded - total_start, compute_seconds=computed - uploaded,
            pack_seconds=packed_at - computed, download_seconds=downloaded - packed_at,
            chain_seconds=downloaded - total_start, slab_upload_bytes=halo.nbytes + reference_valid.nbytes,
            download_bytes=host.nbytes, numerical_download_calls=1,
            note='Solver error checks/boolean allocation may synchronize small metadata; no intermediate tensor/Jacobian/eigen array CPU roundtrips.',
        )
