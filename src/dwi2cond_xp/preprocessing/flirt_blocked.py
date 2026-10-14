"""T1 细层相关比的自适应 FP64 分块规约。"""
import numpy as np
from numba import njit, prange
from .flirt_cost import FlirtWeightedCorrelationRatio, _find_range_x, _trilinear, _finite_4x4
from ._numba import set_available_numba_threads
from numba import get_num_threads, set_num_threads

@njit(cache=True, nogil=True, fastmath=False)
def _tile(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_weight: np.ndarray,
    moving_weight: np.ndarray,
    unit_moving_weight: bool,
    reference_first_x: np.ndarray,
    reference_last_x: np.ndarray,
    reference_bins: np.ndarray,
    voxel_mapping: np.ndarray,
    moving_voxel_sizes: np.ndarray,
    bins: int,
    smooth_size: np.float32,
    row_start: int,
    row_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """保留原坐标递推和插值，用独立 FP64 直方图统计一个空间块。"""

    counts = np.zeros(bins + 1, dtype=np.float64)
    sums = np.zeros(bins + 1, dtype=np.float64)
    square_sums = np.zeros(bins + 1, dtype=np.float64)
    xb1 = reference.shape[0] - 1
    yb1 = reference.shape[1] - 1
    xb2 = np.float32(moving.shape[0]) - np.float32(1.0001)
    yb2 = np.float32(moving.shape[1]) - np.float32(1.0001)
    zb2 = np.float32(moving.shape[2]) - np.float32(1.0001)
    matrix = voxel_mapping.astype(np.float32)
    a11, a12, a13, a14 = matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]
    a21, a22, a23, a24 = matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]
    a31, a32, a33, a34 = matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    smoothx = np.float32(smooth_size / moving_voxel_sizes[0])
    smoothy = np.float32(smooth_size / moving_voxel_sizes[1])
    smoothz = np.float32(smooth_size / moving_voxel_sizes[2])
    zero = np.float32(0.0)

    for row in range(row_start, row_end):
        zindex = row // (yb1 + 1)
        yindex = row % (yb1 + 1)
        zfloat = np.float32(zindex)
        first_weighted_x = reference_first_x[yindex, zindex]
        last_weighted_x = reference_last_x[yindex, zindex]
        if last_weighted_x < first_weighted_x:
            continue
        yfloat = np.float32(yindex)
        o1 = np.float32(np.float32(yfloat * a12 + zfloat * a13) + a14)
        o2 = np.float32(np.float32(yfloat * a22 + zfloat * a23) + a24)
        o3 = np.float32(np.float32(yfloat * a32 + zfloat * a33) + a34)
        xmin, xmax = _find_range_x(o1, o2, o3, a11, a21, a31, xb1, xb2, yb2, zb2)
        o1 = np.float32(o1 + np.float32(xmin) * a11)
        o2 = np.float32(o2 + np.float32(xmin) * a21)
        o3 = np.float32(o3 + np.float32(xmin) * a31)
        first_x = max(xmin, first_weighted_x)
        last_x = min(xmax, last_weighted_x)
        for _ in range(xmin, first_x):
            o1 = np.float32(o1 + a11)
            o2 = np.float32(o2 + a21)
            o3 = np.float32(o3 + a31)
        for xindex in range(first_x, last_x + 1):
            ix, iy, iz = int(o1), int(o2), int(o3)
            interpolation_valid = (
                0 <= ix < moving.shape[0] - 1
                and 0 <= iy < moving.shape[1] - 1
                and 0 <= iz < moving.shape[2] - 1
            )
            if (
                ((xindex != xmin and xindex != xmax) or interpolation_valid)
                and reference_weight[xindex, yindex, zindex] != zero
            ):
                value = _trilinear(moving, o1, o2, o3)
                moving_weight_value = (
                    np.float32(1.0)
                    if unit_moving_weight
                    else _trilinear(moving_weight, o1, o2, o3)
                )
                weight = np.float32(
                    moving_weight_value * reference_weight[xindex, yindex, zindex]
                )
                if o1 < smoothx:
                    weight = np.float32(weight * np.float32(o1 / smoothx))
                elif np.float32(xb2 - o1) < smoothx:
                    weight = np.float32(weight * np.float32(np.float32(xb2 - o1) / smoothx))
                if o2 < smoothy:
                    weight = np.float32(weight * np.float32(o2 / smoothy))
                elif np.float32(yb2 - o2) < smoothy:
                    weight = np.float32(weight * np.float32(np.float32(yb2 - o2) / smoothy))
                if o3 < smoothz:
                    weight = np.float32(weight * np.float32(o3 / smoothz))
                elif np.float32(zb2 - o3) < smoothz:
                    weight = np.float32(weight * np.float32(np.float32(zb2 - o3) / smoothz))
                if weight < zero:
                    weight = zero
                bin_index = reference_bins[xindex, yindex, zindex]
                counts[bin_index] += float(weight)
                sums[bin_index] += float(np.float32(weight * value))
                square_sums[bin_index] += float(np.float32(np.float32(weight * value) * value))
            o1 = np.float32(o1 + a11)
            o2 = np.float32(o2 + a21)
            o3 = np.float32(o3 + a31)

    return counts, sums, square_sums

@njit(cache=True, nogil=True)
def _finish(counts, sums, square_sums, bins):
    """按固定顺序合并分箱，保留原有效计数与零方差规则。"""
    counts[bins-1] += counts[bins]
    sums[bins-1] += sums[bins]
    square_sums[bins-1] += square_sums[bins]
    total_count = 0.0
    total_sum = 0.0
    total_square = 0.0
    within = 0.0
    variance = 0.0
    for b in range(bins):
        count = counts[b]
        if count > 2.0:
            total_count += count
            total_sum += sums[b]
            total_square += square_sums[b]
            variance = (square_sums[b] - sums[b]*sums[b]/count)/(count-1.0)
            within += variance*count
    if total_count > 0.0:
        within /= total_count
    if total_count > 1.0:
        variance = (total_square - total_sum*total_sum/total_count)/(total_count-1.0)
    if total_count <= 1.0 or variance <= 0.0:
        return np.float32(1.0)
    return np.float32(within/variance)

@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _many(reference, moving, reference_weight, moving_weight, unit_moving_weight,
         first_x, last_x, reference_bins, mappings, sizes, bins, smooth_size, tiles):
    """分配独立块统计，合并时保持固定顺序，不使用原子累加。"""
    rows = reference.shape[1]*reference.shape[2]
    hist = np.empty((mappings.shape[0], tiles, 3, bins+1), dtype=np.float64)
    for task in prange(mappings.shape[0]*tiles):
        matrix = task // tiles
        tile = task % tiles
        counts, sums, squares = _tile(reference,moving,reference_weight,moving_weight,
            unit_moving_weight,first_x,last_x,reference_bins,mappings[matrix],sizes,
            bins,smooth_size,rows*tile//tiles,rows*(tile+1)//tiles)
        hist[matrix,tile,0] = counts
        hist[matrix,tile,1] = sums
        hist[matrix,tile,2] = squares
    result = np.empty(mappings.shape[0],dtype=np.float64)
    for matrix in range(mappings.shape[0]):
        counts=np.zeros(bins+1,dtype=np.float64)
        sums=np.zeros(bins+1,dtype=np.float64)
        squares=np.zeros(bins+1,dtype=np.float64)
        for tile in range(tiles):
            counts += hist[matrix,tile,0]
            sums += hist[matrix,tile,1]
            squares += hist[matrix,tile,2]
        result[matrix]=_finish(counts,sums,squares,bins)
    return result

class FlirtBlockedCorrelationRatio(FlirtWeightedCorrelationRatio):
    """单候选按空间分块，多候选按候选并行，统一使用 FP64 统计。"""

    def evaluate_many(self, transforms: np.ndarray, workers: int = 8) -> np.ndarray:
        """验证候选后按调用预算执行，任何异常也恢复线程设置。"""
        matrices = np.asarray(transforms, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices):
            raise ValueError("transforms must have shape (n, 4, 4) with n greater than zero")
        if not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        validated = np.asarray([_finite_4x4(matrix, "transform") for matrix in matrices])
        mappings = self._inverse_moving_sampling @ np.linalg.inv(validated) @ self.reference_sampling
        tiles = 8 if len(matrices) == 1 else 1
        previous_threads = get_num_threads()
        set_available_numba_threads(min(int(workers), len(matrices) * tiles))
        try:
            return _many(
                self.reference, self.moving, self.reference_weight, self.moving_weight,
                self._unit_moving_weight, self._reference_first_x, self._reference_last_x,
                self._reference_bins, mappings, self._moving_voxel_sizes,
                int(self.bins), np.float32(self.smooth_size), tiles,
            )
        finally:
            set_num_threads(previous_threads)

    def __call__(self, moving_to_reference: np.ndarray) -> float:
        """单候选也用同一种统计，避免代价随批量大小改变。"""
        matrix = _finite_4x4(moving_to_reference, "moving_to_reference")
        return float(self.evaluate_many(matrix[None], workers=get_num_threads())[0])

    def evaluate_ordered_parallel(self, moving_to_reference: np.ndarray) -> float:
        """沿用优化器的并行调用入口，规约采用明确的分块顺序。"""
        return self(moving_to_reference)
