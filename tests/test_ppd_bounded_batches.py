import numpy as np
import pytest

from dwi2cond_xp.preprocessing import nonlinear


@pytest.mark.parametrize("jacobian_mode", ["physical", "simnibs46-literal"])
def test_ppd_batches_preserve_halo_jacobians_and_full_grid_qa(monkeypatch, jacobian_mode):
    shape = (9, 8, 7)
    affine = np.diag([-1.2, 0.8, 1.6, 1.0])
    grid = np.indices(shape, dtype=np.float32)
    warp = np.zeros(shape + (3,), dtype=np.float32)
    warp[..., 0] = 0.08 * np.sin(grid[0])
    warp[:2, ..., 0] = -2 * grid[0, :2]
    warp[..., 1] = 0.04 * np.cos(grid[2])
    tensor = np.broadcast_to([3, 0, 0, 2, 0, 1], shape + (6,)).astype(np.float32).copy()
    tensor[3:5] = [1, 0, 0, 1, 0, 1]
    tensor[-1, ..., 0] = np.nan
    reference_mask = np.zeros(shape, dtype=bool)
    reference_mask[2:6] = True
    # 强制跨越多个分块，包含全局边界与内部 halo。
    monkeypatch.setattr(nonlinear, "_PPD_BATCH_VOXELS", 112)
    arguments = dict(reference_mask=reference_mask, jacobian_mode=jacobian_mode)
    serial = nonlinear.resample_tensor_ppd_fsl(
        tensor, affine, shape, affine, warp, workers=1, **arguments
    )
    parallel = nonlinear.resample_tensor_ppd_fsl(
        tensor, affine, shape, affine, warp, workers=4, **arguments
    )
    for name in nonlinear.NonlinearTensorResult.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(serial, name), getattr(parallel, name))
    physical = nonlinear.fsl_warp_jacobians(warp, voxel_sizes=np.array([1.2, 0.8, 1.6]))
    ppd = physical if jacobian_mode == "physical" else nonlinear.fsl_warp_jacobians(warp)
    np.testing.assert_array_equal(parallel.backward_jacobian, ppd[0])
    np.testing.assert_array_equal(parallel.forward_jacobian, ppd[1])
    np.testing.assert_array_equal(parallel.backward_jacobian_determinant, physical[2])
    np.testing.assert_array_equal(parallel.fold_mask, physical[3])
    np.testing.assert_array_equal(parallel.near_singular_mask, physical[4])
    assert np.any(parallel.fold_mask & ~reference_mask)
    assert np.any(parallel.invalid_tensor_mask & ~reference_mask)
    assert np.any(parallel.repeated_eigenvalue_mask & parallel.valid_mask)
    assert np.any(parallel.low_fa_mask & parallel.valid_mask)
    assert not np.any(parallel.valid_mask & ~reference_mask)
