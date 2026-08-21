import numpy as np
import pytest

from dwi2cond_xp.gradients import load_gradients, select_dti_volumes


def test_selects_only_b0_and_requested_shell():
    bvals = np.array([5, 40, 990, 1000, 1010, 2000, 3000, 1005, 995, 1001])
    selected = select_dti_volumes(bvals, shell=1000, tolerance=20, b0_threshold=50)
    assert np.array_equal(selected, [0, 1, 2, 3, 4, 7, 8, 9])


def test_requires_enough_shell_directions():
    with pytest.raises(ValueError, match="fewer than six"):
        select_dti_volumes(np.array([0, 1000, 1000, 1000]))


def test_load_gradients_accepts_both_bvec_layouts(tmp_path):
    bvals_file = tmp_path / "bvals"
    bvecs_file = tmp_path / "bvecs"
    np.savetxt(bvals_file, [[0, 1000, 1000]])
    np.savetxt(bvecs_file, np.eye(3))
    bvals, bvecs = load_gradients(bvals_file, bvecs_file)
    assert bvals.tolist() == [0, 1000, 1000]
    assert np.array_equal(bvecs, np.eye(3))
    np.savetxt(bvecs_file, np.arange(12).reshape(4, 3))
    np.savetxt(bvals_file, [[0, 1000, 1000, 1000]])
    assert load_gradients(bvals_file, bvecs_file)[1].shape == (4, 3)


@pytest.mark.parametrize(
    ("bvals", "bvecs", "message"),
    [
        ([0, 1000], [1, 2, 3], "two-dimensional"),
        ([0, 1000], [[1, 2], [3, 4]], "3xN or Nx3"),
        ([0, 1000, 1000, 1000], [[1, 0, 0], [0, 1, 0]], "different numbers"),
        ([0, np.nan], [[0, 0, 0], [1, 0, 0]], "NaN or Inf"),
        ([-1, 1000], [[0, 0, 0], [1, 0, 0]], "must not be negative"),
    ],
)
def test_load_gradients_rejects_invalid_files(tmp_path, bvals, bvecs, message):
    bvals_file = tmp_path / "bvals"
    bvecs_file = tmp_path / "bvecs"
    np.savetxt(bvals_file, bvals)
    np.savetxt(bvecs_file, bvecs)
    with pytest.raises(ValueError, match=message):
        load_gradients(bvals_file, bvecs_file)


def test_shell_parameter_validation_and_missing_b0():
    with pytest.raises(ValueError, match="above"):
        select_dti_volumes(np.array([0, 1000]), shell=40, b0_threshold=50)
    with pytest.raises(ValueError, match="tolerance"):
        select_dti_volumes(np.array([0, 1000]), tolerance=0)
    with pytest.raises(ValueError, match="No b=0"):
        select_dti_volumes(np.full(6, 1000.0))
