import numpy as np
import pytest

from dwi2cond_xp.gradients import select_dti_volumes


def test_selects_only_b0_and_requested_shell():
    bvals = np.array([5, 40, 990, 1000, 1010, 2000, 3000, 1005, 995, 1001])
    selected = select_dti_volumes(bvals, shell=1000, tolerance=20, b0_threshold=50)
    assert np.array_equal(selected, [0, 1, 2, 3, 4, 7, 8, 9])


def test_requires_enough_shell_directions():
    with pytest.raises(ValueError, match="fewer than six"):
        select_dti_volumes(np.array([0, 1000, 1000, 1000]))
