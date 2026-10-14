"""分块相关比的失败边界及线程预算恢复。"""

import numpy as np
import pytest
from numba import get_num_threads

from dwi2cond_xp.preprocessing import flirt_blocked


def _evaluator():
    reference = np.arange(8 ** 3, dtype=np.float32).reshape(8, 8, 8)
    weight = np.ones_like(reference)
    return flirt_blocked.FlirtBlockedCorrelationRatio(
        reference, reference.copy(), weight, weight,
        np.eye(4), np.eye(4), bins=16,
    )


@pytest.mark.parametrize("transforms", [np.zeros((0, 4, 4)), np.eye(4), np.ones((1, 3, 3))])
def test_blocked_evaluator_rejects_empty_and_malformed_batches(transforms):
    before = get_num_threads()
    with pytest.raises(ValueError, match="shape.*greater than zero"):
        _evaluator().evaluate_many(transforms)
    assert get_num_threads() == before


@pytest.mark.parametrize("workers", [0, -1, 1.5])
def test_blocked_evaluator_rejects_invalid_worker_budget(workers):
    before = get_num_threads()
    with pytest.raises(ValueError, match="positive integer"):
        _evaluator().evaluate_many(np.eye(4)[None], workers=workers)
    assert get_num_threads() == before


def test_blocked_evaluator_restores_thread_budget_after_kernel_error(monkeypatch):
    """内核失败向调用者传播，不能把局部预算留给后续阶段。"""
    evaluator = _evaluator()
    before = get_num_threads()

    def fail(*args):
        assert get_num_threads() == 1
        raise RuntimeError("injected histogram failure")

    monkeypatch.setattr(flirt_blocked, "_many", fail)
    with pytest.raises(RuntimeError, match="histogram failure"):
        evaluator.evaluate_many(np.eye(4)[None], workers=1)
    assert get_num_threads() == before


def test_blocked_ordered_parallel_matches_scalar_entrypoint(monkeypatch):
    evaluator = _evaluator()
    expected = np.array([0.25])
    seen = []

    def evaluate(transforms, workers):
        seen.append((transforms.copy(), workers))
        return expected

    monkeypatch.setattr(evaluator, "evaluate_many", evaluate)
    assert evaluator.evaluate_ordered_parallel(np.eye(4)) == 0.25
    np.testing.assert_array_equal(seen[0][0], np.eye(4)[None])
    assert seen[0][1] == get_num_threads()


def test_blocked_negative_sampling_weights_match_serial_clamping():
    """负采样权重与串行参考一致钳零，不贡献有效直方图质量。"""
    reference = np.arange(8 ** 3, dtype=np.float32).reshape(8, 8, 8)
    reference_weight = np.ones_like(reference)
    moving_weight = -np.ones_like(reference)
    args = (
        reference, reference.copy(), reference_weight, moving_weight,
        np.eye(4), np.eye(4),
    )
    blocked = flirt_blocked.FlirtBlockedCorrelationRatio(*args, bins=16)
    serial = flirt_blocked.FlirtWeightedCorrelationRatio(*args, bins=16)
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    transforms[1, :3, 3] = [0.2, -0.1, 0.3]
    expected = serial.evaluate_many(transforms, workers=1)
    np.testing.assert_array_equal(expected, np.ones(2))
    np.testing.assert_array_equal(blocked.evaluate_many(transforms, workers=1), expected)
