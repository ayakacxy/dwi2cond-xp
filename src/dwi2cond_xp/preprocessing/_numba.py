"""Small runtime helpers shared by Numba-backed preprocessing kernels."""

from __future__ import annotations

from numba import config, set_num_threads


def set_available_numba_threads(workers: int) -> int:
    """Use at most the thread slots exposed by the current Numba runtime."""

    active = min(int(workers), int(config.NUMBA_NUM_THREADS))
    set_num_threads(active)
    return active
