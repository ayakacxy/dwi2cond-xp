from __future__ import annotations

import numpy as np
import pytest

import dwi2cond_xp.preprocessing.flirt_pyramid as pyramid_module
from dwi2cond_xp.preprocessing import (
    build_flirt_pyramid,
    flirt_blur,
    isotropic_resample,
    subsample_by_two,
)


def _volume(shape: tuple[int, int, int] = (9, 8, 7)) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float32)
    return np.asarray(grid[0] + 2.0 * grid[1] + 3.0 * grid[2], dtype=np.float32)


def test_blur_identity_and_filtered_paths() -> None:
    volume = _volume()
    identity = flirt_blur(volume, np.ones(3), 1.0)
    np.testing.assert_array_equal(identity, volume)
    filtered = flirt_blur(volume, np.ones(3), 4.0, padding=-2.0)
    assert filtered.shape == volume.shape
    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))
    assert not np.array_equal(filtered, volume)
    assert pyramid_module._blur_kernel(1.05, 1.0).tolist() == [1.0]
    assert pyramid_module._blur_kernel(1.11, 1.0).tolist() == [1.0]
    assert np.isfinite(pyramid_module._background_value(volume))
    direct = pyramid_module._convolve_axis.py_func(
        volume,
        pyramid_module._blur_kernel(4.0, 1.0),
        0,
        np.float32(-2.0),
    )
    assert direct.shape == volume.shape


def test_isotropic_and_subsample_preserve_fsl_grid_contract() -> None:
    volume = _volume((6, 5, 4))
    sampling = np.diag([1.0, 1.0, 1.0, 1.0])
    isotropic, isotropic_sampling = isotropic_resample(volume, sampling, 1.0)
    np.testing.assert_array_equal(isotropic, volume)
    np.testing.assert_array_equal(isotropic_sampling, sampling)
    assert pyramid_module._linear_sample.py_func(
        volume,
        np.float32(1.25),
        np.float32(1.5),
        np.float32(1.75),
        np.float32(-1.0),
    ) == pytest.approx(9.5)
    direct_isotropic = pyramid_module._isotropic_kernel.py_func(
        volume,
        np.ones(3, dtype=np.float32),
        np.asarray(volume.shape, dtype=np.int64),
        np.float32(0.0),
    )
    np.testing.assert_array_equal(direct_isotropic, volume)

    subsampled, subsampled_sampling = subsample_by_two(volume, sampling, padding=-1.0)
    expected = pyramid_module._subsample_kernel.py_func(volume, np.float32(-1.0))
    np.testing.assert_array_equal(subsampled, expected)
    assert subsampled.shape == (3, 3, 2)
    np.testing.assert_array_equal(
        subsampled_sampling, np.diag([2.0, 2.0, 2.0, 1.0])
    )


def test_pyramid_reuses_active_scale_and_keeps_values_finite() -> None:
    reference = _volume((12, 11, 10))
    moving = np.asarray(reference * np.float32(0.7) + np.float32(2.0), dtype=np.float32)
    reference_weight = np.ones_like(reference)
    reference_weight[:2] = 0.0
    moving_weight = np.linspace(0.2, 1.0, moving.size, dtype=np.float32).reshape(
        moving.shape
    )
    sampling = np.diag([2.0, 2.0, 2.0, 1.0])
    levels = build_flirt_pyramid(
        reference,
        moving,
        reference_weight,
        moving_weight,
        sampling,
        sampling,
    )
    assert list(levels) == [8, 4, 2, 1]
    assert levels[1].reference is levels[2].reference
    assert levels[1].moving is levels[2].moving
    assert levels[1].bins == levels[2].bins == 128
    assert [levels[scale].requested_scale for scale in (8, 4, 2, 1)] == [8.0, 4.0, 2.0, 1.0]
    for level in levels.values():
        assert level.reference.flags.f_contiguous
        assert level.moving.flags.f_contiguous
        assert level.reference.shape == level.reference_weight.shape
        assert level.moving.shape == level.moving_weight.shape
        assert np.all(np.isfinite(level.reference))
        assert np.all(np.isfinite(level.moving))


def test_unweighted_pyramid_uses_background_padding_without_mask_normalization() -> None:
    reference = _volume((9, 8, 7)) + np.float32(20.0)
    moving = np.asarray(reference * np.float32(0.8), dtype=np.float32)
    weights = np.ones_like(reference)
    weights[0] = 0.0
    sampling = np.eye(4)
    unweighted = build_flirt_pyramid(
        reference,
        moving,
        weights,
        weights,
        sampling,
        sampling,
        use_weights=False,
    )
    weighted = build_flirt_pyramid(
        reference,
        moving,
        weights,
        weights,
        sampling,
        sampling,
        use_weights=True,
    )
    assert np.all(unweighted[8].reference_weight == 1.0)
    assert np.all(unweighted[8].moving_weight == 1.0)
    assert not np.array_equal(unweighted[8].reference, weighted[8].reference)
    assert not np.array_equal(unweighted[8].moving, weighted[8].moving)


def test_filter_helpers_cover_binary_and_nonbinary_weights() -> None:
    image = _volume((4, 4, 4))
    binary = np.ones_like(image)
    weighted = np.asarray(binary * np.float32(0.5), dtype=np.float32)

    def operation(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    np.testing.assert_array_equal(
        pyramid_module._filter_image(image, binary, operation), image
    )
    np.testing.assert_array_equal(
        pyramid_module._filter_weight(binary, operation), binary
    )
    filtered_image, filtered_weight = pyramid_module._subsample_image_and_weight(
        image, weighted
    )
    assert filtered_image.shape == filtered_weight.shape == (2, 2, 2)
    assert np.all(np.isfinite(filtered_image))


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (flirt_blur, (np.zeros((2, 2)), np.ones(3), 2.0), "finite 3D"),
        (flirt_blur, (np.zeros((2, 2, 2)), np.ones(2), 2.0), "three positive"),
        (flirt_blur, (np.zeros((2, 2, 2)), np.ones(3), 0.0), "positive"),
        (flirt_blur, (np.zeros((2, 2, 2)), np.ones(3), 2.0), "padding"),
        (isotropic_resample, (np.zeros((2, 2, 2)), np.eye(3), 1.0), "finite 4x4"),
        (isotropic_resample, (np.zeros((2, 2, 2)), np.eye(4), 0.0), "positive"),
        (subsample_by_two, (np.zeros((2, 2, 2)), np.zeros((4, 4))), "invertible"),
    ],
)
def test_pyramid_validation(function, arguments, message) -> None:
    keyword = {"padding": np.inf} if message == "padding" else {}
    with pytest.raises(ValueError, match=message):
        function(*arguments, **keyword)


def test_build_pyramid_rejects_mismatched_weights() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="weight must match"):
        build_flirt_pyramid(
            volume,
            volume,
            np.zeros((2, 3, 3), dtype=np.float32),
            volume,
            np.eye(4),
            np.eye(4),
        )
