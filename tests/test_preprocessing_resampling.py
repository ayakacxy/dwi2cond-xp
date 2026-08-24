import numpy as np
import pytest

import dwi2cond_xp.preprocessing.resampling as resampling_module
from dwi2cond_xp.preprocessing.resampling import (
    output_to_input_voxel_matrix,
    resample_image,
    resample_mask,
)


def test_optimized_sinc_is_bitwise_equal_to_reference():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(8, 7, 6)).astype(np.float32)
    coordinates = rng.uniform([-1, -1, -1], [8, 7, 6], size=(128, 3)).T
    reference = resampling_module._fsl_sinc_sample_reference(
        source, coordinates, -2.0
    )
    optimized = resampling_module._fsl_sinc_sample(source, coordinates, -2.0)
    np.testing.assert_array_equal(optimized, reference)
    weights = resampling_module._fsl_sinc_kernel_values(
        np.array([-4.0, 0.0, 4.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(weights[[0, 2]], 0.0)
    assert weights[1] == 1.0


def test_output_to_input_matrix_and_identity_resampling():
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    np.testing.assert_allclose(
        output_to_input_voxel_matrix(affine, affine, np.eye(4)), np.eye(4)
    )
    image = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    for interpolation in ("nearest", "linear", "spline", "sinc"):
        actual = resample_image(
            image, affine, image.shape, affine, np.eye(4), interpolation=interpolation, z_chunk=2
        )
        np.testing.assert_allclose(actual, image, atol=2e-5)


def test_affine_displacement_and_channel_resampling():
    image = np.indices((5, 4, 3), dtype=np.float32)[0]
    channels = np.stack((image, image + 10), axis=-1)
    transform = np.eye(4)
    transform[0, 3] = 1.0
    moved = resample_image(
        channels, np.eye(4), image.shape, np.eye(4), transform, z_chunk=1
    )
    np.testing.assert_array_equal(moved[1:, ..., 0], image[:-1])
    np.testing.assert_array_equal(moved[1:, ..., 1], image[:-1] + 10)
    np.testing.assert_array_equal(moved[0], 0)

    displacement = np.zeros(image.shape + (3,), dtype=np.float64)
    displacement[..., 0] = 1.0
    displaced = resample_image(
        image,
        np.eye(4),
        image.shape,
        np.eye(4),
        np.eye(4),
        reference_to_moving_displacement=displacement,
        interpolation="nearest",
    )
    np.testing.assert_array_equal(displaced[:-1], image[1:])
    np.testing.assert_array_equal(displaced[-1], 0)


def test_mask_is_binary_and_nearest_only():
    mask = np.zeros((4, 4, 4), dtype=float)
    mask[1:3, 1:3, 1:3] = 2.0
    actual = resample_mask(mask, np.eye(4), mask.shape, np.eye(4), np.eye(4), z_chunk=2)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, mask > 0)
    with pytest.raises(ValueError, match="three-dimensional"):
        resample_mask(mask[..., None], np.eye(4), mask.shape, np.eye(4), np.eye(4))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference_shape": (2, 2)}, "three positive"),
        ({"moving_affine": np.zeros((4, 4))}, "invertible"),
        ({"moving_affine": np.zeros((3, 3))}, "finite 4x4"),
        ({"moving": np.zeros((2, 2))}, "three positive"),
        ({"moving": np.zeros((2, 2, 2, 1, 1))}, "3D or channel-last 4D"),
        ({"interpolation": "lanczos"}, "nearest, linear, spline, or sinc"),
        ({"cval": np.inf}, "cval must be finite"),
        ({"z_chunk": 0}, "z_chunk must be positive"),
        ({"reference_to_moving_displacement": np.zeros((2, 2, 2, 2))}, "match the reference"),
        ({"linear_extrapolation": "nearest"}, "partial, constant, or fsl"),
    ],
)
def test_resampling_validation(kwargs, message):
    arguments = {
        "moving": np.zeros((2, 2, 2)),
        "moving_affine": np.eye(4),
        "reference_shape": (2, 2, 2),
        "reference_affine": np.eye(4),
        "world_transform": np.eye(4),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        resample_image(**arguments)
