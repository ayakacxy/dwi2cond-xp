import os
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pytest

import dwi2cond_xp.preprocessing.resampling as resampling_module
from dwi2cond_xp.preprocessing.resampling import (
    output_to_input_voxel_matrix,
    resample_image,
    resample_mask,
)
from dwi2cond_xp.preprocessing.transforms import fsl_matrix_to_world


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


def test_displacement_is_composed_before_inverse_affine():
    image = np.indices((9, 3, 2), dtype=np.float32)[0]
    transform = np.diag([2.0, 1.0, 1.0, 1.0])
    displacement = np.zeros(image.shape + (3,), dtype=np.float64)
    displacement[..., 0] = 2.0

    actual = resample_image(
        image,
        np.eye(4),
        image.shape,
        np.eye(4),
        transform,
        reference_to_moving_displacement=displacement,
        interpolation="linear",
        linear_extrapolation="constant",
    )

    expected_x = np.arange(image.shape[0], dtype=np.float32) / 2.0 + 1.0
    np.testing.assert_allclose(actual[:, 1, 1], expected_x, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(
    not os.environ.get("FSL_CONVERTWARP") or not os.environ.get("FSL_APPLYWARP"),
    reason="FSL GRE composition reference is not configured",
)
def test_affine_displacement_composition_matches_real_fsl(tmp_path: Path) -> None:
    shape = (9, 9, 3)
    grid = np.indices(shape, dtype=np.float32)
    values = np.asarray(
        100.0 * grid[0] + grid[1] + 0.01 * grid[2], dtype=np.float32
    )
    affine = np.eye(4)
    for name, data in (
        ("input.nii.gz", values),
        ("reference.nii.gz", np.zeros(shape, dtype=np.float32)),
        ("shift.nii.gz", np.ones(shape, dtype=np.float32)),
    ):
        image = nib.Nifti1Image(data, affine)
        image.set_qform(affine, 1)
        image.set_sform(affine, 1)
        nib.save(image, tmp_path / name)
    premat = np.array(
        [[0.0, -1.0, 0.0, 8.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    np.savetxt(tmp_path / "premat.mat", premat, fmt="%.17g")
    environment = os.environ.copy()
    environment.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    subprocess.run(
        [
            os.environ["FSL_CONVERTWARP"],
            "-s",
            str(tmp_path / "shift.nii.gz"),
            "-o",
            str(tmp_path / "warp.nii.gz"),
            "-r",
            str(tmp_path / "reference.nii.gz"),
            "--shiftdir=y",
        ],
        check=True,
        env=environment,
    )
    subprocess.run(
        [
            os.environ["FSL_APPLYWARP"],
            "-i",
            str(tmp_path / "input.nii.gz"),
            "-r",
            str(tmp_path / "reference.nii.gz"),
            "-o",
            str(tmp_path / "fsl.nii.gz"),
            "-w",
            str(tmp_path / "warp.nii.gz"),
            "--abs",
            f"--premat={tmp_path / 'premat.mat'}",
            "--interp=trilinear",
        ],
        check=True,
        env=environment,
    )
    world_transform = fsl_matrix_to_world(
        premat, shape, affine, shape, affine
    )
    displacement = np.zeros(shape + (3,), dtype=np.float64)
    displacement[..., 1] = 1.0
    candidate = resample_image(
        values,
        affine,
        shape,
        affine,
        world_transform,
        reference_to_moving_displacement=displacement,
        interpolation="linear",
        linear_extrapolation="fsl",
    )
    reference = np.asarray(nib.load(tmp_path / "fsl.nii.gz").dataobj)
    np.testing.assert_array_equal(candidate, reference)


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
