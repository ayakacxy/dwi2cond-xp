import numpy as np
import pytest

from dwi2cond_xp.preprocessing.transforms import (
    affine_matrices,
    affine_matrix,
    compose_transforms,
    decompose_affine,
    fsl_matrix_to_world,
    fsl_voxel_to_scaled_mm,
    invert_transform,
    rigid_matrix,
    world_matrix_to_fsl,
)


def test_batched_affine_construction_is_bitwise_equal_for_every_dof():
    random = np.random.default_rng(83)
    center = np.array([3.2, -4.1, 8.7])
    for degrees_of_freedom in range(6, 13):
        parameters = random.normal(size=(17, degrees_of_freedom))
        if degrees_of_freedom > 6:
            scale_count = min(9, degrees_of_freedom) - 6
            parameters[:, 6 : 6 + scale_count] = random.uniform(
                0.7, 1.3, size=(17, scale_count)
            )
        direct = np.asarray(
            [affine_matrix(values, center) for values in parameters]
        )
        np.testing.assert_array_equal(
            affine_matrices(parameters, center), direct
        )


@pytest.mark.parametrize(
    ("parameters", "center", "message"),
    [
        (np.zeros(6), None, "shape"),
        (np.zeros((2, 5)), None, "shape"),
        (np.full((2, 6), np.nan), None, "finite"),
        (np.zeros((2, 6)), np.zeros(2), "three-element"),
        (np.c_[np.zeros((2, 6)), np.zeros(2)], None, "nonzero"),
    ],
)
def test_batched_affine_validation(parameters, center, message):
    with pytest.raises(ValueError, match=message):
        affine_matrices(parameters, center)


def test_rigid_affine_inverse_and_composition_contracts():
    center = np.array([4.0, 5.0, 6.0])
    parameters = np.array([0.1, -0.2, 0.3, 1.0, 2.0, -3.0])
    rigid = rigid_matrix(parameters, center)
    np.testing.assert_allclose(affine_matrix(parameters, center), rigid)
    np.testing.assert_allclose(rigid[:3, :3] @ rigid[:3, :3].T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(
        rigid @ np.r_[center, 1.0], np.r_[center + parameters[3:], 1.0]
    )

    second = np.eye(4)
    second[:3, 3] = [7.0, 8.0, 9.0]
    composed = compose_transforms(rigid, second)
    np.testing.assert_allclose(composed, second @ rigid)
    np.testing.assert_allclose(
        invert_transform(composed) @ composed, np.eye(4), atol=1e-14
    )
    np.testing.assert_array_equal(compose_transforms(), np.eye(4))


def test_twelve_dof_affine_matches_fsl_composition():
    center = np.array([3.0, 4.0, 5.0])
    parameters = np.array(
        [0.03, -0.04, 0.02, 2.0, -1.0, 0.5, 1.1, 0.9, 1.2, 0.1, -0.2, 0.3]
    )
    actual = affine_matrix(parameters, center)
    scale = np.diag([1.1, 0.9, 1.2, 1.0])
    scale[:3, 3] = center - scale[:3, :3] @ center
    skew = np.eye(4)
    skew[0, 1], skew[0, 2], skew[1, 2] = [0.1, -0.2, 0.3]
    skew[:3, 3] = center - skew[:3, :3] @ center
    expected = rigid_matrix(parameters[:6], center) @ skew @ scale
    np.testing.assert_allclose(actual, expected)
    recovered = decompose_affine(actual, center)
    np.testing.assert_allclose(recovered, parameters, atol=2e-7)

    seven = affine_matrix(parameters[:7], center)
    isotropic = np.diag([1.1, 1.1, 1.1, 1.0])
    isotropic[:3, 3] = center - isotropic[:3, :3] @ center
    np.testing.assert_allclose(seven, rigid_matrix(parameters[:6], center) @ isotropic)
    nine = affine_matrix(parameters[:9], center)
    np.testing.assert_allclose(nine, rigid_matrix(parameters[:6], center) @ scale)


def test_fsl_scaled_mm_handedness_and_world_round_trip():
    shape = (11, 12, 13)
    neurological = np.diag([2.0, 3.0, 4.0, 1.0])
    neurological[:3, 3] = [-10.0, 2.0, 4.0]
    scaled = fsl_voxel_to_scaled_mm(shape, neurological)
    np.testing.assert_array_equal(
        scaled,
        np.array([[-2.0, 0, 0, 20.0], [0, 3.0, 0, 0], [0, 0, 4.0, 0], [0, 0, 0, 1]]),
    )
    radiological = neurological.copy()
    radiological[0, 0] = -2.0
    np.testing.assert_array_equal(
        fsl_voxel_to_scaled_mm(shape, radiological), np.diag([2.0, 3.0, 4.0, 1.0])
    )

    moving_shape = (11, 12, 13)
    reference_shape = (9, 10, 8)
    moving_affine = neurological
    reference_affine = np.array(
        [[-1.5, 0, 0, 20], [0, 2.5, 0, -3], [0, 0, 3.5, 8], [0, 0, 0, 1]],
        dtype=float,
    )
    fsl = affine_matrix(
        np.array([0.02, -0.01, 0.03, 4, -2, 1, 1.05, 0.97, 1.02, 0.01, -0.02, 0.03])
    )
    world = fsl_matrix_to_world(
        fsl, moving_shape, moving_affine, reference_shape, reference_affine
    )
    recovered = world_matrix_to_fsl(
        world, moving_shape, moving_affine, reference_shape, reference_affine
    )
    np.testing.assert_allclose(recovered, fsl, atol=1e-13)


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (invert_transform, (np.zeros((3, 3)),), "finite 4x4"),
        (invert_transform, (np.zeros((4, 4)),), "invertible"),
        (rigid_matrix, (np.zeros(5),), "six-element"),
        (rigid_matrix, (np.zeros(6), np.zeros(2)), "three-element"),
        (affine_matrix, (np.zeros(5),), "six through twelve"),
        (affine_matrix, (np.zeros(13),), "six through twelve"),
        (decompose_affine, (np.eye(4), np.zeros(2)), "three-element"),
        (affine_matrix, (np.r_[np.zeros(6), 0, 1, 1, 0, 0, 0],), "nonzero"),
        (affine_matrix, (np.zeros(6), np.zeros(2)), "three-element"),
        (fsl_voxel_to_scaled_mm, ((1, 2), np.eye(4)), "three positive"),
    ],
)
def test_transform_validation(function, args, message):
    with pytest.raises(ValueError, match=message):
        function(*args)
