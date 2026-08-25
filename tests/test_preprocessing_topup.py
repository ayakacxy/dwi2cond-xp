from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing.topup import (
    HESSIAN_PRECISION,
    IMAGE_INTERPOLATION,
    REGULARIZATION_MODEL,
    SCALE_IMAGES_INDIVIDUALLY,
    SCALE_REGULARIZATION_BY_SSD,
    SIMNIBS46_TOPUP_LEVELS,
    TopupFixedMovementObjective,
    TopupRunResult,
    bending_energy,
    bending_energy_hessian,
    cubic_spline_value,
    expand_spline_coefficients,
    field_displacement_voxels,
    field_jacobian,
    fsl_coefficient_shape,
    fsl_knot_spacing,
    fsl_regrid_topup_scan,
    prepare_topup_scan,
    resample_topup_scan,
    run_topup_nifti,
    spline_design_matrix,
    topup_pair_cost,
    validate_acquisition_parameters,
)
from dwi2cond_xp.preprocessing import topup
from dwi2cond_xp.preprocessing.topup import (
    _balanced_spline_hessian_workset,
    _expand_coefficients_fsl_order,
    _expand_coefficients_fsl_order_voxel_parallel,
    _fsl_periodic_cubic_coefficients,
    _movement_interaction_jte_fsl_order,
    _paired_spline_jte_fsl_order,
    _sample_fsl_periodic_cubic,
    _sample_fsl_periodic_cubic_affine,
    _sample_fsl_periodic_cubic_affine_batch,
    _spline_hessian_workset,
    _spline_jte_fsl_order,
    _topup_movement_matrix,
    _topup_matrix_to_movement_parameters,
)


def test_simnibs46_schedule_is_frozen() -> None:
    assert len(SIMNIBS46_TOPUP_LEVELS) == 9
    assert [level.warp_resolution_mm for level in SIMNIBS46_TOPUP_LEVELS] == [
        20,
        16,
        14,
        12,
        10,
        6,
        4,
        4,
        4,
    ]
    assert [level.subsampling for level in SIMNIBS46_TOPUP_LEVELS] == [1] * 9
    assert [level.max_iterations for level in SIMNIBS46_TOPUP_LEVELS] == [
        5,
        5,
        5,
        5,
        5,
        10,
        10,
        20,
        20,
    ]
    assert [level.estimate_movements for level in SIMNIBS46_TOPUP_LEVELS] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert REGULARIZATION_MODEL == "bending_energy"
    assert SCALE_IMAGES_INDIVIDUALLY is True
    assert SCALE_REGULARIZATION_BY_SSD is True
    assert HESSIAN_PRECISION == "double"
    assert IMAGE_INTERPOLATION == "spline"


def test_acquisition_parameter_contract() -> None:
    values = validate_acquisition_parameters(
        np.array([[0, 1, 0, 0.05], [0, -1, 0, 0.05]]), number_of_volumes=2
    )
    assert values.dtype == np.float64
    assert values.flags.owndata
    for invalid, message in (
        (np.ones(4), "shape"),
        (np.ones((2, 3)), "shape"),
        (np.array([[0, 0, 0, 0.05]]), "exactly one"),
        (np.array([[1, 1, 0, 0.05]]), "exactly one"),
        (np.array([[0, 0, 1, 0.05]]), "does not support"),
        (np.array([[1, 0, 0, 0.0]]), "positive"),
        (np.array([[1, 0, 0, np.nan]]), "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_acquisition_parameters(invalid)
    with pytest.raises(ValueError, match="one acquisition"):
        validate_acquisition_parameters(values, number_of_volumes=3)


def test_movement_matrix_matches_fsl604_float_trigonometry() -> None:
    movement = np.asarray((0.1, -0.2, 0.3, 0.001, -0.002, 0.003))
    actual = _topup_movement_matrix(movement, (26, 26, 18), (2.0, 2.0, 2.5))
    expected = np.asarray(
        (
            (
                0.9999934081085004,
                0.002999989237646256,
                0.0019999984732520233,
                -0.017334901210270565,
            ),
            (
                -0.003001993876420777,
                0.9999949518222027,
                0.0009999978871396953,
                -0.14607390374626453,
            ),
            (
                -0.001996988808540155,
                -0.0010059953501772264,
                0.9999974792212415,
                0.37512817051655284,
            ),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3.0e-17)


def test_matrix_to_movement_parameters_reconstructs_rigid_matrix() -> None:
    movement = np.asarray((0.1, -0.2, 0.3, 0.001, -0.002, 0.003))
    shape = (26, 26, 18)
    voxel_sizes = (2.0, 2.0, 2.5)
    matrix = _topup_movement_matrix(movement, shape, voxel_sizes)
    recovered = _topup_matrix_to_movement_parameters(matrix, shape, voxel_sizes)
    reconstructed = _topup_movement_matrix(recovered, shape, voxel_sizes)
    # Matrix2MovePar intentionally narrows Euler intermediates to float, so a
    # matrix-to-parameters-to-matrix round trip is only float accurate.
    np.testing.assert_allclose(reconstructed, matrix, rtol=0.0, atol=1.0e-7)


def test_fsl_knot_and_coefficient_geometry() -> None:
    assert fsl_knot_spacing(4.0, (2.0, 2.2, 2.5)) == (2, 2, 2)
    assert fsl_coefficient_shape((16, 14, 12), (2, 2, 2)) == (11, 10, 9)
    assert fsl_coefficient_shape((4, 5, 6), (1, 1, 1)) == (4, 5, 6)
    with pytest.raises(ValueError, match="warp resolution"):
        fsl_knot_spacing(0.0, (1, 1, 1))
    with pytest.raises(ValueError, match="three finite"):
        fsl_knot_spacing(4.0, (1, 1))
    with pytest.raises(ValueError, match="positive"):
        fsl_knot_spacing(4.0, (1, -1, 1))
    with pytest.raises(ValueError, match="three-dimensional"):
        fsl_coefficient_shape((2, 2), (1, 1, 1))
    with pytest.raises(ValueError, match="positive"):
        fsl_coefficient_shape((2, 0, 2), (1, 1, 1))


def test_cubic_basis_and_derivatives() -> None:
    assert cubic_spline_value(0, 1, 2) == pytest.approx(2.0 / 3.0)
    assert cubic_spline_value(2, 1, 2) == pytest.approx(1.0 / 6.0)
    assert cubic_spline_value(-2, 1, 2) == pytest.approx(1.0 / 6.0)
    assert cubic_spline_value(4, 1, 2) == 0.0
    assert cubic_spline_value(0, 1, 2, 1) == 0.0
    assert cubic_spline_value(1, 1, 2, 1) == pytest.approx(-0.3125)
    assert cubic_spline_value(-1, 1, 2, 1) == pytest.approx(0.3125)
    assert cubic_spline_value(0, 1, 2, 2) == pytest.approx(-0.5)
    assert cubic_spline_value(3, 1, 2, 2) == pytest.approx(0.125)
    with pytest.raises(ValueError, match="positive"):
        cubic_spline_value(0, 0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        cubic_spline_value(0, -1, 1)
    with pytest.raises(ValueError, match="derivative orders"):
        cubic_spline_value(0, 0, 1, 3)


def test_spline_expansion_matches_explicit_tensor_product() -> None:
    field_shape = (5, 4, 3)
    spacing = (2, 2, 2)
    coefficient_shape = fsl_coefficient_shape(field_shape, spacing)
    coefficients = np.arange(np.prod(coefficient_shape), dtype=np.float64).reshape(
        coefficient_shape, order="F"
    )
    basis = tuple(
        spline_design_matrix(field_shape[axis], spacing[axis]) for axis in range(3)
    )
    expected = np.zeros(field_shape, dtype=np.float64)
    for z in range(coefficient_shape[2]):
        for y in range(coefficient_shape[1]):
            for x in range(coefficient_shape[0]):
                for voxel_z in range(field_shape[2]):
                    for voxel_y in range(field_shape[1]):
                        for voxel_x in range(field_shape[0]):
                            kernel = (
                                basis[2][voxel_z, z]
                                * basis[1][voxel_y, y]
                                * basis[0][voxel_x, x]
                            )
                            expected[voxel_x, voxel_y, voxel_z] += (
                                coefficients[x, y, z] * kernel
                            )
    actual = expand_spline_coefficients(coefficients, field_shape, spacing)
    assert np.array_equal(actual, expected)
    derivative = expand_spline_coefficients(
        coefficients, field_shape, spacing, derivative_axis=1
    )
    derivative_basis = (
        basis[0],
        spline_design_matrix(field_shape[1], spacing[1], derivative=1),
        basis[2],
    )
    reference_derivative = _expand_coefficients_fsl_order(
        coefficients, *derivative_basis
    )
    optimized_derivative = _expand_coefficients_fsl_order_voxel_parallel(
        coefficients, *derivative_basis
    )
    assert derivative.shape == field_shape
    assert np.all(np.isfinite(derivative))
    assert np.array_equal(optimized_derivative, reference_derivative)
    with pytest.raises(ValueError, match="coefficient shape"):
        expand_spline_coefficients(coefficients[:-1], field_shape, spacing)
    invalid = coefficients.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        expand_spline_coefficients(invalid, field_shape, spacing)
    with pytest.raises(ValueError, match="derivative axis"):
        expand_spline_coefficients(
            coefficients, field_shape, spacing, derivative_axis=3
        )


def test_displacement_jacobian_and_pair_cost() -> None:
    shape = (5, 6, 4)
    spacing = (2, 2, 2)
    coefficient_shape = fsl_coefficient_shape(shape, spacing)
    coefficients = np.full(coefficient_shape, 8.0, dtype=np.float64)
    field = expand_spline_coefficients(coefficients, shape, spacing)
    row = np.array([0.0, -1.0, 0.0, 0.05])
    displacement = field_displacement_voxels(field, row)
    assert displacement.shape == shape + (3,)
    assert np.array_equal(displacement[..., 0], np.zeros(shape))
    assert np.array_equal(displacement[..., 2], np.zeros(shape))
    assert np.all(displacement[..., 1] <= 0.0)
    jacobian = field_jacobian(
        field,
        row,
        field_coefficients=coefficients,
        knot_spacing=spacing,
    )
    assert np.allclose(jacobian, 1.0, rtol=0.0, atol=1e-15)
    with pytest.raises(ValueError, match="required"):
        field_jacobian(field, row)

    scan = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    corrected = np.stack((scan, scan + np.float32(2.0)), axis=3)
    masks = np.ones(corrected.shape, dtype=np.uint8)
    assert topup_pair_cost(corrected, corrected, masks) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="share a 4D"):
        topup_pair_cost(scan, corrected, masks)
    with pytest.raises(ValueError, match="at least two"):
        topup_pair_cost(corrected[..., :1], corrected[..., :1], masks[..., :1])
    masks.fill(0)
    with pytest.raises(ValueError, match="empty"):
        topup_pair_cost(corrected, corrected, masks)


def test_bending_energy_hessian_contract() -> None:
    shape = (5, 4, 3)
    spacing = (2, 2, 2)
    coefficient_shape = fsl_coefficient_shape(shape, spacing)
    hessian = bending_energy_hessian(shape, (2.0, 2.2, 2.5), spacing)
    assert hessian.shape == (np.prod(coefficient_shape),) * 2
    asymmetry = (hessian - hessian.T).data
    assert asymmetry.size == 0 or np.max(np.abs(asymmetry)) < 1e-15
    coefficients = np.arange(np.prod(coefficient_shape), dtype=np.float64).reshape(
        coefficient_shape, order="F"
    )
    energy = bending_energy(
        coefficients, shape, (2.0, 2.2, 2.5), spacing, hessian=hessian
    )
    assert energy >= 0.0
    assert (
        bending_energy(np.zeros(coefficient_shape), shape, (2.0, 2.2, 2.5), spacing)
        == 0.0
    )
    with pytest.raises(ValueError, match="three finite"):
        bending_energy_hessian(shape, (1.0, 2.0), spacing)
    with pytest.raises(ValueError, match="positive"):
        bending_energy_hessian(shape, (1.0, -2.0, 3.0), spacing)
    with pytest.raises(ValueError, match="finite with shape"):
        bending_energy(coefficients[:-1], shape, (2.0, 2.2, 2.5), spacing)


def test_hessian_workset_preserves_ordered_overlap_pairs() -> None:
    shape = (5, 4, 3)
    spacing = (2, 2, 2)
    coefficient_shape = fsl_coefficient_shape(shape, spacing)
    *_supports, left, right = _spline_hessian_workset(shape, spacing, 1)
    expected = []
    nx, ny, nz = coefficient_shape
    for left_index in range(nx * ny * nz):
        left_x = left_index % nx
        left_y = (left_index // nx) % ny
        left_z = left_index // (nx * ny)
        for right_z in range(max(0, left_z - 3), min(nz, left_z + 4)):
            for right_y in range(max(0, left_y - 3), min(ny, left_y + 4)):
                for right_x in range(max(0, left_x - 3), min(nx, left_x + 4)):
                    expected.append(
                        (left_index, right_z * nx * ny + right_y * nx + right_x)
                    )
    assert list(zip(left.tolist(), right.tolist(), strict=True)) == expected


def test_balanced_hessian_workset_preserves_all_ordered_pairs() -> None:
    shape = (7, 6, 5)
    spacing = (2, 2, 2)
    *reference_supports, reference_left, reference_right = _spline_hessian_workset(
        shape, spacing, 1
    )
    *balanced_supports, balanced_left, balanced_right = (
        _balanced_spline_hessian_workset(shape, spacing, 1, 8)
    )
    for actual, expected in zip(
        balanced_supports, reference_supports, strict=True
    ):
        assert actual is expected
    reference_pairs = sorted(zip(reference_left.tolist(), reference_right.tolist()))
    balanced_pairs = sorted(zip(balanced_left.tolist(), balanced_right.tolist()))
    assert balanced_pairs == reference_pairs


def test_paired_spline_transpose_is_bitwise_equal_to_separate_calls() -> None:
    rng = np.random.default_rng(4401)
    shape = (7, 6, 5)
    spacing = (2, 2, 2)
    direct_image = rng.normal(size=shape).astype(np.float32)
    derivative_image = rng.normal(size=shape).astype(np.float32)
    mask = (rng.random(shape) > 0.2).astype(np.uint8)
    basis = tuple(
        spline_design_matrix(shape[axis], spacing[axis]) for axis in range(3)
    )
    derivative_basis = tuple(
        spline_design_matrix(
            shape[axis], spacing[axis], derivative=1 if axis == 1 else 0
        )
        for axis in range(3)
    )
    expected_direct = _spline_jte_fsl_order(direct_image, mask, *basis)
    expected_derivative = _spline_jte_fsl_order(
        derivative_image, mask, *derivative_basis
    )
    actual_direct, actual_derivative = _paired_spline_jte_fsl_order(
        direct_image,
        derivative_image,
        mask,
        *basis,
        *derivative_basis,
    )
    assert np.array_equal(actual_direct, expected_direct)
    assert np.array_equal(actual_derivative, expected_derivative)


def test_batched_movement_interaction_is_bitwise_equal_to_separate_columns() -> None:
    rng = np.random.default_rng(5721)
    shape = (7, 6, 5)
    spacing = (2, 2, 2)
    movement = rng.normal(size=(*shape, 5)).astype(np.float32)
    alpha = rng.normal(size=shape).astype(np.float32)
    axis_term = rng.normal(size=shape).astype(np.float32)
    mask = (rng.random(shape) > 0.2).astype(np.uint8)
    basis = tuple(
        spline_design_matrix(shape[axis], spacing[axis]) for axis in range(3)
    )
    derivative_basis = tuple(
        spline_design_matrix(
            shape[axis], spacing[axis], derivative=1 if axis == 1 else 0
        )
        for axis in range(3)
    )
    direct, derivative = _movement_interaction_jte_fsl_order(
        movement, alpha, axis_term, mask, *basis, *derivative_basis
    )

    for index in range(5):
        expected_direct = _spline_jte_fsl_order(
            movement[..., index] * alpha, mask, *basis
        ).reshape(-1, order="F")
        expected_derivative = _spline_jte_fsl_order(
            movement[..., index] * axis_term, mask, *derivative_basis
        ).reshape(-1, order="F")
        assert np.array_equal(direct[:, index], expected_direct)
        assert np.array_equal(derivative[:, index], expected_derivative)


def test_prepared_scan_reuses_interpolation_coefficients() -> None:
    shape = (6, 7, 5)
    scan = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    prepared = prepare_topup_scan(scan)
    field = np.zeros(shape, dtype=np.float64)
    jacobian = np.ones(shape, dtype=np.float64)
    row = np.array([0.0, 1.0, 0.0, 0.05])
    direct, direct_mask = resample_topup_scan(scan, field, row, jacobian)
    reused, reused_mask = resample_topup_scan(prepared, field, row, jacobian)
    assert np.array_equal(reused, direct)
    assert np.array_equal(reused_mask, direct_mask)
    assert prepared.values.flags.writeable is False
    assert prepared.interpolation_coefficients.flags.writeable is False
    with pytest.raises(ValueError, match="finite 3D"):
        prepare_topup_scan(np.ones((2, 2)))


def test_default_regrid_preserves_fsl_source_geometry() -> None:
    scan = np.full((6, 5, 4), 7.0, dtype=np.float32)
    regridded, voxel_sizes = fsl_regrid_topup_scan(scan, (2.0, 2.2, 2.5))

    assert regridded.shape == (7, 6, 5)
    np.testing.assert_allclose(
        voxel_sizes,
        ((5 * 2.0 - 1.0e-6) / 7, (4 * 2.2 - 1.0e-6) / 6, (3 * 2.5 - 1.0e-6) / 5),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.all(np.isfinite(regridded))
    assert float(np.max(np.abs(regridded - 7.0))) < 0.05


def test_topup_nifti_writes_public_artifacts(tmp_path, monkeypatch) -> None:
    shape = (6, 5, 4)
    affine = np.diag([-2.0, 2.2, 2.5, 1.0])
    forward_file = tmp_path / "forward.nii.gz"
    reverse_file = tmp_path / "reverse.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine), forward_file)
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.float32) * 2.0, affine), reverse_file
    )
    fake = TopupRunResult(
        np.ones((6, 6, 5), dtype=np.float64),
        np.full(shape, 3.0, dtype=np.float32),
        np.zeros((2, 6), dtype=np.float64),
        np.ones((*shape, 2), dtype=np.float32),
        np.ones(shape, dtype=np.uint8),
        (SimpleNamespace(cost=1.25),),
    )
    monkeypatch.setattr(
        "dwi2cond_xp.preprocessing.topup.run_simnibs46_topup",
        lambda *_args, **_kwargs: fake,
    )

    output = tmp_path / "output"
    report = run_topup_nifti(
        forward_file,
        reverse_file,
        output,
        readout_seconds=0.05,
        phase_encoding_direction="y",
    )

    assert report["status"] == "complete"
    assert report["level_costs"] == [1.25]
    for name in (
        "field_hz.nii.gz",
        "corrected_pair.nii.gz",
        "joint_mask.nii.gz",
        "field_coefficients.nii.gz",
        "movement_parameters.txt",
        "topup_qa.json",
    ):
        assert (output / name).is_file()
    assert nib.load(output / "corrected_pair.nii.gz").shape == (*shape, 2)


def test_periodic_cubic_sampler_matches_fsl604_oracle() -> None:
    image = np.empty((8, 3, 2), dtype=np.float32)
    for z in range(2):
        for y in range(3):
            for x in range(8):
                image[x, y, z] = 0.13 * x * x - 0.7 * y + 1.1 * z + 0.2 * x * y
    coefficients = _fsl_periodic_cubic_coefficients(image)
    coordinates = (-2.3, -0.2, 0.0, 0.2, 6.8, 7.0, 7.2, 8.2, 14.3)
    expected_values = (
        4.087724208831787,
        0.575584888458252,
        -0.6619706153869629,
        -1.2884396314620972,
        6.980093002319336,
        6.586437225341797,
        5.506863117218018,
        -1.288439393043518,
        5.900698184967041,
    )
    expected_dx = (
        1.8906086683273315,
        -7.4233222007751465,
        -4.7096357345581055,
        -1.698503017425537,
        -0.3785356283187866,
        -3.722259759902954,
        -6.830894470214844,
        -1.6985055208206177,
        3.6696557998657227,
    )
    values = []
    derivatives = []
    for coordinate in coordinates:
        displacement = np.zeros(image.shape, dtype=np.float64)
        displacement[0, 1, 0] = np.float32(coordinate)
        sampled, derivative_x, _derivative_y, _derivative_z = (
            _sample_fsl_periodic_cubic(coefficients, displacement, 0)
        )
        values.append(sampled[0, 1, 0])
        derivatives.append(derivative_x[0, 1, 0])

    np.testing.assert_allclose(values, expected_values, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(derivatives, expected_dx, rtol=0.0, atol=2.0e-6)


@pytest.mark.parametrize("phase_axis", (0, 1))
def test_batched_affine_sampler_is_bitwise_equal(phase_axis: int) -> None:
    rng = np.random.default_rng(72 + phase_axis)
    shape = (6, 5, 4)
    coefficients = rng.normal(size=(*shape, 2)).astype(np.float32)
    displacements = rng.normal(0.0, 0.2, size=(*shape, 2)).astype(np.float32)
    pull_matrices = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    pull_matrices[1, :3, 3] = (0.1, -0.15, 0.2)

    actual = _sample_fsl_periodic_cubic_affine_batch(
        coefficients,
        displacements,
        phase_axis,
        pull_matrices,
    )
    for scan_index in range(2):
        expected = _sample_fsl_periodic_cubic_affine(
            coefficients[..., scan_index],
            displacements[..., scan_index],
            phase_axis,
            pull_matrices[scan_index],
        )
        for actual_array, expected_array in zip(actual, expected, strict=True):
            assert np.array_equal(actual_array[..., scan_index], expected_array)


def test_fixed_movement_objective_gradient_matches_directional_difference() -> None:
    rng = np.random.default_rng(41)
    shape = (8, 7, 6)
    base = rng.normal(100.0, 5.0, size=shape).astype(np.float32)
    scans = np.stack((base, np.roll(base, 1, axis=1)), axis=3)
    objective = TopupFixedMovementObjective(
        scans,
        np.asarray([[0.0, 1.0, 0.0, 0.04], [0.0, -1.0, 0.0, 0.04]]),
        (2.0, 2.0, 2.0),
        6.0,
        0.0,
    )
    parameters = rng.normal(0.0, 0.05, size=objective.number_of_parameters)
    direction = rng.normal(size=parameters.size)
    direction /= np.linalg.norm(direction)
    gradient = objective.gradient(parameters)
    step = 2.0e-3
    numerical = (
        objective.cost(parameters + step * direction)
        - objective.cost(parameters - step * direction)
    ) / (2.0 * step)

    assert objective.cost(parameters) > 0.0
    assert float(gradient @ direction) == pytest.approx(
        numerical, rel=4.0e-2, abs=2.0e-3
    )


def test_csr_regularization_product_is_bitwise_equal_to_csc() -> None:
    rng = np.random.default_rng(44)
    scans = rng.normal(size=(8, 7, 6, 2)).astype(np.float32)
    objective = TopupFixedMovementObjective(
        scans,
        np.asarray([[0.0, 1.0, 0.0, 0.04], [0.0, -1.0, 0.0, 0.04]]),
        (2.0, 2.0, 2.0),
        4.0,
        1.0e-4,
    )
    parameters = rng.normal(size=objective.number_of_parameters)
    expected = np.asarray(objective.regularization_hessian @ parameters)

    actual = objective._apply_regularization_hessian(parameters)

    assert np.array_equal(actual, expected)


def test_topup_remaining_scalar_and_objective_contracts() -> None:
    """Cover defensive and extreme-geometry branches not reached by the full E2E path."""

    assert cubic_spline_value(100.0, 0, 2, 2) == 0.0
    assert topup._kernel_overlap(np.ones(2), np.ones(2), 3) == 0.0
    row = np.asarray([0.0, 1.0, 0.0, 0.05])
    with pytest.raises(ValueError, match="field must be"):
        field_displacement_voxels(np.ones((2, 2)), row)
    with pytest.raises(ValueError, match="field must be"):
        field_jacobian(np.ones((2, 2)), row)

    values = np.ones((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="matching 3D"):
        resample_topup_scan(values, np.ones((3, 3, 3)), row, values)
    invalid = values.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        resample_topup_scan(invalid, values, row, values)
    with pytest.raises(ValueError, match="scan must be"):
        prepare_topup_scan(np.ones((2, 2)))
    with pytest.raises(ValueError, match="scan must be"):
        fsl_regrid_topup_scan(np.ones((2, 2)), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="voxel sizes"):
        fsl_regrid_topup_scan(values, (1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="scan must be"):
        topup.fsl_smooth_topup_scan(np.ones((2, 2)), 1.0, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="FWHM"):
        topup.fsl_smooth_topup_scan(values, -1.0, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="voxel sizes"):
        topup.fsl_smooth_topup_scan(values, 1.0, (1.0, 0.0, 1.0))
    assert np.array_equal(
        topup.fsl_smooth_topup_scan(values, 0.0, (1.0, 1.0, 1.0)), values
    )
    one_worker = _balanced_spline_hessian_workset((4, 4, 4), (2, 2, 2), 1, 1)
    assert one_worker[3].size == one_worker[4].size

    scans = np.stack((values, values * 1.01), axis=3)
    rows = np.asarray([[0.0, 1.0, 0.0, 0.05], [0.0, -1.0, 0.0, 0.05]])
    invalid_calls = (
        lambda: TopupFixedMovementObjective(np.ones((2, 2, 2)), rows, (1, 1, 1), 2, 0),
        lambda: TopupFixedMovementObjective(scans, np.asarray([[1, 0, 0, 0.05], [0, -1, 0, 0.05]]), (1, 1, 1), 2, 0),
        lambda: TopupFixedMovementObjective(scans, rows, (1, 1, 1), 2, -1),
        lambda: TopupFixedMovementObjective(scans, rows, (1, 1, 1), 2, 0, target_shape=(0, 2, 2)),
        lambda: TopupFixedMovementObjective(scans, rows, (1, 1, 1), 2, 0, source_voxel_sizes_mm=(1, 0, 1)),
        lambda: TopupFixedMovementObjective(scans, rows, (1, 1, 1), 2, 0, fixed_movements=np.zeros((1, 6))),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()

    objective = TopupFixedMovementObjective(scans, rows, (1, 1, 1), 2, 0)
    with pytest.raises(ValueError, match="field-coefficient vector"):
        objective.cost(np.zeros(objective.number_of_parameters + 1))
    parameters = np.zeros(objective.number_of_parameters)
    hessian = objective.hessian(parameters)
    assert hessian.shape == (parameters.size, parameters.size)

    three_scans = np.stack((values, values, values), axis=3)
    three_rows = np.asarray(
        [[0, 1, 0, 0.05], [0, -1, 0, 0.05], [0, 1, 0, 0.05]], dtype=float
    )
    with pytest.raises(ValueError, match="exactly two"):
        topup.TopupMovingObjective(three_scans, three_rows, (1, 1, 1), 2, 0)
    moving = topup.TopupMovingObjective(scans, rows, (1, 1, 1), 2, 0)
    with pytest.raises(ValueError, match="joint parameter"):
        moving.cost(np.zeros(moving.number_of_parameters + 1))
    assert topup._topup_voxel_pull_matrix(np.zeros(6), values.shape, (1, 1, 1)).shape == (4, 4)


def test_topup_matrix_and_runner_validation_paths(
    tmp_path,
    monkeypatch,
) -> None:
    with pytest.raises(ValueError, match="finite 4x4"):
        topup._fsl_affine_inverse_4x4(np.eye(3))
    non_affine = np.eye(4)
    non_affine[3, 0] = 1.0
    with pytest.raises(ValueError, match="affine final row"):
        topup._fsl_affine_inverse_4x4(non_affine)
    singular = np.eye(4)
    singular[2, 2] = 0.0
    with pytest.raises(ValueError, match="singular"):
        topup._fsl_affine_inverse_4x4(singular)
    with pytest.raises(ValueError, match="six finite"):
        _topup_movement_matrix(np.zeros(5), (4, 4, 4), (1, 1, 1))
    with pytest.raises(ValueError, match="finite 4x4"):
        _topup_matrix_to_movement_parameters(np.eye(3), (4, 4, 4), (1, 1, 1))
    gimbal = np.eye(4)
    gimbal[:3, :3] = np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    assert np.all(np.isfinite(_topup_matrix_to_movement_parameters(gimbal, (4, 4, 4), (1, 1, 1))))
    with pytest.raises(ValueError, match="field must be"):
        topup.fit_spline_coefficients(np.ones((2, 2)), (1, 1, 1))
    rows = np.asarray([[0, 1, 0, 0.05], [0, -1, 0, 0.05]], dtype=float)
    with pytest.raises(ValueError, match="two finite"):
        topup.run_simnibs46_topup(np.ones((2, 2, 2, 3)), rows, (1, 1, 1))
    with pytest.raises(ValueError, match="zero mean"):
        topup.run_simnibs46_topup(np.zeros((2, 2, 2, 2)), rows, (1, 1, 1))

    sizes = iter(((1.0, 1.0, 1.0), (2.0, 1.0, 1.0)))
    monkeypatch.setattr(
        topup,
        "fsl_regrid_topup_scan",
        lambda scan, _voxel_sizes: (np.asarray(scan), next(sizes)),
    )
    with pytest.raises(RuntimeError, match="source grids"):
        topup.run_simnibs46_topup(np.ones((2, 2, 2, 2)), rows, (1, 1, 1))

    affine = np.eye(4)
    forward = tmp_path / "forward.nii.gz"
    reverse = tmp_path / "reverse.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), affine), forward)
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), affine), reverse)
    invalid_calls = (
        lambda: run_topup_nifti(forward, reverse, tmp_path / "a", readout_seconds=0, phase_encoding_direction="y"),
        lambda: run_topup_nifti(forward, reverse, tmp_path / "b", readout_seconds=0.05, phase_encoding_direction="y", workers=0),
        lambda: run_topup_nifti(forward, reverse, tmp_path / "c", readout_seconds=0.05, phase_encoding_direction="z"),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()
    bad_shape = tmp_path / "bad-shape.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3), dtype=np.float32), affine), bad_shape)
    with pytest.raises(ValueError, match="share one 3D shape"):
        run_topup_nifti(forward, bad_shape, tmp_path / "d", readout_seconds=0.05, phase_encoding_direction="y")
    different_shape = tmp_path / "different-shape.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((4, 3, 3), dtype=np.float32), affine),
        different_shape,
    )
    with pytest.raises(ValueError, match="share one 3D shape"):
        run_topup_nifti(
            forward,
            different_shape,
            tmp_path / "d2",
            readout_seconds=0.05,
            phase_encoding_direction="y",
        )
    bad_affine = tmp_path / "bad-affine.nii.gz"
    shifted = affine.copy()
    shifted[0, 3] = 1.0
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), shifted), bad_affine)
    with pytest.raises(ValueError, match="share one affine"):
        run_topup_nifti(forward, bad_affine, tmp_path / "e", readout_seconds=0.05, phase_encoding_direction="y")


def test_topup_runner_rejects_an_optimizer_that_does_not_publish_final_state(
    monkeypatch,
) -> None:
    """Keep final-state assertions so future backends cannot omit numerical artifacts."""

    class EmptyStateObjective:
        def __init__(self, *_args, **_kwargs):
            self._state = None

        @staticmethod
        def cost(_parameters):
            return 0.0

        @staticmethod
        def gradient(parameters):
            return np.zeros_like(parameters)

        @staticmethod
        def hessian(parameters):
            return np.eye(parameters.size)

    level = topup.TopupLevel(2.0, 1, 0.0, 0, 0.0, False, "scaled_conjugate_gradient")
    monkeypatch.setattr(topup, "SIMNIBS46_TOPUP_LEVELS", (level,))
    monkeypatch.setattr(topup, "TopupFixedMovementObjective", EmptyStateObjective)
    import dwi2cond_xp.preprocessing.topup_optimizer as optimizer

    monkeypatch.setattr(
        optimizer,
        "fsl_scaled_conjugate_gradient",
        lambda starting, *_args, **_kwargs: SimpleNamespace(
            parameters=np.asarray(starting), cost=0.0
        ),
    )
    scans = np.ones((3, 3, 3, 2), dtype=np.float32)
    rows = np.asarray([[0, 1, 0, 0.05], [0, -1, 0, 0.05]], dtype=float)
    with pytest.raises(RuntimeError, match="final TOPUP state"):
        topup.run_simnibs46_topup(scans, rows, (1.0, 1.0, 1.0))
