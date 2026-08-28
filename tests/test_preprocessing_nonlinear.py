import json
import os
from pathlib import Path
import subprocess
from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix, kron

import dwi2cond_xp.preprocessing.fnirt as fnirt_module
from dwi2cond_xp.preprocessing.nonlinear import (
    fsl_displacement_gradient,
    fsl_warp_jacobians,
    register_tensor_nonlinear_nifti,
    register_tensor_fnirt_nifti,
    resample_tensor_ppd_fsl,
)
from dwi2cond_xp.preprocessing.fnirt import (
    FnirtLevelImages,
    SIMNIBS46_FNIRT_LEVELS,
    apply_fnirt_intensity_mapping,
    evaluate_fnirt_cost,
    evaluate_fnirt_gradient,
    evaluate_fnirt_hessian,
    expand_fnirt_coefficients,
    fsl_fnirt_smooth,
    fsl_fnirt_full_resolution_knot_spacing,
    fsl_fnirt_bias_knot_spacing,
    fsl_fnirt_level_affine,
    fsl_fnirt_subsampled_shape,
    fsl_spm_like_mean,
    prepare_fnirt_images,
    prepare_fnirt_level_images,
    initialize_fnirt_intensity_mapping,
    optimize_fnirt_level,
    warp_fnirt_moving,
    zoom_fnirt_spline_coefficients,
)
from dwi2cond_xp.preprocessing.fnirt_topology import (
    fsl_constrain_topology,
    fsl_corner_jacobian_range,
    fsl_good_fft_size,
)
from dwi2cond_xp.preprocessing.topup import (
    expand_spline_coefficients,
    fsl_coefficient_shape,
    spline_design_matrix,
)


FSL_VECREG = Path(os.environ.get("FSL_VECREG", "/path/not/configured/vecreg"))
FSL_CONVERTWARP = Path(
    os.environ.get("FSL_CONVERTWARP", "/path/not/configured/convertwarp")
)
FSL_FNIRT = Path(os.environ.get("FSL_FNIRT", "/path/not/configured/fnirt"))


def _diagonal_tensor(shape=(9, 8, 7)):
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[..., 0] = 3.0
    tensor[..., 3] = 2.0
    tensor[..., 5] = 1.0
    return tensor


def test_fnirt_full_resolution_knot_spacing_uses_integer_division():
    assert fsl_fnirt_full_resolution_knot_spacing((2.0, 2.0, 2.0)) == (2, 2, 2)
    assert fsl_fnirt_full_resolution_knot_spacing(
        (1.1, 1.6, 2.4), warp_resolution_mm=10.0, final_subsampling=2
    ) == (4, 3, 2)
    with pytest.raises(ValueError, match="incompatible"):
        fsl_fnirt_full_resolution_knot_spacing((10.0, 10.0, 10.0))


def test_fnirt_topology_constraint_restores_a_folded_absolute_warp():
    shape = (8, 8, 8)
    identity = np.moveaxis(np.indices(shape, dtype=np.float32), 0, -1)
    assert fsl_good_fft_size(7) == 8
    assert fsl_good_fft_size(8193) == 8193
    assert np.array_equal(fsl_constrain_topology(identity, (1.0, 1.0, 1.0)), identity)

    folded = identity.copy()
    folded[:, 4:, :, 1] -= np.float32(3.0)
    constrained = fsl_constrain_topology(folded, (1.0, 1.0, 1.0))

    assert fsl_corner_jacobian_range(folded, (1.0, 1.0, 1.0))[0] == -2.0
    minimum, maximum = fsl_corner_jacobian_range(constrained, (1.0, 1.0, 1.0))
    assert minimum >= 0.01
    assert maximum <= 100.0

    coefficient_shape = fsl_coefficient_shape((9, 9, 9), (2, 2, 2))
    coefficients = np.zeros(coefficient_shape + (3,), dtype=np.float64)
    coefficients[3:, :, :, 0] = -3.0
    updated, analytic_range = fnirt_module._constrain_fnirt_warpfield(
        coefficients,
        (9, 9, 9),
        (1.0, 1.0, 1.0),
        np.eye(4),
        (2, 2, 2),
        max_tries=5,
    )
    assert updated.shape == coefficients.shape
    assert analytic_range[0] >= 0.01
    assert analytic_range[1] <= 100.0

    expanded = identity.copy()
    expanded[..., 0] *= np.float32(1.0e6)
    limited = fsl_constrain_topology(
        expanded,
        (1.0, 1.0, 1.0),
        minimum_jacobian=0.01,
        maximum_jacobian=2.0,
    )
    assert np.all(np.isfinite(limited))


def test_fnirt_warp_constraint_reports_progress_and_handles_affine_mapping(
    monkeypatch,
) -> None:
    shape = (9, 9, 9)
    spacing = (2, 2, 2)
    coefficients = np.zeros(fsl_coefficient_shape(shape, spacing) + (3,))
    affine = np.eye(4)
    affine[0, 3] = 0.25
    events: list[tuple[int, int, float]] = []

    monkeypatch.setattr(
        fnirt_module, "_fnirt_full_jacobian_range", lambda *args: (0.0, 1.0)
    )
    monkeypatch.setattr(
        fnirt_module, "fsl_constrain_topology", lambda values, *args, **kwargs: values
    )
    updated, bounds = fnirt_module._constrain_fnirt_warpfield(
        coefficients,
        shape,
        (1.0, 1.0, 1.0),
        affine,
        spacing,
        max_tries=3,
        progress=lambda *event: events.append(event),
    )

    assert updated.shape == coefficients.shape
    assert bounds == (0.0, 1.0)
    assert events == [(0, 3, 0.0), (1, 3, 0.0)]


@pytest.mark.parametrize(
    "call",
    [
        lambda: fsl_good_fft_size(0),
        lambda: fsl_constrain_topology(np.zeros((2, 2, 2)), (1.0, 1.0, 1.0)),
        lambda: fsl_constrain_topology(np.zeros((2, 2, 2, 3)), (1.0, 0.0, 1.0)),
        lambda: fsl_constrain_topology(
            np.zeros((2, 2, 2, 3)),
            (1.0, 1.0, 1.0),
            minimum_jacobian=2.0,
            maximum_jacobian=1.0,
        ),
        lambda: fsl_corner_jacobian_range(np.zeros((2, 2, 2)), (1.0, 1.0, 1.0)),
    ],
)
def test_fnirt_topology_contract_errors(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.skipif(not FSL_CONVERTWARP.is_file(), reason="FSL convertwarp unavailable")
def test_fnirt_topology_constraint_matches_fsl_warpfns(tmp_path: Path):
    shape = (8, 8, 8)
    affine = np.diag((-1.0, 1.0, 1.0, 1.0))
    identity = np.moveaxis(np.indices(shape, dtype=np.float32), 0, -1)
    folded = identity.copy()
    folded[:, 4:, :, 1] -= np.float32(3.0)
    reference_path = tmp_path / "reference.nii.gz"
    input_path = tmp_path / "input.nii.gz"
    output_path = tmp_path / "fsl.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), reference_path)
    nib.save(nib.Nifti1Image(folded, affine), input_path)
    environment = os.environ.copy()
    environment.setdefault("FSLDIR", str(FSL_CONVERTWARP.parents[1]))
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    subprocess.run(
        [
            str(FSL_CONVERTWARP),
            f"--ref={reference_path}",
            f"--warp1={input_path}",
            f"--out={output_path}",
            "--abs",
            "--absout",
            "--constrainj",
            "--jmin=0.01",
            "--jmax=100",
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    reference = np.asarray(nib.load(output_path).dataobj, dtype=np.float32)
    constrained = fsl_constrain_topology(folded, (1.0, 1.0, 1.0))

    np.testing.assert_allclose(constrained, reference, rtol=0.0, atol=2.0e-6)


def test_simnibs46_fnirt_schedule_and_pyramid_geometry_are_frozen():
    assert [level.subsampling for level in SIMNIBS46_FNIRT_LEVELS] == [8, 4, 2, 2]
    assert [level.regularization_weight for level in SIMNIBS46_FNIRT_LEVELS] == [
        240.0,
        120.0,
        60.0,
        60.0,
    ]
    assert [level.estimate_intensity for level in SIMNIBS46_FNIRT_LEVELS] == [
        True,
        True,
        True,
        False,
    ]
    assert fsl_fnirt_subsampled_shape((24, 23, 22), 8) == (4, 4, 4)
    assert fsl_fnirt_subsampled_shape((24, 23, 22), 4) == (7, 7, 7)
    assert fsl_fnirt_subsampled_shape((24, 23, 22), 2) == (13, 12, 12)
    assert fsl_fnirt_bias_knot_spacing((2.0, 2.0, 2.0), 8) == (3, 3, 3)
    assert fsl_fnirt_bias_knot_spacing((2.0, 2.0, 2.0), 4) == (6, 6, 6)
    assert fsl_fnirt_bias_knot_spacing((2.0, 2.0, 2.0), 2) == (12, 12, 12)


def test_fnirt_spm_mean_scaling_masks_and_level_sampling():
    reference = np.zeros((6, 5, 4), dtype=np.float32)
    moving = np.zeros_like(reference)
    reference[1:5, 1:4, 1:3] = np.arange(1, 25, dtype=np.float32).reshape(4, 3, 2)
    moving[1:5, 1:4, 1:3] = 2.0 * reference[1:5, 1:4, 1:3]
    assert fsl_spm_like_mean(reference) == 12.5
    prepared = prepare_fnirt_images(reference, moving)
    assert prepared.reference_mean == 12.5
    assert prepared.moving_mean == 25.0
    assert np.array_equal(prepared.reference_mask, prepared.moving_mask)
    assert np.array_equal(prepared.reference, prepared.moving)
    level = prepare_fnirt_level_images(
        prepared,
        (2.0, 2.0, 2.0),
        SIMNIBS46_FNIRT_LEVELS[0],
    )
    assert level.reference.shape == fsl_fnirt_subsampled_shape(reference.shape, 8)
    assert level.reference_voxel_sizes_mm == (16.0, 16.0, 16.0)
    assert level.moving.shape == moving.shape
    assert not level.reference_mask[-1, -1, -1]


def test_fnirt_level_smooths_moving_in_its_own_physical_grid():
    shape = (13, 12, 11)
    reference = np.zeros(shape, dtype=np.float32)
    moving = np.zeros(shape, dtype=np.float32)
    reference[2:-2, 2:-2, 2:-2] = 1.0
    moving[2:-2, 2:-2, 2:-2] = (
        np.arange(np.prod(np.asarray(shape) - 4), dtype=np.float32).reshape(
            tuple(np.asarray(shape) - 4)
        )
        + 1.0
    )
    prepared = prepare_fnirt_images(reference, moving)
    specification = SIMNIBS46_FNIRT_LEVELS[0]
    level = prepare_fnirt_level_images(
        prepared,
        (0.7, 0.7, 0.7),
        specification,
        moving_voxel_sizes_mm=(1.25, 1.25, 1.25),
    )
    sigma = specification.input_fwhm_mm / np.sqrt(8.0 * np.log(2.0))
    numerator = fsl_fnirt_smooth(
        prepared.moving * prepared.moving_mask,
        sigma,
        (1.25, 1.25, 1.25),
    )
    denominator = fsl_fnirt_smooth(
        prepared.moving_mask.astype(np.float32),
        sigma,
        (1.25, 1.25, 1.25),
    )
    expected = np.zeros_like(prepared.moving)
    np.divide(numerator, denominator, out=expected, where=denominator != 0.0)
    expected[~prepared.moving_mask] = 0.0
    assert np.array_equal(level.moving, expected)

    legacy = prepare_fnirt_level_images(
        prepared,
        (0.7, 0.7, 0.7),
        specification,
    )
    assert not np.array_equal(level.moving, legacy.moving)


def test_fnirt_affine_pull_and_masks_have_explicit_contracts():
    shape = (5, 4, 3)
    moving = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    moving[0] = 0.0
    prepared = prepare_fnirt_images(np.maximum(moving, 1.0), moving)
    level_specification = SIMNIBS46_FNIRT_LEVELS[-1]
    level = prepare_fnirt_level_images(prepared, (1.0, 1.0, 1.0), level_specification)
    level_affine = fsl_fnirt_level_affine(np.eye(4), 2)
    assert np.array_equal(level_affine, np.diag((2.0, 2.0, 2.0, 1.0)))
    result = warp_fnirt_moving(
        level,
        level_affine,
        np.eye(4),
        np.eye(4),
    )
    assert result.values.shape == level.reference.shape
    assert not np.any(result.coordinates[:, 0, 0, 0])
    assert not result.warped_moving_mask[0, 0, 0]
    assert not result.mask[0, 0, 0]
    assert not result.data_mask[-1, -1, -1]


@pytest.mark.parametrize("x_scale", [2.0, -2.0])
def test_fnirt_scaled_mm_derivatives_match_three_axis_finite_differences(x_scale):
    shape = (9, 8, 7)
    grid = np.indices(shape, dtype=np.float32)
    moving = 10.0 + 3.0 * grid[0] + 2.0 * grid[1] + grid[2]
    mask = np.ones(shape, dtype=bool)
    level = FnirtLevelImages(
        reference=moving.copy(),
        moving=moving,
        reference_mask=mask,
        moving_mask=mask,
        reference_voxel_sizes_mm=(2.0, 2.0, 2.0),
    )
    affine = np.diag((x_scale, 2.0, 2.0, 1.0))
    zero = np.zeros(shape + (3,), dtype=np.float32)
    analytic = warp_fnirt_moving(
        level,
        affine,
        affine,
        np.eye(4),
        zero,
        calculate_derivatives=True,
    ).derivatives_per_mm
    epsilon = np.float32(1.0e-3)
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))

    for component in range(3):
        positive = zero.copy()
        negative = zero.copy()
        positive[..., component] = epsilon
        negative[..., component] = -epsilon
        positive_values = warp_fnirt_moving(
            level,
            affine,
            affine,
            np.eye(4),
            positive,
            calculate_derivatives=False,
        ).values
        negative_values = warp_fnirt_moving(
            level,
            affine,
            affine,
            np.eye(4),
            negative,
            calculate_derivatives=False,
        ).values
        numeric = (positive_values - negative_values) / (2.0 * epsilon)
        np.testing.assert_allclose(
            analytic[interior + (component,)],
            numeric[interior],
            rtol=2.0e-3,
            atol=2.0e-3,
        )


def test_fnirt_coefficients_are_mapped_from_internal_radiological_grid():
    shape = (9, 8, 7)
    spacing = (2, 2, 2)
    rng = np.random.default_rng(61)
    coefficients = rng.normal(
        size=fsl_coefficient_shape(shape, spacing) + (3,)
    )
    radiological = expand_fnirt_coefficients(
        coefficients,
        shape,
        np.diag((-2.0, 2.0, 2.0, 1.0)),
        np.eye(4),
        knot_spacing=spacing,
    )
    neurological = expand_fnirt_coefficients(
        coefficients,
        shape,
        np.diag((2.0, 2.0, 2.0, 1.0)),
        np.eye(4),
        knot_spacing=spacing,
    )

    np.testing.assert_array_equal(
        neurological.nonlinear_displacement,
        np.flip(radiological.nonlinear_displacement, axis=0),
    )
    np.testing.assert_array_equal(
        neurological.nonlinear_jacobian,
        np.flip(radiological.nonlinear_jacobian, axis=0),
    )


def test_fnirt_canonicalizes_only_neurological_moving_grid(monkeypatch):
    original = fnirt_module.run_simnibs46_fnirt
    reference = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    moving = reference + np.float32(10.0)
    canonical = object()
    captured = {}

    def canonical_kernel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return canonical

    monkeypatch.setattr(fnirt_module, "run_simnibs46_fnirt", canonical_kernel)
    result = original(
        reference,
        moving,
        np.diag((-2.0, 2.0, 2.0, 1.0)),
        np.diag((2.0, 2.0, 2.0, 1.0)),
        np.eye(4),
        moving_mask=np.ones_like(moving, dtype=np.uint8),
    )

    assert result is canonical
    np.testing.assert_array_equal(captured["args"][0], reference)
    np.testing.assert_array_equal(captured["args"][1], np.flip(moving, axis=0))
    assert np.linalg.det(captured["args"][3][:3, :3]) < 0
    np.testing.assert_array_equal(
        captured["kwargs"]["moving_mask"],
        np.ones_like(moving, dtype=np.uint8),
    )


@pytest.mark.skipif(not FSL_FNIRT.is_file(), reason="FSL FNIRT is unavailable")
def test_positive_determinant_fnirt_endpoint_matches_real_fsl(tmp_path: Path):
    shape = (24, 23, 22)
    grid = np.indices(shape, dtype=np.float32)
    normalized = [
        (grid[axis] - np.float32((shape[axis] - 1) / 2.0))
        / np.float32(shape[axis])
        for axis in range(3)
    ]
    reference = (
        np.float32(80.0)
        + np.float32(35.0) * normalized[0]
        - np.float32(20.0) * normalized[1]
        + np.float32(15.0) * normalized[2]
        + np.float32(140.0)
        * np.exp(
            -(
                ((normalized[0] + np.float32(0.14)) / np.float32(0.12)) ** 2
                + ((normalized[1] - np.float32(0.08)) / np.float32(0.15)) ** 2
                + ((normalized[2] + np.float32(0.05)) / np.float32(0.11)) ** 2
            )
        )
        + np.float32(95.0)
        * np.exp(
            -(
                ((normalized[0] - np.float32(0.18)) / np.float32(0.10)) ** 2
                + ((normalized[1] + np.float32(0.15)) / np.float32(0.12)) ** 2
                + ((normalized[2] - np.float32(0.10)) / np.float32(0.14)) ** 2
            )
        )
    ).astype(np.float32)
    shift_x = (
        np.float32(1.15)
        * np.sin(
            np.float32(2.0 * np.pi)
            * (grid[1] / np.float32(shape[1] - 1))
        )
        * np.cos(
            np.float32(np.pi) * (grid[2] / np.float32(shape[2] - 1))
        )
    )
    moving = map_coordinates(
        reference,
        np.asarray([grid[0] + shift_x, grid[1], grid[2]], dtype=np.float32),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    affine = np.diag((2.0, 2.0, 2.0, 1.0))
    reference_file = tmp_path / "reference.nii.gz"
    moving_file = tmp_path / "moving.nii.gz"
    affine_file = tmp_path / "identity.mat"
    nib.save(nib.Nifti1Image(reference, affine), reference_file)
    nib.save(nib.Nifti1Image(moving, affine), moving_file)
    np.savetxt(affine_file, np.eye(4), fmt="%.17g")
    environment = os.environ.copy()
    environment["FSLDIR"] = str(FSL_FNIRT.parent.parent)
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    subprocess.run(
        [
            str(FSL_FNIRT),
            f"--in={moving_file}",
            f"--ref={reference_file}",
            f"--aff={affine_file}",
            f"--cout={tmp_path / 'fsl_warp'}",
            f"--fout={tmp_path / 'fsl_field'}",
            f"--jout={tmp_path / 'fsl_jacobian'}",
            f"--iout={tmp_path / 'fsl_iout'}",
            f"--logout={tmp_path / 'fnirt.log'}",
            "--subsamp=8,4,2,2",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    local = fnirt_module.run_simnibs46_fnirt(
        reference,
        moving,
        affine,
        affine,
        np.eye(4),
        workers=1,
    )
    level = FnirtLevelImages(
        reference=reference,
        moving=moving,
        reference_mask=reference != 0,
        moving_mask=moving != 0,
        reference_voxel_sizes_mm=(2.0, 2.0, 2.0),
    )
    local_result = warp_fnirt_moving(
        level,
        affine,
        affine,
        np.eye(4),
        local.expansion.displacement,
        calculate_derivatives=False,
    )
    fsl_field = np.asarray(
        nib.load(tmp_path / "fsl_field.nii.gz").dataobj, dtype=np.float32
    )
    fsl_result = warp_fnirt_moving(
        level,
        affine,
        affine,
        np.eye(4),
        fsl_field,
        calculate_derivatives=False,
    )
    common = local_result.mask & fsl_result.mask & (reference != 0)
    local_to_fsl = np.linalg.norm(
        (local_result.values[common] - fsl_result.values[common]).ravel()
    ) / np.linalg.norm(fsl_result.values[common].ravel())
    field_relative_l2 = np.linalg.norm(
        (local.expansion.displacement - fsl_field).ravel()
    ) / np.linalg.norm(fsl_field.ravel())

    assert np.count_nonzero(common) > 9000
    assert local_to_fsl < 0.005
    assert field_relative_l2 < 0.25
    assert np.max(np.abs(local.expansion.displacement - fsl_field)) < 1.2


def test_fnirt_intensity_mapping_initializes_identity_polynomial():
    shape = (8, 7, 6)
    level = SIMNIBS46_FNIRT_LEVELS[0]
    mapping = initialize_fnirt_intensity_mapping(shape, (2.0, 2.0, 2.0), level)
    reference = np.arange(np.prod(mapping.bias_field.shape), dtype=np.float32).reshape(
        mapping.bias_field.shape
    )
    scaled = apply_fnirt_intensity_mapping(reference, mapping)
    assert np.array_equal(mapping.global_coefficients, [0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(scaled, reference * mapping.bias_field)


def test_fnirt_cost_separates_ssd_and_regularization_terms():
    shape = (17, 17, 17)
    values = np.ones(shape, dtype=np.float32)
    prepared = prepare_fnirt_images(values, values)
    level_specification = SIMNIBS46_FNIRT_LEVELS[0]
    level = prepare_fnirt_level_images(prepared, (2.0, 2.0, 2.0), level_specification)
    mapping = initialize_fnirt_intensity_mapping(
        shape, (2.0, 2.0, 2.0), level_specification
    )
    full_affine = np.diag((-2.0, 2.0, 2.0, 1.0))
    result = evaluate_fnirt_cost(
        level,
        fsl_fnirt_level_affine(full_affine, 8),
        full_affine,
        np.eye(4),
        mapping,
        level_specification,
    )
    expected_difference = result.warped_moving.values - result.scaled_reference
    expected_ssd = np.mean(
        np.square(expected_difference[result.warped_moving.mask], dtype=np.float32),
        dtype=np.float64,
    )
    assert result.voxel_count == np.count_nonzero(result.warped_moving.mask)
    assert result.mean_squared_difference == expected_ssd
    assert result.displacement_regularization == 0.0
    assert result.intensity_regularization >= 0.0
    assert result.total == expected_ssd + result.intensity_regularization


def test_fnirt_gradient_uses_fsl_parameter_block_order():
    shape = (17, 17, 17)
    values = np.ones(shape, dtype=np.float32)
    prepared = prepare_fnirt_images(values, values)
    level_specification = SIMNIBS46_FNIRT_LEVELS[0]
    level = prepare_fnirt_level_images(prepared, (2.0, 2.0, 2.0), level_specification)
    mapping = initialize_fnirt_intensity_mapping(
        shape, (2.0, 2.0, 2.0), level_specification
    )
    full_affine = np.diag((-2.0, 2.0, 2.0, 1.0))
    result = evaluate_fnirt_gradient(
        level,
        fsl_fnirt_level_affine(full_affine, 8),
        full_affine,
        np.eye(4),
        mapping,
        level_specification,
    )
    displacement_size = 3 * np.prod(
        fsl_coefficient_shape(level.reference.shape, (2, 2, 2))
    )
    bias_size = mapping.bias_coefficients.size
    assert result.displacement_gradient.size == displacement_size
    assert result.bias_gradient.size == bias_size
    assert result.global_gradient.size == 5
    assert np.array_equal(
        result.gradient,
        np.concatenate(
            (
                result.displacement_gradient,
                result.bias_gradient,
                result.global_gradient,
            )
        ),
    )
    assert np.all(np.isfinite(result.gradient))

    hessian = evaluate_fnirt_hessian(
        level,
        fsl_fnirt_level_affine(full_affine, 8),
        full_affine,
        np.eye(4),
        mapping,
        level_specification,
    )
    parallel_hessian = evaluate_fnirt_hessian(
        level,
        fsl_fnirt_level_affine(full_affine, 8),
        full_affine,
        np.eye(4),
        mapping,
        level_specification,
        workers=8,
    )
    parameter_count = result.gradient.size
    assert hessian.hessian.shape == (parameter_count, parameter_count)
    difference_parallel = hessian.hessian - parallel_hessian.hessian
    reference_norm = np.sqrt(hessian.hessian.multiply(hessian.hessian).sum())
    difference_norm = np.sqrt(difference_parallel.multiply(difference_parallel).sum())
    assert difference_norm / reference_norm < 1.0e-12
    assert (
        np.max(np.abs(difference_parallel.data)) / np.max(np.abs(hessian.hessian.data))
        < 1.0e-12
    )
    displacement_block_size = int(
        np.prod(fsl_coefficient_shape(level.reference.shape, (2, 2, 2)))
    )
    for row in range(3):
        row_slice = slice(
            row * displacement_block_size, (row + 1) * displacement_block_size
        )
        for column in range(3):
            column_slice = slice(
                column * displacement_block_size,
                (column + 1) * displacement_block_size,
            )
            np.testing.assert_allclose(
                parallel_hessian.hessian[row_slice, column_slice].toarray(),
                hessian.hessian[row_slice, column_slice].toarray(),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
    difference = hessian.hessian - hessian.hessian.T
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-12


def test_fnirt_optimized_hessian_preserves_one_step_lm_trajectory():
    generator = np.random.default_rng(23)
    shape = (17, 19, 15)
    reference = generator.uniform(0.2, 1.0, size=shape).astype(np.float32)
    moving = np.roll(reference, 1, axis=0)
    prepared = prepare_fnirt_images(reference, moving)
    specification = replace(SIMNIBS46_FNIRT_LEVELS[-1], maximum_iterations=1)
    level = prepare_fnirt_level_images(prepared, (2.0, 2.0, 2.0), specification)
    mapping = initialize_fnirt_intensity_mapping(shape, (2.0, 2.0, 2.0), specification)
    affine = np.diag((-2.0, 2.0, 2.0, 1.0))
    reference_result = optimize_fnirt_level(
        level,
        fsl_fnirt_level_affine(affine, specification.subsampling),
        affine,
        np.eye(4),
        mapping,
        specification,
        workers=1,
    )
    optimized_result = optimize_fnirt_level(
        level,
        fsl_fnirt_level_affine(affine, specification.subsampling),
        affine,
        np.eye(4),
        mapping,
        specification,
        workers=8,
    )
    np.testing.assert_allclose(
        optimized_result.parameters,
        reference_result.parameters,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert optimized_result.status == reference_result.status
    assert (
        optimized_result.successful_iterations == reference_result.successful_iterations
    )
    assert len(optimized_result.trace) == len(reference_result.trace)
    np.testing.assert_allclose(
        [entry.attempted_cost for entry in optimized_result.trace],
        [entry.attempted_cost for entry in reference_result.trace],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_fnirt_direct_sparse_basis_is_bitwise_kronecker_equivalent():
    shape = (13, 12, 11)
    spacing = (3, 4, 2)
    axes = tuple(
        csr_matrix(spline_design_matrix(size, knot))
        for size, knot in zip(shape, spacing, strict=True)
    )
    expected = kron(axes[2], kron(axes[1], axes[0], format="csr"), format="csr")
    for workers in (1, 8):
        actual = fnirt_module._spline_basis_sparse(shape, spacing, workers)
        assert np.array_equal(actual.indptr, expected.indptr)
        assert np.array_equal(actual.indices, expected.indices)
        assert np.array_equal(actual.data, expected.data)
    with pytest.raises(ValueError, match="workers"):
        fnirt_module._spline_basis_sparse(shape, spacing, 0)

    selected = np.zeros(int(np.prod(shape)), dtype=bool)
    selected[::3] = True
    expected_selected = expected[selected]
    actual_selected = fnirt_module._selected_spline_basis_sparse(
        shape, spacing, selected, 8
    )
    assert np.array_equal(actual_selected.indptr, expected_selected.indptr)
    assert np.array_equal(actual_selected.indices, expected_selected.indices)
    assert np.array_equal(actual_selected.data, expected_selected.data)
    with pytest.raises(ValueError, match="selected mask"):
        fnirt_module._selected_spline_basis_sparse(shape, spacing, selected[:-1], 8)
    with pytest.raises(ValueError, match="workers"):
        fnirt_module._selected_spline_basis_sparse(shape, spacing, selected, 0)


def test_fnirt_sparse_jtj_helpers_cover_masked_and_empty_contracts():
    left = csr_matrix(np.asarray([[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]]))
    right = csr_matrix(np.asarray([[2.0, 0.0], [1.0, 1.0], [0.0, 5.0]]))
    weights = np.asarray([0.5, 2.0, 1.5])
    mask = np.asarray([True, False, True])

    actual = fnirt_module._masked_sparse_jtj(left, right, weights, mask)
    selected = np.flatnonzero(mask)
    expected = left[selected].T @ right[selected].multiply(weights[selected, None])
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    assert fnirt_module._selected_sparse_jtj_blocks(left, right, (), workers=8) == ()
    with pytest.raises(ValueError, match="weights"):
        fnirt_module._selected_sparse_jtj(left, right, np.ones(2))


def test_fnirt_coefficients_expand_affine_and_analytic_jacobian():
    shape = (9, 8, 7)
    spacing = (2, 2, 2)
    coefficients = np.zeros(fsl_coefficient_shape(shape, spacing) + (3,))
    affine = np.array(
        [
            [1.0, 0.02, 0.0, 1.5],
            [0.0, 1.0, -0.01, -0.5],
            [0.0, 0.0, 1.0, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = expand_fnirt_coefficients(
        coefficients,
        shape,
        np.diag((-2.0, 2.0, 2.0, 1.0)),
        affine,
        knot_spacing=spacing,
    )
    sampling = np.diag((2.0, 2.0, 2.0, 1.0))
    expected_mapping = (np.linalg.inv(affine) - np.eye(4)) @ sampling
    grid = np.indices(shape, dtype=np.float64).reshape(3, -1)
    expected = (
        expected_mapping[:3, :3] @ grid + expected_mapping[:3, 3, None]
    ).T.reshape(shape + (3,))
    assert not np.any(result.nonlinear_displacement)
    assert np.allclose(result.affine_displacement, expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(result.displacement, result.affine_displacement)
    assert np.all(result.nonlinear_jacobian == np.eye(3))
    assert np.all(result.nonlinear_jacobian_determinant == 1.0)


def test_fused_fnirt_expansion_is_exact_to_independent_scalar_expansion():
    shape = (9, 8, 7)
    spacing = (2, 2, 2)
    coefficients = np.random.default_rng(7).normal(
        size=fsl_coefficient_shape(shape, spacing) + (3,)
    )
    result = expand_fnirt_coefficients(
        coefficients,
        shape,
        np.diag((-2.0, 2.0, 2.0, 1.0)),
        np.eye(4),
        knot_spacing=spacing,
    )
    expected_field = np.empty_like(result.nonlinear_displacement)
    expected_jacobian = np.empty_like(result.nonlinear_jacobian)
    for component in range(3):
        expected_field[..., component] = expand_spline_coefficients(
            coefficients[..., component], shape, spacing
        )
        for axis in range(3):
            expected_jacobian[..., component, axis] = (
                expand_spline_coefficients(
                    coefficients[..., component],
                    shape,
                    spacing,
                    derivative_axis=axis,
                )
                / 2.0
            )
    for axis in range(3):
        expected_jacobian[..., axis, axis] += 1.0
    assert np.array_equal(result.nonlinear_displacement, expected_field)
    assert np.array_equal(result.nonlinear_jacobian, expected_jacobian)


def test_fsl_displacement_gradient_and_jacobian_truth():
    shape = (9, 8, 7)
    grid = np.indices(shape, dtype=np.float32)
    displacement = np.zeros(shape + (3,), dtype=np.float32)
    displacement[..., 0] = 0.03 * grid[0] + 0.04 * grid[2]
    displacement[..., 1] = 0.1 * grid[0]
    gradient = fsl_displacement_gradient(displacement)
    expected = np.array(
        [[0.03, 0.0, 0.04], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    assert np.allclose(gradient, expected, rtol=0, atol=4e-8)
    backward, forward, determinant, fold, near_singular = fsl_warp_jacobians(
        displacement
    )
    assert np.allclose(backward, np.eye(3) + expected, rtol=0, atol=4e-8)
    assert np.allclose(forward, np.linalg.inv(np.eye(3) + expected), atol=4e-8)
    assert np.allclose(determinant, 1.03, rtol=0, atol=4e-8)
    assert not np.any(fold)
    assert not np.any(near_singular)


def test_ppd_shear_matches_known_rotation_and_preserves_eigenvalues():
    shape = (9, 8, 7)
    tensor = _diagonal_tensor(shape)
    displacement = np.zeros(shape + (3,), dtype=np.float32)
    displacement[..., 1] = 0.1 * np.indices(shape, dtype=np.float32)[0]
    result = resample_tensor_ppd_fsl(
        tensor,
        np.eye(4),
        shape,
        np.eye(4),
        displacement,
    )
    interior = result.tensor[1:-1, 1:-1, 1:-1]
    expected = np.array(
        [2.9900990099, 0.0990099010, 0.0, 2.0099009901, 0.0, 1.0],
        dtype=np.float32,
    )
    assert np.allclose(interior, expected, rtol=0, atol=3e-7)
    eigenvalues = np.linalg.eigvalsh(
        np.array(
            [
                [interior[0, 0, 0, 0], interior[0, 0, 0, 1], 0.0],
                [interior[0, 0, 0, 1], interior[0, 0, 0, 3], 0.0],
                [0.0, 0.0, interior[0, 0, 0, 5]],
            ]
        )
    )
    assert np.allclose(eigenvalues, [1.0, 2.0, 3.0], atol=3e-7)
    assert not np.any(result.fold_mask)
    assert not np.any(result.near_singular_mask)


def test_fold_singularity_and_masks_are_explicit():
    shape = (4, 4, 4)
    tensor = _diagonal_tensor(shape)
    grid = np.indices(shape, dtype=np.float32)
    displacement = np.zeros(shape + (3,), dtype=np.float32)
    # A neurological NIfTI is x-flipped in FSL storage, so +x yields -1 there.
    displacement[..., 0] = grid[0]
    reference_mask = np.ones(shape, dtype=np.uint8)
    reference_mask[0] = 0
    result = resample_tensor_ppd_fsl(
        tensor,
        np.eye(4),
        shape,
        np.eye(4),
        displacement,
        reference_mask=reference_mask,
        compatibility_mode="robust",
    )
    assert np.all(result.near_singular_mask)
    assert np.all(result.fold_mask)
    assert not np.any(result.valid_mask)
    assert not np.any(result.tensor)

    with pytest.raises(ValueError, match="strict-fsl nonlinear PPD"):
        resample_tensor_ppd_fsl(
            tensor,
            np.eye(4),
            shape,
            np.eye(4),
            displacement,
            reference_mask=reference_mask,
        )


def test_repeated_and_low_fa_are_reported_without_renaming_output():
    shape = (3, 3, 3)
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[..., 0] = tensor[..., 3] = tensor[..., 5] = 1.0
    result = resample_tensor_ppd_fsl(
        tensor,
        np.eye(4),
        shape,
        np.eye(4),
        np.zeros(shape + (3,), dtype=np.float32),
    )
    assert np.all(result.repeated_eigenvalue_mask)
    assert np.all(result.low_fa_mask)
    assert np.array_equal(result.tensor, tensor)


def test_nonlinear_contract_rejects_invalid_inputs():
    tensor = _diagonal_tensor((3, 3, 3))
    warp = np.zeros((3, 3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        fsl_displacement_gradient(warp[..., 0])
    with pytest.raises(ValueError, match="at least two"):
        fsl_displacement_gradient(np.zeros((1, 2, 2, 3)))
    with pytest.raises(ValueError, match="singular_tolerance"):
        fsl_warp_jacobians(warp, singular_tolerance=0)
    with pytest.raises(ValueError, match="three signs"):
        fsl_warp_jacobians(warp, voxel_axis_directions=np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="positive finite"):
        fsl_warp_jacobians(warp, voxel_sizes=np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="tensor must"):
        resample_tensor_ppd_fsl(tensor[..., :5], np.eye(4), (3, 3, 3), np.eye(4), warp)
    with pytest.raises(ValueError, match="displacement"):
        resample_tensor_ppd_fsl(tensor, np.eye(4), (3, 3, 3), np.eye(4), warp[:2])
    with pytest.raises(ValueError, match="source_mask"):
        resample_tensor_ppd_fsl(
            tensor,
            np.eye(4),
            (3, 3, 3),
            np.eye(4),
            warp,
            source_mask=np.ones((2, 2, 2)),
        )
    with pytest.raises(ValueError, match="reference_mask"):
        resample_tensor_ppd_fsl(
            tensor,
            np.eye(4),
            (3, 3, 3),
            np.eye(4),
            warp,
            reference_mask=np.ones((2, 2, 2)),
        )
    with pytest.raises(ValueError, match="compatibility_mode"):
        resample_tensor_ppd_fsl(
            tensor,
            np.eye(4),
            (3, 3, 3),
            np.eye(4),
            warp,
            compatibility_mode="bad",
        )


def test_public_ppd_masks_use_fsl_nonzero_support_for_negative_values():
    shape = (3, 3, 3)
    tensor = _diagonal_tensor(shape)
    warp = np.zeros(shape + (3,), dtype=np.float32)
    positive = resample_tensor_ppd_fsl(
        tensor,
        np.eye(4),
        shape,
        np.eye(4),
        warp,
        source_mask=np.ones(shape),
        reference_mask=np.ones(shape),
    )
    negative = resample_tensor_ppd_fsl(
        tensor,
        np.eye(4),
        shape,
        np.eye(4),
        warp,
        source_mask=-np.ones(shape),
        reference_mask=-np.ones(shape),
    )
    np.testing.assert_array_equal(negative.valid_mask, positive.valid_mask)
    np.testing.assert_array_equal(negative.tensor, positive.tensor)


def test_fnirt_validation_contracts_cover_all_public_boundaries():
    values = np.ones((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="finite 3D"):
        fsl_fnirt_smooth(values[..., None], 1.0, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="positive and finite"):
        fsl_fnirt_smooth(values, 0.0, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        fsl_fnirt_bias_knot_spacing((1.0, 0.0, 1.0), 1)
    with pytest.raises(ValueError, match="coarsest"):
        fsl_fnirt_bias_knot_spacing((100.0, 100.0, 100.0), 8, bias_resolution_mm=1.0)

    spacing = (2, 2, 2)
    coefficient_shape = fsl_coefficient_shape(values.shape, spacing)
    coefficients = np.zeros(coefficient_shape)
    with pytest.raises(ValueError, match="coefficient shape"):
        zoom_fnirt_spline_coefficients(
            coefficients[:-1],
            values.shape,
            (1.0, 1.0, 1.0),
            spacing,
            values.shape,
            (1.0, 1.0, 1.0),
            spacing,
        )
    with pytest.raises(ValueError, match="voxel sizes"):
        zoom_fnirt_spline_coefficients(
            coefficients,
            values.shape,
            (1.0, 1.0, 1.0),
            spacing,
            values.shape,
            (0.0, 1.0, 1.0),
            spacing,
        )
    with pytest.raises(ValueError, match="integer knots"):
        zoom_fnirt_spline_coefficients(
            coefficients,
            values.shape,
            (1.0, 1.0, 1.0),
            spacing,
            values.shape,
            (1.3, 1.0, 1.0),
            spacing,
        )

    mapping = initialize_fnirt_intensity_mapping(
        values.shape, (1.0, 1.0, 1.0), SIMNIBS46_FNIRT_LEVELS[0]
    )
    with pytest.raises(ValueError, match="match the bias"):
        apply_fnirt_intensity_mapping(values[:-1], mapping)
    with pytest.raises(ValueError, match="three-dimensional"):
        fsl_spm_like_mean(values[..., None])
    with pytest.raises(ValueError, match="empty foreground"):
        fsl_spm_like_mean(np.zeros_like(values))
    with pytest.raises(ValueError, match="three-dimensional"):
        prepare_fnirt_images(values[..., None], values)
    invalid = values.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare_fnirt_images(invalid, values)
    with pytest.raises(ValueError, match="binary"):
        prepare_fnirt_images(values, values, reference_mask=np.full(values.shape, 2))
    with pytest.raises(ValueError, match="positive dimensions"):
        fsl_fnirt_subsampled_shape((0, 2, 2), 2)
    with pytest.raises(ValueError, match="positive power"):
        fsl_fnirt_subsampled_shape((2, 2, 2), 0)
    with pytest.raises(ValueError, match="positive power"):
        fsl_fnirt_subsampled_shape((2, 2, 2), 3)
    with pytest.raises(ValueError, match="three finite"):
        prepare_fnirt_level_images(
            prepare_fnirt_images(values, values), (1.0, 1.0), SIMNIBS46_FNIRT_LEVELS[0]
        )
    with pytest.raises(ValueError, match="positive"):
        prepare_fnirt_level_images(
            prepare_fnirt_images(values, values),
            (1.0, 0.0, 1.0),
            SIMNIBS46_FNIRT_LEVELS[0],
        )
    with pytest.raises(ValueError, match="moving_voxel_sizes_mm"):
        prepare_fnirt_level_images(
            prepare_fnirt_images(values, values),
            (1.0, 1.0, 1.0),
            SIMNIBS46_FNIRT_LEVELS[0],
            moving_voxel_sizes_mm=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="moving_voxel_sizes_mm"):
        prepare_fnirt_level_images(
            prepare_fnirt_images(values, values),
            (1.0, 1.0, 1.0),
            SIMNIBS46_FNIRT_LEVELS[0],
            moving_voxel_sizes_mm=(1.0, -1.0, 1.0),
        )


def test_fnirt_cost_warp_and_expansion_reject_invalid_contracts():
    shape = (5, 5, 5)
    values = np.ones(shape, dtype=np.float32)
    prepared = prepare_fnirt_images(values, values)
    level_specification = SIMNIBS46_FNIRT_LEVELS[-1]
    level = prepare_fnirt_level_images(prepared, (1.0, 1.0, 1.0), level_specification)
    mapping = initialize_fnirt_intensity_mapping(
        shape, (1.0, 1.0, 1.0), level_specification
    )
    affine = np.eye(4)
    with pytest.raises(ValueError, match="displacement_coefficients"):
        evaluate_fnirt_cost(
            level,
            affine,
            affine,
            affine,
            mapping,
            level_specification,
            displacement_coefficients=np.zeros((1, 1, 1, 3)),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_fnirt_cost(
            level,
            affine,
            affine,
            affine,
            mapping,
            level_specification,
            bias_regularization_weight=-1.0,
        )
    empty_level = FnirtLevelImages(
        level.reference,
        level.moving,
        np.zeros_like(level.reference_mask),
        level.moving_mask,
        level.reference_voxel_sizes_mm,
    )
    with pytest.raises(ValueError, match="no valid voxels"):
        evaluate_fnirt_cost(
            empty_level, affine, affine, affine, mapping, level_specification
        )
    with pytest.raises(ValueError, match="match the level"):
        evaluate_fnirt_gradient(
            level,
            affine,
            affine,
            affine,
            mapping,
            level_specification,
            displacement_coefficients=np.zeros((1, 1, 1, 3)),
        )
    with pytest.raises(ValueError, match="nonlinear_displacement"):
        warp_fnirt_moving(level, affine, affine, affine, np.zeros((1, 1, 1, 3)))
    bad_reference_mask = FnirtLevelImages(
        level.reference,
        level.moving,
        np.ones((1, 1, 1), dtype=bool),
        level.moving_mask,
        level.reference_voxel_sizes_mm,
    )
    with pytest.raises(ValueError, match="reference_mask"):
        warp_fnirt_moving(bad_reference_mask, affine, affine, affine)
    bad_moving_mask = FnirtLevelImages(
        level.reference,
        level.moving,
        level.reference_mask,
        np.ones((1, 1, 1), dtype=bool),
        level.reference_voxel_sizes_mm,
    )
    with pytest.raises(ValueError, match="moving_mask"):
        warp_fnirt_moving(bad_moving_mask, affine, affine, affine)

    with pytest.raises(ValueError, match="positive integer"):
        fsl_fnirt_full_resolution_knot_spacing((1.0, 1.0, 1.0), final_subsampling=0)
    with pytest.raises(ValueError, match="finite 4x4"):
        expand_fnirt_coefficients(np.zeros((1,)), shape, np.eye(3), affine)
    bad_homogeneous = affine.copy()
    bad_homogeneous[3, 0] = 1.0
    with pytest.raises(ValueError, match="homogeneous"):
        expand_fnirt_coefficients(np.zeros((1,)), shape, bad_homogeneous, affine)
    singular = affine.copy()
    singular[0, 0] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        expand_fnirt_coefficients(np.zeros((1,)), shape, singular, affine)
    with pytest.raises(ValueError, match="dimensions"):
        expand_fnirt_coefficients(np.zeros((1,)), (1, 2, 2), affine, affine)
    with pytest.raises(ValueError, match="positive integers"):
        expand_fnirt_coefficients(
            np.zeros((1,)), shape, affine, affine, knot_spacing=(0, 1, 1)
        )
    with pytest.raises(ValueError, match="coefficients"):
        expand_fnirt_coefficients(
            np.zeros((1,)), shape, affine, affine, knot_spacing=(2, 2, 2)
        )

    objective = fnirt_module._FnirtLevelObjective(
        level, affine, affine, affine, mapping, level_specification, (2, 2, 2), 1.0
    )
    with pytest.raises(ValueError, match="parameters"):
        objective.unpack(np.zeros(1))
    with pytest.raises(ValueError, match="initial displacement"):
        optimize_fnirt_level(
            level,
            affine,
            affine,
            affine,
            mapping,
            level_specification,
            initial_displacement_coefficients=np.zeros((1, 1, 1, 3)),
        )
    with pytest.raises(ValueError, match="workers"):
        optimize_fnirt_level(
            level,
            affine,
            affine,
            affine,
            mapping,
            level_specification,
            workers=0,
        )
    gradient = evaluate_fnirt_gradient(
        level, affine, affine, affine, mapping, level_specification
    )
    with pytest.raises(ValueError, match="workers"):
        evaluate_fnirt_hessian(
            level,
            affine,
            affine,
            affine,
            mapping,
            level_specification,
            workers=0,
            _gradient_evaluation=gradient,
        )
    with pytest.raises(ValueError, match="workers"):
        fnirt_module.run_simnibs46_fnirt(
            values, values, affine, affine, affine, workers=0
        )


def test_fnirt_tiny_sigma_and_explicit_binary_mask_paths():
    values = np.ones((3, 3, 3), dtype=np.float32)
    assert np.array_equal(fsl_fnirt_smooth(values, 1e-7, (1.0, 1.0, 1.0)), values)
    prepared = prepare_fnirt_images(
        values, values, reference_mask=np.ones_like(values, dtype=np.uint8)
    )
    assert np.all(prepared.reference_mask)


def test_fnirt_self_registration_uses_identity_fast_path():
    values = np.arange(7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5) + 1.0
    almost_identity = np.eye(4)
    almost_identity[0, 3] = 5.0e-9
    result = fnirt_module.run_simnibs46_fnirt(
        values,
        values.copy(),
        np.eye(4),
        np.eye(4),
        almost_identity,
        reference_mask=np.ones_like(values, dtype=np.uint8),
        moving_mask=np.zeros_like(values, dtype=np.uint8),
    )

    assert result.levels == ()
    assert result.jacobian_ranges == ()
    assert not np.any(result.coefficients)
    assert not np.any(result.expansion.displacement)
    np.testing.assert_array_equal(
        result.expansion.nonlinear_jacobian_determinant, 1.0
    )
    np.testing.assert_array_equal(
        result.intensity_mapping.global_coefficients,
        [0.0, 1.0, 0.0, 0.0, 0.0],
    )


def test_fnirt_nifti_self_registration_publishes_identity_outputs(tmp_path):
    shape = (7, 6, 5)
    affine = np.diag((-2.0, 2.0, 2.0, 1.0))
    fa = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    tensor = _diagonal_tensor(shape).astype(np.float32)
    mask = np.ones(shape, dtype=np.uint8)
    fa_file = tmp_path / "fa.nii.gz"
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    matrix_file = tmp_path / "identity.mat"
    output = tmp_path / "identity-output"
    nib.save(nib.Nifti1Image(fa, affine), fa_file)
    nib.save(nib.Nifti1Image(tensor, affine), tensor_file)
    nib.save(nib.Nifti1Image(fa.copy(), affine), reference_file)
    nib.save(nib.Nifti1Image(mask, affine), mask_file)
    np.savetxt(matrix_file, np.eye(4))

    qa = register_tensor_fnirt_nifti(
        fa_file,
        tensor_file,
        reference_file,
        matrix_file,
        output,
        brain_mask_file=mask_file,
        workers=2,
    )

    assert qa["identity_fast_path"] is True
    assert qa["levels"] == []
    assert not np.any(np.asarray(nib.load(output / "FA2T1_warp.nii.gz").dataobj))
    np.testing.assert_array_equal(
        np.asarray(nib.load(output / "FA2T1_field.nii.gz").dataobj), 0.0
    )
    np.testing.assert_array_equal(
        np.asarray(nib.load(output / "FA2T1_jacobian.nii.gz").dataobj), 1.0
    )
    np.testing.assert_array_equal(
        np.asarray(nib.load(output / "DTI_FA_nonlin.nii.gz").dataobj), fa
    )
    np.testing.assert_allclose(
        np.asarray(nib.load(output / "DTI_coregT1_tensor.nii.gz").dataobj),
        tensor,
        rtol=0.0,
        atol=1.0e-7,
    )


def test_fnirt_topology_constraint_is_applied_after_each_level(monkeypatch):
    values = np.ones((17, 17, 17), dtype=np.float32)
    constrained: list[int] = []

    def fake_optimize(*args, initial_displacement_coefficients, **kwargs):
        parameter_count = 3 * int(np.prod(initial_displacement_coefficients.shape[:3]))
        return type(
            "Optimization",
            (),
            {
                "displacement_coefficients": initial_displacement_coefficients,
                "intensity_mapping": args[4],
                "parameters": np.zeros(parameter_count),
                "successful_iterations": 0,
                "trace": (),
                "cost": 0.0,
                "status": "test",
            },
        )()

    def fake_constrain(coefficients, *args, max_tries, **kwargs):
        constrained.append(max_tries)
        return coefficients.copy(), (0.5, 1.5)

    monkeypatch.setattr(fnirt_module, "optimize_fnirt_level", fake_optimize)
    monkeypatch.setattr(
        fnirt_module, "_fnirt_full_jacobian_range", lambda *args: (0.0, 1.0)
    )
    monkeypatch.setattr(fnirt_module, "_constrain_fnirt_warpfield", fake_constrain)
    result = fnirt_module.run_simnibs46_fnirt(
        values, values * np.float32(2.0), np.eye(4), np.eye(4), np.eye(4)
    )

    assert constrained == [5, 10, 10, 10]
    assert result.jacobian_ranges == ((0.5, 1.5),) * 4


def test_fnirt_complete_schedule_can_be_audited_without_optimizer_cost(monkeypatch):
    values = np.ones((17, 17, 17), dtype=np.float32)
    progress: list[tuple[int, str]] = []

    def preserve_initial_state(*args, initial_displacement_coefficients, **kwargs):
        return type(
            "Optimization",
            (),
            {
                "displacement_coefficients": initial_displacement_coefficients,
                "intensity_mapping": args[4],
                "successful_iterations": 0,
                "trace": (),
                "cost": 0.0,
                "status": "test",
            },
        )()

    monkeypatch.setattr(fnirt_module, "optimize_fnirt_level", preserve_initial_state)
    result = fnirt_module.run_simnibs46_fnirt(
        values,
        values * np.float32(2.0),
        np.eye(4),
        np.eye(4),
        np.eye(4),
        progress=lambda level, phase, done, total, value: progress.append(
            (level, phase, done, total, value)
        ),
    )
    assert len(result.levels) == 4
    assert len(result.jacobian_ranges) == 4
    assert all(bounds == (1.0, 1.0) for bounds in result.jacobian_ranges)
    assert progress[-1][:4] == (4, "complete", 1, 1)


def test_fnirt_schedule_derives_displacement_spacing_from_reference_voxels(
    monkeypatch,
):
    values = np.ones((17, 17, 17), dtype=np.float32)
    spacings: list[tuple[int, int, int]] = []

    def preserve_initial_state(
        *args, initial_displacement_coefficients, displacement_knot_spacing, **kwargs
    ):
        spacings.append(displacement_knot_spacing)
        return type(
            "Optimization",
            (),
            {
                "displacement_coefficients": initial_displacement_coefficients,
                "intensity_mapping": args[4],
                "successful_iterations": 0,
                "trace": (),
                "cost": 0.0,
                "status": "test",
            },
        )()

    monkeypatch.setattr(fnirt_module, "optimize_fnirt_level", preserve_initial_state)
    affine = np.diag((0.7, 0.7, 0.7, 1.0))
    result = fnirt_module.run_simnibs46_fnirt(
        values,
        values * np.float32(2.0),
        affine,
        affine,
        np.eye(4),
    )

    assert spacings == [(7, 7, 7)] * 4
    assert result.expansion.knot_spacing == (7, 7, 7)


def test_fnirt_parallel_kernels_match_serial_contract():
    shape = (47, 47, 47)
    values = np.zeros(shape, dtype=np.float32)
    values[23, 23, 23] = 1.0
    smoothed = fsl_fnirt_smooth(values, 1.0, (1.0, 1.0, 1.0))
    assert smoothed.shape == shape
    spacing = (8, 8, 8)
    coefficients = np.zeros(fsl_coefficient_shape(shape, spacing) + (3,))
    expanded = expand_fnirt_coefficients(
        coefficients, shape, np.eye(4), np.eye(4), knot_spacing=spacing
    )
    assert not np.any(expanded.displacement)


def test_nonlinear_nifti_writes_tensor_derivatives_and_qa(tmp_path):
    shape = (5, 4, 3)
    tensor = _diagonal_tensor(shape)
    warp = np.zeros(shape + (3,), dtype=np.float32)
    warp[..., 1] = 0.1 * np.indices(shape, dtype=np.float32)[0]
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    warp_file = tmp_path / "warp.nii.gz"
    source_mask_file = tmp_path / "source_mask.nii.gz"
    reference_mask_file = tmp_path / "reference_mask.nii.gz"
    output_file = tmp_path / "registered.nii.gz"
    nib.save(nib.Nifti1Image(tensor, np.eye(4)), tensor_file)
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), reference_file
    )
    warp_image = nib.Nifti1Image(warp, np.eye(4))
    warp_image.header.set_intent("vector")
    nib.save(warp_image, warp_file)
    nib.save(
        nib.Nifti1Image(-np.ones(shape, dtype=np.float32), np.eye(4)), source_mask_file
    )
    nib.save(
        nib.Nifti1Image(-np.ones(shape, dtype=np.float32), np.eye(4)), reference_mask_file
    )
    qa = register_tensor_nonlinear_nifti(
        tensor_file,
        reference_file,
        warp_file,
        output_file,
        source_mask_file=source_mask_file,
        reference_mask_file=reference_mask_file,
        workers=2,
    )
    assert qa["fallback"] == "none"
    assert qa["fold_voxels"] == 0
    assert qa["valid_voxels"] > 0
    for suffix in ("", "_valid_mask", "_jacobian", "_FA", "_V1"):
        assert (tmp_path / f"registered{suffix}.nii.gz").is_file()
    stored = json.loads(
        (tmp_path / "registered_nonlinear_qa.json").read_text(encoding="utf-8")
    )
    assert stored["reorientation"].endswith("principal-direction")


def test_nonlinear_nifti_accepts_explicit_fnirt_coefficients(tmp_path):
    shape = (5, 4, 3)
    spacing = (2, 2, 2)
    tensor = _diagonal_tensor(shape)
    coefficients = np.zeros(fsl_coefficient_shape(shape, spacing) + (3,))
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    coefficient_file = tmp_path / "coefficients.nii.gz"
    affine_file = tmp_path / "affine.mat"
    output_file = tmp_path / "registered.nii.gz"
    nib.save(nib.Nifti1Image(tensor, np.eye(4)), tensor_file)
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), reference_file
    )
    nib.save(nib.Nifti1Image(coefficients, np.eye(4)), coefficient_file)
    np.savetxt(affine_file, np.eye(4))
    qa = register_tensor_nonlinear_nifti(
        tensor_file,
        reference_file,
        coefficient_file,
        output_file,
        warp_kind="coefficients",
        affine_matrix_file=affine_file,
        knot_spacing=spacing,
        workers=2,
    )
    assert qa["warp_kind"] == "coefficients"


def test_nonlinear_output_mask_matches_post_vecreg_source_order(tmp_path):
    shape = (5, 4, 3)
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    warp_file = tmp_path / "warp.nii.gz"
    output_mask_file = tmp_path / "T1_brainmask.nii.gz"
    output_file = tmp_path / "registered.nii.gz"
    mask = np.ones(shape, dtype=np.float32)
    mask[0] = 0
    mask[1] = -1
    nib.save(nib.Nifti1Image(_diagonal_tensor(shape), np.eye(4)), tensor_file)
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), reference_file)
    nib.save(nib.Nifti1Image(np.zeros(shape + (3,)), np.eye(4)), warp_file)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), output_mask_file)

    qa = register_tensor_nonlinear_nifti(
        tensor_file,
        reference_file,
        warp_file,
        output_file,
        output_mask_file=output_mask_file,
        workers=2,
    )

    tensor = np.asarray(nib.load(output_file).dataobj)
    valid = np.asarray(nib.load(tmp_path / "registered_valid_mask.nii.gz").dataobj)
    fa = np.asarray(nib.load(tmp_path / "registered_FA.nii.gz").dataobj)
    assert not np.any(tensor[0])
    assert not np.any(valid[0])
    assert not np.any(fa[0])
    assert not np.any(tensor[1])
    assert not np.any(valid[1])
    assert not np.any(fa[1])
    assert np.any(tensor[2:])
    assert qa["output_mask_contract"].endswith("T1 brain mask")
    assert qa["jacobian_contract"] == "finite-difference complete displacement"
    assert qa["jacobian_determinant_min"] == 1.0
    assert qa["jacobian_determinant_max"] == 1.0
    assert np.array_equal(np.asarray(nib.load(output_file).dataobj), tensor)


def test_nonlinear_nifti_rejects_ambiguous_warp_contract(tmp_path):
    shape = (3, 3, 3)
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    warp_file = tmp_path / "warp.nii.gz"
    affine_file = tmp_path / "affine.mat"
    output_file = tmp_path / "registered.nii.gz"
    nib.save(nib.Nifti1Image(_diagonal_tensor(shape), np.eye(4)), tensor_file)
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), reference_file)
    nib.save(nib.Nifti1Image(np.zeros(shape + (3,)), np.eye(4)), warp_file)
    np.savetxt(affine_file, np.eye(4))
    with pytest.raises(ValueError, match="only valid"):
        register_tensor_nonlinear_nifti(
            tensor_file,
            reference_file,
            warp_file,
            output_file,
            affine_matrix_file=affine_file,
        )
    with pytest.raises(ValueError, match="warp_kind"):
        register_tensor_nonlinear_nifti(
            tensor_file,
            reference_file,
            warp_file,
            output_file,
            warp_kind="automatic",
        )


def test_nonlinear_resampling_rejects_remaining_invalid_inputs():
    tensor = _diagonal_tensor((3, 3, 3))
    warp = np.zeros((3, 3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="positive dimensions"):
        resample_tensor_ppd_fsl(tensor, np.eye(4), (0, 3, 3), np.eye(4), warp)
    with pytest.raises(ValueError, match="source_affine"):
        resample_tensor_ppd_fsl(tensor, np.eye(3), (3, 3, 3), np.eye(4), warp)
    with pytest.raises(ValueError, match="reference_affine"):
        resample_tensor_ppd_fsl(tensor, np.eye(4), (3, 3, 3), np.eye(3), warp)
    with pytest.raises(ValueError, match="eigenvalue tolerance"):
        resample_tensor_ppd_fsl(
            tensor,
            np.eye(4),
            (3, 3, 3),
            np.eye(4),
            warp,
            repeated_eigenvalue_tolerance=0,
        )


def test_nonlinear_nifti_rejects_all_file_contract_errors(tmp_path):
    shape = (3, 3, 3)
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    warp_file = tmp_path / "warp.nii.gz"
    output_file = tmp_path / "registered.nii.gz"
    nib.save(nib.Nifti1Image(_diagonal_tensor(shape), np.eye(4)), tensor_file)
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), reference_file)
    nib.save(nib.Nifti1Image(np.zeros(shape + (3,)), np.eye(4)), warp_file)
    with pytest.raises(ValueError, match="workers"):
        register_tensor_nonlinear_nifti(
            tensor_file, reference_file, warp_file, output_file, workers=0
        )

    bad_tensor = tmp_path / "bad_tensor.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), bad_tensor)
    with pytest.raises(ValueError, match="six-component"):
        register_tensor_nonlinear_nifti(
            bad_tensor, reference_file, warp_file, output_file
        )
    bad_reference = tmp_path / "bad_reference.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape + (1,)), np.eye(4)), bad_reference)
    with pytest.raises(ValueError, match="three-dimensional"):
        register_tensor_nonlinear_nifti(
            tensor_file, bad_reference, warp_file, output_file
        )
    bad_warp = tmp_path / "bad_warp.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape + (3,)), np.diag((2, 1, 1, 1))), bad_warp)
    with pytest.raises(ValueError, match="displacement field"):
        register_tensor_nonlinear_nifti(
            tensor_file, reference_file, bad_warp, output_file
        )
    with pytest.raises(ValueError, match="required"):
        register_tensor_nonlinear_nifti(
            tensor_file,
            reference_file,
            warp_file,
            output_file,
            warp_kind="coefficients",
        )
    with pytest.raises(ValueError, match="could not be read"):
        register_tensor_nonlinear_nifti(
            tensor_file,
            reference_file,
            warp_file,
            output_file,
            warp_kind="coefficients",
            affine_matrix_file=tmp_path / "missing.mat",
        )
    bad_mask = tmp_path / "bad_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), bad_mask)
    with pytest.raises(ValueError, match="source mask"):
        register_tensor_nonlinear_nifti(
            tensor_file,
            reference_file,
            warp_file,
            output_file,
            source_mask_file=bad_mask,
        )
    with pytest.raises(ValueError, match="reference mask"):
        register_tensor_nonlinear_nifti(
            tensor_file,
            reference_file,
            warp_file,
            output_file,
            reference_mask_file=bad_mask,
        )
    with pytest.raises(ValueError, match="output mask"):
        register_tensor_nonlinear_nifti(
            tensor_file,
            reference_file,
            warp_file,
            output_file,
            output_mask_file=bad_mask,
        )
    with pytest.raises(ValueError, match="derivative_base"):
        register_tensor_nonlinear_nifti(
            tensor_file, reference_file, warp_file, output_file, derivative_base="a/b"
        )


def test_fnirt_nifti_runner_rejects_all_early_file_contract_errors(tmp_path):
    shape = (3, 3, 3)
    fa_file = tmp_path / "fa.nii.gz"
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    affine_file = tmp_path / "affine.mat"
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), fa_file)
    nib.save(nib.Nifti1Image(_diagonal_tensor(shape), np.eye(4)), tensor_file)
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), reference_file)
    np.savetxt(affine_file, np.eye(4))
    with pytest.raises(ValueError, match="brain_mask_file is required"):
        register_tensor_fnirt_nifti(
            fa_file, tensor_file, reference_file, affine_file, tmp_path
        )
    with pytest.raises(ValueError, match="workers"):
        register_tensor_fnirt_nifti(
            fa_file, tensor_file, reference_file, affine_file, tmp_path, workers=0
        )
    bad_fa = tmp_path / "bad_fa.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape + (1,)), np.eye(4)), bad_fa)
    with pytest.raises(ValueError, match="three-dimensional"):
        register_tensor_fnirt_nifti(
            bad_fa, tensor_file, reference_file, affine_file, tmp_path
        )
    bad_tensor = tmp_path / "bad_tensor.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape + (5,)), np.eye(4)), bad_tensor)
    with pytest.raises(ValueError, match="source grid"):
        register_tensor_fnirt_nifti(
            fa_file, bad_tensor, reference_file, affine_file, tmp_path
        )
    with pytest.raises(ValueError, match="could not be read"):
        register_tensor_fnirt_nifti(
            fa_file, tensor_file, reference_file, tmp_path / "missing.mat", tmp_path
        )
    mask_cases = (
        (np.ones(shape + (1,)), np.eye(4), "three-dimensional"),
        (np.ones((2, 2, 2)), np.eye(4), "share the reference grid"),
        (np.full(shape, np.nan), np.eye(4), "finite nonzero support"),
        (np.zeros(shape), np.eye(4), "finite nonzero support"),
    )
    for index, (values, affine, message) in enumerate(mask_cases):
        mask = tmp_path / f"invalid-mask-{index}.nii.gz"
        nib.save(nib.Nifti1Image(values, affine), mask)
        with pytest.raises(ValueError, match=message):
            register_tensor_fnirt_nifti(
                fa_file,
                tensor_file,
                reference_file,
                affine_file,
                tmp_path,
                brain_mask_file=mask,
            )
    np.savetxt(affine_file, np.eye(3))
    with pytest.raises(ValueError, match="finite and 4x4"):
        register_tensor_fnirt_nifti(
            fa_file, tensor_file, reference_file, affine_file, tmp_path
        )


def test_fnirt_nifti_runner_writes_complete_contract_with_mocked_optimizer(
    tmp_path, monkeypatch
):
    shape = (5, 4, 3)
    spacing = (2, 2, 2)
    fa_file = tmp_path / "fa.nii.gz"
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    brain_mask_file = tmp_path / "brain_mask.nii.gz"
    affine_file = tmp_path / "affine.mat"
    output = tmp_path / "output"
    stale_attempt = output / f".fnirt-attempt-{os.getpid()}"
    stale_attempt.mkdir(parents=True)
    (stale_attempt / "partial.txt").write_text("partial\n", encoding="utf-8")
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), fa_file)
    nib.save(nib.Nifti1Image(_diagonal_tensor(shape), np.eye(4)), tensor_file)
    reference_affine = np.diag((0.7, 0.8, 0.9, 1.0))
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.float32), reference_affine),
        reference_file,
    )
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.uint8), reference_affine),
        brain_mask_file,
    )
    affine_matrix = np.eye(4)
    affine_matrix[:3, 3] = (1.0, -2.0, 3.0)
    np.savetxt(affine_file, affine_matrix)
    expansion = expand_fnirt_coefficients(
        np.zeros(fsl_coefficient_shape(shape, spacing) + (3,)),
        shape,
        np.eye(4),
        np.eye(4),
        knot_spacing=spacing,
    )
    levels = tuple(
        type(
            "Level",
            (),
            {
                "successful_iterations": index,
                "trace": (),
                "cost": float(index),
                "status": "test",
            },
        )()
        for index in range(4)
    )
    fake_result = type(
        "Result",
        (),
        {
            "coefficients": np.zeros(fsl_coefficient_shape(shape, spacing) + (3,)),
            "expansion": expansion,
            "levels": levels,
            "jacobian_ranges": ((1.0, 1.0),) * 4,
        },
    )()
    monkeypatch.setattr(
        fnirt_module, "run_simnibs46_fnirt", lambda *args, **kwargs: fake_result
    )
    monkeypatch.setattr(
        "dwi2cond_xp.preprocessing.nonlinear.register_tensor_nonlinear_nifti",
        lambda *args, **kwargs: {
            "outputs": {
                "tensor": "tensor",
                "valid_mask": "valid",
                "jacobian": "jacobian",
                "fa": "fa",
                "v1": "v1",
            }
        },
    )
    qa = register_tensor_fnirt_nifti(
        fa_file,
        tensor_file,
        reference_file,
        affine_file,
        output,
        brain_mask_file=brain_mask_file,
        workers=2,
    )
    assert qa["status"] == "completed"
    assert len(qa["levels"]) == 4
    assert ".fnirt-attempt" not in json.dumps(qa)
    assert ".fnirt-attempt" not in (
        output / "nonlinear_registration_qa.json"
    ).read_text(encoding="utf-8")
    coefficient_image = nib.load(output / "FA2T1_warp.nii.gz")
    coefficient_header = coefficient_image.header
    assert int(coefficient_header["intent_code"]) == 2007
    assert np.array_equal(np.asarray(coefficient_header["pixdim"])[1:4], spacing)
    assert np.allclose(
        [
            coefficient_header["intent_p1"],
            coefficient_header["intent_p2"],
            coefficient_header["intent_p3"],
        ],
        (0.7, 0.8, 0.9),
        rtol=0.0,
        atol=1e-7,
    )
    assert np.array_equal(
        [
            coefficient_header["qoffset_x"],
            coefficient_header["qoffset_y"],
            coefficient_header["qoffset_z"],
        ],
        shape,
    )
    assert np.array_equal(coefficient_image.get_sform(), affine_matrix)
    assert (output / "nonlinear_registration_qa.json").is_file()
    for name in (
        "FA2T1_warp.nii.gz",
        "FA2T1_field.nii.gz",
        "FA2T1_jacobian.nii.gz",
        "DTI_FA_nonlin.nii.gz",
    ):
        assert (output / name).is_file()


@pytest.mark.skipif(
    not FSL_VECREG.is_file(),
    reason="FSL reference disabled; set FSL_VECREG to a local vecreg executable",
)
def test_shear_ppd_matches_fsl_vecreg(tmp_path):
    shape = (9, 8, 7)
    tensor = _diagonal_tensor(shape)
    warp = np.zeros(shape + (3,), dtype=np.float32)
    warp[..., 1] = 0.1 * np.indices(shape, dtype=np.float32)[0]
    tensor_file = tmp_path / "tensor.nii.gz"
    reference_file = tmp_path / "reference.nii.gz"
    warp_file = tmp_path / "warp.nii.gz"
    output_file = tmp_path / "fsl.nii.gz"
    nib.save(nib.Nifti1Image(tensor, np.eye(4)), tensor_file)
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), reference_file
    )
    warp_image = nib.Nifti1Image(warp, np.eye(4))
    warp_image.header.set_intent("vector")
    nib.save(warp_image, warp_file)
    environment = os.environ.copy()
    environment["FSLDIR"] = str(FSL_VECREG.parent.parent)
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    subprocess.run(
        [
            str(FSL_VECREG),
            "-i",
            str(tensor_file),
            "-o",
            str(output_file),
            "-r",
            str(reference_file),
            "-w",
            str(warp_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    reference = np.asarray(nib.load(output_file).dataobj)
    ours = resample_tensor_ppd_fsl(tensor, np.eye(4), shape, np.eye(4), warp).tensor
    common = np.any(reference != 0.0, axis=-1) & np.any(ours != 0.0, axis=-1)
    difference = np.abs(reference[common] - ours[common])
    float32_rounding_bound = (
        2.0 * np.finfo(np.float32).eps * np.max(np.abs(reference[common]))
    )
    assert np.max(difference) <= float32_rounding_bound
    assert (
        np.linalg.norm((reference[common] - ours[common]).ravel())
        / np.linalg.norm(reference[common].ravel())
        < 5e-8
    )
