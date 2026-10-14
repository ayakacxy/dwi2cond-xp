import numpy as np
import pytest

from dwi2cond_xp.preprocessing import fnirt
from dwi2cond_xp.preprocessing.topup import (
    _spline_jte_fsl_order,
    expand_spline_coefficients,
    fsl_coefficient_shape,
    spline_design_matrix,
)


@pytest.mark.parametrize("shape,spacing,dtype", [
    ((7, 8, 9), (2, 3, 2), np.float64),
    ((8, 9, 7), (3, 2, 4), np.float32),
])
def test_fused_displacement_matches_three_original_scalar_expansions(shape, spacing, dtype):
    rng = np.random.default_rng(47)
    coefficients = rng.normal(size=fsl_coefficient_shape(shape, spacing) + (3,)).astype(dtype)
    coefficients[::2, ::2, ::2] = 0
    basis = tuple(spline_design_matrix(size, step) for size, step in zip(shape, spacing))
    expected = np.stack([
        expand_spline_coefficients(coefficients[..., component], shape, spacing)
        for component in range(3)
    ], axis=3)
    actual = fnirt._expand_fnirt_displacement_values(coefficients.astype(np.float64), *basis)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", ["finite", "nan", "overflow"])
def test_fused_gradient_projection_preserves_scalar_reductions_and_nonfinite_values(case):
    shape, spacing = (7, 8, 9), (2, 3, 2)
    rng = np.random.default_rng(59)
    difference = rng.normal(size=shape).astype(np.float32)
    derivatives = rng.normal(size=shape + (3,)).astype(np.float32)
    mask = rng.random(shape) > 0.2
    if case == "nan":
        derivatives[-1, 2, 3, 0] = np.nan
        mask[-1, 2, 3] = True
    elif case == "overflow":
        difference[-1, 2, 3] = np.finfo(np.float32).max
        derivatives[-1, 2, 3, 1] = 2
        mask[-1, 2, 3] = True
    basis = tuple(spline_design_matrix(size, step) for size, step in zip(shape, spacing))
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.stack([
            _spline_jte_fsl_order(
                np.multiply(difference, derivatives[..., component], dtype=np.float32),
                mask, *basis,
            )
            for component in range(3)
        ], axis=3)
        actual = fnirt._fnirt_displacement_jte(difference, derivatives, mask, *basis)
    np.testing.assert_array_equal(actual, expected)
