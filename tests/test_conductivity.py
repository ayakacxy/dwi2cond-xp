import numpy as np

from dwi2cond_xp.conductivity import (
    correct_fsl_tensor_basis,
    tensors_to_conductivity,
)


def test_fsl_basis_correction_flips_x_cross_terms_for_las():
    tensor = np.array([[[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 3.0]]])
    affine = np.diag([-1.25, 1.25, 1.25, 1.0])
    corrected = correct_fsl_tensor_basis(tensor, affine)
    expected = tensor.copy()
    expected[:, 0, 1] *= -1
    expected[:, 1, 0] *= -1
    expected[:, 0, 2] *= -1
    expected[:, 2, 0] *= -1
    assert np.allclose(corrected, expected)


def test_vn_preserves_direction_and_sets_determinant():
    tensor = np.diag([4.0, 2.0, 1.0])[None, ...]
    conductivity, report = tensors_to_conductivity(
        tensor,
        np.array([1]),
        {1: 0.126},
        mode="vn",
        anisotropic_tissues=(1,),
        max_cond=10,
        max_ratio=10,
    )
    assert np.allclose(np.linalg.det(conductivity[0]), 0.126**3)
    assert np.argmax(np.diag(conductivity[0])) == 0
    assert report["mode"] == "vn"


def test_mc_is_isotropic_and_uses_simnibs_global_scale():
    tensors = np.stack((np.diag([4.0, 2.0, 1.0]), np.diag([8.0, 4.0, 2.0])))
    conductivity, report = tensors_to_conductivity(
        tensors,
        np.array([1, 1]),
        {1: 0.126},
        mode="mc",
        anisotropic_tissues=(1,),
    )
    scale = 0.126 / (36.0 ** (1.0 / 3.0))
    expected = np.stack((2.0 * scale * np.eye(3), 4.0 * scale * np.eye(3)))
    assert np.allclose(conductivity, expected)
    assert np.isclose(
        report["tissues"]["1"]["mean_conductivity"],
        np.mean([2.0 * scale, 4.0 * scale]),
    )


def test_non_anisotropic_tissue_uses_scalar_value():
    tensor = np.diag([4.0, 2.0, 1.0])[None, ...]
    conductivity, _ = tensors_to_conductivity(
        tensor,
        np.array([3]),
        {3: 0.275},
        mode="dir",
        anisotropic_tissues=(1, 2),
    )
    assert np.array_equal(conductivity[0], 0.275 * np.eye(3))
