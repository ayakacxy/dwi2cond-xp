import numpy as np
import pytest

from dwi2cond_xp.conductivity import (
    _adjust_excentricity,
    _anisotropic_intensity_scale,
    _fix_eigenvalues,
    correct_fsl_tensor_basis,
    tensors_to_conductivity,
)


def test_anisotropic_intensity_scale_rejects_degenerate_aggregates():
    with pytest.raises(ValueError, match="denominator"):
        _anisotropic_intensity_scale({}, {})
    with pytest.raises(ValueError, match="intensity scale"):
        _anisotropic_intensity_scale({1: 1.0}, {1: np.inf})


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


def test_fsl_basis_correction_validates_shapes_axes_and_ras_reflection():
    tensor = np.array([[[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 3.0]]])
    corrected = correct_fsl_tensor_basis(tensor, np.eye(4))
    expected = tensor.copy()
    expected[:, 0, 1] *= -1
    expected[:, 1, 0] *= -1
    expected[:, 0, 2] *= -1
    expected[:, 2, 0] *= -1
    assert np.allclose(corrected, expected)
    with pytest.raises(ValueError, match="Nx3x3"):
        correct_fsl_tensor_basis(np.zeros((2, 6)), np.eye(4))
    affine = np.eye(4)
    affine[0, 0] = 0
    with pytest.raises(ValueError, match="zero-length"):
        correct_fsl_tensor_basis(tensor, affine)


def test_eigenvalue_repairs_and_excentricity_branches():
    values, report = _fix_eigenvalues(
        np.array([[-1.0, -2.0, -3.0], [20.0, 0.01, 0.001]]),
        max_value=2.0,
        max_ratio=10.0,
        fallback=0.3,
    )
    assert np.allclose(values[0], 0.3)
    assert np.allclose(values[1], [2.0, 0.2, 0.2])
    assert report["negative_semidefinite_tensors"] == 1
    assert report["capped_eigenvalues"] == 1

    base = np.array([[4.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    for scaling in (0.25, 0.5, 0.75):
        adjusted = _adjust_excentricity(base, scaling)
        assert np.allclose(np.prod(adjusted, axis=1), np.prod(base, axis=1))
        assert np.array_equal(adjusted[1], base[1])
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        _adjust_excentricity(base, 1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        _adjust_excentricity(np.array([[1.0, 0.0, 0.0]]), 0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "bad"}, "mode"),
        ({"max_ratio": 0.5}, "max_ratio"),
        ({"max_cond": 0}, "max_cond"),
        ({"weights": np.array([0.0])}, "weights"),
    ],
)
def test_conductivity_rejects_invalid_contract(kwargs, message):
    tensor = np.eye(3)[None]
    with pytest.raises(ValueError, match=message):
        tensors_to_conductivity(tensor, np.array([1]), {1: 0.1}, **kwargs)
    with pytest.raises(ValueError, match="Nx3x3"):
        tensors_to_conductivity(np.zeros((1, 6)), np.array([1]), {1: 0.1})


@pytest.mark.parametrize("scalar", [0.0, np.inf])
def test_conductivity_rejects_invalid_scalar(scalar):
    with pytest.raises(ValueError, match="finite and positive"):
        tensors_to_conductivity(np.eye(3)[None], np.array([1]), {1: scalar})


def test_dir_without_intensity_uses_sequence_and_excentricity():
    tensors = np.stack((np.zeros((3, 3)), np.diag([4.0, 2.0, 1.0])))
    conductivity, report = tensors_to_conductivity(
        tensors,
        np.array([1, 1]),
        [0.126],
        mode="dir",
        anisotropic_tissues=(1,),
        correct_intensity=False,
        excentricity_scaling=0.5,
    )
    assert np.all(np.linalg.eigvalsh(conductivity) > 0)
    assert report["tissues"]["1"]["zero_tensors"] == 1
    assert "fix" in report["tissues"]["1"]


def test_vn_excentricity_and_mc_without_intensity_are_supported():
    tensor = np.diag([4.0, 2.0, 1.0])[None]
    vn, _ = tensors_to_conductivity(
        tensor,
        np.array([1]),
        {1: 0.126},
        mode="vn",
        anisotropic_tissues=(1,),
        excentricity_scaling=0.75,
    )
    mc, _ = tensors_to_conductivity(
        tensor,
        np.array([1]),
        {1: 0.126},
        mode="mc",
        anisotropic_tissues=(1,),
        correct_intensity=False,
    )
    assert np.linalg.det(vn[0]) == pytest.approx(0.126**3)
    assert np.allclose(mc[0], np.eye(3) * 4.0 ** (1 / 3))


@pytest.mark.parametrize("mode", ["dir", "mc"])
def test_dir_mc_zero_tensor_follows_post_reconstruction_fix(mode):
    tensors = np.stack(
        (
            np.zeros((3, 3)),
            np.diag([1.5e-3, 0.7e-3, 0.4e-3]),
            np.diag([1.4e-3, 0.8e-3, 0.5e-3]),
            np.diag([1.6e-3, 0.9e-3, 0.6e-3]),
        )
    )
    conductivity, report = tensors_to_conductivity(
        tensors,
        np.ones(4, dtype=int),
        {1: 0.126},
        mode=mode,
        anisotropic_tissues=(1,),
    )
    assert np.array_equal(conductivity[0], np.eye(3) * 0.126)
    assert report["intensity_scale"] == pytest.approx(163.106112448201, rel=1e-12)


@pytest.mark.parametrize("mode", ["dir", "mc"])
def test_dir_mc_reject_all_zero_anisotropic_tissue_explicitly(mode):
    with pytest.raises(ValueError, match="no positive finite mean determinant"):
        tensors_to_conductivity(
            np.zeros((3, 3, 3), dtype=np.float64),
            np.ones(3, dtype=int),
            {1: 0.126},
            mode=mode,
            anisotropic_tissues=(1,),
        )
