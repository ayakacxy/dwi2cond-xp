import numpy as np

from dwi2cond_xp.tensor_fit import fit_tensor_wls, form_design_matrix


def _gradient_fixture():
    directions = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1],
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((2, 3)), directions])
    bvals = np.concatenate([np.zeros(2), np.full(len(directions), 1000.0)])
    return bvals, bvecs


def test_recovers_noiseless_tensor():
    bvals, bvecs = _gradient_fixture()
    expected = np.array([1.5e-3, 1.0e-4, -5.0e-5, 7.0e-4, 8.0e-5, 4.0e-4])
    design = form_design_matrix(bvals, bvecs)
    signal = np.exp(-(design[:, :6] @ expected + design[:, 6] * -np.log(1000.0)))
    fitted = fit_tensor_wls(np.vstack([signal, signal * 0.8]), bvals, bvecs)
    assert np.allclose(fitted[0], expected, rtol=1e-10, atol=1e-12)
    assert np.allclose(fitted[1], expected, rtol=1e-10, atol=1e-12)


def test_grad_dev_matches_explicit_transformed_gradients():
    bvals, bvecs = _gradient_fixture()
    grad = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]])
    actual = form_design_matrix(bvals, bvecs, grad)[0]
    transform = np.array(
        [[1.01, 0.04, 0.07], [0.02, 1.05, 0.08], [0.03, 0.06, 1.09]]
    )
    transformed = (transform @ bvecs.T).T
    expected = np.column_stack(
        [
            bvals * transformed[:, 0] ** 2,
            2 * bvals * transformed[:, 0] * transformed[:, 1],
            2 * bvals * transformed[:, 0] * transformed[:, 2],
            bvals * transformed[:, 1] ** 2,
            2 * bvals * transformed[:, 1] * transformed[:, 2],
            bvals * transformed[:, 2] ** 2,
            np.ones_like(bvals),
        ]
    )
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)
