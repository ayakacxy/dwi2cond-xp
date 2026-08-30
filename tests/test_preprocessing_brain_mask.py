from __future__ import annotations

import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import dwi2cond_xp.preprocessing.brain_mask as brain_mask_module
from dwi2cond_xp.preprocessing import (
    bet_brain_mask,
    robust_intensity_limits,
    write_bet_brain_mask,
)


def _sphere(shape: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    coordinates = np.indices(shape, dtype=np.float64)
    center = (np.asarray(shape) - 1.0)[:, None, None, None] * 0.5
    radius = np.sqrt(np.sum((coordinates - center) ** 2, axis=0))
    return np.where(radius <= 7.5, 500.0 - 8.0 * radius, 0.0).astype(np.float32)


def test_robust_limits_match_frozen_sphere_and_validate() -> None:
    values = _sphere()
    minimum, maximum = robust_intensity_limits(values)
    assert minimum == 0.0
    assert maximum == pytest.approx(473.8420, abs=1e-3)
    assert robust_intensity_limits(np.ones((4, 4, 4), dtype=np.float32)) == (1.0, 1.0)
    long_tail = np.concatenate(
        [np.zeros(1000, dtype=np.float32), np.ones(1000, dtype=np.float32), [1e9]]
    )
    low, high = robust_intensity_limits(long_tail)
    assert np.isfinite(low) and np.isfinite(high)
    with pytest.raises(ValueError, match="non-empty finite"):
        robust_intensity_limits(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty finite"):
        robust_intensity_limits(np.array([np.nan], dtype=np.float32))


def test_mesh_topology_geometry_and_rasterization_are_deterministic() -> None:
    vertices, faces = brain_mask_module._icosphere(5)
    assert vertices.shape == (2562, 3)
    assert faces.shape == (5120, 3)
    neighbours, neighbour_valid = brain_mask_module._mesh_arrays(faces, len(vertices))
    incident, incident_valid = brain_mask_module._incident_face_arrays(
        faces, len(vertices)
    )
    assert set(neighbour_valid.sum(axis=1)) == {5, 6}
    assert np.all(incident_valid.sum(axis=1) == neighbour_valid.sum(axis=1))

    scaled = 9.5 + 7.0 * vertices
    single = brain_mask_module._rasterize_surface(
        scaled, faces, (20, 20, 20), np.ones(3), workers=1
    )
    parallel = brain_mask_module._rasterize_surface(
        scaled, faces, (20, 20, 20), np.ones(3), workers=2
    )
    assert np.array_equal(single, parallel)
    assert single[10, 10, 10] == 1
    assert single[0, 0, 0] == 0

    empty_score = brain_mask_module._self_intersection_score(
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        1.0,
        1.0,
    )
    assert empty_score == 0.0


def test_bet_reference_and_optimized_backends_preserve_contract() -> None:
    values = _sphere()
    reference = bet_brain_mask(values, np.full(3, 2.0), workers=1, backend="reference")
    optimized = bet_brain_mask(values, np.full(3, 2.0), workers=2, backend="optimized")
    repeated = bet_brain_mask(values, np.full(3, 2.0), workers=2, backend="optimized")
    assert optimized.vertices_mm.shape == (2562, 3)
    assert optimized.faces.shape == (5120, 3)
    assert np.array_equal(optimized.mask, repeated.mask)
    assert np.array_equal(optimized.vertices_mm, repeated.vertices_mm)
    intersection = np.count_nonzero((reference.mask > 0) & (optimized.mask > 0))
    dice = 2.0 * intersection / (reference.mask.sum() + optimized.mask.sum())
    assert dice > 0.99
    assert optimized.passes == 0
    assert optimized.self_intersection_score < 4000.0
    assert optimized.center_mm == pytest.approx([19.0, 19.0, 19.0], abs=1e-10)


def test_bet_validation_and_internal_edge_paths(monkeypatch) -> None:
    values = _sphere((8, 8, 8))
    cases = [
        (values[..., None], np.ones(3), {}, "three-dimensional"),
        (values, np.array([1.0, 0.0, 1.0]), {}, "finite positive"),
        (values, np.ones(3), {"fractional_threshold": 0.0}, "between zero and one"),
        (values, np.ones(3), {"gradient_threshold": 2.0}, "minus one and one"),
        (values, np.ones(3), {"workers": 0}, "worker count"),
        (values, np.ones(3), {"backend": "cuda"}, "reference or optimized"),
    ]
    for image, sizes, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            bet_brain_mask(image, sizes, **kwargs)
    bad = values.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        bet_brain_mask(bad, np.ones(3))
    with pytest.raises(ValueError, match="initialize a center"):
        bet_brain_mask(np.zeros((8, 8, 8), dtype=np.float32), np.ones(3))
    one_voxel = np.zeros((8, 8, 8), dtype=np.float32)
    one_voxel[4, 4, 4] = 50.0
    monkeypatch.setattr(
        brain_mask_module, "robust_intensity_limits", lambda values: (0.0, 100.0)
    )
    with pytest.raises(ValueError, match="within-brain median"):
        bet_brain_mask(one_voxel, np.ones(3))

    vertices, faces = brain_mask_module._icosphere(1)
    neighbours, valid = brain_mask_module._mesh_arrays(faces, len(vertices))
    flat_faces = faces.reshape(-1)
    face_ids = np.repeat(np.arange(len(faces)), 3)
    incidence = brain_mask_module.csr_matrix(
        (np.ones(flat_faces.size), (flat_faces, face_ids)),
        shape=(len(vertices), len(faces)),
    )
    average = brain_mask_module.csr_matrix(np.eye(len(vertices), dtype=np.float64))
    with pytest.raises(RuntimeError, match="no valid normal"):
        brain_mask_module._mesh_geometry(
            np.zeros_like(vertices),
            faces,
            incidence,
            average,
            neighbours,
            valid,
        )

    monkeypatch.setattr(
        brain_mask_module, "label", lambda values: (np.zeros_like(values), 1)
    )
    with pytest.raises(RuntimeError, match="centroid lies"):
        brain_mask_module._rasterize_surface(
            4.0 + vertices, faces, (9, 9, 9), np.ones(3), workers=1
        )


def test_optimized_python_body_and_reference_smoothing_paths() -> None:
    assert brain_mask_module._sequential_mean.py_func(np.array([1.0, 3.0])) == 2.0
    assert brain_mask_module._smoothing_increase(1, 0) == 100.0
    assert brain_mask_module._smoothing_increase(1, 800) == pytest.approx(80.2)
    assert brain_mask_module._optimized_smoothing(0.2, -1.0, 1, 800) == 0.2
    assert brain_mask_module._optimized_smoothing(0.2, 1.0, 1, 800) == 1.0
    assert brain_mask_module._fit_force(2.0, 1.0, 2.0) == 1.0
    assert brain_mask_module._fit_force(2.0, 1.0, 0.0) == 2.0
    base_smoothing = np.array([0.2, 0.3])
    assert (
        brain_mask_module._increase_outward_smoothing(
            base_smoothing, np.array([1.0, -1.0]), 0, 0
        )
        is base_smoothing
    )
    assert brain_mask_module._increase_outward_smoothing(
        base_smoothing, np.array([1.0, -1.0]), 1, 800
    ) == pytest.approx([1.0, 0.3])
    image = _sphere((20, 20, 20))
    vertices, faces = brain_mask_module._icosphere(1)
    vertices = 19.0 + 7.0 * vertices
    neighbours, neighbour_valid = brain_mask_module._mesh_arrays(faces, len(vertices))
    incident, incident_valid = brain_mask_module._incident_face_arrays(
        faces, len(vertices)
    )
    inward_distances, inward_updates_maximum = (
        brain_mask_module._fsl_inward_sampling_contract(np.full(3, 2.0))
    )
    optimized = brain_mask_module._evolve_surface_optimized.py_func(
        image,
        vertices,
        faces,
        neighbours,
        neighbour_valid,
        incident,
        incident_valid,
        np.full(3, 2.0),
        inward_distances,
        inward_updates_maximum,
        0.0,
        45.0,
        450.0,
        0.2**0.275,
        19.0,
        15.0,
        0.1,
        1,
    )
    assert optimized.shape == vertices.shape
    dented = vertices.copy()
    dented[0] = 19.0 + (vertices[0] - 19.0) * (6.9 / 7.0)
    reference, score, mean_edge = brain_mask_module._evolve_surface(
        image,
        dented,
        faces,
        np.full(3, 2.0),
        0.0,
        45.0,
        450.0,
        0.2,
        19.0,
        15.0,
        0.1,
        pass_number=1,
        backend="reference",
        workers=1,
    )
    assert reference.shape == vertices.shape
    assert np.isfinite(score) and mean_edge > 0


def test_bet_inward_sampling_matches_fsl_endpoint_and_submillimetre_loop() -> None:
    one_mm_distances, one_mm_maximum = (
        brain_mask_module._fsl_inward_sampling_contract(np.ones(3))
    )
    np.testing.assert_array_equal(one_mm_distances, [1, 7, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(
        one_mm_maximum, [True, False, True, False, False, False, False]
    )

    half_mm_distances, half_mm_maximum = (
        brain_mask_module._fsl_inward_sampling_contract(
            np.array([0.5, 0.8, 1.2])
        )
    )
    np.testing.assert_array_equal(
        half_mm_distances,
        [1, 7, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
    )
    np.testing.assert_array_equal(
        half_mm_maximum,
        [
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )
    with pytest.raises(ValueError, match="positive finite"):
        brain_mask_module._fsl_inward_sampling_contract(np.array([1.0, 0.0, 1.0]))

    image = np.full((12, 3, 3), 100.0, dtype=np.float32)
    image[1, 1, 1] = 0.0
    minimum, maximum, valid = brain_mask_module._sample_inward_extrema(
        image,
        np.array([[8.0, 1.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.ones(3),
        0.0,
        10.0,
        100.0,
    )
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_array_equal(minimum, [0.0])
    np.testing.assert_array_equal(maximum, [100.0])
    force = brain_mask_module._fit_force(
        minimum[0],
        (maximum[0] - 0.0) * (0.2**0.275),
        maximum[0],
    )
    assert force < 0.0

    submillimetre = np.full((20, 3, 3), 100.0, dtype=np.float32)
    submillimetre[13, 1, 1] = 0.0
    minimum, maximum, valid = brain_mask_module._sample_inward_extrema(
        submillimetre,
        np.array([[8.0, 0.5, 0.5]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.full(3, 0.5),
        0.0,
        10.0,
        100.0,
    )
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_array_equal(minimum, [0.0])
    np.testing.assert_array_equal(maximum, [100.0])

    _minimum, _maximum, valid = brain_mask_module._sample_inward_extrema(
        image,
        np.array([[5.0, 1.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.ones(3),
        0.0,
        10.0,
        100.0,
    )
    np.testing.assert_array_equal(valid, [False])


def test_bet_self_intersection_rerun_path(monkeypatch) -> None:
    calls = []

    def fake_evolve(image, vertices, faces, *args, pass_number, **kwargs):
        calls.append(pass_number)
        score = 5000.0 if pass_number == 0 else 0.0
        return vertices, score, 1.0

    monkeypatch.setattr(brain_mask_module, "_evolve_surface", fake_evolve)
    monkeypatch.setattr(
        brain_mask_module,
        "_rasterize_surface",
        lambda vertices, faces, shape, sizes, workers: np.ones(shape, dtype=np.uint8),
    )
    result = bet_brain_mask(_sphere(), np.full(3, 2.0), workers=1)
    assert calls == [0, 1]
    assert result.passes == 1


def test_write_bet_mask_preserves_nifti_geometry(tmp_path: Path) -> None:
    values = _sphere()
    affine = np.array(
        [[-2.0, 0, 0, 19], [0, 2.0, 0, -19], [0, 0, 2.0, -19], [0, 0, 0, 1]]
    )
    input_file = tmp_path / "input.nii.gz"
    output_file = tmp_path / "mask.nii.gz"
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    nib.save(image, input_file)
    result = write_bet_brain_mask(input_file, output_file, workers=2)
    output = nib.load(output_file)
    assert output.get_data_dtype() == np.dtype(np.uint8)
    assert np.array_equal(np.asarray(output.dataobj), result.mask)
    assert output.get_qform(coded=True)[1] == 1
    assert output.get_sform(coded=True)[1] == 2

    four_dimensional = tmp_path / "4d.nii.gz"
    nib.save(nib.Nifti1Image(values[..., None], affine), four_dimensional)
    with pytest.raises(ValueError, match="three-dimensional"):
        write_bet_brain_mask(four_dimensional, output_file)


@pytest.mark.skipif(not os.environ.get("FSL_BET"), reason="FSL_BET is not configured")
def test_bet_matches_configured_fsl(tmp_path: Path) -> None:
    values = _sphere()
    input_file = tmp_path / "input.nii.gz"
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(values, affine), input_file)
    subprocess.run(
        [
            os.environ["FSL_BET"],
            str(input_file),
            str(tmp_path / "fsl"),
            "-f",
            "0.2",
            "-m",
        ],
        check=True,
        env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"},
        capture_output=True,
        text=True,
    )
    ours = bet_brain_mask(values, np.full(3, 2.0), workers=2).mask > 0
    reference = np.asarray(nib.load(tmp_path / "fsl_mask.nii.gz").dataobj) > 0
    dice = 2.0 * np.count_nonzero(ours & reference) / (ours.sum() + reference.sum())
    assert dice > 0.985


@pytest.mark.skipif(not os.environ.get("FSL_BET"), reason="FSL_BET is not configured")
def test_bet_submillimetre_adversarial_mask_matches_configured_fsl(
    tmp_path: Path,
) -> None:
    shape = (64, 64, 64)
    voxel_size = 0.5
    coordinates = np.indices(shape, dtype=np.float64) * voxel_size
    center = (np.asarray(shape) - 1.0) * voxel_size * 0.5
    radius = np.sqrt(
        np.sum((coordinates - center[:, None, None, None]) ** 2, axis=0)
    )
    values = np.where(radius <= 12.0, 500.0 - 5.0 * radius, 0.0)
    values[(radius >= 4.35) & (radius <= 4.65)] = 20.0
    values = values.astype(np.float32)
    input_file = tmp_path / "submillimetre.nii.gz"
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    nib.save(nib.Nifti1Image(values, affine), input_file)
    subprocess.run(
        [
            os.environ["FSL_BET"],
            str(input_file),
            str(tmp_path / "fsl-submillimetre"),
            "-f",
            "0.2",
            "-m",
        ],
        check=True,
        env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"},
        capture_output=True,
        text=True,
    )
    ours = bet_brain_mask(values, np.full(3, voxel_size), workers=2).mask > 0
    reference = np.asarray(
        nib.load(tmp_path / "fsl-submillimetre_mask.nii.gz").dataobj
    ) > 0
    dice = 2.0 * np.count_nonzero(ours & reference) / (ours.sum() + reference.sum())
    assert dice > 0.985
