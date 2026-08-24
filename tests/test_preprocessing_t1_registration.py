from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

import dwi2cond_xp.preprocessing.t1_registration as t1_module
from dwi2cond_xp.preprocessing.flirt_registration import FlirtRegistrationResult
from dwi2cond_xp.preprocessing.t1_registration import (
    prepare_charm_t1_inputs,
    run_t1_registration_nifti,
)


def _save(path, values, affine=None):
    matrix = np.eye(4) if affine is None else affine
    image = nib.Nifti1Image(values, matrix)
    image.set_qform(matrix, 1)
    image.set_sform(matrix, 1)
    nib.save(image, path)


def _fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    shape = (9, 8, 7)
    grid = np.indices(shape, dtype=np.float32)
    anatomy = np.exp(
        -(
            (grid[0] - 3.0) ** 2 / 5.0
            + (grid[1] - 4.0) ** 2 / 7.0
            + (grid[2] - 2.0) ** 2 / 4.0
        )
    ).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int16)
    labels[1:-1, 1:-1, 1:-1] = 2
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[..., 0] = anatomy * np.float32(1.5e-3)
    tensor[..., 3] = anatomy * np.float32(0.6e-3)
    tensor[..., 5] = anatomy * np.float32(0.3e-3)
    paths = {}
    for name, values in (
        ("t1", anatomy),
        ("corrected", anatomy),
        ("labels", labels),
        ("fa", anatomy),
        ("sse", anatomy * 2),
        ("tensor", tensor),
    ):
        paths[name] = tmp_path / f"{name}.nii.gz"
        _save(paths[name], values)
    return paths, labels


def test_prepare_charm_t1_inputs_matches_script_contract(tmp_path):
    paths, labels = _fixture(tmp_path)
    outputs = prepare_charm_t1_inputs(
        paths["t1"], paths["labels"], paths["corrected"], tmp_path / "prepared"
    )
    mask = np.asarray(nib.load(outputs["brain_mask"]).dataobj)
    brain = np.asarray(nib.load(outputs["t1_brain"]).dataobj)
    rim = np.asarray(nib.load(outputs["brain_rim"]).dataobj)
    weight = np.asarray(nib.load(outputs["reference_weight"]).dataobj)
    expected = ((labels >= 1) & (labels <= 499)).astype(np.uint8)
    assert np.array_equal(mask, expected)
    assert np.array_equal(brain, np.asarray(nib.load(paths["corrected"]).dataobj) * expected)
    assert set(np.unique(rim)).issubset({0, 1})
    assert weight.dtype == np.float32
    assert np.all(weight >= 0)


@pytest.mark.parametrize("degrees_of_freedom", [6, 12])
def test_t1_registration_writes_complete_grid_and_qa(
    tmp_path, monkeypatch, degrees_of_freedom
):
    paths, labels = _fixture(tmp_path)
    calls = []

    def fake_estimate(*args, degrees_of_freedom, **kwargs):
        calls.append(degrees_of_freedom)
        return FlirtRegistrationResult(np.eye(4), 0.125, 42, 7)

    monkeypatch.setattr(t1_module, "_estimate", fake_estimate)
    progress = []
    output = tmp_path / "output"
    report = run_t1_registration_nifti(
        paths["tensor"],
        paths["fa"],
        paths["t1"],
        paths["labels"],
        paths["corrected"],
        output,
        sse_file=paths["sse"],
        degrees_of_freedom=degrees_of_freedom,
        workers=8,
        progress=lambda phase, done, total: progress.append((phase, done, total)),
    )
    assert calls == ([6] if degrees_of_freedom == 6 else [12, 6])
    assert report["fallback"] == "none"
    assert report["mode"] == ("rigid" if degrees_of_freedom == 6 else "affine")
    assert report["valid_voxels"] > 0
    assert progress[-1] == ("qa", 1, 1)
    for name in (
        "FA2T1.mat",
        "FA2T1_world.mat",
        "DTI_coregT1_tensor.nii.gz",
        "DTI_coregT1_FA.nii.gz",
        "DTI_coregT1_V1.nii.gz",
        "DTI_FA_6dof_QA.nii.gz",
        "DTI_SSE_6dof_QA.nii.gz",
        "T1_brainrim_QA.nii.gz",
        "t1_registration_qa.json",
    ):
        assert (output / name).is_file()
    tensor_image = nib.load(output / "DTI_coregT1_tensor.nii.gz")
    t1_image = nib.load(paths["t1"])
    assert tensor_image.shape == t1_image.shape + (6,)
    assert np.array_equal(tensor_image.affine, t1_image.affine)
    tensor = np.asarray(tensor_image.dataobj)
    expected_mask = (labels >= 1) & (labels <= 499)
    assert np.all(tensor[~expected_mask] == 0)
    assert np.all(np.isfinite(tensor))
    assert json.loads((output / "t1_registration_qa.json").read_text())["workers"] == 8


def test_t1_registration_has_no_alignment_fallback(tmp_path, monkeypatch):
    paths, _ = _fixture(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(t1_module, "_estimate", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        run_t1_registration_nifti(
            paths["tensor"],
            paths["fa"],
            paths["t1"],
            paths["labels"],
            paths["corrected"],
            tmp_path / "output",
        )


def test_t1_registration_validates_inputs(tmp_path):
    paths, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must be 6 or 12"):
        run_t1_registration_nifti(
            paths["tensor"], paths["fa"], paths["t1"], paths["labels"],
            paths["corrected"], tmp_path / "output", degrees_of_freedom=9
        )
    with pytest.raises(ValueError, match="positive integer"):
        run_t1_registration_nifti(
            paths["tensor"], paths["fa"], paths["t1"], paths["labels"],
            paths["corrected"], tmp_path / "output", workers=0
        )
    bad_t1 = tmp_path / "bad_t1.nii.gz"
    _save(bad_t1, np.ones((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="same spatial grid"):
        prepare_charm_t1_inputs(
            bad_t1, paths["labels"], paths["corrected"], tmp_path / "prepared"
        )


def test_t1_preparation_rejects_dimensions_and_nonfinite_inputs(tmp_path):
    paths, _ = _fixture(tmp_path)
    four_dimensional = tmp_path / "labels_4d.nii.gz"
    _save(four_dimensional, np.zeros((9, 8, 7, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="must be 3D"):
        prepare_charm_t1_inputs(
            paths["t1"], four_dimensional, paths["corrected"], tmp_path / "out1"
        )

    corrected = np.asarray(nib.load(paths["corrected"]).dataobj)
    corrected[0, 0, 0] = np.inf
    _save(paths["corrected"], corrected)
    with pytest.raises(ValueError, match="Bias-corrected T1"):
        prepare_charm_t1_inputs(
            paths["t1"], paths["labels"], paths["corrected"], tmp_path / "out2"
        )

    paths, _ = _fixture(tmp_path / "finite")
    t1 = np.asarray(nib.load(paths["t1"]).dataobj)
    t1[0, 0, 0] = np.nan
    _save(paths["t1"], t1)
    with pytest.raises(ValueError, match="T1 contains"):
        prepare_charm_t1_inputs(
            paths["t1"], paths["labels"], paths["corrected"], tmp_path / "out3"
        )

    with pytest.raises(ValueError, match="must be a 3D NIfTI"):
        t1_module._load_finite_3d(four_dimensional, "test input")


def test_t1_estimator_builds_fsl_coordinate_contract(tmp_path, monkeypatch):
    paths, _ = _fixture(tmp_path)
    reference_image, reference = t1_module._load_finite_3d(paths["t1"], "T1")
    moving_image, moving = t1_module._load_finite_3d(paths["fa"], "FA")
    captured = {}

    def fake_register(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FlirtRegistrationResult(np.eye(4), 0.0, 1, 1)

    monkeypatch.setattr(t1_module, "register_flirt_affine", fake_register)
    result = t1_module._estimate(
        reference_image,
        reference,
        np.ones_like(reference),
        moving_image,
        moving,
        degrees_of_freedom=12,
        workers=8,
        progress=None,
    )
    assert result.matrix.shape == (4, 4)
    assert captured["kwargs"]["degrees_of_freedom"] == 12
    assert captured["kwargs"]["workers"] == 8
    assert captured["kwargs"]["qsform_matrix"].shape == (4, 4)


def test_t1_registration_rejects_nonfinite_and_sse_grid(tmp_path, monkeypatch):
    paths, _ = _fixture(tmp_path)
    values = np.asarray(nib.load(paths["fa"]).dataobj)
    values[0, 0, 0] = np.nan
    _save(paths["fa"], values)
    with pytest.raises(ValueError, match="NaN or Inf"):
        run_t1_registration_nifti(
            paths["tensor"], paths["fa"], paths["t1"], paths["labels"],
            paths["corrected"], tmp_path / "output"
        )

    paths, _ = _fixture(tmp_path / "second")
    monkeypatch.setattr(
        t1_module,
        "_estimate",
        lambda *args, **kwargs: FlirtRegistrationResult(np.eye(4), 0.0, 1, 1),
    )
    _save(paths["sse"], np.ones((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="SSE must match"):
        run_t1_registration_nifti(
            paths["tensor"], paths["fa"], paths["t1"], paths["labels"],
            paths["corrected"], tmp_path / "second-output", sse_file=paths["sse"]
        )
