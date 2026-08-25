from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import dwi2cond_xp.preprocessing.legacy as legacy
from dwi2cond_xp.preprocessing.flirt_registration import FlirtRegistrationResult


def _volume(value: float = 1.0) -> np.ndarray:
    grid = np.indices((7, 7, 7), dtype=np.float32)
    return np.asarray(value * np.exp(-np.sum((grid - 3.0) ** 2, axis=0) / 5.0), dtype=np.float32)


def _write_inputs(tmp_path: Path, bvals: np.ndarray | None = None) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    values = np.stack([_volume(1.0), _volume(0.9), _volume(0.8), _volume(0.7)], axis=3)
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    dwi = tmp_path / "dwi.nii"
    nib.save(image, dwi)
    bvals_file = tmp_path / "bvals"
    np.savetxt(bvals_file, [(np.array([0, 0, 1000, 1000]) if bvals is None else bvals)])
    bvecs = tmp_path / "bvecs"
    np.savetxt(bvecs, np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float))
    return dwi, bvals_file, bvecs


def test_float32_mean_and_validation() -> None:
    volumes = [_volume(1), _volume(2)]
    assert np.array_equal(
        legacy._float32_mean(volumes, np.array([0, 1])),
        np.asarray((volumes[0] + volumes[1]) / np.float32(2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="At least one"):
        legacy._float32_mean(volumes, np.array([], dtype=int))

    position, matrix, cost, count = legacy._optimize_stage_payload(
        (0, volumes[0], volumes[0], 4.0, np.eye(4), 0.8, np.zeros(3), 6)
    )
    assert position == 0
    assert matrix.shape == (4, 4)
    assert np.isfinite(cost)
    assert count > 0


def test_register_series_contract_and_progress(monkeypatch) -> None:
    values = [_volume(1), _volume(0.9), _volume(0.8)]
    progress = []
    process_state = {"created": False, "shutdown": False}

    class InlineProcessPool:
        def __init__(self, *, max_workers):
            process_state["created"] = max_workers == 2

        def map(self, function, payloads):
            return map(function, payloads)

        def shutdown(self):
            process_state["shutdown"] = True

    monkeypatch.setattr(legacy, "_isotropic_resample", lambda value, _sizes, _spacing: value)
    monkeypatch.setattr(
        legacy,
        "_intensity_center_scaled_mm",
        lambda _value, _spacing: np.zeros(3),
    )

    def optimize(_ref, _mov, _spacing, initial, _tol, _center, dof):
        result = initial.copy()
        result[0, 3] += dof / 100.0
        return result, float(dof), 3

    monkeypatch.setattr(legacy, "_optimize_one_stage", optimize)
    monkeypatch.setattr(legacy, "ProcessPoolExecutor", InlineProcessPool)
    monkeypatch.setattr(legacy.sys, "platform", "linux")
    matrices, evaluations, costs = legacy._register_mcflirt_series(
        values,
        values[0],
        np.eye(4),
        degrees_of_freedom=6,
        workers=2,
        stages_mm=(8.0, 4.0),
        progress=lambda done, total: progress.append((done, total)),
    )
    assert len(matrices) == 3
    assert evaluations == [6, 6, 6]
    assert costs == [6.0, 6.0, 6.0]
    assert progress[-1] == (6, 6)
    assert process_state == {"created": True, "shutdown": True}
    monkeypatch.setattr(legacy.sys, "platform", "darwin")
    threaded, _, _ = legacy._register_mcflirt_series(
        values,
        values[0],
        np.eye(4),
        degrees_of_freedom=6,
        workers=2,
        stages_mm=(8.0, 4.0),
    )
    assert len(threaded) == 3
    for kwargs, message in [
        ({"degrees_of_freedom": 7, "workers": 1}, "6 or 12"),
        ({"degrees_of_freedom": 6, "workers": 0}, "positive integer"),
        ({"degrees_of_freedom": 6, "workers": 1, "stages_mm": ()}, "positive spacings"),
        ({"degrees_of_freedom": 6, "workers": 1, "max_evaluations": 0}, "positive"),
    ]:
        with pytest.raises(ValueError, match=message):
            legacy._register_mcflirt_series(values, values[0], np.eye(4), **kwargs)
    with pytest.raises(ValueError, match="at least one"):
        legacy._register_mcflirt_series([], values[0], np.eye(4), degrees_of_freedom=6, workers=1)
    with pytest.raises(ValueError, match="share one grid"):
        legacy._register_mcflirt_series([values[0][:-1]], values[0], np.eye(4), degrees_of_freedom=6, workers=1)
    bad = values[0].copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="only finite"):
        legacy._register_mcflirt_series([bad], values[0], np.eye(4), degrees_of_freedom=6, workers=1)
    with pytest.raises(RuntimeError, match="evaluation limit"):
        legacy._register_mcflirt_series(
            values[:1], values[0], np.eye(4), degrees_of_freedom=6, workers=1, stages_mm=(8,), max_evaluations=1
        )


def test_resample_series_rotate_bvecs_and_save(tmp_path, monkeypatch) -> None:
    values = [_volume(1), _volume(2)]
    monkeypatch.setattr(legacy, "set_available_numba_threads", lambda workers: workers)
    monkeypatch.setattr(legacy, "fsl_matrix_to_world", lambda *_args: np.eye(4))
    monkeypatch.setattr(
        legacy,
        "resample_image",
        lambda volume, *_args, **kwargs: volume + (1 if kwargs["interpolation"] == "sinc" else 2),
    )
    progress = []
    sinc = legacy._resample_series(
        values,
        np.eye(4),
        [np.eye(4), np.eye(4)],
        interpolation="sinc",
        workers=2,
        progress=lambda done, total: progress.append((done, total)),
    )
    linear = legacy._resample_series(
        values, np.eye(4), [np.eye(4), np.eye(4)], interpolation="linear", workers=2
    )
    serial = legacy._resample_series(
        values[:1], np.eye(4), [np.eye(4)], interpolation="linear", workers=1
    )
    assert np.array_equal(sinc[0], values[0] + 1)
    assert np.array_equal(linear[1], values[1] + 2)
    assert np.array_equal(serial[0], values[0] + 2)
    assert progress == [(1, 2), (2, 2)]

    angle = np.pi / 2
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotated = legacy._rotate_bvecs(
        np.array([0.0, 1000.0]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        [np.eye(4), matrix],
    )
    assert np.allclose(rotated[1], [0, 1, 0], atol=1e-12)
    with pytest.raises(RuntimeError, match="Cannot rotate"):
        legacy._rotate_bvecs(
            np.array([1000.0]), np.zeros((1, 3)), [np.eye(4)]
        )

    template = nib.Nifti1Image(values[0], np.eye(4))
    output = tmp_path / "saved.nii.gz"
    legacy._save_nifti(values[0], template, output)
    assert np.array_equal(np.asarray(nib.load(output).dataobj), values[0])

    displacement_file = tmp_path / "displacement.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(values[0].shape + (3,)), np.eye(4)), displacement_file)
    assert legacy._load_displacement(displacement_file, template).shape == values[0].shape + (3,)
    nib.save(nib.Nifti1Image(np.zeros(values[0].shape), np.eye(4)), displacement_file)
    with pytest.raises(ValueError, match="end in xyz"):
        legacy._load_displacement(displacement_file, template)
    nonfinite = np.zeros(values[0].shape + (3,))
    nonfinite[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(nonfinite, np.eye(4)), displacement_file)
    with pytest.raises(ValueError, match="finite values"):
        legacy._load_displacement(displacement_file, template)
    nib.save(nib.Nifti1Image(np.zeros(values[0].shape + (3,)), np.diag([2, 1, 1, 1])), displacement_file)
    with pytest.raises(ValueError, match="affine"):
        legacy._load_displacement(displacement_file, template)


@pytest.mark.parametrize("bvec_mode", ["compat46", "corrected"])
def test_run_legacy_pipeline_contract(tmp_path, monkeypatch, bvec_mode) -> None:
    dwi, bvals, bvecs = _write_inputs(tmp_path)
    output = tmp_path / f"out-{bvec_mode}"

    def aligned(_data, _bvals, target, **_kwargs):
        image = nib.load(dwi)
        legacy._save_nifti(np.asarray(image.dataobj[..., 0]), image, Path(target))
        return Path(target)

    def mask(nodif, target, **_kwargs):
        image = nib.load(nodif)
        legacy._save_nifti(np.ones(image.shape, dtype=np.float32), image, Path(target))
        return object()

    monkeypatch.setattr(legacy, "write_aligned_b0_mean", aligned)
    monkeypatch.setattr(legacy, "write_bet_brain_mask", mask)
    monkeypatch.setattr(
        legacy,
        "_register_mcflirt_series",
        lambda volumes, *_args, **_kwargs: (
            [np.eye(4) for _ in volumes],
            [1 for _ in volumes],
            [0.0 for _ in volumes],
        ),
    )
    monkeypatch.setattr(
        legacy,
        "_resample_series",
        lambda volumes, *_args, **_kwargs: [np.asarray(value, dtype=np.float32) for value in volumes],
    )
    monkeypatch.setattr(
        legacy,
        "register_flirt_nosearch_mutual_information",
        lambda *_args, **_kwargs: FlirtRegistrationResult(np.eye(4), 0.0, 2, 1),
    )

    def fit(_data, _bvals, _bvecs, _mask, base, **kwargs):
        image = nib.load(dwi)
        base = Path(base)
        legacy._save_nifti(np.zeros(image.shape[:3] + (6,), dtype=np.float32), image, base)
        fa = base.with_name("DTI_FA.nii.gz")
        sse = base.with_name("DTI_sse.nii.gz")
        legacy._save_nifti(np.zeros(image.shape[:3], dtype=np.float32), image, fa)
        legacy._save_nifti(np.zeros(image.shape[:3], dtype=np.float32), image, sse)
        legacy._save_nifti(np.ones(image.shape[:3], dtype=np.float32), image, Path(kwargs["valid_mask_file"]))
        Path(kwargs["qa_file"]).write_text(
            json.dumps({"derived_outputs": {"FA": str(fa), "sse": str(sse)}}), encoding="utf-8"
        )

    monkeypatch.setattr(legacy, "fit_dti_nifti", fit)
    raw_fieldmap_kwargs = {}
    if bvec_mode == "compat46":
        magnitude = tmp_path / "magnitude.nii.gz"
        radians = tmp_path / "radians.nii.gz"
        template = nib.load(dwi)
        legacy._save_nifti(_volume(1.0), template, magnitude)
        legacy._save_nifti(_volume(10.0), template, radians)

        def fake_fieldmap(_magnitude, _radians, _b0, target, **_kwargs):
            target = Path(target)
            target.mkdir(parents=True, exist_ok=True)
            legacy._save_nifti(
                np.zeros((7, 7, 7, 3), dtype=np.float32),
                template,
                target / "displacement_world_mm.nii.gz",
            )
            legacy._save_nifti(
                np.ones((7, 7, 7), dtype=np.float32),
                template,
                target / "corrected_mask.nii.gz",
            )
            return {"status": "completed"}

        monkeypatch.setattr(legacy, "run_fieldmap_nifti", fake_fieldmap)
        raw_fieldmap_kwargs = {
            "fieldmap_magnitude_file": magnitude,
            "fieldmap_radians_per_second_file": radians,
            "fieldmap_dwell_milliseconds": 0.5,
            "fieldmap_phase_encoding_direction": "y-",
        }
    if bvec_mode == "corrected":
        monkeypatch.setattr(legacy, "_rotate_bvecs", lambda _a, vectors, _m: vectors)
    displacement = None
    corrected_mask = None
    grad_dev = None
    if bvec_mode == "corrected":
        displacement = tmp_path / "field.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((7, 7, 7, 3)), np.diag([-2, 2, 2, 1])), displacement)
        corrected_mask = tmp_path / "corrected_mask.nii.gz"
        nib.save(
            nib.Nifti1Image(np.ones((7, 7, 7)), np.diag([-2, 2, 2, 1])),
            corrected_mask,
        )
        grad_dev = dwi
    report = legacy.run_legacy_nifti(
        dwi,
        bvals,
        bvecs,
        output,
        bvec_mode=bvec_mode,
        fieldmap_displacement_file=displacement,
        fieldmap_corrected_mask_file=corrected_mask,
        grad_dev_file=grad_dev,
        workers=2,
        progress=lambda *_args: None,
        **raw_fieldmap_kwargs,
    )
    assert report["status"] == "completed"
    assert report["interpolation"]["formal_output_passes_per_volume"] == 1
    assert len(list((output / "DWI_corr.mat").glob("MAT_*"))) == 4
    if bvec_mode == "compat46":
        assert (output / "DWIbvecs").read_bytes() == bvecs.read_bytes()
        assert report["interpolation"]["fieldmap_input"] == "raw-radians-per-second"


def test_run_legacy_input_and_field_validation(tmp_path, monkeypatch) -> None:
    dwi, bvals, bvecs = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="bvec_mode"):
        legacy.run_legacy_nifti(dwi, bvals, bvecs, tmp_path / "bad", bvec_mode="bad")
    with pytest.raises(ValueError, match="positive integer"):
        legacy.run_legacy_nifti(dwi, bvals, bvecs, tmp_path / "bad-workers", workers=0)
    with pytest.raises(ValueError, match="raw fieldmap requires"):
        legacy.run_legacy_nifti(
            dwi,
            bvals,
            bvecs,
            tmp_path / "incomplete-raw-fieldmap",
            fieldmap_magnitude_file=dwi,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        legacy.run_legacy_nifti(
            dwi,
            bvals,
            bvecs,
            tmp_path / "mixed-fieldmap",
            fieldmap_magnitude_file=dwi,
            fieldmap_radians_per_second_file=dwi,
            fieldmap_dwell_milliseconds=0.5,
            fieldmap_phase_encoding_direction="y",
            fieldmap_displacement_file=dwi,
            fieldmap_corrected_mask_file=dwi,
        )
    with pytest.raises(ValueError, match="displacement and corrected mask"):
        legacy.run_legacy_nifti(
            dwi,
            bvals,
            bvecs,
            tmp_path / "incomplete-prepared-fieldmap",
            fieldmap_displacement_file=dwi,
        )
    short_bvals = tmp_path / "short-bvals"
    short_bvecs = tmp_path / "short-bvecs"
    np.savetxt(short_bvals, [[0, 1000]])
    np.savetxt(short_bvecs, np.array([[0, 1], [0, 0], [0, 0]], dtype=float))
    with pytest.raises(ValueError, match="fourth axis"):
        legacy.run_legacy_nifti(dwi, short_bvals, short_bvecs, tmp_path / "short")
    bad_image = nib.load(dwi)
    bad_values = np.asarray(bad_image.dataobj).copy()
    bad_values[0, 0, 0, 0] = np.nan
    bad_dwi = tmp_path / "bad-dwi.nii"
    nib.save(nib.Nifti1Image(bad_values, bad_image.affine), bad_dwi)
    with pytest.raises(ValueError, match="NaN or Inf"):
        legacy.run_legacy_nifti(bad_dwi, bvals, bvecs, tmp_path / "nonfinite")
    for values, message, case_name in [
        (np.array([1000, 1000, 1000, 1000]), "exact b=0", "missing-b0"),
        (np.array([0, 0, 0, 0]), "b>0", "missing-dwi"),
    ]:
        _, invalid_bvals, _ = _write_inputs(tmp_path / case_name, values)
        with pytest.raises(ValueError, match=message):
            legacy.run_legacy_nifti(
                dwi, invalid_bvals, bvecs, tmp_path / f"invalid-{case_name}"
            )

    # Replace the expensive stage before reading the displacement.
    monkeypatch.setattr(legacy, "write_aligned_b0_mean", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        legacy.run_legacy_nifti(dwi, bvals, bvecs, tmp_path / "early-stop")


def test_corrected_fieldmap_mask_requires_matching_grid(tmp_path: Path) -> None:
    dwi, _, _ = _write_inputs(tmp_path)
    reference = nib.load(dwi)
    bad_shape = tmp_path / "bad-shape-mask.nii.gz"
    bad_affine = tmp_path / "bad-affine-mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 7, 7)), reference.affine), bad_shape)
    shifted = reference.affine.copy()
    shifted[0, 3] += 1.0
    nib.save(nib.Nifti1Image(np.ones((7, 7, 7)), shifted), bad_affine)

    with pytest.raises(ValueError, match="match the DWI grid"):
        legacy._replace_mask_with_corrected(
            bad_shape, reference, tmp_path / "shape-output.nii.gz"
        )
    with pytest.raises(ValueError, match="affine"):
        legacy._replace_mask_with_corrected(
            bad_affine, reference, tmp_path / "affine-output.nii.gz"
        )
