from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.nifti_fit import fit_dti_nifti
import dwi2cond_xp.preprocessing.nomoco as nomoco_module
from dwi2cond_xp.tensor_fit import form_design_matrix


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Generate a small deterministic single-shell DWI sufficient to exercise WLS."""

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
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvals = np.concatenate([np.zeros(2), np.full(len(directions), 1000.0)])
    bvecs = np.vstack([np.zeros((2, 3)), directions])
    tensor = np.array([1.3e-3, 6e-5, -3e-5, 7e-4, 4e-5, 4e-4])
    design = form_design_matrix(bvals, bvecs)
    signal = 1000.0 * np.exp(-(design[:, :6] @ tensor))
    data = np.zeros((5, 4, 3, len(bvals)), dtype=np.float32)
    data[1:4, 1:3, :, :] = signal
    data[0, 0, 0, 0] = -2.0
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    dwi = tmp_path / "dwi.nii.gz"
    bvals_file = tmp_path / "bvals"
    bvecs_file = tmp_path / "bvecs"
    grad_dev = tmp_path / "grad_dev.nii.gz"
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, dwi)
    nib.save(nib.Nifti1Image(np.zeros(data.shape[:3] + (9,), dtype=np.float32), affine), grad_dev)
    np.savetxt(bvals_file, bvals[None, :])
    np.savetxt(bvecs_file, bvecs.T)
    return dwi, bvals_file, bvecs_file, grad_dev


@pytest.mark.parametrize("direct_mmap", [False, True])
def test_nomoco_pipeline_matches_existing_fit_and_emits_no_correction_artifacts(
    tmp_path: Path, monkeypatch, direct_mmap: bool
) -> None:
    dwi, bvals, bvecs, grad_dev = _write_fixture(tmp_path)
    if direct_mmap:
        image = nib.load(dwi)
        values = np.maximum(np.asarray(image.dataobj, dtype=np.float32), 0.0)
        dwi = tmp_path / "dwi.nii"
        nib.save(nib.Nifti1Image(values, image.affine, image.header), dwi)

    def fake_aligned(data_file, _bvals_file, output_file, *, progress, qa_file, **kwargs):
        del kwargs
        image = nib.load(str(data_file))
        values = np.asarray(image.dataobj, dtype=np.float32).mean(axis=3)
        nib.save(nib.Nifti1Image(values, image.affine, image.header), output_file)
        Path(qa_file).write_text('{"status":"completed"}\n', encoding="utf-8")
        progress(2, 2)
        return Path(output_file)

    def fake_bet(input_file, output_file, **kwargs):
        del kwargs
        image = nib.load(str(input_file))
        mask = (np.asarray(image.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, image.affine, image.header), output_file)
        return SimpleNamespace(mask=mask, passes=0)

    monkeypatch.setattr(nomoco_module, "write_aligned_b0_mean", fake_aligned)
    monkeypatch.setattr(nomoco_module, "write_bet_brain_mask", fake_bet)
    progress: list[tuple[str, int, int]] = []
    output = tmp_path / "nomoco"
    report = nomoco_module.run_nomoco_nifti(
        dwi,
        bvals,
        bvecs,
        output,
        grad_dev_file=grad_dev,
        workers=1,
        z_chunk=1,
        progress=lambda phase, done, total: progress.append((phase, done, total)),
    )

    direct = tmp_path / "direct.nii.gz"
    fitting_input = dwi if direct_mmap else output / "DWIforfit.nii"
    fit_dti_nifti(
        fitting_input,
        output / "DWIbvals",
        output / "DWIbvecs",
        output / "nodif_brain_mask.nii.gz",
        direct,
        grad_dev_file=output / "grad_dev.nii",
        workers=1,
        z_chunk=1,
    )
    assert np.array_equal(
        np.asarray(nib.load(output / "DTI_tensor.nii.gz").dataobj),
        np.asarray(nib.load(direct).dataobj),
    )
    for suffix in ("FA", "sse"):
        assert np.array_equal(
            np.asarray(nib.load(output / f"DTI_{suffix}.nii.gz").dataobj),
            np.asarray(nib.load(tmp_path / f"direct_{suffix}.nii.gz").dataobj),
        )
    if direct_mmap:
        assert not (output / "DWIforfit.nii").exists()
        assert report["fitting_input"]["strategy"] == "validated_input_mmap"
    else:
        assert np.min(np.asarray(nib.load(output / "DWIforfit.nii").dataobj)) == 0.0
        assert report["fitting_input"]["strategy"] == "single_decode_materialization"
    assert report["correction"] == {
        "motion": "not_applied_to_dwi",
        "eddy_current": "not_applied",
        "susceptibility": "not_applied",
    }
    assert set(report["stage_seconds"]) == {
        "normalize_input",
        "align_b0",
        "brain_mask",
        "fit_dti",
    }
    assert {item[0] for item in progress} == {
        "normalize_input",
        "align_b0",
        "brain_mask",
        "fit_dti",
    }
    names = {path.name.lower() for path in output.iterdir()}
    assert not any("motion" in name or "eddy" in name or "field" in name for name in names)
    assert json.loads((output / "nomoco_qa.json").read_text(encoding="utf-8")) == report


def test_nomoco_validates_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        nomoco_module.run_nomoco_nifti("dwi", "bvals", "bvecs", tmp_path, workers=0)


def test_fitting_input_validation_and_fallbacks(tmp_path: Path) -> None:
    three_dimensional = tmp_path / "three.nii"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)), three_dimensional)
    with pytest.raises(ValueError, match="four-dimensional"):
        nomoco_module._prepare_fitting_input(three_dimensional, tmp_path / "out.nii")

    values = np.zeros((2, 2, 2, 2), dtype=np.float32)
    nonfinite = tmp_path / "nonfinite.nii"
    values[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(values, np.eye(4)), nonfinite)
    materialized, report = nomoco_module._prepare_fitting_input(
        nonfinite, tmp_path / "nonfinite_output.nii"
    )
    assert materialized.name == "nonfinite_output.nii"
    assert report["finite"] is False
    assert report["materialized"] is True

    negative_values = np.zeros((2, 2, 2, 2), dtype=np.float32)
    negative_values[0, 0, 0, 0] = -1.0
    negative = tmp_path / "negative.nii"
    nib.save(nib.Nifti1Image(negative_values, np.eye(4)), negative)
    _, negative_report = nomoco_module._prepare_fitting_input(
        negative, tmp_path / "negative_output.nii"
    )
    assert negative_report["finite"] is True
    assert negative_report["nonnegative"] is False

    with pytest.raises(ValueError, match="z-chunk"):
        nomoco_module._prepare_fitting_input(nonfinite, tmp_path / "unused.nii", z_chunk=0)
