"""验证官方 reverse-PE、TOPUP、BET、EDDY 组合闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import dwi2cond_xp.preprocessing.topup_eddy as module


def _nifti(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.eye(4)), path)


def test_topup_eddy_uses_reverse_4d_and_corrected_b0_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dwi = tmp_path / "dwi.nii.gz"
    reverse = tmp_path / "reverse.nii.gz"
    bvals = tmp_path / "bvals"
    bvecs = tmp_path / "bvecs"
    _nifti(dwi, np.ones((3, 3, 3, 3), dtype=np.float32))
    _nifti(reverse, np.ones((3, 3, 3, 2), dtype=np.float32) * 2)
    bvals.write_text("0 1000 0\n", encoding="utf-8")
    bvecs.write_text("0 1 0\n0 0 0\n0 0 0\n", encoding="utf-8")
    seen: dict[str, object] = {}
    progress: list[tuple[str, int, int]] = []

    def fake_mean(data, mean_bvals, output, **kwargs):
        image = nib.load(str(data))
        values = np.asarray(image.dataobj, dtype=np.float32)
        selected = np.loadtxt(mean_bvals, ndmin=1) == 0
        _nifti(Path(output), np.mean(values[..., selected], axis=3))
        seen.setdefault("means", []).append((Path(data), int(np.count_nonzero(selected))))
        return Path(output)

    def fake_topup(forward, reverse_mean, output, **kwargs):
        output = Path(output)
        _nifti(output / "corrected_pair.nii.gz", np.ones((3, 3, 3, 2)))
        _nifti(output / "field_hz.nii.gz", np.zeros((3, 3, 3)))
        _nifti(output / "field_coefficients.nii.gz", np.zeros((3, 3, 3)))
        (output / "movement_parameters.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
        return {"status": "completed", "forward": str(forward), "reverse": str(reverse_mean)}

    def fake_bet(source, destination, **kwargs):
        seen["bet_source"] = Path(source)
        _nifti(Path(destination), np.ones((3, 3, 3)))
        return Path(destination)

    def fake_eddy(data, _bvals, _bvecs, mask, output, **kwargs):
        seen["eddy_data"] = Path(data)
        seen["eddy_mask"] = Path(mask)
        seen["eddy_field"] = Path(kwargs["susceptibility_field_file"])
        output = Path(output)
        _nifti(output / "corrected_dwi.nii.gz", np.ones((3, 3, 3, 3)))
        return {"status": "completed", "algorithm": "eddy"}

    monkeypatch.setattr(module, "write_aligned_b0_mean", fake_mean)
    monkeypatch.setattr(module, "run_topup_nifti", fake_topup)
    monkeypatch.setattr(module, "write_bet_brain_mask", fake_bet)
    monkeypatch.setattr(module, "run_eddy_nifti", fake_eddy)

    output = tmp_path / "output"
    report = module.run_topup_eddy_nifti(
        dwi,
        bvals,
        bvecs,
        reverse,
        output,
        readout_seconds=0.05,
        phase_encoding_direction="y-",
        workers=8,
        progress=lambda phase, done, total: progress.append((phase, done, total)),
    )

    assert seen["means"][0][1] == 2
    assert seen["means"][1][1] == 2
    assert seen["bet_source"] == output / "topup_preparation/topup_corrected_b0.nii.gz"
    assert seen["eddy_mask"] == output / "topup_preparation/topup_corrected_b0_brain_mask.nii.gz"
    assert seen["eddy_field"] == output / "topup/field_hz.nii.gz"
    assert report["reverse_phase_encoding_volumes"] == 2
    assert json.loads((output / "topup_eddy_qa.json").read_text())["status"] == "completed"
    assert ("forward_b0_mean", 0, 1) in progress
    assert ("forward_b0_mean", 1, 1) in progress
    assert ("reverse_b0_mean", 0, 1) in progress
    assert ("reverse_b0_mean", 1, 1) in progress
    assert ("topup_corrected_b0_bet", 1, 1) in progress


def test_topup_eddy_rejects_three_dimensional_reverse_input(tmp_path: Path) -> None:
    dwi = tmp_path / "dwi.nii.gz"
    reverse = tmp_path / "reverse.nii.gz"
    _nifti(dwi, np.ones((2, 2, 2, 2)))
    _nifti(reverse, np.ones((2, 2, 2)))
    (tmp_path / "bvals").write_text("0 1000\n", encoding="utf-8")
    (tmp_path / "bvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="four-dimensional"):
        module.run_topup_eddy_nifti(
            dwi,
            tmp_path / "bvals",
            tmp_path / "bvecs",
            reverse,
            tmp_path / "output",
            readout_seconds=0.05,
            phase_encoding_direction="y",
        )


def test_corrected_pair_must_have_exactly_two_volumes(tmp_path: Path) -> None:
    pair = tmp_path / "pair.nii.gz"
    _nifti(pair, np.ones((2, 2, 2, 3)))
    with pytest.raises(ValueError, match="exactly two"):
        module._save_first_corrected_b0(pair, tmp_path / "first.nii.gz")


@pytest.mark.parametrize(
    ("readout", "direction", "message"),
    ((0.001, "y", "within"), (0.05, "z", "must be x")),
)
def test_topup_eddy_rejects_global_parameters_before_creating_outputs(
    tmp_path: Path, readout: float, direction: str, message: str
) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        module.run_topup_eddy_nifti(
            tmp_path / "dwi.nii.gz",
            tmp_path / "bvals",
            tmp_path / "bvecs",
            tmp_path / "reverse.nii.gz",
            output,
            readout_seconds=readout,
            phase_encoding_direction=direction,
        )
    assert not output.exists()
