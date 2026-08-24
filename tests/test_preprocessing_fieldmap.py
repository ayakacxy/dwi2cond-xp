from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import dwi2cond_xp.preprocessing.fieldmap as fieldmap_module
from dwi2cond_xp.preprocessing.fieldmap import (
    FieldmapResult,
    displacement_from_voxel_shift,
    extrapolate_field_holes,
    fill_head_mask,
    forward_warp_magnitude,
    phase_encoding_axis_sign,
    prepare_radians_per_second,
    regularize_voxel_shift,
    run_fieldmap,
    run_fieldmap_nifti,
    voxel_shift_from_field,
)
from dwi2cond_xp.preprocessing.flirt_registration import FlirtRegistrationResult


def test_direction_shift_and_world_displacement_contract() -> None:
    field = np.full((4, 5, 6), np.float32(2.0 * np.pi * 10.0))
    affine = np.array(
        [[-2.0, 0.2, 0.0, 3.0], [0.0, 3.0, 0.1, -4.0], [0.0, 0.0, 4.0, 2.0], [0, 0, 0, 1]]
    )
    for direction, axis, sign in (
        ("x", 0, 1),
        ("x-", 0, -1),
        ("y", 1, 1),
        ("y-", 1, -1),
        ("z", 2, 1),
        ("z-", 2, -1),
    ):
        assert phase_encoding_axis_sign(direction) == (axis, sign)
        shift = voxel_shift_from_field(field, 0.001, direction)
        assert np.array_equal(shift, np.full(field.shape, field.shape[axis] * 0.01, np.float32))
        displacement = displacement_from_voxel_shift(shift, affine, direction)
        assert np.allclose(
            displacement,
            shift[..., None] * affine[:3, axis] * sign,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("value", ["i", "j", "k", "Y", "bad"])
def test_invalid_direction_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="must be x"):
        phase_encoding_axis_sign(value)


def test_numeric_validation_errors() -> None:
    valid = np.ones((3, 3, 3), dtype=np.float32)
    invalid = valid.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="field must be finite"):
        prepare_radians_per_second(invalid, valid, median_filter=False)
    with pytest.raises(ValueError, match="finite 3D"):
        voxel_shift_from_field(invalid, 0.001, "y")
    with pytest.raises(ValueError, match="positive"):
        voxel_shift_from_field(valid, 0.0, "y")
    with pytest.raises(ValueError, match="finite 3D"):
        displacement_from_voxel_shift(invalid, np.eye(4), "y")
    with pytest.raises(ValueError, match="finite 4x4"):
        displacement_from_voxel_shift(valid, np.eye(3), "y")
    with pytest.raises(ValueError, match="nonempty"):
        fill_head_mask(np.zeros((3, 3, 3)))


def test_fill_extrapolate_and_regularize_holes() -> None:
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[1:6, 1:6, 1:6] = 1
    mask[3, 3, 3] = 0
    filled = fill_head_mask(mask)
    assert filled[3, 3, 3] == 1
    field = np.indices(mask.shape, dtype=np.float32)[1]
    extrapolated = extrapolate_field_holes(field, mask, filled)
    assert np.isfinite(extrapolated[3, 3, 3])
    unchanged = extrapolate_field_holes(field, filled, filled)
    assert np.array_equal(unchanged, field)
    defaulted = extrapolate_field_holes(
        np.full((2, 2, 2), 3.0, np.float32),
        np.zeros((2, 2, 2), np.uint8),
        np.ones((2, 2, 2), np.uint8),
    )
    assert np.all(defaulted > 0)
    regularized, regularized_mask = regularize_voxel_shift(field, mask, "y-")
    assert regularized_mask[3, 3, 3] == 1
    assert np.all(np.isfinite(regularized))
    with pytest.raises(ValueError, match="share"):
        extrapolate_field_holes(field, mask[:-1], filled)
    with pytest.raises(ValueError, match="share"):
        regularize_voxel_shift(field, mask[:-1], "y")


def test_fill_enclosed_component_and_rigid_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = np.ones((11, 11, 11), dtype=np.uint8)
    shell[3:8, 3:8, 3:8] = 0
    assert np.all(fill_head_mask(shell)[3:8, 3:8, 3:8] == 1)

    custom_mask = np.zeros((2, 5, 2), dtype=np.uint8)
    custom_mask[0, 2, :] = 1
    custom_mask[1, [0, 2, 3, 4], :] = 1
    shift = np.broadcast_to(
        np.arange(5, dtype=np.float32)[None, :, None], custom_mask.shape
    ).copy()
    monkeypatch.setattr(fieldmap_module, "fill_head_mask", lambda _mask: custom_mask)
    monkeypatch.setattr(
        fieldmap_module,
        "extrapolate_field_holes",
        lambda values, _original, _filled: values.copy(),
    )
    extended, _ = regularize_voxel_shift(shift, custom_mask, "y-")
    assert np.all(np.isfinite(extended))
    assert np.array_equal(extended[0, :, 0], np.full(5, 2.0, np.float32))

    single_mask = np.zeros((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(fieldmap_module, "fill_head_mask", lambda _mask: single_mask)
    single, _ = regularize_voxel_shift(
        np.ones((1, 1, 1), dtype=np.float32), single_mask, "y"
    )
    assert single[0, 0, 0] == 0.0


def test_prepare_fieldmap_filter_and_offset() -> None:
    field = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    field[2, 2, 2] = 10000.0
    mask = np.ones(field.shape, dtype=np.uint8)
    prepared, offset = prepare_radians_per_second(field, mask, median_filter=True)
    assert offset == 63.0
    assert prepared[2, 2, 2] == 0.0
    assert prepared.dtype == np.float32
    with pytest.raises(ValueError, match="matching"):
        prepare_radians_per_second(field, mask[:2], median_filter=False)


def test_forward_warp_sign_and_constant_field() -> None:
    source = np.zeros((3, 7, 2), dtype=np.float32)
    source[:, 3, :] = 10.0
    shift = np.ones(source.shape, dtype=np.float32)
    positive = forward_warp_magnitude(source, shift, "y", workers=2)
    negative = forward_warp_magnitude(source, shift, "y-", workers=2)
    assert np.array_equal(positive[:, 4, :], np.full((3, 2), 10.0, np.float32))
    assert np.array_equal(negative[:, 2, :], np.full((3, 2), 10.0, np.float32))
    assert np.count_nonzero(positive) == 6
    assert np.count_nonzero(negative) == 6
    with pytest.raises(ValueError, match="matching"):
        forward_warp_magnitude(source, shift[:-1], "y")
    with pytest.raises(ValueError, match="positive"):
        forward_warp_magnitude(source, shift, "y", workers=0)


def test_run_fieldmap_identity_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (8, 7, 6)
    grid = np.indices(shape, dtype=np.float32)
    magnitude = np.exp(-np.sum((grid - np.array(shape)[:, None, None, None] / 2) ** 2, axis=0) / 8).astype(np.float32)
    mask = np.ones(shape, dtype=np.uint8)
    field = (grid[0] - 3.0).astype(np.float32)
    affine = np.diag([-2.0, 2.5, 3.0, 1.0])

    def identity_registration(*_args, **_kwargs) -> FlirtRegistrationResult:
        return FlirtRegistrationResult(np.eye(4), -1.0, 10, 1)

    monkeypatch.setattr(fieldmap_module, "register_flirt_affine", identity_registration)
    progress = []
    result = run_fieldmap(
        magnitude,
        field,
        mask,
        magnitude,
        mask,
        affine,
        affine,
        dwell_seconds=0.0005,
        phase_encoding_direction="y-",
        workers=2,
        median_filter=False,
        progress=lambda phase, done, total: progress.append((phase, done, total)),
    )
    assert result.voxel_shift.shape == shape
    assert result.displacement_world_mm.shape == shape + (3,)
    assert np.array_equal(result.displacement_world_mm[..., 0], np.zeros(shape))
    assert np.array_equal(result.displacement_world_mm[..., 2], np.zeros(shape))
    assert np.all(result.corrected_b0 >= 0)
    assert progress == [
        ("fieldmap_prepare", 1, 4),
        ("fieldmap_register", 2, 4),
        ("fieldmap_warp", 4, 4),
    ]
    with pytest.raises(ValueError, match="magnitude"):
        run_fieldmap(
            magnitude[:-1], field, mask, magnitude, mask, affine, affine,
            dwell_seconds=0.0005, phase_encoding_direction="y", workers=1,
        )
    with pytest.raises(ValueError, match="b0_brain"):
        run_fieldmap(
            magnitude, field, mask, magnitude[:-1], mask, affine, affine,
            dwell_seconds=0.0005, phase_encoding_direction="y", workers=1,
        )


def test_nifti_runner_writes_explicit_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = (4, 5, 3)
    affine = np.diag([-2.0, 2.0, 2.5, 1.0])
    reference = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    inputs = {}
    for name in ("magnitude", "field", "b0", "mask"):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(reference, path)
        inputs[name] = path
    empty = np.zeros(shape, dtype=np.float32)
    result = FieldmapResult(
        empty,
        empty,
        np.ones(shape, dtype=np.uint8),
        empty,
        np.eye(4),
        empty,
        np.ones(shape, dtype=np.uint8),
        empty,
        np.zeros(shape + (3,), dtype=np.float64),
        empty,
        np.ones(shape, dtype=np.uint8),
        FlirtRegistrationResult(np.eye(4), 0.0, 1, 1),
    )
    monkeypatch.setattr(fieldmap_module, "run_fieldmap", lambda *_args, **_kwargs: result)
    report = run_fieldmap_nifti(
        inputs["magnitude"],
        inputs["field"],
        inputs["b0"],
        tmp_path / "out",
        dwell_milliseconds=0.5,
        phase_encoding_direction="y-",
        magnitude_mask_file=inputs["mask"],
        b0_mask_file=inputs["mask"],
        workers=2,
    )
    assert report["fieldmap_input_units"] == "radians_per_second"
    assert report["dwell_seconds"] == 0.0005
    assert report["voxel_shift_units"] == "voxels"
    assert report["displacement_units"] == "nifti_world_millimeters"
    assert nib.load(report["outputs"]["displacement_world_mm"]).shape == shape + (3,)


def test_nifti_loading_and_runner_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    affine = np.eye(4)
    volume = np.ones((4, 4, 4), dtype=np.float32)

    def save(name: str, values: np.ndarray, matrix: np.ndarray = affine) -> Path:
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(values, matrix), path)
        return path

    magnitude = save("magnitude", volume)
    field = save("field", np.stack((volume, volume), axis=3))
    b0 = save("b0", volume)
    mask = save("mask", volume)
    loaded_image, loaded = fieldmap_module._load_3d(field, "field")
    assert loaded_image.shape == (4, 4, 4, 2)
    assert loaded.shape == volume.shape
    invalid_dimension = save("two_dimensional", np.ones((4, 4), np.float32))
    with pytest.raises(ValueError, match="must be 3D"):
        fieldmap_module._load_3d(invalid_dimension, "bad")
    nonfinite = volume.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fieldmap_module._load_3d(save("nonfinite", nonfinite), "bad")
    with pytest.raises(ValueError, match="positive"):
        run_fieldmap_nifti(magnitude, field, b0, tmp_path / "bad", dwell_milliseconds=0, phase_encoding_direction="y")
    mismatched_affine = save("field_bad_affine", volume, np.diag([2, 1, 1, 1]))
    with pytest.raises(ValueError, match="share a grid"):
        run_fieldmap_nifti(magnitude, mismatched_affine, b0, tmp_path / "bad", dwell_milliseconds=0.5, phase_encoding_direction="y")
    bad_mask = save("bad_mask", np.ones((3, 4, 4), np.float32))
    with pytest.raises(ValueError, match="magnitude mask"):
        run_fieldmap_nifti(magnitude, field, b0, tmp_path / "bad", dwell_milliseconds=0.5, phase_encoding_direction="y", magnitude_mask_file=bad_mask)
    with pytest.raises(ValueError, match="b0 mask"):
        run_fieldmap_nifti(magnitude, field, b0, tmp_path / "bad", dwell_milliseconds=0.5, phase_encoding_direction="y", magnitude_mask_file=mask, b0_mask_file=bad_mask)

    fake_bet = type("FakeBet", (), {"mask": np.ones(volume.shape, np.uint8)})()
    monkeypatch.setattr(fieldmap_module, "bet_brain_mask", lambda *_args, **_kwargs: fake_bet)
    monkeypatch.setattr(
        fieldmap_module,
        "run_fieldmap",
        lambda *_args, **_kwargs: FieldmapResult(
            volume, volume, volume.astype(np.uint8), volume, np.eye(4), volume,
            volume.astype(np.uint8), volume, np.zeros(volume.shape + (3,)),
            volume, volume.astype(np.uint8),
            FlirtRegistrationResult(np.eye(4), 0.0, 1, 1),
        ),
    )
    report = run_fieldmap_nifti(
        magnitude, field, b0, tmp_path / "bet-output",
        dwell_milliseconds=0.5, phase_encoding_direction="y",
    )
    assert report["status"] == "complete"
