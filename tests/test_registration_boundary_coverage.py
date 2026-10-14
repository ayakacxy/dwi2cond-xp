"""核对配准的坐标、流式读取、预载竞态和失败边界。"""

from dataclasses import replace
import gzip
import os
import subprocess

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing import nonlinear, orientation, qa, resampling
from dwi2cond_xp.preprocessing import flirt_registration as flirt
from dwi2cond_xp.preprocessing import t1_registration as t1


def _tensor_inputs(shape=(3, 4, 3)):
    values = np.broadcast_to([3., 0., 0., 2., 0., 1.], shape + (6,)).copy()
    affine = np.diag([-1.2, .8, 1.6, 1.])
    return values, affine, shape, affine, np.zeros(shape + (3,))


def test_full_pull_coordinates_use_independent_header_scales():
    shape, source_shape = (3, 4, 3), (5, 4, 3)
    warp = np.broadcast_to([.3, -.2, .1], shape + (3,)).copy()
    actual = nonlinear._fsl_pull_coordinates(
        shape, np.diag([-1., 2., 2., 1.]), source_shape,
        np.diag([2., 3., 4., 1.]), warp,
        source_voxel_sizes=(2.25, 3.25, 4.25),
        reference_voxel_sizes=(1.5, 2.5, 3.5),
    )
    grid = np.indices(shape, dtype=np.float64).reshape(3, -1)
    expected = np.stack(((9. - 1.5 * grid[0] - .3) / 2.25,
                         (2.5 * grid[1] - .2) / 3.25,
                         (3.5 * grid[2] + .1) / 4.25))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("options,message", [
    ({"singular_tolerance": 0}, "singular_tolerance"),
    ({"singular_tolerance": np.nan}, "singular_tolerance"),
    ({"workers": 0}, "workers"),
    ({"workers": 1.5}, "workers"),
])
def test_ppd_rejects_invalid_numerical_options(options, message):
    with pytest.raises(ValueError, match=message):
        nonlinear.resample_tensor_ppd_fsl(*_tensor_inputs(), **options)


def test_ppd_rejects_thin_reference_grid():
    with pytest.raises(ValueError, match="two voxels per axis"):
        nonlinear.resample_tensor_ppd_fsl(*_tensor_inputs((1, 3, 3)))


@pytest.mark.parametrize("function,options,message", [
    (nonlinear.register_tensor_nonlinear_nifti, {"ppd_device": "bad"}, "ppd_device"),
    (nonlinear.register_tensor_nonlinear_nifti, {"jacobian_mode": "bad"}, "jacobian_mode"),
    (nonlinear.register_tensor_fnirt_nifti, {"ppd_device": "bad"}, "ppd_device"),
    (nonlinear.register_tensor_fnirt_nifti, {"jacobian_mode": "bad"}, "jacobian_mode"),
    (nonlinear.register_tensor_fnirt_nifti, {"hessian_device": "bad"}, "hessian_device"),
])
def test_nonlinear_policy_rejection_precedes_file_access(tmp_path, function, options, message):
    count = 5 if function is nonlinear.register_tensor_fnirt_nifti else 4
    arguments = [tmp_path / f"absent-{i}.nii.gz" for i in range(count)]
    with pytest.raises(ValueError, match=message):
        function(*arguments, **options)
    assert not any(tmp_path.iterdir())


def test_cuda_ppd_entry_point_preserves_all_slab_results(monkeypatch):
    from test_gpu_coverage import numpy_ppd

    _, transfers = numpy_ppd.__wrapped__(monkeypatch)
    monkeypatch.setattr(nonlinear, "_PPD_BATCH_VOXELS", 16)
    arguments = _tensor_inputs()
    expected = nonlinear.resample_tensor_ppd_fsl(*arguments)
    actual = nonlinear.resample_tensor_ppd_fsl(*arguments, ppd_device="cuda")
    for field in expected.__dataclass_fields__:
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field),
                                   rtol=1e-12, atol=1e-12)
    assert len(transfers.downloads) == 3


@pytest.mark.parametrize("vectors,message", [
    (np.zeros((3, 2)), "shape"), (np.array([[np.nan, 0., 0.]]), "NaN"),
])
def test_reorient_gradients_rejects_invalid_arrays(vectors, message):
    with pytest.raises(ValueError, match=message):
        orientation.reorient_bvecs_fsl(vectors, np.eye(4))


def test_canonical_stream_rejects_short_payload_atomically(tmp_path):
    source_path, output = tmp_path / "source.nii.gz", tmp_path / "result.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 3), np.float32), np.eye(4)), source_path)
    source = nib.load(source_path)
    raw = gzip.decompress(source_path.read_bytes())
    source_path.write_bytes(gzip.compress(raw[:-4]))
    output.write_bytes(b"previous-result")
    with pytest.raises(EOFError, match="Incomplete gzip"):
        orientation._stream_canonical_float32(
            source, source_path, output, np.array([[0, 1], [1, 1], [2, 1]]))
    assert output.read_bytes() == b"previous-result"
    assert not list(tmp_path.glob("*.part"))


def test_in_place_reorientation_preserves_world_samples(tmp_path):
    path = tmp_path / "source.nii"
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    old_affine = np.diag([2., 3., 4., 1.])
    nib.save(nib.Nifti1Image(values, old_affine), path)
    orientation.write_fsl_reoriented(path, path, float32=True)
    actual = nib.load(path)
    new_grid = np.indices(actual.shape).reshape(3, -1)
    mapping = np.linalg.inv(old_affine) @ actual.affine
    old_grid = np.rint(mapping[:3, :3] @ new_grid + mapping[:3, 3, None]).astype(int)
    np.testing.assert_array_equal(np.asarray(actual.dataobj).reshape(-1),
                                  values[tuple(old_grid)])


@pytest.mark.parametrize("case", ["dimensions", "short", "padding"])
def test_dwi_mean_stream_validates_shape_and_consumes_tail(tmp_path, case):
    path = tmp_path / "dwi.nii.gz"
    values = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    nib.save(nib.Nifti1Image(values[..., 0] if case == "dimensions" else values,
                           np.eye(4)), path)
    if case != "dimensions":
        raw = gzip.decompress(path.read_bytes())
        path.write_bytes(gzip.compress(raw[:-4] if case == "short" else raw + b"padding"))
    if case in ("dimensions", "short"):
        with pytest.raises(ValueError if case == "dimensions" else EOFError,
                           match="4D" if case == "dimensions" else "Incomplete"):
            qa._mean_artifacts(path, np.array([0]), np.array([1, 2]), tmp_path, "means")
    else:
        _, b0, diffusion, _ = qa._mean_artifacts(
            path, np.array([0]), np.array([1, 2]), tmp_path, "means")
        np.testing.assert_array_equal(b0, values[..., 0])
        np.testing.assert_array_equal(diffusion, values[..., 1:].mean(axis=3))


def test_qa_preloaded_input_still_checks_dimensions(tmp_path):
    from test_preprocessing_qa import _core_inputs

    inputs = _core_inputs(tmp_path)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6, 1), np.uint8), np.eye(4)), inputs.brain_mask)
    with pytest.raises(ValueError, match="must be 3D"):
        qa.build_pipeline_qa(inputs, tmp_path / "qa")


def _directory_alias(path, target):
    # Windows junction 不需要 symlink 特权，两个平台都使用真实文件系统别名。
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(target)],
                       check=True, capture_output=True)
    else:
        path.symlink_to(target, target_is_directory=True)


def test_qa_reloads_alias_changed_after_preload(tmp_path, monkeypatch):
    from test_preprocessing_qa import _core_inputs

    inputs = _core_inputs(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        mask = np.ones((4, 5, 6), np.uint8)
        if directory == second:
            mask[0, 0, 0] = 0
        nib.save(nib.Nifti1Image(mask, np.eye(4)), directory / "mask.nii.gz")
    alias = tmp_path / "alias"
    _directory_alias(alias, first)
    original = qa._load_finite
    loaded = []

    def load(path, **options):
        loaded.append(path)
        item = original(path, **options)
        if path == first / "mask.nii.gz":
            alias.rmdir() if os.name == "nt" else alias.unlink()
            _directory_alias(alias, second)
        return item

    monkeypatch.setattr(qa, "_load_finite", load)
    report = qa.build_pipeline_qa(replace(inputs, brain_mask=alias / "mask.nii.gz"),
                                  tmp_path / "qa")
    assert second / "mask.nii.gz" in loaded
    assert report["brain_mask"]["voxels"] == report["tensor"]["valid_voxels"] == 119


def test_multichannel_sinc_keeps_channels_independent():
    shape = (4, 4, 4)
    values = np.arange(64, dtype=np.float32).reshape(shape)
    source = np.stack((values, values * 2 + 3), axis=-1)
    actual = resampling.resample_image(source, np.eye(4), shape, np.eye(4),
                                       np.eye(4), interpolation="sinc")
    expected = np.stack([resampling.resample_image(source[..., i], np.eye(4), shape,
                        np.eye(4), np.eye(4), interpolation="sinc") for i in range(2)], axis=-1)
    np.testing.assert_array_equal(actual, expected)


def test_blocked_registration_selects_fine_evaluators(monkeypatch):
    # 缩短已另行验证的搜索，只检查真实金字塔与评估器的策略接线。
    candidate = flirt._Candidate(1., np.eye(4))
    monkeypatch.setattr(flirt, "_search", lambda *args, **kwargs: ([candidate], [candidate], 1))

    def optimize(evaluator, initial, matrices, *args, **kwargs):
        return [flirt._Candidate(evaluator(initial @ matrix), matrix) for matrix in matrices], len(matrices)

    monkeypatch.setattr(flirt, "_parallel_optimize", optimize)
    selected = []
    original = flirt._level_evaluator

    def evaluator(level, size, *args, **kwargs):
        item = original(level, size, *args, **kwargs)
        if kwargs.get("blocked"):
            selected.append((size, item))
        return item

    monkeypatch.setattr(flirt, "_level_evaluator", evaluator)
    grid = np.indices((8, 8, 8), dtype=np.float32)
    volume = np.exp(-((grid - 3.5) ** 2).sum(axis=0) / 8).astype(np.float32)
    weight = np.ones_like(volume)
    result = flirt.register_flirt_affine(volume, volume, weight, weight, np.eye(4),
        np.eye(4), workers=1, nosearch=True, _blocked_correlation_ratio=True)
    assert [size for size, _ in selected] == [2., 1.]
    assert all(isinstance(item, flirt.FlirtBlockedCorrelationRatio) for _, item in selected)
    assert np.all(np.isfinite(result.matrix)) and np.isfinite(result.cost)


def test_raw_t1_qa_rejects_mismatched_fa_sse_grids(tmp_path, monkeypatch):
    from test_preprocessing_t1_registration import _fixture

    paths, _ = _fixture(tmp_path)
    raw_sse = tmp_path / "wrong-sse.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 7), np.float32), np.eye(4)), raw_sse)
    monkeypatch.setattr(t1, "_estimate", lambda *args, **kwargs:
                        flirt.FlirtRegistrationResult(np.eye(4), .1, 1, 1))
    with pytest.raises(ValueError, match="SSE must match the DTI FA grid"):
        t1.run_t1_registration_nifti(paths["tensor"], paths["fa"], paths["t1"],
            paths["labels"], paths["corrected"], tmp_path / "output",
            raw_fa_file=paths["fa"], raw_sse_file=raw_sse,
            workers=1, degrees_of_freedom=6, register_tensor_output=False)


def test_ordered_blocked_evaluator_agrees_with_serial_reference():
    from dwi2cond_xp.preprocessing.flirt_cost import FlirtWeightedCorrelationRatio

    grid = np.indices((7, 8, 9), dtype=np.float32)
    reference = np.exp(-((grid - 3.) ** 2).sum(axis=0) / 8).astype(np.float32)
    moving = np.roll(reference, 1, axis=0)
    weight = np.ones_like(reference)
    arguments = (reference, moving, weight, weight, np.eye(4), np.eye(4))
    blocked = flirt.FlirtBlockedCorrelationRatio(*arguments)
    serial = FlirtWeightedCorrelationRatio(*arguments)
    actual = blocked.evaluate_ordered_parallel(np.eye(4))
    assert actual == pytest.approx(serial(np.eye(4)), rel=1e-6, abs=1e-9)
