"""核对并发文件的压缩预算、完整字节和失败时的发布边界。"""

from concurrent.futures import ThreadPoolExecutor
import gzip
import io
from threading import Event, Lock
import time

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing import _nifti_io


def _image(seed):
    values = np.random.default_rng(seed).normal(size=(17, 13, 11, 3)).astype(np.float32)
    values.reshape(-1)[0] = np.nan
    header = nib.Nifti1Header()
    header.extensions.append(nib.nifti1.Nifti1Extension(6, b"codec boundary" * 400))
    image = nib.Nifti1Image(values, np.diag([-1.2, 0.8, 1.7, 1.0]), header)
    image.set_qform(image.affine, 1)
    image.set_sform(image.affine, 2)
    return image


def test_shared_writer_bounds_threads_and_preserves_complete_nifti(tmp_path, monkeypatch):
    monkeypatch.setattr(_nifti_io, "_PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr(_nifti_io, "_GZIP_BLOCK_BYTES", 1024)
    original = gzip.compress
    lock = Lock()
    active = peak = 0

    def compress(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.001)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(gzip, "compress", compress)
    images = [_image(seed) for seed in range(3)]
    baselines = []
    outputs = []
    for index, image in enumerate(images):
        baseline = tmp_path / f"baseline{index}.nii.gz"
        nib.save(image, baseline)
        baselines.append(baseline)
        outputs.append(tmp_path / f"parallel{index}.nii.gz")
    with _nifti_io.NiftiWriter(4) as writer, ThreadPoolExecutor(max_workers=3) as files:
        futures = [files.submit(writer.save, image, path) for image, path in zip(images, outputs)]
        for future in futures:
            future.result(timeout=10)
    assert 1 < peak <= 4
    for baseline, output in zip(baselines, outputs):
        assert gzip.decompress(baseline.read_bytes()) == gzip.decompress(output.read_bytes())
    assert not list(tmp_path.glob(".*"))


def test_writer_failure_preserves_old_output_and_releases_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(_nifti_io, "_PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr(_nifti_io, "_GZIP_BLOCK_BYTES", 1024)
    accepted = tmp_path / "accepted.nii.gz"
    nib.save(_image(1), accepted)
    previous = accepted.read_bytes()
    original = gzip.compress

    def fail(*args, **kwargs):
        raise RuntimeError("injected compression failure")

    with _nifti_io.NiftiWriter(2) as writer:
        monkeypatch.setattr(gzip, "compress", fail)
        attempted = _image(2)
        header_before = attempted.header.binaryblock
        file_map_before = attempted.file_map
        with pytest.raises(RuntimeError, match="injected compression failure"):
            writer.save(attempted, accepted)
        assert accepted.read_bytes() == previous
        assert attempted.header.binaryblock == header_before
        assert attempted.file_map is file_map_before
        assert not list(tmp_path.glob(".*"))
        monkeypatch.setattr(gzip, "compress", original)
        writer.save(_image(3), tmp_path / "recovered.nii.gz")
    np.testing.assert_array_equal(
        np.asarray(nib.load(tmp_path / "recovered.nii.gz").dataobj), _image(3).dataobj,
    )
    assert not list(tmp_path.glob(".*"))


def test_small_and_uncompressed_outputs_keep_nibabel_path(tmp_path, monkeypatch):
    image = _image(4)
    baseline = tmp_path / "baseline.nii.gz"
    nib.save(image, baseline)

    def unexpected(*args, **kwargs):
        raise AssertionError("Parallel codec must not run for this output")

    monkeypatch.setattr(gzip, "compress", unexpected)
    with _nifti_io.NiftiWriter(40) as writer:
        assert writer.workers == 8
        writer.save(image, tmp_path / "small.nii.gz")
        monkeypatch.setattr(_nifti_io, "_PARALLEL_MIN_BYTES", 1)
        writer.save(image, tmp_path / "uncompressed.nii")
    assert (tmp_path / "small.nii.gz").read_bytes() == baseline.read_bytes()
    assert (tmp_path / "uncompressed.nii").read_bytes() == gzip.decompress(baseline.read_bytes())


def test_compression_stream_seek_preserves_position_and_zero_padding(monkeypatch):
    """前进定位补零，非法定位不允许改动已接收的字节。"""
    monkeypatch.setattr(_nifti_io, "_GZIP_BLOCK_BYTES", 4)
    output = io.BytesIO()
    with _nifti_io.NiftiWriter(2) as writer:
        sink = _nifti_io._GzipSink(writer, output)
        assert sink.write(b"abc") == 3
        assert sink.tell() == 3
        assert sink.seek(10) == 10
        assert sink.seek(2, 1) == 12
        for offset, whence in ((0, 0), (-1, 1), (0, 2)):
            with pytest.raises(io.UnsupportedOperation, match="cannot seek backwards"):
                sink.seek(offset, whence)
            assert sink.tell() == 12
        sink.finish()
    assert gzip.decompress(output.getvalue()) == b"abc" + bytes(9)


def test_submit_failure_returns_compression_slot_and_can_recover(monkeypatch):
    """执行器拒绝提交后恢复预算，随后同一个流仍可压缩原始缓冲。"""
    output = io.BytesIO()
    with _nifti_io.NiftiWriter(2) as writer:
        sink = _nifti_io._GzipSink(writer, output)
        sink.write(b"recoverable")
        original_submit = writer._executor.submit

        def reject(*args, **kwargs):
            raise RuntimeError("executor rejected compression")

        monkeypatch.setattr(writer._executor, "submit", reject)
        with pytest.raises(RuntimeError, match="executor rejected"):
            sink.finish()
        assert writer._slots.acquire(blocking=False)
        assert writer._slots.acquire(blocking=False)
        assert not writer._slots.acquire(blocking=False)
        writer._slots.release()
        writer._slots.release()
        monkeypatch.setattr(writer._executor, "submit", original_submit)
        sink.finish()
    assert gzip.decompress(output.getvalue()) == b"recoverable"


def test_stream_without_pending_blocks_waits_for_other_file_budget():
    """另一文件占满预算时等待其写出，不追加压缩任务或丢失数据。"""
    waited = Event()
    first_output, second_output = io.BytesIO(), io.BytesIO()
    with _nifti_io.NiftiWriter(1) as writer:
        semaphore = writer._slots

        class ObservedBudget:
            def acquire(self, blocking=True):
                if blocking:
                    waited.set()
                return semaphore.acquire(blocking=blocking)

            def release(self):
                return semaphore.release()

        writer._slots = ObservedBudget()
        first = _nifti_io._GzipSink(writer, first_output)
        second = _nifti_io._GzipSink(writer, second_output)
        first.write(b"first file")
        first._submit()
        second.write(b"second file")
        with ThreadPoolExecutor(max_workers=1) as files:
            future = files.submit(second.finish)
            try:
                assert waited.wait(timeout=5)
                assert not future.done()
            finally:
                first.finish()
            future.result(timeout=5)
    assert gzip.decompress(first_output.getvalue()) == b"first file"
    assert gzip.decompress(second_output.getvalue()) == b"second file"


@pytest.mark.parametrize("workers", [0, -1])
def test_writer_rejects_nonpositive_worker_budget(workers):
    with pytest.raises(ValueError, match="positive integer"):
        _nifti_io.NiftiWriter(workers)


def test_save_nifti_delegation_and_owned_writer_produce_identical_data(tmp_path):
    """共享及独立写出入口均保留 NiBabel 的完整图像字节。"""
    image = _image(5)
    shared, owned = tmp_path / "shared.nii.gz", tmp_path / "owned.nii.gz"
    with _nifti_io.NiftiWriter(2) as writer:
        _nifti_io.save_nifti(image, shared, writer=writer)
    _nifti_io.save_nifti(image, owned, workers=1)
    assert gzip.decompress(shared.read_bytes()) == gzip.decompress(owned.read_bytes())
    np.testing.assert_array_equal(np.asarray(nib.load(owned).dataobj), image.dataobj)
