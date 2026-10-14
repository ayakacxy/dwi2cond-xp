"""以标准 gzip 并行写出大 NIfTI，保留原始未压缩字节与小文件行为。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import gzip
import io
import operator
import os
from pathlib import Path
import tempfile
from threading import BoundedSemaphore

import nibabel as nib


_PARALLEL_MIN_BYTES = 64 * 1024 * 1024
_GZIP_BLOCK_BYTES = 8 * 1024 * 1024
_MAX_COMPRESSION_WORKERS = 8


class _GzipSink(io.RawIOBase):
    """把 NiBabel 序列化字节直接分块压缩，省去未压缩临时文件。"""

    def __init__(self, writer: NiftiWriter, output) -> None:
        self.writer = writer
        self.output = output
        self.buffer = bytearray()
        self.pending: deque[Future[bytes]] = deque()
        self.position = 0

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        target = offset if whence == 0 else self.position + offset
        if whence not in (0, 1) or target < self.position:
            raise io.UnsupportedOperation("NIfTI compression stream cannot seek backwards")
        while target > self.position:
            self.write(bytes(min(target - self.position, _GZIP_BLOCK_BYTES)))
        return self.position

    def write(self, data) -> int:
        view = memoryview(data).cast("B")
        size = len(view)
        while view:
            take = min(len(view), _GZIP_BLOCK_BYTES - len(self.buffer))
            self.buffer.extend(view[:take])
            view = view[take:]
            if len(self.buffer) == _GZIP_BLOCK_BYTES:
                self._submit()
        self.position += size
        return size

    def _write_next(self) -> None:
        future = self.pending.popleft()
        try:
            self.output.write(future.result())
        finally:
            self.writer._slots.release()

    def _submit(self) -> None:
        # 共用阶段预算；当前文件先消费自己的块，避免多个文件互相等待。
        while not self.writer._slots.acquire(blocking=False):
            if self.pending:
                self._write_next()
            else:
                self.writer._slots.acquire()
                break
        try:
            future = self.writer._executor.submit(
                gzip.compress, bytes(self.buffer), 1, mtime=0,
            )
        except BaseException:
            self.writer._slots.release()
            raise
        self.buffer.clear()
        self.pending.append(future)

    def finish(self) -> None:
        if self.buffer:
            self._submit()
        while self.pending:
            self._write_next()

    def discard(self) -> None:
        # 失败时等自身任务完成并归还名额，不能让随后文件永久阻塞。
        while self.pending:
            future = self.pending.popleft()
            try:
                future.result()
            except BaseException:
                pass
            finally:
                self.writer._slots.release()


class NiftiWriter:
    """一个阶段共用压缩线程及在途块上限，避免逐文件放大线程数。"""

    def __init__(self, workers: int = 1) -> None:
        count = operator.index(workers)
        if count < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = min(count, _MAX_COMPRESSION_WORKERS)
        self._executor = ThreadPoolExecutor(max_workers=self.workers)
        self._slots = BoundedSemaphore(self.workers)

    def __enter__(self) -> NiftiWriter:
        return self

    def __exit__(self, *exception: object) -> None:
        self._executor.shutdown(wait=True)

    def save(self, image: nib.spatialimages.SpatialImage, filename: str | Path) -> None:
        """完整压缩成功后原子替换；临时文件仅创建于目标所在目录。"""

        destination = Path(filename)
        elements = 1
        for size in image.shape:
            elements *= size
        storage_bytes = elements * image.get_data_dtype().itemsize
        if (
            self.workers == 1
            or not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image))
            or not destination.name.lower().endswith(".nii.gz")
            or storage_bytes < _PARALLEL_MIN_BYTES
        ):
            # 原 nib.save 路径也占一个阶段名额，避免小文件与大文件压缩叠加。
            with self._slots:
                nib.save(image, str(destination))
            return

        staged: Path | None = None
        sink = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".gz-part", dir=destination.parent
            )
            os.close(descriptor)
            staged = Path(name)
            with staged.open("wb") as output:
                sink = _GzipSink(self, output)
                # 原 NiBabel writer 负责 header、scaling、字节序和扩展。
                original_map = image.file_map
                consumed = {
                    key: image.header[key].copy()
                    for key in ("scl_slope", "scl_inter", "vox_offset")
                }
                try:
                    image.to_file_map({
                        "image": nib.FileHolder(fileobj=sink),
                    })
                    sink.finish()
                except BaseException:
                    # 压缩可能在序列化中途失败；恢复尚未来得及恢复的 header 消费字段。
                    for key, value in consumed.items():
                        image.header[key] = value
                    raise
                finally:
                    image.file_map = original_map
            staged.replace(destination)
        finally:
            if sink is not None:
                sink.discard()
            if staged is not None:
                staged.unlink(missing_ok=True)


def save_nifti(
    image: nib.spatialimages.SpatialImage,
    filename: str | Path,
    *,
    writer: NiftiWriter | None = None,
    workers: int = 1,
) -> None:
    """使用阶段 writer；独立调用仍明确管理线程与临时文件生命周期。"""

    if writer is not None:
        writer.save(image, filename)
    else:
        with NiftiWriter(workers) as temporary_writer:
            temporary_writer.save(image, filename)
