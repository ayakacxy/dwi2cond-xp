"""Finite NIfTI geometry and orientation operations used by dwi2cond."""

from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from ..registration import matrix_to_tensor6, tensor6_to_matrix


def _f32(value: float | np.floating) -> np.float32:
    return np.float32(value)


def _f32_norm(x: np.float32, y: np.float32, z: np.float32) -> np.float32:
    squared = _f32(_f32(x * x) + _f32(y * y))
    squared = _f32(squared + _f32(z * z))
    return _f32(np.sqrt(squared))


def _f32_dot(
    x1: np.float32,
    y1: np.float32,
    z1: np.float32,
    x2: np.float32,
    y2: np.float32,
    z2: np.float32,
) -> np.float32:
    value = _f32(_f32(x1 * x2) + _f32(y1 * y2))
    return _f32(value + _f32(z1 * z2))


def _fsl_determinant(matrix: np.ndarray) -> np.float32:
    values = np.asarray(matrix, dtype=np.float32)
    r11, r12, r13 = (float(value) for value in values[0])
    r21, r22, r23 = (float(value) for value in values[1])
    r31, r32, r33 = (float(value) for value in values[2])
    return _f32(
        r11 * r22 * r33
        - r11 * r32 * r23
        - r21 * r12 * r33
        + r21 * r32 * r13
        + r31 * r12 * r23
        - r31 * r22 * r13
    )


def _fsl_axis_orientation(affine: np.ndarray) -> np.ndarray:
    q = np.asarray(affine[:3, :3], dtype=np.float32).copy()

    norm = _f32_norm(q[0, 0], q[1, 0], q[2, 0])
    if norm == 0.0:
        raise ValueError("The orientation affine has a zero-length spatial axis")
    q[:, 0] = np.asarray([_f32(value / norm) for value in q[:, 0]])

    norm = _f32_norm(q[0, 1], q[1, 1], q[2, 1])
    if norm == 0.0:
        raise ValueError("The orientation affine has a zero-length spatial axis")
    q[:, 1] = np.asarray([_f32(value / norm) for value in q[:, 1]])

    projection = _f32_dot(*q[:, 0], *q[:, 1])
    if abs(float(projection)) > 1.0e-4:
        q[:, 1] = np.asarray(
            [_f32(q[row, 1] - _f32(projection * q[row, 0])) for row in range(3)]
        )
        norm = _f32_norm(*q[:, 1])
        if norm == 0.0:
            raise ValueError("The orientation affine has parallel spatial axes")
        q[:, 1] = np.asarray([_f32(value / norm) for value in q[:, 1]])

    norm = _f32_norm(*q[:, 2])
    if norm == 0.0:
        q[:, 2] = np.asarray(
            [
                _f32(q[1, 0] * q[2, 1] - q[2, 0] * q[1, 1]),
                _f32(q[2, 0] * q[0, 1] - q[0, 0] * q[2, 1]),
                _f32(q[0, 0] * q[1, 1] - q[1, 0] * q[0, 1]),
            ],
            dtype=np.float32,
        )
    else:
        q[:, 2] = np.asarray([_f32(value / norm) for value in q[:, 2]])

    projection = _f32_dot(*q[:, 0], *q[:, 2])
    if abs(float(projection)) > 1.0e-4:
        q[:, 2] = np.asarray(
            [_f32(q[row, 2] - _f32(projection * q[row, 0])) for row in range(3)]
        )
        norm = _f32_norm(*q[:, 2])
        if norm == 0.0:
            raise ValueError("The orientation affine has dependent spatial axes")
        q[:, 2] = np.asarray([_f32(value / norm) for value in q[:, 2]])

    projection = _f32_dot(*q[:, 1], *q[:, 2])
    if abs(float(projection)) > 1.0e-4:
        q[:, 2] = np.asarray(
            [_f32(q[row, 2] - _f32(projection * q[row, 1])) for row in range(3)]
        )
        norm = _f32_norm(*q[:, 2])
        if norm == 0.0:
            raise ValueError("The orientation affine has dependent spatial axes")
        q[:, 2] = np.asarray([_f32(value / norm) for value in q[:, 2]])

    determinant = _fsl_determinant(q)
    if determinant == 0.0:
        raise ValueError("The orientation affine has singular normalized axes")

    best_trace = _f32(-666.0)
    best = (0, 1, 2, 1, 1, 1)
    for first_axis in range(3):
        for second_axis in range(3):
            if first_axis == second_axis:
                continue
            for third_axis in range(3):
                if third_axis in (first_axis, second_axis):
                    continue
                for first_sign in (-1, 1):
                    for second_sign in (-1, 1):
                        for third_sign in (-1, 1):
                            permutation = np.zeros((3, 3), dtype=np.float32)
                            permutation[0, first_axis] = first_sign
                            permutation[1, second_axis] = second_sign
                            permutation[2, third_axis] = third_sign
                            if _fsl_determinant(permutation) * determinant <= 0.0:
                                continue
                            product = np.empty((3, 3), dtype=np.float32)
                            for row in range(3):
                                for column in range(3):
                                    value = _f32(
                                        permutation[row, 0] * q[0, column]
                                        + permutation[row, 1] * q[1, column]
                                    )
                                    product[row, column] = _f32(
                                        value + permutation[row, 2] * q[2, column]
                                    )
                            trace = _f32(_f32(product[0, 0] + product[1, 1]) + product[2, 2])
                            if trace > best_trace:
                                best_trace = trace
                                best = (
                                    first_axis,
                                    second_axis,
                                    third_axis,
                                    first_sign,
                                    second_sign,
                                    third_sign,
                                )
    return np.asarray(
        [[best[0], best[3]], [best[1], best[4]], [best[2], best[5]]],
        dtype=np.float64,
    )


def fsl_canonical_orientation(affine: np.ndarray) -> np.ndarray:
    """Return the storage transform used by FSL ``fslreorient2std``.

    FSL preserves the input storage handedness: positive-determinant images
    are arranged as RAS and negative-determinant images as LAS.
    """

    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("The orientation affine must be a finite 4x4 matrix")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if abs(determinant) < 1e-12:
        raise ValueError("The orientation affine must have nonsingular spatial axes")
    current = _fsl_axis_orientation(matrix)
    target_codes = ("R", "A", "S") if determinant > 0 else ("L", "A", "S")
    target = nib.orientations.axcodes2ornt(target_codes)
    return nib.orientations.ornt_transform(current, target)


def reorient_spatial_array(values: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Permute and flip the first three axes without interpolation."""

    source = np.asarray(values)
    transform = np.asarray(orientation, dtype=np.float64)
    if source.ndim < 3:
        raise ValueError("Spatial reorientation requires at least three dimensions")
    if transform.shape != (3, 2):
        raise ValueError("The orientation transform must have shape (3, 2)")
    return np.ascontiguousarray(nib.orientations.apply_orientation(source, transform))


def voxel_basis_transform(orientation: np.ndarray) -> np.ndarray:
    """Return the signed permutation mapping old vector components to new ones."""

    transform = np.asarray(orientation, dtype=np.float64)
    if transform.shape != (3, 2):
        raise ValueError("The orientation transform must have shape (3, 2)")
    new_voxel_to_old_voxel = nib.orientations.inv_ornt_aff(
        transform, (1, 1, 1)
    )[:3, :3]
    return new_voxel_to_old_voxel.T


def reorient_bvecs_voxel(
    bvecs: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Transform row-wise b-vectors expressed in the image voxel basis."""

    vectors = np.asarray(bvecs)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("bvecs must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("bvecs contain NaN or Inf")
    basis = voxel_basis_transform(orientation)
    return np.asarray(vectors @ basis.T, dtype=vectors.dtype)


def reorient_tensor6_voxel(
    tensor: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Transform FSL-order tensor components between voxel bases."""

    values = np.asarray(tensor)
    basis = voxel_basis_transform(orientation)
    matrices = tensor6_to_matrix(values)
    transformed = np.einsum(
        "ij,...jk,lk->...il", basis, matrices, basis, optimize=True
    )
    return np.asarray(matrix_to_tensor6(transformed), dtype=values.dtype)


def fsl_component_basis_transform(source_affine: np.ndarray) -> np.ndarray:
    """返回原 FSL 分量基到 canonical 存储 FSL 分量基的正交变换。"""

    orientation = fsl_canonical_orientation(source_affine)
    basis = voxel_basis_transform(orientation)
    old_affine = np.asarray(source_affine, dtype=np.float64)
    new_affine = old_affine @ nib.orientations.inv_ornt_aff(orientation, (1, 1, 1))
    old_handedness = np.diag(
        [-1.0 if np.linalg.det(old_affine[:3, :3]) > 0 else 1.0, 1.0, 1.0]
    )
    new_handedness = np.diag(
        [-1.0 if np.linalg.det(new_affine[:3, :3]) > 0 else 1.0, 1.0, 1.0]
    )
    return new_handedness @ basis @ old_handedness


def reorient_bvecs_fsl(bvecs: np.ndarray, source_affine: np.ndarray) -> np.ndarray:
    """把逐行 (N,3) 的 FSL bvec 转到重排后的图像 FSL 分量基。"""

    vectors = np.asarray(bvecs)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("bvecs must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("bvecs contain NaN or Inf")
    basis = fsl_component_basis_transform(source_affine)
    return np.asarray(vectors @ basis.T, dtype=vectors.dtype)


def validate_fsl_raw_sampling(image: nib.spatialimages.SpatialImage) -> None:
    """raw array-only 算法尚未区分 pixdim 与 sform 尺度时明确拒绝。"""

    if not np.allclose(
        image.header.get_zooms()[:3],
        nib.affines.voxel_sizes(image.affine),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Raw preprocessing requires header pixdim to match the active affine "
            "voxel sizes; use an explicitly resampled grid"
        )


def _stream_canonical_float32(source, input_file, output_file, orientation) -> Path:
    """规范方向、无缩放的 float32 图像直接流式解压，NiBabel 仍生成输出头。"""
    voxel_map = nib.orientations.inv_ornt_aff(orientation, source.shape[:3])
    qform = source.get_qform() @ voxel_map
    sform = source.get_sform() @ voxel_map
    qcode, scode = int(source.header["qform_code"]), int(source.header["sform_code"])
    # 零步长占位视图只提供形状和 dtype，不分配或计算全体积数据。
    placeholder = np.broadcast_to(np.float32(0), source.shape)
    image = nib.Nifti1Image(placeholder, sform if scode > 0 else qform, source.header.copy())
    image.set_qform(qform, qcode)
    image.set_sform(sform, scode)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header.set_data_offset(0)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".part", dir=output.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as target, gzip.open(input_file, "rb") as stream:
            header.write_to(target)
            target.seek(int(header["vox_offset"]))
            stream.seek(int(source.dataobj.offset))
            remaining = source.dataobj.dtype.itemsize
            for size in source.shape:
                remaining *= size
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise EOFError("Incomplete gzip NIfTI image")
                target.write(block)
                remaining -= len(block)
            # 完整图像后的读取触发标准 gzip 尾部校验；不复制额外 padding。
            stream.read(1)
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    return output


def write_fsl_reoriented(
    input_file: str | Path,
    output_file: str | Path,
    *,
    float32: bool = False,
    nonnegative: bool = False,
    component_kind: str | None = None,
    component_mode: str = "physical",
    stream_canonical: bool = False,
) -> Path:
    """无插值地重排存储轴，并按显式类型转换张量或 grad-dev 分量。

    physical 模式保持原物理方向；simnibs46-literal 仅重排体素。
    grad-dev 九分量使用 FSL/HCP 列主序。
    """

    if component_kind not in (None, "tensor", "grad-dev"):
        raise ValueError("component_kind must be None, 'tensor', or 'grad-dev'")
    if component_mode not in ("physical", "simnibs46-literal"):
        raise ValueError("component_mode must be 'physical' or 'simnibs46-literal'")

    output = Path(output_file)
    overwrite_input = output.exists() and Path(input_file).samefile(output)
    # Windows 禁止截断仍被映射的文件；同文件及硬链接读取时就避开 memmap。
    source = nib.load(str(input_file), mmap=not overwrite_input)
    qform_code = int(source.header["qform_code"])
    sform_code = int(source.header["sform_code"])
    if qform_code == 0 and sform_code == 0:
        raise ValueError("NIfTI orientation requires a valid qform or sform")
    active_affine = source.get_sform() if sform_code > 0 else source.get_qform()
    orientation = fsl_canonical_orientation(active_affine)
    if (
        stream_canonical and float32 and not nonnegative and component_kind is None
        and type(source) is nib.Nifti1Image
        and str(input_file).lower().endswith(".nii.gz")
        and str(output_file).lower().endswith(".nii")
        and source.dataobj.dtype.kind == "f" and source.dataobj.dtype.itemsize == 4
        and source.dataobj.slope == 1.0 and source.dataobj.inter == 0.0
        and source.dataobj.order == "F"
        and np.array_equal(orientation, [[0, 1], [1, 1], [2, 1]])
    ):
        return _stream_canonical_float32(source, input_file, output_file, orientation)
    source_values = np.asanyarray(source.dataobj)
    if float32:
        source_values = np.asarray(source_values, dtype=np.float32)
    # 写盘按 NIfTI 的列主序遍历；保留重排视图，避免先复制成行主序再搬回。
    values = nib.orientations.apply_orientation(source_values, orientation)
    if component_kind is not None:
        expected = 6 if component_kind == "tensor" else 9
        if values.ndim != 4 or values.shape[-1] != expected:
            raise ValueError(f"{component_kind} image must have {expected} component channels")
        if component_mode == "physical":
            basis = fsl_component_basis_transform(active_affine)
            if np.issubdtype(values.dtype, np.floating) and np.all(np.isfinite(values)):
                # 基变换是带符号的轴置换，无需展开九分量矩阵并进行稠密乘法。
                axes = np.argmax(np.abs(basis), axis=1)
                signs = basis[np.arange(3), axes]
                components = (
                    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
                    if component_kind == "tensor"
                    else tuple((row, column) for column in range(3) for row in range(3))
                )
                original_components = {pair: index for index, pair in enumerate(components)}
                indices = []
                multipliers = []
                for row, column in components:
                    pair = (int(axes[row]), int(axes[column]))
                    if component_kind == "tensor":
                        pair = tuple(sorted(pair))
                    indices.append(original_components[pair])
                    multipliers.append(signs[row] * signs[column])
                values = values[..., indices]
                values *= np.asarray(multipliers, dtype=values.dtype)
                # 稠密乘法中的零加法会将负零变成正零，保留此序列化细节。
                values[values == 0] = 0
            else:
                # 非有限或整数输入继续沿用原稠密乘法的传播与类型转换规则。
                matrices = (
                    tensor6_to_matrix(values) if component_kind == "tensor"
                    else values.reshape(values.shape[:3] + (3, 3)).swapaxes(-1, -2)
                )
                transformed = np.einsum(
                    "ij,...jk,lk->...il", basis, matrices, basis, optimize=True
                )
                values = np.asarray(
                    matrix_to_tensor6(transformed) if component_kind == "tensor"
                    else transformed.swapaxes(-1, -2).reshape(values.shape),
                    dtype=values.dtype,
                )
    if nonnegative:
        # 钳位不能修改输入 memmap；列主序副本可直接用于后续写盘。
        values = np.array(values, copy=True, order="F")
        np.maximum(values, 0.0, out=values)
    new_voxel_to_old_voxel = nib.orientations.inv_ornt_aff(
        orientation, source.shape[:3]
    )
    qform = source.get_qform() @ new_voxel_to_old_voxel
    sform = source.get_sform() @ new_voxel_to_old_voxel
    new_active_affine = (
        sform if sform_code > 0 else qform
    )

    header = source.header.copy()
    header.set_data_shape(values.shape)
    header.set_data_dtype(np.float32 if float32 else source.get_data_dtype())
    output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite_input:
        # 同文件或硬链接覆盖会先截断 memmap；写盘前必须脱离输入存储。
        values = np.array(values, copy=True, order="F")
    image = nib.Nifti1Image(values, new_active_affine, header)
    image.set_qform(qform, qform_code)
    image.set_sform(sform, sform_code)
    nib.save(image, str(output))
    return output


def copy_nifti_geometry(
    source_file: str | Path,
    destination_file: str | Path,
    output_file: str | Path | None = None,
    *,
    copy_dimensions: bool = True,
) -> Path:
    """Copy the geometry fields used by ``fslcpgeom`` onto existing data.

    The dwi2cond call site uses matching 3D grids. Dimension-changing header
    reinterpretation is rejected because it can expose or truncate raw bytes.
    """

    source = nib.load(str(source_file))
    destination = nib.load(str(destination_file))
    if copy_dimensions and source.shape != destination.shape:
        raise ValueError("Geometry dimension copying requires matching image shapes")
    values = np.asanyarray(destination.dataobj)
    header = destination.header.copy()
    source_zooms = tuple(float(value) for value in source.header["pixdim"][1 : values.ndim + 1])
    header.set_zooms(source_zooms)
    qform_code = int(source.header["qform_code"])
    sform_code = int(source.header["sform_code"])
    active_affine = source.get_sform() if sform_code > 0 else source.get_qform()
    if qform_code == 0 and sform_code == 0:
        active_affine = destination.affine
    output = Path(destination_file) if output_file is None else Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(values, active_affine, header)
    image.set_qform(source.get_qform(), qform_code)
    image.set_sform(source.get_sform(), sform_code)
    nib.save(image, str(output))
    return output
