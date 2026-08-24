from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing.image_ops import (
    apply_positive_mask,
    binarize_positive,
    edge_strength,
    extract_roi,
    gaussian_smooth,
    image_dimensions,
    lower_threshold,
    masked_percentile,
    median_filter_box,
    merge_time,
    multiply,
    read_dwi_z_block,
    select_b0_indices,
    subtract,
    time_mean,
    upper_threshold,
    write_float32_copy,
    write_unaligned_b0_mean,
)


FSLMATHS = Path(os.environ.get("FSLMATHS", "/path/not/configured/fslmaths"))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    data = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    data[0, 0, 0, 2] = -3.0
    affine = np.array(
        [[-1.2, 0, 0, 1.2], [0, 1.3, 0, -1.3], [0, 0, 1.4, -1.4], [0, 0, 0, 1]]
    )
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    data_file = tmp_path / "dwi.nii.gz"
    bvals_file = tmp_path / "bvals"
    nib.save(image, data_file)
    np.savetxt(bvals_file, [[5, 1000, 40, 995]])
    return data_file, bvals_file, data


def test_select_b0_indices_uses_threshold_and_rejects_invalid_values() -> None:
    assert select_b0_indices(np.array([5, 50, 51, 1000])).tolist() == [0, 1]
    with pytest.raises(ValueError, match="nonnegative"):
        select_b0_indices(np.array([0, 1000]), threshold=-1)
    with pytest.raises(ValueError, match="NaN or Inf"):
        select_b0_indices(np.array([0, np.nan]))
    with pytest.raises(ValueError, match="No b0"):
        select_b0_indices(np.array([1000, 1005]))


def test_fsl_scalar_operations_and_mask_broadcasting() -> None:
    values = np.array([-2.0, 0.0, 1.0, 2.0], dtype=np.float64).reshape(2, 1, 2)
    assert np.array_equal(
        lower_threshold(values, 1.0),
        np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32).reshape(2, 1, 2),
    )
    assert np.array_equal(
        upper_threshold(values, 1.0),
        np.array([-2.0, 0.0, 1.0, 0.0], dtype=np.float32).reshape(2, 1, 2),
    )
    assert np.array_equal(
        binarize_positive(values),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32).reshape(2, 1, 2),
    )

    four_d = np.stack([values, values + 1.0], axis=3)
    mask = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32).reshape(2, 1, 2)
    expected_masked = four_d.astype(np.float32) * (mask > 0)[..., None]
    assert np.array_equal(apply_positive_mask(four_d, mask), expected_masked)
    assert np.array_equal(multiply(four_d, 2.0), four_d.astype(np.float32) * 2.0)
    assert np.array_equal(
        multiply(four_d, np.ones_like(four_d)), four_d.astype(np.float32)
    )
    assert np.array_equal(subtract(four_d, 1.0), four_d.astype(np.float32) - 1.0)
    with pytest.raises(ValueError, match="compatible"):
        multiply(four_d, np.ones((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="3D or 4D"):
        lower_threshold(np.zeros((2, 2)), 0.0)


def test_time_merge_roi_percentile_and_dimensions() -> None:
    first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    second = np.stack([first + 2.0, first + 4.0], axis=3)
    merged = merge_time([first, second])
    assert merged.shape == (2, 3, 4, 3)
    assert np.array_equal(merged[..., 0], first)
    assert np.array_equal(time_mean(merged), np.mean(merged, axis=3, dtype=np.float64))
    assert image_dimensions(first) == (2, 3, 4, 1)
    assert image_dimensions(merged) == (2, 3, 4, 3)
    assert np.array_equal(
        extract_roi(merged, (0, 1, 1, 1), (-1, 2, 3, 2)),
        merged[:, 1:3, 1:4, 1:3],
    )

    mask = np.zeros(first.shape, dtype=np.float32)
    mask.reshape(-1)[[2, 4, 6, 8]] = 1.0
    selected = np.sort(
        merged[np.broadcast_to(mask[..., None] >= 0.5, merged.shape) & (merged != 0.0)]
    )
    assert masked_percentile(merged, mask, 50.0) == float(
        selected[int(selected.size * 0.5)]
    )
    assert masked_percentile(merged, mask, 100.0) == float(selected[-1])

    with pytest.raises(ValueError, match="four-dimensional"):
        time_mean(first)
    with pytest.raises(ValueError, match="At least one"):
        merge_time([])
    with pytest.raises(ValueError, match="spatial shapes"):
        merge_time([first, np.zeros((3, 3, 4), dtype=np.float32)])
    with pytest.raises(ValueError, match="match the image dimensions"):
        extract_roi(first, (0, 0), (1, 1))
    for starts, sizes in (
        ((-1, 0, 0), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 1)),
        ((0, 0, 0), (-2, 1, 1)),
        ((0, 0, 3), (1, 1, 2)),
    ):
        with pytest.raises(ValueError, match="nonempty"):
            extract_roi(first, starts, sizes)
    with pytest.raises(ValueError, match="between 0 and 100"):
        masked_percentile(first, mask, 101.0)
    with pytest.raises(ValueError, match="does not select"):
        masked_percentile(first, np.zeros_like(first), 50.0)


def test_fsl_spatial_filter_contracts_and_validation() -> None:
    values = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    filtered = median_filter_box(values)
    center_neighbours = np.sort(values[1:4, 1:4, 0:3], axis=None)
    corner_neighbours = np.sort(values[0:2, 0:2, 0:2], axis=None)
    assert filtered[2, 2, 1] == center_neighbours[center_neighbours.size // 2]
    assert filtered[0, 0, 0] == corner_neighbours[corner_neighbours.size // 2]
    stacked = np.stack([values, values + 100.0], axis=3)
    assert np.array_equal(median_filter_box(stacked)[..., 0], filtered)
    singleton = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    assert median_filter_box(singleton).shape == singleton.shape

    constant = np.full(values.shape, 7.0, dtype=np.float32)
    assert np.allclose(
        gaussian_smooth(constant, 1.0, (1.2, 1.3, 1.4)),
        constant,
        rtol=0.0,
        atol=5e-7,
    )
    with pytest.raises(ValueError, match="must be positive"):
        gaussian_smooth(values, 0.0, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="must be positive"):
        gaussian_smooth(values, 1.0, (1.0, -1.0, 1.0))

    grid = np.indices((5, 5, 5), dtype=np.float32)
    ramp = grid[0] * 2.0 + grid[1] * 3.0 + grid[2] * 4.0
    edges = edge_strength(ramp, (1.0, 1.0, 1.0))
    assert edges[2, 2, 2] == pytest.approx(
        np.sqrt(4**2 + 6**2 + 8**2) / (2 * np.sqrt(3))
    )
    assert edges[0, 0, 0] == ramp[0, 0, 0]
    planar = edge_strength(ramp[:, :, :2], (1.0, 1.0, 1.0))
    assert np.all(planar[2, 2, :] > 0)
    with pytest.raises(ValueError, match="three positive"):
        edge_strength(ramp, (1.0, 1.0))
    with pytest.raises(ValueError, match="three positive"):
        edge_strength(ramp, (1.0, 0.0, 1.0))


def test_read_block_preserves_values_or_clips_negatives(tmp_path: Path) -> None:
    data_file, _, data = _write_fixture(tmp_path)
    image = nib.load(data_file)
    raw = read_dwi_z_block(image, 0, 1)
    clipped = read_dwi_z_block(image, 0, 1, nonnegative=True)
    assert raw.flags.c_contiguous
    assert np.array_equal(raw, data[:, :, :1, :])
    assert clipped[0, 0, 0, 2] == 0
    assert np.all(clipped >= 0)


def test_read_block_rejects_shape_and_bounds(tmp_path: Path) -> None:
    three_d = tmp_path / "three.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), three_d)
    with pytest.raises(ValueError, match="four-dimensional"):
        read_dwi_z_block(nib.load(three_d), 0, 1)
    data_file, _, _ = _write_fixture(tmp_path)
    image = nib.load(data_file)
    for bounds in ((-1, 1), (1, 1), (0, 4)):
        with pytest.raises(ValueError, match="bounds"):
            read_dwi_z_block(image, *bounds)


def test_write_unaligned_mean_preserves_header_and_reports_qa(tmp_path: Path) -> None:
    data_file, bvals_file, data = _write_fixture(tmp_path)
    output = tmp_path / "result.nii.gz"
    qa_file = tmp_path / "custom.json"
    progress = []
    result = write_unaligned_b0_mean(
        data_file,
        bvals_file,
        output,
        z_chunk=2,
        progress=lambda done, total: progress.append((done, total)),
        qa_file=qa_file,
    )
    source = nib.load(data_file)
    image = nib.load(result)
    expected = np.mean(data[..., [0, 2]], axis=3, dtype=np.float64).astype(np.float32)
    assert np.array_equal(np.asanyarray(image.dataobj), expected)
    assert image.get_data_dtype() == np.dtype(np.float32)
    assert np.array_equal(image.affine, source.affine)
    assert int(image.header["qform_code"]) == 1
    assert int(image.header["sform_code"]) == 2
    assert progress == [(2, 3), (3, 3)]
    qa = json.loads(qa_file.read_text(encoding="utf-8"))
    assert qa["b0_indices"] == [0, 2]
    assert qa["b0_values"] == [5.0, 40.0]
    assert qa["negative_measurements"] == 1
    assert qa["nonfinite_measurements"] == 0
    assert qa["b0_alignment"] == "none"


def test_float32_copy_applies_scaling_and_preserves_geometry(tmp_path: Path) -> None:
    source = tmp_path / "scaled.nii"
    output = tmp_path / "nested" / "copy.nii.gz"
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    image = nib.Nifti1Image(values, affine)
    image.header.set_slope_inter(2.0, 5.0)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    nib.save(image, source)

    write_float32_copy(source, output)
    result = nib.load(output)
    assert result.get_data_dtype() == np.dtype(np.float32)
    assert np.array_equal(np.asarray(result.dataobj), values * 2.0 + 5.0)
    assert np.array_equal(result.affine, affine)
    assert int(result.header["qform_code"]) == 1
    assert int(result.header["sform_code"]) == 2


def test_default_qa_name_and_input_validation(tmp_path: Path) -> None:
    data_file, bvals_file, _ = _write_fixture(tmp_path)
    output = tmp_path / "mean.nii.gz"
    write_unaligned_b0_mean(data_file, bvals_file, output)
    assert (tmp_path / "mean_qa.json").is_file()
    with pytest.raises(ValueError, match="positive"):
        write_unaligned_b0_mean(data_file, bvals_file, output, z_chunk=0)

    three_d = tmp_path / "three.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), three_d)
    with pytest.raises(ValueError, match="four-dimensional"):
        write_unaligned_b0_mean(three_d, bvals_file, output)
    bad_bvals = tmp_path / "bad_bvals"
    np.savetxt(bad_bvals, [[0, 1000]])
    with pytest.raises(ValueError, match="fourth axis"):
        write_unaligned_b0_mean(data_file, bad_bvals, output)


def test_b0_nonfinite_values_fail_before_writing(tmp_path: Path) -> None:
    data_file, bvals_file, data = _write_fixture(tmp_path)
    data[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(data, np.eye(4)), data_file)
    output = tmp_path / "mean.nii.gz"
    with pytest.raises(ValueError, match="b0 volumes"):
        write_unaligned_b0_mean(data_file, bvals_file, output)
    assert not output.exists()


@pytest.mark.skipif(
    not FSLMATHS.is_file(),
    reason="FSL image-operation reference disabled; set FSLMATHS",
)
def test_finite_image_operations_match_fsl(tmp_path: Path) -> None:
    values = np.array(
        [
            -2.0,
            -0.0,
            0.25,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
        ],
        dtype=np.float32,
    ).reshape(2, 2, 2, 2)
    mask = np.array(
        [0.0, 0.5, 0.5001, 1.0, -1.0, 2.0, 0.0, 1.0], dtype=np.float32
    ).reshape(2, 2, 2)
    affine = np.diag([-1.2, 1.3, 1.4, 1.0])
    input_file = tmp_path / "input.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(values, affine), input_file)
    nib.save(nib.Nifti1Image(mask, affine), mask_file)

    environment = os.environ.copy()
    environment["FSLDIR"] = str(FSLMATHS.parent.parent)
    environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    def run_math(name: str, arguments: list[str]) -> np.ndarray:
        output = tmp_path / name
        subprocess.run(
            [str(FSLMATHS), str(input_file), *arguments, str(output)],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        return np.asarray(nib.load(output.with_suffix(".nii.gz")).dataobj)

    comparisons = (
        (
            np.asarray(nib.load(write_float32_copy(input_file, tmp_path / "copy.nii.gz")).dataobj),
            run_math("copy_reference", []),
        ),
        (lower_threshold(values, 0.5), run_math("lower", ["-thr", "0.5"])),
        (upper_threshold(values, 4.0), run_math("upper", ["-uthr", "4"])),
        (binarize_positive(values), run_math("binary", ["-bin"])),
        (
            apply_positive_mask(values, mask),
            run_math("masked", ["-mas", str(mask_file)]),
        ),
        (multiply(values, mask), run_math("multiply", ["-mul", str(mask_file)])),
        (subtract(values, 1.5), run_math("subtract", ["-sub", "1.5"])),
        (time_mean(values), run_math("mean", ["-Tmean"])),
    )
    for ours, reference in comparisons:
        assert np.array_equal(ours, reference)

    voxel_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(affine))
    assert np.array_equal(median_filter_box(values), run_math("median", ["-fmedian"]))
    assert np.allclose(
        gaussian_smooth(values, 1.0, voxel_sizes),
        run_math("smooth", ["-s", "1"]),
        rtol=0.0,
        atol=2e-6,
    )
    assert np.allclose(
        edge_strength(values, voxel_sizes),
        run_math("edge", ["-edge"]),
        rtol=0.0,
        atol=2e-6,
    )

    fslroi = FSLMATHS.with_name("fslroi")
    roi_file = tmp_path / "roi.nii.gz"
    subprocess.run(
        [str(fslroi), str(input_file), str(roi_file), "0", "1"],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert np.array_equal(
        extract_roi(values, (0, 0, 0, 0), (-1, -1, -1, 1)),
        np.asarray(nib.load(roi_file).dataobj),
    )

    first_file = tmp_path / "first.nii.gz"
    second_file = tmp_path / "second.nii.gz"
    merged_file = tmp_path / "merged.nii.gz"
    nib.save(nib.Nifti1Image(values[..., 0], affine), first_file)
    nib.save(nib.Nifti1Image(values[..., 1], affine), second_file)
    subprocess.run(
        [
            str(FSLMATHS.with_name("fslmerge")),
            "-t",
            str(merged_file),
            str(first_file),
            str(second_file),
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert np.array_equal(
        merge_time([values[..., 0], values[..., 1]]),
        np.asarray(nib.load(merged_file).dataobj),
    )

    fslstats = FSLMATHS.with_name("fslstats")
    result = subprocess.run(
        [str(fslstats), str(input_file), "-k", str(mask_file), "-P", "50"],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert masked_percentile(values, mask, 50.0) == float(result.stdout.strip())
