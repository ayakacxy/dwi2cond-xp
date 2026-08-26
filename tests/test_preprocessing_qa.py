"""Verify complete and default paths for the P11 aggregate scientific QA."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing.qa import (
    PipelineQaInputs,
    audit_fem_manifest,
    build_pipeline_qa,
)
from dwi2cond_xp.preprocessing import qa
from dwi2cond_xp.cli import main


def _save(path: Path, values: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)


def _core_inputs(tmp_path: Path) -> PipelineQaInputs:
    shape = (4, 5, 6)
    mask = np.ones(shape, dtype=np.uint8)
    tensor = np.zeros(shape + (6,), dtype=np.float32)
    tensor[..., 0] = 2.0
    tensor[..., 3] = 1.0
    tensor[..., 5] = 0.5
    _save(tmp_path / "mask.nii.gz", mask)
    _save(tmp_path / "valid.nii.gz", mask)
    _save(tmp_path / "fa.nii.gz", np.full(shape, 0.6, dtype=np.float32))
    _save(tmp_path / "tensor.nii.gz", tensor)
    (tmp_path / "bvals").write_text("0 1000 1000\n", encoding="utf-8")
    (tmp_path / "bvecs").write_text(
        "0 1 0\n0 0 1\n0 0 0\n", encoding="utf-8"
    )
    return PipelineQaInputs(
        bvals=tmp_path / "bvals",
        original_bvecs=tmp_path / "bvecs",
        brain_mask=tmp_path / "mask.nii.gz",
        fa=tmp_path / "fa.nii.gz",
        tensor=tmp_path / "tensor.nii.gz",
        valid_mask=tmp_path / "valid.nii.gz",
    )


def test_build_pipeline_qa_covers_all_formal_sections(tmp_path: Path) -> None:
    core = _core_inputs(tmp_path)
    shape = (4, 5, 6)
    dwi = np.stack(
        (
            np.full(shape, 10.0, dtype=np.float32),
            np.full(shape, 5.0, dtype=np.float32),
            np.full(shape, 7.0, dtype=np.float32),
        ),
        axis=3,
    )
    _save(tmp_path / "raw.nii.gz", dwi)
    _save(tmp_path / "corrected.nii.gz", dwi + 1.0)
    _save(tmp_path / "sse.nii.gz", np.full(shape, 0.25, dtype=np.float32))
    _save(tmp_path / "field.nii.gz", np.full(shape, 2.0, dtype=np.float32))
    _save(tmp_path / "jac.nii.gz", np.ones(shape, dtype=np.float32))
    _save(tmp_path / "t1.nii.gz", np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    _save(tmp_path / "registered_fa.nii.gz", np.full(shape, 0.6, dtype=np.float32))
    _save(tmp_path / "raw_registered_fa.nii.gz", np.full(shape, 0.5, dtype=np.float32))
    _save(tmp_path / "raw_registered_sse.nii.gz", np.full(shape, 0.75, dtype=np.float32))
    v1 = np.zeros(shape + (3,), dtype=np.float32)
    v1[..., 0] = 1.0
    _save(tmp_path / "v1.nii.gz", v1)
    (tmp_path / "rotated").write_text(
        "0 0.9998477 -0.0174524\n0 0.0174524 0.9998477\n0 0 0\n",
        encoding="utf-8",
    )
    np.savetxt(tmp_path / "parameters.txt", np.zeros((3, 16)))
    np.savetxt(tmp_path / "outliers.txt", np.asarray([[0, 0], [0, 1], [0, 0]]), fmt="%d")
    fem_manifests: dict[str, Path] = {}
    _save(tmp_path / "final_tissues.nii.gz", np.ones(shape, dtype=np.uint8))
    for mode in ("scalar", "vn", "dir", "mc"):
        path = tmp_path / f"{mode}.json"
        mesh = tmp_path / f"{mode}.msh"
        mesh.write_text("mesh\n", encoding="utf-8")
        volume = tmp_path / f"{mode}_E.nii.gz"
        _save(volume, np.ones(shape + (3,), dtype=np.float32))
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "required_simnibs_version": "4.6.0",
                    "input": {
                        "mode": mode,
                        "final_tissues": str(tmp_path / "final_tissues.nii.gz"),
                    },
                    "outputs": [str(mesh)],
                    "masked_subject_volumes": [str(volume)],
                    "volume_tissues": [1],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        fem_manifests[mode] = path

    inputs = replace(
        core,
        raw_dwi=tmp_path / "raw.nii.gz",
        corrected_dwi=tmp_path / "corrected.nii.gz",
        rotated_bvecs=tmp_path / "rotated",
        sse=tmp_path / "sse.nii.gz",
        t1=tmp_path / "t1.nii.gz",
        registered_fa=tmp_path / "registered_fa.nii.gz",
        raw_registered_fa=tmp_path / "raw_registered_fa.nii.gz",
        raw_registered_sse=tmp_path / "raw_registered_sse.nii.gz",
        v1=tmp_path / "v1.nii.gz",
        field_hz=tmp_path / "field.nii.gz",
        jacobian=tmp_path / "jac.nii.gz",
        eddy_parameters=tmp_path / "parameters.txt",
        outlier_map=tmp_path / "outliers.txt",
        readout_seconds=0.05,
        fem_manifests=fem_manifests,
    )
    report = build_pipeline_qa(inputs, tmp_path / "qa")

    assert report["status"] == "completed"
    assert report["brain_mask"]["voxels"] == np.prod(shape)
    assert report["dwi"]["raw"]["mean_dwi_stats"]["mean"] == 6.0
    assert report["dwi"]["corrected"]["mean_dwi_stats"]["mean"] == 7.0
    assert report["motion_eddy"]["rows"] == 3
    assert report["motion_eddy"]["outlier_slices"] == 1
    assert report["bvec_rotation"]["max_angle_degrees"] > 0.9
    assert report["susceptibility"]["voxel_shift"]["mean"] == pytest.approx(0.1)
    assert report["susceptibility"]["nonpositive_jacobian_voxels"] == 0
    assert report["tensor"]["valid_voxels"] == np.prod(shape)
    assert report["tensor"]["eigenvalue_min"] == 0.5
    assert report["raw_fit"]["status"] == "available"
    assert report["raw_fit"]["fa"]["mean"] == 0.5
    assert report["raw_fit"]["sse"]["mean"] == 0.75
    assert all(report["fem_smoke"][mode]["completed"] for mode in fem_manifests)
    assert all(
        report["fem_smoke"][mode]["subject_volumes"][0][
            "max_abs_outside_tissues"
        ]
        == 0.0
        for mode in fem_manifests
    )
    assert (tmp_path / "qa" / "raw_b0_mean.nii.gz").is_file()
    assert (tmp_path / "qa" / "corrected_mean_dwi.nii.gz").is_file()
    assert (tmp_path / "qa" / "dti_fa_t1_overlay.png").is_file()
    stored = json.loads((tmp_path / "qa" / "pipeline_qa.json").read_text())
    assert stored == report


def test_build_pipeline_qa_marks_mode_specific_inputs_as_unavailable(
    tmp_path: Path,
) -> None:
    report = build_pipeline_qa(_core_inputs(tmp_path), tmp_path / "qa")

    assert report["dwi"]["raw"]["status"] == "not_provided"
    assert report["motion_eddy"]["status"] == "not_provided"
    assert report["bvec_rotation"]["status"] == "not_provided"
    assert report["susceptibility"]["status"] == "not_provided"
    assert report["registration_overlay"]["status"] == "not_provided"
    assert report["sse"]["status"] == "not_provided"
    assert report["raw_fit"]["status"] == "not_provided"
    assert report["fem_smoke"]["scalar"]["status"] == "not_provided"


def test_pipeline_qa_default_b0_selection_is_exact_zero(tmp_path: Path) -> None:
    core = _core_inputs(tmp_path)
    core.bvals.write_text("0 20 1000\n", encoding="utf-8")
    shape = (4, 5, 6)
    dwi = np.stack(
        [
            np.full(shape, 10.0, dtype=np.float32),
            np.full(shape, 20.0, dtype=np.float32),
            np.full(shape, 100.0, dtype=np.float32),
        ],
        axis=3,
    )
    _save(tmp_path / "raw.nii.gz", dwi)

    report = build_pipeline_qa(
        replace(core, raw_dwi=tmp_path / "raw.nii.gz"), tmp_path / "qa"
    )

    assert report["dwi"]["raw"]["mean_dwi_stats"]["mean"] == 60.0
    np.testing.assert_array_equal(
        np.asarray(nib.load(tmp_path / "qa/raw_b0_mean.nii.gz").dataobj),
        10.0,
    )


def test_pipeline_qa_uses_separate_raw_and_corrected_mask_lineages(tmp_path: Path) -> None:
    core = _core_inputs(tmp_path)
    shape = (4, 5, 6)
    raw = np.zeros(shape + (3,), dtype=np.float32)
    corrected = np.zeros_like(raw)
    raw[..., 0] = corrected[..., 0] = 10.0
    raw[0, 0, 0, 1:] = 2.0
    raw[1, 0, 0, 1:] = 200.0
    corrected[0, 0, 0, 1:] = 3.0
    corrected[1, 0, 0, 1:] = 300.0
    _save(tmp_path / "raw.nii.gz", raw)
    _save(tmp_path / "corrected.nii.gz", corrected)
    raw_mask = np.zeros(shape, dtype=np.uint8)
    corrected_mask = np.zeros(shape, dtype=np.uint8)
    raw_mask[0, 0, 0] = 1
    corrected_mask[1, 0, 0] = 1
    _save(tmp_path / "raw_mask.nii.gz", raw_mask)
    _save(tmp_path / "corrected_mask.nii.gz", corrected_mask)

    report = build_pipeline_qa(
        replace(
            core,
            raw_dwi=tmp_path / "raw.nii.gz",
            corrected_dwi=tmp_path / "corrected.nii.gz",
            raw_dwi_brain_mask=tmp_path / "raw_mask.nii.gz",
            corrected_dwi_brain_mask=tmp_path / "corrected_mask.nii.gz",
        ),
        tmp_path / "qa",
    )

    assert report["dwi"]["raw"]["mean_dwi_stats"]["mean"] == 2.0
    assert report["dwi"]["corrected"]["mean_dwi_stats"]["mean"] == 300.0

    shifted_mask = tmp_path / "shifted_raw_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(raw_mask, np.diag([2.0, 1.0, 1.0, 1.0])), shifted_mask
    )
    with pytest.raises(ValueError, match="raw DWI and raw DWI mask"):
        build_pipeline_qa(
            replace(
                core,
                raw_dwi=tmp_path / "raw.nii.gz",
                raw_dwi_brain_mask=shifted_mask,
            ),
            tmp_path / "qa_shifted",
        )

    shifted_corrected_mask = tmp_path / "shifted_corrected_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(
            corrected_mask, np.diag([1.0, 2.0, 1.0, 1.0])
        ),
        shifted_corrected_mask,
    )
    with pytest.raises(ValueError, match="corrected DWI and corrected DWI mask"):
        build_pipeline_qa(
            replace(
                core,
                corrected_dwi=tmp_path / "corrected.nii.gz",
                corrected_dwi_brain_mask=shifted_corrected_mask,
            ),
            tmp_path / "qa_shifted_corrected",
        )


def test_pipeline_qa_rejects_core_affine_mismatch(tmp_path: Path) -> None:
    core = _core_inputs(tmp_path)
    fa = np.full((4, 5, 6), 0.6, dtype=np.float32)
    shifted_fa = tmp_path / "shifted_fa.nii.gz"
    nib.save(
        nib.Nifti1Image(fa, np.diag([2.0, 1.0, 1.0, 1.0])), shifted_fa
    )

    with pytest.raises(ValueError, match="mask, FA, tensor and valid mask"):
        build_pipeline_qa(
            replace(core, fa=shifted_fa), tmp_path / "qa_shifted_core"
        )


def test_raw_fit_qa_requires_a_paired_common_grid(tmp_path: Path) -> None:
    core = _core_inputs(tmp_path)
    _save(tmp_path / "raw-fa.nii.gz", np.ones((4, 5, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="must be provided together"):
        build_pipeline_qa(
            replace(core, raw_registered_fa=tmp_path / "raw-fa.nii.gz"),
            tmp_path / "qa-unpaired",
        )

    _save(tmp_path / "raw-sse.nii.gz", np.ones((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="must share one grid"):
        build_pipeline_qa(
            replace(
                core,
                raw_registered_fa=tmp_path / "raw-fa.nii.gz",
                raw_registered_sse=tmp_path / "raw-sse.nii.gz",
            ),
            tmp_path / "qa-grid",
        )


def test_pipeline_qa_cli_writes_report_without_loading_unrelated_routes(
    tmp_path: Path,
) -> None:
    inputs = _core_inputs(tmp_path)
    output = tmp_path / "cli-qa"

    status = main(
        [
            "pipeline-qa",
            str(inputs.bvals),
            str(inputs.original_bvecs),
            str(inputs.brain_mask),
            str(inputs.fa),
            str(inputs.tensor),
            str(inputs.valid_mask),
            str(output),
            "--progress",
            "off",
        ]
    )

    assert status == 0
    assert json.loads((output / "pipeline_qa.json").read_text())["status"] == "completed"


def test_fem_audit_rejects_nonzero_values_outside_selected_tissues(
    tmp_path: Path,
) -> None:
    tissues = np.zeros((2, 2, 2), dtype=np.uint8)
    tissues[0, 0, 0] = 1
    _save(tmp_path / "tissues.nii.gz", tissues)
    values = np.zeros((2, 2, 2, 3), dtype=np.float32)
    values[1, 1, 1, 0] = 1.0
    _save(tmp_path / "E.nii.gz", values)
    (tmp_path / "result.msh").write_text("mesh\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "required_simnibs_version": "4.6.0",
                "input": {
                    "mode": "vn",
                    "final_tissues": str(tmp_path / "tissues.nii.gz"),
                },
                "outputs": [str(tmp_path / "result.msh")],
                "masked_subject_volumes": [str(tmp_path / "E.nii.gz")],
                "volume_tissues": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nonzero outside selected tissues"):
        audit_fem_manifest(manifest, "vn")


def test_qa_low_level_readers_reject_invalid_arrays_and_cover_correlation(
    tmp_path: Path,
) -> None:
    """Cover low-level defensive branches outside the aggregate entry point's contract."""

    _save(tmp_path / "two_d.nii.gz", np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="must be 3D"):
        qa._load_finite(tmp_path / "two_d.nii.gz", ndim=3)
    invalid = np.ones((2, 2, 2), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    _save(tmp_path / "invalid.nii.gz", invalid)
    with pytest.raises(ValueError, match="NaN or Inf"):
        qa._load_finite(tmp_path / "invalid.nii.gz")
    with pytest.raises(ValueError, match="selects no values"):
        qa._masked_stats(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))

    (tmp_path / "bad-bvecs").write_text("1 0\n0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"shape \(3, N\)"):
        qa._load_bvecs(tmp_path / "bad-bvecs", 2)
    _save(tmp_path / "short.nii.gz", np.ones((2, 2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="volume count"):
        qa._mean_artifacts(
            tmp_path / "short.nii.gz",
            np.asarray([0]),
            np.asarray([2]),
            tmp_path,
            "short",
        )
    with pytest.raises(ValueError, match="share one grid"):
        qa._write_overlay(
            np.ones((2, 2, 2)),
            np.ones((2, 2, 3)),
            np.ones((2, 2, 2), dtype=bool),
            tmp_path / "bad.png",
        )
    varying = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    overlay = qa._write_overlay(
        varying,
        varying / 8.0,
        np.ones((2, 2, 2), dtype=bool),
        tmp_path / "correlated.png",
    )
    assert overlay["masked_pearson_correlation"] == pytest.approx(1.0)


def _fem_manifest_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tissues = np.ones((2, 2, 2, 1), dtype=np.uint8)
    _save(tmp_path / "audit-tissues.nii.gz", tissues)
    _save(tmp_path / "audit-E.nii.gz", np.ones((2, 2, 2, 3), dtype=np.float32))
    (tmp_path / "audit.msh").write_text("mesh\n", encoding="utf-8")
    payload: dict[str, object] = {
        "status": "completed",
        "required_simnibs_version": "4.6.0",
        "input": {"mode": "vn", "final_tissues": str(tmp_path / "audit-tissues.nii.gz")},
        "outputs": [str(tmp_path / "audit.msh")],
        "masked_subject_volumes": [str(tmp_path / "audit-E.nii.gz")],
        "volume_tissues": [1],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_fem_audit_covers_planned_singleton_and_all_contract_failures(
    tmp_path: Path,
) -> None:
    path, base = _fem_manifest_fixture(tmp_path)
    assert audit_fem_manifest(path, "vn")["completed"] is True
    with pytest.raises(ValueError, match="expected FEM mode"):
        audit_fem_manifest(path, "bad")

    def run(payload: dict[str, object], message: str) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            audit_fem_manifest(path, "vn")

    planned = {**base, "status": "planned"}
    path.write_text(json.dumps(planned), encoding="utf-8")
    assert audit_fem_manifest(path, "vn")["completed"] is False
    run({**base, "required_simnibs_version": "4.5.0"}, "SimNIBS 4.6.0")
    run({**base, "input": {"mode": "mc"}}, "mode does not match")
    run({**base, "outputs": []}, "mesh outputs")
    run({**base, "masked_subject_volumes": []}, "no subject-volume")
    run({**base, "volume_tissues": []}, "no volume tissue labels")

    _save(tmp_path / "bad-tissues.nii.gz", np.ones((2, 2, 2, 2), dtype=np.uint8))
    bad_input = {"mode": "vn", "final_tissues": str(tmp_path / "bad-tissues.nii.gz")}
    run({**base, "input": bad_input}, "final_tissues must be 3D")
    _save(tmp_path / "wrong-grid.nii.gz", np.ones((3, 2, 2, 3), dtype=np.float32))
    run(
        {**base, "masked_subject_volumes": [str(tmp_path / "wrong-grid.nii.gz")]},
        "does not match final_tissues",
    )


def test_pipeline_qa_rejects_all_invalid_scientific_contracts(tmp_path: Path) -> None:
    """Every exception must fail explicitly instead of producing apparently complete QA."""

    core = _core_inputs(tmp_path)

    def reject(inputs: PipelineQaInputs, message: str, index: int) -> None:
        with pytest.raises(ValueError, match=message):
            build_pipeline_qa(inputs, tmp_path / f"reject-{index}")

    (tmp_path / "bad-bvals").write_text("0 -1\n", encoding="utf-8")
    reject(replace(core, bvals=tmp_path / "bad-bvals"), "nonnegative", 1)
    (tmp_path / "only-b0").write_text("0 0 0\n", encoding="utf-8")
    reject(replace(core, bvals=tmp_path / "only-b0"), "one b0", 2)

    shape = (4, 5, 6)
    _save(tmp_path / "empty-mask.nii.gz", np.zeros(shape, dtype=np.uint8))
    reject(replace(core, brain_mask=tmp_path / "empty-mask.nii.gz"), "at least one voxel", 3)
    reject(replace(core, dwi_brain_mask=tmp_path / "empty-mask.nii.gz"), "DWI brain mask", 4)

    _save(tmp_path / "bad-tensor.nii.gz", np.ones(shape + (5,), dtype=np.float32))
    reject(replace(core, tensor=tmp_path / "bad-tensor.nii.gz"), "six components", 5)
    _save(tmp_path / "bad-fa.nii.gz", np.ones((3, 5, 6), dtype=np.float32))
    reject(replace(core, fa=tmp_path / "bad-fa.nii.gz"), "share one grid", 6)
    _save(tmp_path / "zero-valid.nii.gz", np.zeros(shape, dtype=np.uint8))
    reject(replace(core, valid_mask=tmp_path / "zero-valid.nii.gz"), "selects no voxels", 7)

    (tmp_path / "zero-rotated").write_text("0 0 0\n0 0 1\n0 0 0\n", encoding="utf-8")
    reject(replace(core, rotated_bvecs=tmp_path / "zero-rotated"), "must be nonzero", 8)
    _save(tmp_path / "bad-v1.nii.gz", np.ones(shape + (2,), dtype=np.float32))
    reject(replace(core, v1=tmp_path / "bad-v1.nii.gz"), "final axis 3", 9)
    _save(tmp_path / "bad-field.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
    reject(replace(core, field_hz=tmp_path / "bad-field.nii.gz"), "field must share", 10)
    _save(tmp_path / "field-ok.nii.gz", np.ones(shape, dtype=np.float32))
    reject(replace(core, field_hz=tmp_path / "field-ok.nii.gz", readout_seconds=0.0), "readout_seconds", 11)
    _save(tmp_path / "bad-jac.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
    reject(replace(core, jacobian=tmp_path / "bad-jac.nii.gz"), "Jacobian", 12)
    np.savetxt(tmp_path / "bad-parameters", np.zeros((2, 16)))
    reject(replace(core, eddy_parameters=tmp_path / "bad-parameters"), "one finite row", 13)
    _save(tmp_path / "bad-sse.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
    reject(replace(core, sse=tmp_path / "bad-sse.nii.gz"), "SSE", 14)
