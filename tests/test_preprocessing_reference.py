from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import dwi2cond_xp.preprocessing._numba as numba_helpers
from dwi2cond_xp.preprocessing.reference import (
    ReferenceArtifact,
    ReferenceRunError,
    audit_public_manifest,
    _file_summary,
    _redact,
    _resolve_executable,
    _resource_metrics,
    run_reference_command,
    summarize_fixture_inputs,
)


def _write_nifti(path: Path) -> None:
    data = np.array([[[0.0, 1.0], [np.nan, -2.0]]], dtype=np.float32)
    image = nib.Nifti1Image(data, np.diag([1.0, 2.0, 3.0, 1.0]))
    image.set_qform(image.affine, 1)
    image.set_sform(image.affine, 2)
    nib.save(image, path)


def test_fixture_summary_is_path_free_and_records_nifti_contract(
    tmp_path: Path,
) -> None:
    image = tmp_path / "private_subject.nii.gz"
    text = tmp_path / "bvals"
    _write_nifti(image)
    text.write_text("0 1000\n", encoding="utf-8")

    summary = summarize_fixture_inputs(
        {"dwi": image, "bvals": text},
        nifti_aliases=("dwi",),
        mask_aliases=("dwi",),
        include_digests=True,
    )

    encoded = json.dumps(summary)
    assert str(tmp_path) not in encoded
    assert [item["alias"] for item in summary] == ["bvals", "dwi"]
    nifti = summary[1]["nifti"]
    assert nifti["shape"] == [1, 2, 2]
    assert nifti["dtype"] == "float32"
    assert nifti["finite_count"] == 3
    assert nifti["nonzero_count"] == 2
    assert nifti["mask_count"] == 2
    assert nifti["finite_min"] == -2.0
    assert nifti["finite_max"] == 1.0
    assert nifti["qform_code"] == 1
    assert nifti["sform_code"] == 2
    assert len(summary[0]["sha256"]) == 64


def test_fixture_summary_rejects_missing_and_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        summarize_fixture_inputs({"missing": tmp_path / "missing"})
    file = tmp_path / "file"
    file.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported"):
        _file_summary(
            file, alias="x", kind="directory", mask=False, include_digest=False
        )


def test_uncompressed_nifti_summary_uses_the_same_contract(tmp_path: Path) -> None:
    image = tmp_path / "image.nii"
    _write_nifti(image)
    summary = summarize_fixture_inputs({"image": image}, nifti_aliases=("image",))
    assert summary[0]["nifti"]["finite_count"] == 3
    assert summary[0]["nifti"]["nonzero_count"] == 2


def test_missing_reference_is_an_explicit_skip(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest = run_reference_command(
        stage="nomoco",
        executable=tmp_path / "missing-fsl",
        arguments=(),
        working_directory=tmp_path / "work",
        manifest_path=manifest_file,
        reference_version="FSL 6.0.4",
    )
    assert manifest["status"] == "skipped"
    assert json.loads(manifest_file.read_text())["status"] == "skipped"
    environment = {"PATH": str(Path(sys.executable).parent)}
    assert _resolve_executable(Path(sys.executable).name, environment) is not None
    assert _resolve_executable("definitely-not-an-executable", environment) is None


def test_success_records_provenance_outputs_and_redacts_paths(tmp_path: Path) -> None:
    work = tmp_path / "private" / "subject"
    work.mkdir(parents=True)
    script = tmp_path / "reference-script.sh"
    script.write_text("reference source", encoding="utf-8")
    code = (
        "from pathlib import Path; import sys; "
        "Path('result.txt').write_text('ok'); "
        "print(Path.cwd()); print(sys.argv[1])"
    )
    manifest_file = tmp_path / "public" / "manifest.json"

    manifest = run_reference_command(
        stage="synthetic",
        executable=sys.executable,
        arguments=("-c", code, str(work / "input.nii.gz")),
        working_directory=work,
        manifest_path=manifest_file,
        artifacts=(
            ReferenceArtifact("result.txt"),
            ReferenceArtifact("optional.txt", required=False),
        ),
        environment={"OMP_NUM_THREADS": "1"},
        reference_version="test-reference",
        script_paths=(script,),
        threads=1,
        include_output_digests=True,
    )

    assert manifest["status"] == "completed"
    assert manifest["outputs"][0]["sha256"]
    assert manifest["command"]["environment_keys"] == ["OMP_NUM_THREADS"]
    assert manifest["reference"]["scripts"][0]["name"] == script.name
    encoded = json.dumps(manifest)
    assert str(work) not in encoded
    assert "<path:0>" in manifest["stdout_summary"]
    assert manifest["metrics"]["wall_seconds"] >= 0
    assert manifest["metrics"]["child_cpu_seconds"] >= 0


def test_failed_command_writes_manifest(tmp_path: Path) -> None:
    manifest_file = tmp_path / "failure.json"
    with pytest.raises(ReferenceRunError, match="returned 7") as caught:
        run_reference_command(
            stage="failure",
            executable=sys.executable,
            arguments=("-c", "import sys; sys.exit(7)"),
            working_directory=tmp_path / "work",
            manifest_path=manifest_file,
            reference_version="test-reference",
        )
    assert caught.value.manifest["error_type"] == "ReferenceCommandFailed"
    assert json.loads(manifest_file.read_text())["returncode"] == 7


def test_missing_or_escaping_output_fails_explicitly(tmp_path: Path) -> None:
    common = {
        "stage": "outputs",
        "executable": sys.executable,
        "arguments": ("-c", "pass"),
        "working_directory": tmp_path / "work",
        "reference_version": "test-reference",
    }
    with pytest.raises(ReferenceRunError, match="missing required"):
        run_reference_command(
            **common,
            manifest_path=tmp_path / "missing.json",
            artifacts=(ReferenceArtifact("missing.nii.gz", kind="nifti"),),
        )
    with pytest.raises(ValueError, match="escapes"):
        run_reference_command(
            **common,
            manifest_path=tmp_path / "escape.json",
            artifacts=(ReferenceArtifact("../outside.txt"),),
        )


def test_timeout_writes_manifest(tmp_path: Path) -> None:
    manifest_file = tmp_path / "timeout.json"
    with pytest.raises(ReferenceRunError, match="exceeded") as caught:
        run_reference_command(
            stage="timeout",
            executable=sys.executable,
            arguments=("-c", "import time; time.sleep(1)"),
            working_directory=tmp_path / "work",
            manifest_path=manifest_file,
            reference_version="test-reference",
            timeout_seconds=0.01,
        )
    assert caught.value.manifest["error_type"] == "TimeoutExpired"
    assert json.loads(manifest_file.read_text())["status"] == "failed"


def test_redaction_and_resource_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _redact("/secret/value", ()) == "<absolute-path>"
    assert _redact("--data=/secret/value https://example.com/x", ()) == (
        "--data=<absolute-path> https://example.com/x"
    )
    before = os.times()
    real_import = __import__

    def reject_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_resource)
    metrics = _resource_metrics(before, 0.5)
    assert metrics["peak_rss_bytes"] is None
    assert metrics["peak_rss_method"] == "unavailable"


def test_resource_metrics_use_platform_native_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_resource = SimpleNamespace(
        RUSAGE_CHILDREN=0,
        getrusage=lambda _scope: SimpleNamespace(ru_maxrss=23),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(
        os, "uname", lambda: SimpleNamespace(sysname="Darwin"), raising=False
    )

    metrics = _resource_metrics(os.times(), 0.5)
    assert metrics["peak_rss_bytes"] == 23
    assert metrics["peak_rss_method"] == "RUSAGE_CHILDREN"


def test_public_manifest_audit_accepts_aggregates_and_rejects_private_content() -> None:
    audit_public_manifest(
        {"alias": "private-single-shell-v1", "affine": np.eye(4).tolist()},
        forbidden_terms=("not-present",),
    )
    for payload, message in (
        ({"subject_id": "example"}, "forbidden key"),
        (
            {"log": "/" + "home" + "/user/private/file"},
            "absolute path",
        ),
        ({"log": r"C:\\private\\file"}, "absolute path"),
        ({"alias": "sensitive-code"}, "forbidden identifier"),
    ):
        terms = ("sensitive-code",) if "alias" in payload else ()
        with pytest.raises(ValueError, match=message):
            audit_public_manifest(payload, forbidden_terms=terms)


def test_frozen_reference_assets_are_public_safe_and_structurally_complete() -> None:
    directory = Path(__file__).parent / "fixtures" / "reference"
    manifests = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in directory.glob("*.json")
    }
    assert set(manifests) == {
        "private_nomoco_reference.json",
        "synthetic_b0_motion_manifest.json",
        "synthetic_nomoco_manifest.json",
        "synthetic_nomoco_reference.json",
        "synthetic_legacy_reference.json",
        "synthetic_fieldmap_reference.json",
        "synthetic_eddy_reference.json",
        "synthetic_topup_reference.json",
        "synthetic_preprocessing_manifest.json",
        "synthetic_t1_registration_reference.json",
    }
    for payload in manifests.values():
        audit_public_manifest(payload, forbidden_terms=("100610", "hcp", "cxy"))

    private = manifests["private_nomoco_reference.json"]
    assert private["fixture_id"] == "private-single-shell-v1"
    assert private["status"] == "completed"
    assert len(private["outputs"]) == 10
    for output in private["outputs"]:
        if output["kind"] != "nifti":
            continue
        summary = output["nifti"]
        assert set(
            ("shape", "dtype", "affine", "qform", "sform", "finite_count", "nonzero_count")
        ).issubset(summary)
        assert np.asarray(summary["affine"]).shape == (4, 4)
        assert np.asarray(summary["qform"]).shape == (4, 4)
        assert np.asarray(summary["sform"]).shape == (4, 4)

    synthetic = manifests["synthetic_nomoco_reference.json"]
    assert synthetic["status"] == "completed"
    assert len(synthetic["outputs"]) == 10
    for output in synthetic["outputs"]:
        if output["kind"] == "nifti":
            assert output["grid_ref"] in synthetic["grids"]


def test_synthetic_preprocessing_generator_is_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    tool = root / "tools" / "generate_synthetic_preprocessing_fixtures.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    outputs = []
    for name in ("first", "second"):
        destination = tmp_path / name
        subprocess.run(
            [sys.executable, str(tool), str(destination)],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads((destination / "fixture_suite.json").read_text()))
    assert outputs[0] == outputs[1]
    frozen = json.loads(
        (root / "tests/fixtures/reference/synthetic_preprocessing_manifest.json").read_text()
    )
    assert outputs[0]["sha256"] == frozen["sha256"]
    assert nib.load(tmp_path / "first/reverse_pe_b0.nii.gz").shape == (16, 14, 12, 2)
    assert nib.load(tmp_path / "first/tensor_fsl6.nii.gz").shape == (16, 14, 12, 6)

    motion_tool = root / "tools" / "generate_synthetic_b0_motion_fixture.py"
    motion_outputs = []
    for name in ("motion_first", "motion_second"):
        destination = tmp_path / name
        subprocess.run(
            [sys.executable, str(motion_tool), "--output-directory", str(destination)],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        motion_outputs.append(
            {
                path.name: path.read_bytes()
                for path in destination.iterdir()
                if path.is_file()
            }
        )
    assert motion_outputs[0] == motion_outputs[1]

    eddy_tool = root / "tools" / "generate_synthetic_eddy_fixture.py"
    eddy_outputs = []
    for name in ("eddy_first", "eddy_second"):
        destination = tmp_path / name
        subprocess.run(
            [sys.executable, str(eddy_tool), str(destination)],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        eddy_outputs.append(json.loads((destination / "fixture.json").read_text()))
    assert eddy_outputs[0] == eddy_outputs[1]
    eddy_frozen = json.loads(
        (root / "tests/fixtures/reference/synthetic_eddy_reference.json").read_text()
    )
    stable_inputs = {"mask.nii", "bvals", "bvecs", "acqp.txt", "index.txt"}
    assert {
        name: eddy_outputs[0]["sha256"][name] for name in stable_inputs
    } == {name: eddy_frozen["sha256"][name] for name in stable_inputs}
    assert nib.load(tmp_path / "eddy_first/dwi.nii").shape == (26, 26, 18, 26)
    assert nib.load(tmp_path / "eddy_first/truth_uncorrupted_dwi.nii").shape == (
        26,
        26,
        18,
        26,
    )
    assert [
        [item["volume"], item["slice"]]
        for item in eddy_outputs[0]["slice_corruptions"]
    ] == eddy_frozen["truth_outliers"]


def test_numba_worker_request_is_clamped_to_runtime_capacity(monkeypatch) -> None:
    selected: list[int] = []
    monkeypatch.setattr(numba_helpers.config, "NUMBA_NUM_THREADS", 3)
    monkeypatch.setattr(numba_helpers, "set_num_threads", selected.append)

    assert numba_helpers.set_available_numba_threads(8) == 3
    assert selected == [3]


def test_reference_asset_audit_cli_passes_frozen_manifests() -> None:
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools/audit_reference_assets.py"),
            str(root / "tests/fixtures/reference"),
            "--forbid",
            "100610",
            "--forbid",
            "hcp",
            "--forbid",
            "cxy",
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"status": "passed", "manifests": 10}
