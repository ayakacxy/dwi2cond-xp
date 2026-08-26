"""Verify the P11 explicit DAG, manifests, cache, and numerical gates."""

from __future__ import annotations

import json
import builtins
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.preprocessing import pipeline

from dwi2cond_xp.preprocessing.pipeline import (
    ArtifactContract,
    PipelineRunner,
    StageDefinition,
    fingerprint_file,
    validate_artifacts,
)


def _write_nifti(path: Path, value: float = 1.0) -> None:
    values = np.full((3, 4, 5), value, dtype=np.float32)
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)


def test_pipeline_manifest_records_contract_and_uses_valid_cache(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    output = tmp_path / "stage" / "output.nii.gz"
    calls: list[int] = []
    progress: list[tuple[str, int, int, str]] = []

    def action() -> dict[str, object]:
        calls.append(1)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_nifti(output)
        return {"voxels": 60}

    stage = StageDefinition(
        "fit",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(output, "nifti", ndim=3),),
        parameters={"workers": 8, "algorithm": "reference-equivalent"},
        backend="python-optimized",
        implementation_version="0.1.0",
    )
    def record_progress(name: str, done: int, total: int, status: str) -> None:
        progress.append((name, done, total, status))
    runner = PipelineRunner(tmp_path / "manifests", progress=record_progress)
    first = runner.run((stage,))[0]

    assert first.status == "completed"
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["parameters"]["workers"] == 8
    assert manifest["backend"] == "python-optimized"
    assert manifest["implementation_version"] == "0.1.0"
    assert manifest["inputs"][0]["sha256"] == fingerprint_file(source)["sha256"]
    assert manifest["metrics"]["wall_seconds"] >= 0.0
    assert manifest["metrics"]["cpu_seconds"] >= 0.0
    assert "peak_rss_bytes" in manifest["metrics"]
    assert manifest["artifacts"][0]["shape"] == [3, 4, 5]
    assert manifest["artifacts"][0]["sha256"] == fingerprint_file(output)["sha256"]

    second_runner = PipelineRunner(
        tmp_path / "manifests", progress=record_progress
    )
    second = second_runner.run((stage,))[0]
    assert second.status == "cached"
    assert second.metrics["cache_hit"] is True
    assert len(calls) == 1
    assert progress[-1] == ("fit", 1, 1, "cached")


def test_pipeline_reexecutes_when_cached_artifact_is_structurally_invalid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    output = tmp_path / "output.json"
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        output.write_text('{"status":"ok"}\n', encoding="utf-8")

    stage = StageDefinition(
        "qa",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(output, "json"),),
    )
    PipelineRunner(tmp_path / "manifests").run((stage,))
    output.write_text("not-json\n", encoding="utf-8")

    result = PipelineRunner(tmp_path / "manifests").run((stage,))[0]
    assert result.status == "completed"
    assert len(calls) == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_pipeline_reexecutes_when_cached_artifact_content_is_replaced(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    output = tmp_path / "output.nii.gz"
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        _write_nifti(output, 1.0)

    stage = StageDefinition(
        "fit",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(output, "nifti", ndim=3),),
    )
    PipelineRunner(tmp_path / "manifests").run((stage,))
    _write_nifti(output, 0.0)

    result = PipelineRunner(tmp_path / "manifests").run((stage,))[0]

    assert result.status == "completed"
    assert calls == [1, 1]
    np.testing.assert_array_equal(np.asarray(nib.load(output).dataobj), 1.0)


def test_directory_hash_covers_topology_files_and_rejects_unknown_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "artifact"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (nested / "value.txt").write_text("one\n", encoding="utf-8")
    first = pipeline._sha256_directory(directory)
    (nested / "value.txt").write_text("two\n", encoding="utf-8")
    assert pipeline._sha256_directory(directory) != first

    unsupported = directory / "unsupported"
    unsupported.write_text("entry\n", encoding="utf-8")
    original_is_dir = Path.is_dir
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: False if path == unsupported else original_is_dir(path),
    )
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path == unsupported else original_is_file(path),
    )
    with pytest.raises(ValueError, match="unsupported entry"):
        pipeline._sha256_directory(directory)


def test_pipeline_source_digest_invalidates_development_cache(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    implementation = tmp_path / "algorithm.py"
    output = tmp_path / "output.txt"
    source.write_text("source\n", encoding="utf-8")
    implementation.write_text("VERSION = 1\n", encoding="utf-8")
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        output.write_text("complete\n", encoding="utf-8")

    stage = StageDefinition(
        "compute",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(output, "text"),),
        implementation_version="0+unknown",
    )
    PipelineRunner(
        tmp_path / "manifests", implementation_files=(implementation,)
    ).run((stage,))
    implementation.write_text("VERSION = 2\n", encoding="utf-8")
    result = PipelineRunner(
        tmp_path / "manifests", implementation_files=(implementation,)
    ).run((stage,))[0]

    assert result.status == "completed"
    assert calls == [1, 1]


def test_pipeline_dependency_and_failure_manifest_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    failed = StageDefinition(
        "failed",
        lambda: (_ for _ in ()).throw(RuntimeError("intentional")),
        inputs=(source,),
        outputs=(ArtifactContract(tmp_path / "missing.txt", "text"),),
    )
    progress: list[tuple[str, int, int, str]] = []
    runner = PipelineRunner(tmp_path / "manifests", progress=lambda *args: progress.append(args))
    with pytest.raises(RuntimeError, match="intentional"):
        runner.run((failed,))
    manifest = json.loads(
        (tmp_path / "manifests" / "failed.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    assert progress[-1] == ("failed", 0, 1, "failed")

    dependent = StageDefinition(
        "dependent",
        lambda: None,
        inputs=(source,),
        outputs=(),
        dependencies=("failed",),
    )
    with pytest.raises(ValueError, match="dependencies have not completed"):
        PipelineRunner(tmp_path / "other").run((dependent,))


def test_final_numerical_validation_is_separate_from_cache_structure_check(
    tmp_path: Path,
) -> None:
    output = tmp_path / "values.nii.gz"
    values = np.ones((2, 2, 2), dtype=np.float32)
    values[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(values, np.eye(4)), output)
    contract = ArtifactContract(output, "nifti", ndim=3)

    structural = validate_artifacts((contract,), numerical=False)
    assert structural[0]["shape"] == [2, 2, 2]
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_artifacts((contract,), numerical=True)


@pytest.mark.parametrize("name", ("", "bad/name", "bad\\name"))
def test_pipeline_rejects_unsafe_stage_names(tmp_path: Path, name: str) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    stage = StageDefinition(name, lambda: None, inputs=(source,), outputs=())
    with pytest.raises(ValueError, match="Stage name"):
        PipelineRunner(tmp_path / "manifests").run((stage,))


def test_artifact_contract_rejects_every_invalid_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify each cache structural gate so corrupt artifacts are not accepted as hits."""

    with pytest.raises(FileNotFoundError, match="Pipeline input"):
        fingerprint_file(tmp_path / "missing-input")

    directory = tmp_path / "directory"
    with pytest.raises(FileNotFoundError, match="output directory"):
        validate_artifacts((ArtifactContract(directory, "directory"),))
    directory.mkdir()
    with pytest.raises(ValueError, match="directory is empty"):
        validate_artifacts((ArtifactContract(directory, "directory"),))
    assert validate_artifacts(
        (ArtifactContract(directory, "directory", allow_empty=True),)
    )[0]["entries"] == 0

    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError, match="output artifact"):
        validate_artifacts((ArtifactContract(missing, "file"),))
    empty = tmp_path / "empty.txt"
    empty.touch()
    with pytest.raises(ValueError, match="artifact is empty"):
        validate_artifacts((ArtifactContract(empty, "file"),))
    assert validate_artifacts(
        (ArtifactContract(empty, "file", allow_empty=True),)
    )[0]["size_bytes"] == 0

    payload = tmp_path / "payload.json"
    payload.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        validate_artifacts((ArtifactContract(payload, "json"),))
    text = tmp_path / "blank.txt"
    text.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no values"):
        validate_artifacts((ArtifactContract(text, "text"),))

    nifti = tmp_path / "image.nii.gz"
    _write_nifti(nifti)
    with pytest.raises(ValueError, match="must be 4D"):
        validate_artifacts((ArtifactContract(nifti, "nifti", ndim=4),))
    with pytest.raises(ValueError, match="final axis must be 6"):
        validate_artifacts((ArtifactContract(nifti, "nifti", final_axis=6),))

    fake = SimpleNamespace(
        shape=(3, 4, 5),
        affine=np.full((4, 4), np.nan),
        get_data_dtype=lambda: np.dtype(np.float32),
    )
    monkeypatch.setattr(pipeline.nib, "load", lambda *_args, **_kwargs: fake)
    with pytest.raises(ValueError, match="affine must be finite"):
        validate_artifacts((ArtifactContract(nifti, "nifti"),))


def test_pipeline_duplicate_names_and_unavailable_rss_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("input\n", encoding="utf-8")
    stage = StageDefinition("same", lambda: None, inputs=(source,), outputs=())
    with pytest.raises(ValueError, match="must be unique"):
        PipelineRunner(tmp_path / "manifests").run((stage, stage))

    original_import = builtins.__import__

    def without_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("platform without resource")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_resource)
    assert pipeline._peak_rss_bytes() == (None, "unavailable")


def test_pipeline_peak_rss_uses_platform_native_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _scope: SimpleNamespace(ru_maxrss=17),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(pipeline.sys, "platform", "darwin")

    assert pipeline._peak_rss_bytes() == (17, "RUSAGE_SELF")
