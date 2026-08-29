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


def test_pipeline_can_disable_cache_when_backend_identity_is_incomplete(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    output = tmp_path / "output.txt"
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        output.write_text(f"run-{len(calls)}\n", encoding="utf-8")

    stage = StageDefinition(
        "fem",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(output, "text"),),
        cacheable=False,
    )
    first = PipelineRunner(tmp_path / "manifests").run((stage,))[0]
    second = PipelineRunner(tmp_path / "manifests").run((stage,))[0]

    assert first.status == "completed"
    assert second.status == "completed"
    assert calls == [1, 1]
    assert output.read_text(encoding="utf-8") == "run-2\n"


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


def test_pipeline_reexecutes_when_dynamic_fem_artifact_changes(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    output = tmp_path / "simulation.json"
    field = tmp_path / "field.msh"
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        field.write_text("fresh-field\n", encoding="utf-8")
        output.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "output_directory": str(tmp_path),
                    "outputs": [str(field.resolve())],
                    "artifacts": [
                        {
                            "path": str(field.resolve()),
                            "relative_path": field.name,
                            "type": "file",
                            "bytes": field.stat().st_size,
                            "sha256": pipeline._sha256_file(field),
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    stage = StageDefinition(
        "fem",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(output, "json", dynamic_inventory=True),),
    )
    PipelineRunner(tmp_path / "manifests").run((stage,))
    field.write_text("tampered-field\n", encoding="utf-8")

    result = PipelineRunner(tmp_path / "manifests").run((stage,))[0]

    assert result.status == "completed"
    assert calls == [1, 1]
    assert field.read_text(encoding="utf-8") == "fresh-field\n"


def test_pipeline_failed_attempt_cannot_reuse_stale_declared_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    contracts = (
        ArtifactContract(first, "text"),
        ArtifactContract(second, "text"),
    )

    def failing_action() -> None:
        first.write_text("attempt-one\n", encoding="utf-8")
        second.write_text("stale-second\n", encoding="utf-8")
        raise RuntimeError("attempt failed")

    failed_stage = StageDefinition(
        "transaction",
        failing_action,
        inputs=(source,),
        outputs=contracts,
    )
    with pytest.raises(RuntimeError, match="attempt failed"):
        PipelineRunner(tmp_path / "manifests").run((failed_stage,))
    assert not first.exists()
    assert not second.exists()

    def incomplete_retry() -> None:
        first.write_text("attempt-two\n", encoding="utf-8")

    retry_stage = StageDefinition(
        "transaction",
        incomplete_retry,
        inputs=(source,),
        outputs=contracts,
    )
    with pytest.raises(FileNotFoundError, match="second.txt"):
        PipelineRunner(tmp_path / "manifests").run((retry_stage,))
    assert not first.exists()
    assert not second.exists()


def test_pipeline_numerical_runtime_identity_invalidates_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.txt"
    output = tmp_path / "output.txt"
    source.write_text("source\n", encoding="utf-8")
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        output.write_text("result\n", encoding="utf-8")

    stage = StageDefinition(
        "runtime",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(output, "text"),),
    )
    monkeypatch.setattr(pipeline, "_numerical_runtime_identity", lambda: {"blas": "A"})
    PipelineRunner(tmp_path / "manifests").run((stage,))
    monkeypatch.setattr(pipeline, "_numerical_runtime_identity", lambda: {"blas": "B"})

    result = PipelineRunner(tmp_path / "manifests").run((stage,))[0]

    assert result.status == "completed"
    assert calls == [1, 1]


def test_dynamic_artifact_manifest_rejects_all_invalid_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "simulation.json"
    contract = ArtifactContract(manifest, "json", dynamic_inventory=True)

    for payload, message in (
        (
            {"output_directory": str(tmp_path), "artifacts": {}},
            "contain a list",
        ),
        (
            {"output_directory": str(tmp_path), "artifacts": ["bad"]},
            "Invalid dynamic artifact entry",
        ),
        ({"artifacts": []}, "declare output_directory"),
        (
            {
                "output_directory": str(tmp_path),
                "artifacts": [{"path": str(tmp_path / "missing.msh")}],
            },
            "Missing dynamic output artifact",
        ),
    ):
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError), match=message):
            validate_artifacts((contract,), numerical=False)

    image_path = tmp_path / "field.nii.gz"
    _write_nifti(image_path)
    image = nib.load(image_path)
    entry = {
        "path": str(image_path.resolve()),
        "relative_path": image_path.name,
        "type": "nifti",
        "bytes": image_path.stat().st_size,
        "sha256": pipeline._sha256_file(image_path),
        "shape": list(image.shape),
        "affine": image.affine.tolist(),
        "dtype": str(image.get_data_dtype()),
    }
    manifest.write_text(
        json.dumps({"output_directory": str(tmp_path), "artifacts": [entry]}),
        encoding="utf-8",
    )
    metadata = validate_artifacts((contract,), numerical=False)[0]
    assert metadata["dynamic_artifacts"] == [entry]

    invalid_payloads = (
        (
            {
                "status": "completed",
                "output_directory": str(tmp_path),
                "artifacts": [],
            },
            "must not be empty",
        ),
        (
            {
                "output_directory": str(tmp_path / "nested"),
                "artifacts": [entry],
            },
            "outside output_directory",
        ),
        (
            {
                "output_directory": str(tmp_path),
                "artifacts": [entry, entry],
            },
            "listed more than once",
        ),
        (
            {
                "output_directory": str(tmp_path),
                "outputs": {},
                "artifacts": [entry],
            },
            "must be a list",
        ),
        (
            {
                "output_directory": str(tmp_path),
                "outputs": [3],
                "artifacts": [entry],
            },
            "artifact paths only",
        ),
        (
            {
                "output_directory": str(tmp_path),
                "outputs": [str(tmp_path / "missing-output.msh")],
                "artifacts": [entry],
            },
            "missing from dynamic inventory",
        ),
        (
            {
                "output_directory": str(tmp_path),
                "artifacts": [{**entry, "shape": [99, 99, 99]}],
            },
            "changed after publication",
        ),
    )
    for payload, message in invalid_payloads:
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            validate_artifacts((contract,), numerical=False)

    manifest.write_text(
        json.dumps(
            {
                "output_directory": str(tmp_path),
                "outputs": [image_path.name],
                "artifacts": [entry],
            }
        ),
        encoding="utf-8",
    )
    validate_artifacts((contract,), numerical=False)

    class InvalidAffineImage:
        shape = (2, 2, 2)
        affine = np.full((4, 4), np.nan)

        @staticmethod
        def get_data_dtype():
            return np.dtype(np.float32)

    monkeypatch.setattr(pipeline.nib, "load", lambda *_args, **_kwargs: InvalidAffineImage())
    manifest.write_text(
        json.dumps({"output_directory": str(tmp_path), "artifacts": [entry]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="affine is not finite"):
        validate_artifacts((contract,), numerical=False)


def test_regular_json_artifacts_mapping_is_not_a_dynamic_inventory(
    tmp_path: Path,
) -> None:
    report = tmp_path / "legacy_qa.json"
    report.write_text(
        json.dumps({"status": "completed", "artifacts": {"tensor": "DTI.nii.gz"}}),
        encoding="utf-8",
    )
    metadata = validate_artifacts((ArtifactContract(report, "json"),))[0]
    assert metadata["keys"] == ["artifacts", "status"]
    assert "dynamic_artifacts" not in metadata


def test_output_cleanup_removes_declared_directory(tmp_path: Path) -> None:
    output = tmp_path / "stage-output"
    output.mkdir()
    (output / "partial.txt").write_text("partial\n", encoding="utf-8")

    pipeline._remove_declared_outputs((ArtifactContract(output, "directory"),))

    assert not output.exists()


def test_failure_transaction_stage_preserves_previous_declared_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.txt"
    output = tmp_path / "published.txt"
    source.write_text("source\n", encoding="utf-8")
    output.write_text("previous\n", encoding="utf-8")

    def fail() -> None:
        raise RuntimeError("transaction failed")

    stage = StageDefinition(
        "publish",
        fail,
        inputs=(source,),
        outputs=(ArtifactContract(output, "text"),),
        preserve_outputs_on_attempt=True,
    )
    with pytest.raises(RuntimeError, match="transaction failed"):
        PipelineRunner(tmp_path / "manifests").run((stage,))
    assert output.read_text(encoding="utf-8") == "previous\n"


def test_numerical_runtime_identity_covers_unavailable_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "threadpool_info",
        lambda: [
            {
                "internal_api": "openblas",
                "user_api": "blas",
                "filepath": "/synthetic/libblas.so",
                "version": "test",
                "num_threads": 2,
                "ignored": "not part of the cache identity",
            }
        ],
    )
    libraries = pipeline._numerical_runtime_identity()["threadpool_libraries"]
    assert libraries[0]["internal_api"] == "openblas"
    assert libraries[0]["num_threads"] == 2
    assert "ignored" not in libraries[0]

    real_version = pipeline.importlib.metadata.version

    def missing_scipy(name: str) -> str:
        if name == "scipy":
            raise pipeline.importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(pipeline.importlib.metadata, "version", missing_scipy)
    real_import = builtins.__import__

    def missing_numba(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("numba unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_numba)
    missing = pipeline._numerical_runtime_identity()
    assert missing["distributions"]["scipy"] == "not-installed"
    assert "simnibs" in missing["distributions"]
    assert missing["numba"] == {"status": "not-installed"}

    monkeypatch.setattr(builtins, "__import__", real_import)
    import numba

    monkeypatch.setattr(
        numba,
        "get_num_threads",
        lambda: (_ for _ in ()).throw(RuntimeError("threading unavailable")),
    )
    unavailable = pipeline._numerical_runtime_identity()
    assert unavailable["numba"]["status"] == "unavailable"
    assert unavailable["numba"]["error_type"] == "RuntimeError"


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


def test_pipeline_preserves_producer_failure_manifest_and_phase(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    producer_manifest = tmp_path / "producer.json"

    def action() -> None:
        producer_manifest.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_phase": "solve",
                    "artifacts": [],
                    "output_directory": str(tmp_path),
                }
            ),
            encoding="utf-8",
        )
        error = RuntimeError("producer failed")
        error.dwi2cond_xp_failure_manifest = str(producer_manifest)
        error.dwi2cond_xp_failed_phase = "solve"
        raise error

    stage = StageDefinition(
        "producer",
        action,
        inputs=(source,),
        outputs=(ArtifactContract(producer_manifest, "json"),),
        preserve_outputs_on_attempt=True,
    )
    manifest_directory = tmp_path / "manifests"
    with pytest.raises(RuntimeError, match="producer failed"):
        PipelineRunner(manifest_directory).run((stage,))

    assert producer_manifest.is_file()
    runner_manifest = json.loads(
        (manifest_directory / "producer.json").read_text(encoding="utf-8")
    )
    assert runner_manifest["failed_phase"] == "solve"
    assert runner_manifest["producer_failure"]["failed_phase"] == "solve"
    assert runner_manifest["producer_failure_manifest"] == str(producer_manifest)


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
