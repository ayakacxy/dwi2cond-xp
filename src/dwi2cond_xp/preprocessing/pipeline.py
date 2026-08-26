"""Provide a recoverable, auditable explicit stage DAG for the full dwi2cond pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Literal

import nibabel as nib
import numpy as np
from threadpoolctl import threadpool_info


ArtifactKind = Literal["nifti", "json", "text", "directory", "file"]
ProgressCallback = Callable[[str, int, int, str], None]
StageAction = Callable[[], Mapping[str, object] | None]


@dataclass(frozen=True)
class ArtifactContract:
    """Describe a stage output and its minimum structural contract."""

    path: Path
    kind: ArtifactKind = "file"
    ndim: int | None = None
    final_axis: int | None = None
    allow_empty: bool = False


@dataclass(frozen=True)
class StageDefinition:
    """Describe a DAG stage without implicit dependencies or fallback."""

    name: str
    action: StageAction
    inputs: tuple[Path, ...]
    outputs: tuple[ArtifactContract, ...]
    dependencies: tuple[str, ...] = ()
    parameters: Mapping[str, object] = field(default_factory=dict)
    backend: str = "python-optimized"
    implementation_version: str = "unknown"


@dataclass(frozen=True)
class StageRunResult:
    """Return a summary of one stage execution or cache hit."""

    name: str
    status: Literal["completed", "cached"]
    manifest_path: Path
    fingerprint: str
    metrics: Mapping[str, object]


def _json_bytes(payload: object) -> bytes:
    """Generate a cross-platform stable JSON byte representation."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream file hashing in fixed-size blocks to avoid copying large NIfTI images."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash directory topology and every regular file in stable relative-path order."""

    digest = hashlib.sha256()
    for entry in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        if entry.is_dir():
            digest.update(b"directory\0" + relative + b"\0")
        elif entry.is_file():
            digest.update(b"file\0" + relative + b"\0")
            digest.update(_sha256_file(entry).encode("ascii"))
            digest.update(b"\0")
        else:
            raise ValueError(f"Output directory contains an unsupported entry: {entry}")
    return digest.hexdigest()


def fingerprint_file(path: str | Path) -> dict[str, object]:
    """Return a file's strong content fingerprint and structural metadata."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pipeline input is not a file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _artifact_metadata(
    contract: ArtifactContract,
    *,
    numerical: bool,
) -> dict[str, object]:
    """Validate an artifact and return a structural summary suitable for a manifest."""

    path = contract.path.resolve()
    if contract.kind == "directory":
        if not path.is_dir():
            raise FileNotFoundError(f"Missing output directory: {path}")
        entries = sum(1 for _ in path.iterdir())
        if entries == 0 and not contract.allow_empty:
            raise ValueError(f"Output directory is empty: {path}")
        return {
            "path": str(path),
            "kind": contract.kind,
            "entries": entries,
            "sha256": _sha256_directory(path),
        }

    if not path.is_file():
        raise FileNotFoundError(f"Missing output artifact: {path}")
    size = int(path.stat().st_size)
    if size == 0 and not contract.allow_empty:
        raise ValueError(f"Output artifact is empty: {path}")
    metadata: dict[str, object] = {
        "path": str(path),
        "kind": contract.kind,
        "size_bytes": size,
        "sha256": _sha256_file(path),
    }
    if contract.kind == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact must contain an object: {path}")
        metadata["keys"] = sorted(str(key) for key in payload)
        dynamic = payload.get("artifacts")
        if dynamic is not None:
            if not isinstance(dynamic, list):
                raise ValueError(f"JSON artifacts must contain a list: {path}")
            validated_dynamic: list[dict[str, object]] = []
            for entry in dynamic:
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("path"), str
                ):
                    raise ValueError(f"Invalid dynamic artifact entry: {path}")
                dynamic_path = Path(entry["path"]).resolve()
                if not dynamic_path.is_file():
                    raise FileNotFoundError(
                        f"Missing dynamic output artifact: {dynamic_path}"
                    )
                current = {
                    "path": str(dynamic_path),
                    "relative_path": entry.get("relative_path"),
                    "type": entry.get("type", "file"),
                    "bytes": dynamic_path.stat().st_size,
                    "sha256": _sha256_file(dynamic_path),
                }
                for key in ("shape", "affine", "dtype"):
                    if key in entry:
                        current[key] = entry[key]
                if any(current.get(key) != entry.get(key) for key in current):
                    raise ValueError(
                        f"Dynamic output artifact changed after publication: {dynamic_path}"
                    )
                validated_dynamic.append(current)
            metadata["dynamic_artifacts"] = validated_dynamic
    elif contract.kind == "text":
        if not path.read_text(encoding="utf-8").strip() and not contract.allow_empty:
            raise ValueError(f"Text artifact contains no values: {path}")
    elif contract.kind == "nifti":
        image = nib.load(str(path), mmap=True)
        if contract.ndim is not None and len(image.shape) != contract.ndim:
            raise ValueError(
                f"NIfTI {path} must be {contract.ndim}D; found {len(image.shape)}D"
            )
        if contract.final_axis is not None and image.shape[-1] != contract.final_axis:
            raise ValueError(
                f"NIfTI {path} final axis must be {contract.final_axis}; "
                f"found {image.shape[-1]}"
            )
        affine = np.asarray(image.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError(f"NIfTI affine must be finite and 4x4: {path}")
        metadata.update(
            {
                "shape": [int(value) for value in image.shape],
                "dtype": str(image.get_data_dtype()),
                "affine": affine.tolist(),
            }
        )
        if numerical:
            values = np.asarray(image.dataobj)
            finite = np.isfinite(values)
            if not np.all(finite):
                raise ValueError(f"NIfTI contains NaN or Inf: {path}")
            metadata["finite_values"] = int(np.count_nonzero(finite))
    return metadata


def validate_artifacts(
    contracts: Sequence[ArtifactContract],
    *,
    numerical: bool = False,
) -> list[dict[str, object]]:
    """Validate outputs; cache hits read structure only unless formal validation reads values."""

    return [
        _artifact_metadata(contract, numerical=numerical) for contract in contracts
    ]


def _peak_rss_bytes() -> tuple[int | None, str]:
    """Return the current process's historical peak RSS in platform-native units."""

    try:
        import resource
    except ImportError:
        return None, "unavailable"
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    scale = 1 if sys.platform == "darwin" else 1024
    return value * scale, "RUSAGE_SELF"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a JSON manifest atomically within the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_declared_outputs(contracts: Sequence[ArtifactContract]) -> None:
    """Remove only this stage's declared outputs before or after a failed attempt."""

    for contract in contracts:
        path = contract.path.resolve()
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _numerical_runtime_identity() -> dict[str, object]:
    """Capture numerical-library and threading identities that can change results."""

    distributions: dict[str, str] = {}
    for name in ("numpy", "scipy", "numba", "nibabel", "threadpoolctl"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = "not-installed"
    libraries = []
    for item in threadpool_info():
        libraries.append(
            {
                key: item.get(key)
                for key in (
                    "internal_api",
                    "user_api",
                    "prefix",
                    "filepath",
                    "version",
                    "threading_layer",
                    "architecture",
                    "num_threads",
                )
            }
        )
    libraries.sort(key=lambda item: str(item.get("filepath")))
    try:
        from numba import config as numba_config
        from numba import get_num_threads, threading_layer

        active_numba_threads = get_num_threads()
        active_threading_layer = threading_layer()
        numba_identity = {
            "configured_threading_layer": str(numba_config.THREADING_LAYER),
            "active_threading_layer": active_threading_layer,
            "configured_num_threads": int(numba_config.NUMBA_NUM_THREADS),
            "active_num_threads": int(active_numba_threads),
        }
    except ImportError:
        numba_identity = {"status": "not-installed"}
    except (RuntimeError, ValueError) as error:
        numba_identity = {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "distributions": distributions,
        "threadpool_libraries": libraries,
        "numba": numba_identity,
    }


class PipelineRunner:
    """Run stages in deterministic topological order and preserve backend identity."""

    def __init__(
        self,
        manifest_directory: str | Path,
        *,
        progress: ProgressCallback | None = None,
        implementation_files: Sequence[str | Path] = (),
    ) -> None:
        self.manifest_directory = Path(manifest_directory).resolve()
        self.progress = progress
        self.implementation_files = tuple(
            Path(path).resolve() for path in implementation_files
        )
        self._file_fingerprints: dict[Path, dict[str, object]] = {}
        self._results: dict[str, StageRunResult] = {}

    def _input_fingerprint(self, path: Path) -> dict[str, object]:
        resolved = path.resolve()
        cached = self._file_fingerprints.get(resolved)
        if cached is None:
            cached = fingerprint_file(resolved)
            self._file_fingerprints[resolved] = cached
        return cached

    def _stage_identity(
        self,
        stage: StageDefinition,
        inputs: Sequence[Mapping[str, object]],
    ) -> tuple[str, dict[str, object]]:
        dependencies = {
            name: self._results[name].fingerprint for name in stage.dependencies
        }
        implementation = [
            self._input_fingerprint(path) for path in self.implementation_files
        ]
        payload = {
            "schema_version": 1,
            "stage": stage.name,
            "inputs": list(inputs),
            "dependencies": dependencies,
            "parameters": dict(stage.parameters),
            "backend": stage.backend,
            "implementation_version": stage.implementation_version,
            "implementation_files": implementation,
            "numerical_runtime": _numerical_runtime_identity(),
        }
        return hashlib.sha256(_json_bytes(payload)).hexdigest(), payload

    def _manifest_path(self, stage_name: str) -> Path:
        return self.manifest_directory / f"{stage_name}.json"

    def _cached_result(
        self,
        stage: StageDefinition,
        manifest_path: Path,
        fingerprint: str,
    ) -> StageRunResult | None:
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") != "completed"
                or manifest.get("fingerprint") != fingerprint
            ):
                return None
            current_artifacts = validate_artifacts(stage.outputs, numerical=False)
            if manifest.get("artifacts") != current_artifacts:
                return None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        metrics = dict(manifest.get("metrics", {}))
        metrics["cache_hit"] = True
        return StageRunResult(
            stage.name, "cached", manifest_path, fingerprint, metrics
        )

    def run_stage(self, stage: StageDefinition) -> StageRunResult:
        """Run a stage whose dependencies are met, recording failures without fallback."""

        if not stage.name or any(character in stage.name for character in "/\\"):
            raise ValueError("Stage name must be one nonempty path-safe component")
        missing_dependencies = [
            name for name in stage.dependencies if name not in self._results
        ]
        if missing_dependencies:
            raise ValueError(
                "Stage dependencies have not completed: "
                + ", ".join(missing_dependencies)
            )
        inputs = [self._input_fingerprint(path) for path in stage.inputs]
        fingerprint, identity = self._stage_identity(stage, inputs)
        manifest_path = self._manifest_path(stage.name)
        cached = self._cached_result(stage, manifest_path, fingerprint)
        if cached is not None:
            self._results[stage.name] = cached
            if self.progress is not None:
                self.progress(stage.name, 1, 1, "cached")
            return cached

        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        if self.progress is not None:
            self.progress(stage.name, 0, 1, "running")
        _remove_declared_outputs(stage.outputs)
        try:
            action_result = stage.action()
            artifacts = validate_artifacts(stage.outputs, numerical=False)
        except Exception as error:
            _remove_declared_outputs(stage.outputs)
            peak_rss, peak_rss_method = _peak_rss_bytes()
            failed_manifest: dict[str, object] = {
                **identity,
                "status": "failed",
                "fingerprint": fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
                "metrics": {
                    "wall_seconds": time.perf_counter() - wall_started,
                    "cpu_seconds": time.process_time() - cpu_started,
                    "peak_rss_bytes": peak_rss,
                    "peak_rss_method": peak_rss_method,
                    "cache_hit": False,
                },
            }
            _atomic_json(manifest_path, failed_manifest)
            if self.progress is not None:
                self.progress(stage.name, 0, 1, "failed")
            raise

        peak_rss, peak_rss_method = _peak_rss_bytes()
        metrics: dict[str, object] = {
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_bytes": peak_rss,
            "peak_rss_method": peak_rss_method,
            "cache_hit": False,
        }
        manifest: dict[str, object] = {
            **identity,
            "status": "completed",
            "fingerprint": fingerprint,
            "metrics": metrics,
            "artifacts": artifacts,
            "result": {} if action_result is None else dict(action_result),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        _atomic_json(manifest_path, manifest)
        result = StageRunResult(
            stage.name, "completed", manifest_path, fingerprint, metrics
        )
        self._results[stage.name] = result
        if self.progress is not None:
            self.progress(stage.name, 1, 1, "completed")
        return result

    def run(self, stages: Sequence[StageDefinition]) -> tuple[StageRunResult, ...]:
        """Run all stages in the deterministic topological order supplied by the caller."""

        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")
        return tuple(self.run_stage(stage) for stage in stages)

    def validate_final_outputs(
        self, stages: Sequence[StageDefinition]
    ) -> dict[str, list[dict[str, object]]]:
        """Run formal numerical validation separately from cache structural checks."""

        return {
            stage.name: validate_artifacts(stage.outputs, numerical=True)
            for stage in stages
        }


__all__ = [
    "ArtifactContract",
    "PipelineRunner",
    "StageDefinition",
    "StageRunResult",
    "fingerprint_file",
    "validate_artifacts",
]
