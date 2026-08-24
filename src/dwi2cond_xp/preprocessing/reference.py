"""Run local FSL references without making them runtime dependencies.

The public manifest deliberately stores aliases and structural summaries instead
of source paths. Private voxel arrays and subject identifiers remain outside the
repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class ReferenceArtifact:
    """Describe one expected reference output relative to the working directory."""

    path: str
    kind: str = "file"
    required: bool = True
    mask: bool = False


class ReferenceRunError(RuntimeError):
    """Report a failed reference command after its manifest has been written."""

    def __init__(self, message: str, manifest: Mapping[str, object]) -> None:
        super().__init__(message)
        self.manifest = dict(manifest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _redact(text: str, roots: Sequence[Path]) -> str:
    redacted = text
    for index, root in enumerate(roots):
        redacted = redacted.replace(str(root.resolve()), f"<path:{index}>")
    return re.sub(
        r"(^|[=\s])/(?!/)\S+",
        lambda match: f"{match.group(1)}<absolute-path>",
        redacted,
    )


def _nifti_summary(path: Path, *, mask: bool) -> dict[str, object]:
    image = nib.load(str(path))
    # A compressed proxy restarts gzip decompression for every random slice.
    # Load it sequentially once, then retain bounded per-slab temporary arrays.
    loaded = (
        np.asanyarray(image.dataobj)
        if path.name.lower().endswith((".nii.gz", ".img.gz"))
        else None
    )
    finite_count = 0
    nonzero_count = 0
    finite_min: float | None = None
    finite_max: float | None = None
    for index in range(image.shape[0]):
        values = (
            np.asanyarray(image.dataobj[index, ...])
            if loaded is None
            else loaded[index, ...]
        )
        finite = np.isfinite(values)
        finite_count += int(np.count_nonzero(finite))
        nonzero_count += int(np.count_nonzero(values[finite]))
        if np.any(finite):
            slab_min = float(np.min(values[finite]))
            slab_max = float(np.max(values[finite]))
            finite_min = slab_min if finite_min is None else min(finite_min, slab_min)
            finite_max = slab_max if finite_max is None else max(finite_max, slab_max)
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    summary: dict[str, object] = {
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "affine": np.asarray(image.affine).tolist(),
        "qform": None if qform is None else np.asarray(qform).tolist(),
        "qform_code": int(qform_code),
        "sform": None if sform is None else np.asarray(sform).tolist(),
        "sform_code": int(sform_code),
        "finite_count": finite_count,
        "nonzero_count": nonzero_count,
        "finite_min": finite_min,
        "finite_max": finite_max,
    }
    if mask:
        summary["mask_count"] = nonzero_count
    return summary


def _file_summary(
    path: Path,
    *,
    alias: str,
    kind: str,
    mask: bool,
    include_digest: bool,
) -> dict[str, object]:
    if kind not in {"file", "nifti"}:
        raise ValueError(f"Unsupported reference artifact kind: {kind}")
    summary: dict[str, object] = {
        "alias": alias,
        "kind": kind,
        "size_bytes": path.stat().st_size,
    }
    if include_digest:
        summary["sha256"] = _sha256(path)
    if kind == "nifti":
        summary["nifti"] = _nifti_summary(path, mask=mask)
    return summary


def summarize_fixture_inputs(
    inputs: Mapping[str, str | Path],
    *,
    nifti_aliases: Sequence[str] = (),
    mask_aliases: Sequence[str] = (),
    include_digests: bool = False,
) -> list[dict[str, object]]:
    """Summarize fixture inputs by public aliases without recording source paths."""

    nifti_set = set(nifti_aliases)
    mask_set = set(mask_aliases)
    summaries = []
    for alias in sorted(inputs):
        path = Path(inputs[alias]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Fixture input does not exist: {alias}")
        kind = "nifti" if alias in nifti_set else "file"
        summaries.append(
            _file_summary(
                path,
                alias=alias,
                kind=kind,
                mask=alias in mask_set,
                include_digest=include_digests,
            )
        )
    return summaries


def audit_public_manifest(
    payload: Mapping[str, object], *, forbidden_terms: Sequence[str] = ()
) -> None:
    """Reject paths, direct identifiers, credentials, or voxel arrays.

    Aggregate image geometry and scalar counts are allowed. The audit is a
    release guard, not a claim that arbitrary free text is anonymized.
    """

    forbidden_keys = {
        "api_key",
        "access_token",
        "password",
        "patient_id",
        "subject_id",
        "source_path",
        "voxel_values",
    }
    absolute_path = re.compile(r"(?:^|[=\s])(?:/[A-Za-z0-9_.-]+){2,}|[A-Za-z]:\\")

    def inspect(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys:
                    raise ValueError(f"Public manifest contains forbidden key: {location}.{key}")
                inspect(child, f"{location}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                inspect(child, f"{location}[{index}]")
            return
        if not isinstance(value, str):
            return
        if absolute_path.search(value) and "<absolute-path>" not in value:
            raise ValueError(f"Public manifest contains an absolute path at {location}")
        lowered = value.lower()
        for term in forbidden_terms:
            if term and term.lower() in lowered:
                raise ValueError(
                    f"Public manifest contains forbidden identifier at {location}"
                )

    inspect(payload, "manifest")


def _resolve_executable(
    executable: str | Path, environment: Mapping[str, str]
) -> Path | None:
    value = str(executable)
    if os.sep in value or (os.altsep is not None and os.altsep in value):
        candidate = Path(value).expanduser().resolve()
        return (
            candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
        )
    located = shutil.which(value, path=environment.get("PATH"))
    return None if located is None else Path(located).resolve()


def _resource_metrics(
    before: os.times_result, wall_seconds: float
) -> dict[str, object]:
    after = os.times()
    metrics: dict[str, object] = {
        "wall_seconds": wall_seconds,
        "child_cpu_seconds": (after.children_user - before.children_user)
        + (after.children_system - before.children_system),
        "peak_rss_bytes": None,
        "peak_rss_method": "unavailable",
    }
    try:
        import resource
    except ImportError:
        return metrics
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if os.uname().sysname == "Darwin" else 1024
    metrics["peak_rss_bytes"] = int(usage.ru_maxrss * scale)
    metrics["peak_rss_method"] = "RUSAGE_CHILDREN"
    return metrics


def run_reference_command(
    *,
    stage: str,
    executable: str | Path,
    arguments: Sequence[str],
    working_directory: str | Path,
    manifest_path: str | Path,
    artifacts: Sequence[ReferenceArtifact] = (),
    environment: Mapping[str, str] | None = None,
    reference_version: str,
    script_paths: Sequence[str | Path] = (),
    threads: int = 1,
    timeout_seconds: float | None = None,
    include_output_digests: bool = False,
) -> dict[str, object]:
    """Run one explicit reference stage and write a public-safe manifest."""

    workdir = Path(working_directory).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    run_environment = os.environ.copy()
    run_environment.update(environment or {})
    resolved = _resolve_executable(executable, run_environment)
    roots = (workdir, Path.home())
    manifest: dict[str, object] = {
        "schema_version": 1,
        "stage": stage,
        "status": "pending",
        "reference": {
            "executable": Path(str(executable)).name,
            "version": reference_version,
            "scripts": [
                {
                    "name": Path(path).name,
                    "size_bytes": Path(path).stat().st_size,
                    "sha256": _sha256(Path(path)),
                }
                for path in script_paths
            ],
        },
        "command": {
            "argv": [
                Path(str(executable)).name,
                *[_redact(str(argument), roots) for argument in arguments],
            ],
            "environment_keys": sorted((environment or {}).keys()),
            "threads": threads,
        },
        "outputs": [],
    }
    if resolved is None:
        manifest["status"] = "skipped"
        manifest["reason"] = "reference executable is not configured or executable"
        _write_json(manifest_file, manifest)
        return manifest

    before = os.times()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(resolved), *map(str, arguments)],
            cwd=workdir,
            env=run_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = "TimeoutExpired"
        manifest["error"] = f"reference stage exceeded {timeout_seconds} seconds"
        manifest["metrics"] = _resource_metrics(before, time.perf_counter() - started)
        _write_json(manifest_file, manifest)
        raise ReferenceRunError(str(manifest["error"]), manifest) from exc

    manifest["metrics"] = _resource_metrics(before, time.perf_counter() - started)
    manifest["returncode"] = completed.returncode
    manifest["stdout_summary"] = _redact(completed.stdout[-4000:], roots)
    manifest["stderr_summary"] = _redact(completed.stderr[-4000:], roots)
    if completed.returncode != 0:
        manifest["status"] = "failed"
        manifest["error_type"] = "ReferenceCommandFailed"
        manifest["error"] = f"reference stage returned {completed.returncode}"
        _write_json(manifest_file, manifest)
        raise ReferenceRunError(str(manifest["error"]), manifest)

    outputs = []
    missing = []
    for artifact in artifacts:
        output = (workdir / artifact.path).resolve()
        try:
            output.relative_to(workdir)
        except ValueError as exc:
            raise ValueError(
                f"Reference artifact escapes working directory: {artifact.path}"
            ) from exc
        if not output.is_file():
            if artifact.required:
                missing.append(artifact.path)
            continue
        outputs.append(
            _file_summary(
                output,
                alias=artifact.path,
                kind=artifact.kind,
                mask=artifact.mask,
                include_digest=include_output_digests,
            )
        )
    manifest["outputs"] = outputs
    if missing:
        manifest["status"] = "failed"
        manifest["error_type"] = "MissingReferenceArtifact"
        manifest["error"] = f"missing required reference outputs: {', '.join(missing)}"
        _write_json(manifest_file, manifest)
        raise ReferenceRunError(str(manifest["error"]), manifest)

    manifest["status"] = "completed"
    _write_json(manifest_file, manifest)
    return manifest
