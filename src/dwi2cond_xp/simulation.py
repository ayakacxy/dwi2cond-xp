"""SimNIBS 4.6 scalar/anisotropic tDCS orchestration and result QA."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import nibabel as nib
import numpy as np


SIMNIBS_REQUIRED_VERSION = "4.6.0"


def _validate_tdcs_parameters(
    *,
    anode: str,
    cathode: str,
    current_ma: float,
    shape: str,
    dimensions: tuple[float, float],
    thickness: float,
    fields: str,
    solver: str,
    volume_tissues: tuple[int, ...],
) -> None:
    """Validate the fixed electrodes and solver parameters before creating the SimNIBS object."""

    if not anode or not cathode or anode == cathode:
        raise ValueError("Anode and cathode must be distinct nonempty positions")
    if (
        not np.isfinite(current_ma)
        or current_ma <= 0
        or not np.isfinite(thickness)
        or thickness <= 0
        or len(dimensions) != 2
        or any(not np.isfinite(value) or value <= 0 for value in dimensions)
    ):
        raise ValueError("Current, electrode dimensions, and thickness must be positive")
    if shape not in {"rect", "ellipse"}:
        raise ValueError("Electrode shape must be rect or ellipse")
    if solver not in {"pardiso", "hypre", "mumps", "petsc_pardiso"}:
        raise ValueError("Unsupported FEM solver")
    if not fields:
        raise ValueError("fields must contain at least one SimNIBS output field")
    if not volume_tissues or any(
        not isinstance(tag, (int, np.integer)) or tag <= 0 for tag in volume_tissues
    ):
        raise ValueError("volume_tissues must contain positive integer tissue labels")


def _discover_subject_files(
    subpath: Path,
) -> tuple[Path, Path, Path, Path | None]:
    """Locate the CHARM mesh, T1, final tissues, and EEG 10-10 file."""
    subject = subpath.name.removeprefix("m2m_")
    head_mesh = subpath / f"{subject}.msh"
    if not head_mesh.is_file():
        candidates = sorted(subpath.glob("*.msh"))
        if len(candidates) == 1:
            head_mesh = candidates[0]
    t1_file = subpath / "T1.nii.gz"
    final_tissues = subpath / "final_tissues.nii.gz"
    # Prefer the official SubjectFiles default cap over an alphabetically selected coordinate set.
    eeg_cap = subpath / "eeg_positions" / "EEG10-10_UI_Jurak_2007.csv"
    if not eeg_cap.is_file():
        eeg_candidates = sorted((subpath / "eeg_positions").glob("*10-10*.csv"))
        eeg_cap = eeg_candidates[0] if eeg_candidates else None
    return head_mesh, t1_file, final_tissues, eeg_cap


def validate_simulation_inputs(
    subpath: str | Path,
    *,
    mode: str,
    tensor_file: str | Path | None,
) -> dict[str, Any]:
    """Validate the head model, tensor, and spatial contract before SESSION creation."""
    if mode not in {"scalar", "vn", "dir", "mc"}:
        raise ValueError("mode must be scalar, vn, dir, or mc")
    subject_path = Path(subpath).resolve()
    if not subject_path.is_dir():
        raise FileNotFoundError(f"m2m directory does not exist: {subject_path}")
    if not subject_path.name.startswith("m2m_") or not subject_path.name[4:]:
        raise ValueError("Simulation requires a CHARM directory named m2m_<subject>")
    head_mesh, t1_file, final_tissues, eeg_cap = _discover_subject_files(subject_path)
    required: dict[str, Path | None] = {
        "head_mesh": head_mesh,
        "T1": t1_file,
        "final_tissues": final_tissues,
        "EEG_10_10": eeg_cap,
    }
    missing = [
        f"{name}: {path if path is not None else 'not found'}"
        for name, path in required.items()
        if path is None or not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("The CHARM head model is incomplete:\n" + "\n".join(missing))

    t1_img = nib.load(str(t1_file))
    tissues_img = nib.load(str(final_tissues))
    tissues_shape = tissues_img.shape
    if len(tissues_shape) == 4 and tissues_shape[-1] == 1:
        tissues_shape = tissues_shape[:3]
    if len(t1_img.shape) != 3 or tissues_shape != t1_img.shape or not np.allclose(
        tissues_img.affine, t1_img.affine, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError("T1 and final_tissues must share one three-dimensional grid")

    tensor_path: Path | None = None
    tensor_contract: dict[str, Any] | None = None
    if mode == "scalar" and tensor_file is not None:
        raise ValueError("tensor_file must not be provided for scalar mode")
    if mode != "scalar":
        if tensor_file is None:
            # Match the SimNIBS 4.6 dwi2cond subject-directory contract.
            official_tensor = subject_path / "DTI_coregT1_tensor.nii.gz"
            if not official_tensor.is_file():
                raise ValueError(
                    f"No tensor_file was provided for {mode} and the official path "
                    f"does not exist: {official_tensor}; scalar fallback is forbidden"
                )
            tensor_path = official_tensor
        else:
            tensor_path = Path(tensor_file).resolve()
        if not tensor_path.is_file():
            raise FileNotFoundError(f"Tensor file does not exist: {tensor_path}")
        tensor_img = nib.load(str(tensor_path))
        if tensor_img.shape != t1_img.shape[:3] + (6,):
            raise ValueError("Tensor shape must equal the CHARM T1 shape plus six components")
        if not np.allclose(tensor_img.affine, t1_img.affine):
            raise ValueError("Tensor affine must match the CHARM T1 affine")
        tensor_values = np.asanyarray(tensor_img.dataobj)
        if not np.all(np.isfinite(tensor_values)):
            raise ValueError("Tensor data contains NaN or Inf")
        tensor_contract = {
            "path": str(tensor_path),
            "shape": list(tensor_img.shape),
            "affine": tensor_img.affine.tolist(),
            "component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
            "finite_components_checked": int(tensor_values.size),
        }
    return {
        "subpath": str(subject_path),
        "head_mesh": str(head_mesh),
        "T1": str(t1_file),
        "final_tissues": str(final_tissues),
        "eeg_cap": str(eeg_cap),
        "mode": mode,
        "tensor": tensor_contract,
        "grid": {
            "shape": list(t1_img.shape),
            "affine": t1_img.affine.tolist(),
        },
    }


def build_tdcs_session(
    input_contract: dict[str, Any],
    output_directory: str | Path,
    *,
    anode: str = "C3",
    cathode: str = "C4",
    current_ma: float = 1.0,
    shape: str = "rect",
    dimensions: tuple[float, float] = (50.0, 50.0),
    thickness: float = 4.0,
    fields: str = "E",
    solver: str = "pardiso",
    volume_tissues: tuple[int, ...] = (1, 2, 3),
):
    """Construct a fixed-montage SimNIBS 4.6 SESSION without solving."""
    _validate_tdcs_parameters(
        anode=anode,
        cathode=cathode,
        current_ma=current_ma,
        shape=shape,
        dimensions=dimensions,
        thickness=thickness,
        fields=fields,
        solver=solver,
        volume_tissues=volume_tissues,
    )
    try:
        import simnibs
        from simnibs import sim_struct
    except ImportError as exc:
        raise RuntimeError("Simulation requires simnibs==4.6.0") from exc
    if simnibs.__version__ != SIMNIBS_REQUIRED_VERSION:
        raise RuntimeError(
            f"SimNIBS {SIMNIBS_REQUIRED_VERSION} is required; found {simnibs.__version__}"
        )
    session = sim_struct.SESSION()
    session.subpath = input_contract["subpath"]
    session.fnamehead = input_contract["head_mesh"]
    session.pathfem = str(Path(output_directory).resolve())
    session.open_in_gmsh = False
    session.map_to_surf = False
    session.map_to_fsavg = False
    # README figures use T1-grid voxel NIfTI and do not depend on mesh screenshots.
    session.map_to_vol = True
    session.map_to_MNI = False
    # SimNIBS defaults to GM only; this contract keeps WM/GM/CSF and excludes outer tissues.
    session.tissues_in_niftis = list(volume_tissues)
    session.fields = fields
    session.eeg_cap = input_contract["eeg_cap"]
    if input_contract["tensor"] is not None:
        session.fname_tensor = input_contract["tensor"]["path"]

    tdcs = session.add_tdcslist()
    current_ampere = current_ma * 1e-3
    tdcs.currents = [current_ampere, -current_ampere]
    tdcs.anisotropy_type = input_contract["mode"]
    tdcs.solver_options = solver

    for channel, centre in ((1, anode), (2, cathode)):
        electrode = tdcs.add_electrode()
        electrode.channelnr = channel
        electrode.centre = centre
        electrode.shape = shape
        electrode.dimensions = list(dimensions)
        electrode.thickness = thickness
    return session


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a simulation manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    """Recursively convert SimNIBS return values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _rebase_path_strings(value: Any, source: Path, destination: Path) -> Any:
    """将 attempt 目录中的路径重写为正式发布目录。"""

    if isinstance(value, dict):
        return {
            str(key): _rebase_path_strings(item, source, destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebase_path_strings(item, source, destination) for item in value]
    if isinstance(value, str):
        source_prefix = str(source.resolve())
        destination_prefix = str(destination.resolve())
        if value == source_prefix:
            return destination_prefix
        if value.startswith(source_prefix + os.sep):
            return destination_prefix + value[len(source_prefix) :]
    return value


def _sha256_file(path: Path) -> str:
    """Hash one simulation artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_simulation_outputs(
    output_directory: str | Path,
    *,
    manifest_path: str | Path,
) -> list[dict[str, Any]]:
    """Record every dynamic FEM artifact needed for cache validation."""

    output = Path(output_directory).resolve()
    manifest = Path(manifest_path).resolve()
    artifacts: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        resolved = path.resolve()
        if resolved == manifest or path.name.endswith(".tmp"):
            continue
        entry: dict[str, Any] = {
            "path": str(resolved),
            "relative_path": resolved.relative_to(output).as_posix(),
            "type": "file",
            "bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
        if resolved.name.endswith((".nii", ".nii.gz")):
            image = nib.load(str(resolved))
            entry.update(
                {
                    "type": "nifti",
                    "shape": list(image.shape),
                    "affine": image.affine.tolist(),
                    "dtype": str(image.get_data_dtype()),
                }
            )
        artifacts.append(entry)
    return artifacts


def mask_subject_volume_outputs(
    output_directory: str | Path,
    final_tissues_file: str | Path,
    volume_tissues: tuple[int, ...],
) -> list[str]:
    """Strictly zero subject-volume NIfTI outside the requested tissues."""
    output_path = Path(output_directory).resolve() / "subject_volumes"
    if not output_path.is_dir():
        return []
    tissues_img = nib.load(str(final_tissues_file))
    tissues = np.asanyarray(tissues_img.dataobj)
    if tissues.ndim == 4 and tissues.shape[-1] == 1:
        tissues = tissues[..., 0]
    if tissues.ndim != 3:
        raise ValueError("final_tissues must be a three-dimensional label image")
    keep_mask = np.isin(tissues, volume_tissues)
    masked_outputs: list[str] = []
    for nifti_path in sorted(output_path.glob("*.nii.gz")):
        image = nib.load(str(nifti_path))
        if image.shape[:3] != tissues.shape or not np.allclose(
            image.affine, tissues_img.affine
        ):
            raise ValueError(f"Subject volume does not match the final_tissues grid: {nifti_path}")
        data = np.asanyarray(image.dataobj).copy()
        data[~keep_mask] = 0
        # Write beside the target and replace atomically to avoid partial gzip files.
        temporary = nifti_path.with_name(
            nifti_path.name.removesuffix(".nii.gz") + ".tmp.nii.gz"
        )
        nib.save(nib.Nifti1Image(data, image.affine, image.header), str(temporary))
        temporary.replace(nifti_path)
        masked_outputs.append(str(nifti_path))
    return masked_outputs


def run_tdcs(
    subpath: str | Path,
    output_root: str | Path,
    *,
    mode: str,
    tensor_file: str | Path | None = None,
    anode: str = "C3",
    cathode: str = "C4",
    current_ma: float = 1.0,
    shape: str = "rect",
    dimensions: tuple[float, float] = (50.0, 50.0),
    thickness: float = 4.0,
    fields: str = "E",
    solver: str = "pardiso",
    volume_tissues: tuple[int, ...] = (1, 2, 3),
    cpus: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate, record, and optionally execute one tDCS conductivity mode."""
    if cpus <= 0:
        raise ValueError("cpus must be a positive integer")
    _validate_tdcs_parameters(
        anode=anode,
        cathode=cathode,
        current_ma=current_ma,
        shape=shape,
        dimensions=dimensions,
        thickness=thickness,
        fields=fields,
        solver=solver,
        volume_tissues=volume_tissues,
    )
    contract = validate_simulation_inputs(subpath, mode=mode, tensor_file=tensor_file)
    root = Path(output_root).resolve()
    output_directory = root / "dry-run" / mode if dry_run else root / mode
    attempt_id = uuid4().hex
    attempt_directory = output_directory.with_name(
        f".{output_directory.name}.attempt-{attempt_id}"
    )
    failure_directory = output_directory.with_name(
        f".{output_directory.name}.failed-{attempt_id}"
    )
    backup_directory = output_directory.with_name(
        f".{output_directory.name}.previous-{attempt_id}"
    )
    attempt_directory.mkdir(parents=True, exist_ok=False)
    active_directory = attempt_directory
    manifest_path = active_directory / "dwi2cond_xp_simulation.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if dry_run else "running",
        "required_simnibs_version": SIMNIBS_REQUIRED_VERSION,
        "input": contract,
        "montage": {
            "anode": anode,
            "cathode": cathode,
            "current_ma": current_ma,
            "shape": shape,
            "dimensions_mm": list(dimensions),
            "thickness_mm": thickness,
        },
        "fields": fields,
        "solver": solver,
        "volume_tissues": list(volume_tissues),
        "map_to_subject_volume": True,
        "cpus": cpus,
        "output_directory": str(attempt_directory),
        "attempt_id": attempt_id,
        "attempt_directory": str(attempt_directory),
        "outputs": [],
        "masked_subject_volumes": [],
        "artifacts": [],
    }
    phase = "write_initial_manifest"
    try:
        _write_manifest(manifest_path, manifest)
        if not dry_run:
            phase = "build_session"
            session = build_tdcs_session(
                contract,
                attempt_directory,
                anode=anode,
                cathode=cathode,
                current_ma=current_ma,
                shape=shape,
                dimensions=dimensions,
                thickness=thickness,
                fields=fields,
                solver=solver,
                volume_tissues=volume_tissues,
            )
            phase = "solve"
            from simnibs import run_simnibs

            outputs = run_simnibs(session, cpus=cpus)
            phase = "postprocess_subject_volumes"
            masked_subject_volumes = mask_subject_volume_outputs(
                attempt_directory,
                contract["final_tissues"],
                volume_tissues,
            )
            if not masked_subject_volumes:
                raise ValueError("SimNIBS produced no subject-volume output")
            phase = "inventory_outputs"
            artifacts = inventory_simulation_outputs(
                attempt_directory, manifest_path=manifest_path
            )
            phase = "serialize_outputs"
            safe_outputs = _json_safe(outputs)
            phase = "write_completed_manifest"
            manifest["status"] = "completed"
            manifest["outputs"] = safe_outputs
            manifest["masked_subject_volumes"] = masked_subject_volumes
            manifest["artifacts"] = artifacts
            _write_manifest(manifest_path, manifest)
        phase = "validate_attempt_manifest"
        from .preprocessing.pipeline import ArtifactContract, validate_artifacts

        validate_artifacts(
            (
                ArtifactContract(
                    manifest_path, "json", dynamic_inventory=True
                ),
            )
        )
        phase = "rebase_published_paths"
        manifest = _rebase_path_strings(
            manifest, attempt_directory, output_directory
        )
        _write_manifest(manifest_path, manifest)
        phase = "publish_attempt"
        if output_directory.exists():
            output_directory.replace(backup_directory)
        attempt_directory.replace(output_directory)
        active_directory = output_directory
        manifest_path = output_directory / "dwi2cond_xp_simulation.json"
        phase = "validate_published_manifest"
        validate_artifacts(
            (
                ArtifactContract(
                    manifest_path, "json", dynamic_inventory=True
                ),
            )
        )
        phase = "remove_previous_output"
        if backup_directory.exists():
            shutil.rmtree(backup_directory)
    except Exception as exc:
        failed_source_directory = active_directory
        if active_directory.exists():
            active_directory.replace(failure_directory)
        if backup_directory.exists() and not output_directory.exists():
            backup_directory.replace(output_directory)
        manifest = _rebase_path_strings(
            manifest, failed_source_directory, failure_directory
        )
        manifest_path = failure_directory / "dwi2cond_xp_simulation.json"
        manifest["output_directory"] = str(failure_directory)
        manifest["status"] = "failed"
        manifest["failed_phase"] = phase
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["failed_attempt_directory"] = str(failure_directory)
        manifest["failure_manifest"] = str(manifest_path)
        _write_manifest(manifest_path, manifest)
        setattr(exc, "dwi2cond_xp_failure_manifest", str(manifest_path))
        setattr(exc, "dwi2cond_xp_failed_phase", phase)
        raise
    return manifest
