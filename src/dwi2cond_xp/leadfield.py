"""SimNIBS 4.6 all-electrode tDCS lead-field orchestration and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import h5py
import numpy as np

from .simulation import (
    SIMNIBS_REQUIRED_VERSION,
    _json_safe,
    _write_manifest,
    validate_simulation_inputs,
)


LEADFIELD_DATASET = "mesh_leadfield/leadfields/tdcs_leadfield"
ProgressCallback = Callable[[int, int, str], None]


def _decode_text(value: Any) -> str:
    """Convert HDF5 bytes or NumPy scalars to strings consistently."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def build_tdcs_leadfield(
    input_contract: dict[str, Any],
    output_directory: str | Path,
    *,
    eeg_cap: str | Path | None = None,
    field: str = "E",
    interpolation: str = "none",
    tissues: Sequence[int] = (1, 2),
    interpolation_tissues: Sequence[int] = (2,),
    shape: str = "ellipse",
    dimensions: tuple[float, float] = (10.0, 10.0),
    thickness: float = 4.0,
    solver: str = "pardiso",
):
    """Construct an all-electrode TDCSLEADFIELD without solving."""
    try:
        import simnibs
        from simnibs import sim_struct
    except ImportError as exc:
        raise RuntimeError("Lead-field simulation requires simnibs==4.6.0") from exc
    if simnibs.__version__ != SIMNIBS_REQUIRED_VERSION:
        raise RuntimeError(
            f"SimNIBS {SIMNIBS_REQUIRED_VERSION} is required; found {simnibs.__version__}"
        )
    if field not in {"E", "J"}:
        raise ValueError("field must be E or J")
    if interpolation not in {"none", "middle-gm"}:
        raise ValueError("interpolation must be none or middle-gm")
    if shape not in {"ellipse", "rect"}:
        raise ValueError("Electrode shape must be ellipse or rect")
    if any(value <= 0 for value in dimensions) or thickness <= 0:
        raise ValueError("Electrode dimensions and thickness must be positive")
    if solver not in {"default", "pardiso"}:
        raise ValueError("solver must be default or pardiso")
    if interpolation == "none" and not tissues:
        raise ValueError("A volume lead field requires at least one tissue")
    if interpolation == "middle-gm" and not interpolation_tissues:
        raise ValueError("middle-gm requires interpolation_tissues")

    leadfield = sim_struct.TDCSLEADFIELD()
    leadfield.fnamehead = input_contract["head_mesh"]
    leadfield.subpath = input_contract["subpath"]
    leadfield.pathfem = str(Path(output_directory).resolve())
    leadfield.eeg_cap = str(eeg_cap or input_contract["eeg_cap"])
    leadfield.field = field
    leadfield.anisotropy_type = input_contract["mode"]
    if input_contract["tensor"] is not None:
        leadfield.fname_tensor = input_contract["tensor"]["path"]

    if interpolation == "none":
        leadfield.interpolation = None
        leadfield.tissues = [int(tag) for tag in tissues]
    else:
        leadfield.interpolation = "middle gm"
        # Volume tissues must not be mixed into the surface-interpolation output.
        leadfield.tissues = []
        leadfield.interpolation_tissue = [
            int(tag) for tag in interpolation_tissues
        ]

    leadfield.electrode.shape = shape
    leadfield.electrode.dimensions = list(dimensions)
    leadfield.electrode.thickness = [thickness]
    leadfield.solver_options = None if solver == "default" else "pardiso"
    return leadfield


def _read_axis_contract(handle: h5py.File) -> dict[str, Any]:
    """Read and validate lead-field axes, reference, and spatial grain."""
    if LEADFIELD_DATASET not in handle:
        raise KeyError(f"HDF5 is missing dataset: {LEADFIELD_DATASET}")
    dataset = handle[LEADFIELD_DATASET]
    if dataset.ndim != 3 or dataset.shape[-1] != 3:
        raise ValueError("Lead-field shape must be (N_basis, N_spatial, 3)")
    names = [_decode_text(value) for value in dataset.attrs["electrode_names"]]
    reference = _decode_text(dataset.attrs["reference_electrode"])
    if len(names) != dataset.shape[0] + 1:
        raise ValueError("electrode_names count must equal N_basis + 1")
    if names[0] != reference:
        raise ValueError("reference_electrode must be the first electrode_names entry")
    d_type = _decode_text(dataset.attrs.get("d_type", "unknown"))
    if d_type not in {"element_data", "node_data"}:
        raise ValueError(f"Unsupported lead-field d_type: {d_type}")

    if d_type == "element_data":
        spatial_count = handle["mesh_leadfield/elm/tag1"].shape[0]
    else:
        spatial_count = handle["mesh_leadfield/nodes/node_coord"].shape[0]
    if spatial_count != dataset.shape[1]:
        raise ValueError("The HDF5 mesh spatial count does not match the lead-field axis")
    return {
        "dataset": LEADFIELD_DATASET,
        "shape": list(dataset.shape),
        "axis_order": ["basis_electrode", "spatial", "world_xyz"],
        "matrix_layout": "spatial_xyz_by_basis",
        "reference_electrode": reference,
        "active_electrodes": names[1:],
        "all_electrodes": names,
        "field": _decode_text(dataset.attrs.get("field", "unknown")),
        "units": _decode_text(dataset.attrs.get("units", "unknown")),
        "basis_current": _decode_text(dataset.attrs.get("current", "unknown")),
        "d_type": d_type,
        "interpolation": _decode_text(
            dataset.attrs.get("interpolation", "unknown")
        ),
    }


def validate_and_export_leadfield(
    hdf5_file: str | Path,
    *,
    matrix_file: str | Path | None = None,
    roi_labels: Sequence[int] = (),
    avoid_labels: Sequence[int] = (),
    roi_mask_file: str | Path | None = None,
    avoid_mask_file: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Validate each basis and optionally export a 2-D matrix and immutable ROI masks."""
    hdf_path = Path(hdf5_file).resolve()
    temporary_matrix: Path | None = None
    matrix: np.memmap | None = None
    with h5py.File(hdf_path, "r") as handle:
        report = _read_axis_contract(handle)
        dataset = handle[LEADFIELD_DATASET]
        n_basis, n_spatial, _ = dataset.shape
        mask_outputs: list[tuple[Path, np.ndarray, str, list[int]]] = []
        if roi_labels or avoid_labels:
            if report["d_type"] != "element_data":
                raise ValueError("Tissue-tag ROI export supports element_data only")
            tags = np.asarray(handle["mesh_leadfield/elm/tag1"]).reshape(-1)
            available_tags = set(np.unique(tags))
            for labels, destination, key in (
                (roi_labels, roi_mask_file, "roi_mask_file"),
                (avoid_labels, avoid_mask_file, "avoid_mask_file"),
            ):
                if not labels:
                    continue
                normalized_labels = [int(value) for value in labels]
                missing = sorted(set(normalized_labels) - available_tags)
                if missing:
                    raise ValueError(f"Lead-field space is missing tissue tags: {missing}")
                if destination is None:
                    raise ValueError(f"{key} was not specified")
                destination_path = Path(destination).resolve()
                mask_outputs.append(
                    (
                        destination_path,
                        np.isin(tags, normalized_labels),
                        key,
                        normalized_labels,
                    )
                )
        if matrix_file is not None:
            matrix_path = Path(matrix_file).resolve()
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_matrix = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
            matrix = np.lib.format.open_memmap(
                temporary_matrix,
                mode="w+",
                dtype=np.float64,
                shape=(n_spatial * 3, n_basis),
            )

        minimum = np.inf
        maximum = -np.inf
        squared_sum = 0.0
        value_count = 0
        try:
            for basis in range(n_basis):
                values = np.asarray(dataset[basis], dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Lead-field basis {basis} contains NaN/Inf")
                minimum = min(minimum, float(np.min(values)))
                maximum = max(maximum, float(np.max(values)))
                squared_sum += float(np.sum(values * values, dtype=np.float64))
                value_count += values.size
                if matrix is not None:
                    matrix[:, basis] = values.reshape(-1)
                if progress is not None:
                    progress(basis + 1, n_basis, "validate_export")
        except Exception:
            if matrix is not None:
                del matrix
            if temporary_matrix is not None and temporary_matrix.exists():
                temporary_matrix.unlink()
            raise

        if matrix is not None and temporary_matrix is not None:
            matrix.flush()
            del matrix
            matrix_path = Path(matrix_file).resolve()
            temporary_matrix.replace(matrix_path)
            report["matrix_file"] = str(matrix_path)
            report["matrix_shape"] = [n_spatial * 3, n_basis]
            report["flatten_order"] = "spatial-major then world x/y/z"

        report["finite"] = True
        report["minimum"] = minimum
        report["maximum"] = maximum
        report["root_mean_square"] = float(np.sqrt(squared_sum / value_count))

        for destination_path, mask, key, labels in mask_outputs:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination_path, mask)
            report[key] = str(destination_path)
            report[key.replace("_file", "_labels")] = labels
    return report


def run_tdcs_leadfield(
    subpath: str | Path,
    output_root: str | Path,
    *,
    mode: str,
    tensor_file: str | Path | None = None,
    eeg_cap: str | Path | None = None,
    field: str = "E",
    interpolation: str = "none",
    tissues: Sequence[int] = (1, 2),
    interpolation_tissues: Sequence[int] = (2,),
    shape: str = "ellipse",
    dimensions: tuple[float, float] = (10.0, 10.0),
    thickness: float = 4.0,
    solver: str = "pardiso",
    cpus: int = 1,
    export_matrix: bool = True,
    roi_labels: Sequence[int] = (),
    avoid_labels: Sequence[int] = (),
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Validate and optionally execute one all-electrode conductivity mode."""
    if cpus <= 0:
        raise ValueError("cpus must be a positive integer")
    requested_masks = set(int(tag) for tag in roi_labels) | set(
        int(tag) for tag in avoid_labels
    )
    if interpolation != "none" and requested_masks:
        raise ValueError("middle-gm output cannot export ROI/avoid masks by volume tissue tag")
    missing_mask_tissues = requested_masks - set(int(tag) for tag in tissues)
    if missing_mask_tissues:
        raise ValueError(
            "ROI/avoid labels must be included in --tissues: "
            f"{sorted(missing_mask_tissues)}"
        )
    contract = validate_simulation_inputs(subpath, mode=mode, tensor_file=tensor_file)
    if eeg_cap is not None:
        eeg_path = Path(eeg_cap).resolve()
        if not eeg_path.is_file():
            raise FileNotFoundError(f"EEG cap does not exist: {eeg_path}")
    else:
        eeg_path = Path(contract["eeg_cap"])

    output_directory = Path(output_root).resolve() / mode
    manifest_path = output_directory / "dwi2cond_xp_leadfield.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if dry_run else "running",
        "required_simnibs_version": SIMNIBS_REQUIRED_VERSION,
        "input": contract,
        "leadfield": {
            "field": field,
            "eeg_cap": str(eeg_path),
            "interpolation": interpolation,
            "tissues": [int(tag) for tag in tissues],
            "interpolation_tissues": [int(tag) for tag in interpolation_tissues],
            "electrode_shape": shape,
            "dimensions_mm": list(dimensions),
            "thickness_mm": thickness,
            "solver": solver,
            "cpus": cpus,
            "basis_current": "1A",
            "export_npy": export_matrix,
            "npy_layout": "spatial_xyz_by_basis",
            "roi_labels": [int(tag) for tag in roi_labels],
            "avoid_labels": [int(tag) for tag in avoid_labels],
        },
        "output_directory": str(output_directory),
    }
    if not dry_run:
        existing = sorted(output_directory.glob("*.hdf5"))
        if existing:
            raise FileExistsError(
                "The output directory already contains HDF5; use a new output_root: "
                + ", ".join(str(path) for path in existing)
            )
    _write_manifest(manifest_path, manifest)
    if dry_run:
        return manifest

    try:
        leadfield = build_tdcs_leadfield(
            contract,
            output_directory,
            eeg_cap=eeg_path,
            field=field,
            interpolation=interpolation,
            tissues=tissues,
            interpolation_tissues=interpolation_tissues,
            shape=shape,
            dimensions=dimensions,
            thickness=thickness,
            solver=solver,
        )
        from simnibs import run_simnibs

        run_simnibs(leadfield, cpus=cpus)
        hdf5_candidates = sorted(output_directory.glob("*.hdf5"))
        if len(hdf5_candidates) != 1:
            raise RuntimeError(
                f"Expected one lead-field HDF5; found {len(hdf5_candidates)}"
            )
        matrix_file = output_directory / "leadfield_spatial_xyz_by_basis.npy"
        roi_mask_file = output_directory / "roi_mask.npy"
        avoid_mask_file = output_directory / "avoid_mask.npy"
        qa = validate_and_export_leadfield(
            hdf5_candidates[0],
            matrix_file=matrix_file if export_matrix else None,
            roi_labels=roi_labels,
            avoid_labels=avoid_labels,
            roi_mask_file=roi_mask_file if roi_labels else None,
            avoid_mask_file=avoid_mask_file if avoid_labels else None,
            progress=progress,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        _write_manifest(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    manifest["hdf5"] = str(hdf5_candidates[0])
    manifest["qa"] = _json_safe(qa)
    _write_manifest(manifest_path, manifest)
    return manifest
