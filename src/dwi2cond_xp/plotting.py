"""Create auditable electric-field comparisons from voxel-level NIfTI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np


MODE_ORDER = ("scalar", "vn", "dir", "mc")
MODE_TITLES = {
    "scalar": "Scalar",
    "vn": "VN · volume normalized",
    "dir": "DIR · direct",
    "mc": "MC · mean conductivity",
}
PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
ORIENTATION_LABELS = {
    "axial": ("L", "R", "P", "A"),
    "coronal": ("L", "R", "I", "S"),
    "sagittal": ("P", "A", "I", "S"),
}


def _load_canonical(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI in the nearest RAS+ orientation and return float data and affine."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        # CHARM 4.6 may write final_tissues with a singleton final axis.
        data = data[..., 0]
    return data, np.asarray(image.affine, dtype=np.float64)


def _scalar_field(data: np.ndarray, path: str | Path) -> np.ndarray:
    """Accept a 3-D scalar field or convert a three-component vector to magnitude."""
    if data.ndim == 3:
        return data
    if data.ndim == 4 and data.shape[-1] == 3:
        return np.linalg.norm(data, axis=-1)
    raise ValueError(f"Field NIfTI must be 3-D scalar or end in three vector components: {path}")


def _take_slice(data: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Extract and transpose a slice so the second RAS+ display axis points up."""
    return np.take(data, index, axis=PLANE_AXIS[plane]).T


def _automatic_slice(mask: np.ndarray, plane: str) -> int:
    """Select the largest brain-mask slice independently of field values."""
    axes = tuple(axis for axis in range(3) if axis != PLANE_AXIS[plane])
    areas = np.count_nonzero(mask, axis=axes)
    if not np.any(areas):
        raise ValueError("The brain mask is empty; a slice cannot be selected")
    return int(np.argmax(areas))


def _robust_limits(anatomy: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Compute a stable grayscale display range for T1."""
    values = anatomy[np.isfinite(anatomy) & mask]
    if values.size == 0:
        values = anatomy[np.isfinite(anatomy)]
    if values.size == 0:
        raise ValueError("T1 contains no finite values")
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _draw_panel(
    axis,
    anatomy_slice: np.ndarray,
    field_slice: np.ndarray,
    mask_slice: np.ndarray,
    *,
    title: str,
    plane: str,
    anatomy_limits: tuple[float, float],
    vmax: float,
):
    """Plot one T1 and electric-field overlay using a shared scale."""
    axis.imshow(
        anatomy_slice,
        cmap="gray",
        origin="lower",
        vmin=anatomy_limits[0],
        vmax=anatomy_limits[1],
        interpolation="nearest",
    )
    overlay = np.ma.masked_where(~mask_slice | ~np.isfinite(field_slice), field_slice)
    image = axis.imshow(
        overlay,
        cmap="magma",
        origin="lower",
        vmin=0.0,
        vmax=vmax,
        alpha=0.82,
        interpolation="bilinear",
    )
    if np.any(mask_slice) and np.any(~mask_slice):
        axis.contour(mask_slice.astype(float), levels=[0.5], colors="#f5f5f5", linewidths=0.45)
    left, right, bottom, top = ORIENTATION_LABELS[plane]
    label_style = dict(color="white", fontsize=8, fontweight="bold", alpha=0.9)
    axis.text(0.012, 0.5, left, transform=axis.transAxes, ha="left", va="center", **label_style)
    axis.text(0.988, 0.5, right, transform=axis.transAxes, ha="right", va="center", **label_style)
    axis.text(0.5, 0.012, bottom, transform=axis.transAxes, ha="center", va="bottom", **label_style)
    axis.text(0.5, 0.988, top, transform=axis.transAxes, ha="center", va="top", **label_style)
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=7)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def _write_report(path: Path, report: dict) -> None:
    """Write plotting parameters and voxel statistics atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _plot_magnitude_comparison(
    field_files: Mapping[str, str | Path],
    anatomy_file: str | Path,
    mask_file: str | Path,
    output_file: str | Path,
    *,
    plane: str = "axial",
    slice_index: int | None = None,
    mask_labels: tuple[int, ...] = (1, 2, 3),
    vmax: float | None = None,
    percentile: float = 99.5,
    dpi: int = 220,
    panels_directory: str | Path | None = None,
) -> dict:
    """Export four PNGs, a 2x2 composite, and shared-scale QA JSON.

    Inputs must share one voxel grid. Automatic slice selection depends only on
    mask area. The shared scale uses the pooled masked distribution across all
    modes so per-panel scaling cannot exaggerate differences.
    """
    if set(field_files) != set(MODE_ORDER):
        raise ValueError(f"field_files must contain exactly: {', '.join(MODE_ORDER)}")
    if plane not in PLANE_AXIS:
        raise ValueError("plane must be axial, coronal, or sagittal")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    anatomy, reference_affine = _load_canonical(anatomy_file)
    mask_data, mask_affine = _load_canonical(mask_file)
    if anatomy.ndim != 3 or mask_data.shape != anatomy.shape:
        raise ValueError("T1 and mask must be three-dimensional NIfTIs with equal shape")
    if not np.allclose(mask_affine, reference_affine):
        raise ValueError("The mask affine must match the T1 affine")
    mask = np.isin(np.rint(mask_data).astype(np.int32), mask_labels)
    if not np.any(mask):
        raise ValueError(f"The mask contains none of labels {mask_labels}")

    fields: dict[str, np.ndarray] = {}
    for mode in MODE_ORDER:
        data, affine = _load_canonical(field_files[mode])
        field = _scalar_field(data, field_files[mode])
        if field.shape != anatomy.shape or not np.allclose(affine, reference_affine):
            raise ValueError(f"{mode} field shape/affine must match T1 exactly")
        finite_masked = field[np.isfinite(field) & mask]
        if finite_masked.size == 0:
            raise ValueError(f"{mode} field has no finite voxel inside the brain mask")
        tolerance = max(float(np.max(np.abs(finite_masked))) * 1e-7, 1e-12)
        if float(np.min(finite_masked)) < -tolerance:
            raise ValueError(f"{mode} contains negative values and is not a field magnitude")
        fields[mode] = np.maximum(field, 0.0)

    axis_number = PLANE_AXIS[plane]
    if slice_index is None:
        slice_index = _automatic_slice(mask, plane)
        slice_selection = "maximum_mask_area"
    else:
        if not 0 <= slice_index < anatomy.shape[axis_number]:
            raise ValueError("slice_index is outside the selected plane")
        slice_selection = "explicit"

    combined = np.concatenate([fields[mode][mask] for mode in MODE_ORDER])
    combined = combined[np.isfinite(combined)]
    if vmax is None:
        vmax = float(np.percentile(combined, percentile))
        scale_selection = f"combined_masked_p{percentile:g}"
    else:
        vmax = float(vmax)
        scale_selection = "explicit"
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Shared vmax must be finite and positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path = (
        Path(panels_directory).resolve()
        if panels_directory is not None
        else output_path.parent / f"{output_path.stem}_panels"
    )
    panel_path.mkdir(parents=True, exist_ok=True)
    anatomy_slice = _take_slice(anatomy, plane, slice_index)
    mask_slice = _take_slice(mask, plane, slice_index)
    anatomy_limits = _robust_limits(anatomy, mask)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#20242b",
            "axes.titlecolor": "#20242b",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 9.4), constrained_layout=True)
    images = []
    for axis, mode in zip(axes.flat, MODE_ORDER, strict=True):
        images.append(
            _draw_panel(
                axis,
                anatomy_slice,
                _take_slice(fields[mode], plane, slice_index),
                mask_slice,
                title=MODE_TITLES[mode],
                plane=plane,
                anatomy_limits=anatomy_limits,
                vmax=vmax,
            )
        )
    figure.suptitle(
        "Electric-field magnitude across conductivity models",
        fontsize=15,
        fontweight="semibold",
    )
    colorbar = figure.colorbar(
        images[0],
        ax=axes,
        location="bottom",
        shrink=0.72,
        pad=0.025,
        aspect=35,
    )
    colorbar.set_label("|E| (V/m) · shared scale", fontsize=10)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)

    panel_outputs: dict[str, str] = {}
    for mode in MODE_ORDER:
        panel_figure, panel_axis = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
        panel_image = _draw_panel(
            panel_axis,
            anatomy_slice,
            _take_slice(fields[mode], plane, slice_index),
            mask_slice,
            title=MODE_TITLES[mode],
            plane=plane,
            anatomy_limits=anatomy_limits,
            vmax=vmax,
        )
        panel_colorbar = panel_figure.colorbar(
            panel_image,
            ax=panel_axis,
            location="bottom",
            shrink=0.78,
            pad=0.025,
        )
        panel_colorbar.set_label("|E| (V/m) · shared scale", fontsize=9)
        destination = panel_path / f"{mode}.png"
        panel_figure.savefig(destination, dpi=dpi, facecolor="white")
        plt.close(panel_figure)
        panel_outputs[mode] = str(destination)

    slice_voxel = np.eye(3, dtype=float)[axis_number] * slice_index
    world_coordinate = float(
        nib.affines.apply_affine(reference_affine, slice_voxel)[axis_number]
    )
    report = {
        "schema_version": 1,
        "view": "magnitude_2x2",
        "output": str(output_path),
        "panels": panel_outputs,
        "plane": plane,
        "slice_index": slice_index,
        "slice_world_coordinate_mm": world_coordinate,
        "slice_selection": slice_selection,
        "mask_file": str(Path(mask_file).resolve()),
        "mask_labels": list(mask_labels),
        "masked_voxels": int(np.count_nonzero(mask)),
        "vmin_v_per_m": 0.0,
        "vmax_v_per_m": vmax,
        "scale_selection": scale_selection,
        "fields": {
            mode: {
                "file": str(Path(field_files[mode]).resolve()),
                "masked_mean_v_per_m": float(np.mean(fields[mode][mask])),
                "masked_p99_v_per_m": float(np.percentile(fields[mode][mask], 99.0)),
                "masked_max_v_per_m": float(np.max(fields[mode][mask])),
            }
            for mode in MODE_ORDER
        },
    }
    _write_report(output_path.with_suffix(".json"), report)
    return report


def _plot_component_comparison(
    field_files: Mapping[str, str | Path],
    anatomy_file: str | Path,
    mask_file: str | Path,
    output_file: str | Path,
    *,
    plane: str,
    slice_index: int | None,
    mask_labels: tuple[int, ...],
    vmax: float | None,
    percentile: float,
    dpi: int,
    panels_directory: str | Path | None,
) -> dict:
    """Create a 3x4 comparison of three world components by four modes."""
    if plane != "axial":
        raise ValueError("The 3x4 Ex/Ey/Ez figure currently supports axial view only")
    if set(field_files) != set(MODE_ORDER):
        raise ValueError(f"field_files must contain exactly: {', '.join(MODE_ORDER)}")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    anatomy, reference_affine = _load_canonical(anatomy_file)
    mask_data, mask_affine = _load_canonical(mask_file)
    if anatomy.ndim != 3 or mask_data.shape != anatomy.shape:
        raise ValueError("T1 and mask must be three-dimensional NIfTIs with equal shape")
    if not np.allclose(mask_affine, reference_affine):
        raise ValueError("The mask affine must match the T1 affine")
    mask = np.isin(np.rint(mask_data).astype(np.int32), mask_labels)
    if not np.any(mask):
        raise ValueError(f"The mask contains none of labels {mask_labels}")

    vectors: dict[str, np.ndarray] = {}
    for mode in MODE_ORDER:
        data, affine = _load_canonical(field_files[mode])
        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(f"components view requires {mode} to end in three vector components")
        if data.shape[:3] != anatomy.shape or not np.allclose(affine, reference_affine):
            raise ValueError(f"{mode} field shape/affine must match T1 exactly")
        if not np.all(np.isfinite(data[mask])):
            raise ValueError(f"{mode} vector field contains NaN/Inf inside the brain mask")
        vectors[mode] = data

    if slice_index is None:
        slice_index = _automatic_slice(mask, plane)
        slice_selection = "maximum_mask_area"
    else:
        if not 0 <= slice_index < anatomy.shape[2]:
            raise ValueError("slice_index is outside the axial range")
        slice_selection = "explicit"

    combined_absolute = np.concatenate(
        [np.abs(vectors[mode][mask]).reshape(-1) for mode in MODE_ORDER]
    )
    if vmax is None:
        vmax = float(np.percentile(combined_absolute, percentile))
        scale_selection = f"combined_components_abs_p{percentile:g}"
    else:
        vmax = float(vmax)
        scale_selection = "explicit"
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Shared component vmax must be finite and positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path = (
        Path(panels_directory).resolve()
        if panels_directory is not None
        else output_path.parent / f"{output_path.stem}_panels"
    )
    panel_path.mkdir(parents=True, exist_ok=True)
    anatomy_slice = _take_slice(anatomy, plane, slice_index)
    mask_slice = _take_slice(mask, plane, slice_index)
    anatomy_limits = _robust_limits(anatomy, mask)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#20242b",
            "axes.titlecolor": "#20242b",
            "figure.facecolor": "white",
        }
    )

    def draw_component(axis, values: np.ndarray, title: str):
        axis.imshow(
            anatomy_slice,
            cmap="gray",
            origin="lower",
            vmin=anatomy_limits[0],
            vmax=anatomy_limits[1],
            interpolation="nearest",
        )
        overlay = np.ma.masked_where(~mask_slice, values)
        image = axis.imshow(
            overlay,
            cmap="RdBu_r",
            origin="lower",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.82,
            interpolation="bilinear",
        )
        if np.any(mask_slice) and np.any(~mask_slice):
            axis.contour(
                mask_slice.astype(float),
                levels=[0.5],
                colors="#f5f5f5",
                linewidths=0.4,
            )
        left, right, bottom, top = ORIENTATION_LABELS[plane]
        label_style = dict(color="white", fontsize=7, fontweight="bold", alpha=0.9)
        axis.text(0.012, 0.5, left, transform=axis.transAxes, ha="left", va="center", **label_style)
        axis.text(
            0.988,
            0.5,
            right,
            transform=axis.transAxes,
            ha="right",
            va="center",
            **label_style,
        )
        axis.text(
            0.5,
            0.012,
            bottom,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            **label_style,
        )
        axis.text(0.5, 0.988, top, transform=axis.transAxes, ha="center", va="top", **label_style)
        axis.set_title(title, fontsize=10, fontweight="semibold", pad=5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        return image

    figure, axes = plt.subplots(3, 4, figsize=(14.4, 10.6), constrained_layout=True)
    last_image = None
    for row, component_name in enumerate(("Ex", "Ey", "Ez")):
        for column, mode in enumerate(MODE_ORDER):
            last_image = draw_component(
                axes[row, column],
                _take_slice(vectors[mode][..., row], plane, slice_index),
                MODE_TITLES[mode] if row == 0 else "",
            )
        axes[row, 0].text(
            -0.10,
            0.5,
            component_name,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="semibold",
        )
    figure.suptitle(
        "Electric-field components across conductivity models · axial view",
        fontsize=15,
        fontweight="semibold",
    )
    colorbar = figure.colorbar(
        last_image, ax=axes, location="bottom", shrink=0.68, pad=0.018, aspect=38
    )
    colorbar.set_label("Electric-field component (V/m) · shared symmetric scale", fontsize=10)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)

    panel_outputs: dict[str, str] = {}
    for mode in MODE_ORDER:
        panel_figure, panel_axes = plt.subplots(
            1, 3, figsize=(10.5, 3.8), constrained_layout=True
        )
        panel_image = None
        for component, component_name in enumerate(("Ex", "Ey", "Ez")):
            panel_image = draw_component(
                panel_axes[component],
                _take_slice(vectors[mode][..., component], plane, slice_index),
                component_name,
            )
        panel_figure.suptitle(MODE_TITLES[mode], fontsize=13, fontweight="semibold")
        panel_colorbar = panel_figure.colorbar(
            panel_image, ax=panel_axes, location="bottom", shrink=0.72, pad=0.025
        )
        panel_colorbar.set_label("Electric-field component (V/m) · shared scale", fontsize=9)
        destination = panel_path / f"{mode}_xyz.png"
        panel_figure.savefig(destination, dpi=dpi, facecolor="white")
        plt.close(panel_figure)
        panel_outputs[mode] = str(destination)

    center_voxel = (np.asarray(anatomy.shape, dtype=float) - 1.0) / 2.0
    center_voxel[2] = slice_index
    world_coordinate = float(nib.affines.apply_affine(reference_affine, center_voxel)[2])
    report = {
        "schema_version": 1,
        "view": "components_3x4",
        "component_basis": "SimNIBS output world x/y/z",
        "output": str(output_path),
        "panels": panel_outputs,
        "plane": plane,
        "slice_index": slice_index,
        "slice_world_coordinate_mm": world_coordinate,
        "slice_selection": slice_selection,
        "mask_file": str(Path(mask_file).resolve()),
        "mask_labels": list(mask_labels),
        "masked_voxels": int(np.count_nonzero(mask)),
        "vmin_v_per_m": -vmax,
        "vmax_v_per_m": vmax,
        "scale_selection": scale_selection,
        "fields": {
            mode: {
                "file": str(Path(field_files[mode]).resolve()),
                "component_mean_v_per_m": [
                    float(np.mean(vectors[mode][..., component][mask]))
                    for component in range(3)
                ],
                "component_abs_p99_v_per_m": [
                    float(np.percentile(np.abs(vectors[mode][..., component][mask]), 99.0))
                    for component in range(3)
                ],
                "magnitude_mean_v_per_m": float(
                    np.mean(np.linalg.norm(vectors[mode][mask], axis=-1))
                ),
            }
            for mode in MODE_ORDER
        },
    }
    _write_report(output_path.with_suffix(".json"), report)
    return report


def plot_field_comparison(
    field_files: Mapping[str, str | Path],
    anatomy_file: str | Path,
    mask_file: str | Path,
    output_file: str | Path,
    *,
    plane: str = "axial",
    slice_index: int | None = None,
    mask_labels: tuple[int, ...] = (1, 2, 3),
    vmax: float | None = None,
    percentile: float = 99.5,
    dpi: int = 220,
    panels_directory: str | Path | None = None,
    view: str = "components",
) -> dict:
    """Create a voxel-level electric-field comparison for the requested view."""
    common = dict(
        plane=plane,
        slice_index=slice_index,
        mask_labels=mask_labels,
        vmax=vmax,
        percentile=percentile,
        dpi=dpi,
        panels_directory=panels_directory,
    )
    if view == "components":
        return _plot_component_comparison(
            field_files, anatomy_file, mask_file, output_file, **common
        )
    if view == "magnitude":
        return _plot_magnitude_comparison(
            field_files, anatomy_file, mask_file, output_file, **common
        )
    raise ValueError("view must be components or magnitude")
