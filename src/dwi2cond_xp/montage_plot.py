"""Create a standard 10-20 montage schematic for the public README with MNE."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ANODE_COLOR = "#C45147"
CATHODE_COLOR = "#347FBD"
INACTIVE_COLOR = "#DDE3E7"
OUTLINE_COLOR = "#27343C"


@dataclass(frozen=True)
class CapProjection:
    """Store a spherical-cap projection fitted from 3-D scalp electrodes."""

    center: np.ndarray
    angular_scale: float
    radial_scale: float
    y_sign: float

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Project 3-D electrodes to a 2-D head map using angle from the sphere center."""
        xyz = np.asarray(coordinates, dtype=float)
        one_point = xyz.ndim == 1
        xyz = np.atleast_2d(xyz)
        if xyz.shape[1] != 3:
            raise ValueError(f"coordinates must be (N, 3); found {xyz.shape}")
        centered = xyz - self.center
        in_plane_radius = np.linalg.norm(centered[:, :2], axis=1)
        polar_angle = np.arctan2(in_plane_radius, centered[:, 2])
        projected = np.zeros((len(xyz), 2), dtype=float)
        valid = in_plane_radius > 1e-8
        projected[valid] = (
            centered[valid, :2]
            / in_plane_radius[valid, None]
            * (polar_angle[valid] / self.angular_scale)[:, None]
        )
        projected[:, 1] *= self.y_sign
        projected *= self.radial_scale
        return projected[0] if one_point else projected


def fit_cap_projection(names: list[str], coordinates: np.ndarray) -> CapProjection:
    """Fit a sphere center from all MNE electrodes and build a stable projection."""
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape != (len(names), 3):
        raise ValueError(f"names and coordinates have incompatible shapes: {xyz.shape}")
    for landmark in ("Fpz", "Oz"):
        if landmark not in names:
            raise ValueError(f"montage is missing orientation landmark {landmark}")

    design = np.column_stack((2.0 * xyz, np.ones(len(xyz))))
    squared_radius = np.square(xyz).sum(axis=1)
    solution, *_ = np.linalg.lstsq(design, squared_radius, rcond=None)
    center = np.asarray(solution[:3], dtype=float)

    centered = xyz - center
    in_plane_radius = np.linalg.norm(centered[:, :2], axis=1)
    polar_angle = np.arctan2(in_plane_radius, centered[:, 2])
    angular_scale = float(np.percentile(polar_angle, 99.0))
    if not np.isfinite(angular_scale) or angular_scale <= 0.0:
        raise ValueError("Cannot determine an angular scale from montage coordinates")

    unscaled = CapProjection(center, angular_scale, 1.0, 1.0)
    preliminary = unscaled.transform(xyz)
    by_name = {name: preliminary[index] for index, name in enumerate(names)}
    y_sign = -1.0 if by_name["Fpz"][1] < by_name["Oz"][1] else 1.0
    oriented = preliminary * np.asarray((1.0, y_sign))
    radius_scale = float(np.percentile(np.linalg.norm(oriented, axis=1), 99.0))
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("Cannot determine a radial scale from montage projection")
    return CapProjection(center, angular_scale, 0.96 / radius_scale, y_sign)


def plot_montage_schematic(
    output_file: str | Path,
    *,
    anode: str = "C3",
    cathode: str = "C4",
    current_ma: float = 1.0,
    shape: str = "rect",
    dimensions: tuple[float, float] = (50.0, 50.0),
    thickness: float = 4.0,
    montage_name: str = "standard_1020",
    dpi: int = 220,
    svg_file: str | Path | None = None,
) -> dict:
    """Plot the full electrode topology and highlight anode and cathode.

    This is a 2-D topology schematic of an MNE standard montage, not an
    individual's 3-D CHARM scalp coordinates.
    """
    if anode == cathode:
        raise ValueError("anode and cathode must differ")
    if current_ma <= 0:
        raise ValueError("current_ma must be positive")
    if shape not in {"rect", "ellipse"}:
        raise ValueError("shape must be rect or ellipse")
    if any(value <= 0 for value in dimensions) or thickness <= 0:
        raise ValueError("Electrode dimensions and thickness must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Ellipse, Polygon

    try:
        import mne
    except ImportError as exc:
        raise RuntimeError(
            "Montage plotting requires the optional dependency: pip install '.[viz]'"
        ) from exc

    montage = mne.channels.make_standard_montage(montage_name)
    names = list(montage.ch_names)
    missing = [name for name in (anode, cathode) if name not in names]
    if missing:
        raise ValueError(f"MNE montage {montage_name} is missing electrodes: {missing}")
    positions = montage.get_positions()["ch_pos"]
    coordinates = np.asarray([positions[name] for name in names], dtype=float)
    projection = fit_cap_projection(names, coordinates)
    xy = projection.transform(coordinates)

    figure, axis = plt.subplots(figsize=(11.2, 6.2), constrained_layout=True)
    figure.patch.set_facecolor("#FBFCFD")
    axis.set_facecolor("#FBFCFD")
    axis.add_patch(
        Circle((0.0, 0.0), 0.98, facecolor="white", edgecolor=OUTLINE_COLOR, linewidth=3.0)
    )
    axis.add_patch(
        Polygon(
            [(-0.11, 0.965), (0.0, 1.14), (0.11, 0.965)],
            closed=False,
            fill=False,
            edgecolor=OUTLINE_COLOR,
            linewidth=3.0,
            joinstyle="miter",
        )
    )
    axis.add_patch(
        Ellipse((-1.01, -0.03), 0.18, 0.43, facecolor="white", edgecolor=OUTLINE_COLOR, linewidth=2.6)
    )
    axis.add_patch(
        Ellipse((1.01, -0.03), 0.18, 0.43, facecolor="white", edgecolor=OUTLINE_COLOR, linewidth=2.6)
    )

    inactive_xy = [
        point
        for name, point in zip(names, xy, strict=True)
        if name not in {anode, cathode}
    ]
    axis.scatter(
        [point[0] for point in inactive_xy],
        [point[1] for point in inactive_xy],
        s=42,
        facecolor=INACTIVE_COLOR,
        edgecolor="#94A1AA",
        linewidth=1.0,
        zorder=3,
    )
    active = {name: xy[names.index(name)] for name in (anode, cathode)}
    for name, color in ((anode, ANODE_COLOR), (cathode, CATHODE_COLOR)):
        point = active[name]
        axis.scatter(
            point[0],
            point[1],
            s=430,
            marker="s" if shape == "rect" else "o",
            facecolor=color,
            edgecolor="white",
            linewidth=3.2,
            zorder=5,
        )
        axis.scatter(
            point[0],
            point[1],
            s=500,
            marker="s" if shape == "rect" else "o",
            facecolor="none",
            edgecolor=OUTLINE_COLOR,
            linewidth=1.4,
            zorder=4,
        )
        axis.text(
            point[0],
            point[1],
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    axis.text(
        1.32,
        0.86,
        f"{anode} → {cathode}",
        ha="left",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=OUTLINE_COLOR,
    )
    axis.text(
        1.32,
        0.69,
        "tDCS montage",
        ha="left",
        va="center",
        fontsize=14,
        color="#69757E",
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="s" if shape == "rect" else "o",
            color="none",
            markerfacecolor=ANODE_COLOR,
            markeredgecolor=OUTLINE_COLOR,
            markersize=13,
            label=f"Anode  {anode}  (+{current_ma:g} mA)",
        ),
        Line2D(
            [0],
            [0],
            marker="s" if shape == "rect" else "o",
            color="none",
            markerfacecolor=CATHODE_COLOR,
            markeredgecolor=OUTLINE_COLOR,
            markersize=13,
            label=f"Cathode  {cathode}  (−{current_ma:g} mA)",
        ),
    ]
    axis.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.70, 0.64),
        ncol=1,
        frameon=False,
        fontsize=13,
        handletextpad=0.8,
        labelspacing=1.1,
    )
    axis.text(
        1.32,
        -0.12,
        f"{dimensions[0]:g} × {dimensions[1]:g} × {thickness:g} mm",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="semibold",
        color=OUTLINE_COLOR,
    )
    axis.text(
        1.32,
        -0.28,
        f"{shape} electrode · {current_ma:g} mA",
        ha="left",
        va="center",
        fontsize=12,
        color="#69757E",
    )
    axis.text(
        1.32,
        -0.63,
        "MNE standard 10–20 topology",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#69757E",
    )
    axis.text(
        1.32,
        -0.76,
        "Schematic only · not subject-specific 3D placement",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#87919A",
    )
    axis.set_xlim(-1.28, 2.82)
    axis.set_ylim(-1.20, 1.24)
    axis.set_aspect("equal")
    axis.axis("off")

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    svg_path: Path | None = None
    if svg_file is not None:
        svg_path = Path(svg_file).resolve()
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    report = {
        "schema_version": 1,
        "figure_type": "mne_standard_montage_infographic",
        "coordinate_contract": "MNE standard 10-20 topology; not subject-specific",
        "projection": "fitted_spherical_cap_azimuthal",
        "montage": montage_name,
        "anode": anode,
        "cathode": cathode,
        "current_ma": current_ma,
        "shape": shape,
        "dimensions_mm": list(dimensions),
        "thickness_mm": thickness,
        "output_png": str(output_path),
        "output_svg": None if svg_path is None else str(svg_path),
        "mne_version": mne.__version__,
    }
    report_path = output_path.with_suffix(".json")
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    return report
