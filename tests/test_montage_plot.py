from pathlib import Path

import pytest

import numpy as np

from dwi2cond_xp.montage_plot import (
    fit_cap_projection,
    plot_montage_schematic,
)


mne = pytest.importorskip("mne")


def test_spherical_cap_projection_keeps_montage_in_head_outline() -> None:
    montage = mne.channels.make_standard_montage("standard_1020")
    names = list(montage.ch_names)
    positions = montage.get_positions()["ch_pos"]
    coordinates = np.asarray([positions[name] for name in names], dtype=float)
    projection = fit_cap_projection(names, coordinates)
    projected = projection.transform(coordinates)
    radii = np.linalg.norm(projected, axis=1)
    by_name = {name: projected[index] for index, name in enumerate(names)}
    assert np.percentile(radii, 99.0) == pytest.approx(0.96)
    assert radii.max() < 1.0
    assert by_name["Fpz"][1] > by_name["Oz"][1]


def test_plot_montage_exports_png_svg_and_contract(tmp_path: Path) -> None:
    output = tmp_path / "montage.png"
    svg = tmp_path / "montage.svg"
    report = plot_montage_schematic(output, svg_file=svg, dpi=80)
    assert output.is_file() and output.stat().st_size > 1000
    assert svg.is_file() and svg.stat().st_size > 1000
    assert output.with_suffix(".json").is_file()
    assert report["anode"] == "C3"
    assert report["cathode"] == "C4"
    assert report["coordinate_contract"].endswith("not subject-specific")
    assert report["projection"] == "fitted_spherical_cap_azimuthal"
