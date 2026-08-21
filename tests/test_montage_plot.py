from pathlib import Path
import builtins

import pytest

import numpy as np

from dwi2cond_xp.montage_plot import (
    CapProjection,
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


def test_projection_validates_coordinate_shape_and_handles_center_point() -> None:
    projection = CapProjection(np.zeros(3), 1.0, 1.0, 1.0)
    assert np.array_equal(projection.transform(np.zeros(3)), [0.0, 0.0])
    with pytest.raises(ValueError, match="coordinates"):
        projection.transform(np.zeros((2, 2)))


def test_fit_projection_rejects_shape_landmarks_and_degenerate_coordinates(monkeypatch) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        fit_cap_projection(["Fpz", "Oz"], np.zeros((1, 3)))
    with pytest.raises(ValueError, match="missing orientation landmark"):
        fit_cap_projection(["Fpz", "Cz"], np.ones((2, 3)))
    with pytest.raises(ValueError, match="angular scale"):
        fit_cap_projection(["Fpz", "Oz"], np.zeros((2, 3)))

    original = CapProjection.transform

    def zero_projection(self, coordinates):
        return np.zeros((len(np.atleast_2d(coordinates)), 2))

    monkeypatch.setattr(CapProjection, "transform", zero_projection)
    with pytest.raises(ValueError, match="radial scale"):
        fit_cap_projection(
            ["Fpz", "Oz", "Cz"],
            np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    monkeypatch.setattr(CapProjection, "transform", original)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"anode": "C3", "cathode": "C3"}, "must differ"),
        ({"current_ma": 0}, "positive"),
        ({"shape": "circle"}, "shape"),
        ({"dimensions": (0, 10)}, "positive"),
        ({"thickness": 0}, "positive"),
        ({"dpi": 0}, "dpi"),
    ],
)
def test_plot_montage_rejects_invalid_parameters(tmp_path, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        plot_montage_schematic(tmp_path / "out.png", **kwargs)


def test_plot_montage_rejects_missing_electrode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing electrodes"):
        plot_montage_schematic(tmp_path / "out.png", anode="NOT_A_CHANNEL", dpi=40)


def test_plot_montage_reports_missing_mne(monkeypatch, tmp_path: Path) -> None:
    original_import = builtins.__import__

    def rejecting_import(name, *args, **kwargs):
        if name == "mne":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    with pytest.raises(RuntimeError, match="optional dependency"):
        plot_montage_schematic(tmp_path / "out.png", dpi=40)
