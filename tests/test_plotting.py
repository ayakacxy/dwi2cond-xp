from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp.plotting import (
    MODE_ORDER,
    _automatic_slice,
    _load_canonical,
    _robust_limits,
    _scalar_field,
    plot_field_comparison,
)


def _make_plot_inputs(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    """Create four vector fields with smooth spatial variation."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    shape = (24, 28, 20)
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    x, y, z = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0]),
        np.linspace(-1.0, 1.0, shape[1]),
        np.linspace(-1.0, 1.0, shape[2]),
        indexing="ij",
    )
    radius = np.sqrt(x * x + y * y + z * z)
    mask = radius < 0.85
    anatomy = np.clip(1.0 - radius, 0.0, None) * 1000.0
    labels = np.zeros(shape, dtype=np.int16)
    labels[mask] = 1
    anatomy_file = tmp_path / "T1.nii.gz"
    mask_file = tmp_path / "final_tissues.nii.gz"
    nib.save(nib.Nifti1Image(anatomy.astype(np.float32), affine), anatomy_file)
    nib.save(nib.Nifti1Image(labels, affine), mask_file)
    fields: dict[str, Path] = {}
    for index, mode in enumerate(MODE_ORDER):
        scale = 0.8 + 0.12 * index
        vector = np.stack(
            [scale * x, -scale * y, 0.5 * scale * z], axis=-1
        ).astype(np.float32)
        vector[~mask] = 0.0
        destination = tmp_path / f"{mode}_E.nii.gz"
        nib.save(nib.Nifti1Image(vector, affine), destination)
        fields[mode] = destination
    return fields, anatomy_file, mask_file


def test_components_export_3x4_and_panels(tmp_path: Path) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    output = tmp_path / "components.png"
    report = plot_field_comparison(
        fields, anatomy, mask, output, view="components", dpi=60
    )
    assert report["view"] == "components_3x4"
    assert report["slice_selection"] == "maximum_mask_area"
    assert output.is_file() and output.stat().st_size > 1000
    assert output.with_suffix(".json").is_file()
    assert all(Path(path).is_file() for path in report["panels"].values())
    assert report["vmin_v_per_m"] == -report["vmax_v_per_m"]


def test_magnitude_export_uses_shared_positive_scale(tmp_path: Path) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    output = tmp_path / "magnitude.png"
    report = plot_field_comparison(
        fields, anatomy, mask, output, view="magnitude", dpi=60
    )
    assert output.is_file() and output.stat().st_size > 1000
    assert report["vmin_v_per_m"] == 0.0
    assert report["vmax_v_per_m"] > 0.0


def test_plot_helpers_cover_scalar_singleton_and_invalid_data(tmp_path: Path) -> None:
    singleton = tmp_path / "singleton.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 3, 4, 1)), np.eye(4)), singleton)
    data, _ = _load_canonical(singleton)
    assert data.shape == (2, 3, 4)
    assert _scalar_field(data, singleton).shape == data.shape
    with pytest.raises(ValueError, match="three vector components"):
        _scalar_field(np.zeros((2, 3, 4, 2)), singleton)
    with pytest.raises(ValueError, match="brain mask is empty"):
        _automatic_slice(np.zeros((2, 3, 4), dtype=bool), "axial")
    assert _robust_limits(np.ones((2, 2, 2)), np.ones((2, 2, 2), dtype=bool)) == (1.0, 2.0)
    low, high = _robust_limits(np.arange(8.0).reshape(2, 2, 2), np.zeros((2, 2, 2), bool))
    assert low < high
    with pytest.raises(ValueError, match="no finite values"):
        _robust_limits(np.full((2, 2, 2), np.nan), np.zeros((2, 2, 2), bool))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"plane": "oblique"}, "plane"),
        ({"percentile": 0}, "percentile"),
        ({"dpi": 0}, "dpi"),
        ({"slice_index": 999}, "slice_index"),
        ({"vmax": 0}, "vmax"),
    ],
)
def test_magnitude_rejects_invalid_parameters(tmp_path: Path, kwargs, message) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        plot_field_comparison(
            fields, anatomy, mask, tmp_path / "out.png", view="magnitude", **kwargs
        )


def test_magnitude_rejects_field_keys_and_image_contracts(tmp_path: Path) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        plot_field_comparison(
            {key: value for key, value in fields.items() if key != "mc"},
            anatomy,
            mask,
            tmp_path / "out.png",
            view="magnitude",
        )

    bad_anatomy = tmp_path / "bad_anatomy.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4)), bad_anatomy)
    with pytest.raises(ValueError, match="three-dimensional"):
        plot_field_comparison(fields, bad_anatomy, mask, tmp_path / "out.png", view="magnitude")

    bad_mask = tmp_path / "bad_mask.nii.gz"
    mask_data = np.asanyarray(nib.load(mask).dataobj)
    nib.save(nib.Nifti1Image(mask_data, np.diag([2, 1, 1, 1])), bad_mask)
    with pytest.raises(ValueError, match="mask affine"):
        plot_field_comparison(fields, anatomy, bad_mask, tmp_path / "out.png", view="magnitude")
    empty_mask = tmp_path / "empty_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros_like(mask_data), nib.load(mask).affine), empty_mask)
    with pytest.raises(ValueError, match="contains none"):
        plot_field_comparison(fields, anatomy, empty_mask, tmp_path / "out.png", view="magnitude")


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("dimensions", "three vector components"),
        ("affine", "shape/affine"),
        ("nonfinite", "no finite voxel"),
        ("negative", "negative values"),
    ],
)
def test_magnitude_rejects_bad_field_data(tmp_path: Path, kind: str, message: str) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    image = nib.load(fields["scalar"])
    if kind == "dimensions":
        data = np.zeros(image.shape[:3] + (2,), dtype=np.float32)
        affine = image.affine
    elif kind == "affine":
        data = np.zeros(image.shape[:3], dtype=np.float32)
        affine = np.diag([2, 1, 1, 1])
    elif kind == "nonfinite":
        data = np.full(image.shape[:3], np.nan, dtype=np.float32)
        affine = image.affine
    else:
        data = np.full(image.shape[:3], -1.0, dtype=np.float32)
        affine = image.affine
    nib.save(nib.Nifti1Image(data, affine), fields["scalar"])
    with pytest.raises(ValueError, match=message):
        plot_field_comparison(fields, anatomy, mask, tmp_path / "out.png", view="magnitude")


def test_explicit_plot_scales_and_custom_panel_directories(tmp_path: Path) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    magnitude = plot_field_comparison(
        fields,
        anatomy,
        mask,
        tmp_path / "magnitude.png",
        view="magnitude",
        plane="coronal",
        slice_index=10,
        vmax=2.0,
        dpi=40,
        panels_directory=tmp_path / "magnitude_panels",
    )
    components = plot_field_comparison(
        fields,
        anatomy,
        mask,
        tmp_path / "components.png",
        view="components",
        slice_index=8,
        vmax=2.0,
        dpi=40,
        panels_directory=tmp_path / "component_panels",
    )
    assert magnitude["scale_selection"] == "explicit"
    assert magnitude["slice_selection"] == "explicit"
    assert components["scale_selection"] == "explicit"
    assert components["slice_selection"] == "explicit"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"plane": "coronal"}, "axial view only"),
        ({"percentile": 101}, "percentile"),
        ({"dpi": 0}, "dpi"),
        ({"slice_index": 999}, "slice_index"),
        ({"vmax": np.inf}, "vmax"),
    ],
)
def test_components_rejects_invalid_parameters(tmp_path: Path, kwargs, message) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        plot_field_comparison(fields, anatomy, mask, tmp_path / "out.png", **kwargs)


def test_components_rejects_keys_grids_and_vector_values(tmp_path: Path) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        plot_field_comparison(
            {key: value for key, value in fields.items() if key != "mc"},
            anatomy,
            mask,
            tmp_path / "out.png",
        )

    image = nib.load(fields["scalar"])
    nib.save(nib.Nifti1Image(np.zeros(image.shape[:3]), image.affine), fields["scalar"])
    with pytest.raises(ValueError, match="three vector components"):
        plot_field_comparison(fields, anatomy, mask, tmp_path / "out.png")

    fields, anatomy, mask = _make_plot_inputs(tmp_path / "grid")
    image = nib.load(fields["scalar"])
    nib.save(nib.Nifti1Image(np.zeros(image.shape), np.diag([2, 1, 1, 1])), fields["scalar"])
    with pytest.raises(ValueError, match="shape/affine"):
        plot_field_comparison(fields, anatomy, mask, tmp_path / "grid/out.png")

    fields, anatomy, mask = _make_plot_inputs(tmp_path / "nan")
    image = nib.load(fields["scalar"])
    data = np.asanyarray(image.dataobj).copy()
    labels = np.asanyarray(nib.load(mask).dataobj)
    data[np.argwhere(labels == 1)[0][0], np.argwhere(labels == 1)[0][1], np.argwhere(labels == 1)[0][2], 0] = np.nan
    nib.save(nib.Nifti1Image(data, image.affine), fields["scalar"])
    with pytest.raises(ValueError, match="NaN/Inf"):
        plot_field_comparison(fields, anatomy, mask, tmp_path / "nan/out.png")


def test_components_rejects_bad_anatomy_mask_and_view(tmp_path: Path) -> None:
    fields, anatomy, mask = _make_plot_inputs(tmp_path)
    anatomy_img = nib.load(anatomy)
    bad_mask = tmp_path / "bad_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones(anatomy_img.shape), np.diag([2, 1, 1, 1])), bad_mask
    )
    with pytest.raises(ValueError, match="mask affine"):
        plot_field_comparison(fields, anatomy, bad_mask, tmp_path / "out.png")
    empty_mask = tmp_path / "empty_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(anatomy_img.shape), anatomy_img.affine), empty_mask)
    with pytest.raises(ValueError, match="contains none"):
        plot_field_comparison(fields, anatomy, empty_mask, tmp_path / "out.png")
    bad_anatomy = tmp_path / "bad_anatomy.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(anatomy_img.shape + (2,)), anatomy_img.affine), bad_anatomy)
    with pytest.raises(ValueError, match="three-dimensional"):
        plot_field_comparison(fields, bad_anatomy, mask, tmp_path / "out.png")
    with pytest.raises(ValueError, match="view must"):
        plot_field_comparison(fields, anatomy, mask, tmp_path / "out.png", view="mesh")
