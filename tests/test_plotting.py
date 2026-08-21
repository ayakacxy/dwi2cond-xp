from pathlib import Path

import nibabel as nib
import numpy as np

from dwi2cond_xp.plotting import MODE_ORDER, plot_field_comparison


def _make_plot_inputs(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    """Create four vector fields with smooth spatial variation."""
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
