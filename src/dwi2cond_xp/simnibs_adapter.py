"""Connect pure-Python conductivity conversion to SimNIBS mesh/FEM structures."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .conductivity import correct_fsl_tensor_basis, tensors_to_conductivity
from .registration import tensor6_to_matrix


def tensor_to_mesh_conductivity(
    tensor_file: str | Path,
    mesh_file: str | Path,
    output_mesh_file: str | Path,
    *,
    mode: str = "vn",
    anisotropic_tissues: tuple[int, ...] = (1, 2),
    scalar_conductivity: dict[int, float] | None = None,
    correct_fsl: bool = True,
    max_ratio: float = 10.0,
    max_cond: float = 2.0,
    excentricity_scaling: float | None = None,
    correct_intensity: bool = True,
    vn_singular_policy: str = "error",
    qa_file: str | Path | None = None,
) -> Path:
    """Interpolate tensors to tetrahedra and write nine-component conductivity.

    SimNIBS is used only for mesh I/O and interpolation. The numerical
    ``dir/vn/mc`` conversion is implemented here with pure NumPy.
    """
    try:
        from simnibs.mesh_tools import mesh_io
        from simnibs.utils.cond_utils import standard_cond
    except ImportError as exc:
        raise RuntimeError("tensor-to-mesh requires SimNIBS; DTI fitting does not") from exc

    tensor_img = nib.load(str(tensor_file))
    if tensor_img.shape[-1:] != (6,):
        raise ValueError("The final tensor NIfTI axis must contain six components")
    mesh = mesh_io.read_msh(str(mesh_file))
    sampled = mesh_io.ElementData.from_data_grid(
        mesh,
        tensor_img.dataobj,
        tensor_img.affine,
        "sampled_diffusion_tensor",
        order=1,
        cval=0.0,
        prefilter=True,
    )
    matrices = tensor6_to_matrix(np.asarray(sampled.value, dtype=np.float64))
    if correct_fsl:
        matrices = correct_fsl_tensor_basis(matrices, tensor_img.affine)

    tetrahedra = mesh.elm.get_tetrahedra()
    tetrahedra_array = np.asarray(tetrahedra)
    tetrahedra_count = (
        int(np.count_nonzero(tetrahedra_array))
        if tetrahedra_array.dtype == np.bool_
        else int(tetrahedra_array.size)
    )
    tags = np.asarray(mesh.elm.tag1[tetrahedra], dtype=np.int32)
    volumes = np.asarray(mesh.elements_volumes_and_areas().value[tetrahedra], float)
    if scalar_conductivity is None:
        scalar_conductivity = {
            index + 1: float(item.value)
            for index, item in enumerate(standard_cond())
            if item.value is not None
        }
    conductivity, report = tensors_to_conductivity(
        matrices[tetrahedra],
        tags,
        scalar_conductivity,
        mode=mode,
        anisotropic_tissues=anisotropic_tissues,
        weights=volumes,
        max_ratio=max_ratio,
        max_cond=max_cond,
        excentricity_scaling=excentricity_scaling,
        correct_intensity=correct_intensity,
        vn_singular_policy=vn_singular_policy,
    )
    full = np.zeros((mesh.elm.nr, 9), dtype=np.float64)
    full[tetrahedra] = conductivity.reshape(-1, 9)
    field = mesh_io.ElementData(full, "conductivity", mesh=mesh)
    field.assign_triangle_values()
    mesh.elmdata.append(field)

    output_path = Path(output_mesh_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_io.write_msh(mesh, str(output_path))
    qa_path = (
        Path(qa_file)
        if qa_file is not None
        else output_path.with_suffix(".conductivity.json")
    )
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    report.update(
        {
            "tensor_file": str(tensor_file),
            "mesh_file": str(mesh_file),
            "output_mesh_file": str(output_path),
            "tetrahedra": tetrahedra_count,
            "anisotropic_tissues": list(anisotropic_tissues),
            "correct_fsl": correct_fsl,
            "max_ratio": max_ratio,
            "max_cond": max_cond,
            "correct_intensity": correct_intensity,
            "vn_singular_policy": vn_singular_policy,
        }
    )
    qa_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
