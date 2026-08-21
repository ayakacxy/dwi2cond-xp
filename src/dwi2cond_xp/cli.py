"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .leadfield import run_tdcs_leadfield
from .montage_plot import plot_montage_schematic
from .nifti_fit import fit_dti_nifti, select_shell_nifti
from .plotting import plot_field_comparison
from .registration import make_charm_brain_mask, register_tensor_affine
from .simnibs_adapter import tensor_to_mesh_conductivity
from .simulation import run_tdcs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dwi2cond-xp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser(
        "select-shell", help="Extract one shell from a compressed multishell DWI"
    )
    select.add_argument("data")
    select.add_argument("bvals")
    select.add_argument("bvecs")
    select.add_argument("output_data")
    select.add_argument("output_bvals")
    select.add_argument("output_bvecs")
    select.add_argument("--shell", type=float, default=1000.0)
    select.add_argument("--tolerance", type=float, default=100.0)
    select.add_argument("--b0-threshold", type=float, default=50.0)
    fit = subparsers.add_parser("fit-dti", help="Fit a tensor to preprocessed single-shell DWI")
    fit.add_argument("data")
    fit.add_argument("bvals")
    fit.add_argument("bvecs")
    fit.add_argument("mask")
    fit.add_argument("output")
    fit.add_argument("--grad-dev")
    fit.add_argument("--shell", type=float, default=1000.0)
    fit.add_argument("--tolerance", type=float, default=100.0)
    fit.add_argument("--b0-threshold", type=float, default=50.0)
    fit.add_argument("--z-chunk", type=int, default=4)
    fit.add_argument("--voxel-batch", type=int, default=4096)
    fit.add_argument("--workers", type=int, default=1)
    fit.add_argument("--valid-mask-out")
    fit.add_argument("--qa-json")
    fit.add_argument(
        "--progress",
        choices=("tqdm", "off"),
        default="tqdm",
        help="Show voxel progress with tqdm or disable it explicitly",
    )
    register = subparsers.add_parser(
        "register-tensor", help="Map and reorient a tensor to a T1/head-model grid"
    )
    register.add_argument("tensor")
    register.add_argument("reference")
    register.add_argument("output")
    alignment = register.add_mutually_exclusive_group(required=True)
    alignment.add_argument(
        "--world-transform", help="External 4x4 input-world to reference-world affine text file"
    )
    alignment.add_argument(
        "--assume-aligned",
        action="store_true",
        help="Declare that external preprocessing already aligned the tensor to T1 world space",
    )
    register.add_argument("--source-mask")
    register.add_argument("--reference-mask")
    register.add_argument("--valid-mask-out")
    register.add_argument("--qa-json")
    register.add_argument("--interpolation-order", type=int, choices=(0, 1, 3), default=1)
    conductivity = subparsers.add_parser(
        "tensor-to-mesh", help="Write a SimNIBS mesh with dir/vn/mc conductivity tensors"
    )
    conductivity.add_argument("tensor")
    conductivity.add_argument("mesh")
    conductivity.add_argument("output_mesh")
    conductivity.add_argument("--mode", choices=("dir", "vn", "mc"), default="vn")
    conductivity.add_argument("--aniso-tissues", type=int, nargs="+", default=(1, 2))
    conductivity.add_argument("--cond-json", help="JSON object mapping tissue labels to scalar conductivity")
    conductivity.add_argument("--no-correct-fsl", action="store_true")
    conductivity.add_argument("--max-ratio", type=float, default=10.0)
    conductivity.add_argument("--max-cond", type=float, default=2.0)
    conductivity.add_argument("--excentricity-scaling", type=float)
    conductivity.add_argument("--no-correct-intensity", action="store_true")
    conductivity.add_argument("--qa-json")
    brain_mask = subparsers.add_parser(
        "charm-brain-mask", help="Create the official 1..499 brain mask from CHARM labeling"
    )
    brain_mask.add_argument("labeling")
    brain_mask.add_argument("output")
    brain_mask.add_argument("--reference")
    simulate = subparsers.add_parser(
        "simulate-tdcs", help="Run scalar or anisotropic tDCS with SimNIBS 4.6"
    )
    simulate.add_argument("subpath", help="CHARM m2m_<subject> directory")
    simulate.add_argument("output_root", help="Common output root for conductivity modes")
    simulate.add_argument(
        "--mode", choices=("scalar", "vn", "dir", "mc"), required=True
    )
    simulate.add_argument("--tensor", help="Six-component diffusion tensor NIfTI on the T1 grid")
    simulate.add_argument("--anode", default="C3")
    simulate.add_argument("--cathode", default="C4")
    simulate.add_argument("--current-ma", type=float, default=1.0)
    simulate.add_argument("--shape", choices=("rect", "ellipse"), default="rect")
    simulate.add_argument("--dimensions", type=float, nargs=2, default=(50.0, 50.0))
    simulate.add_argument("--thickness", type=float, default=4.0)
    simulate.add_argument("--fields", default="E")
    simulate.add_argument(
        "--solver",
        choices=("pardiso", "hypre", "mumps", "petsc_pardiso"),
        default="pardiso",
    )
    simulate.add_argument(
        "--volume-tissues",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Tissue labels written to voxel NIfTI; default: WM/GM/CSF",
    )
    simulate.add_argument("--cpus", type=int, default=8)
    simulate.add_argument(
        "--dry-run", action="store_true", help="Validate inputs and write a manifest without solving"
    )
    leadfield = subparsers.add_parser(
        "simulate-leadfield",
        help="Generate scalar/vn/dir/mc lead fields for all EEG-cap electrodes",
    )
    leadfield.add_argument("subpath", help="CHARM m2m_<subject> directory")
    leadfield.add_argument("output_root", help="Common output root for conductivity modes")
    leadfield.add_argument(
        "--mode", choices=("scalar", "vn", "dir", "mc"), required=True
    )
    leadfield.add_argument("--tensor", help="Six-component diffusion tensor NIfTI on the T1 grid")
    leadfield.add_argument("--eeg-cap")
    leadfield.add_argument("--field", choices=("E", "J"), default="E")
    leadfield.add_argument(
        "--interpolation", choices=("none", "middle-gm"), default="none"
    )
    leadfield.add_argument("--tissues", type=int, nargs="+", default=(1, 2))
    leadfield.add_argument(
        "--interpolation-tissues", type=int, nargs="+", default=(2,)
    )
    leadfield.add_argument(
        "--shape", choices=("ellipse", "rect"), default="ellipse"
    )
    leadfield.add_argument("--dimensions", type=float, nargs=2, default=(10.0, 10.0))
    leadfield.add_argument("--thickness", type=float, default=4.0)
    leadfield.add_argument(
        "--solver", choices=("pardiso", "default"), default="pardiso"
    )
    leadfield.add_argument("--cpus", type=int, default=1)
    leadfield.add_argument("--roi-labels", type=int, nargs="*", default=())
    leadfield.add_argument("--avoid-labels", type=int, nargs="*", default=())
    leadfield.add_argument(
        "--no-export-npy",
        action="store_true",
        help="Keep only SimNIBS HDF5; default is chunked downstream NPY export",
    )
    leadfield.add_argument(
        "--dry-run", action="store_true", help="Validate inputs and write a manifest without solving"
    )
    montage_plot = subparsers.add_parser(
        "plot-montage", help="Plot a selected montage on the MNE standard 10-20 layout"
    )
    montage_plot.add_argument("output")
    montage_plot.add_argument("--anode", default="C3")
    montage_plot.add_argument("--cathode", default="C4")
    montage_plot.add_argument("--current-ma", type=float, default=1.0)
    montage_plot.add_argument("--shape", choices=("rect", "ellipse"), default="rect")
    montage_plot.add_argument(
        "--dimensions", type=float, nargs=2, default=(50.0, 50.0)
    )
    montage_plot.add_argument("--thickness", type=float, default=4.0)
    montage_plot.add_argument("--montage", default="standard_1020")
    montage_plot.add_argument("--dpi", type=int, default=220)
    montage_plot.add_argument("--svg")
    compare = subparsers.add_parser(
        "compare-fields", help="Create a shared-scale comparison from four voxel-level field NIfTIs"
    )
    compare.add_argument("scalar")
    compare.add_argument("vn")
    compare.add_argument("dir")
    compare.add_argument("mc")
    compare.add_argument("anatomy", help="T1 NIfTI used as the grayscale background")
    compare.add_argument("mask", help="CHARM final_tissues or a binary brain mask")
    compare.add_argument("output", help="Output PNG for the 3x4 component or 2x2 magnitude figure")
    compare.add_argument(
        "--plane", choices=("axial", "coronal", "sagittal"), default="axial"
    )
    compare.add_argument("--slice-index", type=int)
    compare.add_argument("--mask-labels", type=int, nargs="+", default=(1, 2, 3))
    compare.add_argument("--vmax", type=float)
    compare.add_argument("--percentile", type=float, default=99.5)
    compare.add_argument("--dpi", type=int, default=220)
    compare.add_argument("--panels-dir")
    compare.add_argument(
        "--view",
        choices=("components", "magnitude"),
        default="components",
        help="Default: 3x4 Ex/Ey/Ez components; magnitude: four-mode magnitude",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "select-shell":
        print("Extracting the selected shell in one sequential pass...", flush=True)
        selected = select_shell_nifti(
            args.data,
            args.bvals,
            args.bvecs,
            args.output_data,
            args.output_bvals,
            args.output_bvecs,
            shell=args.shell,
            tolerance=args.tolerance,
            b0_threshold=args.b0_threshold,
        )
        print(f"Done: selected {selected.size} volumes; output: {args.output_data}", flush=True)
        return 0
    if args.command == "register-tensor":
        transform = (
            None
            if args.world_transform is None
            else np.loadtxt(args.world_transform, dtype=np.float64)
        )
        bar = tqdm(total=8, desc="Tensor mapping", unit="stage", dynamic_ncols=True)
        try:
            register_tensor_affine(
                args.tensor,
                args.reference,
                args.output,
                world_transform=transform,
                source_mask_file=args.source_mask,
                reference_mask_file=args.reference_mask,
                output_valid_mask_file=args.valid_mask_out,
                qa_file=args.qa_json,
                interpolation_order=args.interpolation_order,
                progress=lambda done, total: bar.update(done - bar.n),
                alignment_assumption=(
                    "external_world_transform"
                    if args.world_transform is not None
                    else "externally_preprocessed_and_aligned"
                ),
            )
        finally:
            bar.close()
        print(f"Done: {args.output}", flush=True)
        return 0
    if args.command == "tensor-to-mesh":
        conductivity_values = None
        if args.cond_json is not None:
            raw = json.loads(Path(args.cond_json).read_text(encoding="utf-8"))
            conductivity_values = {int(key): float(value) for key, value in raw.items()}
        tensor_to_mesh_conductivity(
            args.tensor,
            args.mesh,
            args.output_mesh,
            mode=args.mode,
            anisotropic_tissues=tuple(args.aniso_tissues),
            scalar_conductivity=conductivity_values,
            correct_fsl=not args.no_correct_fsl,
            max_ratio=args.max_ratio,
            max_cond=args.max_cond,
            excentricity_scaling=args.excentricity_scaling,
            correct_intensity=not args.no_correct_intensity,
            qa_file=args.qa_json,
        )
        print(f"Done: {args.output_mesh}", flush=True)
        return 0
    if args.command == "charm-brain-mask":
        make_charm_brain_mask(
            args.labeling,
            args.output,
            reference_file=args.reference,
        )
        print(f"Done: {args.output}", flush=True)
        return 0
    if args.command == "simulate-tdcs":
        manifest = run_tdcs(
            args.subpath,
            args.output_root,
            mode=args.mode,
            tensor_file=args.tensor,
            anode=args.anode,
            cathode=args.cathode,
            current_ma=args.current_ma,
            shape=args.shape,
            dimensions=tuple(args.dimensions),
            thickness=args.thickness,
            fields=args.fields,
            solver=args.solver,
            volume_tissues=tuple(args.volume_tissues),
            cpus=args.cpus,
            dry_run=args.dry_run,
        )
        print(
            f"{manifest['status']}: {manifest['output_directory']}",
            flush=True,
        )
        return 0
    if args.command == "simulate-leadfield":
        progress_bar: list[tqdm | None] = [None]

        def report_leadfield(done: int, total: int, phase: str) -> None:
            if progress_bar[0] is None:
                progress_bar[0] = tqdm(
                    total=total,
                    desc="Lead-field validation and NPY export",
                    unit="basis",
                    dynamic_ncols=True,
                )
            progress_bar[0].update(done - progress_bar[0].n)
            progress_bar[0].set_postfix_str(phase)

        try:
            manifest = run_tdcs_leadfield(
                args.subpath,
                args.output_root,
                mode=args.mode,
                tensor_file=args.tensor,
                eeg_cap=args.eeg_cap,
                field=args.field,
                interpolation=args.interpolation,
                tissues=tuple(args.tissues),
                interpolation_tissues=tuple(args.interpolation_tissues),
                shape=args.shape,
                dimensions=tuple(args.dimensions),
                thickness=args.thickness,
                solver=args.solver,
                cpus=args.cpus,
                export_matrix=not args.no_export_npy,
                roi_labels=tuple(args.roi_labels),
                avoid_labels=tuple(args.avoid_labels),
                dry_run=args.dry_run,
                progress=report_leadfield,
            )
        finally:
            if progress_bar[0] is not None:
                progress_bar[0].close()
        print(
            f"{manifest['status']}: {manifest['output_directory']}",
            flush=True,
        )
        return 0
    if args.command == "plot-montage":
        report = plot_montage_schematic(
            args.output,
            anode=args.anode,
            cathode=args.cathode,
            current_ma=args.current_ma,
            shape=args.shape,
            dimensions=tuple(args.dimensions),
            thickness=args.thickness,
            montage_name=args.montage,
            dpi=args.dpi,
            svg_file=args.svg,
        )
        print(f"Done: {report['output_png']}", flush=True)
        return 0
    if args.command == "compare-fields":
        report = plot_field_comparison(
            {
                "scalar": args.scalar,
                "vn": args.vn,
                "dir": args.dir,
                "mc": args.mc,
            },
            args.anatomy,
            args.mask,
            args.output,
            plane=args.plane,
            slice_index=args.slice_index,
            mask_labels=tuple(args.mask_labels),
            vmax=args.vmax,
            percentile=args.percentile,
            dpi=args.dpi,
            panels_directory=args.panels_dir,
            view=args.view,
        )
        print(
            f"Done: {args.output} ({args.plane} slice {report['slice_index']})",
            flush=True,
        )
        return 0
    if args.command != "fit-dti":
        raise RuntimeError("Command is not implemented")

    progress_bar: list[tqdm | None] = [None]

    def report(done: int, total: int, z_stop: int) -> None:
        del z_stop
        if args.progress == "off":
            return
        if progress_bar[0] is None:
            progress_bar[0] = tqdm(
                total=total,
                desc="DTI tensor fitting",
                unit="voxel",
                unit_scale=True,
                dynamic_ncols=True,
                mininterval=1.0,
            )
        progress_bar[0].update(done - progress_bar[0].n)

    try:
        fit_dti_nifti(
            args.data,
            args.bvals,
            args.bvecs,
            args.mask,
            args.output,
            grad_dev_file=args.grad_dev,
            shell=args.shell,
            tolerance=args.tolerance,
            b0_threshold=args.b0_threshold,
            z_chunk=args.z_chunk,
            voxel_batch=args.voxel_batch,
            workers=args.workers,
            progress=report,
            valid_mask_file=args.valid_mask_out,
            qa_file=args.qa_json,
        )
    finally:
        if progress_bar[0] is not None:
            progress_bar[0].close()
    print(f"Done: {args.output}", flush=True)
    return 0
