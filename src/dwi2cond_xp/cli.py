"""Command-line interface."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import time

import numpy as np


class _NullProgress:
    """Implement the disabled progress-bar contract without importing tqdm."""

    def __init__(self, initial: int = 0) -> None:
        self.n = int(initial)

    def update(self, amount: int) -> None:
        """Track the internal count without producing terminal output."""

        self.n += int(amount)

    def set_postfix_str(self, _phase: str, *, refresh: bool = True) -> None:
        """Ignore one disabled progress label."""

        del refresh

    def close(self) -> None:
        """Close the no-op progress bar."""

    def write(self, _message: str) -> None:
        """Ignore one disabled standalone progress message."""


class _LazyTqdm:
    """Load tqdm only for a route that displays progress."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        if kwargs.get("disable") is True:
            return _NullProgress(int(kwargs.get("initial", 0)))
        from tqdm.auto import tqdm as implementation

        return implementation(*args, **kwargs)


tqdm = _LazyTqdm()


class _LazyCallable:
    """Load one CLI implementation only when its route is executed."""

    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name
        self._value: object | None = None

    def __call__(self, *args: object, **kwargs: object) -> object:
        value = self._value
        if value is None:
            value = getattr(import_module(self._module, __package__), self._name)
            self._value = value
        return value(*args, **kwargs)


run_tdcs_leadfield = _LazyCallable(".leadfield", "run_tdcs_leadfield")
plot_montage_schematic = _LazyCallable(".montage_plot", "plot_montage_schematic")
fit_dti_nifti = _LazyCallable(".nifti_fit", "fit_dti_nifti")
select_shell_nifti = _LazyCallable(".nifti_fit", "select_shell_nifti")
plot_field_comparison = _LazyCallable(".plotting", "plot_field_comparison")
run_legacy_nifti = _LazyCallable(".preprocessing.legacy", "run_legacy_nifti")
run_nomoco_nifti = _LazyCallable(".preprocessing.nomoco", "run_nomoco_nifti")
run_t1_registration_nifti = _LazyCallable(
    ".preprocessing.t1_registration", "run_t1_registration_nifti"
)
run_fieldmap_nifti = _LazyCallable(".preprocessing.fieldmap", "run_fieldmap_nifti")
run_topup_nifti = _LazyCallable(".preprocessing.topup", "run_topup_nifti")
run_eddy_nifti = _LazyCallable(".preprocessing.eddy", "run_eddy_nifti")
register_tensor_fnirt_nifti = _LazyCallable(
    ".preprocessing.nonlinear", "register_tensor_fnirt_nifti"
)
make_charm_brain_mask = _LazyCallable(".registration", "make_charm_brain_mask")
register_tensor_affine = _LazyCallable(".registration", "register_tensor_affine")
tensor_to_mesh_conductivity = _LazyCallable(
    ".simnibs_adapter", "tensor_to_mesh_conductivity"
)
run_tdcs = _LazyCallable(".simulation", "run_tdcs")


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
    fit = subparsers.add_parser(
        "fit-dti", help="Fit a tensor to preprocessed single-shell DWI"
    )
    fit.add_argument("data")
    fit.add_argument("bvals")
    fit.add_argument("bvecs")
    fit.add_argument("mask")
    fit.add_argument("output")
    fit.add_argument("--grad-dev")
    fit.add_argument("--shell", type=float)
    fit.add_argument("--tolerance", type=float, default=100.0)
    fit.add_argument("--b0-threshold", type=float, default=0.0)
    fit.add_argument("--z-chunk", type=int, default=4)
    fit.add_argument("--voxel-batch", type=int, default=4096)
    fit.add_argument("--workers", type=int, default=1)
    fit.add_argument(
        "--compatibility-mode",
        choices=("strict-fsl", "robust"),
        default="strict-fsl",
    )
    fit.add_argument("--valid-mask-out")
    fit.add_argument("--qa-json")
    fit.add_argument(
        "--progress",
        choices=("tqdm", "off"),
        default="tqdm",
        help="Show voxel progress with tqdm or disable it explicitly",
    )
    nomoco = subparsers.add_parser(
        "preprocess-nomoco",
        help="Run the SimNIBS 4.6 raw-DWI path without motion or eddy correction",
    )
    nomoco.add_argument("data")
    nomoco.add_argument("bvals")
    nomoco.add_argument("bvecs")
    nomoco.add_argument("output_directory")
    nomoco.add_argument("--grad-dev")
    nomoco.add_argument("--shell", type=float)
    nomoco.add_argument("--tolerance", type=float, default=100.0)
    nomoco.add_argument("--b0-threshold", type=float, default=0.0)
    nomoco.add_argument("--z-chunk", type=int, default=4)
    nomoco.add_argument("--voxel-batch", type=int, default=4096)
    nomoco.add_argument("--workers", type=int, default=8)
    nomoco.add_argument(
        "--compatibility-mode",
        choices=("strict-fsl", "robust"),
        default="strict-fsl",
    )
    nomoco.add_argument(
        "--bet-backend", choices=("reference", "optimized"), default="optimized"
    )
    nomoco.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    legacy = subparsers.add_parser(
        "preprocess-legacy",
        help="Run the SimNIBS 4.6 two-pass legacy motion/eddy correction path",
    )
    legacy.add_argument("data")
    legacy.add_argument("bvals")
    legacy.add_argument("bvecs")
    legacy.add_argument("output_directory")
    legacy.add_argument("--grad-dev")
    legacy.add_argument(
        "--bvec-mode", choices=("compat46", "corrected"), default="compat46"
    )
    legacy.add_argument("--fieldmap-displacement")
    legacy.add_argument("--fieldmap-corrected-mask")
    legacy.add_argument("--fieldmap-magnitude")
    legacy.add_argument("--fieldmap-radians-per-second")
    legacy.add_argument("--fieldmap-dwell-ms", type=float)
    legacy.add_argument(
        "--fieldmap-phase-encoding-direction",
        choices=("x", "x-", "y", "y-", "z", "z-"),
    )
    legacy.add_argument("--shell", type=float)
    legacy.add_argument("--tolerance", type=float, default=100.0)
    legacy.add_argument("--z-chunk", type=int, default=4)
    legacy.add_argument("--voxel-batch", type=int, default=4096)
    legacy.add_argument("--workers", type=int, default=8)
    legacy.add_argument(
        "--compatibility-mode",
        choices=("strict-fsl", "robust"),
        default="strict-fsl",
    )
    legacy.add_argument(
        "--bet-backend", choices=("reference", "optimized"), default="optimized"
    )
    legacy.add_argument("--max-evaluations", type=int, default=1200)
    legacy.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    fieldmap = subparsers.add_parser(
        "prepare-fieldmap",
        help="Prepare the SimNIBS 4.6 GRE/FUGUE path from a rad/s fieldmap",
    )
    fieldmap.add_argument("magnitude")
    fieldmap.add_argument("field_radians_per_second")
    fieldmap.add_argument("b0_brain")
    fieldmap.add_argument("output_directory")
    fieldmap.add_argument("--dwell-ms", type=float, required=True)
    fieldmap.add_argument(
        "--phase-encoding-direction",
        choices=("x", "x-", "y", "y-", "z", "z-"),
        required=True,
    )
    fieldmap.add_argument("--magnitude-mask")
    fieldmap.add_argument("--b0-mask")
    fieldmap.add_argument("--workers", type=int, default=8)
    fieldmap.add_argument(
        "--bet-backend", choices=("reference", "optimized"), default="optimized"
    )
    fieldmap.add_argument("--no-median-filter", action="store_true")
    fieldmap.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    topup = subparsers.add_parser(
        "prepare-topup",
        help="Estimate the fixed SimNIBS 4.6 reverse-PE susceptibility field",
    )
    topup.add_argument("forward_b0")
    topup.add_argument("reverse_b0")
    topup.add_argument("output_directory")
    topup.add_argument("--readout-seconds", type=float, required=True)
    topup.add_argument(
        "--phase-encoding-direction",
        choices=("x", "x-", "y", "y-"),
        required=True,
    )
    topup.add_argument("--workers", type=int, default=8)
    topup.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    eddy = subparsers.add_parser(
        "prepare-eddy",
        help="Run the fixed SimNIBS 4.6 EDDY --repol single-shell subset",
    )
    eddy.add_argument("dwi")
    eddy.add_argument("bvals")
    eddy.add_argument("bvecs")
    eddy.add_argument("brain_mask")
    eddy.add_argument("output_directory")
    eddy.add_argument("--readout-seconds", type=float, required=True)
    eddy.add_argument(
        "--phase-encoding-direction",
        choices=("x", "x-", "y", "y-", "z", "z-"),
        required=True,
    )
    eddy.add_argument("--susceptibility-field")
    eddy.add_argument("--random-seed", type=int, default=1)
    eddy.add_argument("--workers", type=int, default=8)
    eddy.add_argument("--no-repol", action="store_true")
    eddy.add_argument("--no-rigid-shell-alignment", action="store_true")
    eddy.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    register = subparsers.add_parser(
        "register-tensor", help="Map and reorient a tensor to a T1/head-model grid"
    )
    register.add_argument("tensor")
    register.add_argument("reference")
    register.add_argument("output")
    alignment = register.add_mutually_exclusive_group(required=True)
    alignment.add_argument(
        "--world-transform",
        help="External 4x4 input-world to reference-world affine text file",
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
    register.add_argument(
        "--interpolation-order", type=int, choices=(0, 1, 3), default=1
    )
    register_t1 = subparsers.add_parser(
        "register-t1",
        help="Automatically register DTI tensor outputs to a CHARM T1 grid",
    )
    register_t1.add_argument("dti_directory")
    register_t1.add_argument("m2m_directory")
    register_t1.add_argument("output_directory")
    register_t1.add_argument("--mode", choices=("rigid", "affine"), default="affine")
    register_t1.add_argument("--workers", type=int, default=8)
    register_t1.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    nonlinear = subparsers.add_parser(
        "register-t1-nonlinear",
        help="Run the fixed SimNIBS 4.6 FNIRT and nonlinear tensor branch",
    )
    nonlinear.add_argument("fa")
    nonlinear.add_argument("tensor")
    nonlinear.add_argument("reference")
    nonlinear.add_argument("affine_matrix")
    nonlinear.add_argument("output_directory")
    nonlinear.add_argument("--brain-mask", required=True)
    nonlinear.add_argument("--workers", type=int, default=8)
    nonlinear.add_argument(
        "--compatibility-mode",
        choices=("strict-fsl", "robust"),
        default="strict-fsl",
    )
    nonlinear.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    pipeline_qa = subparsers.add_parser(
        "pipeline-qa",
        help="Aggregate P11 DWI, registration, tensor and FEM QA",
    )
    pipeline_qa.add_argument("bvals")
    pipeline_qa.add_argument("original_bvecs")
    pipeline_qa.add_argument("brain_mask")
    pipeline_qa.add_argument("fa")
    pipeline_qa.add_argument("tensor")
    pipeline_qa.add_argument("valid_mask")
    pipeline_qa.add_argument("output_directory")
    pipeline_qa.add_argument("--dwi-brain-mask")
    pipeline_qa.add_argument("--raw-dwi-brain-mask")
    pipeline_qa.add_argument("--corrected-dwi-brain-mask")
    pipeline_qa.add_argument("--raw-dwi")
    pipeline_qa.add_argument("--corrected-dwi")
    pipeline_qa.add_argument("--rotated-bvecs")
    pipeline_qa.add_argument("--sse")
    pipeline_qa.add_argument("--raw-registered-fa")
    pipeline_qa.add_argument("--raw-registered-sse")
    pipeline_qa.add_argument("--t1")
    pipeline_qa.add_argument("--registered-fa")
    pipeline_qa.add_argument("--v1")
    pipeline_qa.add_argument("--field-hz")
    pipeline_qa.add_argument("--jacobian")
    pipeline_qa.add_argument("--eddy-parameters")
    pipeline_qa.add_argument("--outlier-map")
    pipeline_qa.add_argument("--readout-seconds", type=float)
    pipeline_qa.add_argument("--b0-threshold", type=float, default=0.0)
    pipeline_qa.add_argument(
        "--fem-manifest",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help="Add one completed scalar/vn/dir/mc simulation manifest",
    )
    pipeline_qa.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Run the explicit cached raw-DWI to tensor/FEM P11 DAG",
    )
    pipeline.add_argument("data")
    pipeline.add_argument("bvals")
    pipeline.add_argument("bvecs")
    pipeline.add_argument("m2m_directory")
    pipeline.add_argument("output_directory")
    pipeline.add_argument(
        "--preprocessing-mode",
        choices=("nomoco", "legacy", "eddy"),
        default="legacy",
    )
    pipeline.add_argument(
        "--t1-mode", choices=("rigid", "affine", "nonlinear"), default="nonlinear"
    )
    pipeline.add_argument("--grad-dev")
    pipeline.add_argument("--dwi-brain-mask")
    pipeline.add_argument("--reverse-phase-encoding")
    pipeline.add_argument("--susceptibility-field")
    pipeline.add_argument("--fieldmap-corrected-mask")
    pipeline.add_argument("--fieldmap-magnitude")
    pipeline.add_argument("--fieldmap-radians-per-second")
    pipeline.add_argument("--fieldmap-dwell-ms", type=float)
    pipeline.add_argument("--readout-seconds", type=float)
    pipeline.add_argument(
        "--phase-encoding-direction", choices=("x", "x-", "y", "y-", "z", "z-")
    )
    pipeline.add_argument("--random-seed", type=int, default=1)
    pipeline.add_argument("--workers", type=int, default=8)
    pipeline.add_argument(
        "--fit-compatibility-mode",
        choices=("strict-fsl", "robust"),
        default="strict-fsl",
    )
    pipeline.add_argument(
        "--fem-smoke", choices=("none", "dry-run", "run"), default="none"
    )
    pipeline.add_argument("--no-publish-to-m2m", action="store_true")
    pipeline.add_argument(
        "--solver",
        choices=("pardiso", "hypre", "mumps", "petsc_pardiso"),
        default="pardiso",
    )
    pipeline.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    prefit_pipeline = subparsers.add_parser(
        "run-prefit-pipeline",
        help="Run the official pre-fitted tensor import, T1 registration, and FEM DAG",
    )
    prefit_pipeline.add_argument("tensor")
    prefit_pipeline.add_argument("m2m_directory")
    prefit_pipeline.add_argument("output_directory")
    prefit_pipeline.add_argument(
        "--t1-mode", choices=("rigid", "affine", "nonlinear"), default="nonlinear"
    )
    prefit_pipeline.add_argument("--workers", type=int, default=8)
    prefit_pipeline.add_argument(
        "--fem-smoke", choices=("none", "dry-run", "run"), default="none"
    )
    prefit_pipeline.add_argument("--no-publish-to-m2m", action="store_true")
    prefit_pipeline.add_argument(
        "--solver",
        choices=("pardiso", "hypre", "mumps", "petsc_pardiso"),
        default="pardiso",
    )
    prefit_pipeline.add_argument("--progress", choices=("tqdm", "off"), default="tqdm")
    conductivity = subparsers.add_parser(
        "tensor-to-mesh",
        help="Write a SimNIBS mesh with dir/vn/mc conductivity tensors",
    )
    conductivity.add_argument("tensor")
    conductivity.add_argument("mesh")
    conductivity.add_argument("output_mesh")
    conductivity.add_argument("--mode", choices=("dir", "vn", "mc"), default="vn")
    conductivity.add_argument("--aniso-tissues", type=int, nargs="+", default=(1, 2))
    conductivity.add_argument(
        "--cond-json", help="JSON object mapping tissue labels to scalar conductivity"
    )
    conductivity.add_argument("--no-correct-fsl", action="store_true")
    conductivity.add_argument("--max-ratio", type=float, default=10.0)
    conductivity.add_argument("--max-cond", type=float, default=2.0)
    conductivity.add_argument("--excentricity-scaling", type=float)
    conductivity.add_argument("--no-correct-intensity", action="store_true")
    conductivity.add_argument("--qa-json")
    brain_mask = subparsers.add_parser(
        "charm-brain-mask",
        help="Create the official 1..499 brain mask from CHARM labeling",
    )
    brain_mask.add_argument("labeling")
    brain_mask.add_argument("output")
    brain_mask.add_argument("--reference")
    simulate = subparsers.add_parser(
        "simulate-tdcs", help="Run scalar or anisotropic tDCS with SimNIBS 4.6"
    )
    simulate.add_argument("subpath", help="CHARM m2m_<subject> directory")
    simulate.add_argument(
        "output_root", help="Common output root for conductivity modes"
    )
    simulate.add_argument(
        "--mode", choices=("scalar", "vn", "dir", "mc"), required=True
    )
    simulate.add_argument(
        "--tensor", help="Six-component diffusion tensor NIfTI on the T1 grid"
    )
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
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a manifest without solving",
    )
    leadfield = subparsers.add_parser(
        "simulate-leadfield",
        help="Generate scalar/vn/dir/mc lead fields for all EEG-cap electrodes",
    )
    leadfield.add_argument("subpath", help="CHARM m2m_<subject> directory")
    leadfield.add_argument(
        "output_root", help="Common output root for conductivity modes"
    )
    leadfield.add_argument(
        "--mode", choices=("scalar", "vn", "dir", "mc"), required=True
    )
    leadfield.add_argument(
        "--tensor", help="Six-component diffusion tensor NIfTI on the T1 grid"
    )
    leadfield.add_argument("--eeg-cap")
    leadfield.add_argument("--field", choices=("E", "J"), default="E")
    leadfield.add_argument(
        "--interpolation", choices=("none", "middle-gm"), default="none"
    )
    leadfield.add_argument("--tissues", type=int, nargs="+", default=(1, 2))
    leadfield.add_argument("--interpolation-tissues", type=int, nargs="+", default=(2,))
    leadfield.add_argument("--shape", choices=("ellipse", "rect"), default="ellipse")
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
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a manifest without solving",
    )
    montage_plot = subparsers.add_parser(
        "plot-montage", help="Plot a selected montage on the MNE standard 10-20 layout"
    )
    montage_plot.add_argument("output")
    montage_plot.add_argument("--anode", default="C3")
    montage_plot.add_argument("--cathode", default="C4")
    montage_plot.add_argument("--current-ma", type=float, default=1.0)
    montage_plot.add_argument("--shape", choices=("rect", "ellipse"), default="rect")
    montage_plot.add_argument("--dimensions", type=float, nargs=2, default=(50.0, 50.0))
    montage_plot.add_argument("--thickness", type=float, default=4.0)
    montage_plot.add_argument("--montage", default="standard_1020")
    montage_plot.add_argument("--dpi", type=int, default=220)
    montage_plot.add_argument("--svg")
    compare = subparsers.add_parser(
        "compare-fields",
        help="Create a shared-scale comparison from four voxel-level field NIfTIs",
    )
    compare.add_argument("scalar")
    compare.add_argument("vn")
    compare.add_argument("dir")
    compare.add_argument("mc")
    compare.add_argument("anatomy", help="T1 NIfTI used as the grayscale background")
    compare.add_argument("mask", help="CHARM final_tissues or a binary brain mask")
    compare.add_argument(
        "output", help="Output PNG for the 3x4 component or 2x2 magnitude figure"
    )
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
        print(
            f"Done: selected {selected.size} volumes; output: {args.output_data}",
            flush=True,
        )
        return 0
    if args.command == "preprocess-nomoco":
        progress_bar: list[tqdm | None] = [None]
        progress_phase: list[str | None] = [None]

        def report_nomoco(phase: str, done: int, total: int) -> None:
            if args.progress == "off":
                return
            if progress_phase[0] != phase:
                if progress_bar[0] is not None:
                    progress_bar[0].close()
                progress_phase[0] = phase
                progress_bar[0] = tqdm(
                    total=total,
                    desc=phase.replace("_", " "),
                    unit="item",
                    dynamic_ncols=True,
                    mininterval=1.0,
                )
            progress_bar[0].update(done - progress_bar[0].n)

        try:
            report = run_nomoco_nifti(
                args.data,
                args.bvals,
                args.bvecs,
                args.output_directory,
                grad_dev_file=args.grad_dev,
                shell=args.shell,
                tolerance=args.tolerance,
                b0_threshold=args.b0_threshold,
                z_chunk=args.z_chunk,
                voxel_batch=args.voxel_batch,
                workers=args.workers,
                compatibility_mode=args.compatibility_mode,
                bet_backend=args.bet_backend,
                progress=report_nomoco,
            )
        finally:
            if progress_bar[0] is not None:
                progress_bar[0].close()
        print(
            f"{report['status']}: {Path(args.output_directory) / 'nomoco_qa.json'}",
            flush=True,
        )
        return 0
    if args.command == "preprocess-legacy":
        progress_bar: list[tqdm | None] = [None]
        progress_phase: list[str | None] = [None]

        def report_legacy(phase: str, done: int, total: int) -> None:
            if args.progress == "off":
                return
            if progress_phase[0] != phase:
                if progress_bar[0] is not None:
                    progress_bar[0].close()
                progress_phase[0] = phase
                progress_bar[0] = tqdm(
                    total=total,
                    desc=phase.replace("_", " "),
                    unit="item",
                    dynamic_ncols=True,
                    mininterval=1.0,
                )
            progress_bar[0].update(done - progress_bar[0].n)

        try:
            report = run_legacy_nifti(
                args.data,
                args.bvals,
                args.bvecs,
                args.output_directory,
                grad_dev_file=args.grad_dev,
                bvec_mode=args.bvec_mode,
                fieldmap_displacement_file=args.fieldmap_displacement,
                fieldmap_corrected_mask_file=args.fieldmap_corrected_mask,
                fieldmap_magnitude_file=args.fieldmap_magnitude,
                fieldmap_radians_per_second_file=args.fieldmap_radians_per_second,
                fieldmap_dwell_milliseconds=args.fieldmap_dwell_ms,
                fieldmap_phase_encoding_direction=(
                    args.fieldmap_phase_encoding_direction
                ),
                shell=args.shell,
                tolerance=args.tolerance,
                z_chunk=args.z_chunk,
                voxel_batch=args.voxel_batch,
                workers=args.workers,
                compatibility_mode=args.compatibility_mode,
                bet_backend=args.bet_backend,
                max_evaluations=args.max_evaluations,
                progress=report_legacy,
            )
        finally:
            if progress_bar[0] is not None:
                progress_bar[0].close()
        print(
            f"{report['status']}: {Path(args.output_directory) / 'legacy_qa.json'}",
            flush=True,
        )
        return 0
    if args.command == "prepare-fieldmap":
        bar = tqdm(
            total=4,
            desc="GRE fieldmap",
            unit="stage",
            dynamic_ncols=True,
            disable=args.progress == "off",
        )
        try:
            report = run_fieldmap_nifti(
                args.magnitude,
                args.field_radians_per_second,
                args.b0_brain,
                args.output_directory,
                dwell_milliseconds=args.dwell_ms,
                phase_encoding_direction=args.phase_encoding_direction,
                magnitude_mask_file=args.magnitude_mask,
                b0_mask_file=args.b0_mask,
                workers=args.workers,
                bet_backend=args.bet_backend,
                median_filter=not args.no_median_filter,
                progress=lambda _phase, done, _total: bar.update(done - bar.n),
            )
        finally:
            bar.close()
        print(
            f"{report['status']}: {Path(args.output_directory) / 'fieldmap_qa.json'}",
            flush=True,
        )
        return 0
    if args.command == "prepare-topup":
        bar = tqdm(
            total=len(range(1, 10)),
            desc="TOPUP",
            unit="level",
            dynamic_ncols=True,
            disable=args.progress == "off",
        )
        completed_levels: set[int] = set()

        def report_topup(level: int, phase: str) -> None:
            if phase == "complete" and level not in completed_levels:
                completed_levels.add(level)
                bar.update(1)

        try:
            report = run_topup_nifti(
                args.forward_b0,
                args.reverse_b0,
                args.output_directory,
                readout_seconds=args.readout_seconds,
                phase_encoding_direction=args.phase_encoding_direction,
                workers=args.workers,
                progress=report_topup,
            )
        finally:
            bar.close()
        print(
            f"{report['status']}: {Path(args.output_directory) / 'topup_qa.json'}",
            flush=True,
        )
        return 0
    if args.command == "prepare-eddy":
        bar = tqdm(
            total=4,
            desc="EDDY",
            unit="stage",
            dynamic_ncols=True,
            disable=args.progress == "off",
        )
        completed_phases: set[str] = set()

        def report_eddy(phase: str, done: int, total: int) -> None:
            if done >= total and phase not in completed_phases:
                completed_phases.add(phase)
                bar.update(1)

        try:
            report = run_eddy_nifti(
                args.dwi,
                args.bvals,
                args.bvecs,
                args.brain_mask,
                args.output_directory,
                readout_seconds=args.readout_seconds,
                phase_encoding_direction=args.phase_encoding_direction,
                susceptibility_field_file=args.susceptibility_field,
                random_seed=args.random_seed,
                workers=args.workers,
                replace_outliers=not args.no_repol,
                align_shells_post_eddy=not args.no_rigid_shell_alignment,
                progress=report_eddy,
            )
        finally:
            bar.close()
        print(
            f"{report['status']}: {Path(args.output_directory) / 'eddy_qa.json'}",
            flush=True,
        )
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
    if args.command == "register-t1":
        dti_directory = Path(args.dti_directory)
        m2m_directory = Path(args.m2m_directory)
        required = {
            "tensor": dti_directory / "DTI_tensor.nii.gz",
            "fa": dti_directory / "DTI_FA.nii.gz",
            "t1": m2m_directory / "T1.nii.gz",
            "labeling": m2m_directory / "segmentation" / "labeling.nii.gz",
            "bias_corrected": (
                m2m_directory / "segmentation" / "T1_bias_corrected.nii.gz"
            ),
            "final_tissues": m2m_directory / "final_tissues.nii.gz",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required dwi2cond/CHARM inputs: " + ", ".join(missing)
            )
        progress_bar: list[tqdm | None] = [None]
        progress_phase: list[str | None] = [None]

        def report_t1(phase: str, done: int, total: int) -> None:
            if args.progress == "off":
                return
            if progress_phase[0] != phase:
                if progress_bar[0] is not None:
                    progress_bar[0].close()
                progress_phase[0] = phase
                progress_bar[0] = tqdm(
                    total=total,
                    desc=phase.replace("_", " "),
                    unit="item",
                    dynamic_ncols=True,
                    mininterval=1.0,
                )
            progress_bar[0].update(done - progress_bar[0].n)

        sse = dti_directory / "DTI_sse.nii.gz"
        try:
            report = run_t1_registration_nifti(
                required["tensor"],
                required["fa"],
                required["t1"],
                required["labeling"],
                required["bias_corrected"],
                args.output_directory,
                sse_file=sse if sse.is_file() else None,
                degrees_of_freedom=6 if args.mode == "rigid" else 12,
                workers=args.workers,
                progress=report_t1,
            )
        finally:
            if progress_bar[0] is not None:
                progress_bar[0].close()
        print(
            f"{report['status']}: {Path(args.output_directory) / 't1_registration_qa.json'}",
            flush=True,
        )
        return 0
    if args.command == "register-t1-nonlinear":
        bar = tqdm(
            total=4,
            desc="FNIRT",
            unit="level",
            dynamic_ncols=True,
            disable=args.progress == "off",
        )
        completed_levels: set[int] = set()
        detail_bar: list[object | None] = [None]
        detail_level: list[int | None] = [None]
        detail_iteration = [0]
        finalize_phases = {
            "finalize_write": "write warp / field / Jacobian / registered FA",
            "finalize_tensor": "resample, reorient, and write PPD tensor",
            "finalize_qa": "generate derivatives and QA",
            "finalize_complete": "post-processing complete",
        }

        def report_fnirt(
            level: int,
            phase: str,
            done: int,
            total: int,
            value: float | None,
        ) -> None:
            if phase in finalize_phases:
                if detail_level[0] != 0:
                    if detail_bar[0] is not None:
                        detail_bar[0].close()
                    detail_level[0] = 0
                    detail_bar[0] = tqdm(
                        total=3,
                        initial=min(done, 3),
                        desc="FNIRT post-processing",
                        unit="stage",
                        dynamic_ncols=True,
                        mininterval=0.5,
                        leave=False,
                        disable=args.progress == "off",
                    )
                else:
                    detail_bar[0].update(done - detail_bar[0].n)
                # Post-processing has few phases, so refresh immediately on transitions
                # to keep long-running tasks from displaying the previous phase.
                detail_bar[0].set_postfix_str(finalize_phases[phase])
                if phase == "finalize_complete":
                    detail_bar[0].close()
                    detail_bar[0] = None
                    detail_level[0] = None
            elif phase in ("gradient", "hessian", "pcg", "lm", "topology"):
                if detail_level[0] != level:
                    if detail_bar[0] is not None:
                        detail_bar[0].close()
                    detail_level[0] = level
                    detail_iteration[0] = 0
                    detail_bar[0] = tqdm(
                        total=5,
                        desc=f"FNIRT L{level}/4",
                        unit="iter",
                        dynamic_ncols=True,
                        mininterval=0.5,
                        leave=False,
                        disable=args.progress == "off",
                    )
                if phase in ("gradient", "hessian", "lm"):
                    detail_iteration[0] = done
                    completed = max(0, min(done - 1, 4))
                    detail_bar[0].update(completed - detail_bar[0].n)
                label = f"{phase}; iteration={detail_iteration[0]}/5"
                if phase in ("pcg", "topology"):
                    label += f"; step={done}/{total}"
                if value is not None:
                    label += f"; value={value:.6g}"
                # The postfix should not force a refresh; update(0) lets tqdm throttle
                # output according to mininterval.
                detail_bar[0].set_postfix_str(label, refresh=False)
                detail_bar[0].update(0)
            if phase == "complete" and level not in completed_levels:
                if detail_bar[0] is not None and detail_level[0] == level:
                    detail_bar[0].update(5 - detail_bar[0].n)
                    detail_bar[0].close()
                    detail_bar[0] = None
                    detail_level[0] = None
                completed_levels.add(level)
                bar.update(1)

        try:
            report = register_tensor_fnirt_nifti(
                args.fa,
                args.tensor,
                args.reference,
                args.affine_matrix,
                args.output_directory,
                brain_mask_file=args.brain_mask,
                workers=args.workers,
                compatibility_mode=args.compatibility_mode,
                progress=report_fnirt,
            )
        finally:
            if detail_bar[0] is not None:
                detail_bar[0].close()
            bar.close()
        print(
            f"{report['status']}: "
            f"{Path(args.output_directory) / 'nonlinear_registration_qa.json'}",
            flush=True,
        )
        return 0
    if args.command == "pipeline-qa":
        qa_module = import_module(".preprocessing.qa", __package__)
        fem_manifests: dict[str, Path] = {}
        for specification in args.fem_manifest:
            mode, separator, raw_path = specification.partition("=")
            if separator != "=" or mode not in ("scalar", "vn", "dir", "mc"):
                raise ValueError("--fem-manifest must use scalar/vn/dir/mc=PATH")
            if mode in fem_manifests:
                raise ValueError(f"Duplicate FEM manifest mode: {mode}")
            fem_manifests[mode] = Path(raw_path)
        bar = tqdm(
            total=8,
            desc="Pipeline QA",
            unit="stage",
            dynamic_ncols=True,
            disable=args.progress == "off",
        )
        qa_completed = [0]

        def report_qa(phase: str, done: int, _total: int) -> None:
            bar.set_postfix_str(phase, refresh=False)
            bar.update(done - qa_completed[0])
            qa_completed[0] = done

        try:
            report = qa_module.build_pipeline_qa(
                qa_module.PipelineQaInputs(
                    bvals=Path(args.bvals),
                    original_bvecs=Path(args.original_bvecs),
                    brain_mask=Path(args.brain_mask),
                    fa=Path(args.fa),
                    tensor=Path(args.tensor),
                    valid_mask=Path(args.valid_mask),
                    dwi_brain_mask=(
                        None
                        if args.dwi_brain_mask is None
                        else Path(args.dwi_brain_mask)
                    ),
                    raw_dwi_brain_mask=(
                        None
                        if args.raw_dwi_brain_mask is None
                        else Path(args.raw_dwi_brain_mask)
                    ),
                    corrected_dwi_brain_mask=(
                        None
                        if args.corrected_dwi_brain_mask is None
                        else Path(args.corrected_dwi_brain_mask)
                    ),
                    raw_dwi=None if args.raw_dwi is None else Path(args.raw_dwi),
                    corrected_dwi=(
                        None if args.corrected_dwi is None else Path(args.corrected_dwi)
                    ),
                    rotated_bvecs=(
                        None if args.rotated_bvecs is None else Path(args.rotated_bvecs)
                    ),
                    sse=None if args.sse is None else Path(args.sse),
                    raw_registered_fa=(
                        None
                        if args.raw_registered_fa is None
                        else Path(args.raw_registered_fa)
                    ),
                    raw_registered_sse=(
                        None
                        if args.raw_registered_sse is None
                        else Path(args.raw_registered_sse)
                    ),
                    t1=None if args.t1 is None else Path(args.t1),
                    registered_fa=(
                        None if args.registered_fa is None else Path(args.registered_fa)
                    ),
                    v1=None if args.v1 is None else Path(args.v1),
                    field_hz=(None if args.field_hz is None else Path(args.field_hz)),
                    jacobian=(None if args.jacobian is None else Path(args.jacobian)),
                    eddy_parameters=(
                        None
                        if args.eddy_parameters is None
                        else Path(args.eddy_parameters)
                    ),
                    outlier_map=(
                        None if args.outlier_map is None else Path(args.outlier_map)
                    ),
                    readout_seconds=args.readout_seconds,
                    fem_manifests=fem_manifests,
                ),
                args.output_directory,
                b0_threshold=args.b0_threshold,
                progress=report_qa,
            )
        finally:
            bar.close()
        print(
            f"{report['status']}: {Path(args.output_directory) / 'pipeline_qa.json'}",
            flush=True,
        )
        return 0
    if args.command in ("run-pipeline", "run-prefit-pipeline"):
        workflow_module = import_module(".preprocessing.workflow", __package__)
        is_prefit = args.command == "run-prefit-pipeline"
        expected_stages = 3
        if not is_prefit:
            expected_stages += 1
        if not is_prefit and args.preprocessing_mode == "eddy":
            expected_stages += 1
        if args.t1_mode == "nonlinear":
            expected_stages += 1
        if not args.no_publish_to_m2m:
            expected_stages += 1
        if args.fem_smoke != "none":
            expected_stages += 4
        bar = tqdm(
            total=expected_stages,
            desc="dwi2cond DAG",
            unit="stage",
            dynamic_ncols=True,
            disable=args.progress == "off",
        )
        completed_stages: set[str] = set()
        stage_started: dict[str, float] = {}
        detail_bar: list[object | None] = [None]
        detail_stage: list[str | None] = [None]
        detail_iteration = [0]
        finalize_phases = {
            "finalize_write": "write warp / field / Jacobian / registered FA",
            "finalize_tensor": "resample, reorient, and write PPD tensor",
            "finalize_qa": "generate derivatives and QA",
            "finalize_complete": "post-processing complete",
        }

        def report_pipeline(stage: str, done: int, total: int, status: str) -> None:
            parts = stage.split(":")
            is_fnirt_detail = len(parts) == 3 and parts[0] == "register_nonlinear"
            if is_fnirt_detail:
                level_stage = ":".join(parts[:2])
                phase = parts[2]
                if phase in finalize_phases:
                    finalize_stage = "register_nonlinear:finalize"
                    if detail_stage[0] != finalize_stage:
                        if detail_bar[0] is not None:
                            detail_bar[0].close()
                        detail_stage[0] = finalize_stage
                        detail_bar[0] = tqdm(
                            total=3,
                            initial=min(done, 3),
                            desc="FNIRT post-processing",
                            unit="stage",
                            dynamic_ncols=True,
                            mininterval=0.5,
                            leave=False,
                            disable=args.progress == "off",
                        )
                    else:
                        detail_bar[0].update(done - detail_bar[0].n)
                    # Post-processing has few phases, so refresh immediately on transitions
                    # to keep long-running tasks from displaying the previous phase.
                    detail_bar[0].set_postfix_str(finalize_phases[phase])
                    if phase == "finalize_complete":
                        detail_bar[0].close()
                        detail_bar[0] = None
                        detail_stage[0] = None
                elif phase == "complete":
                    if detail_bar[0] is not None and detail_stage[0] == level_stage:
                        detail_bar[0].update(5 - detail_bar[0].n)
                        detail_bar[0].close()
                        detail_bar[0] = None
                        detail_stage[0] = None
                elif phase in ("gradient", "hessian", "pcg", "lm", "topology"):
                    if detail_stage[0] != level_stage:
                        if detail_bar[0] is not None:
                            detail_bar[0].close()
                        detail_stage[0] = level_stage
                        detail_iteration[0] = 0
                        detail_bar[0] = tqdm(
                            total=5,
                            desc=level_stage.replace(
                                "register_nonlinear:level_", "FNIRT L"
                            )
                            + "/4",
                            unit="iter",
                            dynamic_ncols=True,
                            mininterval=0.5,
                            leave=False,
                            disable=args.progress == "off",
                        )
                    if phase in ("gradient", "hessian", "lm"):
                        detail_iteration[0] = done
                        completed = max(0, min(done - 1, 4))
                        detail_bar[0].update(completed - detail_bar[0].n)
                    label = f"{phase}; iteration={detail_iteration[0]}/5"
                    if phase in ("pcg", "topology"):
                        label += f"; step={done}/{total}"
                    if ";" in status:
                        label += ";" + status.split(";", 1)[1]
                    # PCG callbacks are frequent, so avoid letting every set_postfix call
                    # bypass tqdm's output throttling.
                    detail_bar[0].set_postfix_str(label, refresh=False)
                    detail_bar[0].update(0)
            elif ":" in stage and total > 1:
                # The eight pipeline_qa phases share one continuous progress bar and must
                # not close and recreate it per phase. Other iterative substages retain
                # their own independent progress bars.
                progress_stage = (
                    "pipeline_qa" if stage.startswith("pipeline_qa:") else stage
                )
                if detail_stage[0] != progress_stage:
                    if detail_bar[0] is not None:
                        detail_bar[0].close()
                    detail_stage[0] = progress_stage
                    detail_bar[0] = tqdm(
                        total=total,
                        initial=min(done, total),
                        desc=progress_stage.replace("register_nonlinear:", "FNIRT "),
                        unit="iter",
                        dynamic_ncols=True,
                        leave=False,
                        disable=args.progress == "off",
                    )
                else:
                    detail_bar[0].update(done - detail_bar[0].n)
                detail_bar[0].set_postfix_str(status, refresh=False)
                detail_bar[0].update(0)
            if ":" not in stage and status == "running":
                bar.set_postfix_str(f"{stage}: {status}", refresh=False)
                stage_started[stage] = time.perf_counter()
                bar.write(
                    f"▶ [{len(completed_stages) + 1}/{expected_stages}] {stage} started"
                )
            if (
                ":" not in stage
                and status in ("completed", "cached")
                and stage not in completed_stages
            ):
                if detail_bar[0] is not None:
                    detail_bar[0].close()
                    detail_bar[0] = None
                    detail_stage[0] = None
                completed_stages.add(stage)
                bar.set_postfix_str(f"{stage}: {status}", refresh=False)
                bar.update(1)
                elapsed = time.perf_counter() - stage_started.get(
                    stage, time.perf_counter()
                )
                suffix = "cache hit" if status == "cached" else f"{elapsed:.2f} s"
                bar.write(
                    f"✓ [{len(completed_stages)}/{expected_stages}] "
                    f"{stage} completed, {suffix}"
                )

        try:
            result = workflow_module.run_dwi2cond_pipeline(
                workflow_module.Dwi2CondPipelineConfig(
                    data=None if is_prefit else Path(args.data),
                    bvals=None if is_prefit else Path(args.bvals),
                    bvecs=None if is_prefit else Path(args.bvecs),
                    m2m_directory=Path(args.m2m_directory),
                    output_directory=Path(args.output_directory),
                    prefit_tensor=Path(args.tensor) if is_prefit else None,
                    preprocessing_mode=("prefit" if is_prefit else args.preprocessing_mode),
                    t1_mode=args.t1_mode,
                    grad_dev=(
                        None
                        if is_prefit or args.grad_dev is None
                        else Path(args.grad_dev)
                    ),
                    dwi_brain_mask=(
                        None
                        if is_prefit or args.dwi_brain_mask is None
                        else Path(args.dwi_brain_mask)
                    ),
                    reverse_phase_encoding=(
                        None
                        if is_prefit or args.reverse_phase_encoding is None
                        else Path(args.reverse_phase_encoding)
                    ),
                    susceptibility_field=(
                        None
                        if is_prefit or args.susceptibility_field is None
                        else Path(args.susceptibility_field)
                    ),
                    fieldmap_corrected_mask=(
                        None
                        if is_prefit or args.fieldmap_corrected_mask is None
                        else Path(args.fieldmap_corrected_mask)
                    ),
                    fieldmap_magnitude=(
                        None
                        if is_prefit or args.fieldmap_magnitude is None
                        else Path(args.fieldmap_magnitude)
                    ),
                    fieldmap_radians_per_second=(
                        None
                        if is_prefit or args.fieldmap_radians_per_second is None
                        else Path(args.fieldmap_radians_per_second)
                    ),
                    fieldmap_dwell_milliseconds=(
                        None if is_prefit else args.fieldmap_dwell_ms
                    ),
                    readout_seconds=None if is_prefit else args.readout_seconds,
                    phase_encoding_direction=(
                        None if is_prefit else args.phase_encoding_direction
                    ),
                    random_seed=1 if is_prefit else args.random_seed,
                    workers=args.workers,
                    fit_compatibility_mode=(
                        "strict-fsl" if is_prefit else args.fit_compatibility_mode
                    ),
                    publish_to_m2m=not args.no_publish_to_m2m,
                    fem_smoke=args.fem_smoke,
                    solver=args.solver,
                ),
                progress=report_pipeline,
            )
        finally:
            if detail_bar[0] is not None:
                detail_bar[0].close()
            bar.close()
        print(f"completed: {result.qa_manifest}", flush=True)
        print(f"final tensor: {result.final_tensor}", flush=True)
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
            compatibility_mode=args.compatibility_mode,
            progress=report,
            valid_mask_file=args.valid_mask_out,
            qa_file=args.qa_json,
        )
    finally:
        if progress_bar[0] is not None:
            progress_bar[0].close()
    print(f"Done: {args.output}", flush=True)
    return 0
