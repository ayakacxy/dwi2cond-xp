"""Closed-loop reverse-PE, TOPUP, and EDDY pipeline for SimNIBS 4.6."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .brain_mask import write_bet_brain_mask
from .eddy import run_eddy_nifti
from .orientation import write_fsl_reoriented
from .rigid import write_aligned_b0_mean
from .topup import run_topup_nifti


TopupEddyProgress = Callable[[str, int, int], None]


def _save_first_corrected_b0(pair_file: Path, output_file: Path) -> Path:
    """Take the first b0 from the TOPUP-corrected pair, as used by the official BET stage."""

    image = nib.load(str(pair_file))
    if image.shape[-1] != 2:
        raise ValueError("TOPUP corrected pair must contain exactly two volumes")
    values = np.asarray(image.dataobj[..., 0], dtype=np.float32)
    header = image.header.copy()
    header.set_data_shape(values.shape)
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(values, image.affine, header)
    output.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output.set_sform(image.get_sform(), int(image.header["sform_code"]))
    nib.save(output, str(output_file))
    return output_file


def run_topup_eddy_nifti(
    dwi_file: str | Path,
    bvals_file: str | Path,
    bvecs_file: str | Path,
    reverse_phase_encoding_file: str | Path,
    output_directory: str | Path,
    *,
    readout_seconds: float,
    phase_encoding_direction: str,
    random_seed: int = 1,
    workers: int = 8,
    bet_backend: str = "optimized",
    progress: TopupEddyProgress | None = None,
) -> dict[str, object]:
    """Execute the official reverse 4D preparation, TOPUP, BET, and EDDY sequence."""

    if not np.isfinite(readout_seconds) or not 0.01 <= readout_seconds <= 0.2:
        raise ValueError("readout seconds must be finite and within [0.01, 0.2]")
    if phase_encoding_direction not in ("x", "x-", "y", "y-"):
        raise ValueError("TOPUP phase-encoding direction must be x, x-, y, or y-")
    root = Path(output_directory)
    preparation = root / "topup_preparation"
    topup = root / "topup"
    eddy = root / "eddy"
    preparation.mkdir(parents=True, exist_ok=True)

    forward_reoriented = write_fsl_reoriented(
        dwi_file,
        preparation / "DWIraw.nii",
        float32=True,
        nonnegative=False,
    )
    reverse_reoriented = write_fsl_reoriented(
        reverse_phase_encoding_file,
        preparation / "reverse_pe_reoriented.nii",
        float32=True,
        nonnegative=False,
    )
    reverse_image = nib.load(str(reverse_reoriented))
    if len(reverse_image.shape) != 4:
        raise ValueError("reverse phase-encoding input must be four-dimensional")
    reverse_bvals = preparation / "reverse_pe_bvals"
    np.savetxt(
        reverse_bvals,
        np.zeros((1, reverse_image.shape[3]), dtype=np.float64),
        fmt="%.10g",
    )

    if progress is not None:
        progress("forward_b0_mean", 0, 1)
    forward_mean = write_aligned_b0_mean(
        forward_reoriented,
        bvals_file,
        preparation / "forward_b0_mean.nii.gz",
        b0_threshold=0.0,
        workers=workers,
    )
    if progress is not None:
        progress("forward_b0_mean", 1, 1)
        progress("reverse_b0_mean", 0, 1)
    reverse_mean = write_aligned_b0_mean(
        reverse_reoriented,
        reverse_bvals,
        preparation / "reverse_b0_mean.nii.gz",
        b0_threshold=0.0,
        workers=workers,
    )
    if progress is not None:
        progress("reverse_b0_mean", 1, 1)

    topup_report = run_topup_nifti(
        forward_mean,
        reverse_mean,
        topup,
        readout_seconds=readout_seconds,
        phase_encoding_direction=phase_encoding_direction,
        workers=workers,
        progress=(
            None
            if progress is None
            else lambda level, _phase: progress("topup", level, 9)
        ),
    )
    corrected_b0 = _save_first_corrected_b0(
        topup / "corrected_pair.nii.gz",
        preparation / "topup_corrected_b0.nii.gz",
    )
    corrected_mask = preparation / "topup_corrected_b0_brain_mask.nii.gz"
    write_bet_brain_mask(
        corrected_b0,
        corrected_mask,
        fractional_threshold=0.2,
        workers=workers,
        backend=bet_backend,
    )
    if progress is not None:
        progress("topup_corrected_b0_bet", 1, 1)

    eddy_report = run_eddy_nifti(
        dwi_file,
        bvals_file,
        bvecs_file,
        corrected_mask,
        eddy,
        readout_seconds=readout_seconds,
        phase_encoding_direction=phase_encoding_direction,
        susceptibility_field_file=topup / "field_hz.nii.gz",
        random_seed=random_seed,
        workers=workers,
        progress=progress,
    )
    report: dict[str, object] = {
        "status": "completed",
        "algorithm": "SimNIBS-4.6-reverse-PE-TOPUP-EDDY-fixed-subset",
        "reverse_phase_encoding_volumes": int(reverse_image.shape[3]),
        "readout_seconds": float(readout_seconds),
        "phase_encoding_direction": phase_encoding_direction,
        "workers": workers,
        "topup": topup_report,
        "eddy": eddy_report,
        "artifacts": {
            "forward_b0_mean": str(forward_mean),
            "reverse_b0_mean": str(reverse_mean),
            "topup_coefficients": str(topup / "field_coefficients.nii.gz"),
            "topup_movement": str(topup / "movement_parameters.txt"),
            "topup_field_hz": str(topup / "field_hz.nii.gz"),
            "corrected_b0_mask": str(corrected_mask),
            "corrected_dwi": str(eddy / "corrected_dwi.nii.gz"),
        },
    }
    (root / "topup_eddy_qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
