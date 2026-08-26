#!/usr/bin/env python3
"""Compare each Python legacy-registration stage with a frozen FSL work directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dwi2cond_xp.preprocessing.legacy import (
    _float32_mean,
    _register_mcflirt_series,
    _resample_series,
)
from dwi2cond_xp.preprocessing.flirt_registration import (
    _level_evaluator,
    _search,
    register_flirt_nosearch_mutual_information,
)
from dwi2cond_xp.preprocessing.flirt_pyramid import build_flirt_pyramid
from dwi2cond_xp.preprocessing.rigid import write_aligned_b0_mean


def _load(path: Path) -> np.ndarray:
    """Load one NIfTI as float64 for stable diagnostic norms."""

    return np.asarray(nib.load(path).dataobj, dtype=np.float64)


def _error(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return absolute and relative L2 stage errors."""

    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    denominator = np.linalg.norm(np.asarray(reference, dtype=np.float64).ravel())
    return {
        "relative_l2": float(np.linalg.norm(delta.ravel()) / denominator),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _mask_dice(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return the Dice score for two binary mask interpretations."""

    candidate_mask = np.asarray(candidate) > 0
    reference_mask = np.asarray(reference) > 0
    denominator = np.count_nonzero(candidate_mask) + np.count_nonzero(reference_mask)
    if denominator == 0:
        return 1.0
    return float(
        2.0 * np.count_nonzero(candidate_mask & reference_mask) / denominator
    )


def _metric_value(report: dict[str, object], dotted_path: str) -> object:
    """Resolve one metric from a dotted report path."""

    value: object = report
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"The threshold metric is absent: {dotted_path}")
        value = value[component]
    return value


def _evaluate_thresholds(
    report: dict[str, object], thresholds: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Evaluate explicit maximum or minimum thresholds and return failures."""

    failures: list[dict[str, object]] = []
    for threshold in thresholds:
        metric = str(threshold.get("metric", ""))
        operator = str(threshold.get("operator", ""))
        limit = threshold["value"]
        observed = _metric_value(report, metric)
        if operator in ("<=", ">="):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or isinstance(limit, bool)
                or not isinstance(limit, (int, float))
            ):
                raise ValueError(f"The ordered threshold is not numeric: {metric}")
            passed = observed <= limit if operator == "<=" else observed >= limit
        elif operator == "==":
            passed = observed == limit
        else:
            raise ValueError(
                f"The threshold operator for {metric} must be <=, >=, or =="
            )
        if not passed:
            failures.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "limit": limit,
                    "observed": observed,
                }
            )
    return failures


def _final_metrics(
    candidate: Path, fsl_preprocessing: Path, fsl_matrix_work: Path
) -> dict[str, object]:
    """Compare completed legacy outputs on their common public fixture grid."""

    fsl_fit = fsl_preprocessing / "dti_results_rawspace"
    fsl_eddy = fsl_preprocessing / "eddycorr"
    candidate_mask = _load(candidate / "nodif_brain_mask.nii.gz")
    reference_mask = _load(fsl_fit / "nodif_brain_mask.nii.gz")
    common_mask = (candidate_mask > 0) & (reference_mask > 0)
    if not np.any(common_mask):
        raise ValueError("The candidate and FSL masks do not overlap")
    matrix_errors = [
        float(
            np.max(
                np.abs(
                    np.loadtxt(matrix)
                    - np.loadtxt(fsl_matrix_work / "DWI_corr.mat" / matrix.name)
                )
            )
        )
        for matrix in sorted((candidate / "DWI_corr.mat").glob("MAT_*"))
    ]
    if not matrix_errors:
        raise ValueError("The candidate final matrix directory is empty")
    tensor_candidate = _load(candidate / "DTI_tensor.nii.gz")
    tensor_reference = _load(fsl_fit / "DTI_tensor.nii.gz")
    fa_candidate = _load(candidate / "DTI_FA.nii.gz")
    fa_reference = _load(fsl_fit / "DTI_FA.nii.gz")
    sse_candidate = _load(candidate / "DTI_sse.nii.gz")
    sse_reference = _load(fsl_fit / "DTI_sse.nii.gz")
    return {
        "corrected_dwi": _error(
            _load(candidate / "DWI_corr.nii"),
            _load(fsl_eddy / "DWI_corr.nii.gz"),
        ),
        "corrected_mean": _error(
            _load(candidate / "DWI_corr_mean.nii.gz"),
            _load(fsl_eddy / "DWI_corr_mean.nii.gz"),
        ),
        "mask_dice": _mask_dice(candidate_mask, reference_mask),
        "final_matrix_max_abs": max(matrix_errors),
        "common_mask_tensor": _error(
            tensor_candidate[common_mask], tensor_reference[common_mask]
        ),
        "common_mask_fa": _error(fa_candidate[common_mask], fa_reference[common_mask]),
        "common_mask_sse": _error(
            sse_candidate[common_mask], sse_reference[common_mask]
        ),
        "compat46_bvec_byte_exact": (
            (candidate / "DWIbvecs").read_bytes()
            == (fsl_fit / "DWIbvecs").read_bytes()
        ),
    }


def main() -> int:
    """Run the candidate stages and emit one machine-readable comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fsl-work", type=Path, required=True)
    parser.add_argument("--fsl-nodif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--candidate-final",
        type=Path,
        help="completed candidate legacy output directory",
    )
    parser.add_argument(
        "--fsl-preprocessing",
        type=Path,
        help="completed FSL dMRI_prep directory",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="JSON file containing release_gate.thresholds",
    )
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    fsl = args.fsl_work.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dwi_file = fixture / "dwi.nii.gz"
    bvals_file = fixture / "bvals"
    image = nib.load(dwi_file)
    bvals = np.asarray(np.loadtxt(bvals_file), dtype=np.float64).reshape(-1)
    volumes = [
        np.asarray(image.dataobj[..., index], dtype=np.float32)
        for index in range(image.shape[3])
    ]
    b0_indices = np.flatnonzero(bvals == 0.0)
    diffusion_indices = np.flatnonzero(bvals > 0.0)

    nodif_file = output / "nodif.nii.gz"
    write_aligned_b0_mean(
        dwi_file,
        bvals_file,
        nodif_file,
        b0_threshold=0.0,
        workers=args.workers,
    )
    nodif = _load(nodif_file).astype(np.float32)
    raw_mean = _float32_mean(volumes, diffusion_indices)
    pass1, _, _ = _register_mcflirt_series(
        volumes,
        raw_mean,
        image.affine,
        degrees_of_freedom=6,
        workers=args.workers,
    )
    diffusion_volumes = [volumes[int(index)] for index in diffusion_indices]
    pass1_images = _resample_series(
        diffusion_volumes,
        image.affine,
        [pass1[int(index)] for index in diffusion_indices],
        interpolation="linear",
        workers=args.workers,
    )
    pass1_all = _resample_series(
        volumes,
        image.affine,
        pass1,
        interpolation="linear",
        workers=args.workers,
    )
    pass1_mean = _float32_mean(
        pass1_images, np.arange(len(pass1_images), dtype=np.int64)
    )
    pass2, _, _ = _register_mcflirt_series(
        volumes,
        pass1_mean,
        image.affine,
        degrees_of_freedom=12,
        workers=args.workers,
    )
    pass2_all = _resample_series(
        volumes,
        image.affine,
        pass2,
        interpolation="sinc",
        workers=args.workers,
    )
    pass2_diffusion = [pass2_all[int(index)] for index in diffusion_indices]
    pass2_mean = _float32_mean(
        pass2_diffusion, np.arange(len(pass2_diffusion), dtype=np.int64)
    )
    nib.save(
        nib.Nifti1Image(pass2_mean, image.affine, image.header),
        output / "candidate_pass2_mean.nii.gz",
    )
    sampling = np.diag([*nib.affines.voxel_sizes(image.affine), 1.0])
    mean_registration = register_flirt_nosearch_mutual_information(
        nodif,
        pass2_mean,
        sampling,
        sampling,
        degrees_of_freedom=12,
        workers=args.workers,
    )
    unit_weight = np.ones(nodif.shape, dtype=np.float32)
    levels = build_flirt_pyramid(
        nodif,
        pass2_mean,
        unit_weight,
        unit_weight,
        sampling,
        sampling,
        use_weights=False,
    )
    search_optimized, search_preoptimized, _ = _search(
        levels[8],
        np.eye(4),
        12,
        args.workers,
        None,
        "correlation_ratio",
        True,
    )
    search_evaluator = _level_evaluator(
        levels[8], 8.0, "correlation_ratio"
    )
    fsl_search_matrix = np.diag([0.979757, 0.979757, 0.979757, 1.0])
    fsl_search_matrix[:3, 3] = [0.384457, 0.384733, 0.384727]
    direct_b0, _, _ = _register_mcflirt_series(
        volumes,
        nodif,
        image.affine,
        degrees_of_freedom=6,
        workers=args.workers,
    )

    report: dict[str, object] = {
        "workers": args.workers,
        "nodif": _error(nodif, _load(args.fsl_nodif.resolve())),
        "raw_mean": _error(raw_mean, _load(fsl / "DWIraw_mean.nii.gz")),
        "pass1_series": _error(
            np.stack(pass1_all, axis=3), _load(fsl / "DWI_pass1_corr.nii.gz")
        ),
        "pass1_mean": _error(pass1_mean, _load(fsl / "DWI_pass1_mean.nii.gz")),
        "pass2_series": _error(
            np.stack(pass2_all, axis=3), _load(fsl / "DWI_corr.nii.gz")
        ),
        "pass2_mean": _error(pass2_mean, _load(fsl / "DWI_corr_mean.nii.gz")),
        "mean_matrix_max_abs": float(
            np.max(
                np.abs(
                    mean_registration.matrix
                    - np.loadtxt(fsl / "meanDWI2nodif.mat")
                )
            )
        ),
        "mean_matrix_candidate": mean_registration.matrix.tolist(),
        "mean_matrix_reference": np.loadtxt(fsl / "meanDWI2nodif.mat").tolist(),
        "search_optimized_matrix": search_optimized[0].matrix.tolist(),
        "search_preoptimized_matrix": search_preoptimized[0].matrix.tolist(),
        "search_cost_identity": float(search_evaluator(np.eye(4))),
        "search_cost_candidate": float(
            search_evaluator(search_optimized[0].matrix)
        ),
        "search_cost_preoptimized": float(
            search_evaluator(search_preoptimized[0].matrix)
        ),
        "search_cost_fsl": float(search_evaluator(fsl_search_matrix)),
        "direct_b0_matrix_max_abs": float(
            max(
                np.max(
                    np.abs(
                        direct_b0[int(index)]
                        - np.loadtxt(fsl / "DWI_b0.mat" / f"MAT_{index:04d}")
                    )
                )
                for index in b0_indices
            )
        ),
    }
    if (args.candidate_final is None) != (args.fsl_preprocessing is None):
        raise ValueError(
            "--candidate-final and --fsl-preprocessing must be provided together"
        )
    if args.candidate_final is not None:
        report["final"] = _final_metrics(
            args.candidate_final.resolve(),
            args.fsl_preprocessing.resolve(),
            fsl,
        )
    exit_code = 0
    if args.thresholds is not None:
        contract = json.loads(args.thresholds.read_text(encoding="utf-8"))
        thresholds = contract.get("release_gate", {}).get("thresholds")
        if not isinstance(thresholds, list) or not thresholds:
            raise ValueError(
                "The threshold file must contain a non-empty release_gate.thresholds list"
            )
        failures = _evaluate_thresholds(report, thresholds)
        report["release_gate"] = {
            "status": "passed" if not failures else "failed",
            "threshold_count": len(thresholds),
            "failures": failures,
        }
        exit_code = 0 if not failures else 1
    report_path = output / "legacy_stage_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
