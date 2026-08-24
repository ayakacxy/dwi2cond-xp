"""Reference harness and preprocessing building blocks for raw DWI support.

The public names are resolved lazily so importing one preprocessing submodule
does not initialize every FLIRT, TOPUP, and EDDY kernel in the package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


def _exports(module: str, *names: str) -> dict[str, str]:
    """Map public names to one relative implementation module."""

    return {name: module for name in names}


_LAZY_EXPORTS: Final[dict[str, str]] = {
    **_exports(
        ".image_ops",
        "apply_positive_mask",
        "binarize_positive",
        "edge_strength",
        "extract_roi",
        "gaussian_smooth",
        "image_dimensions",
        "lower_threshold",
        "masked_percentile",
        "median_filter_box",
        "merge_time",
        "multiply",
        "read_dwi_z_block",
        "select_b0_indices",
        "subtract",
        "time_mean",
        "upper_threshold",
        "write_float32_copy",
        "write_unaligned_b0_mean",
    ),
    **_exports(
        ".reference",
        "ReferenceArtifact",
        "ReferenceRunError",
        "audit_public_manifest",
        "run_reference_command",
        "summarize_fixture_inputs",
    ),
    **_exports(
        ".orientation",
        "copy_nifti_geometry",
        "fsl_canonical_orientation",
        "reorient_bvecs_voxel",
        "reorient_spatial_array",
        "reorient_tensor6_voxel",
        "voxel_basis_transform",
        "write_fsl_reoriented",
    ),
    **_exports(
        ".rigid",
        "RigidRegistrationResult",
        "estimate_rigid_transform",
        "normalized_correlation_cost",
        "resample_rigid",
        "rigid_world_matrix",
        "write_aligned_b0_mean",
    ),
    **_exports(".tensor_ops", "decompose_tensor6"),
    **_exports(
        ".brain_mask",
        "BetResult",
        "bet_brain_mask",
        "robust_intensity_limits",
        "write_bet_brain_mask",
    ),
    **_exports(
        ".resampling",
        "output_to_input_voxel_matrix",
        "resample_image",
        "resample_mask",
    ),
    **_exports(
        ".transforms",
        "affine_matrix",
        "compose_transforms",
        "decompose_affine",
        "fsl_matrix_to_world",
        "fsl_voxel_to_scaled_mm",
        "invert_transform",
        "rigid_matrix",
        "world_matrix_to_fsl",
    ),
    **_exports(
        ".flirt_cost",
        "FlirtWeightedCorrelationRatio",
        "FlirtWeightedMutualInformation",
        "flirt_intensity_cog",
    ),
    **_exports(
        ".flirt_optimizer",
        "FlirtOptimizationResult",
        "flirt_brent_optimize",
        "optimize_flirt_stage",
    ),
    **_exports(
        ".flirt_pyramid",
        "FlirtPyramidLevel",
        "build_flirt_pyramid",
        "flirt_blur",
        "isotropic_resample",
        "subsample_by_two",
    ),
    **_exports(
        ".flirt_registration",
        "FlirtRegistrationResult",
        "register_flirt_affine",
        "register_flirt_nosearch_mutual_information",
    ),
    **_exports(
        ".t1_registration",
        "prepare_charm_t1_inputs",
        "run_t1_registration_nifti",
    ),
    **_exports(
        ".fieldmap",
        "FieldmapResult",
        "displacement_from_voxel_shift",
        "extrapolate_field_holes",
        "fill_head_mask",
        "forward_warp_magnitude",
        "phase_encoding_axis_sign",
        "prepare_radians_per_second",
        "regularize_voxel_shift",
        "run_fieldmap",
        "run_fieldmap_nifti",
        "voxel_shift_from_field",
    ),
    **_exports(".topup", "TopupRunResult", "run_simnibs46_topup", "run_topup_nifti"),
    **_exports(
        ".eddy",
        "EddyB0RegistrationResult",
        "EddyOutlierResult",
        "EddyHyperparameterResult",
        "EddyDerivativeResult",
        "EddyDwiRegistrationResult",
        "EddyGaussNewtonResult",
        "EddyIterationResult",
        "EddyRunResult",
        "EddySliceStatistics",
        "EddySphericalGP",
        "EddyTransformResult",
        "PreparedEddySusceptibilityField",
        "detect_eddy_slice_outliers",
        "apply_eddy_shell_pe_translation",
        "apply_eddy_shell_rigid_alignment",
        "eddy_slice_statistics",
        "eddy_parameter_derivatives",
        "eddy_gauss_newton_update",
        "estimate_spherical_gp_hyperparameters",
        "estimate_eddy_shell_pe_translation",
        "estimate_eddy_shell_rigid_alignment",
        "fit_spherical_gp_weights",
        "invert_eddy_displacement",
        "predict_spherical_gp",
        "prepare_eddy_susceptibility_field",
        "quadratic_eddy_field",
        "rotate_bvecs_eddy",
        "run_eddy_b0_iterations",
        "run_eddy_dwi_iterations",
        "run_eddy_nifti",
        "run_simnibs46_eddy",
        "select_fsl_gp_voxels",
        "spherical_gp_cv_cost",
        "spherical_gp_covariance",
        "transform_eddy_model_to_scan",
        "transform_eddy_scan_to_model",
    ),
    **_exports(
        ".nonlinear",
        "NonlinearTensorResult",
        "fsl_displacement_gradient",
        "fsl_warp_jacobians",
        "register_tensor_fnirt_nifti",
        "register_tensor_nonlinear_nifti",
        "resample_tensor_ppd_fsl",
    ),
    **_exports(
        ".fnirt",
        "FnirtLevel",
        "FnirtLevelImages",
        "FnirtLevelOptimization",
        "FnirtIntensityMapping",
        "FnirtCostEvaluation",
        "FnirtGradientEvaluation",
        "FnirtHessianEvaluation",
        "FnirtPreparedImages",
        "FnirtRunResult",
        "FnirtWarpExpansion",
        "FnirtWarpedMoving",
        "SIMNIBS46_FNIRT_LEVELS",
        "expand_fnirt_coefficients",
        "evaluate_fnirt_cost",
        "evaluate_fnirt_gradient",
        "evaluate_fnirt_hessian",
        "apply_fnirt_intensity_mapping",
        "fsl_fnirt_bias_knot_spacing",
        "fsl_fnirt_full_resolution_knot_spacing",
        "fsl_fnirt_level_affine",
        "fsl_fnirt_smooth",
        "fsl_fnirt_subsampled_shape",
        "fsl_spm_like_mean",
        "prepare_fnirt_images",
        "prepare_fnirt_level_images",
        "run_simnibs46_fnirt",
        "warp_fnirt_moving",
        "initialize_fnirt_intensity_mapping",
        "optimize_fnirt_level",
        "zoom_fnirt_spline_coefficients",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __dir__() -> list[str]:
    """Expose lazy public names to normal module introspection."""

    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def __getattr__(name: str) -> object:
    """Resolve and cache one public preprocessing symbol on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
