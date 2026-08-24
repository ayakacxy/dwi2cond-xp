"""Cross-platform tensor preparation for SimNIBS anisotropic simulations."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "fit_tensor_wls": (".tensor_fit", "fit_tensor_wls"),
    "form_design_matrix": (".tensor_fit", "form_design_matrix"),
    "load_gradients": (".gradients", "load_gradients"),
    "select_dti_volumes": (".gradients", "select_dti_volumes"),
}

# ``reload`` retains the old module dictionary, so discard previously cached
# values before restoring lazy lookups.
globals().pop("__version__", None)
for _lazy_name in _LAZY_EXPORTS:
    globals().pop(_lazy_name, None)


def __getattr__(name: str) -> object:
    """Resolve public APIs only when a caller actually uses them."""

    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            value = version("dwi2cond-xp")
        except PackageNotFoundError:
            value = "0+unknown"
        globals()[name] = value
        return value
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily resolved names in interactive module discovery."""

    return sorted(set(globals()) | set(__all__))

__all__ = [
    "__version__",
    "fit_tensor_wls",
    "form_design_matrix",
    "load_gradients",
    "select_dti_volumes",
]
