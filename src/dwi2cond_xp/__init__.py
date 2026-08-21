"""Cross-platform tensor preparation for SimNIBS anisotropic simulations."""

from importlib.metadata import PackageNotFoundError, version

from .gradients import load_gradients, select_dti_volumes
from .tensor_fit import fit_tensor_wls, form_design_matrix

try:
    __version__ = version("dwi2cond-xp")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "fit_tensor_wls",
    "form_design_matrix",
    "load_gradients",
    "select_dti_volumes",
]
