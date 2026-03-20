"""
CFT
~~~~~~~~~~

CFT - stands for Covalent Field Theory. This package provides functionality for working with 2D manifolds for interfaces.
"""

__version__ = "0.1.0"

from .core import Manifold
from .neb_utils import ForceFit, fit_raw, fit_images, plot_band, get_neb_probe, plot_barrier, get_PMD_structure
from .mesh_utils import (
    fit_line_and_distances,
    slice_atoms_near_point,
    furthest_projected_pairs,
    points_close_to_rotating_line,
    curvature_deformed_cube,
)
from .utils import parse_vec
from .plot_utils import add_linear_fits

__all__ = [
    "Manifold",
    "ForceFit",
    "fit_raw",
    "fit_images",
    "plot_band",
    "get_neb_probe",
    "plot_barrier",
    "get_PMD_structure",
    "fit_line_and_distances",
    "slice_atoms_near_point",
    "furthest_projected_pairs",
    "points_close_to_rotating_line",
    "curvature_deformed_cube",
    "parse_vec",
    "add_linear_fits",
]
