from autoadsorbate import Surface
from ase.io import read, write
from ase.visualize import view
import numpy as np
from ase import Atoms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator

from cft import Manifold
from autoadsorbate.Particle import get_cube_surface_pts, grid_round_cube
from cft.mesh_utils import compute_outward_vertex_normals_quads

from ase.io import read, write
from ase.visualize import view
from ase.optimize import BFGS
from ase.io import Trajectory

from autoadsorbate import Fragment
from autoadsorbate.Surf import attach_fragment
from ase.constraints import FixAtoms, FixedPlane, FixCartesian
import copy
from cft.mesh_utils import values_to_colors

from ase.neb import NEB
import copy
from ase.neb import NEBTools
import matplotlib.pyplot as plt

def fit_reaction_plane(images, fixed_indices=None):
    """
    Fit a plane to moving atoms across NEB images.

    Parameters
    ----------
    images : list of ase.Atoms
        List of NEB images.
    fixed_indices : list of int, optional
        Atom indices to ignore (fixed atoms). Default: None.

    Returns
    -------
    plane_func : function
        Function z = f(x, y) giving plane height for given (x, y).
    normal : np.array
        Normal vector of the plane.
    point : np.array
        Point on the plane (centroid of moving atoms).
    """

    # collect positions of moving atoms
    all_pos = []
    for img in images:
        pos = img.get_positions()
        if fixed_indices is not None:
            mask = np.ones(len(pos), dtype=bool)
            mask[fixed_indices] = False
            pos = pos[mask]
        all_pos.append(pos)
    
    all_pos = np.vstack(all_pos)  # shape (N_total, 3)

    # compute centroid
    centroid = np.mean(all_pos, axis=0)

    # subtract centroid
    X = all_pos - centroid

    # SVD to find normal (smallest singular value)
    _, _, vh = np.linalg.svd(X)
    normal = vh[-1, :]  # normal vector of best-fit plane

    # plane equation: (r - centroid) . normal = 0
    # return a function z = f(x, y)
    def plane_func(x, y):
        a, b, c = normal
        if abs(c) < 1e-8:
            raise ValueError("Plane normal nearly vertical, cannot express z as function of x,y")
        return (-a*(x - centroid[0]) - b*(y - centroid[1])) / c + centroid[2]

    return plane_func, normal, centroid


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)

    # choose a vector not parallel to the normal
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(normal, e1)
    return e1, e2

def project_to_plane(points, centroid, e1, e2):
    rel = points - centroid
    u = rel @ e1
    v = rel @ e2
    return u, v


def generate_plane_mesh(images, resolution=0.1, gutter=2.0):
    """
    Generate a square quad mesh lying in the best-fit reaction plane.

    Parameters
    ----------
    images : list of ase.Atoms
        NEB images
    resolution : float
        Grid spacing in plane coordinates
    gutter : float
        Padding added around projected atom bounds

    Returns
    -------
    verts : (N, 3) ndarray
        Vertex positions
    edges : list of (i, j)
        Edge index pairs
    faces : list of [i0, i1, i2, i3]
        Quad faces
    normals : (N, 3) ndarray
        Per-vertex normals
    """

    # ---- collect fixed atom indices (FixAtoms only) ----
    fixed_indices = set()
    for img in images:
        for c in img.constraints or []:
            if isinstance(c, FixAtoms):
                fixed_indices.update(c.index)

    fixed_indices = list(fixed_indices)

    # ---- fit reaction plane ----
    _, normal, centroid = fit_reaction_plane(
        images, fixed_indices=fixed_indices
    )
    normal = normal / np.linalg.norm(normal)

    # ---- collect moving atom positions ----
    moving_pos = []
    for img in images:
        pos = img.get_positions()
        if fixed_indices:
            mask = np.ones(len(pos), dtype=bool)
            mask[fixed_indices] = False
            pos = pos[mask]
        moving_pos.append(pos)
    moving_pos = np.vstack(moving_pos)

    # ---- build plane basis ----
    e1, e2 = plane_basis(normal)

    # ---- project atoms into plane coordinates ----
    rel = moving_pos - centroid
    u = rel @ e1
    v = rel @ e2

    umin, umax = u.min() - gutter, u.max() + gutter
    vmin, vmax = v.min() - gutter, v.max() + gutter

    # ---- generate grid ----
    us = np.arange(umin, umax + resolution, resolution)
    vs = np.arange(vmin, vmax + resolution, resolution)

    Nu = len(us)
    Nv = len(vs)

    # ---- generate vertices ----
    verts = np.empty((Nu * Nv, 3))
    for i, uu in enumerate(us):
        for j, vv in enumerate(vs):
            idx = i * Nv + j
            verts[idx] = centroid + uu * e1 + vv * e2

    # ---- generate quad faces & edges ----
    faces = []
    edges = set()

    def vid(i, j):
        return i * Nv + j

    for i in range(Nu - 1):
        for j in range(Nv - 1):
            v0 = vid(i,     j)
            v1 = vid(i,     j + 1)
            v2 = vid(i + 1, j)
            v3 = vid(i + 1, j + 1)

            faces.append([v0, v1, v3, v2])

            edges.update({
                (v0, v1),
                (v1, v3),
                (v3, v2),
                (v2, v0),
            })

    edges = list(edges)

    # ---- per-vertex normals ----
    normals = np.tile(normal, (len(verts), 1))

    return verts, edges, faces, normals

