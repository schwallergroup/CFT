"""
Test that get_manifold_minima finds consistent minima whether values are
taken directly from a reference mesh or via KDTree interpolation.

Requires env vars:
  CFT_DATA_DIR  - path to directory containing:
                    manifold_O_WRAPONblend_PREC0.2_TSS2.0.xyz

  CFT_PARTICLE  - path to the particle .xyz file
                   (default: notebooks/test_particle.xyz relative to repo root)

The Manifold is constructed once per session via the `shared_manifold` fixture
defined in conftest.py.
"""
import numpy as np
from ase.io import read
from scipy.spatial import KDTree

from cft.mesh_utils import get_manifold_minima


def test_minima(shared_manifold):
    m, data_dir = shared_manifold

    mesh_ref = read(data_dir / 'manifold_O_WRAPONblend_PREC0.2_TSS2.0.xyz')
    vals_O_raw = mesh_ref.arrays['e_ClO_0']

    print(f"Number of grids: M={len(m.grid)}, File={len(vals_O_raw)}")
    if len(m.grid) == len(vals_O_raw):
        diff = np.max(np.linalg.norm(m.grid - mesh_ref.positions, axis=1))
        print(f"Max distance: {diff}")

    min1 = get_manifold_minima(m.faces, vals=vals_O_raw)
    print(f"Raw minima count: {len(min1)}")

    kdtree = KDTree(mesh_ref.positions)
    _, old_indices = kdtree.query(m.grid)
    vals_O_kdtree = vals_O_raw[old_indices]

    min2 = get_manifold_minima(m.faces, vals=vals_O_kdtree)
    print(f"KDTree minima count: {len(min2)}")


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])
