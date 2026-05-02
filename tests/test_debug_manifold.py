"""
Verify that a freshly built Manifold grid matches a pre-computed anisotropic
CO field map in both length and vertex positions.

Requires env vars:
  CFT_DATA_DIR  - path to directory containing:
                    anisotropic_reactivity_map_CO.xyz

  CFT_PARTICLE  - path to the particle .xyz file
                   (default: notebooks/test_particle.xyz relative to repo root)
"""
import os
import numpy as np
from pathlib import Path
from ase.io import read
import pytest
try:
    from cft import Manifold
except ImportError as e:
    pytest.skip(f"cft not importable: {e}", allow_module_level=True)


def test_debug_manifold():
    import pytest
    data_dir = os.environ.get('CFT_DATA_DIR')
    if not data_dir:
        pytest.skip("CFT_DATA_DIR not set — skipping (requires pre-computed field files)")
    data_dir = Path(data_dir)

    particle_path = os.environ.get('CFT_PARTICLE', 'notebooks/test_particle.xyz')
    particle = read(particle_path)

    m = Manifold(
        particle.copy(),
        mode='particle',
        precision=0.2,
        touch_sphere_size=2.0,
        wrap_on='blend',
    )

    trj = read(data_dir / 'anisotropic_reactivity_map_CO.xyz', index=':')
    grid_xyz = trj[0]

    print(f"Len Manifold Grid: {len(m.grid)}")
    print(f"Len XYZ Grid: {len(grid_xyz)}")

    if len(m.grid) == len(grid_xyz):
        diff = np.max(np.linalg.norm(m.grid - grid_xyz.positions, axis=1))
        print(f"Max distance between corresponding vertices: {diff:.6f}")
        assert diff < 1e-3, f"Grid vertices deviate by {diff:.6f} — manifold is not reproducible"
    else:
        assert False, f"Grid length mismatch: Manifold={len(m.grid)}, File={len(grid_xyz)}"


if __name__ == '__main__':
    test_debug_manifold()
