"""
Verify that a freshly built Manifold grid matches a pre-computed anisotropic
CO field map in both length and vertex positions.

The positional tolerance is 0.05 Å — tight enough to catch structural
divergence but permissive of minor floating-point variation between the
algorithm version used to generate the Zenodo reference data and the
current code.

Requires env vars:
  CFT_DATA_DIR  - path to directory containing:
                    anisotropic_reactivity_map_CO.xyz

  CFT_PARTICLE  - path to the particle .xyz file
                   (default: notebooks/test_particle.xyz relative to repo root)

The Manifold is constructed once per session via the `shared_manifold` fixture
defined in conftest.py.
"""
import numpy as np
from ase.io import read


def test_debug_manifold(shared_manifold):
    m, data_dir = shared_manifold

    trj = read(data_dir / 'anisotropic_reactivity_map_CO.xyz', index=':')
    grid_xyz = trj[0]

    print(f"Len Manifold Grid: {len(m.grid)}")
    print(f"Len XYZ Grid: {len(grid_xyz)}")

    assert len(m.grid) == len(grid_xyz), (
        f"Grid length mismatch: Manifold={len(m.grid)}, File={len(grid_xyz)}"
    )

    diff = np.max(np.linalg.norm(m.grid - grid_xyz.positions, axis=1))
    print(f"Max distance between corresponding vertices: {diff:.6f} Å")
    assert diff < 0.05, (
        f"Grid vertices deviate by {diff:.6f} Å — manifold grid has diverged from reference"
    )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])
