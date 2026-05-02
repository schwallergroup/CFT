"""
Check which manifold vertices are closest to Pt atoms and whether PMD minima
fall on those Pt-associated vertices.

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
    from cft.mesh_utils import get_manifold_minima
except ImportError as e:
    pytest.skip(f"cft not importable: {e}", allow_module_level=True)


def test_pt_sites():
    import pytest
    data_dir = os.environ.get('CFT_DATA_DIR')
    if not data_dir:
        pytest.skip("CFT_DATA_DIR not set — skipping (requires pre-computed field files)")
    data_dir = Path(data_dir)

    particle_path = os.environ.get('CFT_PARTICLE', 'notebooks/test_particle.xyz')
    particle = read(particle_path)

    pt_indices = [atom.index for atom in particle if atom.symbol == 'Pt']
    print(f"Pt atom indices in particle: {pt_indices}")

    m = Manifold(particle.copy(), mode='particle', precision=0.2, touch_sphere_size=2.0, wrap_on='blend')

    dists = np.linalg.norm(m.grid[:, np.newaxis, :] - particle.positions[np.newaxis, :, :], axis=2)
    closest_atoms = np.argmin(dists, axis=1)

    pt_vertices = np.where(np.isin(closest_atoms, pt_indices))[0]
    print(f"Number of manifold vertices closest to a Pt atom: {len(pt_vertices)} (out of {len(m.grid)})")

    aniso_trj = read(data_dir / 'anisotropic_reactivity_map_CO.xyz', index=':')
    all_PMD_raw = np.array([atoms.arrays['PMD-CF_aniso'] for atoms in aniso_trj])

    pmd_raw_0 = all_PMD_raw[0].copy()
    pmd_norm = pmd_raw_0.copy()
    pmd_norm -= pmd_norm.min()
    pmd_norm[pmd_norm > 1] = 1

    pmd_mins_idx_0 = np.array(get_manifold_minima(m.faces, vals=pmd_norm))

    pt_pmd_minima = np.intersect1d(pmd_mins_idx_0, pt_vertices)
    print(f"Number of PMD minima (at angle 0) closest to Pt: {len(pt_pmd_minima)}")

    if len(pt_pmd_minima) > 0:
        print("Pt has valid TS basins.")
    else:
        print("Pt has NO valid TS basins.")
        pt_energies = pmd_norm[pt_vertices]
        print(f"Min normalized PMD energy on a Pt vertex: {pt_energies.min():.3f}")


if __name__ == '__main__':
    test_pt_sites()
