"""
Test that attach_fragment correctly places CO and PMD probe fragments at the origin.

Requires env vars:
  CFT_DATA_DIR  - path to directory containing images_neb.xyz (for PMD test)
                  If unset, the PMD fragment test is skipped.
"""
import os
import numpy as np
from ase import Atoms
from autoadsorbate import Fragment
from autoadsorbate.Surf import attach_fragment

p = np.array([0., 0., 0.])
n = np.array([0., 0., 1.])


def test_co_fragment():
    print("--- CO Fragment ---")
    f = Fragment("ClC#[O+]", to_initialize=1).conformers[0]
    a = Atoms()
    attach_fragment(a, {"coordinates": p, "n_vector": n}, fragment=f, n_rotation=0, height=0)
    print(a.get_chemical_symbols())
    print(a.positions)


def test_pmd_fragment():
    data_dir = os.environ.get('CFT_DATA_DIR')
    if not data_dir:
        print("CFT_DATA_DIR not set — skipping PMD fragment test.")
        return

    from pathlib import Path
    from ase.io import read
    from cft.neb_utils import get_neb_probe

    print("\n--- PMD Fragment ---")
    pmd_image_full = read(Path(data_dir) / 'images_neb.xyz', index=':')[5]
    f_pmd = get_neb_probe(pmd_image_full, keep=['C', 'O']).get_conformer(0)
    b = Atoms()
    attach_fragment(b, {"coordinates": p, "n_vector": n}, fragment=f_pmd, n_rotation=0, height=0)
    print(b.get_chemical_symbols())
    print(b.positions)


if __name__ == '__main__':
    test_co_fragment()
    test_pmd_fragment()
