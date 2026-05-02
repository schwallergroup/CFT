"""
Test that translating a CO fragment's Cl anchor atom to the origin before attaching
it produces correct placement at a grid vertex.
"""
import numpy as np
from ase import Atoms
from autoadsorbate import Fragment
from autoadsorbate.Surf import attach_fragment

p = np.array([0., 0., 0.])
n = np.array([0., 0., 1.])


def test_co_translate():
    print("--- CO Fragment (Zeroed) ---")
    f = Fragment("ClC#[O+]", to_initialize=1).conformers[0]
    # Translate Cl anchor atom to origin
    idx_cl = [a.index for a in f if a.symbol == 'Cl'][0]
    pos_cl = f.positions[idx_cl]
    f.positions -= pos_cl

    a = Atoms()
    attach_fragment(a, {"coordinates": p, "n_vector": n}, fragment=f, n_rotation=0, height=0)
    print(a.get_chemical_symbols())
    print(a.positions)


if __name__ == '__main__':
    test_co_translate()
