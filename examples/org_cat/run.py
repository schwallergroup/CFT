"""Compute covalent-field grids for a set of organocatalyst conformers.

Usage
-----
    MODEL_PATH=/path/to/mace.model python run.py

Environment variables
---------------------
MODEL_PATH : str
    Path to a MACE model file (required).
"""

import os
import copy

from ase.io import read, write
import numpy as np

from cft import Manifold
from autoadsorbate import Fragment
from mace.calculators import mace_mp


def main():
    """Build Manifold grids for every conformer of each catalyst SMILES."""
    # Heavy initialisation is done here so that importing the module does not
    # require MODEL_PATH to be set or trigger a GPU/CPU model load.
    clean_calc = mace_mp(
        model=os.environ.get('MODEL_PATH', ''),  # set MODEL_PATH env var to your MACE model file
        device='cpu',
    )

    print(f'{cats = }')

    cats_traj = []
    for c in cats:
        f = Fragment(c, to_initialize=to_initialize, prune_rms_thresh=prune_rms_thresh)
        conformers = [f.get_conformer(i) for i, _ in enumerate(f.conformers)]

        for i, c in enumerate(conformers):
            c.info['conf_i'] = i
        cats_traj += conformers

    print(f'{len(cats_traj) = }')
    write('cats_traj.xyz', cats_traj)

    for i, a in enumerate(cats_traj):
        calc = copy.deepcopy(clean_calc)
        m = Manifold(a, mode='particle', precision=precision, touch_sphere_size=touch_sphere_size, calc=calc)
        m.run_probe_scan(probes=probes)

        # Subtract reference energy (probe infinitely far away) from each channel.
        for k in m.grid_atoms.arrays.keys():
            if k in m.ref_energy_dict.keys():
                print(k)
                m.grid_atoms.arrays[k] -= m.ref_energy_dict[k]

        m.write_grid("tmp.xyz")
        write('out_trj.xyz', read("tmp.xyz"), append=True)


# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

cats = [
    'C1CN2CCN1CC2',
    'CN1CCCC1',
    'c1ccccc1CN2CCC2',
    'c1ccccc1COC2CCN(C)C2',
    'c1ccccc1COCCN2CCCC2'
]

# mesh settings
precision=2.
touch_sphere_size=2.1

# probe settings
probes = [Fragment('ClC', to_initialize=1)] # this is the stand in for the ts "dummy"

# cat settings
prune_rms_thresh =  10. # same as in rdkit, small numb = more conformers, recommended ~ 0.5
to_initialize = 1 # how many conformers to make before pruning


if __name__ == '__main__':
    main()
