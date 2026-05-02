"""Screen phosphine-type ligands on a Rh28 nanoparticle using covalent fields.

Usage
-----
    MODEL_PATH=/path/to/mace.model python run.py

Environment variables
---------------------
MODEL_PATH : str
    Path to a MACE model file (required).

Input files (expected in the same directory)
--------------------------------------------
rh_28.xyz : ASE-readable XYZ file of the Rh28 nanoparticle.
"""

from autoadsorbate import Surface, Fragment
from ase.io import read, write
from cft import Manifold
from cft.mesh_utils import estimate_radius_decay
from ase import Atoms
import numpy as np
import os
import copy
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
from ase.io import Trajectory
from ase.optimize import BFGS


def main():
    """Run probe scan and ligand relaxation on every frame in the Rh28 trajectory."""
    # Heavy initialisation inside main() so that MODEL_PATH and the XYZ file
    # are only required when actually running the script, not when importing.
    clean_calc = mace_mp(
        model=os.environ.get('MODEL_PATH', ''),  # set MODEL_PATH env var to your MACE model file
        device='cuda',
    )

    traj = [read('./rh_28.xyz', index=0)]

    print(f'{len(traj) = }')

    for trj_i, atoms in enumerate(traj):
        m = Manifold(
            atoms,
            precision=precision,
            mode='particle',
            touch_sphere_size=3.5,
            calc=copy.deepcopy(clean_calc),
        )

        print(f'{len(m.grid) = }')

        m.make_fragment_population(
            population_size=population_size,
            fragment=fragment,
            coverage=coverage,
        )

        m.evaluate_surf_population()

        write(f'surf_pop_{coverage}_{trj_i}.xyz', m.surf_population)

        surf_population = read(f'surf_pop_{coverage}_{trj_i}.xyz', index=':')

        for i in range(0, 1):
            selected_atoms = surf_population[i].copy()
            c = FixAtoms(indices=[atom.index for atom in selected_atoms if atom.symbol == 'Cu'])
            selected_atoms.set_constraint(c)
            selected_atoms.calc = copy.deepcopy(clean_calc)
            optimizer = BFGS(selected_atoms)
            optimizer.run(fmax=fmax)

            write(f'relaxed_{coverage}_top_{trj_i}_{i}_{fragment.smile}.xyz', selected_atoms)


# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

precision = 1.5
population_size = 500
coverage = 0.6
fmax = 0.1

fragment = Fragment('Cl[PH+](CC(C)C)(CC(C)C)', to_initialize=100, prune_rms_thresh=0.0001)


if __name__ == '__main__':
    main()

