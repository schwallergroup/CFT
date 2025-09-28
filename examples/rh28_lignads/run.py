from autoadsorbate import Surface, Fragment
from ase.io import read, write
from cft import Manifold
from cft.mesh_utils import estimate_radius_decay
from ase import Atoms
import numpy as np
from ase.visualize import view
import copy
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
from ase.io import Trajectory
import matplotlib.pyplot as plt
from ase.optimize import BFGS

def main():
    print(f'{len(traj) = }')

    for trj_i, atoms in enumerate(traj):
        m = Manifold(
            atoms,
            precision = precision,
            mode = 'particle',
            touch_sphere_size = 3.5,
            calc = copy.deepcopy(clean_calc)
            )

        print(f'{len(m.grid) = }')

        m.make_fragment_population(
            population_size = population_size,
            fragment = f,
            coverage = coverage
            )

        m.evaluate_surf_population()

        write(f'surf_pop_{coverage}_{trj_i}.xyz', m.surf_population)

        surf_population = read(
            f'/scratch/fako/cft_Cu_smash/surf_pop_{coverage}_{trj_i}.xyz', index=':'
            )

        for i in range(0, 1):

            selected_atoms = surf_population[i].copy()
            c = FixAtoms(indices=[atom.index for atom in selected_atoms if atom.symbol == 'Cu'])
            selected_atoms.set_constraint(c) 
            selected_atoms.calc = copy.deepcopy(clean_calc)
            # traj = Trajectory(f'relaxation_top{i}.traj', 'w', selected_atoms)
            optimizer = BFGS(selected_atoms) #, trajectory=traj)
            optimizer.run(fmax = fmax)

            write(f'relaxed_{coverage}_top_{trj_i}_{i}_{f.smile}.xyz', selected_atoms)

#####################################################################################

precision = 1.5
population_size = 500
coverage = .6
fmax = 0.1

clean_calc = mace_mp(model=
            #    '/mnt/c/Users/ef/Desktop/tmp/mace-mh-nl-pbe.model',
               #'/home/fako/data/mace_models/mace-mh-nl-pbe.model',
               '/home/fako/projects/models/mace-omat-0-medium.model',
               device='cuda',
               #head='matpes_r2scan'
               )

traj = [read(
       '/home/ef/Code/CFT/examples/rh28_lignads/rh_28.xyz',
       index=0)]

f = Fragment('Cl[PH+](CC(C)C)(CC(C)C)', to_initialize=100, prune_rms_thresh=.0001)

if __name__ == '__main__':
    main()
# plt.plot([a.info['static_energy'] for a in m.surf_population])
# plt.show()
