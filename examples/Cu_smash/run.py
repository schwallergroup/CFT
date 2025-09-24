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
    # print(f'{len(atoms) = }')

    # m = Manifold(
    #     atoms,
    #     precision = precision,
    #     mode = 'particle',
    #     touch_sphere_size = 3.5,
    #     calc = copy.deepcopy(clean_calc)
    #     )

    # print(f'{len(m.grid) = }')

    # m.make_fragment_population(
    #     population_size = population_size,
    #     fragment = f,
    #     coverage = coverage
    #     )

    # m.evaluate_surf_population()

    # write('surf_pop.xyz', m.surf_population)

    surf_population = read('surf_pop.xyz', index=':')

    for i in range(10):

        selected_atoms = surf_population[i].copy()
        c = FixAtoms(indices=[atom.index for atom in selected_atoms if atom.symbol == 'Cu'])
        selected_atoms.set_constraint(c) 
        selected_atoms.calc = copy.deepcopy(clean_calc)
        traj = Trajectory(f'relaxation_top{i}.traj', 'w', atoms)
        optimizer = BFGS(selected_atoms, trajectory=traj)
        optimizer.run(fmax = fmax)

        write(f'relaxed_top{i}.xyz', selected_atoms)

        mp = Manifold(
            atoms = selected_atoms[selected_atoms.arrays['fragments'] < 1],
            precision = precision,
            mode = 'particle',
            touch_sphere_size = 2.5,
            calc = copy.deepcopy(clean_calc)
            )

        mp.atoms = selected_atoms
        mp.atoms.calc  = copy.deepcopy(clean_calc)

        probes = [
            Atoms(['H'], [[0,0,0]]),
            Atoms(['C'], [[0,0,0]]),
            Atoms(['O'], [[0,0,0]]), 
            Fragment('ClC#[O+]', to_initialize=1, prune_rms_thresh=.0001),
            Fragment('ClP', to_initialize=1, prune_rms_thresh=.0001),
            ]
        mp.run_probe_scan(probes)

        mp.write_grid(f'grd_top{i}.xyz', inclde_atoms=False)

#####################################################################################

precision = 1.5
population_size = 500
coverage = .6
fmax = 0.1

clean_calc = mace_mp(model=
            #    '/mnt/c/Users/ef/Desktop/tmp/mace-mh-nl-pbe.model',
               '/home/fako/data/mace_models/mace-mh-nl-pbe.model',
               device='cuda',
               head='matpes_r2scan')

atoms = read(
        './run_20250826-072304_2bb6cead_sphere_500-run_20250826-072304_2bb6cead_md.xyz',
        index=20)
    
f = Fragment('Cl[PH+](CC(C)C)(CC(C)C)', to_initialize=100, prune_rms_thresh=.0001)


if __name__ == '__main__':
    main()

# plt.plot([a.info['static_energy'] for a in m.surf_population])
# plt.show()
