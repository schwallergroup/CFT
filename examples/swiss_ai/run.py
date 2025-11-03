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
import pandas as pd
import matplotlib.pyplot as plt
import uuid
from ase.db import connect
                
from cft.core import db_to_traj

def atoms_from_string(raw_text: str) -> Atoms:
    """
    Creates an ASE Atoms object from a single-line, semicolon-delimited string.
    """
    symbols = []
    positions = []

    for part in raw_text.split(';'):
        words = part.strip().split()
        
        if len(words) == 4:
            try:
                pos = [float(c) for c in words[1:]]
                symbols.append(words[0])
                positions.append(pos)
            except ValueError:
                pass
                
    return Atoms(symbols=symbols, positions=positions)


def get_cluster_traj(xdf):
    from ase.db import connect
    from ase.io import read
    import pandas as pd
    
    meta_columns = [c for c in xdf.columns if c != 'structure_poscar_format']
    
    cluster_trj = []

    for idx, row in xdf.iterrows():
        xyz_lines = row['structure_xyz']
        atoms = atoms_from_string(xyz_lines)
        metadata = {col: row[col] for col in meta_columns}

        cluster_trj.append(atoms)
        
    return cluster_trj
        

def main():
    print(f'{len(cluster_trj) = }')

    for trj_i, atoms in enumerate(cluster_trj):

        atoms.calc = copy.deepcopy(clean_calc)
        atoms.info['e_naked_cluster'] = atoms.get_potential_energy()

        m = Manifold(
            atoms,
            precision = precision,
            mode = 'particle',
            touch_sphere_size = touch_sphere_size,
            calc = copy.deepcopy(clean_calc),
            wrap_on='sites'
            )

        print(f'{len(m.grid) = }')

        for f in fragments:
            m.make_fragment_population(
                population_size = population_size,
                fragment = f,
                coverage = coverage
                )

            m.evaluate_surf_population()

            for a in m.surf_population:
                a.info['static_reward'] = 0.
                if a.info['n_fragments'] > 0:
                    a.info['static_reward'] = (a.info['static_energy'] - a.info['e_naked_cluster']) / a.info['n_fragments']
                a.info['stage'] = 'raw'

            surf_population_sorted = sorted(m.surf_population, key=lambda at: at.info.get('static_reward', 0))
 
            relaxed_trj =[]
            grids = []
            for i in range(0, int(to_relax)):

                selected_atoms = surf_population_sorted[i].copy()
                selected_atoms.info['population_i'] = i
                selected_atoms.calc = copy.deepcopy(clean_calc)
                optimizer = BFGS(selected_atoms) #, trajectory=traj)
                optimizer.run(fmax = fmax)

                relaxed_id = uuid.uuid4().hex
                
                selected_atoms.info['stage'] = 'relaxed'
                selected_atoms.info['relaxed_id'] = relaxed_id
                selected_atoms.calc = None
                relaxed_trj.append(selected_atoms)

                naked_relaxed_atoms = selected_atoms[[atom.index for atom in selected_atoms if atom.symbol in metals]]
                print(f'{naked_relaxed_atoms = }')

                m_grid = Manifold(
                    naked_relaxed_atoms,
                    precision = precision,
                    mode = 'particle',
                    touch_sphere_size = touch_sphere_size,
                    calc = copy.deepcopy(clean_calc),
                    wrap_on='sites'
                    )
                m_grid.run_probe_scan(probes=probes
                )
                grid_atoms = m_grid.grid_atoms
                grid_atoms.info['stage'] = 'grid'
                grid_atoms.info['relaxed_id'] = relaxed_id
                pop_keys = []
                for k in grid_atoms.arrays.keys():
                    if 'e_' in k and k not in ['e_Cl[P]']:
                        pop_keys.append(k)
                for k in pop_keys:
                    _ = grid_atoms.arrays.pop(k)
                grids.append(grid_atoms)

                m.write_to_db(surf_population_sorted+relaxed_trj+grids, db_path=db_file)
            exit()
            

#####################################################################################

#hyper params
# to_relax x len(df) x len(dfl) x success_rate ~ total number of configs
# 10 x 679 x 20 x .5 ~ 70 k
precision = 1.5             # default = 1.5
touch_sphere_size = 3.5     # default = 3.5
population_size = 100       # default = 1000
to_relax = 3                # default = 10
coverage = .99              # default = .99
fmax = 0.50                 # default = 0.02
prune_rms_thresh=.01        # default = 0.01
f_conformers = 200          # default = 200
db_file = 'db_out.db'      
probes = [Fragment('Cl[P]', to_initialize=1)]

# calculator
clean_calc = mace_mp(model=
               '/mnt/c/Users/ef/Desktop/tmp/mace-mh-nl-pbe.model',
            #    '/home/fako/data/mace_models/mace-mh-nl-pbe.model',
            #    '/home/fako/projects/models/mace-omat-0-medium.model',
               device='cpu',         # default = 'cuda'
               head='omol'  # default = 1.5
               )

# clusters
metals = ['Au','Ag','Co','Pt','Pd','Rh','Cu','Ir','Ru'] # preselected clusters
df = pd.read_json("data/data.json", lines=False); df = df.T
df = df[(df['energy_relative'] < 0.1) & df.element_symbol.isin(metals)].copy() # & (df['n_atoms'].astype(int) > 50)

cluster_trj = get_cluster_traj(df)
print(f'{len(cluster_trj) = }')

# fragemtns
dfl = pd.read_csv('data/phosphine_ligands_enriched.csv', delimiter=','); dfl = dfl.dropna()
fragments = [Fragment(smi, to_initialize=f_conformers, prune_rms_thresh=prune_rms_thresh) for smi in dfl.surrogateSMILES.values]
print(f'{len(dfl) = }')

if __name__ == '__main__':
    main()