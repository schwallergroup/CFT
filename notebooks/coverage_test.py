from autoadsorbate import Surface, Fragment
from ase.io import read, write
from cft import Manifold
from cft.mesh_utils import estimate_radius_decay
from ase import Atoms
import numpy as np
import os
from ase.visualize import view
import matplotlib.pyplot as plt

# from mace.calculators import mace_mp
# calc = mace_mp(model=os.environ.get('MODEL_PATH', ''),
#                device='cpu',
#                head='matpes_r2scan')

atoms = read(
    os.environ.get('CFT_MD_XYZ', '../examples/Cu_smash/naked_particle.xyz'),
    index=10)
print(f'{len(atoms) = }')

m = Manifold(
    atoms,
    precision = 1.5,
    mode = 'particle',
    touch_sphere_size = 3.5,
    # calc = calc
    )

print(f'{m.grid_atoms = }')
write('xx.xyz', m.grid_atoms+m.atoms)
# m.write_grid('xx.xyz')

# f = Fragment('Cl[PH+](CC(C)C)(CC(C)C)', to_initialize=1, prune_rms_thresh=.0001)

# m.make_fragment_population(
#     population_size = 30,
#     fragment = f,
#     coverage = .8
#     )

# m.evaluate_surf_population()

# write('surf_pop.xyz', m.surf_population)
# plt.plot([a.info['static_energy'] for a in m.surf_population])
# plt.show()
