from autoadsorbate import Surface, Fragment
from ase.io import read, write
from cft import Manifold
from ase import Atoms
import numpy as np
from ase.visualize import view

atoms = read('./test_particle.xyz')
atoms.positions -= np.array([10,10,10])
atoms = atoms[np.linalg.norm(atoms.positions, axis=1) < 4.]
atoms.positions += np.array([10,10,10])

# atoms = read('./test_slab.xyz')
print(f'{len(atoms) = }')

# s = Surface(atoms, mode='particle', precision=3.)
# print(f'{s.grid[0] = }')

# s = Surface(slab)
# print(f'{s.grid[0] = }')

m = Manifold(
    atoms,
    precision = 10.,
    mode = 'particle',
    touch_sphere_size = 2.5
    # calc = calc
    )

# m.view_hedgehog()
# print(f'{m.faces[0] = }')
# print(f'{m.normals[0] = }')
# print(f'{m.grid_atoms = }')

# f = Fragment('ClC(=O)[O-]', to_initialize=2, prune_rms_thresh=.0001)
f = Fragment('ClC', to_initialize=1, prune_rms_thresh=.0001)
# print(f'{f.smile = }')
# print(f'{f.conformers[0].positions.shape = }')

# view(f.conformers)
# view([f.get_conformer(i) for i, _ in enumerate(f.conformers)])

from mace.calculators import mace_mp
calc = mace_mp(model='/mnt/c/Users/ef/Desktop/tmp/mace-mh-nl-pbe.model', device='cpu', head='matpes_r2scan')
m.calc = calc
probes = [f]#Atoms(['H'], [[0,0,0]])
m.run_probe_scan(probes)

# for k, v in m.grid_atoms.arrays.items():
#     print(k, v[0])
# # m.view_grid()

m.write_grid('grd.xyz', inclde_atoms=False)

