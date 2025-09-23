from autoadsorbate import Surface, Fragment
from ase.io import read, write
from cft import Manifold
from ase import Atoms
from ase.visualize import view

from mace.calculators import mace_mp
calc = mace_mp(model='/mnt/c/Users/ef/Desktop/tmp/mace-mh-nl-pbe.model', device='cpu', head='matpes_r2scan')

atoms = read('./test_particle.xyz')
# atoms = read('./test_slab.xyz')
print(f'{atoms = }')

s = Surface(atoms, mode='particle', precision=3.)
print(f'{s.grid[0] = }')

# s = Surface(slab)
# print(f'{s.grid[0] = }')

m = Manifold(
    atoms,
    precision = 15.,
    mode = 'particle',
    calc = calc
    )

print(f'{m.faces[0] = }')
print(f'{m.normals[0] = }')
print(f'{m.grid_atoms = }')

f = Fragment('ClC(=O)[O-]', to_initialize=2, prune_rms_thresh=.0001)
print(f'{f.smile = }')

print(f'{f.conformers[0].positions.shape = }')

# view(f.conformers)
# view([f.get_conformer(i) for i, _ in enumerate(f.conformers)])


probes = [Atoms(['H'], [[0,0,0]])] #f]#
m.run_probe_scan(probes)

for k, v in m.grid_atoms.arrays.items():
    print(k, v[0])
# m.view_grid()

m.write_grid('grd.xyz', inclde_atoms=False)

