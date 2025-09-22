from autoadsorbate import Surface, Fragment
from ase.io import read, write
from cft import Manifold
from ase import Atoms

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
    precision = 5.,
    mode = 'particle',
    calc = calc
    )

print(f'{m.faces[0] = }')
print(f'{m.normals[0] = }')
print(f'{m.grid_atoms = }')

f = Fragment('ClO')

print(f'{f.smile = }')

probes = [Atoms('H', [0,0,0])]
m.run_probe_scan(probes)

# m.view_grid()

