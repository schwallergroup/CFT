from autoadsorbate import Surface, Fragment
from ase.io import read, write
from cft import Manifold
from ase import Atoms
import numpy as np
from ase.visualize import view

from mace.calculators import mace_mp
calc = mace_mp(model=
               '/mnt/c/Users/ef/Desktop/tmp/mace-mh-nl-pbe.model',
               device='cpu',
               head='matpes_r2scan')

atoms = read('./test_particle.xyz')
atoms.positions -= np.array([10,10,10])
atoms = atoms[np.linalg.norm(atoms.positions, axis=1) < 4.]
atoms.positions += np.array([10,10,10])

# atoms = read('./test_slab.xyz')
print(f'{len(atoms) = }')

m = Manifold(
    atoms,
    precision = 10.,
    mode = 'particle',
    touch_sphere_size = 2.5
    # calc = calc
    )

# f = Fragment('ClC(=O)[O-]', to_initialize=2, prune_rms_thresh=.0001)
f = Fragment('ClC', to_initialize=1, prune_rms_thresh=.0001)

m.calc = calc
probes = [f]#Atoms(['H'], [[0,0,0]])
m.run_probe_scan(probes)

m.write_grid('grd.xyz', inclde_atoms=False)

