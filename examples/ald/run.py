from autoadsorbate import Surface
from ase.io import read, write
from ase.visualize import view
import numpy as np
from ase import Atoms
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cft import Manifold
from ase.io import read, write
from ase.visualize import view
from autoadsorbate import Fragment
from ase.constraints import FixAtoms
import copy
from ase.optimize import BFGS


from mace.calculators import mace_mp

def make_TiN_slab(bulk, p, c, scale):
    slab = bulk.copy()*scale
    slab.cell[2][2] += c
    slab.positions[:,2] += c/2
    slab.arrays['fragments'] = np.array([0 for _ in slab])

    #make some vacancies
    _s = Surface(slab)
    n_inds = [atom.index for atom in _s.atoms if atom.symbol =='N' and atom.index in _s.surf_inds]
    n_vac = random.sample(n_inds, math.ceil(p * len(n_inds)))
    slab = slab[[atom.index for atom in slab if atom.index not in n_vac]]

    al_z = slab.positions[np.where(slab.positions[:,2] == np.max(slab.positions[:,2]))[0][0]][2]
    al_pos = slab.cell[0]*0.5+slab.cell[1]*0.5 + [0,0,al_z]

    for x in range(3):
        for y in range(3):
            for z in range(1,3):
                print(x,y,z)
                slab += Atoms(['Al'], [al_pos+[x*2,y*2,z*2]])

    slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < slab.cell[2][2]*.5]))
    slab.rattle(stdev=.2)

    slab.calc = clean_calc
    opt = BFGS(slab, trajectory='relax_run.xyz')
    opt.run(fmax=0.1)

    return slab

def make_custom_probe_scan(slab):
    f = Fragment('Cl[P+](C)(C)C', to_initialize=1)
    for atoms in f.conformers:
        for atom in atoms:
            if atom.symbol =='P':
                atom.symbol ='Al'

    m = Manifold(slab,
                precision=precision,
                touch_sphere_size =2.5,
                wrap_on='atoms',
                calc = clean_calc)
    m.normals*=-1

    m.run_probe_scan(probes=[f, Fragment('ClC', to_initialize=1), (Atoms(['Al'], [0,0,0]))])
    m.write_grid('grd_run.xyz')


def __main__():
    
    slab = make_TiN_slab(bulk, p, c, scale)
    write('slab.xyz', slab)
    
    make_custom_probe_scan(slab)


######################################################## params

clean_calc = mace_mp(model=
                '/home/fako/projects/models/mace-omat-0-medium.model',
                device='cuda',
                )

p = .1
c = 20
scale = [2,2,2]
bulk = read('./TiN.cif')
precision = 0.5

if __name__ == '__main__':
    __main__()



