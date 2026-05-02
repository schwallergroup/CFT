import copy

import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.io import Trajectory
from ase.neb import NEB, NEBTools
from ase.constraints import FixAtoms, FixedPlane, FixCartesian

from autoadsorbate import Fragment
from autoadsorbate.Surf import attach_fragment

from mace.calculators import mace_mp

from cft import Manifold
from cft.mesh_utils import values_to_colors, generate_plane_mesh

clean_calc = mace_mp(
    model="mace-mh-nl-pbe.model",
    head='matpes_r2scan',
    device='cuda',
)


def main():

    particle = read('./test_particle.xyz')

    m = Manifold(particle.copy(), mode='particle', precision=.5, touch_sphere_size=2.5, wrap_on='sites')

    if restart == False:
        initial = endpoint_trj[0].copy()
        final = endpoint_trj[1].copy()

        n_images = 11
        images = [initial]
        for i in range(n_images - 2):
            image = initial.copy()
            images.append(image)
        images.append(final)

        neb = NEB(images, remove_rotation_and_translation=False)
        neb.interpolate(apply_constraint=True)

    else:
        images = images_restart.copy()

    neb = NEB(images, remove_rotation_and_translation=False)
    for image in images:
        image.calc = copy.deepcopy(clean_calc)

    traj = Trajectory(f'neb_CO_in_plane_restart.xyz', 'w', images)

    opt = BFGS(neb, trajectory=traj)
    opt.run(fmax=0.1, steps=500)

    nebtools = NEBTools(images)
    fig = nebtools.plot_band()
    fig.savefig("neb_profile.png")

    verts, edges, faces, normals = generate_plane_mesh(images, resolution=.1)
    m_neb = Manifold(particle, mode='particle', precision=.5, touch_sphere_size=2.5, wrap_on='sites')

    m_neb.calc = copy.deepcopy(clean_calc)
    m_neb.grid = verts
    m_neb.grid_atoms = Atoms(positions=verts, symbols=['X' for _ in verts])
    m_neb.normals = normals
    m_neb.faces = faces

    atoms = images[image_i].copy()
    m_neb.atoms = atoms[[atom.index for atom in atoms if atom.symbol not in exclude_spec]]
    m_neb.run_probe_scan(probes=[f])

    pop_keys = [k for k in m_neb.grid_atoms.arrays.keys() if 'grad' in k]

    for k in pop_keys:
        m_neb.grid_atoms.arrays.pop(k)

    write(f'image_{image_i}_{exclude_spec}_grid.xyz', m_neb.grid_atoms)


_all_images = read('./neb_CO_in_plane.xyz', index=':')
endpoint_trj = [_all_images[0], _all_images[-1]]  # IS and FS for a fresh NEB run

images_restart = read('./neb_CO_in_plane.xyz', index=':')
images_restart = images_restart[-11:]
image_i = 0

exclude_spec = ['O', 'C']
f = Fragment('Cl[O]', to_initialize=1)

# f = Fragment('Cl[C]', to_initialize=1)
# exclude_spec = ['O']

# f = Fragment('Cl[O]', to_initialize=1)
# exclude_spec = ['C']

restart = True

if __name__ == '__main__':
    main()
    # for image_i, _ in enumerate(images_restart):
    #     main()
