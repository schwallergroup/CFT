import os
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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get('MODEL_PATH', 'mace-mh-nl-pbe.model')

RESTART     = True          # True: resume from existing trajectory; False: fresh NEB
N_IMAGES    = 11            # number of NEB images (including endpoints)
IMAGE_I     = 0             # which image to use for the probe scan
EXCLUDE_SPEC = ['O', 'C']   # atom species to exclude when building the probe-scan surface
PROBE       = Fragment('Cl[O]', to_initialize=1)
# ─────────────────────────────────────────────────────────────────────────────

clean_calc = mace_mp(
    model=MODEL_PATH,
    head='matpes_r2scan',
    device='cuda',
)


def main():

    particle = read('./test_particle.xyz')

    m = Manifold(particle.copy(), mode='particle', precision=.5, touch_sphere_size=2.5, wrap_on='sites')

    if not RESTART:
        _all_images = read('./neb_CO_in_plane.xyz', index=':')
        initial = _all_images[0].copy()
        final   = _all_images[-1].copy()

        images = [initial]
        for _ in range(N_IMAGES - 2):
            images.append(initial.copy())
        images.append(final)

        neb = NEB(images, remove_rotation_and_translation=False)
        neb.interpolate(apply_constraint=True)

    else:
        _restart_trj = read('./neb_CO_in_plane.xyz', index=':')
        images = _restart_trj[-N_IMAGES:]

    neb = NEB(images, remove_rotation_and_translation=False)
    for image in images:
        image.calc = copy.deepcopy(clean_calc)

    traj = Trajectory('neb_CO_in_plane_restart.xyz', 'w', images)
    opt  = BFGS(neb, trajectory=traj)
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

    atoms = images[IMAGE_I].copy()
    m_neb.atoms = atoms[[atom.index for atom in atoms if atom.symbol not in EXCLUDE_SPEC]]
    m_neb.run_probe_scan(probes=[PROBE])

    pop_keys = [k for k in m_neb.grid_atoms.arrays.keys() if 'grad' in k]
    for k in pop_keys:
        m_neb.grid_atoms.arrays.pop(k)

    write(f'image_{IMAGE_I}_{EXCLUDE_SPEC}_grid.xyz', m_neb.grid_atoms)


if __name__ == '__main__':
    main()
