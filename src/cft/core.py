import numpy as np
from ase import Atoms
from typing import Literal, Union, Iterable, List
from ase.calculators import calculator
from ase.visualize import view
from ase.io import read, write

from autoadsorbate import Surface, Fragment
from .mesh_utils import reorient_faces_from_seed, compute_outward_vertex_normals_quads, save_ply_quads
from .dynamics import ProbeScan

class Manifold(Surface):
    def __init__(self, *args, calc: calculator = None, viz_marker='X', **kwargs):
        super().__init__(*args, **kwargs)

        self.faces = reorient_faces_from_seed(np.array(self.faces), self.grid)
        self.normals = compute_outward_vertex_normals_quads(self.grid, self.faces)
        self.grid_atoms = Atoms([viz_marker for _ in self.grid], self.grid)
        self.calc = calc
    
    def run_probe_scan(self, probes: List[Union[Fragment, Atoms]]):

        if self.calc is None:
            raise ValueError(f'Please provide ase calculator to Manifold.calc.')
        
        for probe in probes:

            if type(probe) == Atoms and len(probe) != 1:
                Warning(f'{probe = } has more than one atoms. Please pass Fragments object.')
                continue

            ref_atoms = self.atoms.copy()
            dyn = ProbeScan(
                ref_atoms = ref_atoms,
                probe = probe,
                vertices = self.grid,
                normals = self.normals
            )
            energies = dyn.run()

            if len(probe) == 1:
                name = probe.get_chemical_formula()
            else:
                name = probe.smile

            self.atoms.arrays[f'e_{name}']

    def get_grid_atoms(self, inclde_atoms=True):
        out_atoms = self.grid_atoms
        if inclde_atoms:
            out_atoms = self.atoms + self.grid_atoms
        return out_atoms

    def view_grid(self, inclde_atoms=True):
        view_atoms = self.get_grid_atoms(inclde_atoms)
        view(view_atoms)

    def write_grid(self, filename: str = 'tmp.xyz', inclde_atoms=True):
        out_atoms = self.get_grid_atoms(inclde_atoms)
        write(filename, out_atoms)

    def save_ply(self,
            vertex_colors: Union[Iterable, None] = None,
            filename: str = f"./quad_sphere_tmp.ply",
            ):
        """
        Function saves mesh data to ply file using Manifold class data. 
        args:
        vertex_colors - optional list of RGB values
        filename - save file
        """
        print(f'Saving grid to file: {filename}')
        save_ply_quads(
            filename,
            vertices = self.grid,
            faces = self.faces,
            normals = self.normals,
            vertex_colors = vertex_colors
            )
