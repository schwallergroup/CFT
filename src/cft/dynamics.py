from ase.io import read, write
from ase import Atoms
from ase.constraints import FixedLine, FixAtoms
from ase.optimize import BFGS
import numpy as np
from tqdm import tqdm
from autoadsorbate.Surf import attach_fragment
from ase.io.trajectory import Trajectory
from autoadsorbate import Fragment
from itertools import product
from typing import Union, Literal

class ProbeScan:
    def __init__(self, ref_atoms,
                 probe, vertices, normals=None,
                 ):
        """
        Evaluate energies of a probe atom placed at multiple coordinates
        near a reference structure.

        Parameters
        ----------
        ref_atoms : ase.Atoms
            Reference system (must have calculator attached).
        probe_atom : ase.Atoms
            Single-atom Atoms object (the probe).
        coordinates : list of [x, y, z]
            Positions to place the probe atom.
        """
        if normals is not None:
            if not len(vertices) == len(normals):
                raise ValueError(f'{len(vertices) == len(normals) = }. Must be true.')

        self.ref_atoms = ref_atoms
        self.probe = probe
        self.coordinates = np.array(vertices)
        self.normals = normals
        # self.mode = mode

        # if probe_atom.get_global_number_of_atoms() != 1:
        #     raise ValueError("probe_atom must contain exactly one atom.")

        if self.ref_atoms.calc is None:
            raise ValueError("ref_atoms must have a calculator attached.")


    def run(self):
        """
        Run the probe scan.

        Returns
        -------
        energies : np.ndarray
            Array of shape (len(coordinates), len(probe.conformers))
            Interaction energies for each probe position and conformer.
        """

        if isinstance(self.probe, Atoms):
            # Single conformer → treat as 1-column array
            energies = np.zeros((len(self.coordinates), 1))
            iterator = enumerate(tqdm(self.coordinates, desc="Scanning probe positions"))
            for i, pos in iterator:
                probe = self.probe.copy()
                probe.set_positions([pos])
                system = self.ref_atoms + probe
                system.calc = self.ref_atoms.calc
                energies[i, 0] = system.get_potential_energy()

        elif isinstance(self.probe, Fragment):
            n_confs = len(self.probe.conformers)
            energies = np.zeros((len(self.coordinates), n_confs))
            conformers = [self.probe.get_conformer(j) for j in range(n_confs)]
            iterator = enumerate(tqdm(list(product(range(len(self.coordinates)), range(n_confs))),
                                    desc="Scanning probe positions"))
            for idx, (i_coord, j_conf) in iterator:
                pos = self.coordinates[i_coord]
                probe = conformers[j_conf]
                
                en = get_static_energy(
                    atoms = self.ref_atoms.copy(),
                    pos = pos,
                    probe = probe,
                    normal = self.normals[i_coord],
                    n_rotation = 0,
                    height = 0,
                    calc = self.ref_atoms.calc
                )
                energies[i_coord, j_conf] = en

        print(f'{energies.shape = }')
        return energies

def get_static_energy(
        atoms: Atoms,
        pos: Union[list, np.array, tuple],
        probe: Atoms,
        normal: Union[list, np.array, tuple],
        n_rotation: float = 0,
        height=0,
        calc = None
        ):
                
    system = attach_fragment(
        atoms=atoms.copy(),
        site_dict={
            'coordinates': pos,
            'n_vector': normal
        },
        fragment=probe,
        n_rotation=n_rotation,
        height=height,
    )
    system.calc = calc
    # #debug mode
    # write('debug_atoms.xyz', system, append=True)
    return system.get_potential_energy()

# class ProbeLineOpt:
#     def __init__(self, ref_atoms, probe_atom, coordinates, vectors):
#         """
#         Optimize probe atom positions along fixed lines near a frozen reference.

#         Parameters
#         ----------
#         ref_atoms : ase.Atoms
#             Reference system (must have calculator attached).
#         probe_atom : ase.Atoms
#             Single-atom Atoms object (the probe).
#         coordinates : list of [x, y, z]
#             Starting positions for probe atom.
#         vectors : list of [vx, vy, vz]
#             Direction vectors defining allowed movement for probe atom.
#             Must be same length as coordinates.
#         """
#         self.ref_atoms = ref_atoms
#         self.probe_atom = probe_atom
#         self.coordinates = np.array(coordinates)
#         self.vectors = np.array(vectors)

#         if probe_atom.get_global_number_of_atoms() != 1:
#             raise ValueError("probe_atom must contain exactly one atom.")

#         if self.ref_atoms.calc is None:
#             raise ValueError("ref_atoms must have a calculator attached.")

#         if len(self.coordinates) != len(self.vectors):
#             raise ValueError("coordinates and vectors must have same length.")

#     def run(self, fmax=0.01, steps=200, subtract_ref=True,
#             show_progress=True, traj_file=None):
#         """
#         Run probe optimizations along fixed lines.

#         Parameters
#         ----------
#         fmax : float, default 0.01
#             Convergence criterion for maximum force.
#         steps : int, default 200
#             Maximum number of optimization steps per probe.
#         subtract_ref : bool, default True
#             If True, return interaction energies relative to reference system.
#             If False, return total energies of combined system.
#         show_progress : bool, default True
#             If True, display a progress bar.
#         traj_file : str or None
#             If provided, save trajectories to this file (all probes appended).

#         Returns
#         -------
#         energies : list of float
#             Energies after optimization for each probe position.
#         positions : list of [x, y, z]
#             Optimized positions of the probe atom.
#         """
#         energies = []
#         positions = []
#         ref_energy = self.ref_atoms.get_potential_energy()

#         iterator = zip(self.coordinates, self.vectors)
#         if show_progress:
#             iterator = tqdm(iterator, total=len(self.coordinates),
#                             desc="Optimizing probe positions")

#         traj = None
#         if traj_file is not None:
#             traj = Trajectory(traj_file, 'w')

#         for i, (pos, vec) in enumerate(iterator):
#             probe = self.probe_atom.copy()
#             probe.set_positions([pos])

#             system = self.ref_atoms + probe
#             system.calc = self.ref_atoms.calc

#             # Constraints: freeze ref_atoms + constrain probe along line
#             fix_ref = FixAtoms(indices=range(len(self.ref_atoms)))
#             line_constraint = FixedLine(len(system) - 1, vec)
#             system.set_constraint([fix_ref, line_constraint])

#             # Optimizer with trajectory logging if requested
#             if traj is not None:
#                 opt = BFGS(system, trajectory=traj, logfile=None)
#             else:
#                 opt = BFGS(system, logfile=None)

#             opt.run(fmax=fmax, steps=steps)

#             # Results
#             e = system.get_potential_energy()
#             if subtract_ref:
#                 e -= ref_energy
#             energies.append(e)

#             probe_pos = system.positions[-1].copy()
#             positions.append(probe_pos)

#         if traj is not None:
#             traj.close()

#         return energies, positions


# def place_fragment(
#         atoms: Atoms,
#         site_dict: dict,
#         fragment: Atoms,
#         n_rotation: float,
#         height: float = None,
#     ):
#     pass