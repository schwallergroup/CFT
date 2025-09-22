from ase.io import read, write
from ase.constraints import FixedLine, FixAtoms
from ase.optimize import BFGS
import numpy as np
from tqdm import tqdm
from autoadsorbate.Surf import attach_fragment
from ase.io.trajectory import Trajectory


class ProbeScan:
    def __init__(self, ref_atoms, probe_atom, coordinates, normals=None):
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
            if not len(coordinates) == len(normals):
                raise ValueError(f'{len(coordinates) == len(normals) = }. Must be true.')

        self.ref_atoms = ref_atoms
        self.probe_atom = probe_atom
        self.coordinates = np.array(coordinates)
        self.normals = normals

        # if probe_atom.get_global_number_of_atoms() != 1:
        #     raise ValueError("probe_atom must contain exactly one atom.")

        if self.ref_atoms.calc is None:
            raise ValueError("ref_atoms must have a calculator attached.")

    def run(self, subtract_ref=True, show_progress=True):
        """
        Run the probe scan.

        Parameters
        ----------
        subtract_ref : bool, default True
            If True, return interaction energies relative to reference system.
            If False, return total energies of combined system.
        show_progress : bool, default True
            If True, display a progress bar.

        Returns
        -------
        list of float
            Energies for each probe position.
        """
        energies = []
        # ref_energy = self.ref_atoms.get_potential_energy()
        ref_energy = 0

        iterator = self.coordinates
        if show_progress:
            iterator = tqdm(self.coordinates, desc="Scanning probe positions")

        debug_traj = []
        for i, pos in enumerate(iterator):
            
            probe = self.probe_atom.copy()
            
            if len(probe) == 1:
                probe.set_positions([pos])
                system = self.ref_atoms + probe
            else:
                system = attach_fragment(
                    atoms = self.ref_atoms.copy(),
                    site_dict= {
                        'coordinates': pos,
                        'n_vector': self.normals[i]
                        },
                    fragment = probe, # ase.atoms.Atoms,
                    n_rotation = 0,
                    height = 0,
                )
            system.calc = self.ref_atoms.calc

            debug_traj+=[system]

            e = system.get_potential_energy()
            if subtract_ref:
                e -= ref_energy
            energies.append(e)
        
        write('tmp.xyz', debug_traj)
            
        return energies

class ProbeLineOpt:
    def __init__(self, ref_atoms, probe_atom, coordinates, vectors):
        """
        Optimize probe atom positions along fixed lines near a frozen reference.

        Parameters
        ----------
        ref_atoms : ase.Atoms
            Reference system (must have calculator attached).
        probe_atom : ase.Atoms
            Single-atom Atoms object (the probe).
        coordinates : list of [x, y, z]
            Starting positions for probe atom.
        vectors : list of [vx, vy, vz]
            Direction vectors defining allowed movement for probe atom.
            Must be same length as coordinates.
        """
        self.ref_atoms = ref_atoms
        self.probe_atom = probe_atom
        self.coordinates = np.array(coordinates)
        self.vectors = np.array(vectors)

        if probe_atom.get_global_number_of_atoms() != 1:
            raise ValueError("probe_atom must contain exactly one atom.")

        if self.ref_atoms.calc is None:
            raise ValueError("ref_atoms must have a calculator attached.")

        if len(self.coordinates) != len(self.vectors):
            raise ValueError("coordinates and vectors must have same length.")

    def run(self, fmax=0.01, steps=200, subtract_ref=True,
            show_progress=True, traj_file=None):
        """
        Run probe optimizations along fixed lines.

        Parameters
        ----------
        fmax : float, default 0.01
            Convergence criterion for maximum force.
        steps : int, default 200
            Maximum number of optimization steps per probe.
        subtract_ref : bool, default True
            If True, return interaction energies relative to reference system.
            If False, return total energies of combined system.
        show_progress : bool, default True
            If True, display a progress bar.
        traj_file : str or None
            If provided, save trajectories to this file (all probes appended).

        Returns
        -------
        energies : list of float
            Energies after optimization for each probe position.
        positions : list of [x, y, z]
            Optimized positions of the probe atom.
        """
        energies = []
        positions = []
        ref_energy = self.ref_atoms.get_potential_energy()

        iterator = zip(self.coordinates, self.vectors)
        if show_progress:
            iterator = tqdm(iterator, total=len(self.coordinates),
                            desc="Optimizing probe positions")

        traj = None
        if traj_file is not None:
            traj = Trajectory(traj_file, 'w')

        for i, (pos, vec) in enumerate(iterator):
            probe = self.probe_atom.copy()
            probe.set_positions([pos])

            system = self.ref_atoms + probe
            system.calc = self.ref_atoms.calc

            # Constraints: freeze ref_atoms + constrain probe along line
            fix_ref = FixAtoms(indices=range(len(self.ref_atoms)))
            line_constraint = FixedLine(len(system) - 1, vec)
            system.set_constraint([fix_ref, line_constraint])

            # Optimizer with trajectory logging if requested
            if traj is not None:
                opt = BFGS(system, trajectory=traj, logfile=None)
            else:
                opt = BFGS(system, logfile=None)

            opt.run(fmax=fmax, steps=steps)

            # Results
            e = system.get_potential_energy()
            if subtract_ref:
                e -= ref_energy
            energies.append(e)

            probe_pos = system.positions[-1].copy()
            positions.append(probe_pos)

        if traj is not None:
            traj.close()

        return energies, positions
