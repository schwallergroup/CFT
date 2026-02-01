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
from typing import Union, Literal, List
import copy
import torch
try:
    import torch_sim as ts
    from torch_sim.autobatching import BinningAutoBatcher
    from torch_sim.models.mace import MaceModel
except ImportError:
    print("torch-sim-atomistic not installed, defaulting to sequential optimization")
from multiprocessing import Pool
from functools import partial
from itertools import product
from tqdm import tqdm


class ProbeScan:
    def __init__(
        self,
        ref_atoms,
        probe: Union[Atoms, Fragment],
        vertices,
        normals=None,
        use_torch_sim=False,
        n_rotation = 0.
    ):
        """
        Evaluate energies of a probe atom placed at multiple coordinates
        near a reference structure.

        Parameters
        ----------
        ref_atoms : ase.Atoms
            Reference system (must have calculator attached).
        probe : Union[Atoms, Fragment]
            Single-atom Atoms object (the probe).
        coordinates : list of [x, y, z]
            Positions to place the probe atom.
        """
        if normals is not None:
            if not len(vertices) == len(normals):
                raise ValueError(f"{len(vertices) == len(normals) = }. Must be true.")

        self.ref_atoms = ref_atoms
        self.probe = probe
        self.coordinates = np.array(vertices)
        self.normals = normals
        self.use_torch_sim = use_torch_sim
        self.n_rotation = n_rotation
        # self.mode = mode

        # if probe_atom.get_global_number_of_atoms() != 1:
        #     raise ValueError("probe_atom must contain exactly one atom.")

        if self.ref_atoms.calc is None:
            raise ValueError("ref_atoms must have a calculator attached.")

    def run_sequential(self):
        """
        Run the probe scan in a sequential manner.

        Returns
        -------
        energies : np.ndarray
            Array of shape (len(coordinates), len(probe.conformers))
            Interaction energies for each probe position and conformer.
        """

        if isinstance(self.probe, Atoms):
            # Single conformer → treat as 1-column array
            energies = np.zeros((len(self.coordinates), 1))
            iterator = enumerate(
                tqdm(self.coordinates, desc="Scanning probe positions")
            )
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
            iterator = enumerate(
                tqdm(
                    list(product(range(len(self.coordinates)), range(n_confs))),
                    desc="Scanning probe positions",
                )
            )
            for idx, (i_coord, j_conf) in iterator:
                pos = self.coordinates[i_coord]
                probe = conformers[j_conf]

                en = get_static_energy(
                    atoms=self.ref_atoms.copy(),
                    pos=pos,
                    probe=probe,
                    normal=self.normals[i_coord],
                    n_rotation=self.n_rotation,
                    height=0,
                    calc=self.ref_atoms.calc,
                )
                energies[i_coord, j_conf] = en

        return energies

    def run_torch_sim(self):
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
            systems = []
            iterator = enumerate(
                tqdm(self.coordinates, desc="Scanning probe positions")
            )
            for i, pos in iterator:
                probe = self.probe.copy()
                probe.set_positions([pos])
                system = self.ref_atoms + probe
                systems.append(system)

            all_energies = get_batched_single_point(systems)

        elif isinstance(self.probe, Fragment):
            n_confs = len(self.probe.conformers)
            conformers = [self.probe.get_conformer(j) for j in range(n_confs)]
            all_energies = np.zeros((len(self.coordinates), n_confs))
            systems = []
            indices = []
            iterator = enumerate(
                list(product(range(len(self.coordinates)), range(n_confs)))
            )
            for idx, (i_coord, j_conf) in iterator:
                pos = self.coordinates[i_coord]
                probe = conformers[j_conf]

                system = attach_fragment(
                    atoms=self.ref_atoms.copy(),
                    site_dict={"coordinates": pos, "n_vector": self.normals[i_coord]},
                    fragment=probe,
                    n_rotation=0,
                    height=0,
                )
                systems.append(system)
                indices.append((i_coord, j_conf))

            systems[0].calc = copy.deepcopy(self.ref_atoms.calc)
            indices = np.array(indices)

            energies = get_batched_single_point(systems)
            for (i, j), energy in zip(indices, energies):
                all_energies[i, j] = energy

        return all_energies

    def run(self):
        if self.use_torch_sim:
            return self.run_torch_sim()
        else:
            return self.run_sequential()


def get_batched_single_point(systems: list[Atoms]):
    # Extract the MACE model from the calculator
    raw_mace_model = systems[0].calc.models[0]

    # Wrap it for TorchSim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaceModel(model=raw_mace_model, device=device)

    energies = np.zeros((len(systems), 1))
    # now use torch-sim to compute energies in automatic batches
    final_state = ts.static(
        system=systems,
        model=model,
        pbar=True,
        autobatcher=BinningAutoBatcher(
            model=model, memory_scales_with="n_atoms", max_atoms_to_try=20_000
        ),
    )
    # results_atoms = final_state.get_atoms()
    for i, atoms in enumerate(final_state):
        energies[i, 0] = atoms["potential_energy"]

    return energies


def get_static_energy(
    atoms: Atoms,
    pos: Union[list, np.array, tuple],
    probe: Atoms,
    normal: Union[list, np.array, tuple],
    n_rotation: float = 0,
    height=0,
    calc=None,
):
    _atoms = atoms.copy()
    _atoms = _atoms[[atom.index for atom in _atoms if atom.symbol != 'X']]
    system = attach_fragment(
        atoms=_atoms,
        site_dict={"coordinates": pos, "n_vector": normal},
        fragment=probe,
        n_rotation=n_rotation,
        height=height,
    )
    system.calc = calc
    # #debug mode
    # write('debug_atoms.xyz', system, append=True)
    return system.get_potential_energy()


def evaluate_and_sort_atoms_by_energy(
    atoms_list: List[Atoms], calculator
) -> List[Atoms]:
    """
    Compute potential energies for a list of ASE Atoms objects,
    store them in atoms.info['static_energy'], and return a list
    sorted by energy (lowest first).

    Args:
        atoms_list (List[Atoms]): List of ASE Atoms objects.
        calculator: ASE calculator instance to attach to each Atoms object.

    Returns:
        List[Atoms]: Sorted list of Atoms by potential energy.
    """
    raise DeprecationWarning("This function is deprecated. Use torch-sim instead.")
    for atoms in tqdm(atoms_list, desc="Calculating energies"):
        atoms.calc = copy.deepcopy(calculator)  # attach calculator
        energy = atoms.get_potential_energy()  # compute energy
        atoms.info["static_energy"] = energy
        atoms.info["static_energy_per_fragment"] = (
            energy / atoms.info["n_fragments"]
        )  # store energy

    # Sort by stored energy
    sorted_list = sorted(atoms_list, key=lambda x: x.info["static_energy_per_fragment"])
    return sorted_list


from ase import Atoms
from collections import defaultdict
from typing import List
from tqdm import tqdm


from ase import Atoms
from collections import defaultdict
from typing import List
from tqdm import tqdm
import copy


class StaticEval:
    """Efficiently calculate potential energies by grouping identical compositions."""

    def __init__(self, calculator, use_torch_sim=False):
        self.use_torch_sim = use_torch_sim
        self.clean_calc = copy.deepcopy(calculator)

    def run_torch_sim(self, atoms_list: List[Atoms]) -> List[Atoms]:
        """Calculate energies and store in atoms.info['static_energy']."""
        if not atoms_list:
            return atoms_list

        atoms_list[0].calc = copy.deepcopy(self.clean_calc)
        for atoms in atoms_list:
            atoms.center(vacuum=1.5)
            atoms.pbc = False

        energies = get_batched_single_point(atoms_list)
        for i, atoms in enumerate(atoms_list):
            atoms.info["static_energy"] = energies[i, 0]
        return atoms_list

    def run_sequential(self, atoms_list: List[Atoms]) -> List[Atoms]:
        """Calculate energies and store in atoms.info['static_energy']."""
        if not atoms_list:
            return atoms_list

        groups = self._group_atoms(atoms_list)

        with tqdm(total=len(atoms_list), desc="Calculating energies") as pbar:
            for indices, template_atoms in groups.values():
                template_atoms.calc = copy.deepcopy(self.clean_calc)

                for idx in indices:
                    atoms = atoms_list[idx]
                    template_atoms.set_positions(atoms.get_positions())
                    template_atoms.set_cell(atoms.get_cell())

                    # try:
                    energy = template_atoms.get_potential_energy()
                    atoms.info["static_energy"] = energy
                    # except:
                    #     atoms.info['static_energy'] = None

                    pbar.update(1)

        return atoms_list

    def run(self, atoms_list: List[Atoms]) -> List[Atoms]:
        if self.use_torch_sim:
            return self.run_torch_sim(atoms_list)
        else:
            return self.run_sequential(atoms_list)

    def _group_atoms(self, atoms_list: List[Atoms]) -> dict:
        """Group atoms by composition and ordering."""
        groups = defaultdict(list)

        for i, atoms in enumerate(atoms_list):
            key = (len(atoms), tuple(atoms.get_chemical_symbols()))
            groups[key].append(i)

        return {
            key: (indices, atoms_list[indices[0]].copy())
            for key, indices in groups.items()
        }


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
