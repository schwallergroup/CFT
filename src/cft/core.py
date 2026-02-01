import numpy as np
from ase import Atoms
from typing import Literal, Union, Iterable, List, Annotated, Dict, Any, Optional
from ase.calculators import calculator
from ase.visualize import view
from ase.io import read, write
import random
from autoadsorbate.Surf import attach_fragment, get_shrinkwrap_ads_sites
from autoadsorbate.Particle import (
    get_shrinkwrap_particle_ads_sites,
    get_base_grid_particle,
)
from ase.db import connect
from ase.io import Trajectory
import uuid
import os
import json

from autoadsorbate import Surface, Fragment
from .mesh_utils import (
    reorient_faces_from_seed,
    compute_outward_vertex_normals_quads,
    save_ply_quads,
    compute_vertex_gradients,
    compute_gradients_per_column,
    select_non_interacting_vertices,
    estimate_radius_decay,
    compute_vertex_areas,
)
from .dynamics import ProbeScan, StaticEval


class Manifold(Surface):
    """
    Represents a geometric manifold constructed from a grid of vertices and faces,
    supporting probe scanning, visualization, and mesh export functionalities.
    Inherits from:
        Surface
        *args: Variable length argument list for the parent Surface class.
        calc (calculator, optional): ASE calculator for energy and gradient computations.
        viz_marker (str, optional): Marker symbol for grid visualization. Defaults to 'X'.
        **kwargs: Arbitrary keyword arguments for the parent Surface class.
    Attributes:
        faces (np.ndarray): Array of oriented face indices.
        normals (np.ndarray): Array of outward vertex normals.
        grid_atoms (Atoms): Atoms object representing grid vertices.
        calc (calculator): ASE calculator for probe scans.
        probe_names (list): List of probe identifiers used in scans.
    Methods:
        run_probe_scan(probes):
            Runs probe scans over the grid using the provided probes and stores energies and gradients.
        get_non_interacting_vertices(radius=3.0, decay=1.5, randomize=True):
        get_grid_atoms(inclde_atoms=True):
            Returns grid atoms, optionally including the original atoms.
        view_grid(inclde_atoms=True):
            Visualizes the grid atoms, optionally including the original atoms.
        write_grid(filename='tmp.xyz', inclde_atoms=False):
            Writes grid atoms and probe scan results to a file.
        view_hedgehog(marker='X'):
            Visualizes grid vertices with their normals as "hedgehog" markers.
        save_ply(vertex_colors=None, filename='./quad_sphere_tmp.ply'):
            Saves the mesh data to a PLY file, optionally with vertex colors.
    """

    def __init__(
        self,
        *args,
        calc: calculator = None,
        viz_marker="X",
        wrap_on: Literal["atoms", "sites", "blend"] = "sites",
        use_torch_sim: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        grid_atoms = self.grid.copy()
        self.wrap_on = wrap_on
        if self.wrap_on in ['sites', 'blend']:
            self.grid, self.faces, _ = self._shrinkwrap(self.sites_atoms)
            grid_sites = self.grid.copy()

            if self.wrap_on == 'blend':
                d_grid = grid_sites - grid_atoms
                d_grid_norm = np.linalg.norm(d_grid, axis=1)
                d_grid_norm /= np.max(d_grid_norm)

                d_grid_norm = .5 / (1 + np.exp(-(d_grid_norm)))

                # self.grid = grid_atoms + (d_grid) * .5 # blend_factor
                self.grid = grid_atoms + (d_grid) * d_grid_norm.reshape(-1, 1) # blend_factor
            
        self.faces = reorient_faces_from_seed(np.array(self.faces), self.grid)
        self.normals = compute_outward_vertex_normals_quads(self.grid, self.faces, mode=self.mode)
        self.grid_atoms = Atoms([viz_marker for _ in self.grid], self.grid)
        self.calc = calc
        self.probe_names = list()
        self.surf_population = None
        self.grid_area = compute_vertex_areas(self.grid, self.faces)
        self.grid_atoms.arrays["area"] = self.grid_area
        self._id = uuid.uuid4().hex
        self.use_torch_sim = use_torch_sim

    def get_base_grid(self):
        if self.mode == "particle":
            return get_base_grid_particle(
                particle_atoms=self.atoms,
                grid_mode=self.grid_mode,
                precision=self.precision,
                touch_sphere_size=self.touch_sphere_size,
            )
        elif self.mode == "slab":
            raise ValueError(f"{self.mode = }; slab - not supported yet.")
        else:
            raise ValueError(f"{self.mode = }; Unknown error.")

    def run_probe_scan(self, probes: List[Union[Fragment, Atoms]], n_rotation=0.):
        """
        Scans a set of probe molecules or atoms over a predefined grid on the reference structure,
        computes energies and gradients for each probe position, and stores the results.
        Parameters
        ----------
        probes : List[Union[Fragment, Atoms]]
            A list of probe objects, each being either a Fragment or an Atoms instance.
            If an Atoms object contains more than one atom, it is skipped with a warning.
        Raises
        ------
        ValueError
            If the calculator (`self.calc`) is not set.
        Side Effects
        ------------
        - Appends the name of each probe to `self.probe_names`.
        - Stores computed energies and gradients in `self.grid_atoms.arrays` with keys
          based on the probe name.
        Notes
        -----
        - For single-atom probes, the chemical formula is used as the name.
        - For Fragment probes, the SMILES string is used as the name.
        - Energies and gradients are computed using the `ProbeScan` and `compute_gradients_per_column` utilities.
        """

        if self.calc is None:
            raise ValueError(f"Please provide ase calculator to Manifold.calc.")

        self.ref_energy_dict = self.evaluate_references(probes=probes)

        for probe in probes:
            if type(probe) == Atoms and len(probe) != 1:
                Warning(
                    f"{probe = } has more than one atoms. Please pass Fragments object."
                )
                continue

            ref_atoms = self.atoms.copy()
            ref_atoms.calc = self.calc
            dyn = ProbeScan(
                ref_atoms=ref_atoms,
                probe=probe,
                vertices=self.grid,
                normals=self.normals,
                use_torch_sim=self.use_torch_sim,
                n_rotation = n_rotation
            )
            energies = dyn.run()

            if type(probe) == Atoms and len(probe) == 1:
                name = probe.get_chemical_formula()
            else:
                name = probe.smile
            self.probe_names.append(name)

            grads, grad_norms = compute_gradients_per_column(
                energies, self.grid, self.faces
            )
            self.grid_atoms.info['n_rotation'] = n_rotation
            self.grid_atoms.arrays[f"e_{name}"] = energies
            self.grid_atoms.arrays[f"grad_e_{name}"] = grads
            self.grid_atoms.arrays[f"grad_norm_e_{name}"] = grad_norms

    def write_to_db(
        self,
        atoms_list: Union[Atoms, List[Atoms]],
        db_path: Optional[str] = None,
        mode: str = "a",
        **global_metadata,
    ) -> None:
        """
        Writes Atoms objects to an ASE database, saving all info and arrays.

        The database path is handled intelligently:
        - If db_path is None (default), creates 'db_{self._id}.db' in the current dir.
        - If db_path is a directory, creates 'db_{self._id}.db' inside it.
        - If db_path is a full file path, uses that path directly.

        Args:
            atoms_list: A single Atoms object or a list of them.
            db_path: Optional path to a directory or a specific .db file.
            mode: 'w' (write/overwrite) or 'a' (append).
            **global_metadata: Metadata to add to every structure's key-value pairs.
        """
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]
        if not atoms_list:
            print("Warning: List of atoms is empty. Nothing to write.")
            return

        if db_path is None:
            db_file = f"db_{self._id}.db"
        elif os.path.isdir(db_path):
            os.makedirs(db_path, exist_ok=True)
            db_file = os.path.join(db_path, f"db_{self._id}.db")
        else:
            db_file = db_path

        print(f"Connecting to ASE database at: {db_file} (mode='{mode}')")

        with connect(db_file, append=(mode == "a")) as db:
            for i, atoms in enumerate(atoms_list):
                # Prepare key-value pairs from atoms.info
                key_value_pairs = atoms.info.copy()
                key_value_pairs.update(global_metadata)
                safe_key_value_pairs = _serialize_metadata(key_value_pairs)

                # Prepare data dictionary for custom arrays from atoms.arrays
                standard_arrays = [
                    "numbers",
                    "positions",
                    "pbc",
                    "initial_magmoms",
                    "initial_charges",
                    "masses",
                    "tags",
                    "momenta",
                    "constraints",
                ]
                custom_data = {
                    name: array
                    for name, array in atoms.arrays.items()
                    if name not in standard_arrays
                }

                # Write everything to the database
                db.write(atoms, key_value_pairs=safe_key_value_pairs, data=custom_data)

                if (i + 1) % 10 == 0:
                    print(f"  ... Wrote {i + 1}/{len(atoms_list)} structures")

        print(
            f"Successfully wrote {len(atoms_list)} structures to the database: {db_file}"
        )

    def evaluate_references(self, probes: List[Union[Fragment, Atoms]]):
        """
        Computes reference energy for each probe as atoms.get_potential_energy() + probe.get_potential_energy()
        Parameters
        ----------
        probes : List[Union[Fragment, Atoms]]
            A list of probe objects, each being either a Fragment or an Atoms instance.
            If an Atoms object contains more than one atom, it is skipped with a warning.
        returns dict of reference energies
        """
        import copy

        _atoms = self.atoms.copy()
        _atoms.calc = copy.deepcopy(self.calc)
        e_ref = _atoms.get_potential_energy()

        ref_dict = {}

        p_atoms = Atoms()
        for p in probes:
            attach_fragment(
                atoms=p_atoms,
                site_dict={"coordinates": [0, 0, 0], "n_vector": [0, 0, 1]},
                fragment=p.get_conformer(0),
                n_rotation=0,
                height=0.0,
            )
            p_atoms.calc = copy.deepcopy(self.calc)
            ref_dict["e_" + p.smile] = p_atoms.get_potential_energy() + e_ref
        return ref_dict

    def get_non_interacting_vertices(self, radius=3.0, decay=1.5, randomize=True):
        """
        Returns a selection of non-interacting vertices from the grid.

        This method identifies vertices that do not interact within a specified radius,
        using a decay factor to influence selection probability. Optionally, the selection
        can be randomized.

        Args:
            radius (float, optional): The radius within which interactions are considered. Defaults to 3.0.
            decay (float, optional): Decay factor affecting selection probability. Defaults to 1.5.
            randomize (bool, optional): If True, randomizes the selection of vertices. Defaults to True.

        Returns:
            list: A list of non-interacting vertices selected from the grid.
        """
        return select_non_interacting_vertices(
            self.grid, radius=radius, decay=decay, randomize=randomize
        )

    def get_grid_atoms(self, inclde_atoms=True):
        """
        Returns the grid atoms, optionally including additional atoms.

        Args:
            inclde_atoms (bool, optional): If True, includes both self.atoms and self.grid_atoms in the output.
                If False, returns only self.grid_atoms. Defaults to True.

        Returns:
            list: A list of atoms, either self.grid_atoms or self.atoms + self.grid_atoms depending on inclde_atoms.
        """
        """"""
        out_atoms = self.grid_atoms
        if inclde_atoms:
            out_atoms = self.atoms + self.grid_atoms
        return out_atoms

    def view_grid(self, inclde_atoms=True):
        """
        Displays a grid view of atoms.

        Args:
            inclde_atoms (bool, optional): If True, includes atoms in the grid view. Defaults to True.

        Returns:
            None
        """
        view_atoms = self.get_grid_atoms(inclde_atoms)
        view(view_atoms)

    def view_hedgehog(self, marker='X', show_with_atoms=False):
        """
        Visualizes the grid atoms along their normal vectors, creating a "hedgehog" effect.

        For each atom in the grid, additional atoms are placed along the direction of its normal vector,
        spaced at intervals from 0 to 2 (step 0.2). The marker symbol for these atoms can be customized.

        Args:
            marker (str, optional): The symbol used to represent the additional atoms. Defaults to 'X'.

        Returns:
            None
        """
        view_atoms = self.grid_atoms.copy()
        for i, v in enumerate(self.grid):
            for slide in np.arange(0,2, 0.2):
                view_atoms+=Atoms([marker], [v+slide*self.normals[i]])
        if show_with_atoms:
            view(self.atoms+view_atoms)
        else:
            view(view_atoms)
            for slide in np.arange(0, 2, 0.2):
                view_atoms += Atoms([marker], [v + slide * self.normals[i]])

    def write_grid(self, filename: str = "tmp.xyz", inclde_atoms=False):
        """
        Writes the grid atoms and associated probe data to a file in XYZ format.
        Parameters
        ----------
        filename : str, optional
            The name of the output file. Defaults to 'tmp.xyz'.
        inclde_atoms : bool, optional
            If True, includes atoms in the output. Currently not supported and will raise a ValueError.
        Raises
        ------
        ValueError
            If `inclde_atoms` is set to True.
        Notes
        -----
        For each probe name in `self.probe_names`, the method extracts energy, gradient, and gradient norm arrays
        from `self.grid_atoms`, splits them per probe component, and adds them back to the atom arrays with
        appropriately suffixed keys. The resulting atom data is then written to the specified file.
        """

        if inclde_atoms:
            raise ValueError("mode not yet supported")

        out_atoms = self.grid_atoms.copy()

        for name in self.probe_names:
            energies = out_atoms.arrays.pop(f"e_{name}")
            grads = out_atoms.arrays.pop(f"grad_e_{name}")
            grad_norms = out_atoms.arrays.pop(f"grad_norm_e_{name}")

            for j in range(energies.shape[1]):
                out_atoms.arrays[f"e_{name}_{j}"] = energies[:, j]  # (n_atoms,)
                out_atoms.arrays[f"grad_norm_e_{name}_{j}"] = grad_norms[
                    :, j
                ]  # (n_atoms,)
                out_atoms.arrays[f"grad_e_{name}_{j}"] = grads[:, j, :]  # (n_atoms,3)

        write(filename, out_atoms)

    def save_ply(
        self,
        vertex_colors: Union[Iterable, None] = None,
        filename: str = f"./quad_sphere_tmp.ply",
    ):
        """
        Function saves mesh data to ply file using Manifold class data.
        args:
        vertex_colors - optional list of RGB values
        filename - save file
        """
        print(f"Saving grid to file: {filename}")
        save_ply_quads(
            filename,
            vertices=self.grid,
            faces=self.faces,
            normals=self.normals,
            vertex_colors=vertex_colors,
        )

    def make_fragment_population(
        self,
        population_size: int,
        fragment: Fragment,
        radius: Union[float, None] = None,
        decay: Union[float, None] = None,
        coverage: Annotated[float, "in [0,1]"] = 0.8,
        anticipated_bond_len=2.0,
    ):
        """
        Generates a population of molecular structures by attaching a given fragment to non-interacting sites on a surface.
        Args:
            population_size (int): Number of structures to generate in the population.
            fragment (Fragment): The molecular fragment to attach to the surface.
            radius (float, optional): Interaction radius for selecting attachment sites. If None, estimated automatically.
            decay (float, optional): Decay parameter for site selection. If None, estimated automatically.
            coverage (float, optional): Fraction of available sites to use for fragment attachment (between 0 and 1). Default is 0.8.
            anticipated_bond_len (float, optional): Expected bond length between the fragment and the surface. Default is 2.0.
        Returns:
            List[np.ndarray]: A list of atom arrays representing the generated population of structures with attached fragments.
        Notes:
            - The function randomizes fragment orientation and attachment sites for each structure.
            - If both `radius` and `decay` are None, they are estimated from the fragment conformers.
            - The fragment is attached at selected surface sites with randomized rotation.
        """

        oriented_conformers = [
            fragment.get_conformer(i) for i, _ in enumerate(fragment.conformers)
        ]

        if radius is None and decay is None:
            radius, decay = estimate_radius_decay(oriented_conformers)

        surf_population = []

        for _ in range(population_size):
            inds = self.get_non_interacting_vertices(
                radius=radius, decay=decay, randomize=True
            )
            inds = inds[: int(len(inds) * coverage)]

            if len(fragment.conformers) < len(inds):
                conformers_prepped = (
                    oriented_conformers * (len(inds) // len(oriented_conformers) + 1)
                )[: len(inds)]
            else:
                conformers_prepped = oriented_conformers

            rotations = [random.uniform(0, 360) for _ in inds]
            random.shuffle(conformers_prepped)
            random.shuffle(inds)
            atoms = self.atoms.copy()

            if "fragments" not in atoms.arrays.keys():
                atoms.arrays["fragments"] = np.array([0 for _ in atoms])

            for i, (rot, ind) in enumerate(zip(rotations, inds)):
                frg = conformers_prepped[i]
                frg.arrays["fragments"] = np.array(
                    [np.max(atoms.arrays["fragments"]) + 1 for _ in frg]
                )

                atoms = attach_fragment(
                    atoms=atoms,
                    site_dict={
                        "coordinates": self.grid[ind],
                        "n_vector": self.normals[ind],
                    },
                    fragment=frg,
                    n_rotation=rot,
                    height=anticipated_bond_len - self.touch_sphere_size,
                )

            atoms.info["n_fragments"] = np.max(atoms.arrays["fragments"])
            surf_population.append(atoms)

        self.surf_population = surf_population

    def evaluate_surf_population(self):
        """
        Evaluates the energy of atoms in the surface population and sorts them by energy.
        Raises:
            ValueError: If the surface population (`self.surf_population`) has not been generated.
        Side Effects:
            Updates `self.surf_population` with the evaluated and sorted atoms using the provided calculator (`self.calc`).
        Note:
            To generate the surface population, use `Manifold.make_fragment_population()`.
        """

        if self.surf_population is None:
            raise ValueError(
                f"Surface population is not generated: {self.surf_population = }.\n \
                             To create surface population use: Manifold.make_fragment_population()"
            )

        # self.surf_population = evaluate_and_sort_atoms_by_energy(self.surf_population, self.calc)

        dyn = StaticEval(self.calc, use_torch_sim=self.use_torch_sim)
        dyn.run(self.surf_population)


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serializes complex data types in a metadata dictionary to JSON strings.

    This function iterates through a dictionary and converts any lists or
    dictionaries into their JSON string representation, making them safe
    to store in an ASE database.

    Args:
        metadata (Dict[str, Any]): The input dictionary (e.g., atoms.info).

    Returns:
        Dict[str, Any]: A new dictionary with complex types serialized.
    """
    serialized_kvp = {}
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            # If the value is a list or dict, dump it to a JSON string
            serialized_kvp[key] = json.dumps(value)
        elif isinstance(value, np.ndarray):
            # Also handle numpy arrays by converting them to lists first
            serialized_kvp[key] = json.dumps(value.tolist())
        else:
            # Keep simple types (int, float, str, bool) as they are
            serialized_kvp[key] = value
    return serialized_kvp


def _deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserializes any JSON strings found in a metadata dictionary.
    """
    deserialized_kvp = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            try:
                # Check for common JSON list/dict patterns
                if (value.startswith("[") and value.endswith("]")) or (
                    value.startswith("{") and value.endswith("}")
                ):
                    deserialized_kvp[key] = json.loads(value)
                else:
                    deserialized_kvp[key] = value
            except (json.JSONDecodeError, TypeError):
                deserialized_kvp[key] = value
        else:
            deserialized_kvp[key] = value
    return deserialized_kvp


def db_to_traj(
    db_path: str,
    output_traj_path: Optional[str] = None,
    selection_query: Optional[str] = None,
) -> List[Atoms]:
    """
    Reads structures from an ASE database and explicitly reconstructs them
    into a list of Atoms objects, showing the manual process of restoring
    all metadata (.info) and custom arrays (.arrays).

    This function does NOT use the `row.toatoms()` shortcut.

    Args:
        db_path (str): Path to the input ASE database file.
        output_traj_path (str, optional): If provided, the Atoms objects will be
            written to this trajectory file.
        selection_query (str, optional): An ASE database selection query string
            to filter which structures are read.

    Returns:
        List[Atoms]: A list of all the read and fully reconstructed Atoms objects.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    print(f"Reading from ASE database: {db_path}")
    if selection_query:
        print(f"Applying selection filter: '{selection_query}'")

    atoms_list = []
    with connect(db_path) as db:
        for row in db.select(selection_query):
            # --- Step 1: Deserialize key-value pairs for the .info dict ---
            deserialized_info = _deserialize_metadata(row.key_value_pairs)

            # --- Step 2: Create the base Atoms object ---
            # Use the standard attributes from the row object.
            atoms = Atoms(
                numbers=row.numbers,
                positions=row.positions,
                cell=row.cell,
                pbc=row.pbc,
                info=deserialized_info,
            )

            # --- Step 3: Attach all custom per-atom arrays ---
            # The custom arrays are stored in the `row.data` dictionary.
            if row.data:
                for name, array in row.data.items():
                    # Use the .new_array() method to attach each array.
                    atoms.new_array(name, array)

            # --- Step 4: Add the fully reconstructed object to our list ---
            atoms_list.append(atoms)

    print(f"Successfully reconstructed {len(atoms_list)} structures from the database.")

    # Optionally, write the list to a .traj file
    if output_traj_path:
        print(
            f"Writing {len(atoms_list)} structures to trajectory file: {output_traj_path}"
        )
        with Trajectory(output_traj_path, "w") as traj:
            for atoms in atoms_list:
                traj.write(atoms)

    return atoms_list


def db_to_traj_explicit(
    db_path: str,
    output_traj_path: Optional[str] = None,
    selection_query: Optional[str] = None,
) -> List[Atoms]:
    """
    Explicitly reconstructs Atoms objects from a database, skipping corrupted rows.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    print(f"Reading explicitly from ASE database: {db_path}")
    if selection_query:
        print(f"Applying selection filter: '{selection_query}'")

    atoms_list = []
    with connect(db_path) as db:
        for row in db.select(selection_query):
            try:
                # --- This is the part of the process that can fail ---
                # Reading standard attributes is usually safe.
                numbers = row.numbers
                positions = row.positions
                cell = row.cell
                pbc = row.pbc

                # The .data attribute for custom arrays is also usually safe.
                custom_arrays = row.data
                # ----------------------------------------------------

                # Now, reconstruct the object
                deserialized_info = _deserialize_metadata(row.key_value_pairs)
                atoms = Atoms(
                    numbers=numbers,
                    positions=positions,
                    cell=cell,
                    pbc=pbc,
                    info=deserialized_info,
                )
                if custom_arrays:
                    for name, array in custom_arrays.items():
                        atoms.new_array(name, array)

                atoms_list.append(atoms)

            except ValueError as e:
                # --- Gracefully handle the error ---
                print(
                    f"\n[WARNING] Skipping row with ID={row.id} due to a reconstruction error."
                )
                print(f"  > Error Type: ValueError")
                print(f"  > Error Message: {e}")
                print(
                    f"  > This often indicates a corrupted or incompatible array in the database.\n"
                )
                continue  # Move to the next row

    print(f"Successfully reconstructed {len(atoms_list)} structures from the database.")

    # ... (writing to .traj file logic is the same) ...

    return atoms_list
