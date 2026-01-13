from ase import Atoms
from ase.neighborlist import neighbor_list, natural_cutoffs
import numpy as np


def check_phosphorus_bonding(
    metal_cluster: Atoms, ligands: Atoms, mult: float = 1.2
) -> bool:
    """
    Checks if all Phosphorus atoms in the 'ligands' object are bonded to
    at least one atom in the 'metal_cluster'.

    Parameters:
    - metal_cluster: ASE Atoms object containing the metal atoms.
    - ligands: ASE Atoms object containing the ligands.
    - mult: Cutoff multiplier (default 1.2). Increases the covalent radii sum
            slightly to account for bond stretching/tolerance.

    Returns:
    - True if ALL Phosphorus atoms in 'ligands' are bonded to the cluster.
    - False otherwise.
    """

    # 1. Combine systems to use ASE's neighbor list tools efficiently
    # We need to track which atoms come from where
    n_cluster = len(metal_cluster)
    combined = metal_cluster + ligands

    # 2. Identify indices
    # Cluster indices are 0 to n_cluster-1
    cluster_indices = set(range(n_cluster))

    # Ligand indices start from n_cluster
    # We specifically want P atoms (atomic number 15)
    p_indices_in_combined = [
        i + n_cluster
        for i, number in enumerate(ligands.get_atomic_numbers())
        if number == 15
    ]

    if not p_indices_in_combined:
        print("No Phosphorus atoms found in the ligands object.")
        return False

    # 3. Generate adaptive cutoffs
    # natural_cutoffs returns a list of radii for each atom.
    # The neighbor_list function uses these to check if dist < r_i + r_j
    cutoffs = natural_cutoffs(combined, mult=mult)

    # 4. Compute connectivity
    # 'i' and 'j' are lists of bonded atom indices.
    i_list, j_list = neighbor_list("ij", combined, cutoffs)

    # 5. Check each Phosphorus atom
    for p_idx in p_indices_in_combined:
        # Find all neighbors of this specific P atom
        # (Where the 'i' list matches our P atom index)
        neighbors_of_p = j_list[i_list == p_idx]

        # Check if any of these neighbors are in the cluster_indices set
        bonded_to_cluster = False
        for neighbor_idx in neighbors_of_p:
            if neighbor_idx in cluster_indices:
                bonded_to_cluster = True
                break

        if not bonded_to_cluster:
            print(f"Phosphorus atom {p_idx} (index in combined system) is detached.")
            return False

    return True


def check_all_p_bonded(atoms: Atoms, metal_symbol: str, mult: float = 1.2) -> bool:
    """
    Check if all the P atoms are bonded to the metal nanoparticle.
    """

    is_metal = np.array([symbol == metal_symbol for symbol in atoms.symbols])
    metal_cluster = atoms[is_metal]
    ligands = atoms[~is_metal]

    return check_phosphorus_bonding(metal_cluster, ligands, mult)
