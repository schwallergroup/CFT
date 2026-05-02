import copy
import itertools
import os
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from ase import Atoms
from ase.io import read
from mace.calculators import mace_mp

from cft import Manifold
from cft.mesh_utils import get_manifold_minima
from cft.neb_utils import get_neb_probe, get_PMD_structure

# --- Configuration ---
DATA_DIR = Path(os.environ.get('CFT_DATA_DIR', ''))  # set CFT_DATA_DIR to the paper data directory
MODEL_PATH = Path(os.environ.get('MODEL_PATH', ''))  # set MODEL_PATH to your MACE model file
PARTICLE_FILE = Path('./test_particle.xyz')
OUTPUT_FILE = Path('./df_reaction_info.csv')

# Search Thresholds
THR_O = 1.9
THR_C = 1.9
MIN_C_O_IS_DIST = 2.7
ANGLES = range(0, 360, 10)
clamp_pmd = False

def load_calculator():
    """Initialize the MACE calculator."""
    return mace_mp(
        model= MODEL_PATH / "mace-mh-nl-pbe.model",
        head='matpes_r2scan',
        device='cpu'
    )

def get_isolated_references(calculator, particle):
    """Calculate isolated atom and molecule reference energies."""
    particle_calc = copy.deepcopy(calculator)
    particle.calc = particle_calc
    e_particle = particle.get_potential_energy()

    o_atoms = Atoms(['O'], [[0, 0, 0]])
    o_atoms.calc = copy.deepcopy(calculator)
    
    c_atoms = Atoms(['C'], [[0, 0, 0]])
    c_atoms.calc = copy.deepcopy(calculator)

    return {
        'O': o_atoms.get_potential_energy(),
        'C': c_atoms.get_potential_energy(),
        'e_particle': e_particle
    }

def construct_trajectory(row, manifold, particle, fragment_pmd):
    """
    Constructs an ASE trajectory (Initial, PMD, Final) for a given DataFrame row.
    """
    from ase import Atoms
    from autoadsorbate import Fragment
    from autoadsorbate.Surf import attach_fragment
    
    # IS: C and O placed at their static basin vertices
    state_IS = particle.copy()
    c_idx, o_idx = int(row['i_c']), int(row['i_o'])
    state_IS += Atoms("C", positions=[manifold.grid[c_idx]])
    state_IS += Atoms("O", positions=[manifold.grid[o_idx]])
    
    # TS: PMD placed dynamically based on angle and specific basin
    state_TS = particle.copy()
    pmd_idx = int(row['i_pmd'])
    site_dict_ts = {"coordinates": manifold.grid[pmd_idx], "n_vector": manifold.normals[pmd_idx]}
    attach_fragment(state_TS, site_dict_ts, fragment=fragment_pmd, n_rotation=row['angle'], height=0)
    
    # FS: CO placed at the PMD site explicitly, bypassing `attach_fragment` spatial bugs
    state_FS = particle.copy()
    n_vec = manifold.normals[pmd_idx]
    pos_C = manifold.grid[pmd_idx]
    pos_O = pos_C + n_vec * 1.13  # standard C#O bond length (1.13 Angstroms)
    state_FS += Atoms("CO", positions=[pos_C, pos_O])
    
    return [state_IS, state_TS, state_FS]

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print("--- Initializing Calculator and References ---")
    calc = load_calculator()
    particle = read(PARTICLE_FILE)
    refs = get_isolated_references(calc, particle)

    # Load images and probe
    neb_images = read(DATA_DIR / 'images_neb.xyz', index=':')
    pmd_image_full = neb_images[5].copy()
    f_pmd = get_neb_probe(pmd_image_full, keep=['C', 'O']).get_conformer(0)

    # Setup Manifold
    print("--- Setting up Manifold ---")
    m = Manifold(
        particle.copy(),
        mode='particle',
        precision=0.2,
        touch_sphere_size=2.0,
        wrap_on='blend',
    )

    # Load Energy Arrays
    print("--- Loading Energy Data ---")
    mesh_ref = read(DATA_DIR / 'manifold_O_WRAPONblend_PREC0.2_TSS2.0.xyz')
    vals_O = mesh_ref.arrays['e_ClO_0']
    vals_C = read(DATA_DIR / 'manifold_C_WRAPONblend_PREC0.2_TSS2.0.xyz').arrays['e_ClC_0']
    
    # Normalize O and C values
    ref_C_on_particle = np.min(vals_C)
    ref_O_on_particle = np.min(vals_O)
    vals_C -= ref_C_on_particle
    vals_O -= ref_O_on_particle

    # Load CO two-body data
    v_co_o = read(DATA_DIR / 'CO_two_body_O.xyz').arrays['e_ClC#O+_0']
    v_co_c = read(DATA_DIR / 'CO_two_body_C.xyz').arrays['e_ClC#O+_0']
    vals_CO = v_co_o + v_co_c - refs['e_particle']
    vals_CO = vals_CO - ref_C_on_particle - ref_O_on_particle + refs['e_particle']

    # Load Anisotropy results
    print("--- Loading Anisotropy Data ---")
    aniso_trj = read(DATA_DIR / 'anisotropic_reactivity_map_CO.xyz', index=':')
    all_PMD_raw = np.array([atoms.arrays['PMD-CF_aniso'] for atoms in aniso_trj])

    # Using direct spatial mapping instead of KDTree to preserve strict topological gradients for minima finder!


    # Find Static Minima
    print("--- Finding Static Minima ---")
    co_mins_idx  = np.array(get_manifold_minima(m.faces, vals=vals_CO))
    c_mins_idx   = np.array(get_manifold_minima(m.faces, vals=vals_C))
    o_mins_idx   = np.array(get_manifold_minima(m.faces, vals=vals_O))

    c_sites = m.grid[c_mins_idx]
    o_sites = m.grid[o_mins_idx]

    # Reaction Site Search
    print(f"--- Searching Reaction Sites ({len(ANGLES)} angles) ---")
    reactions_info = []

    for idx, angle in enumerate(tqdm(ANGLES, desc="Processing angles")):
        pmd_raw_angle = all_PMD_raw[idx].copy()
        
        # Normalize PMD to find basins at this specific angle
        pmd_norm = pmd_raw_angle.copy()
        pmd_norm -= pmd_norm.min()
        if clamp_pmd:
            pmd_norm[pmd_norm > 1] = 1
        pmd_mins_idx = np.array(get_manifold_minima(m.faces, vals=pmd_norm))

        for site_i, pmd_idx in enumerate(pmd_mins_idx):
            co_at_site = get_PMD_structure(m.grid[pmd_idx], m.normals[pmd_idx], angle, fragment=f_pmd)
            o_pos = co_at_site[co_at_site.symbols == 'O'].positions[0]
            c_pos = co_at_site[co_at_site.symbols == 'C'].positions[0]

            # Find candidates within threshold
            o_candidates = [o_mins_idx[n] for n in np.where(np.linalg.norm(o_sites - o_pos, axis=1) < THR_O)[0]]
            c_candidates = [c_mins_idx[n] for n in np.where(np.linalg.norm(c_sites - c_pos, axis=1) < THR_C)[0]]

            for i_o, i_c in itertools.product(o_candidates, c_candidates):
                d_co = np.linalg.norm(m.grid[i_o] - m.grid[i_c])
                if d_co > MIN_C_O_IS_DIST:
                    # Record the full geometric info and its true PMD energy for this specific rotation
                    reactions_info.append({
                        'site': site_i,
                        'angle': angle,
                        'angle_idx': idx,
                        'i_o': i_o,
                        'i_c': i_c,
                        'i_pmd': pmd_idx,
                        'dCO': d_co,
                        'vals_PMD': pmd_raw_angle[pmd_idx]
                    })

    # Data Processing and Enrichment
    print("--- Enriching Results ---")
    df = pd.DataFrame(reactions_info)
    
    if df.empty:
        print("No valid pathways found! Try relaxing thresholds.")
        return

    # Map pre-calculated values
    df['vals_O'] = vals_O[df.i_o.values]
    df['vals_C'] = vals_C[df.i_c.values]
    df['vals_IS'] = df['vals_C'] + df['vals_O']
    df['vals_FS'] = vals_CO[df.i_pmd.values]

    # Calculate Barriers using the exact saved PMD energy for the angle/site pathway
    df['E_barrier'] = df['vals_PMD'] - df['vals_IS']
    df['E_reaction'] = df['vals_FS'] - df['vals_IS']

    # Neighborhood Analysis
    print("--- Analyzing Chemical Environments ---")
    first_neighbor = []
    first_neighborhood = []
    
    particle_positions = particle.positions
    for i in tqdm(df.i_pmd.values, desc="Analyzing neighbors"):
        point = m.grid[i]
        distances = np.linalg.norm(particle_positions - point, axis=1)
        
        neighbor_indices = np.where(distances < 4.0)[0]
        closest_index = np.where(distances == distances.min())[0]
        
        first_neighborhood.append(particle[neighbor_indices].get_chemical_formula(empirical=True))
        first_neighbor.append(particle[closest_index].get_chemical_formula(empirical=True))

    df['first_neighbor'] = first_neighbor
    df['first_neighborhood'] = first_neighborhood

    # Final Sort and Save
    df = df.sort_values(by=['E_barrier', 'first_neighborhood'])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"--- Done! Results saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()
