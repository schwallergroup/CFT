from autoadsorbate import Fragment
from cft import Manifold
from ase import Atoms
import copy
from mace.calculators import mace_mp
from ase.optimize import BFGS
import pandas as pd
import argparse
import uuid
import torch
import torch_sim as ts
from torch_sim.runners import generate_force_convergence_fn
from torch_sim.models.mace import MaceModel
import numpy as np


from utils import check_all_p_bonded


def atoms_from_string(raw_text: str) -> Atoms:
    """
    Creates an ASE Atoms object from a single-line, semicolon-delimited string.
    """
    symbols = []
    positions = []

    for part in raw_text.split(";"):
        words = part.strip().split()

        if len(words) == 4:
            try:
                pos = [float(c) for c in words[1:]]
                symbols.append(words[0])
                positions.append(pos)
            except ValueError:
                pass

    return Atoms(symbols=symbols, positions=positions)


def get_cluster_traj(xdf):
    meta_columns = [c for c in xdf.columns if c != "structure_poscar_format"]

    cluster_trj = []

    for idx, row in xdf.iterrows():
        xyz_lines = row["structure_xyz"]
        atoms = atoms_from_string(xyz_lines)
        metadata = {col: row[col] for col in meta_columns}

        cluster_trj.append(atoms)

    return cluster_trj


def main(args):
    probes = [Fragment("Cl[P]", to_initialize=1)]

    # calculator
    clean_calc = mace_mp(
        model="/cluster/project/krause/frankem/CFT/models/mace-mh-1.model",
        device="cuda" if torch.cuda.is_available() else "cpu",  # default = 'cuda'
        head="omat_pbe",  # default = 1.5
    )

    mace_model = MaceModel(
        model=copy.deepcopy(clean_calc.models[0]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # clusters
    df = pd.read_json(
        "/cluster/project/krause/frankem/CFT/examples/swiss_ai/data/data.json",
        lines=False,
    )
    df = df.T
    df = df[
        (df["energy_relative"] < 0.1) & df.element_symbol.isin([args.metal])
    ].copy()  # & (df['n_atoms'].astype(int) > 50)

    # shuffle df to randomize the order of the configurations
    df = df.sample(frac=1).reset_index(drop=True)

    cluster_trj = get_cluster_traj(df)

    # fragments
    dfl = pd.read_csv(
        "/cluster/project/krause/frankem/CFT/examples/swiss_ai/data/phosphine_ligands_enriched.csv",
        delimiter=",",
    )
    dfl = dfl.dropna()

    # randomize the order of the fragments as well
    dfl = dfl.sample(frac=1).reset_index(drop=True)

    fragments = [
        Fragment(smi, to_initialize=args.f_conformers)
        for smi in dfl.surrogateSMILES.values
    ]

    print(f"Len cluster_trj: {len(cluster_trj)}")

    print(f"Len fragments: {len(fragments)}")

    print("Number of configurations to process: ", len(cluster_trj) * len(fragments))

    for trj_i, atoms in enumerate(cluster_trj):
        atoms.calc = copy.deepcopy(clean_calc)
        atoms.info["e_naked_cluster"] = atoms.get_potential_energy()
        print(atoms)

        m = Manifold(
            atoms,
            precision=args.precision,
            mode="particle",
            touch_sphere_size=args.touch_sphere_size,
            calc=copy.deepcopy(clean_calc),
            wrap_on="sites",
            use_torch_sim=args.use_torch_sim,
        )

        print(f"{len(m.grid) = }")

        fragment_categories = dfl["Category"].values

        for f, f_category in zip(fragments, fragment_categories):
            # Category can be used to ensure we have a good coverage across all ligand categories
            print(f"{f_category = }")

            # we terminate when either the number of retries is reached or the coverage is met
            max_retries = args.max_retries
            min_num_relaxed = args.to_relax
            num_relaxed = 0

            for retry in range(max_retries):
                if num_relaxed >= min_num_relaxed:
                    print(
                        f"Number of target relaxed configurations reached, stopping iteration"
                    )
                    # Number of target relaxed configurations reached, stop iteration
                    break
                else:
                    print(
                        f"Found {num_relaxed} / {min_num_relaxed} relaxed in try {retry + 1}, need {min_num_relaxed - num_relaxed}"
                    )

                # NOTE this will give a new random surface population each time it's called
                m.make_fragment_population(
                    population_size=args.population_size,
                    fragment=f,
                    coverage=args.coverage,
                )

                m.evaluate_surf_population()

                for a in m.surf_population:
                    a.info["static_reward"] = 0.0
                    if a.info["n_fragments"] > 0:
                        a.info["static_reward"] = (
                            a.info["static_energy"] - a.info["e_naked_cluster"]
                        ) / a.info["n_fragments"]
                    a.info["stage"] = "raw"

                surf_population_sorted = sorted(
                    m.surf_population,
                    key=lambda at: at.info.get("static_reward", 0),
                )

                selected_atoms_list = []

                # Iterate over the lowest energy configurations and relax them
                for i in range(0, max(min_num_relaxed - num_relaxed, 3)):
                    # Take the lowest energy configurations and relax them
                    selected_atoms = surf_population_sorted[i].copy()

                    # centering and pbc necessary for torch sim
                    selected_atoms.center(vacuum=1.5)
                    selected_atoms.pbc = False

                    selected_atoms.info["population_i"] = i
                    selected_atoms.calc = copy.deepcopy(clean_calc)
                    selected_atoms_list.append(selected_atoms)

                if args.use_torch_sim:
                    # relax the atoms in parallel with torch sim
                    relaxed_states = ts.optimize(
                        system=selected_atoms_list,
                        model=mace_model,
                        optimizer=ts.Optimizer.fire,
                        autobatcher=True,
                        convergence_fn=generate_force_convergence_fn(
                            force_tol=args.fmax
                        ),
                        init_kwargs=dict(cell_filter=ts.CellFilter.frechet),
                        pbar=True,
                    )
                    print(f"{relaxed_states = }")

                else:
                    # relax the atoms sequentially with BFGS
                    relaxed_states = []
                    for selected_atoms in selected_atoms_list:
                        optimizer = BFGS(selected_atoms)
                        optimizer.run(fmax=args.fmax)
                        relaxed_states.append(selected_atoms)

                grids = []
                relaxed_trj = []
                for selected_atoms in relaxed_states:
                    # Check if the relaxed atoms are all P-bonded to the metal cluster
                    all_p_bonded = check_all_p_bonded(selected_atoms, args.metal)
                    if not all_p_bonded:
                        continue

                    relaxed_id = uuid.uuid4().hex

                    selected_atoms.info["stage"] = "relaxed"
                    selected_atoms.info["relaxed_id"] = relaxed_id
                    selected_atoms.calc = None
                    relaxed_trj.append(selected_atoms)

                    # Handle grid generation
                    naked_relaxed_atoms = selected_atoms[
                        [
                            atom.index
                            for atom in selected_atoms
                            if atom.symbol == args.metal
                        ]
                    ]
                    print(f"{naked_relaxed_atoms = }")

                    m_grid = Manifold(
                        naked_relaxed_atoms,
                        precision=args.precision,
                        mode="particle",
                        touch_sphere_size=args.touch_sphere_size,
                        calc=copy.deepcopy(clean_calc),
                        wrap_on="sites",
                        use_torch_sim=args.use_torch_sim,
                    )
                    m_grid.run_probe_scan(probes=probes)
                    grid_atoms = m_grid.grid_atoms
                    grid_atoms.info["stage"] = "grid"
                    grid_atoms.info["relaxed_id"] = relaxed_id
                    pop_keys = []
                    for k in grid_atoms.arrays.keys():
                        if "e_" in k and k not in ["e_Cl[P]"]:
                            pop_keys.append(k)
                    for k in pop_keys:
                        _ = grid_atoms.arrays.pop(k)
                    grids.append(grid_atoms)

                # only write to db if there are relaxed atoms
                if len(relaxed_trj) > 0:
                    m.write_to_db(
                        surf_population_sorted + relaxed_trj + grids,
                        db_path=args.db_file,
                    )

                num_relaxed += len(relaxed_trj)


#####################################################################################

if __name__ == "__main__":
    # hyper params
    # to_relax x len(df) x len(dfl) x success_rate ~ total number of configs
    # 10 x 679 x 20 x .5 ~ 70 k
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", type=str)
    parser.add_argument("--precision", type=float, default=1.5)
    parser.add_argument("--touch_sphere_size", type=float, default=3.5)
    parser.add_argument("--population_size", type=int, default=200)
    parser.add_argument("--to_relax", type=int, default=10)
    parser.add_argument("--coverage", type=float, default=0.99)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--prune_rms_thresh", type=float, default=0.01)
    parser.add_argument("--f_conformers", type=int, default=200)
    parser.add_argument("--db_file", type=str, default="db_out.db")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_torch_sim", action="store_true")
    parser.add_argument("--max_retries", type=int, default=5)
    args = parser.parse_args()

    main(args)
