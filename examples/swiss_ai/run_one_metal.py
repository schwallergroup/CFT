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
    from ase.db import connect
    from ase.io import read
    import pandas as pd

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

    # clusters
    df = pd.read_json(
        "/cluster/project/krause/frankem/CFT/examples/swiss_ai/data/data.json",
        lines=False,
    )
    df = df.T
    df = df[
        (df["energy_relative"] < 0.1) & df.element_symbol.isin([args.metal])
    ].copy()  # & (df['n_atoms'].astype(int) > 50)

    cluster_trj = get_cluster_traj(df)

    # fragments
    dfl = pd.read_csv(
        "/cluster/project/krause/frankem/CFT/examples/swiss_ai/data/phosphine_ligands_enriched.csv",
        delimiter=",",
    )
    dfl = dfl.dropna()
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
        )

        print(f"{len(m.grid) = }")

        for f in fragments:
            m.make_fragment_population(
                population_size=args.population_size, fragment=f, coverage=args.coverage
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
                m.surf_population, key=lambda at: at.info.get("static_reward", 0)
            )

            relaxed_trj = []
            grids = []
            for i in range(0, int(args.to_relax)):
                selected_atoms = surf_population_sorted[i].copy()
                selected_atoms.info["population_i"] = i
                selected_atoms.calc = copy.deepcopy(clean_calc)
                optimizer = BFGS(selected_atoms)  # , trajectory=traj)
                optimizer.run(fmax=args.fmax)

                relaxed_id = uuid.uuid4().hex

                selected_atoms.info["stage"] = "relaxed"
                selected_atoms.info["relaxed_id"] = relaxed_id
                selected_atoms.calc = None
                relaxed_trj.append(selected_atoms)

                naked_relaxed_atoms = selected_atoms[
                    [atom.index for atom in selected_atoms if atom.symbol == args.metal]
                ]
                print(f"{naked_relaxed_atoms = }")

                m_grid = Manifold(
                    naked_relaxed_atoms,
                    precision=args.precision,
                    mode="particle",
                    touch_sphere_size=args.touch_sphere_size,
                    calc=copy.deepcopy(clean_calc),
                    wrap_on="sites",
                )
                m_grid.run_probe_scan(probes=probes, num_workers=args.num_workers)
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

            m.write_to_db(
                surf_population_sorted + relaxed_trj + grids, db_path=args.db_file
            )


#####################################################################################

if __name__ == "__main__":
    # hyper params
    # to_relax x len(df) x len(dfl) x success_rate ~ total number of configs
    # 10 x 679 x 20 x .5 ~ 70 k
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", type=str)
    parser.add_argument("--precision", type=float, default=1.5)
    parser.add_argument("--touch_sphere_size", type=float, default=3.5)
    parser.add_argument("--population_size", type=int, default=1000)
    parser.add_argument("--to_relax", type=int, default=10)
    parser.add_argument("--coverage", type=float, default=0.99)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--prune_rms_thresh", type=float, default=0.01)
    parser.add_argument("--f_conformers", type=int, default=200)
    parser.add_argument("--db_file", type=str, default="db_out.db")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    main(args)
