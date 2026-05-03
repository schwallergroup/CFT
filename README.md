# Covalent Field Theory (CFT) 🌐🔬

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ASE](https://img.shields.io/badge/integration-ASE-green.svg)](https://wiki.fysik.dtu.dk/ase/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19980912.svg)](https://doi.org/10.5281/zenodo.19980912)

**This repository contains the code for the paper: [On the Covalent Fields of Molecule–Surface Interactions](https://github.com/schwallergroup/CFT)**

**CFT (Covalent Field Theory)** is a representation theory and Python framework designed to move heterogeneous catalysis and surface science beyond the "static-site" paradigm. It provides the computational tools to recast chemical affinity as a continuous, spatially-resolved field across reactive interfaces.

Instead of asking *"Where is the active site?"*, CFT asks *"What is the covalent field?"*. By decomposing interfaces into a field-generating reference and chemical probes parameterized via surrogate-SMILES (*SMILES), CFT maps surface energetics onto a two-dimensional manifold at a fraction of stochastic sampling costs.

---

## 🌟 Key Features

*   **Geometric Manifold Generation**: "Shrinkwrap" 2D manifolds over arbitrary atomistic structures (slabs, nanoparticles, or disordered high-entropy alloys).
*   **Continuous Affinity Mapping**: Evaluate scalar, spectral, and anisotropic covalent fields using any ASE-compatible calculator (e.g., MACE, DeepMD, DFT).
*   **Probe-Based Scanning**: Automates the orientation and positioning of chemical fragments (probes) using **surrogate-SMILES (*SMILES)**.
*   **Linear Scaling Relationships (LSR)**: Preserves linear scaling relationships at every point on the manifold, enabling quantitative identification of regions that break traditional limits. The same manifold representation algebraically recovers Brønsted–Evans–Polanyi (BEP) correlations, connecting barrier heights to reaction energies without explicit transition-state calculations.
*   **Transition-State Estimation via PMD**: The Point of Maximal Deviation locates configurations of maximal probe–probe coupling on the manifold — a path-independent, geometry-based surrogate for the transition state that avoids explicit NEB or TS optimization.
*   **Advanced Visualization**: Built-in utilities for viewing surface grids, normal vectors, and energy gradients.

---

## 🧬 Scientific Foundation

CFT is built on the realization that the traditional "active site" concept lacks a rigorous definition and conflicts with the dynamic reality of working catalysts. The continuous field approach rests on **Four Postulates of CFT**:

1.  **Anchor Definition**: Every probe must have a designated anchor point, specified explicitly in its molecular graph.
2.  **Configuration Mapping**: All configurations of the probe that preserve its molecular graph map to the anchor position.
3.  **Covalent Field**: The value of the covalent field is defined by the interaction energy between a probe and reference as a function of the probe's anchor position.
4.  **Field Validity (Additivity)**: For isolated probes, the total energy is approximately additive. Deviations from additivity quantify inter-probe coupling (many-body effects) such as bond formation.

---

## Getting Started

### 1. Clone and install

> Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is already installed

git clone https://github.com/schwallergroup/CFT.git
cd CFT
uv sync       # creates .venv and installs all dependencies
```

*Fallback (plain pip):*
```bash
python -m venv .venv && source .venv/bin/activate
pip install .
```

### 2. Download the MACE model

```bash
mkdir -p models
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model \
     -O models/mace-mh-1.model

export MODEL_PATH=$PWD/models/mace-mh-1.model
```

### 3. Download the paper data *(optional: needed for full tests and notebooks)*

The dataset is archived on Zenodo at
[doi.org/10.5281/zenodo.19650620](https://doi.org/10.5281/zenodo.19650620).
Files are **access-restricted**: log in to Zenodo and click **Request access**
on the record page. Once approved, download `cft_paper.zip` and extract it:

```bash
unzip cft_paper.zip -d /path/to/data/

export CFT_DATA_DIR=/path/to/data/cft_paper
export CFT_PARTICLE=$CFT_DATA_DIR/test_particle.xyz
export CFT_SCRATCH_DIR=./outputs   # output dir for meshes, PLY files, figures
```

### 4. Verify the installation

```bash
uv run pytest tests/ -v
```

Without `CFT_DATA_DIR` (install-only check):

```
3 passed, 3 skipped
```

With `CFT_DATA_DIR` set (full suite):

```
6 passed
```

---

## 📓 Notebook

[`notebooks/cft.ipynb`](notebooks/cft.ipynb) is the primary reference for reproducing all paper figures and exploring the full CFT workflow. It covers:

- Manifold fitting and visualisation
- Probe fragment setup (*SMILES)
- Continuous field scans and energy maps
- NEB energy decomposition on the manifold
- Linear scaling relations (LSR) and LSR-breaking maps
- Descriptor maps across Cu / Pd / Au / HEO surfaces
- Point of Maximal Deviation (PMD) and anisotropy estimates
- BEP benchmark

Set `CFT_DATA_DIR`, `MODEL_PATH`, and `CFT_SCRATCH_DIR` (see step 3 above), then run the notebook top-to-bottom. Each section is self-contained.

---

## 💻 Quick Start

```python
from ase.visualize import view
from ase.io import write
from ase.build import fcc111, add_adsorbate
from cft import Manifold
from cft.mesh_utils import values_to_colors
from autoadsorbate import Fragment

# 0. Setup ASE calculator
from mace.calculators import mace_mp
calc = mace_mp(model="small", device="cpu") # any ASE calculator can be provided

# 1. Setup the surface (Reference Structure)
atoms = fcc111('Cu', size=(4, 4, 3), vacuum=10.0)

# 2. Create the CFT Manifold
# Mode can be 'slab' or 'particle'
manifold = Manifold(atoms, mode='slab', precision=.3, calc=calc, wrap_on='atoms')

# 3. Define a Probe (*SMILES for a Methyl fragment)
probes = [Fragment("ClC", to_initialize=1)] # Cl atom serves as a surrogate atom in this surrogate-SMILES formula.

# 4. Run a continuous field scan
manifold.run_probe_scan(probes)
vals = manifold.grid_atoms.arrays[f'e_{probes[0].smile}']
colors = [values_to_colors(v, [vals.min(), vals.max()], palette_nam="plasma") for v in vals.flatten()]

# 5. Visualize the "Hedgehog" (Normals) and Field
manifold.save_ply(filename='./test.ply', vertex_colors=colors)
manifold.view_hedgehog(show_with_atoms=True)
write('./test_grid_atoms.xyz', manifold.grid_atoms) # covalent field evals stored in atoms.arrays, visualize with e.g. Ovito
```

---

## 🏗️ Architecture

CFT follows a modular design for extensibility:

*   **`core.py`**: The `Manifold` class. Extends `autoadsorbate.Surface` to handle grid generation and global field operations.
*   **`dynamics.py`**: Logic for `ProbeScan` (moving fragments across the manifold) and `StaticEval`.
*   **`mesh_utils.py`**: High-performance geometric routines for face orientation, gradient estimation, and PLY I/O.
*   **`examples/CO_neb/`**: End-to-end NEB calculation with CFT probe scanning on the NEB plane mesh.

---

## 🤝 Contributing

We welcome contributions from the surface science and ML communities! Whether it's adding support for new manifold topologies or improving gradient descent algorithms on the covalent field, please feel free to open an issue or a PR.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🔗 Citation

If you use CFT in your research, please cite our papers:

```bibtex
@unpublished{fako2026cft,
  author  = {Fako, Edvin and Schwaller, Philippe},
  title   = {On the Covalent Fields of 
             Molecule--Surface Interactions},
  year    = {2026},
  note    = {Manuscript submitted for publication},
}
```
```bibtex
@article{Leemans2026,
  title = {Ligand-Modulated Release of Copper Active Sites Extends Ethylene Production in CO
                    <sub>2</sub>
                    Electroreduction},
  volume = {148},
  ISSN = {1520-5126},
  url = {http://dx.doi.org/10.1021/jacs.5c22701},
  DOI = {10.1021/jacs.5c22701},
  number = {12},
  journal = {Journal of the American Chemical Society},
  publisher = {American Chemical Society (ACS)},
  author = {Leemans,  Jari and Fako,  Edvin and Zaza,  Ludovic and Tritschler,  Moritz and Chen,  Junwu and Schwaller,  Philippe and Buonsanti,  Raffaella},
  year = {2026},
  month = Mar,
  pages = {13118–13127}
}
```
