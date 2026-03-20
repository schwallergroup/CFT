# Covalent Field Theory (CFT) 🌐🔬

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ASE](https://img.shields.io/badge/integration-ASE-green.svg)](https://wiki.fysik.dtu.dk/ase/)

**This repository contains the code for the paper: [On the Covalent Fields of Molecule–Surface Interactions]**

**CFT (Covalent Field Theory)** is a representation theory and Python framework designed to move heterogeneous catalysis and surface science beyond the "static-site" paradigm. It provides the computational tools to recast chemical affinity as a continuous, spatially-resolved field across reactive interfaces.

Instead of asking *"Where is the active site?"*, CFT asks *"What is the covalent field?"*. By decomposing interfaces into a field-generating reference and chemical probes parameterized via surrogate-SMILES (*SMILES), CFT maps surface energetics onto a two-dimensional manifold at a fraction of stochastic sampling costs.

---

## 🌟 Key Features

*   **Geometric Manifold Generation**: "Shrinkwrap" 2D manifolds over arbitrary atomistic structures (slabs, nanoparticles, or disordered high-entropy alloys).
*   **Continuous Affinity Mapping**: Evaluate scalar, spectral, and anisotropic covalent fields using any ASE-compatible calculator (e.g., MACE, DeepMD, DFT).
*   **Probe-Based Scanning**: Automates the orientation and positioning of chemical fragments (probes) using **surrogate-SMILES (*SMILES)**.
*   **Linear Scaling Relationships (LSR)**: Preserves linear scaling relationships at every point on the manifold, enabling quantitative identification of regions that break traditional limits.
*   **Advanced Visualization**: Built-in utilities for viewing surface grids, normal vectors, and energy gradients.

---

## 🧬 Scientific Foundation

CFT is built on the realization that the traditional "active site" concept lacks a rigorous definition and conflicts with the dynamic reality of working catalysts. The continuous field approach rests on **Four Postulates of CFT**:

1.  **Anchor Definition**: Every probe must have a designated anchor point, specified explicitly in its molecular graph.
2.  **Configuration Mapping**: All configurations of the probe that preserve its molecular graph map to the anchor position.
3.  **Covalent Field**: The value of the covalent field is defined by the interaction energy between a probe and reference as a function of the probe's anchor position.
4.  **Field Validity (Additivity)**: For isolated probes, the total energy is approximately additive. Deviations from additivity quantify inter-probe coupling (many-body effects) such as bond formation.

---

## 🚀 Installation

### Prerequisites
*   Python >= 3.12
*   **ASE** (Atomic Simulation Environment) must be installed.
*   **AutoAdsorbate**: Provides basic functionality for probe generation. See: [https://github.com/basf/autoadsorbate](https://github.com/basf/autoadsorbate)

### From Source
```bash
# Clone the repository
git clone https://github.com/schwallergroup/CFT.git
cd CFT

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install .
```

---

## 💻 Quick Start

```python
from ase.visualize import view
from ase.build import fcc111, add_adsorbate
from cft import Manifold
from autoadsorbate import Fragment

# 0. Setup ASE calculator
from mace.calculators import mace_mp
calc = mace_mp(model="small", device="cpu") # any ASE calculator can be provided

# 1. Setup the surface (Reference Structure)
atoms = fcc111('Cu', size=(4, 4, 3), vacuum=10.0)

# 2. Create the CFT Manifold
# Mode can be 'slab' or 'particle'
manifold = Manifold(atoms, mode='slab', precision=.5, calc=calc, wrap_on='atoms')

# 3. Define a Probe (*SMILES for a Methyl fragment)
probe_smiles = [Fragment("ClC", to_initialize=1)] # Cl atom serves as a surrogate atom in this surrogate-SMILES formula.

# 4. Run a continuous field scan
results = manifold.run_probe_scan(probe_smiles)

# 5. Visualize the "Hedgehog" (Normals) and Field
manifold.save_ply(filename='./test.ply')
manifold.view_hedgehog(show_with_atoms=True)

```

---

## 🏗️ Architecture

CFT follows a modular design for extensibility:

*   **`core.py`**: The `Manifold` class. Extends `autoadsorbate.Surface` to handle grid generation and global field operations.
*   **`dynamics.py`**: Logic for `ProbeScan` (moving fragments across the manifold) and `StaticEval`.
*   **`mesh_utils.py`**: High-performance geometric routines for face orientation, gradient estimation, and PLY I/O.
*   **`examples/`**: Notebooks demonstrating Alatomic Layer Deposition (ALD) on TiN, nanoparticle fragmentation, and scaling relationship analysis.

---

## 🤝 Contributing

We welcome contributions from the surface science and ML communities! Whether it's adding support for new manifold topologies or improving gradient descent algorithms on the covalent field, please feel free to open an issue or a PR.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🔗 Citation

If you use CFT in your research, please cite our paper:

> Fako, E., & Schwaller, P. (2025). On the Covalent Fields of Molecule–Surface Interactions.
