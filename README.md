# Covalent Field Theory (CFT) 🌐🔬

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ASE](https://img.shields.io/badge/integration-ASE-green.svg)](https://wiki.fysik.dtu.dk/ase/)

**CFT (Covalent Field Theory)** is a Python framework designed to move heterogeneous catalysis and surface science beyond the "static-site" paradigm. It provides the computational tools to recast reactive interfaces as continuous, spatially-resolved **Covalent Fields**.

Instead of asking *"Where is the active site?"*, CFT asks *"What is the covalent field?"*. By mapping the interaction potential of chemical probes across a dynamic interface, CFT generates continuous affinity landscapes—enabling the predictive modeling of complex industrial catalysts, nanoparticles, and dynamic surfaces.

---

## 🌟 Key Features

*   **Geometric Manifold Generation**: "Shrinkwrap" 2D manifolds over arbitrary atomistic structures (slabs, nanoparticles, or disordered clusters).
*   **Continuous Affinity Mapping**: Evaluate scalar, spectral, and anisotropic covalent fields using any ASE-compatible calculator (e.g., MACE, DeepMD, DFT).
*   **Probe-Based Scanning**: Automates the orientation and positioning of chemical fragments (probes) using **surrogate-SMILES (*SMILES)**.
*   **Dynamic Surface Support**: Tools for tracking reactive manifolds during Molecular Dynamics (MD) or structural reconstruction.
*   **Advanced Visualization**: Built-in utilities for viewing surface grids, normal vectors ("hedgehog" plots), and energy gradients.

---

## 🧬 Scientific Foundation

CFT is built on the realization that the traditional "active site" concept is a bottleneck for rational catalyst design. The package implements the **Four Postulates of CFT**:

1.  **Anchor Definition**: The probe position is defined by a specific anchor point.
2.  **Deterministic Geometry**: Internal probe coordinates are set with respect to the anchor.
3.  **Local Mapping**: Interactions are mapped directly to the manifold position.
4.  **Field Orthogonality**: Each unique probe (*SMILES) defines one unique covalent field.

---

## 🚀 Installation

### Prerequisites
*   Python >= 3.12
*   **ASE** (Atomic Simulation Environment) must be installed in your environment.
*   **AutoAdsorbate** (Chemical intuition for surface science in a package) provides basic functionality. For more details see: https://github.com/basf/autoadsorbate

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
calc = mace_mp(model="small", device="cpu") #any ASE calulator can be provided

# 1. Setup the surface (Reference Structure)
atoms = fcc111('Cu', size=(4, 4, 3), vacuum=10.0)

# 2. Create the CFT Manifold
# Mode can be 'slab' or 'particle'
manifold = Manifold(atoms, mode='slab', precision=.5, calc=calc, wrap_on='atoms')

# 3. Define a Probe (*SMILES for a Methyl fragment)
probe_smiles = [Fragment("ClC", to_initialize=1)] # Cl atom surves as a surrogate atom in this surrogate-SMILES formula.

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

If you use CFT in your research, please cite:
> *tbd., "title" (2025).*
