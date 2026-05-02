# Tests

Scripts for verifying CFT functionality. Some tests require pre-computed data
files that are not part of this repository (e.g. paper data, MACE model weights).

## Environment variables

| Variable | Description |
|----------|-------------|
| `CFT_DATA_DIR` | Path to directory containing pre-computed field/manifold `.xyz` files (e.g. paper data). Required by `test_minima.py`, `check_pt_sites.py`, `debug_manifold.py`, and the PMD part of `test_attach.py`. |
| `CFT_PARTICLE` | Path to a particle `.xyz` file. Defaults to `notebooks/test_particle.xyz`. |
| `MODEL_PATH` | Full path to a MACE model file (`.model`). Required by any test that initializes a calculator. |

## Running

Tests that have no external data dependency run standalone:

```bash
python tests/test_translate.py
python tests/test_attach.py   # PMD section skipped if CFT_DATA_DIR not set
```

Tests that require paper data:

```bash
export CFT_DATA_DIR=/path/to/cft_paper_data
export CFT_PARTICLE=notebooks/test_particle.xyz
python tests/test_minima.py
python tests/check_pt_sites.py
python tests/debug_manifold.py
```
