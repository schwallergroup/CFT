# Tests

Scripts for verifying CFT functionality. Some tests require pre-computed data
files that are not part of this repository (e.g. paper data, MACE model weights).

## Environment variables

| Variable | Description |
|----------|-------------|
| `CFT_DATA_DIR` | Path to directory containing pre-computed field/manifold `.xyz` files (e.g. paper data). Required by `test_minima.py`, `test_pt_sites.py`, and `test_debug_manifold.py`. |
| `CFT_PARTICLE` | Path to a particle `.xyz` file. Defaults to `notebooks/test_particle.xyz`. |
| `MODEL_PATH` | Full path to a MACE model file (`.model`). Required by any test that initializes a calculator. |

## Running

All tests are pytest-discoverable. Run the full suite with:

```bash
pytest tests/ -v
```

Tests that require `CFT_DATA_DIR` will be **skipped** automatically if the
variable is not set. To include them:

```bash
export CFT_DATA_DIR=/path/to/cft_paper_data
export CFT_PARTICLE=notebooks/test_particle.xyz  # optional, this is the default
pytest tests/ -v
```

## Test inventory

| File | Needs `CFT_DATA_DIR` | Needs `MODEL_PATH` | Description |
|---|---|---|---|
| `test_translate.py` | No | No | CO fragment anchor translation |
| `test_attach.py` | Partial (PMD section) | No | Fragment placement at grid vertex |
| `test_minima.py` | Yes | No | Manifold minima consistency check |
| `test_pt_sites.py` | Yes | No | PMD minima on Pt-associated vertices |
| `test_debug_manifold.py` | Yes | No | Grid reproducibility vs pre-computed field |
