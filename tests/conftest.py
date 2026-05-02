"""
Shared pytest fixtures for the CFT test suite.

The `shared_manifold` fixture builds the particle Manifold once per session
and makes it available to all data-dependent tests, avoiding repeated ~6-minute
grid construction runs.
"""
import os
from pathlib import Path

import pytest
from ase.io import read

try:
    from cft import Manifold
except ImportError as e:
    pytest.skip(f"cft not importable: {e}", allow_module_level=True)


@pytest.fixture(scope="session")
def shared_manifold():
    """Build a Manifold once for the entire test session.

    Skips all consuming tests if CFT_DATA_DIR is not set.

    Returns
    -------
    tuple[Manifold, Path]
        ``(m, data_dir)`` — the constructed Manifold and the resolved
        CFT_DATA_DIR path.
    """
    data_dir = os.environ.get("CFT_DATA_DIR")
    if not data_dir:
        pytest.skip(
            "CFT_DATA_DIR not set — skipping (requires pre-computed data files)"
        )
    data_dir = Path(data_dir)

    particle_path = os.environ.get("CFT_PARTICLE", "notebooks/test_particle.xyz")
    particle = read(particle_path)

    m = Manifold(
        particle.copy(),
        mode="particle",
        precision=0.2,
        touch_sphere_size=2.0,
        wrap_on="blend",
    )

    return m, data_dir
