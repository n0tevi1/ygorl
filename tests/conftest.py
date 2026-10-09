"""Shared fixtures."""

import os
import sys
from pathlib import Path

# Tool regression modules must collect independently, including CI without torch.
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

# Small linear-algebra problems (the surrogate's ridge fits) run orders of magnitude slower when
# OpenBLAS spreads them over threads on a loaded machine; the suite never needs threaded BLAS.
# Set before anything imports numpy.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def real_graph():
    """The synergy graph mined from CardScripts ``official/`` (built once, cached on disk)."""
    from ygorl.build.synergy_graph import load_or_build

    return load_or_build()
