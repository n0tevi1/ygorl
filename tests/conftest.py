"""Shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def real_graph():
    """The synergy graph mined from CardScripts ``official/`` (built once, cached on disk)."""
    from ygorl.build.synergy_graph import load_or_build

    return load_or_build()
