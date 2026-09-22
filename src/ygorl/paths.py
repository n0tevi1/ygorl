"""Locations of bundled third-party data (submodules under third_party/)."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


@cache
def third_party() -> Path:
    """Directory containing ygopro-core, CardScripts, BabelCDB and LFLists.

    Resolution order: ``$YGORL_THIRD_PARTY``; ``third_party/`` next to the source
    tree (editable installs); ``third_party/`` in the current directory or a parent.
    """
    env = os.environ.get("YGORL_THIRD_PARTY")
    if env:
        return Path(env)
    candidates = [Path(__file__).resolve().parents[2] / "third_party"]
    cwd = Path.cwd().resolve()
    candidates += [p / "third_party" for p in (cwd, *cwd.parents)]
    for c in candidates:
        if (c / "BabelCDB").is_dir() and (c / "CardScripts").is_dir():
            return c
    raise FileNotFoundError(
        "third_party/ data not found; run `git submodule update --init --recursive` "
        "or set YGORL_THIRD_PARTY"
    )


def cards_cdb() -> Path:
    return third_party() / "BabelCDB" / "cards.cdb"


def card_scripts() -> Path:
    return third_party() / "CardScripts"


def script_directories() -> list[Path]:
    """Script search path in priority order (first match wins), as EDOPro uses it."""
    root = card_scripts()
    subdirs = ["official", "pre-release", "pre-errata", "goat", "rush", "skill", "unofficial"]
    return [root, *(root / s for s in subdirs)]


def lflists() -> Path:
    return third_party() / "LFLists"
