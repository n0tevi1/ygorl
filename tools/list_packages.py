"""Enumerate engine packages on the synergy graph (T5.4) and print the Top-N.

Usage: uv run python tools/list_packages.py [--top 50] [--json out.json] [--cross-only]

Each package prints its score (reachability density: k-robustly reachable
members per starter), size, starters, archetypes, the cross-archetype flag and
its members (password + name; names are display only). The proxy engine
packages (tests/data/proxy_packages.json) are matched against the Top-N by
overlap coefficient ``|P & R| / min(|P|, |R|)`` as a sanity check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from ygorl.build.packages import DEFAULTS, enumerate_packages, setcodes_from_db
from ygorl.build.synergy_graph import load_or_build
from ygorl.cards.cdb import CardDB

ROOT = Path(__file__).resolve().parents[1]


def _setname_table() -> dict[int, str]:
    from ygorl import paths
    from ygorl.build.lua import read_constants

    src = (paths.card_scripts() / "archetype_setcode_constants.lua").read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for name, value in read_constants(src).items():
        out.setdefault(value, name[len("SET_") :].lower())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--json", type=Path, help="also write the ranked packages as JSON")
    parser.add_argument("--cross-only", action="store_true", help="list only cross-archetype packages")
    for key in ("max_size", "min_size", "k"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=DEFAULTS[key])
    for key in ("min_affinity", "min_share", "max_overlap"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=DEFAULTS[key])
    args = parser.parse_args()

    db = CardDB.load()
    graph = load_or_build()
    t0 = time.perf_counter()
    pkgs = enumerate_packages(
        graph, setcodes=setcodes_from_db(db), max_size=args.max_size, min_size=args.min_size, k=args.k,
        min_affinity=args.min_affinity, min_share=args.min_share, max_overlap=args.max_overlap,
    )  # fmt: skip
    elapsed = time.perf_counter() - t0
    shown = [p for p in pkgs if p.cross_archetype] if args.cross_only else pkgs
    top = shown[: args.top]
    sets = _setname_table()

    def label(p) -> str:
        # name each base setcode after the most common full setcode (sub-archetype) among members
        names = []
        for base in p.archetypes:
            full = [sc for m in p.members for sc in db[m].setcodes if sc & 0xFFF == base]
            code = max(set(full), key=lambda sc: (full.count(sc), -sc)) if full else base
            names.append(sets.get(code, sets.get(base, hex(code))))
        return ", ".join(names) or "-"

    n_cross = sum(p.cross_archetype for p in pkgs)
    print(f"{len(pkgs)} packages ({n_cross} cross-archetype) in {elapsed:.1f}s; top {len(top)}:\n")
    for rank, p in enumerate(top, 1):
        arch = label(p)
        flag = " CROSS" if p.cross_archetype else ""
        print(f"#{rank:<3d} score {p.score:5.2f}  size {len(p):2d}  starters {len(p.starters)}  density {p.density:.2f}  [{arch}]{flag}")
        starters = set(p.starters)
        print("     " + "; ".join(f"{'*' if m in starters else ''}{db[m].name} ({m})" for m in p.members))
    if args.json:
        args.json.write_text(json.dumps([p.to_json() for p in top], indent=1) + "\n")

    spec = importlib.util.spec_from_file_location("make_proxy_packages", ROOT / "tools" / "make_proxy_packages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"\nproxy packages vs. all {len(pkgs)} packages (best overlap coefficient |P & R| / min(|P|, |R|), rank):")
    for name, members in sorted(mod.load_packages().items()):
        ref = {db.canonical(m) for m in members}
        best = max(((len(ref & set(p.members)) / min(len(ref), len(p)), -i) for i, p in enumerate(pkgs, 1)), default=(0.0, 0))
        print(f"  {name:18s} {best[0]:.2f}  (#{-best[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
