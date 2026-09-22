"""Derive *proxy* engine packages from the test-deck specs for the synergy-graph recall check.

Usage: uv run python tools/make_proxy_packages.py [--out tests/data/proxy_packages.json]

Real meta engine packages (masterduelmeta / YGOPRODECK deck lists) arrive with
T5.1; until then the recall of the synergy graph (T5.3, acceptance "recall >= 80%
of meta engine packages") is measured on proxies derived from the archetype
cores in ``tools/make_test_decks.py`` ``DECKS``:

* every spec card of a deck belongs to that deck's package, except the generic
  tech listed in ``GENERIC`` (hand traps and staples are already outside ``DECKS``;
  these are generic traps and generic Extra Deck monsters any deck could play);
* engines that are played across decks get their own package (``SPLIT``):
  the Fiendsmith cards of the Snake-Eye and Fiendsmith-Ryzeal decks form one
  "fiendsmith" package, the Ryzeal cards a "ryzeal" package.

Names are resolved to passwords through cards.cdb (like make_test_decks.py);
the output lists passwords (the key) with names for readability only. The
packages are an *evaluation* set: never use them to build the graph.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from ygorl.cards.cdb import CardDB

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("make_test_decks", ROOT / "tools" / "make_test_decks.py")
make_test_decks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_test_decks)

GENERIC = {
    # generic Normal Traps and trap support played by Labrynth
    "Transaction Rollback", "Dogmatika Punishment", "Destructive Daruma Karma Cannon", "Dimensional Barrier",
    "Torrential Tribute", "Compulsory Evacuation Device", "Trap Trick",
    # generic Extra Deck monsters
    "Muckraker From the Underworld", "Chaos Angel", "Crystal Wing Synchro Dragon", "Stardust Dragon",
    "Black Rose Dragon", "Mudragon of the Swamp", "El Shaddoll Construct", "Guardian Chimera",
}  # fmt: skip

SPLIT = {
    "fiendsmith": {
        "Fiendsmith Engraver", "Fiendsmith in Paradise", "Fiendsmith's Tract", "Fiendsmith's Sanct", "Fiendsmith Kyrie",
        "Lacrima the Crimson Tears", "Fiendsmith's Requiem", "Fiendsmith's Sequence", "Fiendsmith's Lacrima",
        "Fiendsmith's Desirae", "Fiendsmith's Agnumday", "Necroquip Princess",
    },
    "ryzeal": {"Ice Ryzeal", "Sword Ryzeal", "Node Ryzeal", "Star Ryzeal", "Ryzeal Cross", "Ryzeal Duo Drive", "Ext Ryzeal", "Ryzeal Detonator"},
}  # fmt: skip


def derive(db: CardDB) -> dict:
    names = make_test_decks.by_name(db)
    packages: dict[str, dict[int, str]] = {}
    excluded: dict[int, str] = {}
    for deck, spec in make_test_decks.DECKS.items():
        for name, _count in spec:
            pw = names[name]
            if name in GENERIC:
                excluded[pw] = name
                continue
            target = next((pkg for pkg, members in SPLIT.items() if name in members), deck)
            packages.setdefault(target, {})[pw] = name
    return {
        "description": "Proxy engine packages derived from tools/make_test_decks.py DECKS (see tools/make_proxy_packages.py); "
        "an evaluation set for the synergy graph until T5.1 provides real meta deck lists.",
        "packages": {pkg: [[pw, members[pw]] for pw in sorted(members)] for pkg, members in sorted(packages.items())},
        "excluded_generic": [[pw, excluded[pw]] for pw in sorted(excluded)],
    }


def dumps(data: dict) -> str:
    """JSON with one ``[password, name]`` member per line (reviewable diffs)."""

    def members(rows: list, indent: str) -> str:
        return "[\n" + ",\n".join(f"{indent} " + json.dumps(r, ensure_ascii=False) for r in rows) + f"\n{indent}]"

    pkgs = ",\n".join(f"  {json.dumps(k)}: {members(v, '  ')}" for k, v in data["packages"].items())
    return (
        "{\n"
        f' "description": {json.dumps(data["description"], ensure_ascii=False)},\n'
        f' "packages": {{\n{pkgs}\n }},\n'
        f' "excluded_generic": {members(data["excluded_generic"], " ")}\n'
        "}\n"
    )


def load_packages(path: str | Path = ROOT / "tests" / "data" / "proxy_packages.json") -> dict[str, list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: [pw for pw, _name in members] for name, members in data["packages"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "data" / "proxy_packages.json")
    args = parser.parse_args()
    data = derive(CardDB.load())
    args.out.write_text(dumps(data), encoding="utf-8")
    sizes = {k: len(v) for k, v in data["packages"].items()}
    print(f"wrote {len(sizes)} packages to {args.out}: {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
