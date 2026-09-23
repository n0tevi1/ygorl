"""Derive engine packages from an environment's meta decks for the synergy-graph recall check (T5.3).

Usage: uv run python tools/make_meta_packages.py [md-2026-09] [--out PATH] [--per-deck] [--min-size 3]

Writes ``environments/<version>/artifacts/meta_packages.json`` (stamped with
``Environment.stamp()``). The packages are an *evaluation* set: the meta deck
lists never enter the synergy graph. The rule is fixed and documented in
docs/synergy.md; nothing is tuned per deck:

1. Deck cards = Main + Extra Deck of each meta ``.ydk`` (passwords are already
   canonical in an environment; the Side Deck is ignored).
2. Known generic cards are removed everywhere: ``tests/data/generic_pool.json``
   (hand traps, board breakers, staples, generic Extra Deck monsters) and the
   ``GENERIC`` list of tools/make_proxy_packages.py -- both curated before T5.1
   and independent of these decks.
3. Shared engines are split off like the proxy script did: cards are grouped by
   their *signature*, the set of meta decks that play them. A group whose
   signature has >= 2 decks is a candidate shared engine (e.g. the Branded cards
   of Branded and Dracotail); a single-deck group is that deck's own candidate.
4. Each candidate group is split into the connected components of a *declared
   relation* taken from the card database only (never from the synergy graph or
   from Yugipedia): two cards are related when they share a base setcode
   (``setcode & 0xfff``), when one card's text quotes the other's name or a
   string contained in it ("Dracotail" -> "Dracotail Faimena"), or when both
   texts quote the same string (both mention "Temple of the Kings"). Quotes of a
   card's own name ("You can only use this effect of "X" once per turn") are
   ignored.
5. Components with >= ``--min-size`` (3) cards are packages. The other cards are
   generic for this purpose: cards played in >= 2 decks outside a shared engine
   (``shared``: Maxx "C", Fuwalos, S:P Little Knight, ...), and single-deck cards
   that belong to no archetype of the deck and name nothing in it
   (``unrelated``: Pot of Desires, Garura, Solemn Strike, ...).

``--per-deck`` skips the split of step 4 (the same members, one package per
signature group) -- the stricter variant that also asks the graph to link two
engines played side by side (Vanquish Soul + K9).

Cards are keyed by password; names are comments for review only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from ygorl.build.scripts import load_constants
from ygorl.cards.cdb import CardDB
from ygorl.data.environment import Environment, load_environment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = "md-2026-09"
ARTIFACT = "meta_packages.json"
MIN_SIZE = 3
GENERIC_POOL = ROOT / "tests" / "data" / "generic_pool.json"

_QUOTE = re.compile(r'"([^"\n]+)"')


def known_generic(db: CardDB) -> set[int]:
    """Curated generic cards (generic_pool.json + make_proxy_packages.GENERIC), canonical passwords."""
    spec = importlib.util.spec_from_file_location("make_proxy_packages", ROOT / "tools" / "make_proxy_packages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pool = {c["password"] for c in json.loads(GENERIC_POOL.read_text(encoding="utf-8"))["cards"]}
    return {db.canonical(p) if p in db else p for p in pool | mod.GENERIC}


class Relation:
    """The declared relation of step 4 (card database only)."""

    def __init__(self, db: CardDB) -> None:
        self.db = db

    def _bases(self, p: int) -> set[int]:
        return {s & 0xFFF for s in self.db[p].setcodes}

    def _quotes(self, p: int) -> set[str]:
        card = self.db[p]
        return {q for q in _QUOTE.findall(card.desc) if q != card.name}

    def __call__(self, a: int, b: int) -> bool:
        if self._bases(a) & self._bases(b):
            return True
        qa, qb = self._quotes(a), self._quotes(b)
        if qa & qb:
            return True
        na, nb = self.db[a].name, self.db[b].name
        return any(q in nb for q in qa) or any(q in na for q in qb)

    def components(self, cards: Iterable[int]) -> list[list[int]]:
        cards = sorted(cards)
        parent = {p: p for p in cards}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(cards):
            for b in cards[i + 1 :]:
                if find(a) != find(b) and self(a, b):
                    parent[find(a)] = find(b)
        groups: dict[int, list[int]] = defaultdict(list)
        for p in cards:
            groups[find(p)].append(p)
        return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _set_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for name, code in sorted(load_constants().items()):
        if name.startswith("SET_") and (code not in names or len(name) < len(names[code])):
            names[code] = name
    return names


def _label(db: CardDB, members: list[int], set_names: Mapping[int, str], fallback: str) -> str:
    bases = Counter(s & 0xFFF for p in members for s in {s & 0xFFF for s in db[p].setcodes})
    if bases:
        code, count = max(bases.items(), key=lambda kv: (kv[1], -kv[0]))
        if count >= 2 and code in set_names:
            return _slug(set_names[code][4:])
    return _slug(fallback)


def meta_decks(env: Environment) -> dict[str, set[int]]:
    return {md.name: set(md.deck.main) | set(md.deck.extra) for md in env.meta_decks}


def derive(env: Environment, db: CardDB, *, min_size: int = MIN_SIZE, split: bool = True) -> dict:
    decks = meta_decks(env)
    generic = known_generic(db)
    relation = Relation(db)
    set_names = _set_names()

    signature: dict[int, frozenset[str]] = {}
    for p in sorted(set().union(*decks.values())):
        signature[p] = frozenset(name for name, cards in decks.items() if p in cards)
    groups: dict[frozenset[str], set[int]] = defaultdict(set)
    excluded: dict[int, str] = {}
    for p, sig in signature.items():
        if p in generic:
            excluded[p] = "generic_list"
        else:
            groups[sig].add(p)

    order = [e.name for e in env.meta_decks]
    packages: dict[str, dict] = {}
    # single-deck groups first (in meta.json order), then shared engines by number of decks
    for sig in sorted(groups, key=lambda s: (len(s) > 1, len(s), sorted(order.index(d) for d in s))):
        comps = relation.components(groups[sig])
        kept = [c for c in comps if len(c) >= min_size]
        for comp in comps:
            if len(comp) < min_size:
                for p in comp:
                    excluded[p] = "shared" if len(sig) >= 2 else "unrelated"
        if not split and kept:
            kept = [sorted(p for c in kept for p in c)]
        deck_names = sorted(sig, key=order.index)
        for i, comp in enumerate(kept):
            if len(sig) == 1 and len(kept) == 1:
                key = _slug(deck_names[0])  # a deck's only own package is named after the deck
            elif not split:
                key = _slug(" + ".join(deck_names))
            else:
                key = _label(db, comp, set_names, fallback=f"{deck_names[0]} {i + 1}")
                if len(sig) > 1 and key in packages:
                    key += "-shared"
            base, n = key, 2
            while key in packages:
                key, n = f"{base}-{n}", n + 1
            packages[key] = {"decks": deck_names, "members": comp}

    def rows(ps: Iterable[int]) -> list[list]:
        return [[p, db[p].name] for p in sorted(ps)]

    return {
        "description": "Engine packages derived from the environment's meta decks by tools/make_meta_packages.py "
        "(rule in its docstring and docs/synergy.md); an evaluation set for the synergy graph, never used to build it.",
        "environment": env.stamp(),
        "rule": {"min_size": min_size, "split_by_declared_relation": split,
                 "generic_lists": ["tests/data/generic_pool.json", "tools/make_proxy_packages.py:GENERIC"]},  # fmt: skip
        "packages": {k: {"decks": v["decks"], "members": rows(v["members"])} for k, v in sorted(packages.items())},
        "excluded": {reason: rows(p for p, r in excluded.items() if r == reason) for reason in ("generic_list", "shared", "unrelated")},
    }


def dumps(data: dict) -> str:
    """JSON with one ``[password, name]`` member per line (reviewable diffs)."""

    def members(rows: list, indent: str) -> str:
        if not rows:
            return "[]"
        return "[\n" + ",\n".join(f"{indent} " + json.dumps(r, ensure_ascii=False) for r in rows) + f"\n{indent}]"

    pkgs = ",\n".join(
        f'  {json.dumps(k)}: {{"decks": {json.dumps(v["decks"], ensure_ascii=False)}, "members": {members(v["members"], "   ")}}}'
        for k, v in data["packages"].items()
    )
    excl = ",\n".join(f"  {json.dumps(k)}: {members(v, '  ')}" for k, v in data["excluded"].items())
    return (
        "{\n"
        f' "description": {json.dumps(data["description"], ensure_ascii=False)},\n'
        f' "environment": {json.dumps(data["environment"])},\n'
        f' "rule": {json.dumps(data["rule"])},\n'
        f' "packages": {{\n{pkgs}\n }},\n'
        f' "excluded": {{\n{excl}\n }}\n'
        "}\n"
    )


def load_packages(path: str | Path) -> dict[str, list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: [pw for pw, _name in pkg["members"]] for name, pkg in data["packages"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("environment", nargs="?", default=DEFAULT_ENVIRONMENT, help="environment version or directory")
    parser.add_argument("--out", type=Path, help=f"default: <environment>/artifacts/{ARTIFACT}")
    parser.add_argument("--min-size", type=int, default=MIN_SIZE)
    parser.add_argument("--per-deck", action="store_true", help="do not split a signature group by declared relation")
    args = parser.parse_args()
    env = load_environment(args.environment)
    data = derive(env, CardDB.load(), min_size=args.min_size, split=not args.per_deck)
    out = args.out or env.artifact_path(ARTIFACT)
    Path(out).write_text(dumps(data), encoding="utf-8")
    sizes = {k: len(v["members"]) for k, v in data["packages"].items()}
    excl = {k: len(v) for k, v in data["excluded"].items()}
    print(f"wrote {len(sizes)} packages to {out}: {sizes}\nexcluded: {excl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
