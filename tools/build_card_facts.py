"""Card facts for the policy network: archetypes, referenced archetypes, effect categories, script queries.

  uv run --no-sync python tools/build_card_facts.py OUT_DIR [--workers 8]

Writes ``OUT_DIR/card_facts.npz`` (layout in ``ygorl.nets.text``, docs/nets.md「卡片事实」), keyed by password, for
every card of the database: its archetypes (cdb setcode, archetype part), the archetypes its scripts refer to
(``IsSetCard`` in search / summon filters, listed series, material archetypes), the cdb effect-category bits and
the (action, location) pairs its scripts take cards from (``ygorl.build.scripts``). Put it in the same directory
as the text tables: the network loads both with ``TextFeatures.load``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LOCATIONS = {0x1: "deck", 0x2: "hand", 0x4: "mzone", 0x8: "szone", 0x10: "grave", 0x20: "banished", 0x40: "extra"}
MAX_REFERENCES = 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-query-cards", type=int, default=20, help="keep (action, location) pairs this common")
    args = ap.parse_args()

    from ygorl import paths
    from ygorl.build.filters import Filter, Not, Pred
    from ygorl.build.synergy_graph import analyze_scripts
    from ygorl.engine.duel import default_cards

    def setcards(f: Filter, out: set[int]) -> None:
        """Archetype parts (low 12 bits) of the ``setcard`` predicates in a filter tree, outside negations."""
        if isinstance(f, Pred):
            if f.kind == "setcard":
                out.update(int(a) & 0x0FFF for a in f.args if isinstance(a, int) and a)
        elif isinstance(f, Not):
            pass  # "not archetype X" excludes X: it is no reference to it
        else:  # And / Or
            for item in f.items:
                setcards(item, out)

    cards = default_cards()
    facts = {}
    for d in reversed(paths.script_directories()):  # later directories first so the first match wins
        facts.update(analyze_scripts(d, workers=args.workers))
    pws = sorted(cards.keys())
    setcodes = np.zeros((len(pws), 4), np.int64)
    refs = np.zeros((len(pws), MAX_REFERENCES), np.int64)
    cats = np.zeros(len(pws), np.uint64)
    query_sets: list[set[str]] = []
    counts: dict[str, int] = {}
    for k, pw in enumerate(pws):
        c = cards[pw]
        own = [s for s in c.setcodes if s][:4]
        setcodes[k, : len(own)] = own
        cats[k] = np.uint64(c.category & 0xFFFFFFFFFFFFFFFF)
        f = facts.get(pw)
        r: set[int] = set()
        q: set[str] = set()
        if f is not None:
            r.update(int(s) & 0x0FFF for s in (*f.listed_series, *f.material_setcodes) if s)
            for x in f.queries:
                setcards(x.filter, r)
                for bit, name in LOCATIONS.items():
                    if x.locations & bit:
                        q.add(f"{x.action}:{name}")
        r -= {0}
        chosen = sorted(r)[:MAX_REFERENCES]
        refs[k, : len(chosen)] = chosen
        query_sets.append(q)
        for x in q:
            counts[x] = counts.get(x, 0) + 1
    names = sorted(x for x, n in counts.items() if n >= args.min_query_cards)
    col = {x: i for i, x in enumerate(names)}
    queries = np.zeros((len(pws), len(names)), np.uint8)
    for k, q in enumerate(query_sets):
        for x in q:
            if x in col:
                queries[k, col[x]] = 1
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "card_facts.npz", passwords=np.asarray(pws, np.int64), setcodes=setcodes,
                        references=refs, categories=cats, queries=queries, query_names=np.asarray(names))  # fmt: skip
    summary = {"cards": len(pws), "scripts": len(facts), "with_archetype": int((setcodes > 0).any(1).sum()),
               "with_references": int((refs > 0).any(1).sum()), "queries": len(names),
               "with_queries": int(queries.any(1).sum())}  # fmt: skip
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
