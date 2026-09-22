"""T2.2 acceptance: the C++ host (actions + encoder) equals the Python reference at N decision points.

Usage: uv run python tools/check_cpp_encoder.py [--points 10000]

Plays random games with the Python Duel while a C++ HostDuel follows the same
choices; at every decision point the action lists, all observation arrays and
finally the game results must be identical.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env.encoding import ACTION_KINDS, ObservationEncoder

DECK_DIR = Path(__file__).resolve().parents[1] / "tests" / "decks"


def py_action(a):
    loc = a.card.loc if a.card is not None else None
    return (ACTION_KINDS.index(a.kind), a.index, a.card is not None, a.card.code if a.card else 0,
            (loc.controller, loc.location, loc.sequence, loc.position) if loc else (0, 0, 0, 0), a.description, a.value)  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=10000)
    args = parser.parse_args()
    db = CardDB.load()
    vocab = CardVocab.from_db(db)
    passwords = [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]
    encoder = ObservationEncoder(db, vocab)
    decks = {p.stem: load_ydk(p) for p in sorted(DECK_DIR.glob("*.ydk"))}
    names = sorted(decks)
    points, games, mismatches = 0, 0, []
    by_type = collections.Counter()
    t0 = time.time()
    while points < args.points:
        seed = 900_000 + games
        a, b, first = names[games % 10], names[(games * 3 + 7) % 10], games % 2
        cfg = DuelConfig()
        duel = Duel(seed, None, decks[a], decks[b], cards=db, first=first, config=cfg)
        host = _core.HostDuel(db.to_core(), default_scripts(), passwords)
        host.start(expand_seed(seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
                   [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
        rng = RandomAgent(seed)

        class Lockstep:
            def act(self, point):
                nonlocal points
                ok = host.actions() == [py_action(x) for x in point.actions]
                cpp, py = host.observe(), encoder.encode(point, duel._core)
                ok = ok and all(np.array_equal(cpp[k], py[k]) for k in py)
                if not ok:
                    mismatches.append((games, point.index, point.decision.name))
                points += 1
                by_type[point.decision.name] += 1
                idx = rng.act(point)
                host.act(idx)
                return idx

        agent = Lockstep()
        result = duel.run(agent, agent)
        r = host.result()
        if r["responses"] != result.responses or r["turns"] != result.turns:
            mismatches.append((games, "result", ""))
        games += 1
    summary = {"points": points, "games": games, "mismatches": len(mismatches), "seconds": round(time.time() - t0, 1),
               "decision_types": dict(by_type.most_common())}  # fmt: skip
    print(json.dumps(summary, indent=2))
    for m in mismatches[:10]:
        print("MISMATCH", m, file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
