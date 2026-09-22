"""T2.4 cross-check: the C++ event token stream equals the Python reference at N decision points.

Usage: uv run python tools/check_cpp_events.py [--points 10000] [--length 128]

Plays random games with the Python Duel while a C++ HostDuel (with an event
stream of ``--length`` tokens) follows the same choices; at every decision point
``events`` / ``event_mask`` must be identical to ``ygorl.env.events.EventHistory``.
Also reports how many tokens of each type and abstain trigger were seen.
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
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env.events import EVENT_TYPES, TRIGGERS, EventHistory

DECK_DIR = Path(__file__).resolve().parents[1] / "tests" / "decks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--length", type=int, default=128)
    args = parser.parse_args()
    db = CardDB.load()
    vocab = CardVocab.from_db(db)
    passwords = [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]
    decks = {p.stem: load_ydk(p) for p in sorted(DECK_DIR.glob("*.ydk"))}
    names = sorted(decks)
    points, games, mismatches, tokens = 0, 0, [], 0
    by_type, by_trigger = collections.Counter(), collections.Counter()
    t0 = time.time()
    while points < args.points:
        seed = 700_000 + games
        a, b, first = names[games % 10], names[(games * 3 + 7) % 10], games % 2
        cfg = DuelConfig()
        duel = Duel(seed, None, decks[a], decks[b], cards=db, first=first, config=cfg)
        host = _core.HostDuel(db.to_core(), default_scripts(), passwords, event_length=args.length)
        host.start(expand_seed(seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
                   [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
        history = EventHistory(db, vocab, length=args.length)
        full = EventHistory(db, vocab, length=10**7)  # whole game, for the statistics
        rng = RandomAgent(seed)

        class Lockstep:
            def act(self, point):
                nonlocal points
                history.feed(point.events)
                full.feed(point.events)
                cpp, py = host.observe(), history.encode(point.player)
                if not all(np.array_equal(cpp[k], py[k]) for k in py):
                    mismatches.append((games, point.index, point.decision.name))
                points += 1
                idx = rng.act(point)
                host.act(idx)
                return idx

        agent = Lockstep()
        duel.run(agent, agent)
        for row in full.tokens[0]:
            tokens += 1
            by_type[EVENT_TYPES[row[0] - 1]] += 1
            if EVENT_TYPES[row[0] - 1] == "abstain":
                for name, bit in TRIGGERS.items():
                    if row[12] & bit:
                        by_trigger[name] += 1
        games += 1
    summary = {"points": points, "games": games, "mismatches": len(mismatches), "seconds": round(time.time() - t0, 1),
               "tokens_viewer0": tokens, "abstain_triggers": dict(by_trigger.most_common()),
               "token_types": dict(by_type.most_common())}  # fmt: skip
    print(json.dumps(summary, indent=2))
    for m in mismatches[:10]:
        print("MISMATCH", m, file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
