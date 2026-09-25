"""Replay-determinism sweep (T1.6).

Usage: uv run python tools/check_determinism.py [--games 500] [--workers N]

Each game is played by random agents, then replayed from its response log in
a fresh Duel within the same worker process (so heap addresses differ). The
two raw message streams must be byte-identical and the outcomes equal.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "tests" / "decks"


def check(game: int):
    from ygorl.agents import RandomAgent
    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import Duel

    names = sorted(p.stem for p in DECK_DIR.glob("*.ydk"))
    a, b = names[game % len(names)], names[(game * 7 + 3) % len(names)]
    if a == b:
        b = names[(game + 1) % len(names)]
    da, db = load_ydk(DECK_DIR / f"{a}.ydk"), load_ydk(DECK_DIR / f"{b}.ydk")
    o = Duel(game, None, da, db, first=game % 2, record_messages=True).run(RandomAgent(game), RandomAgent(~game))
    r = Duel(game, None, da, db, first=game % 2, record_messages=True).replay(o.responses)
    same_stream = o.message_log == r.message_log
    same_outcome = (o.winner, o.reason, o.turns, o.lp) == (r.winner, r.reason, r.turns, r.lp)
    first_diff = None
    if not same_stream:
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(o.message_log, r.message_log)) if x != y),
            min(len(o.message_log), len(r.message_log)),
        )
    return {
        "game": game,
        "a": a,
        "b": b,
        "same_stream": same_stream,
        "same_outcome": same_outcome,
        "first_diff": first_diff,
        "buffers": len(o.message_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    with mp.Pool(args.workers) as pool:
        results = pool.map(check, range(args.games), chunksize=2)
    bad = [r for r in results if not (r["same_stream"] and r["same_outcome"])]
    print(json.dumps({"games": len(results), "stream_mismatches": sum(not r["same_stream"] for r in results),
                      "outcome_mismatches": sum(not r["same_outcome"] for r in results)}, indent=2))  # fmt: skip
    for r in bad[:10]:
        print("MISMATCH", r, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
