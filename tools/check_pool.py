"""T2.1 acceptance: games on the multi-threaded DuelPool equal sequential Duel.run, game by game.

Usage: uv run python tools/check_pool.py [--games 1000] [--threads 4] [--envs 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel
from ygorl.env import GameSpec, run_games

DECK_DIR = Path(__file__).resolve().parents[1] / "tests" / "decks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--envs", type=int, default=32)
    args = parser.parse_args()

    decks = {p.stem: load_ydk(p) for p in sorted(DECK_DIR.glob("*.ydk"))}
    names = sorted(decks)
    specs = [GameSpec(seed=500_000 + i, deck_a=decks[names[i % 10]], deck_b=decks[names[(i * 7 + 3) % 10]], first=i % 2)
             for i in range(args.games)]  # fmt: skip

    def agents(i, spec):
        return RandomAgent(spec.seed), RandomAgent(spec.seed ^ 0xABC)

    t0 = time.time()
    seq = []
    for i, s in enumerate(specs):
        a, b = agents(i, s)
        seq.append(Duel(s.seed, None, s.deck_a, s.deck_b, first=s.first, config=s.config).run(a, b))
    t_seq = time.time() - t0

    t0 = time.time()
    pooled = run_games(specs, agents, num_envs=args.envs, num_threads=args.threads)
    t_pool = time.time() - t0

    def key(r):
        return (r.winner, r.reason, r.win_reason, r.turns, r.lp, r.decisions, r.responses)

    mismatches = [i for i, (x, y) in enumerate(zip(seq, pooled)) if key(x) != key(y)]
    decisions = sum(r.decisions for r in seq)
    print(json.dumps({"games": len(specs), "threads": args.threads, "envs": args.envs, "mismatches": len(mismatches),
                      "decisions": decisions, "sequential_s": round(t_seq, 1), "pool_s": round(t_pool, 1),
                      "sequential_decisions_per_s": round(decisions / t_seq), "pool_decisions_per_s": round(decisions / t_pool)},
                     indent=2))  # fmt: skip
    if mismatches:
        print("first mismatching games:", mismatches[:10], file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
