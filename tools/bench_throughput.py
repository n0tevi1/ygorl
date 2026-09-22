"""T2.7 throughput benchmark: decisions per second versus worker threads.

Compares the two vectorized environments on the same games:

- ``encoded``: EncodedVecEnv, the whole step loop (tracker, action state machines,
  observation encoder) in C++; Python only picks an index with a trivial chooser.
- ``python``: run_games on the DuelPool, with message decoding, action generation and
  the tracker in Python (RandomAgent), no observation encoding.

Usage: uv run python tools/bench_throughput.py [--games 200] [--threads 1,2,4] [--envs-per-thread 8]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import default_cards
from ygorl.env import GameSpec, run_games
from ygorl.env.encoded import EncodedVecEnv, chooser

DECK_DIR = Path(__file__).resolve().parents[1] / "tests" / "decks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--envs-per-thread", type=int, default=8)
    parser.add_argument("--skip-python", action="store_true", help="only benchmark EncodedVecEnv")
    args = parser.parse_args()

    decks = {p.stem: load_ydk(p) for p in sorted(DECK_DIR.glob("*.ydk"))}
    names = sorted(decks)
    specs = [GameSpec(seed=700_000 + i, deck_a=decks[names[i % 10]], deck_b=decks[names[(i * 7 + 3) % 10]], first=i % 2)
             for i in range(args.games)]  # fmt: skip
    cards = default_cards()
    vocab = CardVocab.from_db(cards)

    rows = []
    for threads in (int(t) for t in args.threads.split(",")):
        envs = threads * args.envs_per_thread
        env = EncodedVecEnv(envs, threads, cards=cards, vocab=vocab)
        t0 = time.perf_counter()
        results = env.play(specs, chooser)
        elapsed = time.perf_counter() - t0
        decisions = sum(r["decisions"] for r in results)
        row = {"threads": threads, "envs": envs, "games": len(specs), "encoded_decisions": decisions,
               "encoded_s": round(elapsed, 2), "encoded_decisions_per_s": round(decisions / elapsed)}  # fmt: skip
        if not args.skip_python:
            t0 = time.perf_counter()
            pooled = run_games(specs, lambda i, s: (RandomAgent(s.seed), RandomAgent(s.seed ^ 0xABC)),
                               num_envs=envs, num_threads=threads)  # fmt: skip
            elapsed = time.perf_counter() - t0
            decisions = sum(r.decisions for r in pooled)
            row.update({"python_decisions": decisions, "python_s": round(elapsed, 2),
                        "python_decisions_per_s": round(decisions / elapsed)})  # fmt: skip
        rows.append(row)
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
