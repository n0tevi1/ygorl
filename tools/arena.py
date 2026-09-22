"""Paired-seed arena benchmark over the test decks (T3.2 acceptance).

Usage: uv run python tools/arena.py [--agent-a greedy] [--agent-b random] [--games 2000] [--workers 2]
                                    [--seed 0] [--decks a,b,...] [--env VERSION] [--out report.json]

Agent a and agent b each pilot every deck against every deck (mirrors
included), so ``--games`` is spread evenly over all ordered deck pairings,
``2 * pairs`` paired games per pairing. Prints the pooled report (win rate of
agent a with its Wilson interval, draws, first/second split) and agent a's
win rate per deck it piloted; ``--out`` writes the pooled report as JSON with
every game record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from ygorl.agents import AGENTS
from ygorl.cards.ydk import load_ydk
from ygorl.data import load_environment
from ygorl.eval.arena import Arena, derive_seed, merge

ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "tests" / "decks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-a", default="greedy", choices=sorted(AGENTS))
    parser.add_argument("--agent-b", default="random", choices=sorted(AGENTS))
    parser.add_argument("--games", type=int, default=2000, help="total games (rounded to whole pairs per pairing)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decks", help="comma-separated deck names from tests/decks (default: all)")
    parser.add_argument("--env", help="environment version or directory (default: none)")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out", type=Path, help="write the pooled report (with game records) as JSON")
    args = parser.parse_args()

    names = args.decks.split(",") if args.decks else sorted(p.stem for p in DECK_DIR.glob("*.ydk"))
    decks = [load_ydk(DECK_DIR / f"{n}.ydk") for n in names]
    cells = [(a, b, derive_seed(args.seed, i, j)) for i, a in enumerate(decks) for j, b in enumerate(decks)]
    pairs = max(1, round(args.games / (2 * len(cells))))
    env = load_environment(args.env) if args.env else None

    arena = Arena(AGENTS[args.agent_a], AGENTS[args.agent_b], env=env, workers=args.workers, confidence=args.confidence)
    t0 = time.time()
    reports = arena.run_many(cells, pairs)
    elapsed = time.time() - t0
    pooled = merge(reports)

    print(f"{len(cells)} deck pairings x {2 * pairs} games = {pooled.games} games in {elapsed:.0f}s "
          f"with {args.workers} workers (environment: {env.version if env else 'none'})")  # fmt: skip
    print(pooled.summary())
    print(f"reasons: {pooled.reasons}; mean turns {pooled.mean_turns:.1f}")
    by_deck = defaultdict(list)
    for rep in reports:
        by_deck[rep.deck_a].append(rep)
    print(f"\n{args.agent_a} win rate by the deck it piloted:")
    for name in names:
        r = merge(by_deck[name], deck_a=name)
        print(f"  {name:20s} {r.win_rate:.3f}  ({r.ci[0]:.3f}-{r.ci[1]:.3f}, {r.games} games)")
    if args.out:
        args.out.write_text(json.dumps(pooled.to_dict(), indent=1))
        print(f"\nreport written to {args.out}")
    return 0 if pooled.errors == 0 and pooled.retries == 0 and pooled.unknown_messages == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
