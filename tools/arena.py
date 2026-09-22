"""Paired-seed arena benchmark over the test decks (T3.2 acceptance); a wrapper around `ygorl arena`.

Usage: uv run python tools/arena.py [--agent-a greedy] [--agent-b random] [--games 2000] [--workers 2]
                                    [--seed 0] [--decks a,b,...] [--env VERSION] [--out report.json]

Same as ``uv run ygorl arena tests/decks --games 2000 --workers 2 ...`` (``--decks`` picks deck names from
tests/decks): agent a and agent b each pilot every deck against every deck (mirrors included), so ``--games``
is spread evenly over all ordered deck pairings, ``2 * pairs`` paired games per pairing. Prints the pooled
report and agent a's win rate per deck; ``--out`` writes the pooled report as JSON with every game record.
Exit code 1 if any game raised, retried or produced an unknown message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ygorl.cli import main as ygorl

DECK_DIR = Path(__file__).resolve().parents[1] / "tests" / "decks"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-a", default="greedy")
    parser.add_argument("--agent-b", default="random")
    parser.add_argument("--games", default="2000", help="total games (rounded to whole pairs per pairing)")
    parser.add_argument("--workers", default="2")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--decks", help="comma-separated deck names from tests/decks (default: all)")
    parser.add_argument("--env", help="environment version or directory (default: none)")
    parser.add_argument("--confidence", default="0.95")
    parser.add_argument("--out", help="write the pooled report (with game records) as JSON")
    args = parser.parse_args(argv)

    decks = [str(DECK_DIR / f"{n}.ydk") for n in args.decks.split(",")] if args.decks else [str(DECK_DIR)]
    forwarded = ["arena", *decks]
    for flag in ("agent_a", "agent_b", "games", "workers", "seed", "env", "confidence", "out"):
        value = getattr(args, flag)
        if value is not None:
            forwarded += [f"--{flag.replace('_', '-')}", value]
    return ygorl(forwarded)


if __name__ == "__main__":
    sys.exit(main())
