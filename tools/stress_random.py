"""Random-vs-random stress test over the test decks (T1.5 acceptance).

Usage: uv run python tools/stress_random.py [--games-per-deck 1000] [--workers N]

Every deck in tests/decks plays ``--games-per-deck`` games as deck A against
the other decks in rotation (alternating who goes first). Reports unhandled
messages, retries, exceptions and Lua script errors; exits non-zero if any
game had an exception, a retry, or an unknown/undecodable message.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "tests" / "decks"


def play(job):
    seed, name_a, name_b, first = job
    from ygorl.agents import RandomAgent
    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import Duel

    try:
        a, b = load_ydk(DECK_DIR / f"{name_a}.ydk"), load_ydk(DECK_DIR / f"{name_b}.ydk")
        r = Duel(seed, None, a, b, first=first).run(RandomAgent(seed), RandomAgent(seed ^ 0x5EED))
        return {"seed": seed, "a": name_a, "b": name_b, "winner": r.winner, "reason": r.reason, "turns": r.turns,
                "decisions": r.decisions, "retries": r.retries, "unknown": r.unknown_messages,
                "undecodable": r.undecodable_messages, "script_errors": r.script_errors[:3],
                "n_script_errors": len(r.script_errors), "error": r.error, "exception": None}  # fmt: skip
    except Exception:  # noqa: BLE001 - reported, not swallowed
        return {"seed": seed, "a": name_a, "b": name_b, "exception": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-deck", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--report", type=Path, help="write a JSON summary here")
    args = parser.parse_args()

    names = sorted(p.stem for p in DECK_DIR.glob("*.ydk"))
    jobs = []
    for i, a in enumerate(names):
        others = names[:i] + names[i + 1 :]
        for g in range(args.games_per_deck):
            jobs.append((i * 1_000_000 + g, a, others[g % len(others)], g % 2))

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play, jobs, chunksize=4)
    elapsed = time.time() - t0

    exc = [r for r in results if r.get("exception")]
    ok = [r for r in results if not r.get("exception")]
    bad = [r for r in ok if r["retries"] or r["unknown"] or r["undecodable"] or r["reason"] == "error"]
    reasons = collections.Counter(r["reason"] for r in ok)
    script_err = collections.Counter(e.splitlines()[0][:160] for r in ok for e in r["script_errors"])
    decisions = sum(r["decisions"] for r in ok)
    summary = {
        "games": len(results), "decks": len(names), "seconds": round(elapsed, 1), "workers": args.workers,
        "decisions": decisions, "decisions_per_second": round(decisions / elapsed),
        "exceptions": len(exc), "games_with_retry_or_unhandled_message": len(bad),
        "reasons": dict(reasons), "mean_turns": round(sum(r["turns"] for r in ok) / max(len(ok), 1), 1),
        "games_with_script_errors": sum(1 for r in ok if r["n_script_errors"]),
        "top_script_errors": script_err.most_common(10),
    }  # fmt: skip
    print(json.dumps(summary, indent=2))
    for r in exc[:3]:
        print(f"EXCEPTION seed={r['seed']} {r['a']} vs {r['b']}\n{r['exception']}", file=sys.stderr)
    for r in bad[:5]:
        print(f"BAD seed={r['seed']} {r['a']} vs {r['b']}: {r}", file=sys.stderr)
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if exc or bad else 0


if __name__ == "__main__":
    sys.exit(main())
