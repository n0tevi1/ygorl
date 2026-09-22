"""Re-verify a demonstration file from scratch (T4a.1): every stored line in a fresh duel.

Usage: uv run python tools/verify_demos.py out/demos/solver.jsonl [--env VERSION]

For every line of every record: a new duel from the record's start position is driven by the stored action
indices; every step must be legal, no MSG_RETRY, the responses must equal the stored ones, the final board must
equal the stored summary and hold the target cards (ygorl.solver.verify_line). Prints counts; exit code 1 on any
failure. Independent of the solver binary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ygorl.solver import DemoError, read_jsonl, verify_line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--env", default=None, metavar="PATH|VERSION", help="environment the records are bound to")
    args = parser.parse_args(argv)
    env = None
    if args.env is not None:
        from ygorl.data import load_environment

        env = load_environment(args.env)
    records = lines = steps = 0
    failures: list[str] = []
    start = time.monotonic()
    for demo in read_jsonl(args.path):
        records += 1
        for i, line in enumerate(demo.lines):
            lines += 1
            steps += len(line.actions)
            try:
                verify_line(demo, i, env=env)
            except (DemoError, ValueError) as exc:
                failures.append(f"{demo.deck['name']} hand {demo.hand_index} {demo.variant} line {i}: {exc}")
    elapsed = time.monotonic() - start
    print(f"{records} records, {lines} lines, {steps} steps re-verified in {elapsed:.1f} s: {len(failures)} failures")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
