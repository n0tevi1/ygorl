"""Turn-1 demonstrations towards blocking boards for any deck, with automatic targets (docs/solver.md「阻断场面」).

Usage: uv run python tools/solve_blocking.py DECK.ydk|DIR ... [--sample 20] [--sample-seed 0] [--hands 8]
           [--seed 0] [--solve-ms 8000] [--pieces 3] [--no-pair] [--workers N] [--env VERSION]
           [--out PATH] [--summary PATH]

No targets file: for every opening hand, ``ygorl.solver.blocking`` reads the deck's scripts and plans solver
targets -- the hand's interrupting traps / quick-play spells set, its hand traps kept, plus one of the deck's
field interrupters (a monster whose quick effect negates or disrupts, archetype pieces first). Each attempt is
a full solve + fresh-replay verification (tools/solve_openings.py); the record kept is the line whose end board
has the most interruptions. Records are the usual demonstration JSONL (tools/train_bc.py reads them), appended
as they finish, so a rerun resumes. Run it under ``nice -n 19``: the solver is CPU only, one thread per worker.

--summary writes per deck: solve rate, solver wall time per hand, interruptions and the end boards by card name.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import tempfile
import time
from collections import Counter
from functools import partial
from pathlib import Path

from ygorl.cards.ydk import load_ydk
from ygorl.solver import Demonstration, HandJob, Workdir, find_solver
from ygorl.solver.batch import done_keys, hand_seed, repair_jsonl
from ygorl.solver.combo_solver import SolverNotFound

ROOT = Path(__file__).resolve().parents[1]


def deck_files(paths: list[Path], sample: int | None, seed: int) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted(p.glob("*.ydk"))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"solve_blocking: error: deck file not found: {p}")
    if sample is not None and sample < len(out):
        out = sorted(random.Random(seed).sample(out, sample))
    return out


def _run(job: HandJob, pieces: int, pair: bool) -> dict:
    from ygorl.solver.blocking import solve_blocking

    return solve_blocking(job, pieces=pieces, pair=pair)


def summarize(records: list[dict], cards) -> dict:
    """Per deck: hands, solved, solve rate, wall per hand, interruptions and the kept boards by card name."""
    decks: dict[str, dict] = {}
    for r in records:
        row = decks.setdefault(r["deck"]["name"], {"hands": 0, "solved": 0, "wall_s": 0.0, "interruptions": [],
                                                    "negates": [], "boards": []})  # fmt: skip
        row["hands"] += 1
        row["wall_s"] += r.get("solver", {}).get("wall_s", 0.0)
        if r["status"] != "solved":
            continue
        row["solved"] += 1
        b = r["solver"].get("blocking", {})
        row["interruptions"].append(b.get("interruptions", 0))
        row["negates"].append(b.get("negates", 0))
        side = r["lines"][0]["board"]["players"][0]

        def name(code: int) -> str:
            return cards[code].name if code in cards else str(code)

        board = {
            "hand": r["hand_index"],
            "targets": r["targets"],
            "interruptions": [f"{name(c)} ({w}{', negate' if n else ''})" for c, w, n in b.get("pieces", [])],
            "monsters": [name(c["code"]) for c in side["mzone"]],
            "spells_traps": [name(c["code"]) + ("" if c["position"] & 0x5 else " (set)") for c in side["szone"]],
            "hand_left": [name(c) for c in side["hand"]],
        }
        row["boards"].append(board)
    for row in decks.values():
        n = row["hands"]
        row["solve_rate"] = round(row["solved"] / n, 3) if n else 0.0
        row["wall_per_hand_s"] = round(row["wall_s"] / n, 1) if n else 0.0
        row["wall_s"] = round(row["wall_s"], 1)
        row["mean_interruptions"] = (
            round(sum(row["interruptions"]) / len(row["interruptions"]), 2) if row["interruptions"] else 0.0
        )
        row["top_pieces"] = Counter(p.split(" (")[0] for b in row["boards"] for p in b["interruptions"]).most_common(6)
    return decks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decks", nargs="+", type=Path, help=".ydk files or directories of them")
    parser.add_argument("--sample", type=int, default=None, help="solve a random sample of this many decks")
    parser.add_argument("--sample-seed", type=int, default=0, help="seed of --sample (default 0)")
    parser.add_argument("--hands", type=int, default=8, help="opening hands per deck (default 8)")
    parser.add_argument("--first-hand", type=int, default=0, help="index of the first hand (default 0)")
    parser.add_argument("--seed", type=int, default=0, help="base seed of the hand shuffles (default 0)")
    parser.add_argument("--solve-ms", type=int, default=8_000, help="solver budget per attempt in ms (default 8000)")
    parser.add_argument("--pieces", type=int, default=3, help="field interrupters tried per hand (default 3)")
    parser.add_argument("--no-pair", action="store_true", help="do not try two solved pieces together")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel hands")
    parser.add_argument("--env", default=None, metavar="PATH|VERSION", help="environment (rules; binds the output)")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "demos" / "blocking.jsonl", help="JSONL output")
    parser.add_argument("--summary", type=Path, default=None, help="per-deck summary (JSON)")
    parser.add_argument("--binary", type=Path, default=None, help="solver binary (default: find_solver)")
    parser.add_argument("--timeout", type=float, default=None, help="wall-clock limit per solver run in seconds")
    parser.add_argument("--scratch", type=Path, default=None, help="scratch directory (default: a temporary one)")
    args = parser.parse_args(argv)

    try:
        binary = find_solver(args.binary)
    except SolverNotFound as exc:
        print(f"solve_blocking: error: {exc}", file=sys.stderr)
        return 2
    env_stamp = None
    if args.env is not None:
        from ygorl.data import load_environment

        env = load_environment(args.env)
        env_stamp = {"version": env.version, "fingerprint": env.fingerprint}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl(args.out)
    done = done_keys(args.out, env_stamp)
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="ygorl-blocking-"))
    workdir = Workdir.create(scratch / "workdir").path

    jobs: list[HandJob] = []
    for path in deck_files(args.decks, args.sample, args.sample_seed):
        name = load_ydk(path).name
        for i in range(args.first_hand, args.first_hand + args.hands):
            if (name, i, "plain", None) in done:
                continue
            jobs.append(HandJob(deck_path=path.resolve(), hand_index=i, hand_seed=hand_seed(args.seed, name, i),
                                targets=(), workdir=workdir, scratch=scratch / f"{name}_{i}", solve_ms=args.solve_ms,
                                threads=1, binary=binary, env=args.env, timeout_s=args.timeout))  # fmt: skip
    print(f"solving {len(jobs)} hands with {args.workers} workers, {args.solve_ms} ms per attempt, "
          f"{args.pieces} pieces{'' if args.no_pair else ' + pair'}", flush=True)  # fmt: skip
    start = time.monotonic()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        run = partial(_run, pieces=args.pieces, pair=not args.no_pair)
        for n, rec in enumerate(pool.imap_unordered(run, jobs), start=1):
            demo = Demonstration.from_json(rec)
            demo.append_to(args.out)
            b = demo.solver.get("blocking", {})
            print(f"[{n}/{len(jobs)} {time.monotonic() - start:7.1f}s] {demo.deck['name']:>28} hand {demo.hand_index:3d}: "
                  f"{demo.status:9} interruptions {b.get('interruptions', '-')} ({demo.solver.get('wall_s', 0):.0f}s, "
                  f"{len(b.get('attempts', []))} attempts)", flush=True)  # fmt: skip
    elapsed = time.monotonic() - start

    from ygorl.engine.duel import default_cards
    from ygorl.solver import read_jsonl

    names = {load_ydk(p).name for p in deck_files(args.decks, args.sample, args.sample_seed)}
    records = [d.to_json() for d in read_jsonl(args.out) if d.deck["name"] in names]
    table = summarize(records, default_cards())
    hands = sum(r["hands"] for r in table.values())
    solved = sum(r["solved"] for r in table.values())
    print(f"\n{len(jobs)} hands in {elapsed:.0f} s wall -> {args.out}; {solved}/{hands} solved over {len(table)} decks")
    for name, row in sorted(table.items()):
        print(f"  {name:28} {row['solved']}/{row['hands']} solved, {row['wall_per_hand_s']:5.1f} s/hand, "
              f"interruptions {row['interruptions']}", flush=True)  # fmt: skip
    if args.summary is not None:
        args.summary.write_text(json.dumps({"elapsed_s": round(elapsed, 1), "workers": args.workers,
                                            "solve_ms": args.solve_ms, "pieces": args.pieces, "pair": not args.no_pair,
                                            "hands": hands, "solved": solved, "decks": table}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
