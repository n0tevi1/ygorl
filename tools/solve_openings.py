"""Solve opening hands of meta decks with ygo-combo-solver and write verified demonstrations (T4a.1).

Usage: uv run python tools/solve_openings.py DECK.ydk|DIR ... [--targets tests/decks/solver_targets.json]
           [--hands 1000] [--seed 0] [--solve-ms 60000] [--fire 14558127 ...] [--fire-ms 20000]
           [--workers N] [--threads 1] [--lines 1] [--env VERSION] [--out PATH] [--summary PATH]

For every deck, hand i is the top of the host shuffle with seed hand_seed(--seed, deck, i); the solver
searches a line from that hand to the deck's target board (--targets, keyed by deck name), and every line
it writes is re-run in our engine, mapped to our action indices and checked (all responses consumed, no
MSG_RETRY, target board present) before it becomes a demonstration. --fire adds one record per hand trap:
the opponent plays it at every legal window of the demonstrated line and the solver rebuilds the board.

Records are appended to --out (JSONL, one per deck x hand x variant; format in docs/solver.md) as they
finish. A content manifest rejects incompatible resumes; a lock prevents concurrent writers.
Hand ranges may be extended; summary and exit status include all records, including previous errors.
Without --env the output defaults to out/demos/solver.jsonl; with --env to
environments/<version>/artifacts/demos/solver.jsonl. Build the solver first: tools/build_combo_solver.sh.
Exit code 1 if any record ends in "error" or "unverified" (a solver line failed the fresh replay).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path

from ygorl.commands import CommandError, load_decks
from ygorl.solver import DEFAULT_FIRE_MS, Demonstration, HandJob, Workdir, find_solver, read_jsonl
from ygorl.solver.batch import done_keys, hand_seed, job_done, load_targets, repair_jsonl, resume_job, summarize
from ygorl.solver.combo_solver import SolverNotFound
from ygorl.solver.resume import bind_output, implementation_identity, output_lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "tests" / "decks" / "solver_targets.json"


def deck_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted(p.glob("*.ydk"))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"solve_openings: error: deck file not found: {p}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decks", nargs="+", type=Path, help=".ydk files or directories of them")
    parser.add_argument(
        "--targets", type=Path, default=DEFAULT_TARGETS, help="per-deck targets JSON (default: %(default)s)"
    )
    parser.add_argument("--hands", type=int, default=10, help="opening hands per deck (default 10)")
    parser.add_argument("--first-hand", type=int, default=0, help="index of the first hand (default 0)")
    parser.add_argument("--seed", type=int, default=0, help="base seed of the hand shuffles (default 0)")
    parser.add_argument(
        "--solve-ms", type=int, default=60_000, help="solver search budget per hand in ms (default 60000)"
    )
    parser.add_argument(
        "--fire",
        type=int,
        action="append",
        default=None,
        metavar="PASSWORD",
        help="hand trap for the --fire variant (repeatable; default: the targets file's list per deck)",
    )
    parser.add_argument(
        "--no-fire", action="store_true", help="skip the --fire variant even if the targets file lists one"
    )
    parser.add_argument("--fire-ms", type=int, default=DEFAULT_FIRE_MS, help="solver budget per --fire window in ms")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel solver processes"
    )
    parser.add_argument("--threads", type=int, default=1, help="solver threads per process (default 1)")
    parser.add_argument("--lines", type=int, default=1, help="verified lines kept per hand (default 1)")
    parser.add_argument("--env", default=None, metavar="PATH|VERSION", help="environment (rules; binds the output)")
    parser.add_argument("--out", type=Path, default=None, help="demonstration file (JSONL, appended)")
    parser.add_argument("--summary", type=Path, default=None, help="also write the per-deck summary as JSON")
    parser.add_argument(
        "--binary", type=Path, default=None, help="solver binary (default: build/combo-solver/bin/combosolver)"
    )
    parser.add_argument("--timeout", type=float, default=None, help="wall-clock limit per solver run in seconds")
    parser.add_argument("--solver-seed", type=int, default=None, help="fixed solver seed (default: the solver's clock)")
    parser.add_argument(
        "--keep-files", action="store_true", help="keep per-hand solver files under the scratch directory"
    )
    parser.add_argument("--scratch", type=Path, default=None, help="scratch directory (default: a temporary one)")
    args = parser.parse_args(argv)

    try:
        binary = find_solver(args.binary)
    except SolverNotFound as exc:
        print(f"solve_openings: error: {exc}", file=sys.stderr)
        return 2
    targets = load_targets(args.targets)
    env_stamp = None
    env = None
    if args.env is not None:
        from ygorl.data import load_environment

        env = load_environment(args.env)
        env_stamp = {"version": env.version, "fingerprint": env.fingerprint}
        env_dir = env.root
    out = args.out or (
        env_dir / "artifacts" / "demos" / "solver.jsonl" if args.env else ROOT / "out" / "demos" / "solver.jsonl"
    )
    if (
        args.hands < 1
        or args.first_hand < 0
        or min(args.workers, args.threads, args.lines, args.solve_ms, args.fire_ms) < 1
    ):
        parser.error("hands, workers, threads, lines and budgets must be positive; first-hand must be nonnegative")
    specs = {}
    try:
        for path in deck_files(args.decks):
            deck = load_decks([path], env)[0]
            if deck.name in specs:
                raise ValueError(f"duplicate deck name: {deck.name}")
            if deck.name not in targets:
                raise ValueError(f"no targets for deck {deck.name} in {args.targets}")
            spec = targets[deck.name]
            fire = () if args.no_fire else tuple(args.fire if args.fire is not None else spec["fire"])
            specs[deck.name] = (path, deck, spec["targets"], sorted(set(fire)))
        if not specs:
            raise ValueError("no decks selected")
        identity = {
            "format": "ygorl-solver-batch-1",
            "environment": env_stamp,
            "decks": {
                name: {"main": list(d.main), "extra": list(d.extra), "side": list(d.side), "targets": t, "fire": f}
                for name, (_, d, t, f) in specs.items()
            },
            "search": {
                key: getattr(args, key)
                for key in ("seed", "solver_seed", "solve_ms", "fire_ms", "threads", "lines", "timeout")
            },
            "implementation": implementation_identity(binary, Path(__file__)),
        }
        with output_lock(out):
            bind_output(out, identity)
            return run_batch(args, binary, env_stamp, out, specs)
    except (ValueError, CommandError) as exc:
        print(f"solve_openings: error: {exc}", file=sys.stderr)
        return 2


def run_batch(args, binary: Path, env_stamp: dict | None, out: Path, specs: dict) -> int:
    cut = repair_jsonl(out)
    if cut:
        print(f"note: removed a partial last record ({cut} bytes) from {out}")
    done = done_keys(out, env_stamp)
    records = [d.to_json() for d in read_jsonl(out)] if out.exists() else []
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="ygorl-solve-"))
    workdir = Workdir.create(scratch / "workdir").path
    jobs: list[tuple[str, HandJob]] = []
    skipped = 0
    for name, (path, _, targets, fire) in specs.items():
        for i in range(args.first_hand, args.first_hand + args.hands):
            job = HandJob(deck_path=path.resolve(), hand_index=i, hand_seed=hand_seed(args.seed, name, i),
                          targets=tuple(targets), workdir=workdir, scratch=scratch / f"{name}_{i}",
                          solve_ms=args.solve_ms, threads=args.threads, lines=args.lines, fire=tuple(fire), fire_ms=args.fire_ms,
                          binary=binary, solver_seed=args.solver_seed, env=args.env, timeout_s=args.timeout,
                          keep_files=args.keep_files)  # fmt: skip
            if job_done(name, job, done):
                skipped += 1
            else:
                jobs.append((name, job))
    print(f"solving {len(jobs)} hands ({skipped} already in {out}) with {args.workers} workers x {args.threads} threads, "
          f"{args.solve_ms} ms per hand{', fire ' + ','.join(map(str, jobs[0][1].fire)) if jobs and jobs[0][1].fire else ''}")  # fmt: skip

    bases = {
        (r["deck"]["name"], r["hand_index"]): Demonstration.from_json(r) for r in records if r["variant"] == "plain"
    }
    pending = [(job, bases.get((name, job.hand_index))) for name, job in jobs]
    previous = len(records)
    start = time.monotonic()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for n, recs in enumerate(pool.imap_unordered(resume_job, pending), start=1):
            for rec in recs:
                demo = Demonstration.from_json(rec)
                if demo.key in done:
                    continue
                demo.append_to(out)
                done.add(demo.key)
                records.append(rec)
                variant = demo.variant if demo.fire is None else f"fire:{demo.fire}"
                info = f"{len(demo.lines)} line(s)" if demo.lines else (demo.error[:100] or "-")
                print(f"[{n}/{len(jobs)} {time.monotonic() - start:7.1f}s] {demo.deck['name']:>18} hand {demo.hand_index:4d} "
                      f"{variant:>14}: {demo.status:10} {info} ({demo.solver.get('wall_s', 0):.1f}s)", flush=True)  # fmt: skip
    elapsed = time.monotonic() - start

    table = summarize(records)
    print(f"\n{len(records) - previous} new records ({len(records)} total) in {elapsed:.1f} s wall -> {out}")
    for key, row in sorted(table.items()):
        rest = ", ".join(f"{k} {v}" for k, v in row.items() if k not in ("hands",))
        print(f"  {key:32} hands {row['hands']:4d}: {rest}")
    if args.summary is not None:
        args.summary.write_text(json.dumps({"elapsed_s": round(elapsed, 1), "new_records": len(records) - previous, "records": len(records), "workers": args.workers, "threads": args.threads,
                                            "solve_ms": args.solve_ms, "fire_ms": args.fire_ms, "table": table}, indent=1))  # fmt: skip
    bad = [r for r in records if r["status"] in ("error", "unverified")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
