"""Funnel stage 1 on candidate decks: solver opening analysis, filter verdict and QD descriptors (T5.6).

Usage: uv run python tools/funnel_eval.py DECK.ydk|DIR ... [--targets tests/decks/solver_targets.json]
           [--hands 12] [--seed 0] [--solve-ms 5000] [--fire 14558127 ...] [--no-fire] [--fire-ms 3000]
           [--workers 1] [--max-rollouts N] [--max-brick-rate 0.5] [--min-survival 0] [--no-filter] [--keep-demos]
           [--env VERSION] [--out out/funnel/results.jsonl]

Every deck gets --hands opening hands with fixed seeds (the same shuffle permutations for every deck);
the combo solver searches each for a line to the deck's target board (--targets, keyed by deck name;
several alternatives per deck are allowed as a list of lists), and every solved hand is re-run with each
--fire hand trap. One JSON record per deck is appended to --out (format "ygorl-funnel", docs/funnel.md);
decks already in the file are skipped, so an interrupted run resumes. With the filter on (default), a
deck stops as soon as its bricks exceed --max-brick-rate of the planned hands.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ygorl.build.funnel import DEFAULT_BUDGET_S, FunnelConfig, FunnelFilter, FunnelResult, evaluate_deck
from ygorl.cards.ydk import load_ydk
from ygorl.solver import find_solver
from ygorl.solver.combo_solver import SolverNotFound

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
            raise SystemExit(f"funnel_eval: error: deck file not found: {p}")
    return out


def load_deck_targets(path: Path) -> dict[str, list]:
    """``{deck name: targets}``: the solver targets file (``{"targets": [...]}`` or a plain list per deck)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: (v["targets"] if isinstance(v, dict) else v) for k, v in raw.items() if not k.startswith("_")}


def read_results(path: Path) -> list[FunnelResult]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(FunnelResult.from_json(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue  # a partial last line of an interrupted run
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decks", nargs="+", type=Path, help=".ydk files or directories of them")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="per-deck targets JSON (default: %(default)s)")
    parser.add_argument("--hands", type=int, default=FunnelConfig.hands, help="opening hands per deck")
    parser.add_argument("--seed", type=int, default=0, help="base seed of the hand shuffles")
    parser.add_argument("--solve-ms", type=int, default=FunnelConfig.solve_ms, help="solver budget per hand (ms)")
    parser.add_argument("--fire", type=int, action="append", default=None, metavar="PASSWORD", help="hand trap (repeatable)")
    parser.add_argument("--no-fire", action="store_true", help="skip the --fire variant")
    parser.add_argument("--fire-ms", type=int, default=FunnelConfig.fire_ms, help="solver budget per --fire window (ms)")
    parser.add_argument("--workers", type=int, default=1, help="parallel solver processes per deck")
    parser.add_argument("--max-rollouts", type=int, default=None,
                        help="solver count budget per phase and worker (reproducible runs; set --solve-ms well above it)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_S, help="stage-1 budget, solver process-seconds per deck")
    parser.add_argument("--max-brick-rate", type=float, default=FunnelFilter.max_brick_rate)
    parser.add_argument("--min-survival", type=float, default=FunnelFilter.min_hand_trap_survival)
    parser.add_argument("--no-filter", action="store_true", help="no verdict and no early stopping: every hand is solved")
    parser.add_argument("--keep-demos", action="store_true", help="keep the solver records in the output (for validation)")
    parser.add_argument("--env", default=None, metavar="PATH|VERSION", help="environment (rules; stamps the output)")
    parser.add_argument("--binary", type=Path, default=None, help="solver binary")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "funnel" / "results.jsonl", help="output JSONL (appended)")
    args = parser.parse_args(argv)

    try:
        binary = find_solver(args.binary)
    except SolverNotFound as exc:
        print(f"funnel_eval: error: {exc}", file=sys.stderr)
        return 2
    targets = load_deck_targets(args.targets)
    fire = () if args.no_fire else tuple(args.fire if args.fire is not None else FunnelConfig.fire)
    config = FunnelConfig(hands=args.hands, seed=args.seed, solve_ms=args.solve_ms, fire=fire, fire_ms=args.fire_ms,
                          workers=args.workers, env=args.env, binary=binary, budget_s=args.budget, max_rollouts=args.max_rollouts,
                          keep_demos=args.keep_demos)  # fmt: skip
    gate = None if args.no_filter else FunnelFilter(args.max_brick_rate, args.min_survival)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {r.deck["name"] for r in read_results(args.out)}
    start = time.monotonic()
    rows = []
    for path in deck_files(args.decks):
        deck = load_ydk(path)
        if deck.name not in targets:
            print(f"funnel_eval: error: no targets for deck {deck.name} in {args.targets}", file=sys.stderr)
            return 2
        if deck.name in done:
            print(f"{deck.name}: already in {args.out}, skipped")
            continue

        def show(h, name=deck.name):
            extra = " ".join(f"fire {k}: {v['status']} {v['converted']}/{v['windows']}" for k, v in h.fire.items())
            print(f"  {name} hand {h.index:3d}: {h.status:6} {h.combo_actions or '':>3} {h.solver_s:6.1f}s {extra}", flush=True)

        result = evaluate_deck(deck, targets[deck.name], config, filter=gate, progress=show)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_json(), separators=(",", ":")) + "\n")
        s = result.summary()
        rows.append(s)
        verdict = "" if result.passed is None else (" PASS" if result.passed else f" FAIL ({'; '.join(result.reasons)})")
        print(f"{deck.name}: brick {s['bricks']}/{s['valid_hands']} = {s['brick_rate']:.2f} {s['brick_rate_ci95']}, "
              f"survival {s['hand_trap_survival']}, combo {s['combo_actions']['mean']}, solver {s['solver_s']} s "
              f"(budget {config.budget_s:.0f} s: {'ok' if s['within_budget'] else 'OVER'}), wall {s['wall_s']} s{verdict}",
              flush=True)  # fmt: skip
    print(f"{len(rows)} decks in {time.monotonic() - start:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
