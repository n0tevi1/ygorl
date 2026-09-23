"""Validate funnel stage 1 against real first turns: paired brick tests and time per deck (T5.6 acceptance).

Usage: uv run python tools/validate_funnel.py out/funnel/validate.jsonl [--rollouts 1000] [--greedy-seeds 4]
           [--reference-ms 30000] [--workers 2] [--out out/funnel/validation.json]

Input: a tools/funnel_eval.py output run with --no-filter --keep-demos (every hand solved, solver records
kept). For every hand of every deck, a real first turn (standard duel: same deck order and core seed as the
solver, DUEL_PSEUDO_SHUFFLE off, passive opponent) is played by
  * the solver's own line, followed move by move (solved hands only),
  * the Greedy agent (--greedy-seeds seeds),
  * --rollouts randomized explorer rollouts from a snapshot of the opening,
and the board at the start of turn 2 is checked for the target cards. A hand is "real non-brick" when any
of them reaches the target. Funnel bricks are also re-solved with a larger budget (--reference-ms).
Printed and written (--out): paired 2x2 tables with the exact McNemar test for funnel brick vs
(a) real first turns (all players), (b) Greedy + explorer only, (c) the larger solver budget,
(d) the best-known verdict (b or c or the line); per-deck brick rates; evaluation time per deck vs budget.
Per-hand observations are cached next to --out (observers.jsonl, reference.jsonl) so a rerun resumes.
Decks must have a single target board (no alternatives).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import statistics
import sys
import tempfile
import time
from pathlib import Path

from ygorl.build.funnel import DEFAULT_BUDGET_S, FunnelResult, paired_table
from ygorl.cards.ydk import Deck

sys.path.insert(0, str(Path(__file__).resolve().parent))
from funnel_eval import read_results  # noqa: E402


def _deck(result: FunnelResult) -> Deck:
    d = result.deck
    return Deck(tuple(d["main"]), tuple(d["extra"]), (), d["name"])


def observe(job: dict) -> dict:
    """Worker: real first turns of one hand (line replay, Greedy, explorer rollouts)."""
    from ygorl.agents.greedy import GreedyAgent
    from ygorl.build.first_turn import explore, first_turn_duel, play_first_turn, replay_line
    from ygorl.engine.duel import DuelSession, default_cards
    from ygorl.solver import Demonstration

    cards = default_cards()
    start = time.monotonic()
    deck = Deck(tuple(job["main"]), tuple(job["extra"]), (), job["deck"])
    targets = job["targets"]
    out = {"deck": job["deck"], "index": job["index"], "line": None, "line_error": "", "line_steps": 0}
    if job.get("demo") is not None:
        demo = Demonstration.from_json(job["demo"])
        ft = replay_line(demo, 0, cards=cards)
        out["line"], out["line_error"], out["line_steps"] = ft.reached, ft.error, ft.steps
        out["line_missing"] = ft.missing
    greedy = []
    for s in range(job["greedy_seeds"]):
        session = DuelSession(first_turn_duel(deck, job["hand_seed"], cards=cards))
        try:
            greedy.append(play_first_turn(session, GreedyAgent(s, cards=cards), targets, cards=cards).reached)
        finally:
            session.close()
    out["greedy"] = greedy
    out["explorer"] = explore(deck, job["hand_seed"], targets, job["rollouts"], seed=job["index"], cards=cards)
    out["wall_s"] = round(time.monotonic() - start, 2)
    return out


def reference(job: dict) -> dict:
    """Worker: re-solve a funnel brick with the larger reference budget."""
    from ygorl.solver import HandJob, solve_hand

    scratch = Path(job["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    deck_path = scratch / f"{job['deck']}.ydk"
    deck_path.write_text(Deck(tuple(job["main"]), tuple(job["extra"]), (), job["deck"]).to_ydk(), encoding="utf-8")
    hj = HandJob(deck_path=deck_path, hand_index=job["index"], hand_seed=job["hand_seed"], targets=tuple(job["targets"]),
                 workdir=Path(job["workdir"]), scratch=scratch / "s", solve_ms=job["solve_ms"], threads=1,
                 solver_seed=job["solver_seed"])  # fmt: skip
    demo = solve_hand(hj)
    return {"deck": job["deck"], "index": job["index"], "status": demo.status, "wall_s": demo.solver.get("wall_s"),
            "combo_actions": demo.lines[0].score.get("actions") if demo.lines else None,
            "best_placed": demo.solver.get("best_placed"), "error": demo.error}  # fmt: skip


def _cached(path: Path) -> dict[tuple, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(rec["deck"], rec["index"])] = rec
    return out


def _run(fn, jobs: list[dict], workers: int, cache: Path, label: str) -> None:
    if not jobs:
        return
    start = time.monotonic()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool, open(cache, "a", encoding="utf-8") as f:
        for n, rec in enumerate(pool.imap_unordered(fn, jobs), start=1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{label} {n}/{len(jobs)} {time.monotonic() - start:6.1f}s] {rec['deck']} hand {rec['index']}: "
                  f"{ {k: v for k, v in rec.items() if k in ('line', 'greedy', 'status', 'wall_s')} }", flush=True)  # fmt: skip


def _fmt(t: dict) -> str:
    return (f"n={t['n']} both={t['both']} only_funnel={t['only_a']} only_other={t['only_b']} neither={t['neither']} "
            f"funnel={t['rate_a']:.3f} other={t['rate_b']:.3f} agreement={t['agreement']:.3f} McNemar p={t['p_value']:.4g}")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="tools/funnel_eval.py output (--no-filter --keep-demos)")
    parser.add_argument("--rollouts", type=int, default=1000, help="explorer rollouts per hand")
    parser.add_argument("--greedy-seeds", type=int, default=4, help="Greedy first turns per hand")
    parser.add_argument("--reference-ms", type=int, default=30_000, help="solver budget of the reference re-solve (0: skip)")
    parser.add_argument("--reference-seed", type=int, default=2, help="solver seed of the reference re-solve")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="summary JSON (default: next to the results)")
    args = parser.parse_args(argv)

    results = read_results(args.results)
    if not results:
        print(f"validate_funnel: error: no funnel results in {args.results}", file=sys.stderr)
        return 2
    out = args.out or args.results.with_name("validation.json")
    obs_cache, ref_cache = out.with_name("observers.jsonl"), out.with_name("reference.jsonl")
    for r in results:
        if len(r.targets) != 1:
            print(f"validate_funnel: error: deck {r.deck['name']} has target alternatives", file=sys.stderr)
            return 2
        if any(h.status == "solved" and not h.demos for h in r.hands):
            print("validate_funnel: error: the results carry no solver records; rerun funnel_eval.py with --keep-demos",
                  file=sys.stderr)  # fmt: skip
            return 2
        if r.stopped_early:
            print(f"validate_funnel: error: deck {r.deck['name']} stopped early; rerun funnel_eval.py with --no-filter",
                  file=sys.stderr)  # fmt: skip
            return 2

    # -- real first turns ---------------------------------------------------
    have = _cached(obs_cache)
    jobs = []
    for r in results:
        d = _deck(r)
        for h in r.hands:
            if (d.name, h.index) in have or h.status == "error":
                continue
            demo = h.demos[0] if h.status == "solved" else None
            jobs.append({"deck": d.name, "main": list(d.main), "extra": list(d.extra), "index": h.index, "hand_seed": h.hand_seed,
                         "targets": r.targets[0], "demo": demo, "greedy_seeds": args.greedy_seeds, "rollouts": args.rollouts})  # fmt: skip
    _run(observe, jobs, args.workers, obs_cache, "first turns")
    observed = _cached(obs_cache)

    # -- reference solver on the funnel's bricks ----------------------------
    if args.reference_ms > 0:
        from ygorl.solver import Workdir

        have = _cached(ref_cache)
        scratch = Path(tempfile.mkdtemp(prefix="ygorl-funnel-ref-"))
        workdir = Workdir.create(scratch / "workdir").path
        jobs = []
        for r in results:
            d = _deck(r)
            for h in r.hands:
                if h.status == "brick" and (d.name, h.index) not in have:
                    jobs.append({"deck": d.name, "main": list(d.main), "extra": list(d.extra), "index": h.index,
                                 "hand_seed": h.hand_seed, "targets": r.targets[0], "solve_ms": args.reference_ms,
                                 "solver_seed": args.reference_seed, "workdir": str(workdir),
                                 "scratch": str(scratch / f"{d.name}_{h.index}")})  # fmt: skip
        _run(reference, jobs, args.workers, ref_cache, "reference")
    refs = _cached(ref_cache)

    # -- paired comparison ---------------------------------------------------
    funnel, real, indep, ref, best, per_deck, fidelity = [], [], [], [], [], {}, []
    for r in results:
        name = r.deck["name"]
        row = per_deck.setdefault(name, {"hands": 0, "funnel_brick": 0, "real_brick": 0, "indep_brick": 0, "ref_brick": 0,
                                         "best_brick": 0, "line_ok": 0, "lines": 0})  # fmt: skip
        for h in r.hands:
            if h.status == "error":
                continue
            o = observed.get((name, h.index))
            if o is None:
                continue
            f_brick = h.status == "brick"
            indep_ok = any(o["greedy"]) or o["explorer"]["reached"] > 0
            line_ok = bool(o["line"])
            real_ok = indep_ok or line_ok
            ref_ok = not f_brick or (refs.get((name, h.index), {}).get("status") == "solved")
            best_ok = real_ok or ref_ok
            for lst, v in ((funnel, f_brick), (real, not real_ok), (indep, not indep_ok), (ref, not ref_ok), (best, not best_ok)):
                lst.append(v)
            row["hands"] += 1
            row["funnel_brick"] += f_brick
            row["real_brick"] += not real_ok
            row["indep_brick"] += not indep_ok
            row["ref_brick"] += not ref_ok
            row["best_brick"] += not best_ok
            if o["line"] is not None:
                row["lines"] += 1
                row["line_ok"] += line_ok
                fidelity.append({"deck": name, "index": h.index, "ok": line_ok, "error": o["line_error"],
                                 "missing": o.get("line_missing", [])})  # fmt: skip
    tables = {"real": paired_table(funnel, real), "independent": paired_table(funnel, indep),
              "reference": paired_table(funnel, ref), "best_known": paired_table(funnel, best)}  # fmt: skip

    # -- time -----------------------------------------------------------------
    times = {r.deck["name"]: {"solver_s": r.solver_s, "wall_s": r.wall_s, "hands": len(r.hands),
                              "budget_s": r.config.get("budget_s", DEFAULT_BUDGET_S), "within_budget": r.within_budget,
                              "per_hand_s": round(r.solver_s / max(1, len(r.hands)), 2)} for r in results}  # fmt: skip
    solver_s = [t["solver_s"] for t in times.values()]
    time_summary = {"mean_solver_s": round(statistics.fmean(solver_s), 1), "max_solver_s": max(solver_s),
                    "min_solver_s": min(solver_s), "within_budget": sum(t["within_budget"] for t in times.values()),
                    "decks": len(times)}  # fmt: skip
    obs_wall = [o["wall_s"] for o in observed.values()]
    summary = {"results": str(args.results), "config": results[0].config, "rollouts": args.rollouts,
               "greedy_seeds": args.greedy_seeds, "reference_ms": args.reference_ms, "tables": tables, "per_deck": per_deck,
               "line_fidelity": {"lines": len(fidelity), "ok": sum(f["ok"] for f in fidelity),
                                 "failures": [f for f in fidelity if not f["ok"]]},
               "time": {"per_deck": times, **time_summary},
               "observer_wall_s": {"mean": round(statistics.fmean(obs_wall), 2) if obs_wall else None},
               "funnel_summaries": [r.summary() for r in results]}  # fmt: skip
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("\nfunnel brick vs ...")
    for k, t in tables.items():
        print(f"  {k:12} {_fmt(t)}")
    print("\nper deck (hands, funnel / real / independent / reference / best-known bricks, solver line ok in real duel):")
    for name, row in per_deck.items():
        print(f"  {name:18} {row['hands']:3d}  {row['funnel_brick']:3d} {row['real_brick']:3d} {row['indep_brick']:3d} "
              f"{row['ref_brick']:3d} {row['best_brick']:3d}   lines {row['line_ok']}/{row['lines']}")  # fmt: skip
    print(f"\ntime per deck (solver process-seconds): mean {time_summary['mean_solver_s']}, min {time_summary['min_solver_s']}, "
          f"max {time_summary['max_solver_s']}; within budget {time_summary['within_budget']}/{time_summary['decks']}")  # fmt: skip
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
