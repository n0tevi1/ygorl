"""Graded hand-trap labels: value of firing at window k = best blocking board the combo can still reach without the
interruption minus the best it can reach after firing at k, over the hand's solved ladder of target boards
(chosen board, other solved piece boards, the base of set traps / kept hand traps). Uses the solver's --fire mode
with an explicit --target ladder level and --target-subset (the board must contain the level, not equal it).
  uv run --no-sync python out/claude-scale/handtrap_labels_graded.py BASE.jsonl --out OUT.jsonl [--min-hand 40 --max-hand 50]
      [--recovery-out RECOVERY.jsonl]
--recovery-out (2026-10-24): also keep every solved level's fire record (variant "fire", its lines = the recovery lines,
the level in solver["level"]), highest level first per hand, so LineLibrary.load(recovery=[...]) and the P0b probe can
use the line that reaches the best board left after firing at each window. YGORL_ROOT picks the checkout whose src/ is
imported (default: the claude-rl-review worktree); levels are solved with ygorl.solver.solve_fire(targets=level),
which needs that checkout to have it (branch claude/recovery-coverage).
--fire P (repeatable, 2026-10-26; default Ash 14558127): the hand traps to label, one job per (hand, card); every
graded record carries "fire" (records without it are Ash), the fire records carry the card in "fire" and their start
has the card appended to the responder's deck. --hands deck:index,... restricts to those hands (timing trials);
--max-levels N solves only the N highest levels of a hand (budget cut: the labels then only see those levels).
"""

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ.get("YGORL_ROOT", "/home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review"))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/home/ya0guang/Code/ygorl/out/claude-scale")
BINARY = Path("/home/ya0guang/Code/ygorl/build/combo-solver/bin/combosolver")
DECKS = ROOT / "environments/md-2026-09/meta"
ASH = 14558127


def fire_level(job, base, fire, targets):
    """solve_fire with an explicit target level (subset match); returns the Demonstration (lines = recovered windows)."""
    from ygorl.solver.batch import solve_fire

    return solve_fire(job, base, fire, targets=targets)


def work(args):
    job, rec, card, max_levels = args
    from ygorl.train.line_labels import fire_windows
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards
    from ygorl.solver import Demonstration

    env = load_environment("md-2026-09")
    cards = default_cards()
    base = Demonstration.from_json(rec)
    b = rec["solver"].get("blocking", {})
    levels = []
    seen = set()
    for a in b.get("attempts", []):
        if a.get("status") == "solved" and a.get("targets"):
            key = tuple(sorted(a["targets"]))
            if key in seen:
                continue
            seen.add(key)
            levels.append(
                {
                    "targets": a["targets"],
                    "score": 3 * a.get("interruptions", 0) + a.get("negates", 0),
                    "label": a.get("label"),
                }
            )
    if b.get("base") and tuple(sorted(b["base"])) not in seen:
        levels.append({"targets": list(b["base"]), "score": 3 * len(b["base"]), "label": "base"})
    levels.sort(key=lambda l: -l["score"])
    all_levels = len(levels)
    if max_levels:
        levels = levels[:max_levels]
    out = {
        "deck": rec["deck"]["name"],
        "hand_index": rec["hand_index"],
        "fire": card,
        "levels": [],
        "windows": None,
        "values": {},
        "levels_total": all_levels,
    }
    best_without = levels[0]["score"] if levels else 0
    per_window_best_after = {}
    fires = []  # solved level fire records with a recovered window, highest level first
    for lv in levels:
        d = fire_level(job, base, card, lv["targets"])
        W = d.solver.get("windows") or 0
        if out["windows"] is None and "windows" in d.solver:
            out["windows"] = W
        rec_w = set()
        if d.status == "solved":
            fd = d
            if fd.start is None:
                fd.start = json.loads(json.dumps(base.start))
                fd.start["decks"]["b"]["main"].append(card)
            rec_w = set(fire_windows(fd, cards=cards, env=env, fire=card))
            if rec_w:
                fd.solver["level"] = {
                    "targets": lv["targets"],
                    "score": lv["score"],
                    "label": lv["label"],
                    "recovered": sorted(rec_w),
                }
                fires.append(fd.to_json())
        out["levels"].append(
            {
                **lv,
                "status": d.status,
                "windows": W,
                "converted": d.solver.get("converted"),
                "recovered": sorted(rec_w),
                "wall_s": d.solver.get("wall_s"),
            }
        )
        for k in rec_w:
            per_window_best_after[k] = max(per_window_best_after.get(k, 0), lv["score"])
    W = out["windows"] or 0
    out["values"] = {str(k): best_without - per_window_best_after.get(k, 0) for k in range(1, W + 1)}
    out["best_without"] = best_without
    return out, fires


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("--out", required=True)
    p.add_argument("--min-hand", type=int, default=0)
    p.add_argument("--max-hand", type=int, default=10**9)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--fire-ms", type=int, default=20000)
    p.add_argument("--solve-ms", type=int, default=10000)
    p.add_argument("--lines", type=int, default=8)
    p.add_argument("--recovery-out", default=None, help="JSONL of the solved level fire records (recovery lines)")
    p.add_argument("--fire", type=int, action="append", default=None, help="hand trap password (repeatable; Ash)")
    p.add_argument("--hands", default=None, help="deck:index,... only these hands")
    p.add_argument("--max-levels", type=int, default=0, help="solve only the N highest levels (0: all)")
    a = p.parse_args()
    from ygorl.solver import HandJob, Workdir
    from ygorl.solver.batch import hand_seed

    done = set()
    if Path(a.out).exists():
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["deck"], r["hand_index"], r.get("fire", ASH)))
    fires = a.fire or [ASH]
    only = None if a.hands is None else {(h.split(":")[0], int(h.split(":")[1])) for h in a.hands.split(",")}
    scratch = Path(tempfile.mkdtemp(prefix="ygorl-graded-"))
    workdir = Workdir.create(scratch / "workdir").path
    jobs = []
    for l in open(a.base):
        r = json.loads(l)
        if r["variant"] != "plain" or r["status"] != "solved" or not (a.min_hand <= r["hand_index"] < a.max_hand):
            continue
        name = r["deck"]["name"]
        if only is not None and (name, r["hand_index"]) not in only:
            continue
        path = DECKS / f"{name}.ydk"
        for card in fires:
            if (name, r["hand_index"], card) in done:
                continue
            jobs.append(
                (
                    HandJob(
                        deck_path=path.resolve(),
                        hand_index=r["hand_index"],
                        hand_seed=hand_seed(0, name, r["hand_index"]),
                        targets=(),
                        workdir=workdir,
                        scratch=scratch / f"{name}_{r['hand_index']}_{card}",
                        solve_ms=a.solve_ms,
                        threads=1,
                        fire=(card,),
                        fire_ms=a.fire_ms,
                        binary=BINARY,
                        env="md-2026-09",
                        solver_seed=0,
                        lines=a.lines,
                    ),
                    r,
                    card,
                    a.max_levels,
                )
            )
    print(f"{len(jobs)} hands, {a.workers} workers", flush=True)
    t0 = time.time()
    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        for n, (o, fires) in enumerate(pool.imap_unordered(work, jobs), 1):
            if a.recovery_out and fires:
                with open(a.recovery_out, "a") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in fires))
            with open(a.out, "a") as f:
                f.write(json.dumps(o) + "\n")
            print(
                f"[{n}/{len(jobs)} {time.time() - t0:6.0f}s] {o['deck']:>18} hand {o['hand_index']} fire {o['fire']}: windows {o['windows']} levels {len(o['levels'])} values {o['values']}",
                flush=True,
            )
    print("GRADED-DONE")


if __name__ == "__main__":
    main()
