"""Turn-1 demonstrations whose end boards survive the opponent's turn 2 (docs/solver.md「存活场面」).

Usage: uv run python tools/solve_survival.py DECK.ydk|DIR ... --checkpoint CKPT [--opponents DIR] [--sample 20]
           [--hands 8] [--solve-ms 10000] [--lines 3] [--pieces 3] [--bosses 2] [--samples 8] [--to-end 0]
           [--workers 12] [--out PATH] [--summary PATH]
       uv run python tools/solve_survival.py --report OUT.jsonl [--summary PATH]

For every opening hand (``ygorl.solver.survival.solve_survival``): solve candidate lines -- the blocking targets
of ``ygorl.solver.blocking`` (strict interruption counter) and plain boss monsters, ``--lines`` verified lines per
attempt -- then replay each line in a real duel against ``--samples`` random opponents of ``--opponents`` (the
same samples for every line of the hand) and let the policy ``--checkpoint`` play both seats through turn 2; the
first ``--to-end`` samples are played to the end of the game (calibration). The kept line is the one with the best
mean survival score (the first player's monsters, cards in hand and LP when its turn 3 starts, less the opponent's
board). Records are the usual demonstration JSONL (tools/train_bc.py reads them; ``solver.survival`` describes every
candidate), appended as they finish, so a rerun resumes; every sample is one row of ``<out>.samples.jsonl``.
Run it under ``nice -n 19`` with ``CUDA_VISIBLE_DEVICES=`` (CPU: one solver thread and one torch thread per worker).

--report (also printed at the end of a run) summarizes a finished output: how often the kept line differs from the
old interruption-count choice, the boards of both, and how well each turn-2 count predicts the game's winner on the
samples played to the end.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from solve_blocking import deck_files  # noqa: E402

from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.solver import Demonstration, HandJob, Workdir, find_solver  # noqa: E402
from ygorl.solver.batch import done_keys, hand_seed, repair_jsonl  # noqa: E402
from ygorl.solver.combo_solver import SolverNotFound  # noqa: E402


def _run(job: HandJob, **kwargs) -> tuple[dict, list[dict]]:
    import torch

    from ygorl.solver.survival import solve_survival

    torch.set_num_threads(1)
    try:
        return solve_survival(job, **kwargs)
    except Exception as exc:  # noqa: BLE001 - one hand; recorded as an error
        from ygorl.engine.duel import DuelConfig
        from ygorl.solver.batch import sample_hand

        deck = load_ydk(job.deck_path)
        hand, _ = sample_hand(deck, job.hand_seed, DuelConfig().player.starting_hand)
        demo = Demonstration.new(deck, hand, hand_index=job.hand_index, hand_seed=job.hand_seed, variant="plain",
                                 targets=())  # fmt: skip
        demo.status, demo.error = "error", f"{type(exc).__name__}: {exc}"
        return demo.to_json(), []


# ------------------------------------------------------------------ report


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def _pearson(xs, ys) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if not sxx or not syy:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy), 3)


def _auc(xs, ys) -> float | None:
    """P(score of a win > score of a loss), ties half (the Mann-Whitney AUC)."""
    pos = [x for x, y in zip(xs, ys) if y]
    neg = [x for x, y in zip(xs, ys) if not y]
    if not pos or not neg:
        return None
    order = sorted((x, i) for i, x in enumerate(pos + neg))
    ranks = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k][1]] = (i + j) / 2 + 1
        i = j + 1
    r = sum(ranks[: len(pos)])
    return round((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 3)


FEATURES = ("score", "t1.monsters", "t1.hand", "t1.interruptions", "t2.monsters", "t2.survivors", "t2.hand",
            "t2.lp", "t2.opp_board", "t2.interruptions")  # fmt: skip


def _feature(row: dict, name: str) -> float:
    if "." not in name:
        return row[name]
    part, key = name.split(".")
    return row[part].get(key, 0)


def calibration(rows: list[dict], old: dict | None = None) -> dict:
    """Over the samples played to the end: each count's correlation with a first-player win, and the AUC."""
    games = [r for r in rows if r.get("end_turn") and not r.get("error") and r.get("t2")]
    ys = [int(r["winner"] == 0) for r in games]
    out = {"games": len(games), "win_rate": _mean(ys), "features": {}}
    for f in FEATURES:
        xs = [_feature(r, f) for r in games]
        out["features"][f] = {"pearson": _pearson(xs, ys), "auc": _auc(xs, ys)}
    # candidate level: mean score of a line vs its win rate over the same samples
    by_cand: dict[tuple, list[dict]] = {}
    for r in games:
        by_cand.setdefault((r["deck"], r["hand_index"], r["cand"]), []).append(r)
    xs = [sum(r["score"] for r in v) / len(v) for v in by_cand.values() if len(v) >= 2]
    ys2 = [sum(r["winner"] == 0 for r in v) / len(v) for v in by_cand.values() if len(v) >= 2]
    out["lines"] = {"n": len(xs), "pearson_score_vs_win_rate": _pearson(xs, ys2)}
    # within a hand: does the best-scoring line win more often than the worst one (paired samples)?
    by_hand: dict[tuple, dict[int, list[dict]]] = {}
    for r in games:
        if r["cand"] >= 0:
            by_hand.setdefault((r["deck"], r["hand_index"]), {}).setdefault(r["cand"], []).append(r)
    diffs = []
    for cands in by_hand.values():
        scored = [(sum(r["score"] for r in v) / len(v), sum(r["winner"] == 0 for r in v) / len(v))
                  for v in cands.values() if v]  # fmt: skip
        if len(scored) >= 2:
            scored.sort()
            if scored[-1][0] > scored[0][0]:
                diffs.append(scored[-1][1] - scored[0][1])
    out["hands"] = {"n": len(diffs), "win_rate_best_minus_worst_score": _mean(diffs)}
    by_dead = [r for r in games if r["t2"].get("lp", 1) <= 0]
    out["dead_by_turn_2"] = len(by_dead)
    out["logistic"] = logistic_fit(games)
    out["holdout"] = holdout_selection(games, old or {})
    return out


LOGIT_FEATURES = ("t2.monsters", "t2.hand", "t2.lp", "t2.opp_board", "t2.survivors", "t2.interruptions")


def logistic_fit(games: list[dict], steps: int = 3000, l2: float = 1e-3) -> dict:
    """A logistic regression of a first-player win on the turn-2 counts (LP in thousands); in-sample AUC."""
    import numpy as np

    if len(games) < 20:
        return {}
    X = np.array([[_feature(r, f) / (1000.0 if f == "t2.lp" else 1.0) for f in LOGIT_FEATURES] for r in games])
    y = np.array([float(r["winner"] == 0) for r in games])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(steps):  # Newton steps on a small, well-conditioned problem
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    raw = w[1:] / sd  # per unit of each count
    pred = Z @ w
    return {"features": list(LOGIT_FEATURES), "coef_per_unit": [round(float(x), 4) for x in raw],
            "coef_standardized": [round(float(x), 4) for x in w[1:]], "auc": _auc(list(pred), list(y.astype(int)))}  # fmt: skip


def holdout_selection(games: list[dict], old: dict) -> dict:
    """Choose a line per hand by the mean score of the first half of the samples; win rates on the second half.

    Compared on the same held-out samples: the chosen line, the old interruption-count choice (``old``: (deck,
    hand) -> candidate index), a uniformly random candidate (the mean over candidates) and the policy's own turn 1.
    """
    by_hand: dict[tuple, dict[int, list[dict]]] = {}
    for r in games:
        by_hand.setdefault((r["deck"], r["hand_index"]), {}).setdefault(r["cand"], []).append(r)
    chosen, mean_c, policy, olds, n = [], [], [], [], 0
    for cands in by_hand.values():
        lines = {k: v for k, v in cands.items() if k >= 0}
        if not lines:
            continue
        half = max(r["sample"] for v in lines.values() for r in v) // 2 + 1
        fit = {k: [r for r in v if r["sample"] < half] for k, v in lines.items()}
        test = {k: [r for r in v if r["sample"] >= half] for k, v in lines.items()}
        if not all(fit.values()) or not all(test.values()):
            continue
        n += 1
        best = max(fit, key=lambda k: (sum(r["score"] for r in fit[k]) / len(fit[k]), -k))
        win = lambda rows: sum(r["winner"] == 0 for r in rows) / len(rows)  # noqa: E731
        chosen.append(win(test[best]))
        o = old.get((next(iter(lines.values()))[0]["deck"], next(iter(lines.values()))[0]["hand_index"]))
        if o is not None and o in test:
            olds.append(win(test[o]))
        mean_c.append(sum(win(v) for v in test.values()) / len(test))
        pol = [r for r in cands.get(-1, []) if r["sample"] >= half]
        if pol:
            policy.append(win(pol))
    return {"hands": n, "win_rate_chosen": _mean(chosen), "win_rate_old_choice": _mean(olds),
            "old_hands": len(olds), "win_rate_random_candidate": _mean(mean_c), "win_rate_policy_turn1": _mean(policy)}  # fmt: skip


def report(out: Path, samples: Path | None = None, names: set[str] | None = None, weights: dict | None = None) -> dict:
    from ygorl.engine.duel import default_cards
    from ygorl.solver import read_jsonl

    cards = default_cards()

    def name(code: int) -> str:
        return cards[code].name if code in cards else str(code)

    recs = [d for d in read_jsonl(out) if names is None or d.deck["name"] in names]
    solved = [d for d in recs if d.status == "solved" and d.solver.get("survival", {}).get("chosen") is not None]
    differ, rows_new, rows_old, rows_pol, examples = 0, [], [], [], []
    labels: dict[str, int] = {}
    for d in solved:
        s = d.solver["survival"]
        cands = s["candidates"]
        new = cands[s["chosen"]]
        labels[new["label"].split(":")[0]] = labels.get(new["label"].split(":")[0], 0) + 1
        old = cands[s["old_choice"]] if s.get("old_choice") is not None else None
        rows_new.append(new)
        if old is not None:
            rows_old.append(old)
            differ += s["chosen"] != s["old_choice"]
        if s.get("policy"):
            rows_pol.append(s["policy"])
        if len(examples) < 12 and old is not None and s["chosen"] != s["old_choice"]:

            def board(c):
                b = c["board"]
                return {"monsters": [name(x) for x in b["mzone"]], "spells_traps": [name(x) for x in b["szone"]],
                        "hand": [name(x) for x in b["hand"]], "score": c["eval"]["score"],
                        "strict_interruptions": c["strict"][0], "old_interruptions": c["old"][0],
                        "t2_monsters": c["eval"].get("t2_monsters"), "t2_hand": c["eval"].get("t2_hand"),
                        "win_rate": c["eval"].get("win_rate")}  # fmt: skip

            examples.append({"deck": d.deck["name"], "hand": d.hand_index, "new": board(new), "old": board(old)})

    def stats(rows: list[dict]) -> dict:
        ev = [r["eval"] for r in rows if r.get("eval") and r["eval"].get("n")]
        return {
            "n": len(rows), "monsters": _mean(r["monsters"] for r in rows), "hand": _mean(r["hand"] for r in rows),
            "set_st": _mean(r["set_st"] for r in rows), "interruptions_old": _mean(r["old"][0] for r in rows),
            "interruptions_strict": _mean(r["strict"][0] for r in rows),
            **{k: _mean(e[k] for e in ev if e.get(k) is not None)
               for k in ("score", "t2_monsters", "t2_survivors", "t2_hand", "t2_lp", "t2_opp_board", "t2_dead",
                         "deviated", "opp_t1_actions", "win_rate")},
        }  # fmt: skip

    policy = {k: _mean(p[k] for p in rows_pol if p.get(k) is not None)
              for k in ("score", "t1_monsters", "t1_hand", "t1_interruptions", "t2_monsters", "t2_hand", "t2_dead",
                        "win_rate")}  # fmt: skip
    out_d = {"hands": len(recs), "solved": len(solved), "chosen_differs_from_old": differ,
             "compared": len(rows_old), "chosen_labels": labels, "new": stats(rows_new), "old": stats(rows_old),
             "policy_turn1": policy,
             "candidates_per_hand": _mean(len(d.solver["survival"]["candidates"]) for d in solved),
             "solve_s_per_hand": _mean(d.solver["survival"]["solve_s"] for d in recs if "survival" in d.solver),
             "eval_s_per_hand": _mean(d.solver["survival"]["eval_s"] for d in recs if "survival" in d.solver),
             "examples": examples}  # fmt: skip
    samples = samples or out.with_suffix(".samples.jsonl")
    if samples.exists():
        rows = [json.loads(x) for x in samples.read_text().splitlines() if x.strip()]
        rows = [r for r in rows if names is None or r["deck"] in names]
        if weights:  # rescore the samples (the calibration and the held-out selection use these weights)
            from ygorl.solver.survival import survival_score

            for r in rows:
                r["score"] = survival_score(r, weights)
        old = {(d.deck["name"], d.hand_index): d.solver["survival"]["old_choice"] for d in solved}
        out_d["calibration"] = calibration(rows, old)
    return out_d


def print_report(r: dict) -> None:
    print(f"\n{r['solved']}/{r['hands']} hands solved; {r['candidates_per_hand']} candidate lines per hand; "
          f"solve {r['solve_s_per_hand']} s + evaluation {r['eval_s_per_hand']} s per hand")  # fmt: skip
    print(f"kept line differs from the old interruption-count choice: {r['chosen_differs_from_old']}/{r['compared']}"
          f"; kept by label {r['chosen_labels']}")  # fmt: skip
    keys = (
        "monsters",
        "hand",
        "set_st",
        "interruptions_old",
        "interruptions_strict",
        "score",
        "t2_monsters",
        "t2_survivors",
        "t2_hand",
        "t2_lp",
        "t2_opp_board",
        "t2_dead",
        "deviated",
        "opp_t1_actions",
        "win_rate",
    )
    print(f"  {'':22}{'new':>10}{'old':>10}")
    for k in keys:
        print(f"  {k:22}{r['new'].get(k)!s:>10}{r['old'].get(k)!s:>10}")
    print(f"  policy's own turn 1: {r['policy_turn1']}")
    cal = r.get("calibration")
    if cal:
        print(f"calibration: {cal['games']} games played to the end, first-player win rate {cal['win_rate']}")
        for f, v in cal["features"].items():
            print(f"  {f:20} pearson {v['pearson']!s:>7}  auc {v['auc']!s:>6}")
        print(
            f"  per line: score vs win rate pearson {cal['lines']['pearson_score_vs_win_rate']} (n={cal['lines']['n']})"
        )
        print(f"  per hand: win rate of the best-scoring minus the worst-scoring line "
              f"{cal['hands']['win_rate_best_minus_worst_score']} (n={cal['hands']['n']})")  # fmt: skip
        print(f"  logistic fit (win ~ turn-2 counts): {cal.get('logistic')}")
        print(f"  held-out selection (choose on half the samples, win rate on the other half): {cal['holdout']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decks", nargs="*", type=Path, help=".ydk files or directories of them")
    parser.add_argument("--checkpoint", type=Path, help="policy checkpoint that plays turn 2 (both seats)")
    parser.add_argument("--opponents", type=Path, default=None, help="opponent decks (default: the decks given)")
    parser.add_argument("--sample", type=int, default=None, help="solve a random sample of this many decks")
    parser.add_argument("--sample-seed", type=int, default=0, help="seed of --sample (default 0)")
    parser.add_argument("--hands", type=int, default=8, help="opening hands per deck (default 8)")
    parser.add_argument("--first-hand", type=int, default=0, help="index of the first hand (default 0)")
    parser.add_argument("--seed", type=int, default=0, help="base seed of the hand shuffles and opponents (default 0)")
    parser.add_argument("--solve-ms", type=int, default=10_000, help="solver budget per attempt in ms (default 10000)")
    parser.add_argument("--lines", type=int, default=3, help="verified lines kept per attempt (default 3)")
    parser.add_argument("--pieces", type=int, default=3, help="field interrupters tried per hand (default 3)")
    parser.add_argument("--no-pair", action="store_true", help="do not try two solved pieces together")
    parser.add_argument("--bosses", type=int, default=2, help="plain boss monster targets per hand (default 2)")
    parser.add_argument("--samples", type=int, default=8, help="opponent samples per line (default 8)")
    parser.add_argument("--to-end", type=int, default=0, help="samples per line played to the game's end (default 0)")
    parser.add_argument(
        "--weights",
        type=json.loads,
        default=None,
        help="survival score weights as JSON, e.g. '{\"hand\": 0.25}' (default: SCORE_WEIGHTS)",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel hands")
    parser.add_argument("--env", default=None, metavar="PATH|VERSION", help="environment (rules; binds the output)")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "demos" / "survival.jsonl", help="JSONL output")
    parser.add_argument("--summary", type=Path, default=None, help="write the report (JSON)")
    parser.add_argument("--report", type=Path, default=None, help="only report on this finished output")
    parser.add_argument("--binary", type=Path, default=None, help="solver binary (default: find_solver)")
    parser.add_argument("--timeout", type=float, default=None, help="wall-clock limit per solver run in seconds")
    parser.add_argument("--scratch", type=Path, default=None, help="scratch directory (default: a temporary one)")
    args = parser.parse_args(argv)

    if args.report is not None:
        r = report(args.report, weights=args.weights)
        print_report(r)
        if args.summary is not None:
            args.summary.write_text(json.dumps(r, indent=1))
        return 0
    if not args.decks or args.checkpoint is None:
        parser.error("decks and --checkpoint are required (or --report)")
    try:
        binary = find_solver(args.binary)
    except SolverNotFound as exc:
        print(f"solve_survival: error: {exc}", file=sys.stderr)
        return 2
    env_stamp = None
    if args.env is not None:
        from ygorl.data import load_environment

        env = load_environment(args.env)
        env_stamp = {"version": env.version, "fingerprint": env.fingerprint}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples_path = args.out.with_suffix(".samples.jsonl")
    repair_jsonl(args.out)
    repair_jsonl(samples_path)
    done = done_keys(args.out, env_stamp)
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="ygorl-survival-"))
    workdir = Workdir.create(scratch / "workdir").path
    opponents = [str(p) for p in deck_files([args.opponents] if args.opponents else args.decks, None, 0)]

    files = deck_files(args.decks, args.sample, args.sample_seed)
    jobs: list[HandJob] = []
    for path in files:
        name = load_ydk(path).name
        for i in range(args.first_hand, args.first_hand + args.hands):
            if (name, i, "plain", None) in done:
                continue
            jobs.append(HandJob(deck_path=path.resolve(), hand_index=i, hand_seed=hand_seed(args.seed, name, i),
                                targets=(), workdir=workdir, scratch=scratch / f"{name}_{i}", solve_ms=args.solve_ms,
                                threads=1, lines=args.lines, binary=binary, env=args.env,
                                timeout_s=args.timeout))  # fmt: skip
    print(f"{len(jobs)} hands, {args.workers} workers, {args.solve_ms} ms per attempt, {args.lines} lines per attempt, "
          f"{args.samples} opponent samples per line ({args.to_end} to the end)", flush=True)  # fmt: skip
    start = time.monotonic()
    run = partial(_run, checkpoint=str(args.checkpoint.resolve()), opponents=opponents, samples=args.samples,
                  to_end=args.to_end, pieces=args.pieces, pair=not args.no_pair, bosses=args.bosses,
                  seed=args.seed, weights=args.weights)  # fmt: skip
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for n, (rec, rows) in enumerate(pool.imap_unordered(run, jobs), start=1):
            if rows:
                with open(samples_path, "a", encoding="utf-8") as f:
                    f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
            demo = Demonstration.from_json(rec)
            demo.append_to(args.out)
            s = demo.solver.get("survival", {})
            ch, old = s.get("chosen"), s.get("old_choice")
            cands = s.get("candidates", [])
            desc = "-"
            if ch is not None:
                c = cands[ch]
                desc = (f"kept {c['label']}/{c['line']} score {c['eval']['score']} "
                        f"({'same as' if ch == old else 'old was'} "
                        f"{cands[old]['label'] + '/' + str(cands[old]['line']) if old is not None else '-'})")  # fmt: skip
            print(f"[{n}/{len(jobs)} {time.monotonic() - start:7.1f}s] {demo.deck['name']:>28} hand "
                  f"{demo.hand_index:3d}: {demo.status:9} {len(cands)} lines, {desc} "
                  f"({s.get('solve_s', 0):.0f}+{s.get('eval_s', 0):.0f}s)", flush=True)  # fmt: skip
    print(f"\n{len(jobs)} hands in {time.monotonic() - start:.0f} s wall -> {args.out}")
    names = {load_ydk(p).name for p in files}
    r = report(args.out, samples_path, names)
    print_report(r)
    if args.summary is not None:
        args.summary.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
