"""Budget study of funnel stage 1 (T5.6): hands per deck x solver time per hand x --fire time.

Usage:
  uv run python tools/funnel_budget.py fire RESULTS.jsonl --fire-ms 10000 [--max-index 12] [--workers 2]
      [--out out/budget/fire_10000.jsonl]
  uv run python tools/funnel_budget.py analyze REFERENCE.jsonl [--direct RUN.jsonl ...] [--fire-run FIRE.jsonl ...]
      [--observers out/funnel/observers.jsonl --re-solved out/funnel/reference.jsonl] [--budget 120]
      [--bootstrap 2000] [--out out/budget/study.json]

The runs themselves are tools/funnel_eval.py outputs with --no-filter (every hand solved):
  * REFERENCE: a large budget, e.g. --hands 48 --solve-ms 20000 (with --fire at the default --fire-ms and
    --keep-demos, so the --fire sweep can reuse its lines);
  * --direct: the same decks at the budgets under test (e.g. --solve-ms 2000 / 5000 / 10000, --no-fire).
    The solver splits --solve-ms between its phases in fixed proportions (probe 1/6, rollouts 0.7 of the
    rest, finisher 0.8 of the rest), so a smaller budget is NOT a truncation of a larger run: every
    budget is measured directly. Two runs at the same budget measure the run-to-run noise.
``fire`` re-runs the --fire variant of the solved hands of a run (its kept lines) at another --fire-ms and
writes one JSON line per hand (resumable). ``analyze`` prints markdown tables (docs/funnel.md, "预算研究")
and writes them as JSON: per-budget bias against the reference, the error of the brick rate split into
bias and binomial sampling noise per (hands, solve time), deck ranking and filter-verdict agreement
(first-n hands and a bootstrap), QD bin stability, --fire time, and the throughput of each configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ygorl.build.funnel import DEFAULT_BUDGET_S, FunnelFilter, fire_survives
from ygorl.cards.ydk import Deck

sys.path.insert(0, str(Path(__file__).resolve().parent))
from funnel_eval import read_results  # noqa: E402

HANDS = (6, 12, 24, 48)
BINS = (3, 5, 10)
CORES, DAY_S = 16, 86_400


# ------------------------------------------------------------------ fire sweep


def _fire_job(job: dict) -> dict:
    from ygorl.solver import Demonstration, HandJob, solve_fire

    scratch = Path(job["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    deck_path = scratch / f"{job['deck']}.ydk"
    deck_path.write_text(Deck(tuple(job["main"]), tuple(job["extra"]), (), job["deck"]).to_ydk(), encoding="utf-8")
    hj = HandJob(deck_path=deck_path, hand_index=job["index"], hand_seed=job["hand_seed"], targets=tuple(job["targets"]),
                 workdir=Path(job["workdir"]), scratch=scratch / "s", solve_ms=job["solve_ms"], threads=1,
                 fire_ms=job["fire_ms"], solver_seed=job["solver_seed"])  # fmt: skip
    rec = solve_fire(hj, Demonstration.from_json(job["demo"]), job["fire"]).to_json()
    solver = rec.get("solver", {})
    return {"deck": job["deck"], "index": job["index"], "fire": job["fire"], "fire_ms": job["fire_ms"],
            "status": rec["status"], "windows": solver.get("windows"), "converted": solver.get("converted"),
            "survives": fire_survives(rec), "wall_s": solver.get("wall_s", 0.0), "error": rec.get("error", "")}  # fmt: skip


def cmd_fire(args) -> int:
    from ygorl.solver import Workdir

    results = read_results(args.results)
    out = args.out or Path("out/budget") / f"fire_{args.fire_ms}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["deck"], r["index"], r["fire"]))
    scratch = Path(tempfile.mkdtemp(prefix="ygorl-budget-fire-"))
    workdir = Workdir.create(scratch / "workdir").path
    jobs = []
    for r in results:
        fires = args.fire or r.config.get("fire") or []
        for h in r.hands:
            if h.status != "solved" or h.index >= args.max_index:
                continue
            if not h.demos:
                print("funnel_budget: error: the run carries no solver records (--keep-demos)", file=sys.stderr)
                return 2
            plain = next(d for d in h.demos if d.get("variant") == "plain" and d.get("status") == "solved")
            for fire in fires:
                if (r.deck["name"], h.index, fire) in done:
                    continue
                jobs.append({"deck": r.deck["name"], "main": r.deck["main"], "extra": r.deck["extra"], "index": h.index,
                             "hand_seed": h.hand_seed, "targets": r.targets[h.target or 0], "demo": plain, "fire": fire,
                             "fire_ms": args.fire_ms, "solve_ms": r.config["solve_ms"],
                             "solver_seed": r.config.get("solver_seed"), "workdir": str(workdir),
                             "scratch": str(scratch / f"{r.deck['name']}_{h.index}_{fire}")})  # fmt: skip
    start = time.monotonic()
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool, open(out, "a", encoding="utf-8") as f:
        for n, rec in enumerate(pool.imap_unordered(_fire_job, jobs), start=1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[fire {args.fire_ms} ms {n}/{len(jobs)} {time.monotonic() - start:6.1f}s] {rec['deck']} hand {rec['index']}: "
                  f"{rec['status']} {rec['converted']}/{rec['windows']} {rec['wall_s']}s", flush=True)  # fmt: skip
    print(f"-> {out}")
    return 0


# ------------------------------------------------------------------ statistics


def _ranks(x: list[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    order = a.argsort(kind="stable")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):  # average ranks of ties
        m = a == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_b(x: list[float], y: list[float]) -> float:
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else math.nan


def binom_wrong_verdict(p: float, n: int, threshold: float = 0.5) -> float:
    """P(the n-hand brick rate lands on the other side of ``threshold`` than the true rate p)."""
    k_max = math.floor(threshold * n + 1e-9)  # pass: bricks <= k_max
    p_pass = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(k_max + 1))
    return 1 - p_pass if p <= threshold + 1e-9 else p_pass


def _bin(x: float, k: int) -> int:
    return min(k - 1, int(x * k + 1e-9))


# ------------------------------------------------------------------ loading


def load_run(path: Path) -> dict:
    """{deck: {index: hand}} of a funnel_eval output, with its config; a hand is a dict of the measured numbers."""
    results = read_results(path)
    if not results:
        raise SystemExit(f"funnel_budget: error: no funnel results in {path}")
    decks, config = {}, results[0].config
    for r in results:
        if r.stopped_early:
            raise SystemExit(f"funnel_budget: error: {path}: deck {r.deck['name']} stopped early (run with --no-filter)")
        hands = {}
        for h in r.hands:
            if h.status == "error":
                continue
            fire_s = sum(v.get("wall_s") or 0.0 for v in h.fire.values())
            surv = [v.get("survives") for v in h.fire.values()]
            hands[h.index] = {"brick": h.status == "brick", "plain_s": h.solver_s - fire_s, "fire_s": fire_s if h.fire else None,
                              "survives": (all(surv) if surv and None not in surv else None) if h.status == "solved" else False,
                              "combo": h.combo_actions}  # fmt: skip
        decks[r.deck["name"]] = hands
    return {"path": str(path), "solve_ms": config["solve_ms"], "fire_ms": config.get("fire_ms"),
            "fire": config.get("fire") or [], "decks": decks}  # fmt: skip


def load_fire(path: Path) -> dict:
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = defaultdict(dict)
    for r in recs:
        by[r["deck"]][r["index"]] = r
    return {"path": str(path), "fire_ms": recs[0]["fire_ms"] if recs else None, "decks": by}


# ------------------------------------------------------------------ analysis


def _rate(hands: dict, idx) -> float:
    idx = [i for i in idx if i in hands]
    return sum(hands[i]["brick"] for i in idx) / len(idx) if idx else math.nan


def _table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.3f}"
        return str(v)

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def analyze(args) -> int:
    ref = load_run(args.reference)
    runs = [load_run(p) for p in args.direct]
    fires = [load_fire(p) for p in args.fire_run]
    names = sorted(ref["decks"])
    ref_n = max(max(ref["decks"][d], default=-1) + 1 for d in names)  # hand indices 0..ref_n-1 (errors skipped)
    ref_p = {d: _rate(ref["decks"][d], range(ref_n)) for d in names}
    gate = FunnelFilter()
    rng = np.random.default_rng(args.seed)
    report: dict = {"reference": {"path": ref["path"], "solve_ms": ref["solve_ms"], "fire_ms": ref["fire_ms"], "hands": ref_n,
                                  "brick_rate": ref_p}}  # fmt: skip
    all_runs = [*runs, ref]
    budgets = sorted({r["solve_ms"] for r in all_runs})

    # -- 1. per budget: brick rate, false bricks against the reference, run-to-run noise, time per hand
    rows = []
    for run in all_runs:
        common = [(d, i) for d in names for i in run["decks"].get(d, {}) if i in ref["decks"][d] and i < ref_n]
        a = [run["decks"][d][i]["brick"] for d, i in common]
        b = [ref["decks"][d][i]["brick"] for d, i in common]
        bricks = [h["plain_s"] for d in names for h in run["decks"].get(d, {}).values() if h["brick"]]
        solved = [h["plain_s"] for d in names for h in run["decks"].get(d, {}).values() if not h["brick"]]
        rows.append({"run": Path(run["path"]).name, "solve_s": run["solve_ms"] / 1000, "hands": len(common),
                     "brick_rate": sum(a) / len(a), "ref_brick_rate": sum(b) / len(b),
                     "only_run_brick": sum(x and not y for x, y in zip(a, b)), "only_ref_brick": sum(y and not x for x, y in zip(a, b)),
                     "bias_pp": 100 * (sum(a) - sum(b)) / len(a), "brick_s": statistics.fmean(bricks) if bricks else math.nan,
                     "solved_s": statistics.fmean(solved) if solved else math.nan})  # fmt: skip
    report["per_budget"] = rows
    print("\n## per budget vs reference (same hands)\n")
    print(_table(rows, ["run", "solve_s", "hands", "brick_rate", "ref_brick_rate", "only_run_brick", "only_ref_brick", "bias_pp",
                        "brick_s", "solved_s"]))  # fmt: skip
    same = defaultdict(list)
    for run in all_runs:
        same[run["solve_ms"]].append(run)
    repeat = []
    for ms, rs in same.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                common = [(d, k) for d in names for k in rs[i]["decks"].get(d, {}) if k in rs[j]["decks"].get(d, {})]
                disc = sum(rs[i]["decks"][d][k]["brick"] != rs[j]["decks"][d][k]["brick"] for d, k in common)
                repeat.append({"solve_s": ms / 1000, "a": Path(rs[i]["path"]).name, "b": Path(rs[j]["path"]).name,
                               "hands": len(common), "discordant": disc})  # fmt: skip
    per_deck = []
    for d in names:
        hands = ref["decks"][d]
        combos = [h["combo"] for h in hands.values() if h["combo"] is not None]
        row = {"deck": d, "combo": statistics.fmean(combos) if combos else math.nan, "ref": ref_p[d]}
        for run in sorted(runs, key=lambda r: r["solve_ms"]):
            key = f"{run['solve_ms'] / 1000:g}s"
            if key in row:  # a repeat run of the same budget: the first one is shown
                continue
            common = [i for i in run["decks"].get(d, {}) if i in hands and i < ref_n]
            row[key] = _rate(run["decks"][d], common)
            row[f"false_{key}"] = sum(run["decks"][d][i]["brick"] and not hands[i]["brick"] for i in common)
            row[f"n_{key}"] = len(common)
        per_deck.append(row)
    report["per_deck"] = per_deck
    print("\n## per deck: brick rate per budget (hands the run has), false bricks vs the reference\n")
    print(_table(per_deck, list(per_deck[0])))
    false_cols = [k for k in per_deck[0] if k.startswith("false_")]
    for k in false_cols:
        n_k = "n_" + k.removeprefix("false_")
        rates = [r[k] / r[n_k] for r in per_deck]
        print(f"spearman(combo length, {k} rate) = {spearman([r['combo'] for r in per_deck], rates):.3f}")
    report["repeatability"] = repeat
    if repeat:
        print("\n## run-to-run (same budget, same hands)\n")
        print(_table(repeat, ["solve_s", "a", "b", "hands", "discordant"]))

    # -- 2. best-known check of the reference on the hands of the acceptance experiment
    if args.observers and args.observers.exists():
        obs = {}
        for line in args.observers.read_text(encoding="utf-8").splitlines():
            o = json.loads(line)
            obs[(o["deck"], o["index"])] = bool(o["line"]) or any(o["greedy"]) or o["explorer"]["reached"] > 0
        resolved = {}
        if args.re_solved and args.re_solved.exists():
            for line in args.re_solved.read_text(encoding="utf-8").splitlines():
                o = json.loads(line)
                resolved[(o["deck"], o["index"])] = o["status"] == "solved"
        rb = [(d, i) for d in names for i, h in ref["decks"][d].items() if (d, i) in obs and h["brick"]]
        missed = [(d, i) for d, i in rb if obs[(d, i)] or resolved.get((d, i))]
        report["reference_check"] = {"hands": sum((d, i) in obs for d in names for i in ref["decks"][d]),
                                     "reference_bricks": len(rb), "reached_by_others": missed}  # fmt: skip
        print(f"\nreference bricks on the {report['reference_check']['hands']} acceptance hands: {len(rb)}; "
              f"reached by real first turns or the 30 s re-solve: {missed}")  # fmt: skip

    # -- 3. error of the brick rate per (hands, solve time): bias + binomial noise; ranking; verdicts; bins
    per_ms = {ms: same[ms][0] for ms in budgets}  # first run of each budget
    cost_fire = {d: statistics.fmean([h["fire_s"] for h in ref["decks"][d].values() if h["fire_s"] is not None] or [0.0])
                 for d in names}  # fmt: skip
    grid = []
    for ms in budgets:
        run = per_ms[ms]
        avail = max(max(run["decks"].get(d, {}), default=-1) + 1 for d in names)
        p_t = {d: _rate(run["decks"][d], range(avail)) for d in names}
        bias = {d: p_t[d] - _rate(ref["decks"][d], range(avail)) for d in names}  # on the same hands
        plain = {d: statistics.fmean([h["plain_s"] for h in run["decks"][d].values()]) for d in names}
        for n in HANDS:
            if n > ref_n:
                continue
            sd = [math.sqrt(ref_p[d] * (1 - ref_p[d]) / n) for d in names]
            rmse = math.sqrt(statistics.fmean([bias[d] ** 2 + s**2 for d, s in zip(names, sd)]))
            cost = [n * (plain[d] + (1 - p_t[d]) * cost_fire[d]) for d in names]
            row = {"solve_s": ms / 1000, "hands": n, "bias_pp": 100 * statistics.fmean(bias.values()),
                   "sd_pp": 100 * statistics.fmean(sd), "rmse_pp": 100 * rmse, "cost_s": statistics.fmean(cost),
                   "max_cost_s": max(cost), "decks_per_day": CORES * DAY_S / statistics.fmean(cost)}  # fmt: skip
            if n <= avail:  # the funnel's own hands: the first n seeds
                first = {d: _rate(run["decks"][d], range(n)) for d in names}
                row["spearman_first"] = spearman([first[d] for d in names], [ref_p[d] for d in names])
                row["kendall_first"] = kendall_b([first[d] for d in names], [ref_p[d] for d in names])
                row["verdict_first"] = sum(gate.hopeless(round(first[d] * n), n) == gate.hopeless(round(ref_p[d] * ref_n), ref_n)
                                           for d in names) / len(names)  # fmt: skip
            # bootstrap: n hands drawn with replacement from this budget's hands of each deck
            sp, agree, same_bin = [], [], {k: [] for k in BINS}
            arrays = {d: np.array([h["brick"] for h in run["decks"][d].values()], dtype=float) for d in names}
            for _ in range(args.bootstrap):
                est = {d: float(rng.choice(arrays[d], n).mean()) for d in names}
                sp.append(spearman([est[d] for d in names], [ref_p[d] for d in names]))
                agree.append(statistics.fmean([(est[d] > 0.5 + 1e-9) == (ref_p[d] > 0.5 + 1e-9) for d in names]))
                for k in BINS:
                    same_bin[k].append(statistics.fmean([_bin(est[d], k) == _bin(ref_p[d], k) for d in names]))
            sp = [s for s in sp if not math.isnan(s)]
            row["spearman_boot"] = statistics.median(sp) if sp else math.nan
            row["verdict_boot"] = statistics.fmean(agree)
            for k in BINS:
                row[f"same_bin_{k}"] = statistics.fmean(same_bin[k])
            grid.append(row)
    report["grid"] = grid
    print("\n## brick rate per (solve time, hands): error, ranking, verdict, bins, cost\n")
    print(_table(grid, ["solve_s", "hands", "bias_pp", "sd_pp", "rmse_pp", "spearman_first", "kendall_first", "spearman_boot",
                        "verdict_first", "verdict_boot", *[f"same_bin_{k}" for k in BINS], "cost_s", "max_cost_s",
                        "decks_per_day"]))  # fmt: skip

    # -- 4. analytic: wrong filter verdict and noise for a true brick rate p
    ana = []
    for p in (0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8):
        ana.append({"p": p, **{f"n{n}": binom_wrong_verdict(p, n) for n in HANDS}})
    report["wrong_verdict"] = ana
    print("\n## P(wrong filter verdict at brick <= 0.5) for a true brick rate p (binomial)\n")
    print(_table(ana, ["p", *[f"n{n}" for n in HANDS]]))
    between = statistics.pvariance(list(ref_p.values()))
    rel = []
    for n in HANDS:
        noise = statistics.fmean([ref_p[d] * (1 - ref_p[d]) / n for d in names])
        rel.append({"hands": n, "se_at_0.4": math.sqrt(0.24 / n), "ci95_halfwidth_at_0.4": 1.96 * math.sqrt(0.24 / n),
                    "expected_corr": math.sqrt(between / (between + noise))})  # fmt: skip
    report["reliability"] = {"between_deck_sd": math.sqrt(between), "rows": rel}
    print(f"\nbetween-deck SD of the reference brick rate: {math.sqrt(between):.3f}")
    print(_table(rel, ["hands", "se_at_0.4", "ci95_halfwidth_at_0.4", "expected_corr"]))

    # -- 5. hand-trap survival and combo length (reference solve time, --fire at the reference fire_ms)
    if ref["fire"]:
        surv_ref = {d: statistics.fmean([bool(h["survives"]) for h in ref["decks"][d].values()]) for d in names}
        combo_ref = {d: statistics.fmean([h["combo"] for h in ref["decks"][d].values() if h["combo"] is not None]) for d in names}
        desc = []
        for n in HANDS:
            if n > ref_n:
                continue
            s_first = {d: statistics.fmean([bool(ref["decks"][d][i]["survives"]) for i in range(n) if i in ref["decks"][d]])
                       for d in names}
            c_first = {d: statistics.fmean([ref["decks"][d][i]["combo"] for i in range(n)
                                            if i in ref["decks"][d] and ref["decks"][d][i]["combo"] is not None]
                                           or [math.nan]) for d in names}  # fmt: skip
            desc.append({"hands": n, "survival_spearman": spearman([s_first[d] for d in names], [surv_ref[d] for d in names]),
                         "survival_mae": statistics.fmean([abs(s_first[d] - surv_ref[d]) for d in names]),
                         "combo_spearman": spearman([c_first[d] for d in names], [combo_ref[d] for d in names]),
                         "combo_mae": statistics.fmean([abs(c_first[d] - combo_ref[d]) for d in names])})  # fmt: skip
        report["descriptors"] = {"survival_ref": surv_ref, "combo_ref": combo_ref, "rows": desc}
        print("\n## descriptors: first n hands vs all reference hands (reference solve time)\n")
        print(_table(desc, ["hands", "survival_spearman", "survival_mae", "combo_spearman", "combo_mae"]))

    # -- 6. --fire time: survival verdicts per fire_ms on the same lines
    if fires:
        fruns = sorted(fires, key=lambda f: f["fire_ms"])
        top = fruns[-1]
        keys = [(d, i) for d in top["decks"] for i in top["decks"][d]]
        frows = []
        for f in fruns:
            common = [k for k in keys if k[1] in f["decks"].get(k[0], {})]
            sv = [bool(f["decks"][d][i]["survives"]) for d, i in common]
            tv = [bool(top["decks"][d][i]["survives"]) for d, i in common]
            per_deck = {d: statistics.fmean([bool(f["decks"][d][i]["survives"]) for dd, i in common if dd == d]) for d in names
                        if any(dd == d for dd, _ in common)}  # fmt: skip
            top_deck = {d: statistics.fmean([bool(top["decks"][d][i]["survives"]) for dd, i in common if dd == d]) for d in per_deck}
            frows.append({"fire_s": f["fire_ms"] / 1000, "lines": len(common), "survive_rate": statistics.fmean(sv),
                          "only_here": sum(a and not b for a, b in zip(sv, tv)), "only_top": sum(b and not a for a, b in zip(sv, tv)),
                          "spearman_vs_top": spearman(list(per_deck.values()), list(top_deck.values())),
                          "fire_s_per_line": statistics.fmean([f["decks"][d][i]["wall_s"] for d, i in common])})  # fmt: skip
        report["fire"] = frows
        print(f"\n## --fire time (same lines; top = {top['fire_ms'] / 1000:.0f} s)\n")
        print(_table(frows, ["fire_s", "lines", "survive_rate", "only_here", "only_top", "spearman_vs_top", "fire_s_per_line"]))

    # -- 7. throughput with early stopping (sequential hands; a brick costs the full solve time)
    stop = []
    for ms in budgets:
        run = per_ms[ms]
        brick_s = statistics.fmean([h["plain_s"] for d in names for h in run["decks"][d].values() if h["brick"]])
        solved_s = statistics.fmean([h["plain_s"] for d in names for h in run["decks"][d].values() if not h["brick"]])
        fire_s = statistics.fmean(cost_fire.values())
        for n in (12, 24):
            row = {"solve_s": ms / 1000, "hands": n}
            for p in (0.4, 0.7, 0.9):
                costs = []
                for _ in range(2000):
                    c = b = 0
                    for _h in range(n):
                        if rng.random() < p:
                            b += 1
                            c += brick_s
                        else:
                            c += solved_s + fire_s
                        if gate.hopeless(b, n):
                            break
                    costs.append(c)
                row[f"cost_p{p}"] = statistics.fmean(costs)
            stop.append(row)
    report["early_stop"] = stop
    print("\n## expected solver-seconds per deck with early stopping, by true brick rate p\n")
    print(_table(stop, ["solve_s", "hands", "cost_p0.4", "cost_p0.7", "cost_p0.9"]))
    print(f"\nbudget {args.budget:.0f} s/deck -> {CORES * DAY_S / args.budget:,.0f} decks/day on {CORES} cores")

    out = args.out or args.reference.with_name("study.json")
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"-> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fire", help="re-run --fire on the solved hands of a run at another --fire-ms")
    f.add_argument("results", type=Path, help="tools/funnel_eval.py output with --keep-demos")
    f.add_argument("--fire-ms", type=int, required=True)
    f.add_argument("--fire", type=int, action="append", default=None, metavar="PASSWORD", help="default: the run's")
    f.add_argument("--max-index", type=int, default=12, help="only hands with index below this")
    f.add_argument("--workers", type=int, default=2)
    f.add_argument("--out", type=Path, default=None)
    a = sub.add_parser("analyze", help="tables of the budget study")
    a.add_argument("reference", type=Path, help="large-budget tools/funnel_eval.py output (--no-filter)")
    a.add_argument("--direct", type=Path, action="append", default=[], help="run at a budget under test (repeatable)")
    a.add_argument("--fire-run", type=Path, action="append", default=[], help="output of the fire command (repeatable)")
    a.add_argument("--observers", type=Path, default=None, help="validate_funnel.py real-first-turn cache")
    a.add_argument("--re-solved", type=Path, default=None, help="validate_funnel.py reference re-solve cache")
    a.add_argument("--budget", type=float, default=DEFAULT_BUDGET_S)
    a.add_argument("--bootstrap", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    return cmd_fire(args) if args.cmd == "fire" else analyze(args)


if __name__ == "__main__":
    sys.exit(main())
