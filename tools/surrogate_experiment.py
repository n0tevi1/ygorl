"""Label genotypes with real games and measure the surrogate's held-out error (T5.7 acceptance).

Usage: uv run python tools/surrogate_experiment.py [--n-train 500] [--n-test 100] [--train-pairs 2]
           [--test-pairs 10] [--workers 4] [--seed 2026] [--out out/surrogate] [--analyze-only]

Format (no environment): the 10 test decks in tests/decks are the meta pool. The genotype space's
packages are the 10 decks' card sets plus the ``--top-packages`` best engine packages of the
synergy graph; generic cards come from tests/data/generic_pool.json. Candidates are a mix of random
genotypes (share ``--random-share``) and mutants of the meta decks (``from_deck`` + 1..``--max-ops``
mutation operators), so labels range from near-meta decks to incoherent piles. Candidate ``i`` of a
split is drawn from its own seed, so a larger run extends a smaller one and reuses its cache.

Labels (``ygorl.build.labels.label_decks``, cached in ``<out>/labels.jsonl``): each candidate
(``--agent``) plays ``2 * pairs`` paired games against every meta deck (``--opponent``):

* train: ``--train-pairs`` per opponent (default 40 games per candidate), seed S;
* test: the held-out candidates with ``--test-pairs`` per opponent (default 200 games), seed S + 1;
  these near-noise-free labels are the reference for the held-out error;
* retest: the same held-out candidates at the train budget with seed S + 2, i.e. what one cheap
  real evaluation would say -- the label-noise floor the surrogate is compared with.

The analysis fits the surrogate on the train split and reports the held-out MAE (percentage points)
per target against the test labels, next to baselines (train mean, cheap retest), feature ablations,
a learning curve, uncertainty calibration and a DSA-ME acquisition simulation; ``<out>/results.json``
keeps the numbers.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # see tests/conftest.py
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from ygorl.build.genotype import GenotypeSpace  # noqa: E402
from ygorl.build.labels import LabelCache, label_decks  # noqa: E402
from ygorl.build.packages import enumerate_packages, setcodes_from_db  # noqa: E402
from ygorl.build.surrogate import DEFAULT_TARGETS, FeatureMap, RidgeEnsemble, acquire  # noqa: E402
from ygorl.build.synergy_graph import load_or_build  # noqa: E402
from ygorl.cards.cdb import CardDB  # noqa: E402
from ygorl.cards.ydk import load_ydk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": 0, "test": 1}


def build_space(db, top_packages: int):
    meta = [load_ydk(p) for p in sorted((ROOT / "tests" / "decks").glob("*.ydk"))]
    generic = {c["password"]: c["role"] for c in json.loads((ROOT / "tests" / "data" / "generic_pool.json").read_text())["cards"]}
    syn = enumerate_packages(load_or_build(), setcodes=setcodes_from_db(db))[:top_packages]
    deck_pkgs = [sorted(set(d.main) | set(d.extra)) for d in meta]
    return GenotypeSpace(db, deck_pkgs + list(syn), generic, max_packages=3), meta


def candidates(sp: GenotypeSpace, meta, split: str, n: int, seed: int, random_share: float, max_ops: int):
    out = []
    for i in range(n):
        rng = np.random.default_rng([seed, SPLITS[split], i])
        if rng.random() < random_share:
            g, origin = sp.sample(rng), "random"
        else:
            j = int(rng.integers(len(meta)))
            k = int(rng.integers(1, max_ops + 1))
            g, origin = sp.mutate(sp.from_deck(meta[j], rng), rng, n_ops=k), f"{meta[j].name}+{k}"
        out.append((g, origin))
    return out


# ------------------------------------------------------------------ metrics


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_predict(fm: FeatureMap, c_train, y_train, w_train, c_test, members: int = 16, seed: int = 0):
    model = RidgeEnsemble(n_members=members, seed=seed).fit(fm.transform(c_train), y_train, w_train)
    mean, std = model.predict(fm.transform(c_test))
    return np.clip(mean, 0, 1), std, model


def label_matrix(recs, targets=DEFAULT_TARGETS):
    return np.array([[r["labels"][t] for t in targets] for r in recs])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--train-pairs", type=int, default=2)
    ap.add_argument("--test-pairs", type=int, default=10)
    ap.add_argument("--agent", default="greedy")
    ap.add_argument("--opponent", default="greedy")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--top-packages", type=int, default=20)
    ap.add_argument("--random-share", type=float, default=0.4)
    ap.add_argument("--max-ops", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "surrogate")
    ap.add_argument("--analyze-only", action="store_true", help="use cached labels only; skip unlabelled candidates")
    args = ap.parse_args()

    t0 = time.perf_counter()
    db = CardDB.load()
    sp, meta = build_space(db, args.top_packages)
    fm_full = FeatureMap(sp, db)
    print(f"space: {len(sp)} cards ({sp.n_main} Main / {len(sp) - sp.n_main} Extra), {len(sp.packages)} packages, "
          f"{int(sp.is_generic.sum())} generic; features {fm_full.dim} "
          + ", ".join(f"{g} {s.stop - s.start}" for g, s in fm_full.groups.items()) + f" ({time.perf_counter() - t0:.1f}s)")  # fmt: skip

    train = candidates(sp, meta, "train", args.n_train, args.seed, args.random_share, args.max_ops)
    test = candidates(sp, meta, "test", args.n_test, args.seed, args.random_share, args.max_ops)
    cache = LabelCache(args.out / "labels.jsonl")
    common = dict(agent=args.agent, opponent=args.opponent, max_turns=args.max_turns, workers=args.workers, cache=cache)
    games_played = {}

    def run(name, cands, pairs, seed):
        decks = [sp.decode(g, name=f"{name}{i}") for i, (g, _) in enumerate(cands)]
        if args.analyze_only:
            from ygorl.build.labels import LabelConfig, deck_fingerprint

            cfg = LabelConfig.for_pool(meta, agent=args.agent, opponent=args.opponent, pairs=pairs, seed=seed,
                                       max_turns=args.max_turns).fingerprint()  # fmt: skip
            recs = [cache.get(deck_fingerprint(d), cfg) for d in decks]
            keep = [i for i, r in enumerate(recs) if r is not None]
            return [recs[i] for i in keep], keep
        start = time.perf_counter()
        before = len(cache)

        def progress(done, total):
            el = time.perf_counter() - start
            print(f"  {name}: {done}/{total} new candidates labelled, {el:.0f}s", flush=True)

        recs = label_decks(decks, meta, pairs=pairs, seed=seed, progress=progress, chunk=16, **common)
        new = len(cache) - before
        el = time.perf_counter() - start
        games_played[name] = (new * 2 * pairs * len(meta), el)
        if new:
            print(f"{name}: {new} candidates x {2 * pairs * len(meta)} games in {el:.0f}s "
                  f"({new * 2 * pairs * len(meta) / max(el, 1e-9):.1f} games/s)", flush=True)  # fmt: skip
        return recs, list(range(len(decks)))

    # held-out labels first: an interrupted run can be analysed (--analyze-only) with a partial train split
    te_recs, te_idx = run("test", test, args.test_pairs, args.seed + 1)
    rt_recs, rt_idx = run("retest", test, args.train_pairs, args.seed + 2)
    tr_recs, tr_idx = run("train", train, args.train_pairs, args.seed)
    seen = {r["deck"] for r in tr_recs}
    te_fp = {i: r["deck"] for i, r in zip(te_idx, te_recs)}
    common_te = sorted(i for i in set(te_idx) & set(rt_idx) if te_fp[i] not in seen)  # held out = not in train
    if len(common_te) < len(set(te_idx) & set(rt_idx)):
        print(f"dropped {len(set(te_idx) & set(rt_idx)) - len(common_te)} test candidates that also occur in train")
    te_recs = [te_recs[te_idx.index(i)] for i in common_te]
    rt_recs = [rt_recs[rt_idx.index(i)] for i in common_te]
    train_g = [train[i][0] for i in tr_idx]
    test_g = [test[i][0] for i in common_te]
    if len(train_g) < 20 or len(test_g) < 5:
        print("not enough labelled candidates for the analysis")
        return 1

    targets = list(DEFAULT_TARGETS)
    y_tr, y_te, y_rt = label_matrix(tr_recs), label_matrix(te_recs), label_matrix(rt_recs)
    c_tr = np.stack([sp.vector(g) for g in train_g]).astype(np.int16)
    c_te = np.stack([sp.vector(g) for g in test_g]).astype(np.int16)
    res: dict = {"n_train": len(train_g), "n_test": len(test_g), "targets": targets}

    # -- labels and noise
    se_tr = np.array([r["labels"]["se"] for r in tr_recs])
    hw_tr = np.array([r["labels"]["ci_half_width"] for r in tr_recs])
    se_te = np.array([r["labels"]["se"] for r in te_recs])
    hw_te = np.array([r["labels"]["ci_half_width"] for r in te_recs])
    origins = [train[i][1] for i in tr_idx]
    rnd = np.array([o == "random" for o in origins])
    print(f"\nlabels: train {len(train_g)} x {tr_recs[0]['labels']['games']} games, "
          f"test {len(test_g)} x {te_recs[0]['labels']['games']} (+ retest x {rt_recs[0]['labels']['games']})")  # fmt: skip
    for j, t in enumerate(targets):
        v = y_tr[:, j]
        print(f"  train {t:16s} mean {v.mean():.3f} sd {v.std():.3f} quartiles "
              + " ".join(f"{q:.3f}" for q in np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])))  # fmt: skip
    print(f"  train win_rate: random genotypes {y_tr[rnd, 0].mean():.3f} (n={rnd.sum()}), "
          f"meta mutants {y_tr[~rnd, 0].mean():.3f} (n={(~rnd).sum()})")  # fmt: skip
    true_var = float(y_tr[:, 0].var() - np.mean(se_tr**2))
    print(f"  noise: train se {se_tr.mean():.3f} (Wilson half-width {hw_tr.mean():.3f}); test se {se_te.mean():.3f} "
          f"(half-width {hw_te.mean():.3f}); train label var {y_tr[:, 0].var():.4f} = signal {true_var:.4f} + noise "
          f"{np.mean(se_tr ** 2):.4f}")  # fmt: skip
    res["labels"] = {
        "train_mean": y_tr.mean(0).tolist(), "train_sd": y_tr.std(0).tolist(), "test_sd": y_te.std(0).tolist(),
        "train_se": float(se_tr.mean()), "train_half_width": float(hw_tr.mean()), "test_se": float(se_te.mean()),
        "test_half_width": float(hw_te.mean()), "signal_var": true_var, "noise_var": float(np.mean(se_tr**2)),
        "random_mean": float(y_tr[rnd, 0].mean()), "mutant_mean": float(y_tr[~rnd, 0].mean()),
        "games": {k: list(v) for k, v in games_played.items()},
    }  # fmt: skip

    # -- held-out error
    rows = []

    def report(name, pred):
        errs = [100 * mae(pred[:, j], y_te[:, j]) for j in range(len(targets))]
        rm = [100 * rmse(pred[:, j], y_te[:, j]) for j in range(len(targets))]
        # the test labels' own noise inflates the error; subtract it in quadrature for the RMSE vs truth
        corr = [100 * float(np.sqrt(max(rmse(pred[:, j], y_te[:, j]) ** 2 - np.mean(se_te**2), 0))) for j in range(len(targets))]
        sp_rho = spearman(pred[:, 0], y_te[:, 0]) if np.ptp(pred[:, 0]) > 0 else float("nan")
        rows.append({"model": name, "mae_pp": errs, "rmse_pp": rm, "rmse_vs_truth_pp": corr, "spearman": sp_rho})
        print(f"  {name:34s} MAE " + " / ".join(f"{e:5.2f}" for e in errs) + " pp   RMSE " + " / ".join(f"{e:5.2f}" for e in rm)
              + f"   (win_rate RMSE vs truth ~{corr[0]:.2f})   Spearman {sp_rho:.3f}")  # fmt: skip

    print(f"\nheld-out error vs the {te_recs[0]['labels']['games']}-game labels ({len(test_g)} candidates); "
          f"targets {' / '.join(targets)}")  # fmt: skip
    report("constant (train mean)", np.tile(y_tr.mean(0), (len(test_g), 1)))
    report(f"one cheap evaluation ({rt_recs[0]['labels']['games']} games)", y_rt)
    variants = {
        "counts": FeatureMap(sp, groups=("counts",)),
        "counts+packages": FeatureMap(sp, groups=("counts", "packages")),
        "counts+packages+roles": FeatureMap(sp, groups=("counts", "packages", "roles")),
        "structure+roles+packages": FeatureMap(sp, db, groups=("packages", "structure", "roles")),
        "all (counts+packages+structure+roles)": fm_full,
    }
    preds = {}
    for name, fm in variants.items():
        mean, std, model = fit_predict(fm, c_tr, y_tr, None, c_te)
        preds[name] = (mean, std, model)
        report(f"surrogate: {name}", mean)
    res["held_out"] = rows

    # -- uncertainty
    mean, std, model = preds["all (counts+packages+structure+roles)"]
    err = np.abs(mean[:, 0] - y_te[:, 0])
    total_sd = np.sqrt(std[:, 0] ** 2 + model.residual_std_[0] ** 2 - np.mean(se_tr**2) + np.mean(se_te**2))
    cover = float(np.mean(err <= 1.96 * total_sd))
    rho = spearman(std[:, 0], err)
    print(f"\nuncertainty (full model, win_rate): ensemble sd mean {std[:, 0].mean():.3f}; residual sd (GCV) "
          f"{model.residual_std_[0]:.3f}; 95% predictive interval coverage {cover:.2f}; Spearman(sd, |error|) {rho:.3f}; "
          f"alpha {model.alpha_[0]:.1f}")  # fmt: skip
    res["uncertainty"] = {"ensemble_sd": float(std[:, 0].mean()), "residual_sd": float(model.residual_std_[0]),
                          "coverage95": cover, "spearman_sd_error": rho}  # fmt: skip

    te_rnd = np.array([test[i][1] == "random" for i in common_te])
    by_origin = {}
    for label, mask in (("random", te_rnd), ("mutant", ~te_rnd)):
        if mask.any():
            by_origin[label] = (int(mask.sum()), 100 * mae(mean[mask, 0], y_te[mask, 0]), float(y_te[mask, 0].mean()))
    print("  win_rate MAE by origin: " + ", ".join(f"{k} {e:.2f}pp (n={n}, mean label {m:.3f})" for k, (n, e, m) in by_origin.items()))
    res["by_origin"] = by_origin

    # -- learning curve
    print("\nlearning curve (full features, win_rate MAE pp, 5 random subsets each):")
    curve = []
    for n in [x for x in (25, 50, 100, 200, 300, 400) if x < len(train_g)] + [len(train_g)]:
        errs = []
        for r in range(5 if n < len(train_g) else 1):
            idx = np.random.default_rng(r).choice(len(train_g), n, replace=False)
            m, _, _ = fit_predict(fm_full, c_tr[idx], y_tr[idx], None, c_te, members=4)
            errs.append(100 * mae(m[:, 0], y_te[:, 0]))
        curve.append((n, float(np.mean(errs))))
        print(f"  n={n:4d}: {np.mean(errs):.2f}")
    res["learning_curve"] = curve

    # -- DSA-ME acquisition simulation on the labelled train pool
    start_n = min(50, len(train_g) // 2)
    print(f"\nacquisition simulation (pool = train split, start {start_n}, +25 per round; win_rate):")
    sim = {}
    for rule in ("random", "uncertainty", "ucb"):
        rng = np.random.default_rng(0)
        chosen = [int(i) for i in rng.choice(len(train_g), start_n, replace=False)]
        trace = []
        while True:
            m, s, _ = fit_predict(fm_full, c_tr[chosen], y_tr[chosen], None, np.vstack([c_te, c_tr]), members=8)
            trace.append((len(chosen), 100 * mae(m[: len(test_g), 0], y_te[:, 0]), float(y_tr[chosen, 0].mean())))
            rest = np.setdiff1d(np.arange(len(train_g)), chosen)
            if len(chosen) >= min(300, len(train_g)) or not len(rest):
                break
            mp, sp_ = m[len(test_g) :][rest, 0], s[len(test_g) :][rest, 0]
            if rule == "random":
                pick = rng.choice(rest, min(25, len(rest)), replace=False)
            elif rule == "uncertainty":
                pick = rest[acquire(mp, sp_, 25, beta=0.0, explore=1.0)]
            else:
                pick = rest[acquire(mp, sp_, 25, beta=1.0)]
            chosen += [int(i) for i in pick]
        sim[rule] = trace
        print(f"  {rule:12s} " + "  ".join(f"n={n}: {e:.2f}pp (mean label {mw:.3f})" for n, e, mw in trace[:: 2]))
    res["acquisition"] = sim

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res["meta"] = {"commit": commit, "date": date.today().isoformat(), "machine": f"{platform.machine()} {os.cpu_count()} cpus",
                   "args": {k: str(v) for k, v in vars(args).items()}}  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"\nresults written to {args.out / 'results.json'} ({time.perf_counter() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
