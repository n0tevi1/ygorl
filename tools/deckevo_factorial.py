"""Games per detected +2 pp effect: fractional-factorial evaluation of edit groups against one-at-a-time screening,
on real decks (#152, ygorl.build.factorial, docs/tuning.md「析因评估」).

Parents are ``N_DECKS`` legal training-split corpus lists drawn with ``--seed`` (as tools/deckevo_eval_compare.py).
Per parent, ``--k`` random legal single swaps from the tuner's tech pool, each on its own card copy
(``ygorl.build.factorial.place``), then two arms at the same game count, each on its own evaluator seed:

- **factorial**: the 2^(k−p) design (``--p``; default resolution IV where the runs allow) on ``--pairs`` pairs; main
  effects and interaction chains from the regression with a fixed effect per pair and cluster-robust standard errors;
- **one at a time**: the parent and each single edit on round(runs × pairs / (k + 1)) pairs; paired differences.

Per arm: the effects with standard errors, the games, and the **games per detected +2 pp effect** per edit: the games
at which a true +2 pp is detected (one-sided α = 0.025, power 0.8), scaled from the measured standard error as
1 / √games and divided by k (every edit of the arm shares its games). ``parent_free`` does the same without the
parent's games (run 0 of the design; the parent of one-at-a-time), for when the parent's pairs are already played
(e.g. by the diagnosis). Pooled over decks by the mean squared standard error. The variance ratio is one-at-a-time
over factorial at equal games (synthetic: 2.4 at k = 4, theory (k + 1) / 2).

Usage: tools/deckevo_factorial.py CHECKPOINT N_DECKS OUT.json [--env md-2026-09] [--device cuda] [--k 4] [--p P]
       [--pairs 200] [--envs 256] [--threads 8] [--seed 0]"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ygorl.build.factorial import evaluate_edits, games_to_detect, one_at_a_time, place, summarize
from ygorl.build.tuner import PairedEvaluator, neighbors, tech_pool
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.train.checkpoint import load_actor, torch_device


def pooled(ses, games, k, parent_games=0):
    """Games per edit to detect +2 pp at the mean squared standard error (all edits share the arm's games)."""
    se = math.sqrt(float(np.mean(np.square(ses))))
    return {"stderr": se, "games_per_edit": games_to_detect(se, games) / k,
            "parent_free": games_to_detect(se, games - parent_games) / k if parent_games else None}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("n_decks", type=int)
    ap.add_argument("out")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=4, help="edits per design")
    ap.add_argument("--p", type=int, default=None, help="fraction 2^(k-p) (default: resolution IV where possible)")
    ap.add_argument("--pairs", type=int, default=200, help="pairs every design variant plays")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch_device(args.device)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    names = {pw: c.name for pw, c in cards.items()}
    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], e["file"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]
    meta = [m.deck for m in env.meta_decks]
    weights = [m.share for m in env.meta_decks]
    pol = load_actor(args.checkpoint)
    net = pol.net.to(device)

    def is_extra(pw):
        return cards[pw].is_extra_deck if pw in cards else False

    def legal(d):
        return not env.validate_deck(d, cards)

    def evaluator(seed):
        return PairedEvaluator(
            env_factory=lambda n: EncodedVecEnv(n, args.threads, cards=cards, vocab=pol.vocab,
                                                event_length=pol.event_length, skip_forced=True),
            policy=net, opponents=meta, weights=weights, seed=seed, device=args.device, num_envs=args.envs)  # fmt: skip

    train = {p.name for p in Path("out/corpus/train").glob("*.ydk")}
    bases = [t for t in lists if Path(t[1]).name in train and not env.validate_deck(t[2], cards)]
    picks = [bases[i] for i in np.random.default_rng(args.seed).choice(len(bases), args.n_decks, replace=False)]
    res = {"checkpoint": args.checkpoint, "environment": env.version, "args": vars(args), "parents": []}
    for di, (dtype, file, base) in enumerate(picks):
        pool = tech_pool(base, [d for t, _, d in lists if t == dtype and d is not base], meta)
        cands = [e for e, _ in neighbors(base, pool, is_extra, legal, limit=8 * args.k,
                                         rng=np.random.default_rng([args.seed, di]))]  # fmt: skip
        edits = [pl.edit for pl in place(base, cands)[0]][: args.k]
        fe = evaluator(derive(args.seed, di, 1))
        fr = evaluate_edits(base, edits, fe, legal=legal, pairs=args.pairs, p=args.p)
        edits = fr.edits  # the ones the design kept
        k, runs = len(edits), fr.design.runs
        oat_pairs = round(runs * args.pairs / (k + 1))
        oe = evaluator(derive(args.seed, di, 2))
        oat = one_at_a_time(base, edits, oe, pairs=oat_pairs)
        f_ses = [e.stderr for e in fr.main()]
        o_ses = [se for _, se in oat]
        row = {"type": dtype, "file": file, "k": k, "edits": [e.describe(names) for e in edits],
               "factorial": {**fr.to_dict(), "errors": fe.errors, "seconds": fe.seconds,
                             **pooled(f_ses, fe.games, k, 2 * args.pairs)},
               "one_at_a_time": {"pairs": oat_pairs, "games": oe.games, "errors": oe.errors, "seconds": oe.seconds,
                                 "effects": [{"edit": e.describe(names), "diff": m, "stderr": se}
                                             for e, (m, se) in zip(edits, oat, strict=True)],
                                 **pooled(o_ses, oe.games, k, 2 * oat_pairs)}}  # fmt: skip
        row["variance_ratio"] = (row["one_at_a_time"]["stderr"] ** 2 * oe.games) / (
            row["factorial"]["stderr"] ** 2 * fe.games)  # fmt: skip
        res["parents"].append(row)
        print(f"{dtype}: {summarize(fr, names)}", flush=True)
        for e, (m, se) in zip(edits, oat, strict=True):
            print(f"  one at a time {e.describe(names)}: {m:+.4f} ± {se:.4f}", flush=True)
        print(f"  games per detected +2 pp per edit: factorial {row['factorial']['games_per_edit']:.0f} "
              f"({fe.games} games), one at a time {row['one_at_a_time']['games_per_edit']:.0f} ({oe.games} games); "
              f"variance ratio {row['variance_ratio']:.2f}", flush=True)  # fmt: skip
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    rows = res["parents"]
    if rows:
        k = rows[0]["k"]
        f_all = [e["stderr"] for r in rows for e in r["factorial"]["effects"] if len(e["term"]) == 1]
        o_all = [e["stderr"] for r in rows for e in r["one_at_a_time"]["effects"]]
        fg = float(np.mean([r["factorial"]["games"] for r in rows]))
        og = float(np.mean([r["one_at_a_time"]["games"] for r in rows]))
        res["pooled"] = {"factorial": pooled(f_all, fg, k, 2 * args.pairs),
                         "one_at_a_time": pooled(o_all, og, k, 2 * round(rows[0]["factorial"]["design"]["runs"]
                                                                          * args.pairs / (k + 1)))}  # fmt: skip
        pf, po = res["pooled"]["factorial"], res["pooled"]["one_at_a_time"]
        res["pooled"]["variance_ratio"] = po["stderr"] ** 2 * og / (pf["stderr"] ** 2 * fg)
        print(f"pooled over {len(rows)} decks: games per detected +2 pp per edit: factorial {pf['games_per_edit']:.0f}"
              f" (parent free {pf['parent_free']:.0f}), one at a time {po['games_per_edit']:.0f} (parent free "
              f"{po['parent_free']:.0f}); variance ratio {res['pooled']['variance_ratio']:.2f}")  # fmt: skip
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


def derive(seed, deck, arm):
    return (seed * 1_000 + deck) * 10 + arm


if __name__ == "__main__":
    main()
