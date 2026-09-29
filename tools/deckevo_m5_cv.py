"""M5 (docs/spikes/deck-evolution.md): does the critic control variate lower the variance of paired differences?
Same decks and children as M2 (tools/deckevo_m2_loo.py: one copy of a card replaced by a fixed blank), parent and
children on the same common-random-number pairs against the environment's meta decks by share. Every game also gets
its opening luck (ygorl.build.control): the privileged critic's V at the first decision of the actual deal minus its
mean over K other shuffles of the same deck, each deck with its own cards. Per child and pooled, compare the variance
of the per-pair difference child - parent before and after subtracting beta * luck, with two coefficients that keep
the estimate unbiased because neither is fitted on the games it adjusts:
- ``--beta`` (default 0.5: the critic's +1 / -1 scale to the score's 1 / 0, right only for a calibrated critic);
- held out: for each deck, the variance-minimizing beta fitted on the other decks' pairs (leave one deck out),
  applied to this deck; the pooled out-of-sample reduction is the number that decides M5 (needs >= 2 decks).
Also reported, as diagnostics only: the in-sample best beta and its reduction (optimistic), and the luck's explained
variance of single games. The evaluator enables the control variate only if the out-of-sample reduction is >= 30%.

Usage: tools/deckevo_m5_cv.py CHECKPOINT N_DECKS CHILDREN PAIRS K OUT.json [--env md-2026-09] [--device cuda]
       [--beta 0.5] [--envs 256]"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ygorl.build.control import CriticControlVariate, critic_opening_values
from ygorl.build.tuner import Edit, apply
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import derive_seed
from ygorl.eval.batched import paired_specs, play_policies
from ygorl.train.checkpoint import load_actor_critic, torch_device

BLANK = 65957473  # Metal Armored Bug, as in M2


def reduction(d_raw: np.ndarray, d_adj: np.ndarray) -> float:
    v = d_raw.var(ddof=1)
    return float(1 - d_adj.var(ddof=1) / v) if v > 0 else float("nan")


def fit_beta(d_raw: np.ndarray, l_diff: np.ndarray) -> float:
    """The variance-minimizing coefficient cov(d, l) / var(l)."""
    v = l_diff.var(ddof=1) if len(l_diff) > 1 else 0.0
    return float(np.cov(d_raw, l_diff)[0, 1] / v) if v > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a PPO checkpoint (the critic is needed)")
    ap.add_argument("n_decks", type=int)
    ap.add_argument("children", type=int)
    ap.add_argument("pairs", type=int)
    ap.add_argument("k", type=int, help="alternative shuffles per game")
    ap.add_argument("out")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--envs", type=int, default=256)
    args = ap.parse_args()
    device = torch_device(args.device)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], e["file"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]
    train = {p.name for p in Path("out/corpus/train").glob("*.ydk")}
    bases = [t for t in lists if Path(t[1]).name in train and not env.validate_deck(t[2], cards)]
    rng = np.random.default_rng(0)
    picks = [bases[i] for i in rng.choice(len(bases), args.n_decks, replace=False)]  # the M1 / M2 decks
    meta = [m.deck for m in env.meta_decks]
    weights = np.array([m.share for m in env.meta_decks], dtype=float)
    ac = load_actor_critic(args.checkpoint)
    model = ac.model.to(device)
    config = DuelConfig(max_decisions=4000)
    opp_idx = [int(np.random.default_rng(derive_seed(11, 2, k)).choice(len(meta), p=weights / weights.sum()))
               for k in range(args.pairs)]  # fmt: skip

    def critic_env(n):
        return EncodedVecEnv(n, 8, cards=cards, vocab=ac.vocab, event_length=ac.event_length, privileged=True,
                             skip_forced=True)  # fmt: skip

    cv = CriticControlVariate(lambda sp: critic_opening_values(model, critic_env, sp, device=device,
                                                               num_envs=args.envs), k=args.k, beta=args.beta)  # fmt: skip
    res = {"checkpoint": args.checkpoint, "environment": env.version, "pairs": args.pairs, "k": args.k,
           "beta": args.beta, "decks": []}  # fmt: skip
    pooled_raw, pooled_luck, game_scores, game_luck, per_deck = [], [], [], [], []
    for di, (dtype, file, base) in enumerate(picks):
        distinct = sorted({pw for pw in base.main if pw != BLANK})
        chosen = [int(x) for x in np.random.default_rng(100 + di).choice(distinct, min(args.children, len(distinct)),
                                                                         replace=False)]  # fmt: skip
        decks = [base] + [apply(base, Edit(out=pw, into=BLANK, section="main")) for pw in chosen]
        specs = [s for d in decks for k in range(args.pairs)
                 for s in paired_specs(d, meta[opp_idx[k]], 1, derive_seed(11, 1, k), config)]  # fmt: skip
        vec = EncodedVecEnv(min(args.envs, len(specs)), 8, cards=cards, vocab=ac.vocab, event_length=ac.event_length,
                            skip_forced=True)  # fmt: skip
        records, stats = play_policies(vec, specs, model, device=device)
        s = np.array([np.nan if r.reason == "exception" else 1.0 if r.winner == 0 else 0.5 if r.winner is None else 0.0
                      for r in records]).reshape(len(decks), args.pairs, 2)  # fmt: skip
        t0 = time.perf_counter()
        luck = cv.luck(specs).reshape(len(decks), args.pairs, 2)
        critic_s = time.perf_counter() - t0
        luck = np.nan_to_num(luck, nan=0.0)
        ok = np.isfinite(s)
        game_scores.extend(s[ok].tolist())
        game_luck.extend(luck[ok].tolist())
        rows = []
        for c in range(1, len(decks)):
            d_raw = s[c].mean(-1) - s[0].mean(-1)
            l_diff = luck[c].mean(-1) - luck[0].mean(-1)
            m = np.isfinite(d_raw)
            d_raw, l_diff = d_raw[m], l_diff[m]
            d_adj = d_raw - args.beta * l_diff
            best_beta = fit_beta(d_raw, l_diff)
            rows.append({"card": chosen[c - 1], "name": cards[chosen[c - 1]].name, "pairs": int(m.sum()),
                         "diff_raw": float(d_raw.mean()), "diff_adj": float(d_adj.mean()),
                         "sd_raw": float(d_raw.std(ddof=1)), "sd_adj": float(d_adj.std(ddof=1)),
                         "reduction": reduction(d_raw, d_adj), "best_beta": best_beta,
                         "reduction_best_beta": reduction(d_raw, d_raw - best_beta * l_diff)})  # fmt: skip
            pooled_raw.extend(d_raw.tolist())
            pooled_luck.extend(l_diff.tolist())
            per_deck.append((di, d_raw, l_diff))
        entry = {"type": dtype, "file": file, "parent": float(np.nanmean(s[0])), "children": rows,
                 "game_seconds": stats["seconds"], "critic_seconds": critic_s}  # fmt: skip
        res["decks"].append(entry)
        print(f"{dtype:28s} parent {entry['parent']:.3f}  sd/pair raw {np.mean([r['sd_raw'] for r in rows]):.3f} "
              f"adj {np.mean([r['sd_adj'] for r in rows]):.3f}  reduction {np.mean([r['reduction'] for r in rows]):+.3f}"
              f"  games {stats['seconds']:.0f}s critic {critic_s:.0f}s", flush=True)  # fmt: skip
    d_raw, l_diff = np.array(pooled_raw), np.array(pooled_luck)
    d_adj = d_raw - args.beta * l_diff
    boot = np.random.default_rng(0)
    idx = boot.integers(0, len(d_raw), size=(1000, len(d_raw)))
    reds = [reduction(d_raw[i], d_adj[i]) for i in idx]
    gs, gl = np.array(game_scores), np.array(game_luck)
    best_beta = fit_beta(d_raw, l_diff)
    held_raw, held_adj, held_beta = [], [], {}
    for di in range(len(picks)):  # leave one deck out: beta from the other decks' pairs only
        rest = [(r, lk) for dj, r, lk in per_deck if dj != di]
        if not rest:
            continue
        b = fit_beta(np.concatenate([r for r, _ in rest]), np.concatenate([lk for _, lk in rest]))
        held_beta[di] = b
        for dj, r, lk in per_deck:
            if dj == di:
                held_raw.extend(r.tolist())
                held_adj.extend((r - b * lk).tolist())
    heldout = reduction(np.array(held_raw), np.array(held_adj)) if held_raw else float("nan")
    for di, b in held_beta.items():
        res["decks"][di]["heldout_beta"] = b
    res["pooled"] = {
        "pairs": len(d_raw), "sd_raw": float(d_raw.std(ddof=1)), "sd_adj": float(d_adj.std(ddof=1)),
        "reduction": reduction(d_raw, d_adj), "reduction_ci": [float(np.nanquantile(reds, 0.025)),
                                                                float(np.nanquantile(reds, 0.975))],
        "best_beta": best_beta, "reduction_best_beta": reduction(d_raw, d_raw - best_beta * l_diff),
        "game_luck_r2": float(np.corrcoef(gs, gl)[0, 1] ** 2) if gl.var() > 0 else 0.0,
        "heldout_reduction": heldout, "heldout_betas": list(held_beta.values()),
        "critic_readings": cv.readings, "enable": bool(heldout >= 0.3),
    }  # fmt: skip
    print("pooled:", json.dumps(res["pooled"]))
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
