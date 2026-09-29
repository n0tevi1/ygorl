"""M1 (docs/spikes/deck-evolution.md): how correlated are a parent deck's and a one-card child's results on the same
common-random-number game? Set up exactly like tools/tune_deck.py (tech pool, legal single-card swaps, paired games
against the environment's meta decks by share), but keep every game's outcome. Report, per deck and pooled, the
Pearson correlation rho between parent and child outcomes game by game and pair by pair, and the standard deviation
of the paired difference (what sets the games needed to confirm a change).

Usage: tools/deckevo_m1_rho.py CHECKPOINT N_DECKS CHILDREN PAIRS OUT.json [--env md-2026-09] [--device cuda]"""

import argparse
import json
from pathlib import Path

import numpy as np

from ygorl.build.tuner import neighbors, tech_pool
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import derive_seed
from ygorl.eval.batched import paired_specs, play_policies
from ygorl.train.checkpoint import load_actor, torch_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("n_decks", type=int)
    ap.add_argument("children", type=int)
    ap.add_argument("pairs", type=int)
    ap.add_argument("out")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ckpt, n_decks, n_children, pairs, out = args.checkpoint, args.n_decks, args.children, args.pairs, args.out
    device = torch_device(args.device)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], e["file"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]
    train = {p.name for p in Path("out/corpus/train").glob("*.ydk")}
    bases = [t for t in lists if Path(t[1]).name in train and not env.validate_deck(t[2], cards)]
    rng = np.random.default_rng(0)
    picks = [bases[i] for i in rng.choice(len(bases), n_decks, replace=False)]
    meta = [m.deck for m in env.meta_decks]
    weights = np.array([m.share for m in env.meta_decks], dtype=float)
    pol = load_actor(ckpt)
    net = pol.net.to(device)
    config = DuelConfig(max_decisions=4000)
    res = {"checkpoint": ckpt, "environment": env.version, "pairs": pairs, "decks": []}
    all_g, all_p, all_d = [], [], []
    for dtype, file, base in picks:
        same = [d for t, _, d in lists if t == dtype and d is not base]
        pool = tech_pool(base, same, meta)
        edits = neighbors(base, pool, lambda pw: cards[pw].is_extra_deck if pw in cards else False,
                          lambda d: not env.validate_deck(d, cards), limit=n_children, rng=rng)  # fmt: skip
        decks = [base] + [d for _, d in edits]
        opp = [meta[int(np.random.default_rng(derive_seed(7, 2, k)).choice(len(meta), p=weights / weights.sum()))]
               for k in range(pairs)]  # fmt: skip
        specs = [
            s for d in decks for k in range(pairs) for s in paired_specs(d, opp[k], 1, derive_seed(7, 1, k), config)
        ]
        vec = EncodedVecEnv(min(256, len(specs)), 8, cards=cards, vocab=pol.vocab, event_length=pol.event_length,
                            skip_forced=True)  # fmt: skip
        records, stats = play_policies(vec, specs, net, device=device)
        s = np.array([np.nan if r.reason == "exception" else 1.0 if r.winner == 0 else 0.5 if r.winner is None else 0.0
                      for r in records]).reshape(len(decks), pairs, 2)  # fmt: skip
        par_g, par_p = s[0].reshape(-1), s[0].mean(-1)
        rows = []
        for c in range(1, len(decks)):
            g, p = s[c].reshape(-1), s[c].mean(-1)
            mg, mp = np.isfinite(g) & np.isfinite(par_g), np.isfinite(p) & np.isfinite(par_p)
            d = p[mp] - par_p[mp]
            rows.append({"edit": edits[c - 1][0].describe(), "rho_game": float(np.corrcoef(par_g[mg], g[mg])[0, 1]),
                         "rho_pair": float(np.corrcoef(par_p[mp], p[mp])[0, 1]), "diff": float(d.mean()),
                         "sd_diff_pair": float(d.std(ddof=1))})  # fmt: skip
            all_g.append((par_g[mg], g[mg]))
            all_p.append((par_p[mp], p[mp]))
            all_d.extend(d.tolist())
        entry = {"type": dtype, "file": file, "parent_win_rate": float(np.nanmean(s[0])), "children": rows,
                 "games_per_s": stats["games_per_s"]}  # fmt: skip
        res["decks"].append(entry)
        print(f"{dtype:30s} parent {entry['parent_win_rate']:.3f}  rho_game "
              f"{np.mean([r['rho_game'] for r in rows]):.3f}  rho_pair {np.mean([r['rho_pair'] for r in rows]):.3f}  "
              f"sd(diff/pair) {np.mean([r['sd_diff_pair'] for r in rows]):.3f}  {stats['games_per_s']:.1f} games/s",
              flush=True)  # fmt: skip
    pg = np.corrcoef(np.concatenate([a for a, _ in all_g]), np.concatenate([b for _, b in all_g]))[0, 1]
    pp = np.corrcoef(np.concatenate([a for a, _ in all_p]), np.concatenate([b for _, b in all_p]))[0, 1]
    sd = float(np.std(all_d, ddof=1))
    res["pooled"] = {"rho_game": float(pg), "rho_pair": float(pp), "sd_diff_pair": sd,
                     "pairs_for_2pp": float((2.8 * sd / 0.02) ** 2), "pairs_for_5pp": float((2.8 * sd / 0.05) ** 2)}  # fmt: skip
    print("pooled:", json.dumps(res["pooled"]))
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
