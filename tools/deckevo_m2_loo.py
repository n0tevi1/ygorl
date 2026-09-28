"""M2 (docs/spikes/deck-evolution.md): leave-one-out ground truth. For the same decks as M1, pick CARDS distinct main-
deck cards per deck and replace ONE copy with a fixed blank (Metal Armored Bug: a level-8 vanilla normal monster that
needs two tributes); play parent and every child on the same common-random-number pairs. The paired difference
parent - child is that card's (one copy's) marginal value. The parent's games are also kept with each game's opening
hand (the engine draws from the end of the main deck list, so the opening hand is the last 5 cards of the shuffled
deck in the game spec) for M3.

Usage: tools/deckevo_m2_loo.py CHECKPOINT N_DECKS CARDS PAIRS OUT.npz"""

import json
import sys
from pathlib import Path

import numpy as np

from ygorl.build.tuner import Edit, apply
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import derive_seed
from ygorl.eval.batched import paired_specs, play_policies
from ygorl.train.checkpoint import load_actor

BLANK = 65957473  # Metal Armored Bug


def main():
    ckpt, n_decks, n_cards, pairs, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    cards = default_cards()
    env = load_environment("md-2026-09", cards=cards)
    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], e["file"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]
    train = {p.name for p in Path("out/corpus/train").glob("*.ydk")}
    bases = [t for t in lists if Path(t[1]).name in train and not env.validate_deck(t[2], cards)]
    rng = np.random.default_rng(0)
    picks = [bases[i] for i in rng.choice(len(bases), n_decks, replace=False)]  # the same decks as M1
    meta = [m.deck for m in env.meta_decks]
    weights = np.array([m.share for m in env.meta_decks], dtype=float)
    pol = load_actor(ckpt)
    net = pol.net.to("cuda")
    config = DuelConfig(max_decisions=4000)
    opp_idx = [int(np.random.default_rng(derive_seed(11, 2, k)).choice(len(meta), p=weights / weights.sum()))
               for k in range(pairs)]  # fmt: skip
    save = {}
    summary = []
    for di, (dtype, file, base) in enumerate(picks):
        distinct = sorted({pw for pw in base.main if pw != BLANK})
        chosen = [
            int(x) for x in np.random.default_rng(100 + di).choice(distinct, min(n_cards, len(distinct)), replace=False)
        ]
        decks = [base] + [apply(base, Edit(out=pw, into=BLANK, section="main")) for pw in chosen]
        specs = [s for d in decks for k in range(pairs)
                 for s in paired_specs(d, meta[opp_idx[k]], 1, derive_seed(11, 1, k), config)]  # fmt: skip
        vec = EncodedVecEnv(min(256, len(specs)), 8, cards=cards, vocab=pol.vocab, event_length=pol.event_length,
                            skip_forced=True)  # fmt: skip
        records, stats = play_policies(vec, specs, net, device="cuda")
        s = np.array([np.nan if r.reason == "exception" else 1.0 if r.winner == 0 else 0.5 if r.winner is None else 0.0
                      for r in records]).reshape(len(decks), pairs, 2)  # fmt: skip
        parent_specs = specs[: pairs * 2]
        hands = np.array([list(sp.deck_a.main[-5:]) for sp in parent_specs]).reshape(pairs, 2, 5)
        firsts = np.array([sp.first for sp in parent_specs]).reshape(pairs, 2)
        loo = []
        for c, pw in enumerate(chosen, start=1):
            d = s[0].mean(-1) - s[c].mean(-1)
            d = d[np.isfinite(d)]
            loo.append((pw, float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), base.main.count(pw)))
        summary.append({"type": dtype, "file": file, "parent": float(np.nanmean(s[0])), "games_per_s": stats["games_per_s"],
                        "loo": [{"card": pw, "name": cards[pw].name, "copies": k, "value": v, "ci": h} for pw, v, h, k in loo]})  # fmt: skip
        save[f"d{di}_scores"] = s
        save[f"d{di}_hands"] = hands
        save[f"d{di}_first"] = firsts
        save[f"d{di}_chosen"] = np.array(chosen)
        save[f"d{di}_main"] = np.array(base.main)
        np.savez_compressed(out, **save)
        Path(out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
        print(f"{dtype:28s} parent {summary[-1]['parent']:.3f}  " + "  ".join(f"{cards[pw].name[:14]}:{v:+.3f}"
              for pw, v, _, _ in sorted(loo, key=lambda t: -t[1])), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
