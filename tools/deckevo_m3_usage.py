"""M3 for the usage signals (#109, docs/spikes/deck-evolution.md §3): do dead-card rate, use rate or the advantage
after use (Q - V̄ of the actions on a card) predict the M2 leave-one-out truth?

Replays the parent decks of M2 on M2's own game specs (same decks, opponents and seeds; the policy's sampling is fresh,
from a seeded stream per game) with a checkpoint's actor-critic on the game driver (ygorl.env.driver), and records for deck a's player, per game and card: whether a copy was in hand,
whether an action on it was ever legal, whether one was chosen, and Q(s, a) - sum_b pi(b) Q(s, b) of the chosen
actions on it. Per card of the M2 sample:

- dead rate: games where it was in hand but never had a legal action / games where it was in hand;
- idle rate: games where it had a legal action but none was chosen / games where it had one;
- use rate: games where an action on it was chosen / all games;
- advantage: mean Q - V̄ of the chosen actions on it.

Spearman (average ranks) against the LOO values within each deck and pooled on per-deck z-scores, with and without
Exodia (whose win-condition pieces are never "used").

Usage: tools/deckevo_m3_usage.py CHECKPOINT M2.json OUT.json [--pairs 500] [--env md-2026-09] [--device cuda]"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckevo_m3_signals import spearman  # noqa: E402

from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.data.environment import load_environment  # noqa: E402
from ygorl.engine.duel import DuelConfig, default_cards  # noqa: E402
from ygorl.env.driver import drive  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.eval.arena import derive_seed  # noqa: E402
from ygorl.eval.batched import paired_specs  # noqa: E402
from ygorl.nets import collate  # noqa: E402
from ygorl.nets.actor_critic import collate_privileged  # noqa: E402
from ygorl.train.checkpoint import load_actor_critic, torch_device  # noqa: E402

HAND, CONTROLLER, CARD = 2, 4, 0  # card table: location 2 = hand, column 4 controller (0 = viewer), column 0 index
ACTION_CARD = 2  # action table column 2: the card's vocab index
SAMPLE_SEED = 0  # game i samples its decisions from np.random.default_rng([SAMPLE_SEED, i])


def usage(model, vocab, event_length, base, meta, opp_idx, pairs, device):
    cards = default_cards()
    config = DuelConfig(max_decisions=4000)
    specs = [s for k in range(pairs) for s in paired_specs(base, meta[opp_idx[k]], 1, derive_seed(11, 1, k), config)]
    idx = {vocab.index(pw): pw for pw in set(base.main)}
    if len(idx) != len(set(base.main)) or 1 in idx:
        raise ValueError("a deck card is missing from the vocab (shares the unknown index)")
    env = EncodedVecEnv(256, 8, cards=cards, vocab=vocab, event_length=event_length, privileged=True, skip_forced=True)
    games = [defaultdict(lambda: [False, False, False]) for _ in specs]  # card -> [in hand, legal, chosen]
    adv = defaultdict(list)

    @torch.no_grad()
    def decide(ready):
        evs = [ev for _, ev in ready]
        o = model(collate([ev.obs for ev in evs], device), collate_privileged([ev.privileged for ev in evs], device))
        pi = torch.softmax(o.logits.float(), -1)
        cdf = np.cumsum(pi.cpu().numpy().astype(np.float64), -1)
        # each game samples from its own seeded stream: the choices do not depend on how the games were batched
        acts = [min(int(np.searchsorted(c, game.state.random() * c[-1], side="right")), len(c) - 1)
                for (game, _), c in zip(ready, cdf)]  # fmt: skip
        vbar = (pi * o.q.float()).sum(-1)
        gain = (o.q.float().gather(1, torch.tensor(acts, device=pi.device)[:, None]).squeeze(1) - vbar).cpu().numpy()
        for (game, ev), a, g in zip(ready, acts, gain):
            if ev.player == game.spec.seat_of_deck(0):  # deck a's player
                rec, obs = games[game.index], ev.obs
                c = obs["cards"]
                for ci in c[(c[:, 1] == HAND) & (c[:, CONTROLLER] == 0), CARD].tolist():
                    if ci in idx:
                        rec[ci][0] = True
                legal = obs["actions"][obs["action_mask"] == 1, ACTION_CARD].tolist()
                for ci in set(legal):
                    if ci in idx:
                        rec[ci][1] = True
                chosen = int(obs["actions"][a, ACTION_CARD])
                if chosen in idx:
                    rec[chosen][2] = True
                    adv[chosen].append(float(g))
        return acts

    drive(env, specs, decide, lambda game, result: None, min_batch=64,
          start=lambda i, spec: np.random.default_rng([SAMPLE_SEED, i]))  # fmt: skip
    out = {}
    for ci, pw in idx.items():
        hand = sum(g[ci][0] for g in games if ci in g)
        legal = sum(g[ci][1] for g in games if ci in g)
        dead = sum(g[ci][0] and not g[ci][1] for g in games if ci in g)
        idle = sum(g[ci][1] and not g[ci][2] for g in games if ci in g)
        used = sum(g[ci][2] for g in games if ci in g)
        out[pw] = {"dead_rate": dead / hand if hand else np.nan, "idle_rate": idle / legal if legal else np.nan,
                   "use_rate": used / len(specs), "advantage": float(np.mean(adv[ci])) if adv[ci] else np.nan}  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("m2", help="the .json summary written by deckevo_m2_loo.py")
    ap.add_argument("out")
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ckpt, m2, out, pairs = args.checkpoint, args.m2, args.out, args.pairs
    device = torch_device(args.device)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    meta = [m.deck for m in env.meta_decks]
    w = np.array([m.share for m in env.meta_decks], dtype=float)
    summary = json.loads(Path(m2).read_text())
    opp_idx = [int(np.random.default_rng(derive_seed(11, 2, k)).choice(len(meta), p=w / w.sum())) for k in range(pairs)]
    ac = load_actor_critic(ckpt)
    model, vocab, event_length = ac.model.to(device), ac.vocab, ac.event_length
    names = ("dead_rate", "idle_rate", "use_rate", "advantage")
    signs = {"dead_rate": -1, "idle_rate": -1, "use_rate": 1, "advantage": 1}  # higher = more valuable
    pooled = {n: ([], [], [], []) for n in names}  # sig, truth, sig without Exodia, truth without Exodia
    res = []
    for deck in summary:
        base = load_ydk(env.artifacts_dir / deck["file"])
        sig = usage(model, vocab, event_length, base, meta, opp_idx, pairs, device)
        loo = {r["card"]: r["value"] for r in deck["loo"]}
        row = {"type": deck["type"], "cards": {str(c): {**sig[c], "loo": loo[c]} for c in loo}}
        line = []
        for n in names:
            cs = [c for c in loo if np.isfinite(sig[c][n])]
            s = signs[n] * np.array([sig[c][n] for c in cs])
            t = np.array([loo[c] for c in cs])
            if len(cs) < 3:
                continue
            row[n] = spearman(s, t)
            line.append(f"{n} {row[n]:+.2f}")
            zs, zt = (s - s.mean()) / (s.std() + 1e-9), (t - t.mean()) / (t.std() + 1e-9)
            pooled[n][0].extend(zs)
            pooled[n][1].extend(zt)
            if deck["type"] != "Exodia":
                pooled[n][2].extend(zs)
                pooled[n][3].extend(zt)
        res.append(row)
        print(f"{deck['type']:28s} " + "  ".join(line), flush=True)
    total = {n: {"pooled": spearman(np.array(p[0]), np.array(p[1])),
                 "pooled_without_exodia": spearman(np.array(p[2]), np.array(p[3])), "cards": len(p[0])}
             for n, p in pooled.items() if len(p[0]) > 2}  # fmt: skip
    for n, t in total.items():
        print(
            f"pooled {n:10s} {t['pooled']:+.2f}  without Exodia {t['pooled_without_exodia']:+.2f}  ({t['cards']} cards)"
        )
    Path(out).write_text(json.dumps({"checkpoint": ckpt, "pairs": pairs, "decks": res, "pooled": total}, indent=1))


if __name__ == "__main__":
    main()
