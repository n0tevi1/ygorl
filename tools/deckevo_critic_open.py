"""Critic opening-hand signal (for M3): per M2 deck, K random shuffles x both first players against meta opponents by
share; advance each game to its first decision and read the privileged critic's V from the deck's side (negated when
the first decision is the opponent's: zero-sum). Per card: mean V with it in the opening hand minus without. No games
are finished (the game driver abandons each game at that decision), so it costs only forward passes.

Usage: tools/deckevo_critic_open.py CHECKPOINT M2.json K OUT.json [--env md-2026-09] [--device cuda]"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ygorl.build.diagnose import opening_hand
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import DuelConfig, default_cards, shuffle_deck
from ygorl.env import GameSpec
from ygorl.env.driver import ABANDON, drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import derive_seed
from ygorl.nets.actor_critic import collate_privileged
from ygorl.nets.batch import collate
from ygorl.train.checkpoint import load_actor_critic, torch_device


@torch.no_grad()
def opening_values(model, vocab, event_length, deck, opponents, weights, k, seed, device):
    cards = default_cards()
    base = DuelConfig(max_decisions=4000, shuffle_decks=False)
    specs, hands = [], []
    for i in range(k):
        s = derive_seed(seed, i)
        opp = opponents[int(np.random.default_rng(derive_seed(seed, 9, i)).choice(len(opponents), p=weights))]
        a = replace(deck, main=tuple(shuffle_deck(deck.main, s, 0)))
        b = replace(opp, main=tuple(shuffle_deck(opp.main, s, 1)))
        for first in (0, 1):
            specs.append(GameSpec(seed=s, deck_a=a, deck_b=b, first=first, config=base))
            hands.append(opening_hand(a.main))
    env = EncodedVecEnv(min(256, len(specs)), 8, cards=cards, vocab=vocab, event_length=event_length,
                        privileged=True, skip_forced=True)  # fmt: skip
    values = np.full(len(specs), np.nan)

    def decide(ready):
        out = model(
            collate([ev.obs for _, ev in ready], device), collate_privileged([ev.privileged for _, ev in ready], device)
        )
        for (game, ev), v in zip(ready, out.v.float().cpu().tolist()):
            values[game.index] = v if ev.player == game.spec.seat_of_deck(0) else -v  # from deck a's side
        return [ABANDON] * len(ready)  # one reading per game: start the next spec on this slot

    drive(env, specs, decide, lambda game, result: None, min_batch=64)
    return values, hands


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("m2", help="the .json summary written by deckevo_m2_loo.py")
    ap.add_argument("k", type=int, help="shuffles per deck (each played with both first players)")
    ap.add_argument("out")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ckpt, m2, k, out = args.checkpoint, args.m2, args.k, args.out
    device = torch_device(args.device)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    meta = [m.deck for m in env.meta_decks]
    w = np.array([m.share for m in env.meta_decks], dtype=float)
    ac = load_actor_critic(ckpt)
    model, vocab, event_length = ac.model.to(device), ac.vocab, ac.event_length
    res = []
    for di, deck in enumerate(json.loads(Path(m2).read_text())):
        d = load_ydk(env.artifacts_dir / deck["file"])
        v, hands = opening_values(model, vocab, event_length, d, meta, w / w.sum(), k, 1000 + di, device)
        per = {}
        for card in sorted(set(d.main)):
            held = np.array([card in h for h in hands]) & np.isfinite(v)
            rest = ~held & np.isfinite(v)
            per[int(card)] = float(v[held].mean() - v[rest].mean()) if held.any() and rest.any() else None
        res.append(
            {"type": deck["type"], "k": k, "checkpoint": ckpt, "mean_v": float(np.nanmean(v)), "critic_opening": per}
        )
        print(f"{deck['type']:28s} mean V {np.nanmean(v):+.3f} over {int(np.isfinite(v).sum())} openings", flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
