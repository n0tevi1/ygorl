"""Critic opening-hand signal (for M3): per M2 deck, K random shuffles x both first players against meta opponents by
share; advance each game to its first decision and read the privileged critic's V from the deck's side (negated when
the first decision is the opponent's: zero-sum). Per card: mean V with it in the opening hand minus without. No games
are finished, so it costs only forward passes.

Usage: tools/deckevo_critic_open.py CHECKPOINT M2.json K OUT.json"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ygorl.build.diagnose import opening_hand
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import DuelConfig, default_cards, shuffle_deck
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import derive_seed
from ygorl.nets import NetConfig
from ygorl.nets.actor_critic import ActorCritic, collate_privileged
from ygorl.nets.batch import collate
from ygorl.train.checkpoint import load_checkpoint, vocab_from_text


def load_actor_critic(path, device):
    state = load_checkpoint(path)
    c = state["config"]
    cfg = NetConfig.from_dict(state["net_config"])
    model = ActorCritic(cfg, None, privileged=c["privileged_critic"], privileged_dim=c["privileged_dim"],
                        critic_hidden=c["critic_hidden"], shared_backbone=c["shared_backbone"])  # fmt: skip
    model.load_state_dict(state["learner"]["model"])
    return model.to(device).eval(), vocab_from_text(state["vocab"]), int(c["event_length"])


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
    queue = iter(range(len(specs)))
    running = {}

    def launch(e):
        for i in queue:
            env.reset(e, specs[i])
            running[e] = i
            return True
        return False

    active = sum(launch(e) for e in range(env.num_envs))
    while active:
        evs = env.recv(min(64, active))
        ready = [ev for ev in evs if ev.result is None]
        if ready:
            out = model(
                collate([ev.obs for ev in ready], device), collate_privileged([ev.privileged for ev in ready], device)
            )
            for ev, v in zip(ready, out.v.float().cpu().tolist()):
                i = running[ev.env_id]
                side = specs[i].first  # engine seat holding deck a
                values[i] = v if ev.player == side else -v
        for ev in evs:  # one reading per game: start the next spec on this slot
            if not launch(ev.env_id):
                active -= 1
    return values, hands


def main():
    ckpt, m2, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    cards = default_cards()
    env = load_environment("md-2026-09", cards=cards)
    meta = [m.deck for m in env.meta_decks]
    w = np.array([m.share for m in env.meta_decks], dtype=float)
    model, vocab, event_length = load_actor_critic(ckpt, "cuda")
    res = []
    for di, deck in enumerate(json.loads(Path(m2).read_text())):
        d = load_ydk(env.artifacts_dir / deck["file"])
        v, hands = opening_values(model, vocab, event_length, d, meta, w / w.sum(), k, 1000 + di, "cuda")
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
