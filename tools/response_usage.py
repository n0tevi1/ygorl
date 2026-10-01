"""Does the policy use its interruptions on the opponent's turn? (strength diagnosis, #83)

Self-play of a checkpoint on random corpus pairings (batched path). At every chain prompt the responding player
gets on the **opponent's** turn (``MSG_SELECT_CHAIN`` with ``is_my_turn = 0``), record whether an activation was
available and whether the policy took one. Reports, by the opponent's turn number and for the first player on the
second player's turn 2 in particular:

- prompts with an activation available, and the share where the policy activated (vs passed);
- per game: did the first player activate anything during turn 2, and the first player's win rate with / without.

Usage: tools/response_usage.py CHECKPOINT [--pairings 400] [--greedy] [--out games.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelConfig
from ygorl.env.driver import drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.encoding import ACTION_KINDS
from ygorl.eval.arena import derive_seed
from ygorl.eval.batched import paired_specs
from ygorl.nets import collate
from ygorl.nets.batch import policy_logits
from ygorl.train.checkpoint import load_actor

CHAIN = ACTION_KINDS.index("chain") + 1  # action-table column 0 is kind index + 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--decks", default="out/corpus/train")
    p.add_argument("--pairings", type=int, default=400)
    p.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    decks = [load_ydk(f) for f in sorted(Path(a.decks).glob("*.ydk"))]
    rng = np.random.default_rng(a.seed)
    config = DuelConfig(max_decisions=4000)
    specs = []
    for k in range(a.pairings):
        i, j = (int(x) for x in rng.choice(len(decks), 2, replace=False))
        specs.extend(paired_specs(decks[i], decks[j], 1, derive_seed(a.seed, k), config))
    pol = load_actor(a.checkpoint)
    net = pol.net.to(a.device).eval()
    env = EncodedVecEnv(min(256, len(specs)), 8, vocab=pol.vocab, event_length=pol.event_length, skip_forced=True)
    gen = torch.Generator(device=a.device).manual_seed(a.seed)

    def start(i, spec):
        return {"prompts": defaultdict(lambda: [0, 0, 0])}  # (responder seat, opp turn) -> [prompts, available, used]

    def decide(batch):
        obs = [ev.obs for _, ev in batch]
        with torch.no_grad():
            logits = policy_logits(net, collate(obs, a.device)).float()
            acts = (logits.argmax(-1) if a.greedy else
                    torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1)).tolist()  # fmt: skip
        for (game, ev), act in zip(batch, acts):
            g = ev.obs["globals"]
            if int(g[18]) == C.MSG_SELECT_CHAIN and int(g[2]) == 0:
                kinds = ev.obs["actions"][:, 0][ev.obs["action_mask"] == 1]
                rec = game.state["prompts"][(ev.player, int(g[3]))]
                rec[0] += 1
                if (kinds == CHAIN).any():
                    rec[1] += 1
                    rec[2] += int(ev.obs["actions"][act, 0] == CHAIN)
        return acts

    games = []

    def on_result(game, res):
        if str(res.get("reason", "")) == "error":
            return
        w = res.get("winner")
        games.append({"turns": int(res.get("turns", 0)), "winner": w,
                      "prompts": {f"{s}:{t}": v for (s, t), v in game.state["prompts"].items()}})  # fmt: skip

    drive(env, specs, decide, on_result, min_batch=max(1, env.num_envs // 2), start=start)
    if a.out:
        Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in games))
    # engine seat 0 always goes first: on turn t the turn player is seat (t - 1) % 2, the responder the other one
    by_turn = defaultdict(lambda: np.zeros(3))
    for x in games:
        for key, v in x["prompts"].items():
            _, t = key.split(":")
            by_turn[min(int(t), 8)] += v
    print(f"{len(games)} games, {'greedy' if a.greedy else 'sampled'} play")
    print("opponent's turn | chain prompts | with an activation available | activated when available")
    for t in sorted(by_turn):
        n, av, used = by_turn[t]
        print(
            f"  turn {t}{'+' if t == 8 else ' '}       {int(n):6d}          {int(av):6d}                  {used / max(av, 1):.3f}"
        )
    # first player (seat 0) responding on turn 2
    used2 = [x for x in games if x["prompts"].get("0:2", [0, 0, 0])[1] > 0]
    act2 = [x for x in used2 if x["prompts"]["0:2"][2] > 0]
    pas2 = [x for x in used2 if x["prompts"]["0:2"][2] == 0]
    wr = lambda xs: (
        np.mean([0.5 if x["winner"] is None else float(x["winner"] == 0) for x in xs]) if xs else float("nan")
    )  # noqa: E731
    print(f"first player had an activation available on turn 2 in {len(used2)}/{len(games)} games; "
          f"activated at least once in {len(act2)} (win {wr(act2):.3f}), never in {len(pas2)} (win {wr(pas2):.3f}); "
          f"first-player win overall {wr(games):.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
