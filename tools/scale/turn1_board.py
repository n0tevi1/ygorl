"""How much interruption does a policy's own turn 1 put on board, on the solver's labelled hands?
For each plain record: the policy plays turn 1 as player 0 from the same start (passive opponent), the end board is
scored with ygorl.solver.blocking.board_interruptions, and compared with the solver's blocking line for that hand.
  uv run --no-sync python out/claude-scale/turn1_board.py LABELS.jsonl CKPT [CKPT...] [--limit N]
"""

import torch

torch.set_num_threads(2)
import argparse
import json
import os
import sys
import time

os.chdir("/home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review")
sys.path.insert(0, "src")

p = argparse.ArgumentParser()
p.add_argument("labels")
p.add_argument("ckpts", nargs="+")
p.add_argument("--limit", type=int, default=200)
p.add_argument("--sampled", action="store_true")
p.add_argument("--min-hand", type=int, default=0)
a = p.parse_args()
from ygorl.agents.registry import agent_factory
from ygorl.data import load_environment
from ygorl.engine.duel import default_cards
from ygorl.solver import Demonstration
from ygorl.solver.blocking import board_interruptions
from ygorl.train.bc import play_opening

env = load_environment("md-2026-09")
cards = default_cards()
plain = [Demonstration.from_json(json.loads(l)) for l in open(a.labels)]
plain = [d for d in plain if d.variant == "plain" and d.status == "solved" and d.hand_index >= a.min_hand][: a.limit]
solver = [board_interruptions(d.lines[0].board, cards) for d in plain]


def stat(xs):
    return sum(xs) / max(1, len(xs))


print(
    f"{len(plain)} hands | solver line: interruptions {stat([s.interruptions for s in solver]):.2f}, negates {stat([s.negates for s in solver]):.2f}, hands with >=1 {sum(s.interruptions > 0 for s in solver)}"
)
for ck in a.ckpts:
    f = agent_factory(f"policy:{ck}" + ("" if a.sampled else "@greedy"))
    t0 = time.time()
    rows = []
    fails = 0
    for i, d in enumerate(plain):
        try:
            r = play_opening(d.replay(None), f(i), d.targets, env=env, cards=cards)
            rows.append(board_interruptions(r.board, cards))
        except Exception as e:  # noqa
            fails += 1
    mon = None
    name = ck.split("/ab1/")[-1] if "/ab1/" in ck else ck.split("/")[-1]
    print(
        f"{name}: interruptions {stat([s.interruptions for s in rows]):.2f}, negates {stat([s.negates for s in rows]):.2f}, hands with >=1 {sum(s.interruptions > 0 for s in rows)}/{len(rows)}, failures {fails} ({time.time() - t0:.0f}s)"
    )
