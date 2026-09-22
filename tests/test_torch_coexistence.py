"""The extension links libstdc++ statically and replaces operator new (T2.8); it must coexist with PyTorch."""

import subprocess
import sys

import pytest

pytest.importorskip("torch")

SCRIPT = """
import {first}, {second}
import torch
from pathlib import Path
from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel, DuelConfig, DuelSession

decks = {{p.stem: load_ydk(p) for p in Path("tests/decks").glob("*.ydk")}}
ref = Duel(3, None, decks["yubel"], decks["tenpai"], config=DuelConfig(max_turns=4)).run(RandomAgent(1), RandomAgent(2))
s = DuelSession(Duel(3, None, decks["yubel"], decks["tenpai"], config=DuelConfig(max_turns=4), snapshots=True))
for i in range(len(ref.actions) // 2):
    s.act(ref.actions[i])
snap = s.snapshot()
x = torch.randn(256, 256)
for _ in range(2):
    s.restore(snap)
    y = (x @ x.T).sum()  # allocate through PyTorch while a duel is live
    while not s.done:
        s.act(ref.actions[s.point.index])
    assert s.result().responses == ref.responses
    assert s.core.arena_escapes() == 0
print("ok", float(y) == float((x @ x.T).sum()))
"""


@pytest.mark.parametrize("order", [("torch", "ygorl._core"), ("ygorl._core", "torch")])
def test_duel_snapshots_and_torch_in_one_process(order):
    out = subprocess.run([sys.executable, "-c", SCRIPT.format(first=order[0], second=order[1])],
                         capture_output=True, text=True, timeout=300)  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok True"
