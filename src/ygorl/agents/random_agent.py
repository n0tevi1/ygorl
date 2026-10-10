"""Uniformly random legal actions (seeded, reproducible); ``choosable_only`` skips the masked no-op undos."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ygorl.engine.duel import DecisionPoint


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None, *, choosable_only: bool = False) -> None:
        self.rng = random.Random(seed)
        # pick among DecisionPoint.choosable, like a network policy (off by default: the baseline's numbers predate it)
        self.choosable_only = choosable_only

    def act(self, point: DecisionPoint) -> int:
        if self.choosable_only and point.undo:
            return self.rng.choice(point.choosable)
        return self.rng.randrange(len(point.actions))
