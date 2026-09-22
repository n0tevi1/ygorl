"""Uniformly random legal actions (seeded, reproducible)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ygorl.engine.duel import DecisionPoint


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, point: DecisionPoint) -> int:
        return self.rng.randrange(len(point.actions))
