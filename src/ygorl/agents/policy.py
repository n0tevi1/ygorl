"""``PolicyAgent``: turn a scoring function over the legal actions into an agent.

The policy is any callable ``point -> scores`` returning one real-valued score
(logit) per entry of ``point.actions``. The agent plays ``softmax(scores / T)``
with its own seeded RNG, or the argmax with ``greedy=True``. The learned
policies of M4 plug in here; ``last_probs`` is recorded by
``Duel(record_steps=True)``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ygorl.engine.duel import DecisionPoint

Policy = Callable[["DecisionPoint"], Iterable[float]]


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    top = max(scores)
    exps = [math.exp((s - top) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


class PolicyAgent:
    name = "policy"

    def __init__(self, policy: Policy, *, seed: int | None = None, greedy: bool = False, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.policy = policy
        self.rng = random.Random(seed)
        self.greedy = greedy
        self.temperature = temperature
        self.last_probs: list[float] | None = None

    def act(self, point: DecisionPoint) -> int:
        scores = [float(s) for s in self.policy(point)]
        n = len(point.actions)
        if len(scores) != n:
            raise ValueError(f"policy returned {len(scores)} scores for {n} actions")
        probs = softmax(scores, self.temperature)
        self.last_probs = probs
        if self.greedy:
            return max(range(n), key=lambda i: (scores[i], -i))
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        return max(i for i, p in enumerate(probs) if p > 0)  # rounding: r >= sum(probs)
