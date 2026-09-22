"""Single-duel environment with a reset/step interface (T2.1)."""

from __future__ import annotations

from ygorl.engine.duel import DecisionPoint, DuelResult
from ygorl.env.pool import GameSpec, VecDuelEnv


class DuelEnv:
    """``reset(spec) -> point``; ``step(action) -> (point, None)`` or ``(None, result)`` when the game ends."""

    def __init__(self, cards=None, scripts=None, record_steps: bool = False) -> None:
        self._vec = VecDuelEnv(1, 1, cards, scripts, record_steps)
        self.result: DuelResult | None = None

    def reset(self, spec: GameSpec) -> DecisionPoint | None:
        self.result = None
        self._vec.start(0, spec)
        return self._next()

    def step(self, action: int, probs=None) -> tuple[DecisionPoint | None, DuelResult | None]:
        self._vec.act(0, action, probs)
        point = self._next()
        return point, self.result

    def _next(self) -> DecisionPoint | None:
        (event,) = self._vec.recv(min_events=1)
        if event.result is not None:
            self.result = event.result
            return None
        return event.point
