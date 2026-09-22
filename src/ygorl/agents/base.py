"""Agent protocol: given a decision point, return the index of a legal action."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ygorl.engine.duel import DecisionPoint


@runtime_checkable
class Agent(Protocol):
    def act(self, point: DecisionPoint) -> int:
        """Return an index into ``point.actions``."""
        ...
