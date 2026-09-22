"""Agent protocol: given a decision point, return the index of a legal action.

Contract (what :class:`ygorl.engine.duel.Duel` and :class:`ygorl.eval.arena.Arena`
rely on):

* ``act(point)`` returns an ``int`` in ``range(len(point.actions))``. Every
  action offered is legal, so any index is a valid answer; the duel raises
  ``ValueError`` for anything else.
* One call = one step. Multi-selects are asked element by element, so an agent
  may see several consecutive points for the same ``point.decision``.
* ``point.player`` is the *engine* player (0 moves first), not the deck side.
* Agents are stateful (RNG, per-turn memory). Use one instance per duel and
  seat; build them from an :data:`AgentFactory` so that a seed fully determines
  play. Given the same seed and the same duel, an agent must make the same
  choices (reproducible games, worker-count independent arenas).
* Optionally an agent exposes ``last_probs``: the probabilities it assigned to
  ``point.actions`` at its last call (recorded by ``Duel(record_steps=True)``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ygorl.engine.duel import DecisionPoint


@runtime_checkable
class Agent(Protocol):
    def act(self, point: DecisionPoint) -> int:
        """Return an index into ``point.actions``."""
        ...


AgentFactory = Callable[[int], Agent]
"""Builds a fresh agent from a seed. Must be picklable to be used by a parallel arena
(a class such as ``GreedyAgent``, a module-level function or a ``functools.partial``)."""


def agent_name(factory_or_agent: object) -> str:
    """Readable name of an agent class, instance, function or ``functools.partial``."""
    obj = getattr(factory_or_agent, "func", factory_or_agent)  # functools.partial
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(obj, type) or callable(obj) and hasattr(obj, "__qualname__"):
        return obj.__qualname__
    return type(obj).__name__
