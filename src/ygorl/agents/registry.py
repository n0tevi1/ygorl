"""Named agent factories, so command-line tools can take ``--policy <spec>``.

A spec is ``name`` or ``name:arg`` (e.g. a checkpoint path, later). A factory
is called as ``factory(arg, seed)`` with ``arg=None`` when the spec has no
argument, and returns a fresh agent. Register more agents (policy
checkpoints) with :func:`register_agent`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ygorl.agents.base import Agent
from ygorl.agents.greedy import GreedyAgent
from ygorl.agents.random_agent import RandomAgent

AgentFactory = Callable[[str | None, int], Agent]

_REGISTRY: dict[str, tuple[AgentFactory, str]] = {}


def register_agent(name: str, factory: AgentFactory | None, description: str = "") -> None:
    """Register ``factory`` under ``name``; ``factory=None`` removes the entry."""
    if factory is None:
        _REGISTRY.pop(name, None)
        return
    if not name or ":" in name:
        raise ValueError(f"invalid agent name {name!r}")
    if name in _REGISTRY:
        raise ValueError(f"agent {name!r} is already registered")
    _REGISTRY[name] = (factory, description)


def available_agents() -> dict[str, str]:
    """Registered agent names and their descriptions."""
    return {name: desc for name, (_, desc) in sorted(_REGISTRY.items())}


def make_agent(spec: str, seed: int = 0) -> Agent:
    """Build the agent named by ``spec`` (``name`` or ``name:arg``), seeded with ``seed``."""
    name, sep, arg = spec.partition(":")
    if name not in _REGISTRY:
        raise ValueError(f"unknown agent {name!r}; available: {', '.join(available_agents())}")
    factory, _ = _REGISTRY[name]
    return factory(arg if sep else None, seed)


@dataclass(frozen=True)
class AgentSpec:
    """Picklable :data:`~ygorl.agents.base.AgentFactory` for a spec: ``AgentSpec("greedy")(seed)``.

    ``name`` is the spec, so arena reports and matrices show it. Workers of a
    parallel arena rebuild agents by name, so the spec must be registered in
    them too (true for the built-in agents and anything registered at import).
    """

    spec: str

    @property
    def name(self) -> str:
        return self.spec

    def __call__(self, seed: int) -> Agent:
        return make_agent(self.spec, seed)


def agent_factory(spec: str) -> AgentSpec:
    """A named, picklable factory ``seed -> agent`` for ``spec``; a bad spec raises ``ValueError`` here."""
    make_agent(spec, 0)
    return AgentSpec(spec)


def _random(arg: str | None, seed: int) -> Agent:
    if arg is not None:
        raise ValueError("random takes no argument (use the seed option)")
    return RandomAgent(seed)


def _greedy(arg: str | None, seed: int) -> Agent:
    if arg is not None:
        raise ValueError("greedy takes no argument (use the seed option)")
    return GreedyAgent(seed)


register_agent("random", _random, "uniformly random legal actions (seeded)")
register_agent("greedy", _greedy, "one-ply heuristic baseline (docs/evaluation.md)")

__all__ = ["AgentFactory", "AgentSpec", "agent_factory", "available_agents", "make_agent", "register_agent"]
