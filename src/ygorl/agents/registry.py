"""Named agent factories, so command-line tools can take ``--policy <spec>``.

A spec is ``name`` or ``name:arg`` (e.g. a checkpoint path, later). A factory
is called as ``factory(arg, seed)`` with ``arg=None`` when the spec has no
argument, and returns a fresh agent. Register more agents (policy
checkpoints) with :func:`register_agent`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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


def _policy(arg: str | None, seed: int) -> Agent:
    """``policy:PATH[@greedy][@t=T]``: a PPO training checkpoint (``ygorl.train.checkpoint``, followed through a
    lockstep C++ host) or a policy checkpoint (``ygorl.nets.agent``, e.g. from BC), told apart by the file's
    ``format`` field. Sampling at temperature 1 by default."""
    if not arg:
        raise ValueError("policy needs a checkpoint: policy:PATH[@greedy][@t=T]")
    path, *opts = arg.split("@")
    greedy, temperature = False, 1.0
    for opt in opts:
        if opt == "greedy":
            greedy = True
        elif opt.startswith("t="):
            temperature = float(opt[2:])
            if not temperature > 0:
                raise ValueError(f"policy temperature must be positive, got {opt!r}")
        else:
            raise ValueError(f"unknown policy option {opt!r} (use @greedy or @t=T)")
    try:
        from ygorl.nets.agent import CHECKPOINT_FORMAT, policy_agent_factory
        from ygorl.train.checkpoint import FORMAT as PPO_FORMAT
    except ImportError as exc:  # PyTorch is the optional ``train`` extra
        raise ValueError(f"policy needs PyTorch (uv sync --extra train): {exc}") from None
    fmt = _checkpoint_format(path)
    if fmt == PPO_FORMAT:
        from ygorl.agents.checkpoint import make_policy_agent

        return make_policy_agent(path, seed, greedy=greedy, temperature=temperature)
    if fmt == CHECKPOINT_FORMAT:
        return policy_agent_factory(arg, seed)
    raise ValueError(f"{path}: not a ygorl checkpoint (format {fmt!r})")


def _checkpoint_format(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no policy checkpoint at {path}")
    return _format_of(str(p.resolve()), p.stat().st_mtime_ns)


@functools.lru_cache(maxsize=16)
def _format_of(path: str, mtime_ns: int) -> str | None:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)  # mmap: tensors are not read
    return data.get("format") if isinstance(data, dict) else None


def _policy_greedy(arg: str | None, seed: int) -> Agent:
    return _policy(f"{arg}@greedy" if arg else arg, seed)


register_agent("random", _random, "uniformly random legal actions (seeded)")
register_agent("greedy", _greedy, "one-ply heuristic baseline (docs/evaluation.md)")
register_agent("policy", _policy,
               "a policy checkpoint (PPO training or BC): policy:PATH[@greedy][@t=T] (docs/evaluation.md, docs/bc.md)")
register_agent("policy-greedy", _policy_greedy, "shorthand for policy:PATH@greedy")

__all__ = ["AgentFactory", "AgentSpec", "agent_factory", "available_agents", "make_agent", "register_agent"]
