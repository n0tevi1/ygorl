"""Named agent factories, so command-line tools can take ``--policy <spec>``.

A spec is ``name`` or ``name:arg`` (e.g. a checkpoint path, later). A factory
is called as ``factory(arg, seed)`` with ``arg=None`` when the spec has no
argument, and returns a fresh agent. Register more agents (policy
checkpoints) with :func:`register_agent`.
"""

from __future__ import annotations

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


def parse_policy_arg(arg: str | None) -> tuple[str, bool, float]:
    """``PATH[@greedy][@t=T]`` -> ``(path, greedy, temperature)``; options are peeled from the right, so a path may
    itself contain ``@``."""
    if not arg:
        raise ValueError("policy needs a checkpoint: policy:PATH[@greedy][@t=T]")
    path, greedy, temperature = arg, False, 1.0
    while "@" in path:
        head, _, opt = path.rpartition("@")
        if opt == "greedy":
            greedy = True
        elif opt.startswith("t="):
            try:
                temperature = float(opt[2:])
            except ValueError:
                raise ValueError(f"bad policy temperature {opt!r}") from None
            if not temperature > 0:
                raise ValueError(f"policy temperature must be positive, got {opt!r}")
        else:
            break
        path = head
    if not Path(path).is_file() and "@" in path:
        raise ValueError(f"unknown policy option {path.rpartition('@')[2]!r} (use @greedy or @t=T)")
    return path, greedy, temperature


def policy_checkpoint_of(spec: str) -> str | None:
    """The checkpoint path of a ``policy`` / ``policy-greedy`` agent spec, None for other agents."""
    name, _, arg = spec.partition(":")
    return parse_policy_arg(arg)[0] if name in ("policy", "policy-greedy") else None


def _policy(arg: str | None, seed: int) -> Agent:
    """``policy:PATH[@greedy][@t=T]``: a PPO training checkpoint (``ygorl.train.checkpoint``, followed through a
    lockstep C++ host) or a policy checkpoint (``ygorl.nets.agent``, e.g. from BC), told apart by the file's
    ``format`` field. Sampling at temperature 1 by default."""
    path, greedy, temperature = parse_policy_arg(arg)
    try:
        from ygorl.agents.policy import PolicyAgent
        from ygorl.nets.agent import CHECKPOINT_FORMAT, NetPolicy
        from ygorl.train.checkpoint import FORMAT as PPO_FORMAT
        from ygorl.train.checkpoint import checkpoint_format
    except ImportError as exc:  # PyTorch is the optional ``train`` extra
        raise ValueError(f"policy needs PyTorch (uv sync --extra train): {exc}") from None
    fmt = checkpoint_format(path)
    if fmt == PPO_FORMAT:
        from ygorl.agents.checkpoint import make_policy_agent

        return make_policy_agent(path, seed, greedy=greedy, temperature=temperature)
    if fmt == CHECKPOINT_FORMAT:
        return PolicyAgent(NetPolicy.from_checkpoint(path), seed=seed, greedy=greedy, temperature=temperature)
    raise ValueError(f"{path}: not a ygorl policy checkpoint (format {fmt!r})")


def _policy_greedy(arg: str | None, seed: int) -> Agent:
    return _policy(f"{arg}@greedy" if arg else arg, seed)


register_agent("random", _random, "uniformly random legal actions (seeded)")
register_agent("greedy", _greedy, "one-ply heuristic baseline (docs/evaluation.md)")
register_agent("policy", _policy,
               "a policy checkpoint (PPO training or BC): policy:PATH[@greedy][@t=T] (docs/evaluation.md, docs/bc.md)")
register_agent("policy-greedy", _policy_greedy, "shorthand for policy:PATH@greedy")

__all__ = ["AgentFactory", "AgentSpec", "agent_factory", "available_agents", "make_agent", "register_agent"]
