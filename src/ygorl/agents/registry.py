"""Named agent factories, so command-line tools can take ``--policy <spec>``.

A spec is ``name`` or ``name:arg`` (e.g. a checkpoint path). A factory
is called as ``factory(arg, seed)`` with ``arg=None`` when the spec has no
argument, and returns a fresh agent. Register more agents with
:func:`register_agent`; a *wrapper* (``wraps=True``, e.g. ``lethal``) takes
another agent spec as its argument.

This module is the only place that reads spec strings: :func:`parse_spec`
turns one into a :class:`ParsedSpec` (kind, checkpoint and sampling of a
policy, inner spec of a wrapper), :func:`policy_spec` writes the spec of a
policy checkpoint, and :func:`parse_named_spec` reads the ``NAME=SPEC``
arguments of ``ygorl strength``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ygorl.agents.base import Agent
from ygorl.agents.greedy import GreedyAgent
from ygorl.agents.random_agent import RandomAgent

AgentFactory = Callable[[str | None, int], Agent]

POLICY_KINDS = ("policy", "policy-greedy")  # agents that play a policy checkpoint (policy:PATH[@greedy][@t=T])


@dataclass(frozen=True)
class _Entry:
    factory: AgentFactory
    description: str
    wraps: bool  # the argument is another agent spec


_REGISTRY: dict[str, _Entry] = {}


def register_agent(name: str, factory: AgentFactory | None, description: str = "", *, wraps: bool = False) -> None:
    """Register ``factory`` under ``name``; ``factory=None`` removes the entry. ``wraps=True``: the agent wraps
    another one, and its argument is that agent's spec (``name:<agent spec>``)."""
    if factory is None:
        _REGISTRY.pop(name, None)
        return
    if not name or ":" in name:
        raise ValueError(f"invalid agent name {name!r}")
    if name in _REGISTRY:
        raise ValueError(f"agent {name!r} is already registered")
    _REGISTRY[name] = _Entry(factory, description, wraps)


def available_agents() -> dict[str, str]:
    """Registered agent names and their descriptions."""
    return {name: e.description for name, e in sorted(_REGISTRY.items())}


@dataclass(frozen=True)
class ParsedSpec:
    """An agent spec, parsed by :func:`parse_spec`.

    ``kind`` is the registered agent name and ``arg`` the text after the first ':' (None without one). A policy
    (:data:`POLICY_KINDS`) has its ``checkpoint`` path and sampling (``greedy``: argmax; else ``temperature``); a
    wrapper has the parsed spec it wraps in ``inner``."""

    spec: str
    kind: str
    arg: str | None = None
    checkpoint: str | None = None
    greedy: bool = False
    temperature: float = 1.0
    inner: ParsedSpec | None = None

    @property
    def is_policy(self) -> bool:
        """A plain policy checkpoint (not wrapped)."""
        return self.checkpoint is not None

    @property
    def policy(self) -> ParsedSpec | None:
        """The policy this agent plays, through any wrappers (``lethal:policy:P`` -> ``policy:P``); None if none."""
        p: ParsedSpec | None = self
        while p is not None and not p.is_policy:
            p = p.inner
        return p


def parse_spec(spec: str) -> ParsedSpec:
    """Parse an agent spec (``name``, ``name:arg``, ``policy:PATH[@greedy][@t=T]``, ``lethal:<agent spec>``); an
    unknown agent, a missing checkpoint or inner agent and a bad sampling option raise ``ValueError``. The
    checkpoint file itself is not read."""
    kind, sep, arg = spec.partition(":")
    entry = _REGISTRY.get(kind)
    if entry is None:
        raise ValueError(f"unknown agent {kind!r}; available: {', '.join(available_agents())}")
    if kind in POLICY_KINDS:
        path, greedy, temperature = _policy_arg(arg)
        return ParsedSpec(spec, kind, arg, path, greedy or kind == "policy-greedy", temperature)
    if entry.wraps:
        if not arg:
            raise ValueError(f"{kind} needs an inner agent: {kind}:<agent spec>")
        return ParsedSpec(spec, kind, arg, inner=parse_spec(arg))
    return ParsedSpec(spec, kind, arg if sep else None)


def policy_spec(checkpoint: str | Path, *, greedy: bool = False, temperature: float = 1.0) -> str:
    """The agent spec of a policy checkpoint: ``policy:PATH[@greedy][@t=T]`` (``parse_spec`` reads it back)."""
    if not temperature > 0:
        raise ValueError(f"policy temperature must be positive, got {temperature!r}")
    return f"policy:{checkpoint}" + ("@greedy" if greedy else "") + (f"@t={temperature}" if temperature != 1.0 else "")


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_named_spec(arg: str) -> tuple[str, str]:
    """``[NAME=]SPEC`` -> ``(name, spec)``; the name defaults to the spec. A ``NAME=`` prefix has no ':' (specs like
    ``policy:P@t=1`` do) and is letters, digits, '.', '_' and '-', not starting with '.' or '-'. The spec itself is
    not checked here."""
    name, sep, spec = arg.partition("=")
    if not sep or ":" in name:
        return arg, arg
    if not spec:
        raise ValueError(f"agent {arg!r} has an empty spec")
    if not NAME_RE.match(name):
        raise ValueError(f"agent name {name!r}: letters, digits, '.', '_' and '-', not starting with '.' or '-'")
    return name, spec


def make_agent(spec: str, seed: int = 0) -> Agent:
    """Build the agent named by ``spec`` (``name`` or ``name:arg``), seeded with ``seed``."""
    parsed = parse_spec(spec)
    return _REGISTRY[parsed.kind].factory(parsed.arg, seed)


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


def _policy_arg(arg: str | None) -> tuple[str, bool, float]:
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


def _policy(arg: str | None, seed: int, greedy: bool = False) -> Agent:
    """``policy:PATH[@greedy][@t=T]``: a PPO training checkpoint (``ygorl.train.checkpoint``, followed through a
    lockstep C++ host) or a policy checkpoint (``ygorl.nets.agent``, e.g. from BC), told apart by the file's
    ``format`` field. Sampling at temperature 1 by default."""
    path, suffix_greedy, temperature = _policy_arg(arg)
    greedy = greedy or suffix_greedy
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
    return _policy(arg, seed, greedy=True)


register_agent("random", _random, "uniformly random legal actions (seeded)")
register_agent("greedy", _greedy, "one-ply heuristic baseline (docs/evaluation.md)")
register_agent(
    "policy",
    _policy,
    "a policy checkpoint (PPO training or BC): policy:PATH[@greedy][@t=T] (docs/evaluation.md, docs/bc.md)",
)
register_agent("policy-greedy", _policy_greedy, "shorthand for policy:PATH@greedy")


def _lethal(arg: str | None, seed: int) -> Agent:
    from ygorl.agents.lethal import make_lethal_agent

    return make_lethal_agent(arg, seed)


register_agent("lethal", _lethal, "any agent plus a lethal search this turn: lethal:<agent spec> (#62, prototype)",
               wraps=True)  # fmt: skip

__all__ = ["POLICY_KINDS", "AgentFactory", "AgentSpec", "ParsedSpec", "agent_factory", "available_agents",
           "make_agent", "parse_named_spec", "parse_spec", "policy_spec", "register_agent"]  # fmt: skip
