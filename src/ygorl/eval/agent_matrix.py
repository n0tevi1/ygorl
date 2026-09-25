"""Agent-vs-agent matchup matrix (T4b.8, #83): how strong each agent is.

``build_agent_matrix(agents, decks, pairings=...)`` plays every pair of agents over one fixed sample of ordered deck
pairings ``(d1, d2)`` drawn from a deck pool. Each pairing is played with both deck assignments (x on d1 against y
on d2, then x on d2 against y on d1), each with both players going first once: four games per pairing, so the
pairing's deck and seat advantages cancel inside the cell. Common random numbers: each pairing's two decks
("slots") are shuffled by slot and whoever holds a slot gets that slot's agent seed, never keyed by the seat (which
is set by name order), so an agent sees the same opening hands in every cell. The matrix therefore does not depend on
the order or the names of the agents, adding an agent never changes an existing cell, and two rows are compared on
the same games. All cells share one worker pool.

``win_rate[i][j]`` is agent i's win rate against agent j; draws count half. Games that raised (``exception``) or
that the host stopped as an error (``error``: engine loop, script budget) are not wins, losses or draws: they are
counted in ``errors`` and left out of ``games`` and the win rate.

The meta game is the one of :mod:`ygorl.eval.matchup` (symmetric, zero-sum, payoff ``M - 0.5``): the Nash mixture
and alpha-rank rank the agents without assuming strength is transitive (design C7: matrices, not a single Elo).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ygorl.agents.base import AgentFactory, agent_name
from ygorl.agents.registry import AgentSpec, agent_factory, parse_policy_arg, policy_checkpoint_of
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine.duel import DuelConfig, shuffle_deck
from ygorl.eval.arena import Arena, GameRecord, GameSpec, derive_seed, wilson_interval
from ygorl.eval.matchup import DEFAULT_ALPHA, DEFAULT_POPULATION, _deck_hash, _named, _tuples, alpha_rank, nash_mixture

FORMAT = "ygorl-agent-matrix"
FORMAT_VERSION = 1
ARTIFACT_DIR = "agent-matrix"
ERROR_REASONS = ("exception", "error")  # not a result: counted apart


def sample_pairings(n_decks: int, count: int, seed: int) -> list[tuple[int, int]]:
    """``count`` ordered pairs of different deck indices, drawn with replacement from a seeded generator."""
    if n_decks < 2:
        raise ValueError("a deck pool needs at least two decks")
    if count < 1:
        raise ValueError("at least one deck pairing is needed")
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int]] = []
    while len(out) < count:
        i, j = (int(v) for v in rng.integers(n_decks, size=2))
        if i != j:
            out.append((i, j))
    return out


@dataclass(frozen=True)
class CellResult:
    """Agent x (the first in name order) against agent y over the matrix's games."""

    wins: int
    losses: int
    draws: int
    errors: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.5


def score(records: Sequence[GameRecord]) -> CellResult:
    """Count agent a's results; error games are counted apart."""
    ok = [r for r in records if r.reason not in ERROR_REASONS]
    return CellResult(wins=sum(r.winner == 0 for r in ok), losses=sum(r.winner == 1 for r in ok),
                      draws=sum(r.winner is None for r in ok), errors=len(records) - len(ok))  # fmt: skip


@dataclass(frozen=True)
class AgentMatrix:
    """An agent matchup matrix and its meta game (``nash`` / ``alpha_rank`` indexed like ``agents``)."""

    agents: tuple[str, ...]
    specs: tuple[str, ...]  # how each agent was built (agent spec or factory name)
    win_rate: tuple[tuple[float, ...], ...]
    games: tuple[tuple[int, ...], ...]
    errors: tuple[tuple[int, ...], ...]
    ci_low: tuple[tuple[float, ...], ...]
    ci_high: tuple[tuple[float, ...], ...]
    decks: tuple[str, ...]  # the deck pool
    deck_hashes: tuple[str, ...]
    pairings: tuple[tuple[int, int], ...]  # indices into ``decks``
    seed: int
    max_turns: int
    max_decisions: int
    nash: tuple[float, ...]
    alpha_rank: tuple[float, ...]
    alpha: float = DEFAULT_ALPHA
    population_size: int = DEFAULT_POPULATION
    confidence: float = 0.95
    environment: dict[str, str] | None = None
    fingerprints: tuple[str, ...] = ()  # checkpoint content hash of each agent ("" for rule agents)
    batched: bool = False  # policy-vs-policy cells played on the batched C++ path (ygorl.eval.batched), not the arena

    def __post_init__(self) -> None:
        if not self.fingerprints:  # files written before fingerprints existed
            object.__setattr__(self, "fingerprints", ("",) * len(self.agents))

    def array(self) -> np.ndarray:
        return np.array(self.win_rate, dtype=float)

    def index(self, agent: str) -> int:
        try:
            return self.agents.index(agent)
        except ValueError:
            raise KeyError(f"no agent {agent!r} in the matrix (agents: {', '.join(self.agents)})") from None

    def against(self, opponent: str) -> dict[str, float]:
        """Every other agent's win rate against ``opponent``."""
        j = self.index(opponent)
        return {a: self.win_rate[i][j] for i, a in enumerate(self.agents) if i != j}

    def ranking(self) -> list[str]:
        """Agents by alpha-rank mass, then Nash weight, then mean win rate (strongest first)."""
        m = self.array()
        mean = [(m[i].sum() - 0.5) / max(1, len(self.agents) - 1) for i in range(len(self.agents))]
        order = sorted(
            range(len(self.agents)), key=lambda i: (-self.alpha_rank[i], -self.nash[i], -mean[i], self.agents[i])
        )
        return [self.agents[i] for i in order]

    def total_errors(self) -> int:
        return sum(self.errors[i][j] for i in range(len(self.agents)) for j in range(i + 1, len(self.agents)))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"format": FORMAT, "format_version": FORMAT_VERSION}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            d[k] = [list(r) for r in v] if k in ("win_rate", "games", "errors", "ci_low", "ci_high", "pairings") else (
                list(v) if isinstance(v, tuple) else v)  # fmt: skip
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AgentMatrix:
        if not isinstance(d, Mapping):
            raise ValueError(f"not a {FORMAT} file: the top level is a {type(d).__name__}, not an object")
        if d.get("format") != FORMAT or d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} file: {d.get('format')!r} v{d.get('format_version')!r}")
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("win_rate", "games", "errors", "ci_low", "ci_high", "pairings"):
            kw[k] = _tuples(kw[k])
        for k in ("agents", "specs", "decks", "deck_hashes", "nash", "alpha_rank", "fingerprints"):
            kw[k] = tuple(kw.get(k, ()))
        return cls(**kw)

    def save(self, path: str | Path | None = None, *, env: Environment | None = None, name: str = "agents") -> Path:
        """Write JSON. With ``env``: to ``artifacts/agent-matrix/<name>.json`` of that environment, which must be the
        one the matrix was built under; otherwise to ``path``."""
        if env is not None:
            if path is not None:
                raise ValueError("give either a path or an environment, not both")
            env.check_stamp(self.environment or {})
            path = env.artifact_path(ARTIFACT_DIR, f"{name}.json")
        elif path is None:
            raise ValueError("a path is required when no environment is given")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, env: Environment | None = None) -> AgentMatrix:
        """Read a saved matrix; with ``env``, refuse one built under another environment."""
        matrix = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if env is not None:
            env.check_stamp(matrix.environment or {})
        return matrix


def _named_agents(agents: Mapping[str, AgentFactory] | Sequence[AgentFactory]) -> list[tuple[str, AgentFactory]]:
    items = list(agents.items()) if isinstance(agents, Mapping) else [(agent_name(a), a) for a in agents]
    names = [n for n, _ in items]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError(f"agent names must be non-empty and unique: {names}")
    return items


def pairing_slots(decks: Sequence[Deck], pairings: Sequence[tuple[int, int]], seed: int,
                  config: DuelConfig) -> list[tuple[int, tuple[Deck, Deck], tuple[int, int]]]:  # fmt: skip
    """Per pairing ``k``: its game seed, its two decks (slot 0 = the pairing's first deck) already shuffled, and the
    agent seed of whoever holds each slot. Everything is keyed by the slot, never by the seat, so an agent holding a
    slot sees the same opening hand and gets the same seed in every cell (common random numbers)."""
    out = []
    for k, (i, j) in enumerate(pairings):
        s = derive_seed(seed, k)
        slots = (decks[i], decks[j])
        if config.shuffle_decks:
            slots = tuple(replace(d, main=tuple(shuffle_deck(d.main, s, slot))) for slot, d in enumerate(slots))
        out.append((s, slots, (derive_seed(s, 0), derive_seed(s, 1))))
    return out


def cell_specs(agent_a: AgentFactory, agent_b: AgentFactory, slots, env: Environment | None,
               config: DuelConfig) -> list[GameSpec]:  # fmt: skip
    """The 4 games per pairing of one cell: agent a holds slot ``m`` (0, 1) and slot ``g`` (0, 1) goes first."""
    config = replace(config, shuffle_decks=False)  # decks come shuffled by slot
    specs = []
    for k, (s, decks, seeds) in enumerate(slots):
        for m in (0, 1):
            for g in (0, 1):
                specs.append(GameSpec(pair=k, first=0 if g == m else 1, seed=s, agent_seeds=(seeds[m], seeds[1 - m]),
                                      deck_a=decks[m], deck_b=decks[1 - m], agent_a=agent_a, agent_b=agent_b, env=env,
                                      config=config))  # fmt: skip
    return specs


def agent_fingerprint(spec: str) -> str:
    """Content hash of the checkpoint a ``policy`` agent spec plays (through any ``lethal:`` wrapper), "" otherwise:
    the same name with other weights must not be mistaken for an agent already in a matrix."""
    inner = spec
    while inner.startswith("lethal:"):
        inner = inner[len("lethal:") :]
    try:
        path = policy_checkpoint_of(inner)
    except ValueError:
        return ""
    if path is None or not Path(path).is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()[:16]


def _play(old: AgentMatrix | None, rows: list[tuple[str, str, str, AgentFactory, bool]],
          pool: list[tuple[str, Deck]], sample: list[tuple[int, int]], seed: int, env: Environment | None,
          config: DuelConfig, workers: int, confidence: float, alpha: float, population_size: int,
          device: str | None = None) -> AgentMatrix:  # fmt: skip
    """The matrix of ``rows`` = (name, spec, fingerprint, factory, is new), in name order: cells between two agents of
    ``old`` are copied from it, every cell with a new agent is played."""
    names = [r[0] for r in rows]
    n = len(rows)
    deck_list = [d for _, d in pool]
    slots = pairing_slots(deck_list, sample, seed, config)
    cells = [(i, j) for i in range(n) for j in range(i + 1, n) if rows[i][4] or rows[j][4]]
    per_cell = 4 * len(sample)
    played: dict[tuple[int, int], list[GameRecord]] = {}
    if device is not None:
        played.update(_play_batched(cells, rows, slots, config, device))
    rest = [c for c in cells if c not in played]
    specs = [sp for i, j in rest for sp in cell_specs(rows[i][3], rows[j][3], slots, env, config)]
    records = Arena(rows[0][3], rows[1][3], env=env, config=config, workers=workers).play(specs)  # one pool
    for c, cell in enumerate(rest):
        played[cell] = records[c * per_cell : (c + 1) * per_cell]

    rate, count, errs = np.full((n, n), 0.5), np.zeros((n, n), dtype=int), np.zeros((n, n), dtype=int)
    lo, hi = np.full((n, n), 0.5), np.full((n, n), 0.5)
    if old is not None:
        for a in old.agents:
            for b in old.agents:
                i, j, oi, oj = names.index(a), names.index(b), old.index(a), old.index(b)
                rate[i, j], count[i, j], errs[i, j] = old.win_rate[oi][oj], old.games[oi][oj], old.errors[oi][oj]
                lo[i, j], hi[i, j] = old.ci_low[oi][oj], old.ci_high[oi][oj]
    for i, j in cells:
        cell = score(played[i, j])
        rate[i, j], rate[j, i] = cell.win_rate, 1.0 - cell.win_rate
        count[i, j] = count[j, i] = cell.games
        errs[i, j] = errs[j, i] = cell.errors
        lo[i, j], hi[i, j] = wilson_interval(cell.wins + 0.5 * cell.draws, cell.games, confidence)
        lo[j, i], hi[j, i] = 1.0 - hi[i, j], 1.0 - lo[i, j]
    return AgentMatrix(
        agents=tuple(names), specs=tuple(r[1] for r in rows), fingerprints=tuple(r[2] for r in rows),
        win_rate=_tuples(rate.tolist()), games=_tuples(count.tolist()), errors=_tuples(errs.tolist()),
        ci_low=_tuples(lo.tolist()), ci_high=_tuples(hi.tolist()),
        decks=tuple(name for name, _ in pool), deck_hashes=tuple(_deck_hash(d) for d in deck_list),
        pairings=tuple(tuple(p) for p in sample), seed=seed, max_turns=config.max_turns,
        max_decisions=config.max_decisions, nash=tuple(float(v) for v in nash_mixture(rate)),
        alpha_rank=tuple(float(v) for v in alpha_rank(rate, alpha=alpha, population_size=population_size)),
        alpha=alpha, population_size=population_size, confidence=confidence,
        environment=env.stamp() if env is not None else None, batched=device is not None,
    )  # fmt: skip


def policy_setting(spec: str) -> tuple[str, bool, float] | None:
    """``(checkpoint, greedy, temperature)`` of a plain ``policy`` / ``policy-greedy`` spec (no wrapper), else None."""
    name, _, arg = spec.partition(":")
    if name not in ("policy", "policy-greedy") or not arg:
        return None
    return parse_policy_arg(arg if name == "policy" else f"{arg}@greedy")


def _play_batched(cells, rows, slots, config: DuelConfig, device: str, envs: int = 128,
                  threads: int = 4) -> dict[tuple[int, int], list[GameRecord]]:  # fmt: skip
    """The cells whose two agents are plain policy checkpoints with the same card vocab and event length, played on
    the batched C++ path (``play_policies``): the arena's games (slots, seeds, first players), decisions sampled from
    each agent's slot seed, statistics counted the same way. Other cells are left to the arena."""
    import torch

    from ygorl.env import GameSpec as EnvSpec
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.eval.batched import play_policies
    from ygorl.train.checkpoint import load_actor, vocab_passwords

    loaded: dict[str, Any] = {}

    def policy(spec: str):
        setting = policy_setting(spec)
        if setting is None:
            return None
        if setting[0] not in loaded:
            pol = load_actor(setting[0])
            loaded[setting[0]] = (pol, pol.net.to(torch.device(device)), tuple(vocab_passwords(pol.vocab)))
        return loaded[setting[0]], setting

    groups: dict[tuple, list] = {}
    for i, j in cells:
        a, b = policy(rows[i][1]), policy(rows[j][1])
        if a is None or b is None:
            continue
        (pa, _, va), (pb, _, vb) = a[0], b[0]
        if va != vb or pa.event_length != pb.event_length:
            continue
        groups.setdefault((va, pa.event_length), []).append((i, j, a, b))
    base = replace(config, shuffle_decks=False)
    out: dict[tuple[int, int], list[GameRecord]] = {}
    for (_, event_length), group in groups.items():
        env = EncodedVecEnv(min(envs, 4 * len(slots)), threads, vocab=group[0][2][0][0].vocab,
                            event_length=event_length, skip_forced=True)  # fmt: skip
        for i, j, (ca, sa), (cb, sb) in group:
            specs, seeds = [], []
            for s, decks, slot_seeds in slots:
                for m in (0, 1):
                    for g in (0, 1):
                        specs.append(EnvSpec(seed=s, deck_a=decks[m], deck_b=decks[1 - m], first=0 if g == m else 1,
                                             config=base))  # fmt: skip
                        seeds.append((slot_seeds[m], slot_seeds[1 - m]))
            records, _ = play_policies(env, specs, ca[1], cb[1], device=device, pairs_per_spec=4, sample_seeds=seeds,
                                       sampling=((sa[1], sa[2]), (sb[1], sb[2])))  # fmt: skip
            out[i, j] = records
    return out


def spec_of(factory: AgentFactory) -> str:
    """How an agent was built: the agent spec of an :class:`AgentSpec` (rebuildable from the registry), otherwise
    ``factory:<name>`` (a class, function or partial whose settings the name does not capture: never rebuilt)."""
    return factory.spec if isinstance(factory, AgentSpec) else f"factory:{agent_name(factory)}"


def _row(name: str, factory: AgentFactory, new: bool) -> tuple[str, str, str, AgentFactory, bool]:
    spec = spec_of(factory)
    return (name, spec, agent_fingerprint(spec), factory, new)


def build_agent_matrix(agents: Mapping[str, AgentFactory] | Sequence[AgentFactory],
                       decks: Sequence[Deck] | Mapping[str, Deck], *, pairings: int, seed: int = 0,
                       env: Environment | None = None, config: DuelConfig | None = None, workers: int = 1,
                       confidence: float = 0.95, alpha: float = DEFAULT_ALPHA,
                       population_size: int = DEFAULT_POPULATION, device: str | None = None) -> AgentMatrix:  # fmt: skip
    """Play every pair of ``agents`` on ``pairings`` sampled deck pairings (4 games each) and solve the meta game.

    With ``device`` (e.g. "cuda"), cells between two plain policy checkpoints of the same vocab and event length are
    played on the batched C++ path with the networks on that device; everything else goes through the arena."""
    items = _named_agents(agents)
    if len(items) < 2:
        raise ValueError("an agent matrix needs at least two agents")
    pool = _named(decks)
    sample = sample_pairings(len(pool), pairings, seed)
    config = config or (DuelConfig.from_environment(env) if env is not None else DuelConfig())
    rows = sorted((_row(name, f, True) for name, f in items), key=lambda r: r[0])
    return _play(None, rows, pool, sample, seed, env, config, workers, confidence, alpha, population_size, device)


def extend_agent_matrix(matrix: AgentMatrix, agents: Mapping[str, AgentFactory] | Sequence[AgentFactory],
                        decks: Sequence[Deck] | Mapping[str, Deck], *, env: Environment | None = None,
                        config: DuelConfig | None = None, workers: int = 1,
                        device: str | None = None) -> AgentMatrix:  # fmt: skip
    """``matrix`` plus the agents not in it yet: only their cells are played, on the matrix's own deck pairings,
    seed and rules, so the result equals building all agents at once. A matrix built with the batched path is
    extended with it too (``device`` defaults to "cpu" then; the arena is never used for its policy cells). An agent already in the matrix under the
    same name is skipped if its spec and checkpoint content are the same, an error otherwise."""
    if env is not None:
        env.check_stamp(matrix.environment or {})
    elif matrix.environment is not None:
        raise ValueError(f"the matrix belongs to environment {matrix.environment.get('environment')!r}: pass it")
    pool = _named(decks)
    if tuple(n for n, _ in pool) != matrix.decks or tuple(_deck_hash(d) for _, d in pool) != matrix.deck_hashes:
        raise ValueError("the deck pool differs from the one the matrix was built on (names, order or contents)")
    config = config or (DuelConfig.from_environment(env) if env is not None else DuelConfig())
    if (config.max_turns, config.max_decisions) != (matrix.max_turns, matrix.max_decisions):
        raise ValueError(f"rules differ from the matrix's: max_turns {config.max_turns} vs {matrix.max_turns}, "
                         f"max_decisions {config.max_decisions} vs {matrix.max_decisions}")  # fmt: skip
    given = dict(_named_agents(agents))
    rows = []
    for i, name in enumerate(matrix.agents):  # agents already in the matrix: rebuilt from their spec
        spec, fp = matrix.specs[i], matrix.fingerprints[i]
        now = agent_fingerprint(spec)
        if fp and not now:
            raise ValueError(f"the checkpoint of agent {name!r} ({spec!r}) is not found from here "
                             "(a relative path resolves against the current directory)")  # fmt: skip
        if fp and now != fp:
            raise ValueError(f"the checkpoint of agent {name!r} ({spec!r}) changed since it entered the matrix")
        fp = fp or now  # a file written before fingerprints existed: not checkable, recorded from now on
        if name in given:
            if spec_of(given[name]) != spec or (
                matrix.fingerprints[i] and agent_fingerprint(spec_of(given[name])) != fp
            ):
                raise ValueError(f"agent {name!r} is already in the matrix as {spec!r} ({fp or 'no checkpoint'}): "
                                 "give the new one another name")  # fmt: skip
            factory = given.pop(name)
        else:
            try:
                factory = agent_factory(spec)
            except ValueError as exc:
                raise ValueError(f"agent {name!r} ({spec!r}) cannot be rebuilt from its spec ({exc}): "
                                 "pass it with the new agents") from None  # fmt: skip
        rows.append((name, spec, fp, factory, False))
    if not given:
        return matrix
    rows = sorted(rows + [_row(name, f, True) for name, f in given.items()], key=lambda r: r[0])
    if matrix.batched:
        device = device or "cpu"
    elif device is not None:
        raise ValueError("the matrix was played without the batched path: extend it without a device")
    return _play(matrix, rows, pool, [tuple(p) for p in matrix.pairings], matrix.seed, env, config, workers,
                 matrix.confidence, matrix.alpha, matrix.population_size, device)  # fmt: skip


__all__ = [
    "AgentMatrix",
    "CellResult",
    "agent_fingerprint",
    "build_agent_matrix",
    "cell_specs",
    "extend_agent_matrix",
    "pairing_slots",
    "sample_pairings",
    "score",
]
