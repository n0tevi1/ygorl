"""Deck-vs-deck matchup matrix and its meta-game: Nash mixture and alpha-rank.

``build_matrix(decks, agent, pairs=...)`` plays every unordered pair of decks
with the same agent factory piloting both sides (an :class:`Arena` with
``2 * pairs`` paired games per cell) and fills a win-rate matrix
``M[i][j]`` = deck i's win rate against deck j (draws count half), with
``M[j][i] = 1 - M[i][j]`` and 0.5 on the diagonal (mirrors are not played).
Each cell is played with the two decks in name order and a seed derived from
``seed`` and both deck names, so the matrix does not depend on the order of the
deck list and adding a deck does not change existing cells.

The meta game is symmetric and zero-sum with payoff ``M - 0.5``:

* :func:`nash_mixture` solves it with nashpy's linear program: a deck mixture
  that no single deck beats on average.
* :func:`alpha_rank` is single-population alpha-rank (Omidshafiei et al.,
  2019): a population of ``population_size`` players each using one deck; a
  mutant deck ``r`` fixates in a monomorphic population of ``s`` with the
  Fermi-process probability ``rho(r, s)``; the Markov chain over monomorphic
  states moves ``s -> r`` with probability ``rho(r, s) / (n - 1)`` and its
  stationary distribution ranks the decks. ``alpha`` is the selection
  intensity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nashpy
import numpy as np

from ygorl.agents.base import AgentFactory, agent_name
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine.duel import DuelConfig
from ygorl.eval.arena import Arena, derive_seed

FORMAT = "ygorl-matchup"
FORMAT_VERSION = 1
ARTIFACT_DIR = "matrix"
DEFAULT_ALPHA = 10.0
DEFAULT_POPULATION = 50


# ------------------------------------------------------------------ meta-game solvers


def _as_matrix(win_rate) -> np.ndarray:
    m = np.asarray(win_rate, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise ValueError(f"expected a non-empty square win-rate matrix, got shape {m.shape}")
    return m


def nash_mixture(win_rate) -> np.ndarray:
    """Nash mixture of the symmetric zero-sum game with payoff ``win_rate - 0.5``."""
    m = _as_matrix(win_rate)
    payoff = m - 0.5
    row, _ = nashpy.Game(payoff).linear_program()
    x = np.clip(np.asarray(row, dtype=float), 0.0, None)
    x[x < 1e-12] = 0.0
    return x / x.sum()


def fixation_probability(win_rate, mutant: int, resident: int, *, alpha: float = DEFAULT_ALPHA,
                         population_size: int = DEFAULT_POPULATION) -> float:  # fmt: skip
    """Probability that one ``mutant`` takes over a population of ``resident`` players.

    With ``k`` mutants among ``m`` players (each meets one of the other ``m - 1``):
    ``f_r(k) = ((k-1) M[r,r] + (m-k) M[r,s]) / (m-1)``,
    ``f_s(k) = (k M[s,r] + (m-k-1) M[s,s]) / (m-1)`` and
    ``rho = 1 / (1 + sum_{l=1}^{m-1} exp(-alpha sum_{k=1}^{l} (f_r(k) - f_s(k))))``.
    """
    m = _as_matrix(win_rate)
    n = population_size
    if n < 2:
        raise ValueError("population_size must be at least 2")
    r, s = mutant, resident
    k = np.arange(1, n)
    f_r = ((k - 1) * m[r, r] + (n - k) * m[r, s]) / (n - 1)
    f_s = (k * m[s, r] + (n - k - 1) * m[s, s]) / (n - 1)
    exponents = -alpha * np.cumsum(f_r - f_s)
    # 1 / (1 + sum exp(e)) computed in log space so a large alpha cannot overflow
    log_sum = np.logaddexp.reduce(np.concatenate(([0.0], exponents)))
    return float(np.exp(-log_sum))


def alpha_rank(win_rate, *, alpha: float = DEFAULT_ALPHA, population_size: int = DEFAULT_POPULATION) -> np.ndarray:
    """Stationary distribution of single-population alpha-rank over the decks."""
    m = _as_matrix(win_rate)
    n = m.shape[0]
    if n == 1:
        return np.ones(1)
    c = np.zeros((n, n))
    for s in range(n):
        for r in range(n):
            if r != s:
                c[s, r] = fixation_probability(m, r, s, alpha=alpha, population_size=population_size) / (n - 1)
        c[s, s] = 1.0 - c[s].sum()
    # pi C = pi, sum(pi) = 1
    a = np.vstack([c.T - np.eye(n), np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


# ------------------------------------------------------------------ matrix from games


def _deck_hash(deck: Deck) -> str:
    text = "|".join(",".join(map(str, part)) for part in (deck.main, deck.extra, deck.side))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _name_key(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def _named(decks: Sequence[Deck] | Mapping[str, Deck]) -> list[tuple[str, Deck]]:
    items = list(decks.items()) if isinstance(decks, Mapping) else [(d.name, d) for d in decks]
    seen: set[str] = set()
    for name, _ in items:
        if not name:
            raise ValueError("every deck needs a name (Deck.name or a mapping key)")
        if name in seen:
            raise ValueError(f"duplicate deck name {name!r}")
        seen.add(name)
    return items


def _tuples(a) -> tuple:
    return tuple(tuple(row) for row in a)


@dataclass(frozen=True)
class MatchupMatrix:
    """``win_rate[i][j]``: deck i's win rate against deck j (draws half), from ``games[i][j]`` games."""

    decks: tuple[str, ...]
    win_rate: tuple[tuple[float, ...], ...]
    games: tuple[tuple[int, ...], ...]
    ci_low: tuple[tuple[float, ...], ...]
    ci_high: tuple[tuple[float, ...], ...]
    agent: str
    seed: int
    pairs: int
    confidence: float = 0.95
    errors: int = 0  # games that raised (see ArenaReport.errors)
    deck_hashes: tuple[str, ...] = ()
    environment: dict[str, str] | None = None

    def array(self) -> np.ndarray:
        return np.array(self.win_rate, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return {k: (list(map(list, v)) if k in ("win_rate", "games", "ci_low", "ci_high") else
                    list(v) if isinstance(v, tuple) else v) for k, v in d.items()}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MatchupMatrix:
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("win_rate", "games", "ci_low", "ci_high"):
            kw[k] = _tuples(kw[k])
        kw["decks"], kw["deck_hashes"] = tuple(kw["decks"]), tuple(kw.get("deck_hashes", ()))
        return cls(**kw)


def build_matrix(decks: Sequence[Deck] | Mapping[str, Deck], agent: AgentFactory, *, pairs: int, seed: int = 0,
                 env: Environment | None = None, config: DuelConfig | None = None, workers: int = 1,
                 confidence: float = 0.95) -> MatchupMatrix:  # fmt: skip
    """Play every pair of decks (``2 * pairs`` paired games each) with ``agent`` on both sides."""
    items = _named(decks)
    names = [n for n, _ in items]
    index = {n: i for i, n in enumerate(names)}
    cells = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (x, dx), (y, dy) = sorted((items[i], items[j]), key=lambda it: it[0])
            cells.append((x, y, (dx, dy, derive_seed(seed, _name_key(x), _name_key(y)))))
    arena = Arena(agent, agent, env=env, config=config, workers=workers, confidence=confidence)
    reports = arena.run_many([c[2] for c in cells], pairs)

    n = len(items)
    rate, games = np.full((n, n), 0.5), np.zeros((n, n), dtype=int)
    lo, hi = np.full((n, n), 0.5), np.full((n, n), 0.5)
    for (x, y, _), rep in zip(cells, reports):
        i, j = index[x], index[y]
        rate[i, j], rate[j, i] = rep.win_rate, 1.0 - rep.win_rate
        games[i, j] = games[j, i] = rep.games
        lo[i, j], hi[i, j] = rep.ci
        lo[j, i], hi[j, i] = 1.0 - rep.ci[1], 1.0 - rep.ci[0]
    return MatchupMatrix(
        decks=tuple(names), win_rate=_tuples(rate.tolist()), games=_tuples(games.tolist()),
        ci_low=_tuples(lo.tolist()), ci_high=_tuples(hi.tolist()), agent=agent_name(agent), seed=seed,
        pairs=pairs, confidence=confidence, errors=sum(r.errors for r in reports),
        deck_hashes=tuple(_deck_hash(d) for _, d in items), environment=env.stamp() if env is not None else None,
    )  # fmt: skip


# ------------------------------------------------------------------ meta game result


@dataclass(frozen=True)
class MetaGame:
    """A matchup matrix with its Nash mixture and alpha-rank distribution (both indexed like ``matrix.decks``)."""

    matrix: MatchupMatrix
    nash: tuple[float, ...]
    alpha_rank: tuple[float, ...]
    alpha: float = DEFAULT_ALPHA
    population_size: int = DEFAULT_POPULATION

    def nash_by_deck(self) -> dict[str, float]:
        return dict(zip(self.matrix.decks, self.nash))

    def alpha_rank_by_deck(self) -> dict[str, float]:
        return dict(zip(self.matrix.decks, self.alpha_rank))

    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT, "format_version": FORMAT_VERSION, **self.matrix.to_dict(),
                "nash": list(self.nash), "alpha_rank": list(self.alpha_rank), "alpha": self.alpha,
                "population_size": self.population_size}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MetaGame:
        if d.get("format") != FORMAT or d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} file: {d.get('format')!r} v{d.get('format_version')!r}")
        return cls(matrix=MatchupMatrix.from_dict(d), nash=tuple(d["nash"]), alpha_rank=tuple(d["alpha_rank"]),
                   alpha=d["alpha"], population_size=d["population_size"])  # fmt: skip

    def save(self, path: str | Path | None = None, *, env: Environment | None = None, name: str = "matrix") -> Path:
        """Write JSON. With ``env``: to ``artifacts/matrix/<name>.json`` of that environment,
        which must be the one the matrix was built under; otherwise to ``path``."""
        if env is not None:
            if path is not None:
                raise ValueError("give either a path or an environment, not both")
            env.check_stamp(self.matrix.environment or {})
            path = env.artifact_path(ARTIFACT_DIR, f"{name}.json")
        elif path is None:
            raise ValueError("a path is required when no environment is given")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, env: Environment | None = None) -> MetaGame:
        """Read a saved result; with ``env``, refuse one produced under another environment."""
        meta = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if env is not None:
            env.check_stamp(meta.matrix.environment or {})
        return meta


def analyze(
    matrix: MatchupMatrix, *, alpha: float = DEFAULT_ALPHA, population_size: int = DEFAULT_POPULATION
) -> MetaGame:
    """Solve the meta game of ``matrix``: Nash mixture and alpha-rank."""
    m = matrix.array()
    return MetaGame(matrix=matrix, nash=tuple(float(v) for v in nash_mixture(m)),
                    alpha_rank=tuple(float(v) for v in alpha_rank(m, alpha=alpha, population_size=population_size)),
                    alpha=alpha, population_size=population_size)  # fmt: skip


__all__ = ["MatchupMatrix", "MetaGame", "alpha_rank", "analyze", "build_matrix", "fixation_probability",
           "nash_mixture"]  # fmt: skip
