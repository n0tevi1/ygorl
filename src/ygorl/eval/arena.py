"""Paired-seed arena: play two agents against each other and report the win rate.

Pairing: pair ``p`` uses one duel seed ``s_p = derive_seed(seed, p)`` for two
games with the same decks in the same order, once with deck a going first
(``first=0``) and once with deck b going first (``first=1``). The arena
shuffles both main decks itself from ``s_p`` (deck a with stream 0, deck b with
stream 1) and loads them unshuffled, so both games of a pair start from the
same opening hands; agents get the same seeds in both games too. This cancels
most of the luck of the draw and the first-player advantage.

Every game is fully determined by its :class:`GameSpec`, so a run gives the
same records for any number of worker processes.

Win rate is from agent a's side and counts a draw as half a win. The interval
is Wilson's score interval on that rate; with draws it is conservative (the
variance of a win/draw/loss score is at most ``p(1 - p)``). It treats games as
independent, which the pairing does not guarantee.
"""

from __future__ import annotations

import multiprocessing as mp
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ygorl.agents.base import AgentFactory, agent_name
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine.duel import Duel, DuelConfig, shuffle_deck

MASK64 = (1 << 64) - 1


# ------------------------------------------------------------------ seeds and statistics


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def derive_seed(seed: int, *keys: int) -> int:
    """Stable 63-bit child seed of ``seed`` for the path ``keys`` (order matters)."""
    state = _splitmix64(seed & MASK64)
    for k in keys:
        state = _splitmix64(state ^ (k & MASK64))
    return state >> 1


def wilson_interval(successes: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion ``successes / n``."""
    if n <= 0:
        return 0.0, 1.0
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    lo = 0.0 if successes <= 0 else max(0.0, centre - half)
    hi = 1.0 if successes >= n else min(1.0, centre + half)
    return lo, hi


# ------------------------------------------------------------------ records and reports


@dataclass(frozen=True)
class GameRecord:
    """Outcome of one arena game; ``winner`` is 0 (agent a), 1 (agent b) or None."""

    pair: int
    seed: int
    first: int  # 0: deck/agent a went first
    winner: int | None
    reason: str  # DuelResult.reason, or "exception"
    turns: int = 0
    decisions: int = 0
    win_reason: int | None = None
    lp: tuple[int, int] = (0, 0)
    retries: int = 0
    unknown_messages: int = 0
    undecodable_messages: int = 0
    script_errors: int = 0
    error: str = ""


@dataclass(frozen=True)
class SideStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "win_rate": self.win_rate}


def _side(records: Iterable[GameRecord]) -> SideStats:
    rs = list(records)
    return SideStats(len(rs), sum(r.winner == 0 for r in rs), sum(r.winner == 1 for r in rs), sum(r.winner is None for r in rs))


@dataclass(frozen=True)
class ArenaReport:
    """Agent a vs agent b over ``games`` paired games, from agent a's side."""

    agent_a: str
    agent_b: str
    deck_a: str
    deck_b: str
    seed: int
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float  # (wins + draws / 2) / games
    ci: tuple[float, float]
    confidence: float
    as_first: SideStats  # games where agent a went first
    as_second: SideStats
    first_player_win_rate: float  # how often the player going first won (draws half)
    reasons: dict[str, int]
    retries: int
    unknown_messages: int
    errors: int  # games that raised
    mean_turns: float
    environment: dict[str, str] | None = None
    records: tuple[GameRecord, ...] = field(default=(), repr=False)

    def significant(self) -> bool:
        """The interval excludes 50%: a is significantly better or worse than b."""
        return self.ci[0] > 0.5 or self.ci[1] < 0.5

    def to_dict(self, records: bool = True) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "records"}
        d["ci"] = list(self.ci)
        d["as_first"], d["as_second"] = self.as_first.to_dict(), self.as_second.to_dict()
        if records:
            d["records"] = [asdict(r) for r in self.records]
        return d

    def summary(self) -> str:
        lo, hi = self.ci
        return (f"{self.agent_a}[{self.deck_a}] vs {self.agent_b}[{self.deck_b}]: {self.games} games, "
                f"W/L/D {self.wins}/{self.losses}/{self.draws}, win rate {self.win_rate:.3f} "
                f"({self.confidence:.0%} CI {lo:.3f}-{hi:.3f}); first {self.as_first.win_rate:.3f}, "
                f"second {self.as_second.win_rate:.3f}; first player wins {self.first_player_win_rate:.3f}; "
                f"errors {self.errors}, retries {self.retries}")  # fmt: skip


def summarize(records: Sequence[GameRecord], *, agent_a: str, agent_b: str, deck_a: str, deck_b: str, seed: int,
              confidence: float = 0.95, environment: dict[str, str] | None = None) -> ArenaReport:  # fmt: skip
    """Aggregate game records (agent a's side) into a report."""
    total = _side(records)
    first_player = sum(1.0 if r.winner == r.first else 0.5 if r.winner is None else 0.0 for r in records)
    n = total.games
    return ArenaReport(
        agent_a=agent_a, agent_b=agent_b, deck_a=deck_a, deck_b=deck_b, seed=seed, games=n,
        wins=total.wins, losses=total.losses, draws=total.draws, win_rate=total.win_rate,
        ci=wilson_interval(total.wins + 0.5 * total.draws, n, confidence), confidence=confidence,
        as_first=_side(r for r in records if r.first == 0), as_second=_side(r for r in records if r.first == 1),
        first_player_win_rate=first_player / n if n else 0.0,
        reasons=dict(sorted(Counter(r.reason for r in records).items())),
        retries=sum(r.retries for r in records), unknown_messages=sum(r.unknown_messages for r in records),
        errors=sum(r.reason == "exception" for r in records),
        mean_turns=sum(r.turns for r in records) / n if n else 0.0,
        environment=environment, records=tuple(records),
    )  # fmt: skip


def merge(reports: Sequence[ArenaReport], *, deck_a: str = "*", deck_b: str = "*") -> ArenaReport:
    """Pool several reports of the same two agents (e.g. one per deck pairing) into one."""
    if not reports:
        raise ValueError("nothing to merge")
    first = reports[0]
    records = [r for rep in reports for r in rep.records]
    return summarize(records, agent_a=first.agent_a, agent_b=first.agent_b, deck_a=deck_a, deck_b=deck_b,
                     seed=first.seed, confidence=first.confidence, environment=first.environment)  # fmt: skip


# ------------------------------------------------------------------ games


@dataclass(frozen=True)
class GameSpec:
    """Everything needed to replay one arena game exactly."""

    pair: int
    first: int
    seed: int
    agent_seeds: tuple[int, int]
    deck_a: Deck  # already in load order when config.shuffle_decks is False
    deck_b: Deck
    agent_a: AgentFactory
    agent_b: AgentFactory
    env: Environment | None
    config: DuelConfig

    def duel(self) -> Duel:
        return Duel(self.seed, self.env, self.deck_a, self.deck_b, config=self.config, first=self.first)

    def play(self) -> GameRecord:
        base = dict(pair=self.pair, seed=self.seed, first=self.first)
        try:
            r = self.duel().run(self.agent_a(self.agent_seeds[0]), self.agent_b(self.agent_seeds[1]))
        except Exception as exc:  # noqa: BLE001 - recorded in the report, counted as an error
            return GameRecord(**base, winner=None, reason="exception", error=f"{type(exc).__name__}: {exc}")
        return GameRecord(**base, winner=r.winner, reason=r.reason, turns=r.turns, decisions=r.decisions,
                          win_reason=r.win_reason, lp=r.lp, retries=r.retries, unknown_messages=r.unknown_messages,
                          undecodable_messages=r.undecodable_messages, script_errors=len(r.script_errors),
                          error=r.error)  # fmt: skip


def _play(spec: GameSpec) -> GameRecord:
    return spec.play()


class Arena:
    """Plays ``agent_a`` against ``agent_b`` (factories ``seed -> agent``) on paired seeds."""

    def __init__(self, agent_a: AgentFactory, agent_b: AgentFactory, *, env: Environment | None = None,
                 config: DuelConfig | None = None, workers: int = 1, confidence: float = 0.95,
                 mp_context: str | None = None) -> None:  # fmt: skip
        self.agent_a, self.agent_b = agent_a, agent_b
        self.env = env
        self.config = config or (DuelConfig.from_environment(env) if env is not None else DuelConfig())
        self.workers = max(1, workers)
        self.confidence = confidence
        self.mp_context = mp_context

    def game_specs(self, deck_a: Deck, deck_b: Deck, pairs: int, seed: int = 0) -> list[GameSpec]:
        shuffle = self.config.shuffle_decks
        config = replace(self.config, shuffle_decks=False)
        specs = []
        for p in range(pairs):
            s = derive_seed(seed, p)
            a, b = deck_a, deck_b
            if shuffle:
                a = replace(deck_a, main=tuple(shuffle_deck(deck_a.main, s, 0)))
                b = replace(deck_b, main=tuple(shuffle_deck(deck_b.main, s, 1)))
            agent_seeds = (derive_seed(s, 0), derive_seed(s, 1))
            for first in (0, 1):
                specs.append(GameSpec(p, first, s, agent_seeds, a, b, self.agent_a, self.agent_b, self.env, config))
        return specs

    def play(self, specs: Sequence[GameSpec]) -> list[GameRecord]:
        """Play games in order-preserving parallel; the result does not depend on ``workers``."""
        if self.workers == 1 or len(specs) <= 1:
            return [spec.play() for spec in specs]
        workers = min(self.workers, len(specs))
        chunk = max(1, len(specs) // (workers * 8))
        # Forking a process that has initialized PyTorch (e.g. a policy:<checkpoint> agent validated in the
        # parent) can deadlock the children's thread pools: start fresh interpreters instead.
        context = self.mp_context or ("spawn" if "torch" in sys.modules else None)
        with mp.get_context(context).Pool(workers) as pool:
            return pool.map(_play, specs, chunksize=chunk)

    def run(self, deck_a: Deck, deck_b: Deck, pairs: int, seed: int = 0) -> ArenaReport:
        """``2 * pairs`` games of deck a (agent a) against deck b (agent b)."""
        return self.run_many([(deck_a, deck_b)], pairs, seed)[0]

    def run_many(self, matchups: Sequence[tuple[Deck, Deck] | tuple[Deck, Deck, int]], pairs: int,
                 seed: int = 0) -> list[ArenaReport]:  # fmt: skip
        """One report per ``(deck_a, deck_b[, seed])``, all games sharing one worker pool.

        A matchup without its own seed uses ``seed``.
        """
        cells = [(deck_a, deck_b, rest[0] if rest else seed) for deck_a, deck_b, *rest in matchups]
        records = self.play([spec for a, b, s in cells for spec in self.game_specs(a, b, pairs, s)])
        stamp = self.env.stamp() if self.env is not None else None
        per_cell = 2 * pairs
        return [
            summarize(records[i * per_cell : (i + 1) * per_cell], agent_a=agent_name(self.agent_a),
                      agent_b=agent_name(self.agent_b), deck_a=a.name, deck_b=b.name, seed=s,
                      confidence=self.confidence, environment=stamp)
            for i, (a, b, s) in enumerate(cells)
        ]  # fmt: skip
