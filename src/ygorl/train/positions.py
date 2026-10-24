"""Position library: game starts recorded as (start, action prefix), replayed by prefix start (docs/engine.md 「前缀起局」).

A :class:`Position` is a decision point of some game, stored as the game's exact start (environment version, seed,
who moves first, both decks in the order the host loads them, rule flags / player rules / limits, optional explicit
core seed words) plus the action indices that lead from the start to it, as ``EncodedVecEnv.step`` received them
under the recorded ``skip_forced`` and ``cancel_budget`` (both change which decisions come back and which rows are
choosable, so the replaying env must use the same values). ``EncodedVecEnv.reset(env_id, spec, prefix)`` plays the
prefix on the worker thread, so starting from a position costs one reset plus about 0.16 ms per prefix action, and
works in any env (unlike a snapshot, which restores only into its own game). Uses: re-dealing the turn-2 positions of
the first-seat plan (docs/spikes/first-seat-interruption-plan-2026-10-24.md §5.2, §5.3) as training starts.

Optional fields: ``learner_seat`` (the engine seat the learner plays from the position), ``board_key`` (the field-only
board key of :func:`board_key`, the coarse cache key of §10.3), ``origin`` (free-form provenance: source tool, line
id, model update, ...).

Storage is JSONL, one position per line (:func:`write_positions` / :func:`read_positions`);
:class:`PositionLibrary` holds a list of them and deals one with :meth:`PositionLibrary.sample`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment, PlayerRules
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelConfig
from ygorl.env.pool import GameSpec

FORMAT = "ygorl-position"
FORMAT_VERSION = 1
CANCEL_BUDGETS = (0, 1, 4, 32)


# ------------------------------------------------------------------ board keys


def board_keys(board: dict, player: int = 0) -> dict[str, tuple]:
    """Board keys of ``player``'s side of ``board`` (:func:`ygorl.solver.targets.board_summary`), first-seat plan §10.3:

    - ``field``: the sorted multiset of the player's monster and spell / trap zone cards as (zone, password, set);
    - ``field_grave``: ``field`` plus the sorted graveyard and banished passwords (the finer key P0c measured).

    P0c (``tools/turn1_board_modes.py``) found the field-only key concentrated enough to cache by; :func:`board_key`
    is that key in storable form.
    """
    side = board["players"][player]
    cards = sorted((z, int(c["code"]), not c["position"] & C.POS_FACEUP) for z in ("mzone", "szone") for c in side[z])
    piles = sorted(("grave", int(x)) for x in side["grave"]) + sorted(("banished", int(x)) for x in side["banished"])
    return {"field": tuple(cards), "field_grave": (tuple(cards), tuple(piles))}


def board_key(board: dict, player: int = 0) -> str:
    """The field-only board key of :func:`board_keys` as a compact JSON string (equal boards, equal strings)."""
    return json.dumps(board_keys(board, player)["field"], separators=(",", ":"))


# ------------------------------------------------------------------ positions


def _deck_to_dict(deck: Deck) -> dict:
    return {"name": deck.name, "main": list(deck.main), "extra": list(deck.extra), "side": list(deck.side)}


def _deck_from_dict(d: dict) -> Deck:
    return Deck(tuple(int(c) for c in d["main"]), tuple(int(c) for c in d["extra"]),
                tuple(int(c) for c in d.get("side", ())), str(d.get("name", "")))  # fmt: skip


def _config_from_dict(d: dict) -> DuelConfig:
    return DuelConfig(**{**d, "player": PlayerRules(**d["player"])})


@dataclass(frozen=True)
class Position:
    """A decision point as (exact game start, action prefix); see the module docstring."""

    environment: str | None  # environment version the prefix was recorded in (None: no environment)
    seed: int
    first: int  # GameSpec.first: 0 = deck a moves first
    deck_a: Deck
    deck_b: Deck
    prefix: tuple[int, ...]  # EncodedVecEnv.step action indices from the start
    skip_forced: bool
    config: DuelConfig = field(default_factory=DuelConfig)
    core_seed: tuple[int, ...] | None = None  # explicit core seed words (GameSpec.core_seed)
    cancel_budget: int = 32
    learner_seat: int | None = None
    board_key: str | None = None
    origin: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.first not in (0, 1):
            raise ValueError(f"first must be 0 or 1, not {self.first!r}")
        if self.learner_seat not in (None, 0, 1):
            raise ValueError(f"learner_seat must be None, 0 or 1, not {self.learner_seat!r}")
        if self.cancel_budget not in CANCEL_BUDGETS:
            raise ValueError(f"cancel_budget must be one of {CANCEL_BUDGETS}, not {self.cancel_budget!r}")
        if any(type(a) is not int or not 0 <= a < 1 << 32 for a in self.prefix):
            raise ValueError("prefix actions must be non-negative 32-bit ints")
        if self.config.curriculum != "full" or self.config.augmented_start:
            raise ValueError("positions start full games (the C++ env runs no curriculum mode)")

    @classmethod
    def from_spec(cls, spec: GameSpec, prefix: Sequence[int], *, skip_forced: bool, cancel_budget: int = 32,
                  env: Environment | None = None, learner_seat: int | None = None, board_key: str | None = None,
                  origin: dict | None = None) -> Position:  # fmt: skip
        """The position ``prefix`` reaches from ``spec`` (actions as an env with ``skip_forced`` / ``cancel_budget``
        received them; e.g. ``EncodedVecEnv(record_actions=True)``'s trace)."""
        return cls(environment=env.version if env is not None else None, seed=int(spec.seed), first=int(spec.first),
                   deck_a=spec.deck_a, deck_b=spec.deck_b, prefix=tuple(int(a) for a in prefix),
                   skip_forced=bool(skip_forced), config=spec.config,
                   core_seed=None if spec.core_seed is None else tuple(int(w) for w in spec.core_seed),
                   cancel_budget=cancel_budget, learner_seat=learner_seat, board_key=board_key,
                   origin=dict(origin or {}))  # fmt: skip

    def spec(self) -> GameSpec:
        return GameSpec(seed=self.seed, deck_a=self.deck_a, deck_b=self.deck_b, first=self.first, config=self.config,
                        core_seed=self.core_seed)  # fmt: skip

    def check(self, venv, env: Environment | None = None) -> None:
        """ValueError unless ``venv`` (an EncodedVecEnv) and ``env`` replay this position's prefix as recorded."""
        problems = []
        if self.environment is not None and (env is None or env.version != self.environment):
            problems.append(f"recorded in environment {self.environment}, not {env.version if env else None}")
        if venv.skip_forced != self.skip_forced:
            problems.append(f"recorded with skip_forced={self.skip_forced}, the env has {venv.skip_forced}")
        if venv.cancel_budget != self.cancel_budget:
            problems.append(f"recorded with cancel_budget={self.cancel_budget}, the env has {venv.cancel_budget}")
        if problems:
            raise ValueError("position cannot start here: " + "; ".join(problems))

    def start(self, venv, env_id: int, env: Environment | None = None) -> None:
        """Queue the prefix start of this position in ``venv``'s env ``env_id`` (after :meth:`check`)."""
        self.check(venv, env)
        venv.reset(env_id, self.spec(), self.prefix)

    def with_prefix(self, prefix: Sequence[int], **changes) -> Position:
        """The same game start with another prefix (e.g. one more action)."""
        return replace(self, prefix=tuple(int(a) for a in prefix), **changes)

    # -- storage ----------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        d = {"format": FORMAT, "format_version": FORMAT_VERSION, "environment": self.environment, "seed": self.seed,
             "first": self.first, "deck_a": _deck_to_dict(self.deck_a), "deck_b": _deck_to_dict(self.deck_b),
             "prefix": list(self.prefix), "skip_forced": self.skip_forced, "cancel_budget": self.cancel_budget,
             "config": asdict(self.config)}  # fmt: skip
        if self.core_seed is not None:
            d["core_seed"] = list(self.core_seed)
        for k in ("learner_seat", "board_key"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.origin:
            d["origin"] = self.origin
        return d

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Position:
        """Position from its JSON form; ValueError for a malformed record."""
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(
                f"not a ygorl position (format={data.get('format') if isinstance(data, dict) else None!r})"
            )
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported position format_version {data.get('format_version')} "
                             f"(expected {FORMAT_VERSION})")  # fmt: skip
        try:
            return cls(environment=data["environment"], seed=int(data["seed"]), first=int(data["first"]),
                       deck_a=_deck_from_dict(data["deck_a"]), deck_b=_deck_from_dict(data["deck_b"]),
                       prefix=tuple(data["prefix"]), skip_forced=bool(data["skip_forced"]),
                       config=_config_from_dict(data["config"]),
                       core_seed=tuple(int(w) for w in data["core_seed"]) if "core_seed" in data else None,
                       cancel_budget=int(data["cancel_budget"]), learner_seat=data.get("learner_seat"),
                       board_key=data.get("board_key"), origin=dict(data.get("origin", {})))  # fmt: skip
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed position record: {e!r}") from e


def write_positions(path: str | Path, positions: Iterable[Position], append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for p in positions:
            f.write(json.dumps(p.to_json(), separators=(",", ":")) + "\n")


def read_positions(path: str | Path) -> list[Position]:
    """Every position of a JSONL file (blank lines skipped); ValueError naming the line of a malformed record."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(Position.from_json(json.loads(line)))
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    return out


class PositionLibrary:
    """Positions to start games from, dealt by :meth:`sample`."""

    def __init__(self, positions: Iterable[Position] = ()) -> None:
        self.positions = list(positions)

    @classmethod
    def load(cls, paths: str | Path | Sequence[str | Path], env: Environment | None = None) -> PositionLibrary:
        """Positions of one or more JSONL files; with ``env``, a position recorded in another environment is a
        ValueError (positions are bound to the environment their prefix was recorded in)."""
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        positions = [p for path in paths for p in read_positions(path)]
        if env is not None:
            for i, p in enumerate(positions):
                if p.environment is not None and p.environment != env.version:
                    raise ValueError(f"position {i} was recorded in environment {p.environment}, not {env.version}")
        return cls(positions)

    def save(self, path: str | Path) -> None:
        write_positions(path, self.positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> Position:
        return self.positions[i]

    def sample(self, rng: np.random.Generator, learner_seat: int | None = None) -> Position:
        """A position drawn uniformly (from ``rng`` only: the same generator state deals the same position); with
        ``learner_seat``, only among the positions whose learner plays that seat."""
        pool = (self.positions if learner_seat is None
                else [p for p in self.positions if p.learner_seat == learner_seat])  # fmt: skip
        if not pool:
            raise ValueError("the position library is empty" if not self.positions
                             else f"no position with learner_seat={learner_seat}")  # fmt: skip
        return pool[int(rng.integers(len(pool)))]


__all__ = ["FORMAT", "FORMAT_VERSION", "Position", "PositionLibrary", "board_key", "board_keys", "read_positions",
           "write_positions"]  # fmt: skip
