"""Self-play league (T4b.4, design I1): current-policy self-play + a small snapshot pool + keep-best, and
deck-pool sampling with paired first / second games.

- :class:`SnapshotPool`: frozen copies of the learner taken every few updates; the oldest is evicted beyond
  ``capacity``, except the pinned **best** snapshot (keep-best: the checkpoint that scored highest in the
  periodic evaluation), which stays until a better one replaces it.
- :class:`DeckPool`: the decks and the (deck_a, deck_b) pairings to sample (all ordered pairs, distinct decks
  only, or mirrors; uniform for now -- meta shares come with T4d.1).
- :class:`SelfPlaySchedule`: ``next_game() -> Assignment`` for the rollout collector. Games are dealt in
  pairs: one seed, one deck pairing, one opponent, played once with deck a first and once with deck b first
  (先后攻配平). A deal is self-play with probability ``selfplay_fraction``, else against a pool snapshot with
  the learner on a random side. The snapshot is resolved when the pair is dealt and held until the next deal
  (``opponent(sid)``), so an eviction between the pair's two games cannot lose it.
"""

from __future__ import annotations

import copy
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from torch import nn

from ygorl.cards.ydk import Deck
from ygorl.engine.duel import DuelConfig
from ygorl.env.pool import GameSpec
from ygorl.eval.arena import derive_seed
from ygorl.train.rollout import Assignment

PAIRINGS = ("all", "cross", "mirror")


@dataclass
class Snapshot:
    id: int
    update: int
    model: nn.Module
    tag: str = "snapshot"
    score: float | None = None


class SnapshotPool:
    """Frozen past policies. Regular snapshots are evicted oldest first beyond ``capacity``; the keep-best one and
    *pinned* opponents (fixed policies from the configuration, e.g. a BC checkpoint) stay. Pinned opponents get
    ``pinned_share`` of the draws together (the rest uniform over the others); they are not part of
    :meth:`state_dict` — the configuration pins them again, with the same negative ids, when a run resumes."""

    def __init__(self, capacity: int = 8, pinned_share: float = 0.5) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if not 0 <= pinned_share <= 1:
            raise ValueError("pinned_share must be in [0, 1]")
        self.capacity = capacity
        self.pinned_share = pinned_share
        self._snaps: OrderedDict[int, Snapshot] = OrderedDict()
        self._pinned: OrderedDict[int, Snapshot] = OrderedDict()
        self._next_id = 0
        self.best_id: int | None = None

    @staticmethod
    def _freeze(model: nn.Module) -> nn.Module:
        m = copy.deepcopy(model)
        m.eval()
        m.requires_grad_(False)
        return m

    def _put(self, model: nn.Module, update: int, tag: str, score: float | None, frozen: bool = False) -> int:
        sid = self._next_id
        self._next_id += 1
        self._snaps[sid] = Snapshot(sid, update, model if frozen else self._freeze(model), tag, score)
        return sid

    def add(self, model: nn.Module, update: int, tag: str = "snapshot") -> int:
        """Freeze a copy of ``model``; evict the oldest non-best snapshots beyond ``capacity``."""
        sid = self._put(model, update, tag, None)
        regular = [i for i in self._snaps if i != self.best_id]
        for old in regular[: max(0, len(regular) - self.capacity)]:
            del self._snaps[old]
        return sid

    def set_best(self, model: nn.Module, update: int, score: float) -> int:
        """Pin a copy of ``model`` as the best snapshot, replacing the previous best."""
        if self.best_id is not None:
            self._snaps.pop(self.best_id, None)
        self.best_id = self._put(model, update, "best", score)
        return self.best_id

    @property
    def best_score(self) -> float | None:
        return None if self.best_id is None else self._snaps[self.best_id].score

    def pin(self, model: nn.Module, tag: str = "pinned") -> int:
        """Keep a frozen copy of ``model`` for the whole run (ids -1, -2, ... in pinning order)."""
        sid = -(len(self._pinned) + 1)
        self._pinned[sid] = Snapshot(sid, 0, self._freeze(model), tag)
        return sid

    def sample(self, rng: np.random.Generator) -> int | None:
        """A snapshot id (pinned ones take ``pinned_share`` of the draws), or None when the pool is empty."""
        pinned, regular = list(self._pinned), list(self._snaps)
        if pinned and (not regular or rng.random() < self.pinned_share):
            return pinned[int(rng.integers(len(pinned)))]
        return None if not regular else regular[int(rng.integers(len(regular)))]

    def get(self, sid: int) -> nn.Module:
        return (self._pinned if sid < 0 else self._snaps)[sid].model

    def ids(self) -> list[int]:
        return [*self._pinned, *self._snaps]

    def info(self) -> list[dict]:
        return [{"id": s.id, "update": s.update, "tag": s.tag, "score": s.score}
                for s in (*self._pinned.values(), *self._snaps.values())]  # fmt: skip

    def __len__(self) -> int:
        return len(self._pinned) + len(self._snaps)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "next_id": self._next_id, "best_id": self.best_id,
                "snapshots": [{"id": s.id, "update": s.update, "tag": s.tag, "score": s.score,
                               "state": s.model.state_dict()} for s in self._snaps.values()]}  # fmt: skip

    def load_state_dict(self, state: dict, factory: Callable[[], nn.Module]) -> None:
        """Rebuild the snapshots with ``factory()`` (a fresh model of the same architecture)."""
        self.capacity = int(state["capacity"])
        self._snaps.clear()
        for s in state["snapshots"]:
            model = factory()
            model.load_state_dict(s["state"])
            self._snaps[int(s["id"])] = Snapshot(int(s["id"]), int(s["update"]), self._freeze(model), s["tag"], s["score"])
        self._next_id = int(state["next_id"])
        self.best_id = state["best_id"]


class DeckPool:
    """``pairings``: ``"all"`` (every ordered pair, mirrors included), ``"cross"`` (distinct decks),
    ``"mirror"``, or an explicit list of ``(i, j)`` deck-index pairs."""

    def __init__(self, decks: Sequence[Deck], pairings: str | Sequence[tuple[int, int]] = "cross") -> None:
        if not decks:
            raise ValueError("the deck pool is empty")
        self.decks = list(decks)
        n = len(self.decks)
        if isinstance(pairings, str):
            if pairings not in PAIRINGS:
                raise ValueError(f"pairings must be one of {PAIRINGS} or a list of index pairs")
            pairs = [(i, j) for i in range(n) for j in range(n)
                     if pairings == "all" or (pairings == "cross") == (i != j)]  # fmt: skip
        else:
            pairs = [(int(i), int(j)) for i, j in pairings]
        if not pairs:
            raise ValueError(f"no deck pairings ({pairings!r} with {n} deck{'s' if n != 1 else ''})")
        if any(not (0 <= i < n and 0 <= j < n) for i, j in pairs):
            raise ValueError("a deck pairing refers to a deck outside the pool")
        self.pairs = pairs

    def sample(self, rng: np.random.Generator) -> tuple[Deck, Deck]:
        i, j = self.pairs[int(rng.integers(len(self.pairs)))]
        return self.decks[i], self.decks[j]


class SelfPlaySchedule:
    def __init__(self, decks: DeckPool, pool: SnapshotPool, config: DuelConfig | None = None, *,
                 selfplay_fraction: float = 0.75, seed: int = 0) -> None:  # fmt: skip
        if not 0 <= selfplay_fraction <= 1:
            raise ValueError("selfplay_fraction must be in [0, 1]")
        self.decks = decks
        self.pool = pool
        self.config = config or DuelConfig()
        self.selfplay_fraction = selfplay_fraction
        self.seed = seed
        self.rng = np.random.default_rng(derive_seed(seed, 1))
        self.deals = 0
        self._queue: deque[Assignment] = deque()
        self._held: dict[int, nn.Module] = {}  # snapshots of the current deal, resolved when it was dealt

    def __call__(self) -> Assignment:
        if not self._queue:
            self._deal()
        return self._queue.popleft()

    def opponent(self, sid: int) -> nn.Module:
        """The snapshot module of a dealt pool game (the collector's ``opponents``), even if since evicted."""
        held = self._held.get(sid)
        return held if held is not None else self.pool.get(sid)

    def _deal(self) -> None:
        deck_a, deck_b = self.decks.sample(self.rng)
        seed = derive_seed(self.seed, 2, self.deals)
        self.deals += 1
        # Only called with an empty queue, and the collector resolves each game's opponent as it starts it,
        # so nothing still needs the previous deal's snapshot.
        self._held.clear()
        opponent = None
        if self.rng.random() >= self.selfplay_fraction:
            opponent = self.pool.sample(self.rng)
            if opponent is not None:
                self._held[opponent] = self.pool.get(opponent)
        side = int(self.rng.integers(2))  # the learner's deck in a pool game: 0 = deck_a, 1 = deck_b
        for first in (0, 1):
            spec = GameSpec(seed=seed, deck_a=deck_a, deck_b=deck_b, first=first, config=self.config)
            info = {"deck_a": deck_a.name, "deck_b": deck_b.name, "first": first}
            if opponent is not None:
                info["learner_deck"] = (deck_a, deck_b)[side].name
            # engine seat p holds deck (first + p) % 2, so deck `side` sits at seat (side + first) % 2
            self._queue.append(Assignment(spec, opponent, (side + first) % 2, info))

    def state_dict(self) -> dict:
        return {"deals": self.deals, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.deals = int(state["deals"])
        self.rng.bit_generator.state = state["rng"]
        self._queue.clear()
        self._held.clear()


__all__ = ["DeckPool", "PAIRINGS", "SelfPlaySchedule", "Snapshot", "SnapshotPool"]
