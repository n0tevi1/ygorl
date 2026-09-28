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
- :class:`EvolvedDecks` (#108): the decks an evolution process adds to the pool while training runs, read from a
  deck-pool manifest (``ygorl-deck-pool`` JSON) that the trainer re-reads every few updates. With probability
  ``share`` a deal pairs one probation / active evolved deck (weight x ``(1 - p) ** power``, ``p`` the current
  policy's score piloting it) with a corpus or history deck; with no probation / active deck the schedule draws
  nothing extra, so training is bit-identical to the fixed pool.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from torch import nn

from ygorl.cards.ydk import Deck, parse_ydk
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
    games: int = 0  # pool games of the learner against this snapshot
    learner_points: float = 0.0  # the learner's points in them (1 / 0.5 / 0)

    @property
    def learner_win_rate(self) -> float:
        """The learner's score against this snapshot, with one prior game at 0.5 (0.5 before any game)."""
        return (self.learner_points + 0.5) / (self.games + 1)


class SnapshotPool:
    """Frozen past policies. Regular snapshots are evicted oldest first beyond ``capacity``; the keep-best one and
    *pinned* opponents (fixed policies from the configuration, e.g. a BC checkpoint) stay. Pinned opponents get
    ``pinned_share`` of the draws together (the rest uniform over the others); they are not part of
    :meth:`state_dict` — the configuration pins them again, with the same negative ids, when a run resumes.

    ``sampling``: ``"uniform"`` over the regular snapshots, or ``"pfsp"`` (prioritized fictitious self-play,
    AlphaStar's "hard" weighting): snapshot ``s`` is drawn with weight ``(1 - p_s) ** pfsp_power``, ``p_s`` the
    learner's score against it (:meth:`record`; 0.5 before any game), so the opponents it loses to come up more."""

    SAMPLING = ("uniform", "pfsp")

    def __init__(self, capacity: int = 8, pinned_share: float = 0.5, sampling: str = "uniform",
                 pfsp_power: float = 2.0) -> None:  # fmt: skip
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if not 0 <= pinned_share <= 1:
            raise ValueError("pinned_share must be in [0, 1]")
        if sampling not in self.SAMPLING:
            raise ValueError(f"sampling must be one of {self.SAMPLING}, got {sampling!r}")
        if pfsp_power < 0:
            raise ValueError("pfsp_power must be >= 0")
        self.capacity = capacity
        self.pinned_share = pinned_share
        self.sampling = sampling
        self.pfsp_power = pfsp_power
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
        if not regular:
            return None
        if self.sampling == "uniform":
            return regular[int(rng.integers(len(regular)))]
        w = np.array([(1.0 - self._snaps[i].learner_win_rate) ** self.pfsp_power for i in regular]) + 1e-6
        return regular[int(rng.choice(len(regular), p=w / w.sum()))]

    def record(self, sid: int, learner_score: float) -> None:
        """One finished pool game against snapshot ``sid`` (ignored if it has left the pool)."""
        snap = (self._pinned if sid < 0 else self._snaps).get(sid)
        if snap is not None:
            snap.games += 1
            snap.learner_points += learner_score

    def get(self, sid: int) -> nn.Module:
        return (self._pinned if sid < 0 else self._snaps)[sid].model

    def ids(self) -> list[int]:
        return [*self._pinned, *self._snaps]

    def info(self) -> list[dict]:
        return [{"id": s.id, "update": s.update, "tag": s.tag, "score": s.score, "games": s.games,
                 "learner_win_rate": round(s.learner_win_rate, 3)}
                for s in (*self._pinned.values(), *self._snaps.values())]  # fmt: skip

    def __len__(self) -> int:
        return len(self._pinned) + len(self._snaps)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "next_id": self._next_id, "best_id": self.best_id,
                "snapshots": [{"id": s.id, "update": s.update, "tag": s.tag, "score": s.score, "games": s.games,
                               "learner_points": s.learner_points, "state": s.model.state_dict()}
                              for s in self._snaps.values()]}  # fmt: skip

    def load_state_dict(self, state: dict, factory: Callable[[], nn.Module]) -> None:
        """Rebuild the snapshots with ``factory()`` (a fresh model of the same architecture)."""
        self.capacity = int(state["capacity"])
        self._snaps.clear()
        for s in state["snapshots"]:
            model = factory()
            model.load_state_dict(s["state"])
            self._snaps[int(s["id"])] = Snapshot(
                int(s["id"]),
                int(s["update"]),
                self._freeze(model),
                s["tag"],
                s["score"],
                int(s.get("games", 0)),
                float(s.get("learner_points", 0.0)),
            )
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


MANIFEST_FORMAT = "ygorl-deck-pool"
DECK_STATUSES = ("probation", "active", "history")


class EvolvedDecks:
    """The evolved part of the deck pool, from a manifest::

        {"format": "ygorl-deck-pool", "version": 1,
         "decks": [{"id": "evo-0001", "file": "decks/evo-0001.ydk", "status": "probation", "weight": 1.0,
                    "parent": "...", ...}]}

    ``file`` is relative to the manifest; ``status`` is ``probation`` / ``active`` (dealt as the evolved side of a
    pairing) or ``history`` (a superseded elite: only an opponent deck); ``weight`` (>= 0, default 1) scales the
    sampling weight; other fields (lineage, descriptors) are the evolution process's and ignored here. Deck ids name
    the decks in the logs. The manifest is written by another process while training runs, so nothing in it stops
    training: an unreadable manifest keeps the previous pool, and an entry that is illegal (``validate(deck) ->
    problems``, e.g. the environment's legality check), unreadable, duplicated or malformed is left out; both with a
    note in ``problems``. The learner's scores piloting each deck (``record``) are discounted by ``decay`` per game
    (about the last 1 / (1 - decay) games count), so they follow the current policy; they survive reloads and, with
    the decks' card lists, checkpoints (a resume needs neither the manifest nor the deck files)."""

    def __init__(self, path: str | Path, *, share: float = 0.3, power: float = 1.0, decay: float = 0.99,
                 validate: Callable[[Deck], Sequence[str]] | None = None) -> None:  # fmt: skip
        if not 0 <= share <= 1:
            raise ValueError("the evolved share must be in [0, 1]")
        if power < 0:
            raise ValueError("the evolved power must be >= 0")
        if not 0 < decay <= 1:
            raise ValueError("the score decay must be in (0, 1]")
        self.path = Path(path)
        self.share = share
        self.power = power
        self.decay = decay
        self.validate = validate
        self.entries: list[dict] = []
        self.decks: dict[str, Deck] = {}
        self.stats: dict[
            str, list[float]
        ] = {}  # id -> [discounted games, discounted points] of the learner piloting it
        self.problems: list[str] = []
        self.signature = ""

    def _read(self) -> tuple[str, dict[str, str]]:
        """The manifest text and the text of every deck file it names (unreadable ones left out)."""
        text = self.path.read_text() if self.path.exists() else ""
        files = {}
        try:
            for e in json.loads(text).get("decks", []) if text.strip() else []:
                f = self.path.parent / str(e.get("file", ""))
                if f.is_file():
                    files[str(e.get("file"))] = f.read_text()
        except (ValueError, AttributeError, TypeError):
            pass  # _parse reports it
        return text, files

    def reload(self) -> bool:
        """Re-read the manifest and its deck files; returns whether the pool changed. A missing manifest is an empty
        pool; a manifest that cannot be read keeps the previous pool (with a note in ``problems``)."""
        text, files = self._read()
        signature = json.dumps([text, sorted(files.items())])
        if signature == self.signature:
            return False
        self.signature = signature
        try:
            entries, decks, problems = self._parse(text, files)
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            self.problems = [f"{self.path}: kept the previous pool: {exc}"]
            return True
        self.entries, self.decks, self.problems = entries, decks, problems
        return True

    def _parse(self, text: str, files: dict[str, str]) -> tuple[list[dict], dict[str, Deck], list[str]]:
        entries, decks, problems = [], {}, []
        if not text.strip():
            return entries, decks, problems
        data = json.loads(text)
        if data.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"not a {MANIFEST_FORMAT} manifest")
        for e in data.get("decks", []):
            did = str(e.get("id", ""))
            status, file = e.get("status", "probation"), str(e.get("file", ""))
            try:
                weight = float(e.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = -1.0
            bad = ([] if did else ["no id"]) + (["duplicate id"] if did in decks else [])
            bad += [] if status in DECK_STATUSES else [f"status must be one of {DECK_STATUSES}"]
            bad += [] if weight >= 0 else ["weight must be a number >= 0"]
            bad += [] if file in files else [f"cannot read {file}"]
            if not bad:
                deck = replace(parse_ydk(files[file]), name=did)
                bad = list(self.validate(deck)) if self.validate is not None else []
            if bad:
                problems.append(f"{did or '?'}: {'; '.join(bad)}")
                continue
            entries.append({**e, "id": did, "status": status, "weight": weight})
            decks[did] = deck
        return entries, decks, problems

    def ids(self, *statuses: str) -> list[str]:
        return [e["id"] for e in self.entries if e["status"] in statuses]

    def win_rate(self, did: str) -> float:
        """The learner's recent score piloting deck ``did``, with a prior of one game at 0.5."""
        games, points = self.stats.get(did, (0.0, 0.0))
        return (points + 0.5) / (games + 1)

    def weights(self) -> tuple[list[str], np.ndarray]:
        """The dealt (probation / active) decks and their sampling probabilities."""
        live = [e for e in self.entries if e["status"] != "history"]
        w = np.array([e["weight"] * (1.0 - self.win_rate(e["id"])) ** self.power for e in live]) + 1e-9
        return [e["id"] for e in live], w / w.sum()

    def record(self, did: str, score: float) -> None:
        games, points = self.stats.get(did, (0.0, 0.0))
        self.stats[did] = [games * self.decay + 1, points * self.decay + score]

    def state_dict(self) -> dict:
        return {"signature": self.signature, "entries": [dict(e) for e in self.entries],
                "decks": {k: {"main": list(d.main), "extra": list(d.extra), "side": list(d.side)}
                          for k, d in self.decks.items()},
                "stats": {k: list(v) for k, v in self.stats.items()}}  # fmt: skip

    def load_state_dict(self, state: dict) -> None:
        self.signature = state["signature"]
        self.entries = [dict(e) for e in state["entries"]]
        self.decks = {k: Deck(main=tuple(d["main"]), extra=tuple(d["extra"]), side=tuple(d["side"]), name=k)
                      for k, d in state["decks"].items()}  # fmt: skip
        self.stats = {k: [float(v[0]), float(v[1])] for k, v in state["stats"].items()}
        self.problems = []


class SelfPlaySchedule:
    def __init__(self, decks: DeckPool, pool: SnapshotPool, config: DuelConfig | None = None, *,
                 selfplay_fraction: float = 0.75, seed: int = 0, evolved: EvolvedDecks | None = None) -> None:  # fmt: skip
        if not 0 <= selfplay_fraction <= 1:
            raise ValueError("selfplay_fraction must be in [0, 1]")
        self.decks = decks
        self.evolved = evolved
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

    def _sample_decks(self) -> tuple[Deck, Deck, int | None]:
        """A deck pairing, and which of the two (0 = a, 1 = b) is an evolved deck (None: a corpus pairing).
        Draws nothing beyond the corpus pairing while no evolved deck is live."""
        ev = self.evolved
        if ev is not None and ev.share > 0 and ev.ids("probation", "active") and self.rng.random() < ev.share:
            ids, p = ev.weights()
            deck = ev.decks[ids[int(self.rng.choice(len(ids), p=p))]]
            others = self.decks.decks + [ev.decks[i] for i in ev.ids("history")]
            other = others[int(self.rng.integers(len(others)))]
            side = int(self.rng.integers(2))
            return (deck, other, 0) if side == 0 else (other, deck, 1)
        deck_a, deck_b = self.decks.sample(self.rng)
        return deck_a, deck_b, None

    def _deal(self) -> None:
        deck_a, deck_b, evolved = self._sample_decks()
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
            if evolved is not None:
                info["evolved"] = (deck_a, deck_b)[evolved].name
                info["evolved_seat"] = (evolved + first) % 2
            # engine seat p holds deck (first + p) % 2, so deck `side` sits at seat (side + first) % 2
            self._queue.append(Assignment(spec, opponent, (side + first) % 2, info))

    def state_dict(self) -> dict:
        return {"deals": self.deals, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.deals = int(state["deals"])
        self.rng.bit_generator.state = state["rng"]
        self._queue.clear()
        self._held.clear()


__all__ = ["DECK_STATUSES", "MANIFEST_FORMAT", "DeckPool", "EvolvedDecks", "PAIRINGS", "SelfPlaySchedule", "Snapshot",
           "SnapshotPool"]  # fmt: skip
