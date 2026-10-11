"""The league (design I1): who the learner plays against and how it is doing against them.

:class:`League` owns the bookkeeping between rollouts: the snapshot pool's results, the admission window that decides
whether a due snapshot joins the pool (``snapshot_min_win_rate``), keep-best, and the scores of evolved decks. Its
interface is finished games in (:meth:`League.record`) and decisions out (:meth:`League.admit`,
:meth:`League.offer_best`); the :class:`~ygorl.train.selfplay.SelfPlaySchedule` deals opponents from the same
pool. It plays no games and trains nothing, so it is tested with hand-made :class:`FinishedGame` records.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import nn

from ygorl.train.rollout import FinishedGame
from ygorl.train.selfplay import EvolvedDecks, SnapshotPool


class League:
    def __init__(self, pool: SnapshotPool, evolved: EvolvedDecks | None = None, *,
                 min_win_rate: float | None = None, min_games: int = 0) -> None:  # fmt: skip
        self.pool = pool
        self.evolved = evolved
        self.min_win_rate = min_win_rate  # None: every due snapshot joins
        self.min_games = min_games
        self.window = {"games": 0, "points": 0.0}  # the learner's pool games since the last snapshot joined
        self.best = {"score": None, "update": None}  # keep-best: the evaluation score of best.pt

    def record(self, games: Iterable[FinishedGame]) -> int:
        """Credit finished games: the learner's pool games (not truncated) go to their snapshot's record and the
        admission window; evolved decks the learner piloted get their score. Returns how many games had an evolved
        deck (finished, no error), the learner's or not."""
        evolved_games = 0
        for g in games:
            if g.learner_score is not None and not g.truncated:
                self.pool.record(g.assignment.opponent, g.learner_score)
                self.window["games"] += 1
                self.window["points"] += g.learner_score
            did = g.assignment.info.get("evolved")
            if did is None or g.truncated or g.reason == "error":
                continue
            seat = g.assignment.info["evolved_seat"]
            evolved_games += 1
            if self.evolved is not None and g.assignment.is_learner(seat):
                self.evolved.record(did, 0.5 if g.winner is None else float(g.winner == seat))
        return evolved_games

    def admit(self, model: nn.Module, update: int) -> bool:
        """Add a snapshot of ``model`` to the pool, unless the admission threshold holds it back: with
        ``min_win_rate`` set and snapshots in the pool, the learner must have scored above it in at least
        ``min_games`` pool games since the last one joined. Returns whether it joined (the window restarts)."""
        w = self.window
        regular = [i for i in self.pool.ids() if i >= 0]
        if self.min_win_rate is not None and regular:
            if not w["games"] or w["games"] < self.min_games or w["points"] / w["games"] <= self.min_win_rate:
                return False
        self.pool.add(model, update=update)
        self.window = {"games": 0, "points": 0.0}
        return True

    def offer_best(self, model: nn.Module, update: int, score: float | None) -> bool:
        """Keep-best: ``model`` (evaluated at ``score``, None = no valid evaluation) becomes the pool's best snapshot
        if it beats the best so far. Returns whether it did."""
        if score is None or (self.best["score"] is not None and score <= self.best["score"]):
            return False
        self.best = {"score": score, "update": update}
        self.pool.set_best(model, update=update, score=score)
        return True

    # -- persistence (the keys of a PPO checkpoint, docs/training.md) ------------------------------------------------
    def state_dict(self) -> dict:
        return {"pool": self.pool.state_dict(), "best": dict(self.best), "league": dict(self.window),
                "evolved": self.evolved.state_dict() if self.evolved is not None else None}  # fmt: skip

    def load_state_dict(self, state: dict, new_model: Callable[[], nn.Module]) -> None:
        self.pool.load_state_dict(state["pool"], new_model)
        self.best.update(state["best"])
        self.window.update(state.get("league", {}))
        if self.evolved is not None and state.get("evolved") is not None:
            self.evolved.load_state_dict(state["evolved"])


__all__ = ["League"]
