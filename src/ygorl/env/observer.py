"""Network observations of Python :class:`DecisionPoint` s: the four board arrays plus the event window.

:class:`PointObserver` combines the reference encoders — :class:`~ygorl.env.encoding.ObservationEncoder`
(card table, globals, candidate actions; needs the live core) and :class:`~ygorl.env.events.EventHistory`
(the event-token window, fed with every point's messages) — into the same observation dict
``EncodedVecEnv`` produces in C++ (docs/encoding.md). It serves the Python paths that see
``DecisionPoint`` s instead of encoded events: behaviour cloning on replayed demonstrations (T4a.2) and
network agents in ``Duel.run`` / the arena (``Agent.observe``).

One observer per duel. Every decision point of the duel, whoever decides it, must pass through
:meth:`observe` in order (the event stream is omniscient and filtered per viewer); :meth:`encode`
observes the point too. With selection history enabled, call :meth:`on_decision` after
every accepted action and before observing its successor, including forced choices.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ygorl.cards.cdb import CardVocab
from ygorl.engine.duel import DecisionPoint
from ygorl.env.encoding import ObservationEncoder
from ygorl.env.events import DEFAULT_EVENT_LENGTH, EventHistory


class PointObserver:
    """Observation dicts (``cards``, ``globals``, ``actions``, ``action_mask``, ``events``, ``event_mask``) of one duel."""

    def __init__(self, cards: Mapping, vocab: CardVocab, event_length: int = DEFAULT_EVENT_LENGTH,
                 starting_lp: int | None = None, *, selection_history: bool = False) -> None:  # fmt: skip
        """``starting_lp``: the duel's starting LP (event tokens track LP); default: read from the first point."""
        if selection_history and event_length <= 0:
            raise ValueError("selection history requires a nonempty event window")
        self.selection_history = selection_history
        self._last_action = -1
        self._last_action_index = None
        self.encoder = ObservationEncoder(cards, vocab)
        self.cards, self.vocab = cards, vocab
        self.event_length = event_length
        self.starting_lp = starting_lp
        self.history: EventHistory | None = None
        self._last = -1  # DecisionPoint.index of the last point fed

    def observe(self, point: DecisionPoint) -> None:
        """Feed the messages that led to ``point`` (idempotent per point; points must come in order)."""
        if point.index <= self._last:
            return
        if self.history is None:
            lp = self.starting_lp if self.starting_lp is not None else point.lp[0]
            self.history = EventHistory(
                self.cards, self.vocab, self.event_length, starting_lp=lp, selection_history=self.selection_history
            )
        self.history.feed(point.events)
        self._last = point.index

    def on_decision(self, point: DecisionPoint, index: int) -> None:
        """Call once after an accepted action, before observing the next point."""
        if not 0 <= index < len(point.actions):
            raise IndexError("selection action index out of range")
        if point.index == self._last_action:
            if index != self._last_action_index:
                raise ValueError("conflicting action for an already recorded decision")
            return
        if point.index < self._last_action:
            raise ValueError("selection actions must arrive in order")
        if point.index < self._last:
            raise ValueError("selection action arrived after a later observation")
        self.observe(point)
        self.history.on_action(point.player, point.decision.TYPE, point.actions[index])
        self._last_action = point.index
        self._last_action_index = index

    def encode(self, point: DecisionPoint, core) -> dict[str, np.ndarray]:
        """The observation of ``point`` for its deciding player (``core``: the duel's live ``_core.Duel``)."""
        self.observe(point)
        obs = self.encoder.encode(point, core)
        if self.event_length:
            obs.update(self.history.encode(point.player))
        if self.selection_history:
            obs["selection_history"] = np.ones(1, dtype=np.int32)
        return obs


__all__ = ["PointObserver"]
