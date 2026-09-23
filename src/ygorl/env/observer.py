"""Network observations of Python :class:`DecisionPoint` s: the four board arrays plus the event window.

:class:`PointObserver` combines the reference encoders — :class:`~ygorl.env.encoding.ObservationEncoder`
(card table, globals, candidate actions; needs the live core) and :class:`~ygorl.env.events.EventHistory`
(the event-token window, fed with every point's messages) — into the same observation dict
``EncodedVecEnv`` produces in C++ (docs/encoding.md). It serves the Python paths that see
``DecisionPoint`` s instead of encoded events: behaviour cloning on replayed demonstrations (T4a.2) and
network agents in ``Duel.run`` / the arena (``Agent.observe``).

One observer per duel. Every decision point of the duel, whoever decides it, must pass through
:meth:`observe` in order (the event stream is omniscient and filtered per viewer); :meth:`encode`
observes the point too, so a caller that encodes every point needs nothing else.
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
                 starting_lp: int | None = None) -> None:  # fmt: skip
        """``starting_lp``: the duel's starting LP (event tokens track LP); default: read from the first point."""
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
            self.history = EventHistory(self.cards, self.vocab, self.event_length, starting_lp=lp)
        self.history.feed(point.events)
        self._last = point.index

    def encode(self, point: DecisionPoint, core) -> dict[str, np.ndarray]:
        """The observation of ``point`` for its deciding player (``core``: the duel's live ``_core.Duel``)."""
        self.observe(point)
        obs = self.encoder.encode(point, core)
        if self.event_length:
            obs.update(self.history.encode(point.player))
        return obs


__all__ = ["PointObserver"]
