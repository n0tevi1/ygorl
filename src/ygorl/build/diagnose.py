"""Per-card signals of one deck from its own games (#107, docs/spikes/deck-evolution.md §3).

The opening-hand effect is a natural experiment: the shuffle deals the opening hand, so within one deck, one pilot
and one opponent distribution, the win rate of games where a card was in the opening hand minus the win rate of
games where it was not is that card's causal effect relative to an average other card of the deck being there
instead. It needs no extra games: only each game's shuffled deck order and its result.

The engine draws from the END of the main deck list it was given, so with the host-side shuffle already applied
(``DuelConfig.shuffle_decks = False`` in a game spec, as the arena and the batched path use), a player's opening
hand is the last ``OPENING_HAND`` cards of that player's main deck in the spec (``tests/test_diagnose.py`` locks this
against the engine).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

OPENING_HAND = 5


def opening_hand(main: Sequence[int], size: int = OPENING_HAND) -> tuple[int, ...]:
    """The opening hand dealt from a main deck already in load order (the last ``size`` cards)."""
    return tuple(main[-size:])


@dataclass(frozen=True)
class CardEffect:
    """One card's opening-hand effect in one deck: win rate with it in the opening hand minus without."""

    card: int
    copies: int
    games_in: int  # games with at least one copy in the opening hand
    games_out: int
    win_in: float
    win_out: float

    @property
    def effect(self) -> float:
        return self.win_in - self.win_out

    @property
    def stderr(self) -> float:
        """Standard error of the difference of two independent proportions (draws count half)."""
        if self.games_in < 2 or self.games_out < 2:
            return math.inf
        return math.sqrt(
            self.win_in * (1 - self.win_in) / self.games_in + self.win_out * (1 - self.win_out) / self.games_out
        )


def opening_effects(hands: Sequence[Sequence[int]], scores: Sequence[float], deck: Sequence[int]) -> list[CardEffect]:
    """Per distinct card of ``deck``: opening-hand effect over games with ``hands[i]`` and result ``scores[i]``
    (1 / 0.5 / 0 from the deck's side; NaN games are skipped). Sorted by effect, highest first."""
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    out = []
    for card in sorted(set(deck)):
        held = np.array([card in h for h in hands]) & ok
        rest = ~held & ok
        n_in, n_out = int(held.sum()), int(rest.sum())
        out.append(CardEffect(card=card, copies=list(deck).count(card), games_in=n_in, games_out=n_out,
                              win_in=float(s[held].mean()) if n_in else math.nan,
                              win_out=float(s[rest].mean()) if n_out else math.nan))  # fmt: skip
    return sorted(out, key=lambda e: -e.effect if math.isfinite(e.effect) else math.inf)


__all__ = ["OPENING_HAND", "CardEffect", "opening_effects", "opening_hand"]
