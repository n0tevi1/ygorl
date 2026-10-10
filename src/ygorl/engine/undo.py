"""The no-op undo rules (docs/encoding.md 「撤销类空操作」): which actions of a decision are masked.

Rules 1 and 2 mask actions that only undo the previous step (backing out of a command just started from a menu,
reversing the previous select / unselect); rules 3–6 mask actions that change nothing (shuffling the hand) or would
let a loop run until the decision limit (the same menu effect again, a long selection moving backwards, cancelling
target / material prompts over and over). :class:`UndoRules` is the whole rule set behind four calls; the tracker
(:class:`ygorl.engine.tracker.DuelTracker`) feeds it and puts :meth:`UndoRules.undo` into ``DecisionPoint.undo``.

The C++ mirror is ``UndoRules`` in ``csrc/host.h``. Both read their limits from ``_core.undo_limits()`` when a duel
starts, so the limits are written once (in C++); :func:`override_limits` lowers them for both hosts, which is how
tests/test_undo.py makes random lockstep play reach every rule.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ygorl import _core
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action

_MENUS = (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)
_SELECTIONS = (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD)
_INVERSE = {"select": "unselect", "unselect": "select"}


@dataclass(frozen=True)
class UndoLimits:
    """Limits of rules 3, 5 and 6 (defaults: ``_core.DEFAULT_UNDO_LIMITS``)."""

    # rule 3: an effect activated this many times from a menu in one turn is masked there. A negated monster can
    # activate an unlimited ignition effect for free forever (Expurrely Happiness); real play rarely activates one
    # effect from the menu more than 3 times a turn
    menu_activations: int
    # rule 5: after this many steps in one SELECT_UNSELECT_CARD selection unselecting is masked, so the selection can
    # only move forward: a policy that never picks the card a finish needs (Swordsoul Blackout needs your Wyrm) would
    # otherwise toggle the others until the decision limit
    selection_steps: int
    # rule 6: cancellations across target / material prompts (SELECT_CARD, SELECT_UNSELECT_CARD) of one player with
    # no game event in between
    selection_cancels: int

    @classmethod
    def current(cls) -> UndoLimits:
        """The limits duels started now use (``_core.undo_limits()``)."""
        return cls(**_core.undo_limits())


DEFAULT_LIMITS = UndoLimits(**_core.DEFAULT_UNDO_LIMITS)


@contextmanager
def override_limits(**limits: int) -> Iterator[UndoLimits]:
    """Change some limits (``menu_activations``, ``selection_steps``, ``selection_cancels``) for duels started inside
    the block, on both hosts; restores the old ones after. Process-wide: not for concurrent duels with other limits."""
    old = _core.set_undo_limits(**limits)
    try:
        yield UndoLimits.current()
    finally:
        _core.set_undo_limits(**old)


def card_key(action: Action) -> tuple | None:
    """A card by password and place (controller, location, sequence), as the rules tell cards apart."""
    c = action.card
    return None if c is None else (c.code, c.loc.controller, c.loc.location, c.loc.sequence)


class UndoRules:
    """The rule state of one duel. Feed it, in order: :meth:`on_game_event` for every message that changes the game
    (anything but decisions, retries and hints), :meth:`on_decision` when a decision is asked, and :meth:`on_action`
    for every action taken; :meth:`undo` gives the masked actions of the current decision. Limits are read when the
    duel starts (:meth:`UndoLimits.current`)."""

    def __init__(self, limits: UndoLimits | None = None) -> None:
        self.limits = limits if limits is not None else UndoLimits.current()
        self._inside: int | None = None  # rule 1: the player inside a command just started from a menu
        self._toggle: tuple | None = None  # rule 2: (player, kind, card key) of the last select / unselect
        self._selection_steps = 0  # rule 5: steps of the current SELECT_UNSELECT_CARD selection
        self._selection_player = -1
        self._selection_cancels = 0  # rule 6
        self._selection_cancel_player = -1
        self._activations: dict[tuple, int] = {}  # rule 3: (player, card key, description) -> activations this turn
        self._activations_turn = -1

    @property
    def selection_cancels(self) -> int:
        """Rule 6's count of cancellations in the current target / material prompts (the cancel budget reads it)."""
        return self._selection_cancels

    def on_game_event(self) -> None:
        self._inside = self._toggle = None
        self._selection_steps = 0
        self._selection_cancels = 0

    def on_decision(self, decision: M.Decision) -> None:
        player, kind = decision.player, decision.TYPE
        if player != self._inside or kind in _MENUS:
            self._inside = None  # another player's decision, or back at a menu: not inside a command any more
        if self._toggle is not None and self._toggle[0] != player:
            self._toggle = None
        if kind != C.MSG_SELECT_UNSELECT_CARD or player != self._selection_player:
            self._selection_steps = 0  # not the same selection any more
            self._selection_player = player
        if player != self._selection_cancel_player or kind not in _SELECTIONS:
            self._selection_cancels = 0
        self._selection_cancel_player = player

    def undo(self, decision: M.Decision, actions: list[Action], turn: int) -> tuple[int, ...]:
        """Indices of ``actions`` (offered at ``decision`` in turn ``turn``) that are masked; never all of them."""
        kind, limits = decision.TYPE, self.limits
        undo = []
        for i, a in enumerate(actions):
            if a.kind == "cancel" and self._inside is not None:
                undo.append(i)  # backs out of the command to the unchanged menu (rule 1)
            elif kind in _SELECTIONS and a.kind == "cancel" and self._selection_cancels >= limits.selection_cancels:
                undo.append(i)  # repeated target/material cancellation without game progress (rule 6)
            elif (self._toggle is not None and kind == C.MSG_SELECT_UNSELECT_CARD
                  and (_INVERSE.get(a.kind), card_key(a)) == self._toggle[1:]):  # fmt: skip
                undo.append(i)  # reverses the previous select / unselect (rule 2)
            elif (kind == C.MSG_SELECT_UNSELECT_CARD and a.kind == "unselect"
                  and self._selection_steps >= limits.selection_steps):  # fmt: skip
                undo.append(i)  # a long selection only moves forward from here (rule 5)
            elif kind in _MENUS and a.kind == "shuffle":
                undo.append(i)  # reorders the hand, changes nothing else (rule 4)
            elif (kind in _MENUS and a.kind == "activate" and self._activations_turn == turn
                  and self._activations.get((decision.player, card_key(a), a.description), 0) >= limits.menu_activations):  # fmt: skip
                undo.append(i)  # the same effect again: repeated activation limit (rule 3)
        return tuple(undo) if len(undo) < len(actions) else ()

    def on_action(self, decision: M.Decision, action: Action, turn: int) -> None:
        kind = decision.TYPE
        if kind in _MENUS:
            self._inside = decision.player
            if action.kind == "activate":
                if self._activations_turn != turn:
                    self._activations, self._activations_turn = {}, turn
                key = (decision.player, card_key(action), action.description)
                self._activations[key] = self._activations.get(key, 0) + 1
        self._toggle = None
        if kind == C.MSG_SELECT_UNSELECT_CARD and action.kind in _INVERSE:
            self._toggle = (decision.player, action.kind, card_key(action))
        if kind == C.MSG_SELECT_UNSELECT_CARD:
            self._selection_steps += 1
        if kind in _SELECTIONS and action.kind == "cancel":
            self._selection_cancels += 1


__all__ = ["DEFAULT_LIMITS", "UndoLimits", "UndoRules", "card_key", "override_limits"]
