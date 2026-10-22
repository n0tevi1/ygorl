"""The ygopro shadow duel of the Solo player (design 10 「自博弈的架构」): a reconstruction that keeps going.

:func:`ygorl.advisor.reconstruct.build` stops at the advisor's first idle decision; :class:`ShadowDuel` steps the
same core on with the advisor's choices and stops only at decisions Master Duel would actually ask. On the way it
answers, and logs (:class:`AutoStep`), what MD never asks:

- ``single``: one choosable option besides cancel / unselect (a chain window with only "pass", the last free zone);
- ``equivalent``: a multi-select already under way whose remaining options are copies of one card in one location
  (MD fills in the Queen's Knight from the Deck by itself). A fresh decision is asked even then: MD listed the three
  Arcana Knight Joker of the Extra Deck in the live Face Card Fusion case;
- ``opponent``: every decision of the opponent (engine player 1). Its hidden cards are sampled, so the shadow never
  lets it respond: pass / no / cancel / finish when offered, else the first option. ``could_respond`` marks the
  steps where the sampled cards offered a response the shadow swallowed.

Command points (idle, battle) are always asked. Multi-selects are fed one pick per :meth:`ShadowDuel.step`, as MD
asks them (``SelectCard`` sub-steps of the tracker; ``SelectUnselectCard`` is one card per core message anyway).
The shadow stops for good once the duel is over or the opponent's turn starts (it is rebuilt at the next command
point, design 10 「重新同步」).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from ygorl.advisor.reconstruct import DECLINE, OpponentPrior, Reconstruction, build, complete
from ygorl.advisor.snapshot import BoardSnapshot
from ygorl.cards.ydk import Deck
from ygorl.engine import messages as M
from ygorl.engine.actions import Action
from ygorl.engine.duel import advance, default_cards
from ygorl.engine.tracker import DecisionPoint
from ygorl.selfplay.bridge import Option, find

ADVISOR = 0
MAX_AUTO_STEPS = 512  # auto answers between two advisor decisions before the shadow gives up (an engine loop)
NOT_CHOICES = ("cancel", "unselect")  # backing out of a decision; never what MD would ask about
COMMANDS = (M.SelectIdleCmd, M.SelectBattleCmd)  # command points: MD always waits for the player


class ShadowError(RuntimeError):
    """The shadow duel cannot go on (no advisor decision pending, an engine loop)."""


@dataclass(frozen=True)
class AutoStep:
    """One decision the shadow answered by itself."""

    index: int  # DecisionPoint.index
    player: int  # engine player: 0 advisor, 1 opponent
    decision: str  # decision class name (SelectChain, SelectUnselectCard, ...)
    reason: str  # "single" | "equivalent" | "opponent"
    action: Action
    options: int  # choosable actions offered
    could_respond: bool = False  # opponent: a non-declining option existed and was not taken


def _card_key(a: Action) -> tuple[int, int, int] | None:
    return None if a.card is None else (a.card.code, a.card.loc.controller, a.card.loc.location)


def _picking(point: DecisionPoint) -> bool:
    """A multi-select with at least one card already picked."""
    if getattr(point.state, "picked", None):
        return True
    return isinstance(point.decision, M.SelectUnselectCard) and bool(point.decision.unselectable)


def auto_answer(point: DecisionPoint) -> tuple[str, int] | None:
    """(reason, action index) for an advisor decision MD does not ask, else None (see the module docstring)."""
    if isinstance(point.decision, COMMANDS):
        return None
    choosable = point.choosable
    choices = [i for i in choosable if point.actions[i].kind not in NOT_CHOICES]
    if len(choices) <= 1:
        return "single", (choices or list(choosable))[0]
    keys = {_card_key(point.actions[i]) for i in choices}
    if _picking(point) and len(keys) == 1 and None not in keys:
        return "equivalent", choices[0]
    return None


def opponent_answer(point: DecisionPoint) -> tuple[int, bool]:
    """(action index, could_respond) for an opponent decision: decline when it can, else the first option."""
    choosable = point.choosable
    decline = next((i for i in choosable if point.actions[i].kind in DECLINE), None)
    if decline is None:
        return choosable[0], False
    return decline, any(point.actions[i].kind not in DECLINE for i in choosable)


@dataclass
class ShadowDuel:
    """A reconstruction stepped on by the advisor's choices, stopping only where MD asks."""

    rec: Reconstruction
    log: list[AutoStep] = field(default_factory=list)  # every auto answer, in order
    stopped: str = ""  # "" while running; "end" (duel over) or "opponent_turn"

    def __post_init__(self) -> None:
        self._settle()

    @classmethod
    def from_snapshot(cls, snapshot: BoardSnapshot, own_deck: Deck, prior: OpponentPrior, rule_flags: int, *,
                      cards: Mapping | None = None, scripts=None, rng: random.Random | int = 0, seed: int = 1
                      ) -> ShadowDuel:  # fmt: skip
        """Complete ``snapshot`` (one sample of the hidden cards, drawn with ``rng``) and build the shadow duel."""
        cards = cards if cards is not None else default_cards()
        rng = rng if isinstance(rng, random.Random) else random.Random(rng)
        board = complete(snapshot, own_deck, prior, cards, rng)
        return cls(build(board, rule_flags, cards=cards, scripts=scripts, seed=seed))

    @property
    def done(self) -> bool:
        return bool(self.stopped)

    @property
    def point(self) -> DecisionPoint | None:
        """The advisor's pending decision (None once stopped)."""
        return None if self.stopped else self.rec.tracker.point()

    def step(self, index: int) -> list[AutoStep]:
        """Answer the advisor's decision with action ``index`` (one pick of a multi-select) and run to the next
        decision MD would ask; return the auto answers given on the way."""
        point = self.point
        if point is None:
            raise ShadowError(f"the shadow duel has stopped ({self.stopped})")
        start = len(self.log)
        self._act(index)
        self._settle()
        return self.log[start:]

    def pick(self, option: Option, kind: str | None = None) -> list[AutoStep]:
        """:meth:`step` with the action (of ``kind``, if given) on the card, zone or value MD names: one pick of a
        multi-select as MD asks it, or a command on a card."""
        point = self.point
        if point is None:
            raise ShadowError(f"the shadow duel has stopped ({self.stopped})")
        index = find(point, option, kind)
        if index is None:
            raise ShadowError(f"{option} is not an option of {type(point.decision).__name__}")
        return self.step(index)

    def _act(self, index: int) -> None:
        tracker = self.rec.tracker
        response = tracker.act(index)
        if response is not None and not tracker.done:
            advance(self.rec.core, tracker, response)

    def _settle(self) -> None:
        """Answer what MD does not ask until an advisor decision is left, or stop."""
        tracker = self.rec.tracker
        for _ in range(MAX_AUTO_STEPS):
            point = None if tracker.done else tracker.point()
            if point is None:
                self.stopped = "end"
                return
            if point.turn_player != ADVISOR:
                self.stopped = "opponent_turn"
                return
            could_respond = False
            if point.player != ADVISOR:
                reason = "opponent"
                index, could_respond = opponent_answer(point)
            else:
                auto = auto_answer(point)
                if auto is None:
                    return
                reason, index = auto
            self.log.append(AutoStep(point.index, point.player, type(point.decision).__name__, reason,
                                     point.actions[index], len(point.choosable), could_respond))  # fmt: skip
            self._act(index)
        raise ShadowError(f"no advisor decision after {MAX_AUTO_STEPS} auto answers")


__all__ = ["AutoStep", "ShadowDuel", "ShadowError", "auto_answer", "opponent_answer"]
