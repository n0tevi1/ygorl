"""Host-side bookkeeping of one duel, independent of who drives the core: :class:`DuelTracker`.

The tracker consumes every engine message buffer (LP, turn, phase, events, the pending decision), turns the
pending decision into :class:`DecisionPoint` sub-steps and their answers into ``set_response`` bytes, masks the
no-op undo actions (docs/encoding.md 「撤销类空操作」), answers for a restricted opponent under a curriculum
(:func:`ygorl.engine.curriculum.auto_action`) and reports the :class:`DuelResult`. :class:`ygorl.engine.duel.Duel`,
:class:`ygorl.engine.duel.DuelSession` and :class:`ygorl.env.pool.VecDuelEnv` all drive the core through it; the C++
mirror is ``Tracker`` in ``csrc/host.h``.

This module does not import :mod:`ygorl.engine.duel` (which re-exports its public names), so there is no cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ygorl import _core
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action, DecisionState, make_decision
from ygorl.engine.curriculum import FULL, allowed_actions, auto_action

if TYPE_CHECKING:
    from ygorl.engine.duel import DuelConfig

WIN_REASON_LP = 1  # MSG_WIN reasons written by the core (processor.cpp); 0x10+ come from card scripts
WIN_REASON_DECK_OUT = 2
MAX_CONSECUTIVE_RETRIES = 8


def deck_of_seat(first: int, seat: int) -> int:
    """Index into (a, b) of the deck at engine ``seat`` when ``first`` names the deck that moves first.

    Engine seat 0 always moves first, so seat ``p`` holds deck ``(first + p) % 2``; the one place this rule is written.
    """
    return (first + seat) % 2


def seat_of_deck(first: int, deck: int) -> int:
    """Engine seat of deck ``deck`` (0 = a, 1 = b): the inverse of :func:`deck_of_seat` (the rule is its own inverse)."""
    return deck_of_seat(first, deck)


@dataclass(frozen=True)
class DecisionPoint:
    index: int  # decision counter over the whole duel (each sub-step counts)
    player: int  # engine player asked to act (0 moves first)
    turn: int
    phase: int
    lp: tuple[int, int]  # indexed by engine player
    decision: M.Decision
    actions: list[Action]
    state: DecisionState
    events: tuple[M.Message, ...]  # messages since the previous decision point
    turn_player: int = 0  # engine player whose turn it is
    augmented_start: bool = False  # DuelConfig.augmented_start
    response_index: int = 0  # position this decision's response takes in DuelResult.responses
    undo: tuple[int, ...] = ()  # actions that only undo the previous step (docs/encoding.md 「撤销类空操作」)


@dataclass
class DuelResult:
    winner: int | None  # 0 = deck_a / agent_a, 1 = deck_b / agent_b, None = draw
    reason: str  # "win", "turn_limit", "decision_limit", "end", "error", "log_exhausted" (replay)
    win_reason: int | None = None  # MSG_WIN reason: WIN_REASON_LP, WIN_REASON_DECK_OUT, or 0x10+ (card effects)
    turns: int = 0
    decisions: int = 0
    lp: tuple[int, int] = (0, 0)  # in (a, b) order
    first: int = 0
    retries: int = 0
    unknown_messages: int = 0
    undecodable_messages: int = 0
    script_errors: list[str] = field(default_factory=list)
    responses: list[bytes] = field(default_factory=list)  # every set_response, in order
    actions: list[int] = field(default_factory=list)  # every agent action index, in order
    message_log: list[bytes] = field(default_factory=list)  # raw engine buffers (record_messages=True)
    steps: list[dict] = field(default_factory=list)  # per-action candidates/choice/probs (record_steps=True)
    error: str = ""
    auto_decisions: int = 0  # decisions the host answered for the opponent (curriculum); their bytes are in responses

    def summary(self) -> str:
        who = {0: "a", 1: "b", None: "draw"}[self.winner]
        return (f"winner={who} reason={self.reason} win_reason={self.win_reason} turns={self.turns} "
                f"decisions={self.decisions} lp={self.lp} retries={self.retries} unknown={self.unknown_messages} "
                f"undecodable={self.undecodable_messages} script_errors={len(self.script_errors)} {self.error}")  # fmt: skip


_NOT_EVENTS = (M.Decision, M.Retry, M.Hint, M.Waiting, M.CardHint, M.PlayerHint, M.ShowHint)  # no game change
_MENUS = (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)
# an effect activated this many times from a menu in one turn is masked there (docs/encoding.md 「撤销类空操作」 rule 3):
# a negated monster can activate an unlimited ignition effect for free forever (Expurrely Happiness); real play
# rarely activates one effect from the menu more than 3 times a turn
MAX_MENU_ACTIVATIONS = 8
# engine steps (process() calls) allowed between two decisions: a core that keeps processing without ever asking a
# player would hang the duel (the decision limit never fires); real chains take a few thousand at most (mirror of
# the C++ g_max_engine_steps)
MAX_ENGINE_STEPS = 100_000
# after this many steps in one SELECT_UNSELECT_CARD selection, unselecting is masked, so the selection can only move
# forward (docs/encoding.md 「撤销类空操作」 rule 5): a policy that never picks the card a finish needs (Swordsoul
# Blackout needs your Wyrm) would otherwise toggle the others until the decision limit
MAX_SELECTION_STEPS = 32
# Across target/material prompts with no game event, bound cancellations that only return to target selection.
MAX_SELECTION_CANCELS = 32
_INVERSE = {"select": "unselect", "unselect": "select"}


def _card_key(action: Action) -> tuple | None:
    c = action.card
    return None if c is None else (c.code, c.loc.controller, c.loc.location, c.loc.sequence)


class DuelTracker:
    """Host-side state of one duel, independent of who drives the core.

    Feed every engine buffer to :meth:`on_buffer`. When :attr:`awaiting`,
    either answer with recorded bytes (:meth:`use_response`) or first let the
    host answer for a restricted opponent (:meth:`auto_response`, curriculum
    modes) and otherwise ask agents: :meth:`point` gives the current decision
    point and :meth:`act` applies an action index, returning the response bytes
    once the decision is complete.
    Engine player indices are used internally; :meth:`finish` reports in
    (a, b) order.
    """

    def __init__(self, config: DuelConfig, first: int, cards, *, record_messages: bool = False,
                 record_steps: bool = False, reference_log: list[bytes] | None = None) -> None:  # fmt: skip
        self.config = config
        self.first = first
        self.cards = cards
        self.record_messages = record_messages
        self.record_steps = record_steps
        self.reference_log = reference_log
        self.result = DuelResult(winner=None, reason="", first=first)
        self.lp = [config.player.starting_lp, config.player.starting_lp]
        self.turn = 0
        self.turn_player = 0
        self.phase = 0
        self.events: list[M.Message] = []
        self.state: DecisionState | None = None
        self.decision: M.Decision | None = None
        self.done = False
        self._engine_winner: int | None = None
        self._last_decision: M.Decision | None = None
        self._consecutive_retries = 0
        self._engine_steps = 0  # process() calls since the last decision
        self._buffers = 0
        self._point: DecisionPoint | None = None
        self._allowed: list[int] | None = None  # point.actions -> state.actions() indices when filtered
        # no-op undo tracking (docs/encoding.md 「撤销类空操作」): the player inside a command just started from a
        # menu, and the last select / unselect of a SELECT_UNSELECT_CARD; any game event clears both
        self._inside: int | None = None
        self._toggle: tuple | None = None
        # rule 5: steps of the current SELECT_UNSELECT_CARD selection (same player, no game event in between)
        self._selection_steps = 0
        self._selection_player = -1
        self._selection_cancels = 0
        self._selection_cancel_player = -1
        self._activations: dict[tuple, int] = {}  # (player, card key, description) -> menu activations this turn
        self._activations_turn = -1
        self._stepped = False
        self._learner = seat_of_deck(first, config.learner)  # engine player of the learning deck

    @property
    def awaiting(self) -> bool:
        return not self.done and self.state is not None

    def stop(self, reason: str, error: str = "") -> None:
        self.result.reason = reason
        if error:
            self.result.error = error
        self.done = True
        self.state = None

    def on_buffer(self, buf: bytes, status: int, logs=()) -> None:
        """Consume one engine message buffer and the status of the ``process()`` that produced it."""
        res = self.result
        for t, text in logs:
            if t == _core.LOG_TYPE_ERROR:
                res.script_errors.append(text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text))
        if self.record_messages:
            res.message_log.append(buf)
        if self.reference_log is not None:
            i = self._buffers
            if i >= len(self.reference_log) or self.reference_log[i] != buf:
                raise ValueError(f"replay differs from the reference at message buffer {i}")
        self._buffers += 1
        self.state = None
        decision: M.Decision | None = None
        retried = False
        lp = self.lp
        for msg in M.decode_buffer(buf):
            self.events.append(msg)
            if not isinstance(msg, _NOT_EVENTS):
                self._inside = self._toggle = None
                self._selection_steps = 0
                self._selection_cancels = 0
            if isinstance(msg, M.Decision):
                decision = msg
            elif isinstance(msg, M.NewTurn):
                self.turn += 1
                self.turn_player = msg.player
            elif isinstance(msg, M.NewPhase):
                self.phase = msg.phase
            elif isinstance(msg, (M.Damage, M.PayLpCost)):
                lp[msg.player] -= msg.amount  # the core does not clamp at 0
            elif isinstance(msg, M.Recover):
                lp[msg.player] += msg.amount
            elif isinstance(msg, M.LpUpdate):
                lp[msg.player] = msg.amount
            elif isinstance(msg, M.Win):
                self._engine_winner = msg.player if msg.player in (0, 1) else None
                res.win_reason = msg.reason
                res.reason = "win"
            elif isinstance(msg, M.Retry):
                retried = True
                res.retries += 1
            elif isinstance(msg, M.UnknownMessage):
                res.unknown_messages += 1
            elif isinstance(msg, M.UndecodableMessage):
                res.undecodable_messages += 1

        # The core keeps processing after MSG_WIN; like EDOPro's host we stop at the first one.
        if res.reason == "win":
            self.done = True
            return
        if status == _core.DUEL_STATUS_END:
            self.stop("end")
            return
        if status != _core.DUEL_STATUS_AWAITING:
            self._engine_steps += 1
            if self._engine_steps >= MAX_ENGINE_STEPS:
                self.stop("error", f"engine loop: no decision after {MAX_ENGINE_STEPS} engine steps")
            return
        self._engine_steps = 0
        if decision is None and retried:
            self._consecutive_retries += 1
            if self._consecutive_retries > MAX_CONSECUTIVE_RETRIES:
                self.stop("error", "response rejected repeatedly (MSG_RETRY)")
                return
            decision = self._last_decision
        else:
            self._consecutive_retries = 0
        if decision is None:
            self.stop("error", "engine awaits a response but sent no decodable decision")
            return
        self._last_decision = decision
        decision = M.hide_private(decision)  # what the decider may see (the message log keeps the raw bytes)
        if self.turn > self.config.max_turns:
            self.stop("turn_limit")
            return
        self.decision = decision
        if decision.player != self._inside or decision.TYPE in _MENUS:
            self._inside = None  # another player's decision, or back at a menu: not inside a command any more
        if self._toggle is not None and self._toggle[0] != decision.player:
            self._toggle = None
        if decision.TYPE != C.MSG_SELECT_UNSELECT_CARD or decision.player != self._selection_player:
            self._selection_steps = 0  # not the same selection any more
            self._selection_player = decision.player
        if (decision.player != self._selection_cancel_player
                or decision.TYPE not in (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD)):  # fmt: skip
            self._selection_cancels = 0
        self._selection_cancel_player = decision.player
        self.state = make_decision(decision, self.cards)
        self._point = None
        self._stepped = False  # host answers only fresh decisions, never a half-built multi-select

    def point(self) -> DecisionPoint | None:
        """The current sub-step to ask the deciding agent about (None once the duel had to stop)."""
        if not self.awaiting:
            return None
        if self._point is not None:
            return self._point
        res, state, decision = self.result, self.state, self.decision
        if res.decisions >= self.config.max_decisions:
            self.stop("decision_limit")
            return None
        actions = state.actions()
        if not actions:
            self.stop("error", f"no legal action for {decision.name}")
            return None
        self._allowed = None
        if self._restricted():
            allowed = allowed_actions(self.config.curriculum, decision, actions)
            if len(allowed) < len(actions):
                self._allowed, actions = allowed, [actions[i] for i in allowed]
        self._point = DecisionPoint(res.decisions, decision.player, self.turn, self.phase, (self.lp[0], self.lp[1]),
                                    decision, actions, state, tuple(self.events), self.turn_player,
                                    self.config.augmented_start, len(res.responses), self._undo(decision, actions))  # fmt: skip
        self.events = []
        return self._point

    def act(self, idx, probs=None) -> bytes | None:
        """Apply action ``idx`` to the current point; return the response once the decision is complete."""
        point = self.point()
        if point is None:
            raise RuntimeError("no decision is pending")
        n = len(point.actions)
        if not isinstance(idx, int) or not 0 <= idx < n:
            raise ValueError(f"agent returned action {idx!r}, out of range 0..{n - 1} for {point.decision.name}")
        res = self.result
        if self.record_steps:
            res.steps.append(_step_record(point, idx, probs))
        res.actions.append(idx)
        res.decisions += 1
        self._note_undo(point.decision, point.actions[idx])
        self._point = None
        self._stepped = True
        response = self.state.step(idx if self._allowed is None else self._allowed[idx])
        if response is not None:
            res.responses.append(response)
            self.state = None
        return response

    def _undo(self, decision: M.Decision, actions: list[Action]) -> tuple[int, ...]:
        """Indices of ``actions`` that only undo the previous step (never all of them)."""
        undo = []
        for i, a in enumerate(actions):
            if a.kind == "cancel" and self._inside is not None:
                undo.append(i)  # backs out of the command to the unchanged menu
            elif (decision.TYPE in (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD) and a.kind == "cancel"
                  and self._selection_cancels >= MAX_SELECTION_CANCELS):  # fmt: skip
                undo.append(i)  # repeated target/material cancellation without game progress (rule 6)
            elif (self._toggle is not None and decision.TYPE == C.MSG_SELECT_UNSELECT_CARD
                  and (_INVERSE.get(a.kind), _card_key(a)) == self._toggle[1:]):  # fmt: skip
                undo.append(i)  # reverses the previous select / unselect
            elif (decision.TYPE == C.MSG_SELECT_UNSELECT_CARD and a.kind == "unselect"
                  and self._selection_steps >= MAX_SELECTION_STEPS):  # fmt: skip
                undo.append(i)  # a long selection only moves forward from here (rule 5)
            elif decision.TYPE in _MENUS and a.kind == "shuffle":
                undo.append(i)  # reorders the hand, changes nothing else
            elif (decision.TYPE in _MENUS and a.kind == "activate" and self._activations_turn == self.turn
                  and self._activations.get((decision.player, _card_key(a), a.description), 0) >= MAX_MENU_ACTIVATIONS):  # fmt: skip
                undo.append(i)  # the same effect again: repeated activation limit
        return tuple(undo) if len(undo) < len(actions) else ()

    def _note_undo(self, decision: M.Decision, action: Action) -> None:
        if decision.TYPE in _MENUS:
            self._inside = decision.player
            if action.kind == "activate":
                if self._activations_turn != self.turn:
                    self._activations, self._activations_turn = {}, self.turn
                key = (decision.player, _card_key(action), action.description)
                self._activations[key] = self._activations.get(key, 0) + 1
        self._toggle = None
        if decision.TYPE == C.MSG_SELECT_UNSELECT_CARD and action.kind in _INVERSE:
            self._toggle = (decision.player, action.kind, _card_key(action))
        if decision.TYPE == C.MSG_SELECT_UNSELECT_CARD:
            self._selection_steps += 1
        if decision.TYPE in (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD) and action.kind == "cancel":
            self._selection_cancels += 1

    def _restricted(self) -> bool:
        """The pending decision is the opponent's, in the learner's turn, under a restricting curriculum."""
        return (self.config.curriculum != FULL and self.decision is not None and self.decision.player != self._learner
                and self.turn_player == self._learner)  # fmt: skip

    def auto_response(self) -> bytes | None:
        """Answer the pending decision on the opponent's behalf if the curriculum says so; else None.

        The answer is always a passive one (pass / no / cancel, see
        :mod:`ygorl.engine.curriculum`); it is logged in ``responses`` like any
        other, so replays need no knowledge of the curriculum.
        """
        if not self.awaiting or not self._restricted() or self._point is not None or self._stepped:
            return None
        idx = auto_action(self.config.curriculum, self.decision, self.state.actions())
        if idx is None:
            return None
        response = self.state.step(idx)
        assert response is not None, "passive answers complete a decision in one step"
        self.result.responses.append(response)
        self.result.auto_decisions += 1
        self.state = None
        return response

    def use_response(self, response: bytes) -> None:
        """Answer the pending decision with recorded bytes (replay)."""
        self.result.responses.append(response)
        self.state = None
        self.events = []

    def finish(self) -> DuelResult:
        res, lp = self.result, self.lp
        engine_winner = self._engine_winner
        if res.reason in ("turn_limit", "decision_limit", "error"):
            # the turn limit is a rule (the higher LP wins); the decision limit only stops a loop: a draw, so that
            # looping while ahead never counts as a win (mirror of Tracker::winner)
            engine_winner = (None if lp[0] == lp[1] or res.reason != "turn_limit"
                             else (0 if lp[0] > lp[1] else 1))  # fmt: skip
        res.winner = None if engine_winner is None else deck_of_seat(self.first, engine_winner)
        res.turns = self.turn
        res.lp = (lp[seat_of_deck(self.first, 0)], lp[seat_of_deck(self.first, 1)])
        return res


def _step_record(point: DecisionPoint, chosen: int, probs) -> dict:
    def action(a: Action) -> dict:
        d = {
            "kind": a.kind,
            "index": a.index,
            "code": a.card.code if a.card else 0,
            "description": a.description,
            "value": a.value,
        }
        if a.card is not None:
            d["location"] = [a.card.loc.controller, a.card.loc.location, a.card.loc.sequence]
        return d

    rec = {"index": point.index, "player": point.player, "turn": point.turn, "decision": point.decision.name,
           "actions": [action(a) for a in point.actions], "chosen": chosen}  # fmt: skip
    if probs is not None:
        rec["probs"] = [float(p) for p in probs]
    return rec


__all__ = [
    "MAX_ENGINE_STEPS",
    "MAX_MENU_ACTIVATIONS",
    "MAX_SELECTION_STEPS",
    "MAX_SELECTION_CANCELS",
    "WIN_REASON_DECK_OUT",
    "WIN_REASON_LP",
    "DecisionPoint",
    "DuelResult",
    "DuelTracker",
    "deck_of_seat",
    "seat_of_deck",
]
