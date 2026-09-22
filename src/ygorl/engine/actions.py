"""Legal actions for each decision message and encoding of the core response.

One environment step = one action (design §5.2). Decisions whose answer is a
set or a sequence (SELECT_CARD / TRIBUTE / SUM / COUNTER / PLACE / SORT /
ANNOUNCE_RACE|ATTRIB) are split into several steps: each step picks one element
or ``finish``. Every step offers only actions that can still be completed into a
response the core accepts, so an agent can never produce ``MSG_RETRY`` (this is
the Python reference for the C++ feasible-set code of T2.3).

Usage::

    state = make_decision(msg, cards=card_db)
    while not state.done:
        state.step(agent.choose(state.actions()))
    duel.set_response(state.response)
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from ygorl.engine import constants as C
from ygorl.engine import messages as M

CARD_MARINE_DOLPHIN = 78734254
CARD_TWINKLE_MOSS = 13857930


@dataclass(frozen=True, slots=True)
class Action:
    """One legal choice.

    ``kind`` names the choice (e.g. ``summon``, ``activate``, ``select``,
    ``finish``); ``index`` is the position in the decision's own list;
    ``card`` / ``description`` / ``value`` carry what the choice refers to.
    """

    kind: str
    index: int = -1
    card: M.CardInfo | None = None
    description: int = 0
    value: int = 0


def _i32(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def _cards_response(indices: Sequence[int]) -> bytes:
    """An explicit selection (possibly empty); ``-1`` (cancel) is encoded separately.

    The core treats them differently: for a cancelable script ``-1`` returns nil
    (and cancels a summon's tributes), an empty list returns an empty group.
    """
    return struct.pack(f"<iI{len(indices)}I", 0, len(indices), *indices)


class DecisionState:
    """Step-wise builder of the response to one decision message."""

    def __init__(self, decision: M.Decision) -> None:
        self.decision = decision
        self.player = decision.player
        self.response: bytes | None = None
        self._actions: list[Action] | None = None

    @property
    def done(self) -> bool:
        return self.response is not None

    def actions(self) -> list[Action]:
        if self.done:
            return []
        if self._actions is None:
            self._actions = self._legal()
        return self._actions

    def step(self, index: int) -> bytes | None:
        """Apply action ``index``; return the response bytes once complete."""
        if self.done:
            raise RuntimeError("decision already complete")
        acts = self.actions()
        if not 0 <= index < len(acts):
            raise IndexError(f"action {index} out of range 0..{len(acts) - 1}")
        self._actions = None
        self.response = self._apply(acts[index])
        return self.response

    # subclasses implement:
    def _legal(self) -> list[Action]:
        raise NotImplementedError

    def _apply(self, action: Action) -> bytes | None:
        raise NotImplementedError


# ------------------------------------------------------------------ single step


class _Single(DecisionState):
    """Decisions answered in one step; ``_encode`` maps an action to bytes."""

    def _apply(self, action: Action) -> bytes:
        return self._encode(action)

    def _encode(self, action: Action) -> bytes:
        return _i32(action.value)


class IdleCmdState(_Single):
    _CODES = {"summon": 0, "spsummon": 1, "reposition": 2, "mset": 3, "sset": 4, "activate": 5,
              "battle_phase": 6, "end_phase": 7, "shuffle": 8}  # fmt: skip

    def _legal(self) -> list[Action]:
        d: M.SelectIdleCmd = self.decision
        acts: list[Action] = []
        for kind, cards in (("summon", d.summonable), ("spsummon", d.spsummonable), ("reposition", d.repositionable),
                            ("mset", d.msetable), ("sset", d.ssetable)):  # fmt: skip
            acts.extend(Action(kind, i, c) for i, c in enumerate(cards))
        acts.extend(Action("activate", i, M.CardInfo(o.code, o.loc), o.description) for i, o in enumerate(d.activatable))
        for kind, ok in (("battle_phase", d.can_battle_phase), ("end_phase", d.can_end_phase), ("shuffle", d.can_shuffle)):
            if ok:
                acts.append(Action(kind))
        return acts

    def _encode(self, action: Action) -> bytes:
        return _i32(self._CODES[action.kind] | (max(action.index, 0) << 16))


class BattleCmdState(_Single):
    _CODES = {"activate": 0, "attack": 1, "main2": 2, "end_phase": 3}

    def _legal(self) -> list[Action]:
        d: M.SelectBattleCmd = self.decision
        acts = [Action("activate", i, M.CardInfo(o.code, o.loc), o.description) for i, o in enumerate(d.activatable)]
        acts += [Action("attack", i, M.CardInfo(o.code, o.loc), value=int(o.direct_attackable)) for i, o in enumerate(d.attackable)]
        if d.can_main2:
            acts.append(Action("main2"))
        if d.can_end_phase:
            acts.append(Action("end_phase"))
        return acts

    def _encode(self, action: Action) -> bytes:
        return _i32(self._CODES[action.kind] | (max(action.index, 0) << 16))


class YesNoState(_Single):
    def _legal(self) -> list[Action]:
        d = self.decision
        card = d.card if isinstance(d, M.SelectEffectYn) else None
        return [Action("yes", card=card, description=d.description, value=1), Action("no", card=card, description=d.description, value=0)]


class OptionState(_Single):
    def _legal(self) -> list[Action]:
        return [Action("option", i, description=desc, value=i) for i, desc in enumerate(self.decision.options)]


class NumberState(_Single):
    def _legal(self) -> list[Action]:
        return [Action("number", i, value=v) for i, v in enumerate(self.decision.options)]

    def _encode(self, action: Action) -> bytes:
        return _i32(action.index)


class RpsState(_Single):
    def _legal(self) -> list[Action]:
        return [Action("rps", i, value=i + 1) for i in range(3)]


class ChainState(_Single):
    def _legal(self) -> list[Action]:
        d: M.SelectChain = self.decision
        acts = [Action("chain", i, M.CardInfo(o.code, o.loc), o.description, value=i) for i, o in enumerate(d.chains)]
        if not d.forced:
            acts.append(Action("pass", value=-1))
        return acts


class PositionState(_Single):
    def _legal(self) -> list[Action]:
        d: M.SelectPosition = self.decision
        return [Action("position", value=p, card=M.CardInfo(d.code, M.Location(d.player, 0, 0))) for p in
                (C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_ATTACK, C.POS_FACEUP_DEFENSE, C.POS_FACEDOWN_DEFENSE) if d.positions & p]  # fmt: skip


class UnselectCardState(_Single):
    def _legal(self) -> list[Action]:
        d: M.SelectUnselectCard = self.decision
        acts = [Action("select", i, c, value=i) for i, c in enumerate(d.selectable)]
        n = len(d.selectable)
        acts += [Action("unselect", i, c, value=n + i) for i, c in enumerate(d.unselectable)]
        if d.finishable:
            acts.append(Action("finish", value=-1))
        elif d.cancelable:
            acts.append(Action("cancel", value=-1))
        return acts

    def _encode(self, action: Action) -> bytes:
        return _i32(-1) if action.value == -1 else _i32(1, action.value)


# ------------------------------------------------------------------ announce card


def is_declarable(card, opcodes: Sequence[int]) -> bool:
    """Evaluate an ``ANNOUNCE_CARD`` RPN filter like the core's ``is_declarable``.

    ``card`` needs ``password, alias, type, race, attribute, setcodes``.
    """
    stack: list[int] = []
    alias = token = False

    def binary(fn):
        if len(stack) >= 2:
            rhs, lhs = stack.pop(), stack.pop()
            stack.append(int(fn(lhs, rhs)))

    def unary(fn):
        if stack:
            stack.append(int(fn(stack.pop())))

    for op in opcodes:
        if op == C.OPCODE_ADD:
            binary(lambda a, b: a + b)
        elif op == C.OPCODE_SUB:
            binary(lambda a, b: a - b)
        elif op == C.OPCODE_MUL:
            binary(lambda a, b: a * b)
        elif op == C.OPCODE_DIV:
            binary(lambda a, b: int(a / b) if b else 0)
        elif op == C.OPCODE_AND:
            binary(lambda a, b: bool(a) and bool(b))
        elif op == C.OPCODE_OR:
            binary(lambda a, b: bool(a) or bool(b))
        elif op == C.OPCODE_NEG:
            unary(lambda a: -a)
        elif op == C.OPCODE_NOT:
            unary(lambda a: not a)
        elif op == C.OPCODE_BAND:
            binary(lambda a, b: a & b)
        elif op == C.OPCODE_BOR:
            binary(lambda a, b: a | b)
        elif op == C.OPCODE_BXOR:
            binary(lambda a, b: a ^ b)
        elif op == C.OPCODE_BNOT:
            unary(lambda a: ~a)
        elif op == C.OPCODE_LSHIFT:
            binary(lambda a, b: a << b if 0 <= b < 64 else 0)
        elif op == C.OPCODE_RSHIFT:
            binary(lambda a, b: a >> b if 0 <= b < 64 else 0)
        elif op == C.OPCODE_ISCODE:
            unary(lambda a: card.password == (a & 0xFFFFFFFF))
        elif op == C.OPCODE_ISTYPE:
            unary(lambda a: card.type & a)
        elif op == C.OPCODE_ISRACE:
            unary(lambda a: card.race & a)
        elif op == C.OPCODE_ISATTRIBUTE:
            unary(lambda a: card.attribute & a)
        elif op == C.OPCODE_GETCODE:
            stack.append(card.password)
        elif op == C.OPCODE_GETTYPE:
            stack.append(card.type)
        elif op == C.OPCODE_GETRACE:
            stack.append(card.race)
        elif op == C.OPCODE_GETATTRIBUTE:
            stack.append(card.attribute)
        elif op == C.OPCODE_ISSETCARD:
            if stack:
                set_code = stack.pop() & 0xFFFFFFFF
                settype, subtype = set_code & 0xFFF, set_code & 0xF000
                stack.append(int(any((sc & 0xFFF) == settype and (sc & 0xF000 & subtype) == subtype for sc in card.setcodes)))
        elif op == C.OPCODE_ALLOW_ALIASES:
            alias = True
        elif op == C.OPCODE_ALLOW_TOKENS:
            token = True
        else:
            stack.append(op)
    if len(stack) != 1 or stack[0] == 0:
        return False
    if card.password in (CARD_MARINE_DOLPHIN, CARD_TWINKLE_MOSS):
        return True
    monster_token = C.TYPE_MONSTER | C.TYPE_TOKEN
    return (alias or not card.alias) and (token or (card.type & monster_token) != monster_token)


class AnnounceCardState(_Single):
    def __init__(self, decision: M.AnnounceCard, cards: Mapping[int, object] | None) -> None:
        super().__init__(decision)
        if cards is None:
            raise ValueError("ANNOUNCE_CARD needs the card database (cards=...) to enumerate declarable cards")
        self._cards = cards

    def _legal(self) -> list[Action]:
        ops = self.decision.opcodes
        return [Action("declare", value=p) for p in sorted(self._cards) if is_declarable(self._cards[p], ops)]


# ------------------------------------------------------------------- multi-step


class _Picks(DecisionState):
    """Base for decisions built from a sequence of picks."""

    def __init__(self, decision: M.Decision) -> None:
        super().__init__(decision)
        self.picked: list[int] = []


class SelectCardState(_Picks):
    def _legal(self) -> list[Action]:
        d: M.SelectCard = self.decision
        n = len(self.picked)
        acts = [Action("select", i, c) for i, c in enumerate(d.cards) if i not in self.picked] if n < d.max else []
        if n >= d.min:  # with min 0 an empty selection is a valid answer of its own
            acts.append(Action("finish"))
        if d.cancelable and n == 0:
            acts.append(Action("cancel"))
        return acts

    def _apply(self, action: Action) -> bytes | None:
        d: M.SelectCard = self.decision
        if action.kind == "cancel":
            return _i32(-1)
        if action.kind == "select":
            self.picked.append(action.index)
            if len(self.picked) < d.max and len(self.picked) < len(d.cards):
                return None
        return _cards_response(self.picked)


class TributeState(_Picks):
    def _sum(self, idx) -> int:
        return sum(self.decision.cards[i].release_param for i in idx)

    def _can_reach(self, chosen: list[int]) -> bool:
        d: M.SelectTribute = self.decision
        slots = d.max - len(chosen)
        rest = sorted((c.release_param for i, c in enumerate(d.cards) if i not in chosen), reverse=True)
        return self._sum(chosen) + sum(rest[:max(slots, 0)]) >= d.min

    def _legal(self) -> list[Action]:
        d: M.SelectTribute = self.decision
        acts = []
        if len(self.picked) < d.max:
            for i, c in enumerate(d.cards):
                if i not in self.picked and self._can_reach(self.picked + [i]):
                    acts.append(Action("select", i, M.CardInfo(c.code, c.loc), value=c.release_param))
        if self._sum(self.picked) >= d.min:
            acts.append(Action("finish"))
        if d.cancelable and not self.picked:
            acts.append(Action("cancel"))
        return acts

    def _apply(self, action: Action) -> bytes | None:
        if action.kind == "cancel":
            return _i32(-1)
        if action.kind == "select":
            self.picked.append(action.index)
            if not self.actions():  # nothing else possible (cannot happen for valid input)
                return _cards_response(self.picked)
            if len(self.picked) >= self.decision.max:
                return _cards_response(self.picked)
            return None
        return _cards_response(self.picked)


class SelectSumState(_Picks):
    """SELECT_SUM with dead-end pruning (exact and at-least modes)."""

    def _values(self, opt: M.SumOption) -> tuple[int, ...]:
        return opt.values

    # exact mode ---------------------------------------------------------
    def _sums(self, opts) -> set[int]:
        target = self.decision.target
        sums = {0}
        for o in opts:
            sums = {s + v for s in sums for v in self._values(o) if s + v <= target}
        return sums

    def _exact_feasible(self, chosen: list[int]) -> bool:
        d: M.SelectSum = self.decision
        base = self._sums(list(d.must) + [d.cards[i] for i in chosen])
        max_count = min(max(d.max, d.min), len(d.cards))  # never more picks than offered cards
        if len(chosen) > max_count:
            return False
        # dp[k] = sums reachable with k additional cards from the rest
        dp: list[set[int]] = [base] + [set() for _ in range(max_count - len(chosen))]
        for i, o in enumerate(d.cards):
            if i in chosen:
                continue
            for k in range(len(dp) - 1, 0, -1):
                dp[k] |= {s + v for s in dp[k - 1] for v in self._values(o) if s + v <= d.target}
        return any(d.target in dp[k] for k in range(len(dp)) if len(chosen) + k >= d.min)

    def _exact_complete(self, chosen: list[int]) -> bool:
        d: M.SelectSum = self.decision
        return d.min <= len(chosen) <= max(d.max, d.min) and d.target in self._sums(list(d.must) + [d.cards[i] for i in chosen])

    # at-least mode ------------------------------------------------------
    def _lo_hi(self, o: M.SumOption) -> tuple[int, int]:
        lo, hi = o.param & 0xFFFF, o.param >> 16
        ms = hi if hi and hi < lo else lo
        return ms, max(lo, hi)

    def _greater_valid(self, chosen: Sequence[int]) -> bool:
        d: M.SelectSum = self.decision
        opts = list(d.must) + [d.cards[i] for i in chosen]
        if not opts:
            return False
        pairs = [self._lo_hi(o) for o in opts]
        total_ms = sum(p[0] for p in pairs)
        return sum(p[1] for p in pairs) >= d.target and total_ms - min(p[0] for p in pairs) < d.target

    def _greater_dead(self, chosen: Sequence[int]) -> bool:
        """Adding cards can never fix a set whose sum minus its smallest reaches the target."""
        d: M.SelectSum = self.decision
        pairs = [self._lo_hi(o) for o in list(d.must) + [d.cards[i] for i in chosen]]
        return bool(pairs) and sum(p[0] for p in pairs) - min(p[0] for p in pairs) >= d.target

    def _greater_feasible(self, chosen: tuple[int, ...]) -> bool:
        @lru_cache(maxsize=4096)
        def search(state: tuple[int, ...]) -> bool:
            if self._greater_dead(state):
                return False
            if self._greater_valid(state):
                return True
            return any(search(tuple(sorted(state + (i,)))) for i in range(len(self.decision.cards)) if i not in state)

        return search(tuple(sorted(chosen)))

    # common -------------------------------------------------------------
    def _feasible(self, chosen: list[int]) -> bool:
        return self._exact_feasible(chosen) if self.decision.exact else self._greater_feasible(tuple(chosen))

    def _complete(self, chosen: list[int]) -> bool:
        return self._exact_complete(chosen) if self.decision.exact else self._greater_valid(chosen)

    def _legal(self) -> list[Action]:
        d: M.SelectSum = self.decision
        acts = [Action("select", i, M.CardInfo(o.code, o.loc), value=o.param)
                for i, o in enumerate(d.cards) if i not in self.picked and self._feasible(self.picked + [i])]  # fmt: skip
        if self._complete(self.picked):  # may be empty: the must-select cards alone can complete it
            acts.append(Action("finish"))
        return acts

    def _apply(self, action: Action) -> bytes | None:
        if action.kind == "select":
            self.picked.append(action.index)
            if self._complete(self.picked) and not any(a.kind == "select" for a in self.actions()):
                return _cards_response(self.picked)
            return None
        return _cards_response(self.picked)


class CounterState(_Picks):
    def _remaining(self, i: int) -> int:
        return self.decision.cards[i].counters - self.picked.count(i)

    def _legal(self) -> list[Action]:
        d: M.SelectCounter = self.decision
        return [Action("counter", i, M.CardInfo(c.code, c.loc), value=self._remaining(i)) for i, c in enumerate(d.cards) if self._remaining(i) > 0]

    def _apply(self, action: Action) -> bytes | None:
        d: M.SelectCounter = self.decision
        self.picked.append(action.index)
        if len(self.picked) < d.count:
            return None
        return struct.pack(f"<{len(d.cards)}h", *(self.picked.count(i) for i in range(len(d.cards))))


class SortState(_Picks):
    def _legal(self) -> list[Action]:
        d: M.SortCard = self.decision
        acts = [Action("sort", i, c) for i, c in enumerate(d.cards) if i not in self.picked]
        if not self.picked:
            acts.append(Action("default"))
        return acts

    def _apply(self, action: Action) -> bytes | None:
        d: M.SortCard = self.decision
        if action.kind == "default":
            return struct.pack("<b", -1)
        self.picked.append(action.index)
        if len(self.picked) < len(d.cards) - 1:
            return None
        self.picked += [i for i in range(len(d.cards)) if i not in self.picked]
        positions = [self.picked.index(i) for i in range(len(d.cards))]
        return struct.pack(f"<{len(positions)}b", *positions)


class PlaceState(_Picks):
    """SELECT_PLACE / SELECT_DISFIELD. Action ``value`` = ``player<<16 | location<<8 | sequence``."""

    def _zones(self) -> list[int]:
        d: M.SelectPlace = self.decision
        zones = []
        for rel, player in ((0, d.player), (16, 1 - d.player)):
            for seq in range(7):
                if not d.flag >> (rel + seq) & 1:
                    zones.append(player << 16 | C.LOCATION_MZONE << 8 | seq)
            for seq in range(8):
                if not d.flag >> (rel + 8 + seq) & 1:
                    zones.append(player << 16 | C.LOCATION_SZONE << 8 | seq)
        return zones

    def _legal(self) -> list[Action]:
        return [Action("place", value=z) for z in self._zones() if z not in self.picked]

    def _apply(self, action: Action) -> bytes | None:
        self.picked.append(action.value)
        if len(self.picked) < self.decision.count:
            return None
        return bytes(b for z in self.picked for b in (z >> 16, (z >> 8) & 0xFF, z & 0xFF))


class _AnnounceBits(_Picks):
    _fmt = "<Q"
    _kind = "race"

    def _legal(self) -> list[Action]:
        d = self.decision
        bits = [1 << i for i in range(64) if d.available >> i & 1]
        return [Action(self._kind, value=b) for b in bits if b not in self.picked]

    def _apply(self, action: Action) -> bytes | None:
        self.picked.append(action.value)
        if len(self.picked) < self.decision.count:
            return None
        mask = 0
        for b in self.picked:
            mask |= b
        return struct.pack(self._fmt, mask)


class AnnounceRaceState(_AnnounceBits):
    pass


class AnnounceAttribState(_AnnounceBits):
    _fmt = "<I"
    _kind = "attribute"


# ------------------------------------------------------------------- factory

_STATES: dict[type, type[DecisionState]] = {
    M.SelectIdleCmd: IdleCmdState,
    M.SelectBattleCmd: BattleCmdState,
    M.SelectEffectYn: YesNoState,
    M.SelectYesNo: YesNoState,
    M.SelectOption: OptionState,
    M.SelectCard: SelectCardState,
    M.SelectUnselectCard: UnselectCardState,
    M.SelectChain: ChainState,
    M.SelectPlace: PlaceState,
    M.SelectDisfield: PlaceState,
    M.SelectPosition: PositionState,
    M.SelectTribute: TributeState,
    M.SelectCounter: CounterState,
    M.SelectSum: SelectSumState,
    M.SortCard: SortState,
    M.SortChain: SortState,
    M.AnnounceRace: AnnounceRaceState,
    M.AnnounceAttrib: AnnounceAttribState,
    M.AnnounceNumber: NumberState,
    M.RockPaperScissors: RpsState,
}
SUPPORTED_DECISIONS: tuple[type, ...] = (*_STATES, M.AnnounceCard)


def make_decision(decision: M.Decision, cards: Mapping[int, object] | None = None) -> DecisionState:
    """Build the step-wise action state for ``decision``.

    ``cards`` (password -> card with ``alias/type/race/attribute/setcodes``) is
    required only for ``MSG_ANNOUNCE_CARD``.
    """
    if isinstance(decision, M.AnnounceCard):
        return AnnounceCardState(decision, cards)
    try:
        cls = _STATES[type(decision)]
    except KeyError:
        raise TypeError(f"no action model for {type(decision).__name__}") from None
    return cls(decision)
