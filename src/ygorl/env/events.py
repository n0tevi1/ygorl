"""Event token stream with response-window / abstain tokens (T2.4); the spec is docs/encoding.md.

:class:`EventHistory` consumes every engine message of a duel (the omniscient
stream: feed each ``DecisionPoint.events`` in order) and keeps, for each
viewer, the tokens that viewer is allowed to see. ``encode(viewer)`` returns the
last ``length`` tokens as a fixed ``[length, E_EVENT]`` int32 array plus mask.

Visibility is decided when a token is created: card identities hidden from the
viewer become ``CardVocab.UNKNOWN``. Abstain tokens are derived from public
events only, never from the decision messages (whose existence and contents
depend on hidden cards), so "the opponent could have responded but passed" and
"the opponent could not respond" produce identical streams for the viewer.

The C++ port (csrc/event_encoder.cpp) must match this module element by
element, so the logic is kept plain.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ygorl.cards.cdb import CardVocab
from ygorl.engine import constants as C
from ygorl.engine import messages as M

E_EVENT = 20
DEFAULT_EVENT_LENGTH = 128
CLAMP = 65535

EVENT_TYPES = [
    "draw", "move", "pos_change", "set", "swap", "summoning", "spsummoning", "flipsummoning", "chaining",
    "chain_solving", "chain_negated", "chain_disabled", "chain_end", "new_turn", "new_phase", "damage", "recover",
    "pay_lpcost", "lp_update", "attack", "battle", "attack_disabled", "equip", "unequip", "card_target",
    "cancel_target", "become_target", "card_selected", "random_selected", "add_counter", "remove_counter",
    "confirm_cards", "confirm_decktop", "confirm_extratop", "deck_top", "shuffle_deck", "shuffle_hand",
    "shuffle_extra", "shuffle_set_card", "swap_grave_deck", "reverse_deck", "field_disabled", "toss_coin",
    "toss_dice", "hand_res", "abstain",
]  # fmt: skip
EV = {name: i + 1 for i, name in enumerate(EVENT_TYPES)}

TRIGGERS = {"search": 1, "spsummon_deck": 2, "send_deck_grave": 4, "fifth_summon": 8, "attack": 16, "other": 32}

# columns
(TYPE, PLAYER, CARD, CARD2, FROM_CONTROLLER, FROM_LOCATION, FROM_SEQUENCE, FROM_POSITION, TO_CONTROLLER, TO_LOCATION,
 TO_SEQUENCE, TO_POSITION, VALUE1, VALUE2, VALUE3, TURN, PHASE, MY_TURN, MY_LP, OP_LP) = range(E_EVENT)  # fmt: skip

LOCATION_ENUM = {C.LOCATION_DECK: 1, C.LOCATION_HAND: 2, C.LOCATION_MZONE: 3, C.LOCATION_SZONE: 4,
                 C.LOCATION_GRAVE: 5, C.LOCATION_REMOVED: 6, C.LOCATION_EXTRA: 7}  # fmt: skip
POSITION_ENUM = {C.POS_FACEUP_ATTACK: 1, C.POS_FACEDOWN_ATTACK: 2, C.POS_FACEUP_DEFENSE: 3, C.POS_FACEDOWN_DEFENSE: 4}
FIELD_SLOTS = {C.LOCATION_MZONE: 7, C.LOCATION_SZONE: 8}
CLOSING_DECISIONS = (M.SelectIdleCmd, M.SelectBattleCmd)


def _clamp(value: int, lo: int = 0, hi: int = CLAMP) -> int:
    return max(lo, min(hi, value))


def _bit_index(value: int) -> int:
    return (value & -value).bit_length() if value else 0


def _faceup(position: int) -> bool:
    return bool(position & C.POS_FACEUP)


def _position(position: int) -> int:
    if position in POSITION_ENUM:
        return POSITION_ENUM[position]
    if position & C.POS_FACEUP:
        return 5
    if position & C.POS_FACEDOWN:
        return 6
    return 0


def is_public(loc: M.Location) -> bool:
    """Whether a card at ``loc`` is known to both players (docs/encoding.md, 事件 token 流)."""
    location = loc.location
    if location & (C.LOCATION_OVERLAY | C.LOCATION_GRAVE):
        return True
    if location & (C.LOCATION_DECK | C.LOCATION_HAND):
        return False
    if location in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED, C.LOCATION_EXTRA):
        return _faceup(loc.position)
    return False


@dataclass
class _Link:
    player: int
    code: int
    loc: M.Location
    triggers: int = 0


@dataclass
class _Window:
    abstainer: int
    trigger: int
    card: tuple  # (code, seen by viewer 0, seen by viewer 1)
    loc: M.Location
    responded: bool = False


class EventHistory:
    """Per-viewer event token history of one duel (see the module docstring)."""

    def __init__(self, cards: Mapping, vocab: CardVocab, length: int = DEFAULT_EVENT_LENGTH,
                 starting_lp: int = 8000) -> None:  # fmt: skip
        if length < 0:
            raise ValueError("length must be >= 0")
        self.cards = cards
        self.vocab = vocab
        self.length = length
        self.tokens: tuple[deque, deque] = (deque(maxlen=length), deque(maxlen=length))
        self.turn = 0
        self.turn_player = 0
        self.phase = 0
        self.lp = [starting_lp, starting_lp]
        self.hand = [0, 0]
        self.field: dict[tuple[int, int, int], tuple[int, int]] = {}  # (con, location, seq) -> (code, position)
        self.summons = 0  # the turn player's summons this turn
        self.links: dict[int, _Link] = {}
        self.solving = 0
        self.windows: list[_Window] = []

    # -- public ------------------------------------------------------------
    def feed(self, messages: Iterable[M.Message]) -> None:
        for msg in messages:
            self._on(msg)

    def encode(self, viewer: int) -> dict[str, np.ndarray]:
        events = np.zeros((self.length, E_EVENT), dtype=np.int32)
        mask = np.zeros(self.length, dtype=np.int32)
        toks = self.tokens[viewer]
        if toks:
            events[: len(toks)] = np.asarray(toks, dtype=np.int64).astype(np.int32)
            mask[: len(toks)] = 1
        return {"events": events, "event_mask": mask}

    def hand_count(self, player: int) -> int:
        return self.hand[player]

    def field_count(self, player: int) -> int:
        return sum(1 for (con, _, _) in self.field if con == player)

    # -- token building ----------------------------------------------------
    @staticmethod
    def _card(code: int, owner: int | None = None, public: bool = True) -> tuple:
        """A card reference: (code, seen by viewer 0, seen by viewer 1); ``owner`` always sees it."""
        return (code, public or owner == 0, public or owner == 1)

    def _field_card(self, loc: M.Location) -> tuple | None:
        """The card at a monster / spell-trap zone from the field map (None when untracked)."""
        entry = self.field.get((loc.controller, loc.location, loc.sequence))
        if entry is None:
            return None
        code, pos = entry
        return self._card(code, loc.controller, _faceup(pos))

    def _emit(self, kind: str, player: int | None = None, card=None, card2=None, frm: M.Location | None = None,
              to: M.Location | None = None, v1: int = 0, v2: int = 0, v3: int = 0,
              only: int | None = None) -> None:  # fmt: skip
        for viewer in (0, 1) if only is None else (only,):
            row = [0] * E_EVENT
            row[TYPE] = EV[kind]
            row[PLAYER] = self._rel(player, viewer)
            row[CARD] = self._card_col(card, viewer)
            row[CARD2] = self._card_col(card2, viewer)
            row[FROM_CONTROLLER : FROM_POSITION + 1] = self._loc_cols(frm, viewer)
            row[TO_CONTROLLER : TO_POSITION + 1] = self._loc_cols(to, viewer)
            row[VALUE1], row[VALUE2], row[VALUE3] = v1, v2, v3
            row[TURN] = _clamp(self.turn, 0, 999)
            row[PHASE] = _bit_index(self.phase)
            row[MY_TURN] = int(self.turn_player == viewer)
            row[MY_LP] = _clamp(self.lp[viewer])
            row[OP_LP] = _clamp(self.lp[1 - viewer])
            self.tokens[viewer].append(row)

    @staticmethod
    def _rel(player: int | None, viewer: int) -> int:
        if player is None or player not in (0, 1):
            return 0
        return 1 if player == viewer else 2

    def _card_col(self, card, viewer: int) -> int:
        if card is None:
            return 0
        code, seen0, seen1 = card
        if not (seen0 if viewer == 0 else seen1):
            return CardVocab.UNKNOWN
        return self.vocab.index(code)

    def _loc_cols(self, loc: M.Location | None, viewer: int) -> list[int]:
        if loc is None:
            return [0, 0, 0, 0]
        overlay = bool(loc.location & C.LOCATION_OVERLAY)
        location = 8 if overlay else LOCATION_ENUM.get(loc.location, 0)
        if location == 0:
            return [0, 0, 0, 0]
        seq = 0 if location == 1 else _clamp(loc.sequence)
        return [self._rel(loc.controller, viewer), location, seq, 0 if overlay else _position(loc.position)]

    # -- field / hand bookkeeping -------------------------------------------
    @staticmethod
    def _slot(loc: M.Location) -> tuple[int, int, int] | None:
        n = FIELD_SLOTS.get(loc.location)
        if n is None or loc.controller not in (0, 1) or loc.sequence >= n:
            return None
        return (loc.controller, loc.location, loc.sequence)

    def _place(self, code: int, loc: M.Location) -> None:
        slot = self._slot(loc)
        if slot is not None:
            self.field[slot] = (code, loc.position)

    def _hand_delta(self, loc: M.Location, delta: int) -> None:
        if loc.location == C.LOCATION_HAND and loc.controller in (0, 1):
            self.hand[loc.controller] += delta

    # -- windows -------------------------------------------------------------
    def _close_windows(self) -> None:
        windows, self.windows = self.windows, []
        for w in windows:
            if not w.responded:
                self._abstain(w.abstainer, w.trigger, w.card, w.loc)

    def _abstain(self, abstainer: int, trigger: int, card, loc: M.Location) -> None:
        self._emit("abstain", abstainer, card, frm=loc, v1=trigger, v2=self.field_count(abstainer),
                   v3=self.hand[abstainer])  # fmt: skip

    # -- dispatch ------------------------------------------------------------
    def _on(self, msg: M.Message) -> None:  # noqa: C901 - one branch per message type
        if isinstance(msg, CLOSING_DECISIONS):
            self._close_windows()
            return
        t = msg.TYPE if not isinstance(msg, (M.UnknownMessage, M.UndecodableMessage)) else -1
        if t == C.MSG_DRAW:
            for code, pos in msg.cards:
                self._emit("draw", msg.player, self._card(code, msg.player, _faceup(pos)),
                           to=M.Location(msg.player, C.LOCATION_HAND, 0, 0), v1=len(msg.cards))  # fmt: skip
            if msg.player in (0, 1):
                self.hand[msg.player] += len(msg.cards)
        elif t == C.MSG_MOVE:
            prev, cur = msg.previous, msg.current
            owner = cur.controller if cur.location else prev.controller
            public = is_public(cur) or (prev.location != 0 and is_public(prev))
            card = self._card(msg.code, owner, public)
            self._emit("move", owner, card, frm=prev, to=cur, v1=msg.reason & 0x7FFFFFFF)
            slot = self._slot(prev)
            if slot is not None and self.field.get(slot, (None,))[0] in (msg.code, 0):  # 0: identity unknown
                del self.field[slot]
            self._place(msg.code, cur)
            self._hand_delta(prev, -1)
            self._hand_delta(cur, +1)
            if self.solving and self.solving in self.links and prev.location == C.LOCATION_DECK:
                trig = {C.LOCATION_HAND: "search", C.LOCATION_MZONE: "spsummon_deck", C.LOCATION_GRAVE: "send_deck_grave"}
                if cur.location in trig:
                    self.links[self.solving].triggers |= TRIGGERS[trig[cur.location]]
        elif t == C.MSG_POS_CHANGE:
            loc = msg.loc
            up = _faceup(msg.previous_position) or _faceup(msg.current_position)
            self._emit("pos_change", loc.controller, self._card(msg.code, loc.controller, up),
                       frm=M.Location(loc.controller, loc.location, loc.sequence, msg.previous_position),
                       to=M.Location(loc.controller, loc.location, loc.sequence, msg.current_position))  # fmt: skip
            self._place(msg.code, M.Location(loc.controller, loc.location, loc.sequence, msg.current_position))
        elif t == C.MSG_SET:
            con = msg.loc.controller
            self._emit("set", con, self._card(msg.code, con, False), to=msg.loc)
            self._place(msg.code, msg.loc)
        elif t == C.MSG_SWAP:
            a, b = msg.loc1, msg.loc2  # card 1 moves to b's place and card 2 to a's
            card1 = self._card(msg.code1, b.controller, is_public(a))
            card2 = self._card(msg.code2, a.controller, is_public(b))
            self._emit("swap", None, card1, card2, frm=a, to=b)
            self._place(msg.code1, M.Location(b.controller, b.location, b.sequence, a.position))
            self._place(msg.code2, M.Location(a.controller, a.location, a.sequence, b.position))
        elif t in (C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING):
            loc, con = msg.loc, msg.loc.controller
            card = self._card(msg.code, con, is_public(loc))
            kind = {
                C.MSG_SUMMONING: "summoning",
                C.MSG_SPSUMMONING: "spsummoning",
                C.MSG_FLIPSUMMONING: "flipsummoning",
            }[t]
            self._emit(kind, con, card, to=loc)
            self._place(msg.code, loc)
            if t != C.MSG_FLIPSUMMONING and con == self.turn_player and con in (0, 1):
                self.summons += 1
                if self.summons == 5:
                    self.windows.append(_Window(1 - self.turn_player, TRIGGERS["fifth_summon"], card, loc))
        elif t == C.MSG_CHAINING:
            desc = msg.description
            card2, v2, v3 = None, 0, 0
            if desc:
                if (desc >> 20) in self.cards:
                    card2, v2 = self._card(desc >> 20), _clamp((desc & 0xFFFFF) + 1)
                else:
                    v3 = _clamp(desc)
            player = msg.triggering_controller
            self._emit(
                "chaining", player, self._card(msg.code), card2, frm=msg.loc, v1=_clamp(msg.chain_count), v2=v2, v3=v3
            )
            for n in [n for n in self.links if n >= msg.chain_count]:
                del self.links[n]
            self.links[msg.chain_count] = _Link(player, msg.code, msg.loc)
            for w in self.windows:
                if w.abstainer == player:
                    w.responded = True
        elif t in (C.MSG_CHAIN_SOLVING, C.MSG_CHAIN_NEGATED, C.MSG_CHAIN_DISABLED):
            link = self.links.get(msg.chain_count)
            kind = {C.MSG_CHAIN_SOLVING: "chain_solving", C.MSG_CHAIN_NEGATED: "chain_negated",
                    C.MSG_CHAIN_DISABLED: "chain_disabled"}[t]  # fmt: skip
            self._emit(kind, link.player if link else None, self._card(link.code) if link else None, v1=msg.chain_count)
            if t == C.MSG_CHAIN_SOLVING:
                self.solving = msg.chain_count
        elif t == C.MSG_CHAIN_SOLVED:
            n = msg.chain_count
            link = self.links.get(n)
            if link is not None and link.player in (0, 1):
                abstainer = 1 - link.player
                if not any(k > n and self.links[k].player == abstainer for k in self.links):
                    self._abstain(abstainer, link.triggers or TRIGGERS["other"], self._card(link.code), link.loc)
            self.solving = 0
        elif t == C.MSG_CHAIN_END:
            self.links.clear()
            self.solving = 0
            self._emit("chain_end")
        elif t == C.MSG_NEW_TURN:
            self._close_windows()
            self.turn += 1
            self.turn_player = msg.player
            self.summons = 0
            self._emit("new_turn", msg.player)
        elif t == C.MSG_NEW_PHASE:
            self._close_windows()
            self.phase = msg.phase
            self._emit("new_phase", v1=_bit_index(msg.phase))
        elif t in (C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_PAY_LPCOST, C.MSG_LPUPDATE):
            if msg.player in (0, 1):
                if t == C.MSG_RECOVER:
                    self.lp[msg.player] += msg.amount
                elif t == C.MSG_LPUPDATE:
                    self.lp[msg.player] = msg.amount
                else:
                    self.lp[msg.player] -= msg.amount
            kind = {C.MSG_DAMAGE: "damage", C.MSG_RECOVER: "recover", C.MSG_PAY_LPCOST: "pay_lpcost",
                    C.MSG_LPUPDATE: "lp_update"}[t]  # fmt: skip
            self._emit(kind, msg.player, v1=_clamp(msg.amount))
        elif t == C.MSG_ATTACK:
            attacker, target = msg.card, msg.target
            card = self._field_card(attacker)
            direct = target.location == 0
            self._emit("attack", attacker.controller, card, None if direct else self._field_card(target), frm=attacker,
                       to=None if direct else target, v1=int(direct))  # fmt: skip
            if attacker.controller in (0, 1):
                self.windows.append(_Window(1 - attacker.controller, TRIGGERS["attack"], card, attacker))
        elif t == C.MSG_BATTLE:
            self._emit("battle", msg.attacker.controller, self._field_card(msg.attacker), self._field_card(msg.target),
                       frm=msg.attacker, to=msg.target, v1=_clamp(msg.attacker_atk), v2=_clamp(msg.target_atk),
                       v3=_clamp(msg.target_def))  # fmt: skip
        elif t == C.MSG_ATTACK_DISABLED:
            self._emit("attack_disabled")
        elif t in (C.MSG_EQUIP, C.MSG_CARD_TARGET, C.MSG_CANCEL_TARGET):
            kind = {C.MSG_EQUIP: "equip", C.MSG_CARD_TARGET: "card_target", C.MSG_CANCEL_TARGET: "cancel_target"}[t]
            card, target = self._field_card(msg.card), self._field_card(msg.target)
            self._emit(kind, msg.card.controller, card, target, frm=msg.card, to=msg.target)
        elif t == C.MSG_UNEQUIP:
            self._emit("unequip", msg.card.controller, self._field_card(msg.card), frm=msg.card)
        elif t in (C.MSG_BECOME_TARGET, C.MSG_CARD_SELECTED):
            kind = "become_target" if t == C.MSG_BECOME_TARGET else "card_selected"
            for loc in msg.locations:
                self._emit(kind, loc.controller, self._field_card(loc), frm=loc)
        elif t == C.MSG_RANDOM_SELECTED:
            for loc in msg.locations:
                self._emit("random_selected", msg.player, self._field_card(loc), frm=loc)
        elif t in (C.MSG_ADD_COUNTER, C.MSG_REMOVE_COUNTER):
            kind = "add_counter" if t == C.MSG_ADD_COUNTER else "remove_counter"
            self._emit(
                kind, msg.loc.controller, self._field_card(msg.loc), frm=msg.loc, v1=msg.counter_type, v2=msg.count
            )
        elif t in (C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP):
            kind = {C.MSG_CONFIRM_CARDS: "confirm_cards", C.MSG_CONFIRM_DECKTOP: "confirm_decktop",
                    C.MSG_CONFIRM_EXTRATOP: "confirm_extratop"}[t]  # fmt: skip
            for c in msg.cards:
                if t == C.MSG_CONFIRM_CARDS:
                    shown = c.loc.location != C.LOCATION_DECK
                    card = self._card(c.code, msg.player, shown)
                else:
                    card = self._card(c.code)
                self._emit(kind, msg.player, card, frm=c.loc)
        elif t == C.MSG_DECK_TOP:
            self._emit("deck_top", msg.player, self._card(msg.code, None, _faceup(msg.position)))
        elif t == C.MSG_SHUFFLE_DECK:
            self._emit("shuffle_deck", msg.player)
        elif t in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_EXTRA):
            self._emit("shuffle_hand" if t == C.MSG_SHUFFLE_HAND else "shuffle_extra", msg.player, v1=len(msg.codes))
        elif t == C.MSG_SHUFFLE_SET_CARD:
            self._emit("shuffle_set_card", frm=M.Location(2, msg.location, 0, 0), v1=len(msg.new_locations))
            for loc in msg.new_locations:  # the cards' zones before the shuffle: still occupied, identity now unknown
                slot = self._slot(loc)
                if slot in self.field:
                    self.field[slot] = (0, self.field[slot][1])
        elif t == C.MSG_SWAP_GRAVE_DECK:
            self._emit("swap_grave_deck", msg.player)
        elif t == C.MSG_REVERSE_DECK:
            self._emit("reverse_deck")
        elif t == C.MSG_FIELD_DISABLED:
            halves = (msg.flag & 0xFFFF, (msg.flag >> 16) & 0xFFFF)
            for viewer in (0, 1):  # the values are relative to the viewer
                self._emit("field_disabled", v1=halves[viewer], v2=halves[1 - viewer], only=viewer)
        elif t == C.MSG_TOSS_COIN:
            bits = sum((r & 1) << i for i, r in enumerate(msg.results[:30]))
            self._emit("toss_coin", msg.player, v1=len(msg.results), v2=bits)
        elif t == C.MSG_TOSS_DICE:
            packed = sum((r & 7) << (3 * i) for i, r in enumerate(msg.results[:10]))
            self._emit("toss_dice", msg.player, v1=len(msg.results), v2=packed)
        elif t == C.MSG_HAND_RES:
            hands = (msg.hand0, msg.hand1)
            for viewer in (0, 1):
                self._emit("hand_res", v1=hands[viewer], v2=hands[1 - viewer], only=viewer)


__all__ = ["DEFAULT_EVENT_LENGTH", "EVENT_TYPES", "E_EVENT", "TRIGGERS", "EventHistory", "is_public"]
