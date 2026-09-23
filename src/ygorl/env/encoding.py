"""Python reference observation encoder (T2.2); the specification is docs/encoding.md.

The C++ encoder must match this implementation element by element, so the
logic here is deliberately plain: fixed query flags, a fixed row order, and
integer features only (normalisation belongs to the network).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ygorl.cards.cdb import CardVocab
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import DecisionPoint
from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location
from ygorl.env.privileged import encode_privileged

N_CARDS = 160
F_CARD = 23
G_GLOBAL = 22
MAX_OPTIONS = 128
A_ACTION = 10
CLAMP = 65535

ACTION_KINDS = [
    "summon", "spsummon", "reposition", "mset", "sset", "activate", "battle_phase", "end_phase", "shuffle",
    "attack", "main2", "yes", "no", "option", "number", "rps", "chain", "pass", "position", "select",
    "unselect", "finish", "cancel", "counter", "sort", "default", "place", "race", "attribute", "declare",
]  # fmt: skip

LOCATION_ENUM = {C.LOCATION_DECK: 1, C.LOCATION_HAND: 2, C.LOCATION_MZONE: 3, C.LOCATION_SZONE: 4,
                 C.LOCATION_GRAVE: 5, C.LOCATION_REMOVED: 6, C.LOCATION_EXTRA: 7, C.LOCATION_OVERLAY: 8}  # fmt: skip
POSITION_ENUM = {C.POS_FACEUP_ATTACK: 1, C.POS_FACEDOWN_ATTACK: 2, C.POS_FACEUP_DEFENSE: 3, C.POS_FACEDOWN_DEFENSE: 4}
SIDE_LOCATIONS = (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED,
                  C.LOCATION_EXTRA)  # fmt: skip
COUNT_LOCATIONS = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA)

# card-table columns
(CARD_INDEX, LOCATION, SEQUENCE, OVERLAY_INDEX, CONTROLLER, OWNER, POSITION, VISIBLE, PUBLIC, TYPE, ATTRIBUTE, RACE,
 LEVEL, RANK, LINK, LSCALE, RSCALE, ATTACK, DEFENSE, LINK_MARKER, COUNTERS, MATERIALS, DISABLED) = range(F_CARD)  # fmt: skip


def _bit_index(value: int) -> int:
    """1-based index of the lowest set bit, 0 for no bits."""
    return (value & -value).bit_length() if value else 0


def _clamp(value: int, lo: int = 0, hi: int = CLAMP) -> int:
    return max(lo, min(hi, value))


# Zones where the sequence of a card says nothing a choice between copies could use (LOCATION_ENUM values: deck,
# hand, Extra Deck). Monster / spell-trap zone sequences are columns and link arrows; GY / banished order is age
# (newest last), which tells apart per-card state such as "sent to the GY this turn".
_UNORDERED = (1, 2, 7)


def _equivalence_key(cards: np.ndarray, action: np.ndarray, decision: int) -> bytes | None:
    """What makes an action row distinct to the decider, or None for a row that is never merged (docs/encoding.md)."""
    card_row, card_index = int(action[1]), int(action[2])
    if card_row == 0 or card_index == 0:  # no card-table row, or a card hidden from the decider
        return None
    a = action.copy()
    a[1] = a[9] = 0
    if decision == C.MSG_SELECT_UNSELECT_CARD:
        a[8] = 0  # the list index there
    c = cards[card_row - 1].copy()
    if not c[VISIBLE]:  # never let the mask say that two hidden cards are the same
        return None
    c[OVERLAY_INDEX] = 0
    if c[LOCATION] in _UNORDERED:
        c[SEQUENCE] = 0
    return a.tobytes() + c.tobytes()


def action_representatives(cards: np.ndarray, actions: np.ndarray, n: int, decision: int) -> np.ndarray:
    """``rep[i]`` = the first of the first ``n`` action rows that is equivalent to row ``i`` (``i`` if none is)."""
    rep = np.arange(n)
    first: dict[bytes, int] = {}
    for i in range(n):
        key = _equivalence_key(cards, actions[i], decision)
        if key is not None:
            rep[i] = first.setdefault(key, i)
    return rep


def mask_duplicates(cards: np.ndarray, actions: np.ndarray, mask: np.ndarray, decision: int) -> None:
    """Keep only the first row of each class of equivalent actions in ``mask`` (in place; docs/encoding.md)."""
    n = int(np.count_nonzero(mask))  # the legal rows are a prefix before deduplication
    rep = action_representatives(cards, actions, n, decision)
    mask[:n] = rep == np.arange(n)


def canonical_action(obs: Mapping[str, np.ndarray], index: int) -> int:
    """The row the policy can choose for action ``index`` of an encoded observation (its class representative)."""
    n = min(int(obs["globals"][20]), MAX_OPTIONS)
    if not 0 <= index < n:
        raise IndexError(f"action {index} out of range for {n} encoded actions")
    return int(action_representatives(obs["cards"], obs["actions"], n, int(obs["globals"][18]))[index])


class ObservationEncoder:
    """Encode a :class:`DecisionPoint` for its deciding player (see docs/encoding.md).

    ``privileged=True`` (training mode) additionally enables :meth:`encode_privileged`, the
    opponent ground truth for critic / belief losses (T2.5); it is never part of :meth:`encode`.
    """

    def __init__(self, cards: Mapping, vocab: CardVocab, privileged: bool = False) -> None:
        self.cards = cards
        self.vocab = vocab
        self.privileged = privileged

    # -- public ------------------------------------------------------------
    def encode(self, point: DecisionPoint, core) -> dict[str, np.ndarray]:
        viewer = point.player
        table = np.zeros((N_CARDS, F_CARD), dtype=np.int32)
        rows: list[list[int]] = []
        keys: dict[tuple, int] = {}
        deck_rows: dict[int, int] = {}

        for side, con in ((0, viewer), (1, 1 - viewer)):
            mzone: list[dict | None] = []
            for loc in SIDE_LOCATIONS:
                slots = parse_query_location(core.query_location(CARD_QUERY_FLAGS, con, loc))
                if loc == C.LOCATION_MZONE:
                    mzone = slots
                for seq, card in enumerate(slots):
                    if card is None or (loc == C.LOCATION_EXTRA and side == 1 and not card["public"]):
                        continue
                    visible = bool(card["public"]) or side == 0
                    keys[(con, loc, seq)] = len(rows)
                    rows.append(self._card_row(card, loc, seq, side, viewer, visible))
            for seq, card in enumerate(mzone):
                for k, code in enumerate(card["overlay"] if card else ()):
                    keys[(con, C.LOCATION_OVERLAY, seq, k)] = len(rows)
                    rows.append(self._material_row(code, seq, k, side))
        deck = [c for c in parse_query_location(core.query_location(CARD_QUERY_FLAGS, viewer, C.LOCATION_DECK)) if c]
        for card in sorted(deck, key=lambda c: self.vocab.index(c["code"])):
            deck_rows.setdefault(card["code"], len(rows))
            rows.append(self._card_row(card, C.LOCATION_DECK, 0, 0, viewer, True))

        n = min(len(rows), N_CARDS)
        if n:
            table[:n] = np.asarray(rows[:n], dtype=np.int64).astype(np.int32)
        actions, mask = self._actions(point, viewer, keys, deck_rows, n)
        mask_duplicates(table, actions, mask, point.decision.TYPE)
        return {"cards": table, "globals": self._globals(point, core, viewer), "actions": actions, "action_mask": mask}

    def encode_privileged(self, point: DecisionPoint, core) -> dict[str, np.ndarray]:
        """Opponent ground truth for ``point.player`` (training mode only; never give it to the actor)."""
        if not self.privileged:
            raise RuntimeError("privileged tensors are disabled (inference mode); construct with privileged=True")
        return encode_privileged(core, point.player, self.vocab)

    # -- rows --------------------------------------------------------------
    def _card_row(self, card: dict, loc: int, seq: int, side: int, viewer: int, visible: bool) -> list[int]:
        row = [0] * F_CARD
        row[LOCATION] = LOCATION_ENUM[loc]
        row[SEQUENCE] = seq
        row[CONTROLLER] = side
        row[OWNER] = 0 if card.get("owner", viewer) == viewer else 1
        row[POSITION] = POSITION_ENUM.get(card.get("position", 0), 0)
        if not visible:
            row[CARD_INDEX] = CardVocab.UNKNOWN
            return row
        row[CARD_INDEX] = self.vocab.index(card["code"])
        row[VISIBLE] = 1
        row[PUBLIC] = int(bool(card["public"]))
        row[TYPE] = card.get("type", 0) & 0x7FFFFFFF
        row[ATTRIBUTE] = _bit_index(card.get("attribute", 0))
        row[RACE] = _bit_index(card.get("race", 0))
        row[LEVEL] = card.get("level", 0)
        row[RANK] = card.get("rank", 0)
        row[LINK] = card.get("link", 0)
        row[LSCALE] = card.get("lscale", 0)
        row[RSCALE] = card.get("rscale", 0)
        row[ATTACK] = _clamp(card.get("attack", 0))
        row[DEFENSE] = _clamp(card.get("defense", 0))
        row[LINK_MARKER] = card.get("link_marker", 0)
        row[COUNTERS] = sum(c >> 16 for c in card["counters"])
        row[MATERIALS] = len(card["overlay"])
        row[DISABLED] = int(bool(card.get("status", 0) & C.STATUS_DISABLED))
        return row

    def _material_row(self, code: int, seq: int, k: int, side: int) -> list[int]:
        row = [0] * F_CARD
        row[CARD_INDEX] = self.vocab.index(code)
        row[LOCATION] = LOCATION_ENUM[C.LOCATION_OVERLAY]
        row[SEQUENCE] = seq
        row[OVERLAY_INDEX] = k + 1
        row[CONTROLLER] = side
        row[OWNER] = side
        row[VISIBLE] = 1
        row[PUBLIC] = 1
        card = self.cards.get(code)
        if card is not None:
            row[TYPE] = card.type & 0x7FFFFFFF
            row[ATTRIBUTE] = _bit_index(card.attribute)
            row[RACE] = _bit_index(card.race)
            row[LEVEL] = card.level if not (card.is_xyz or card.is_link) else 0
            row[RANK] = card.rank
            row[LINK] = card.link
            row[LSCALE] = card.lscale
            row[RSCALE] = card.rscale
            row[ATTACK] = _clamp(card.attack)
            row[DEFENSE] = _clamp(card.defense)
            row[LINK_MARKER] = card.link_marker
        return row

    # -- globals -----------------------------------------------------------
    def _globals(self, point: DecisionPoint, core, viewer: int) -> np.ndarray:
        g = np.zeros(G_GLOBAL, dtype=np.int32)
        g[0] = viewer
        g[1] = int(viewer == 0)
        g[2] = int(point.turn_player == viewer)
        g[3] = _clamp(point.turn, 0, 999)
        g[4] = _bit_index(point.phase)
        g[5] = _clamp(point.lp[viewer])
        g[6] = _clamp(point.lp[1 - viewer])
        for side, con in ((0, viewer), (1, 1 - viewer)):
            for i, loc in enumerate(COUNT_LOCATIONS):
                g[7 + side * 5 + i] = core.query_count(con, loc)
        field = M.decode_message(bytes([C.MSG_RELOAD_FIELD]) + core.query_field())
        g[17] = len(field.chain) if isinstance(field, M.ReloadField) else 0
        g[18] = point.decision.TYPE
        g[19] = len(getattr(point.state, "picked", ()))
        g[20] = len(point.actions)
        g[21] = int(point.augmented_start)  # augmented (mid-game) start, DuelConfig.augmented_start (T2.6)
        return g

    # -- actions -----------------------------------------------------------
    def _actions(self, point: DecisionPoint, viewer: int, keys: dict, deck_rows: dict, n_rows: int):
        table = np.zeros((MAX_OPTIONS, A_ACTION), dtype=np.int32)
        mask = np.zeros(MAX_OPTIONS, dtype=np.int32)
        for i, action in enumerate(point.actions[:MAX_OPTIONS]):
            row = table[i]
            mask[i] = 1
            row[0] = ACTION_KINDS.index(action.kind) + 1
            card = action.card
            if card is not None:
                if card.code:  # 0 when hidden from the decider (messages.hide_private)
                    row[2] = self.vocab.index(card.code)
                ref = self._card_ref(card, viewer, keys, deck_rows)
                if ref is not None and ref < n_rows:
                    row[1] = ref + 1
            if action.kind == "declare":
                row[2] = self.vocab.index(action.value)
            desc = action.description
            if desc:  # aux.Stringid(code, n) == code << 20 | n; smaller values are system strings
                if (desc >> 20) in self.cards:
                    row[3] = self.vocab.index(desc >> 20)
                    row[4] = _clamp((desc & 0xFFFFF) + 1)
                else:
                    row[5] = _clamp(desc)
            if action.kind == "position":
                row[6] = POSITION_ENUM.get(action.value, 0)
            elif action.kind == "place":
                player, loc, seq = action.value >> 16, (action.value >> 8) & 0xFF, action.value & 0xFF
                row[7] = (0 if player == viewer else 1) * 16 + (0 if loc == C.LOCATION_MZONE else 8) + seq + 1
            row[8] = self._value(action)
            row[9] = _clamp(action.index + 1, 0, 255)
        return table, mask

    @staticmethod
    def _value(action) -> int:
        kind = action.kind
        if kind in ("race", "attribute"):
            return _bit_index(action.value)
        if kind in ("number", "rps", "counter", "select", "unselect"):
            return _clamp(action.value & 0xFFFF)
        return 0

    @staticmethod
    def _card_ref(card: M.CardInfo, viewer: int, keys: dict, deck_rows: dict) -> int | None:
        loc = card.loc
        if loc.location & C.LOCATION_OVERLAY:
            return keys.get((loc.controller, C.LOCATION_OVERLAY, loc.sequence, loc.position))
        if loc.location == C.LOCATION_DECK:
            return deck_rows.get(card.code) if loc.controller == viewer else None
        return keys.get((loc.controller, loc.location, loc.sequence))
