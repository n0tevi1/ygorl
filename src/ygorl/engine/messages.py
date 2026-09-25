"""Typed decoding of ygopro-core (edo9300, OCG API 11) ``MSG_*`` messages.

``OCG_DuelGetMessage`` returns a buffer of ``[u32 length][u8 msg_type][payload]``
records. :func:`split_messages` cuts the buffer and :func:`decode_message` turns
each record into a frozen dataclass. Layouts were transcribed from the core's
``new_message(...)`` write sites (playerop.cpp, processor.cpp, operations.cpp,
field.cpp, libduel.cpp, card.cpp, libdebug.cpp).

Decoding never raises: an unknown type yields :class:`UnknownMessage` and a
malformed payload yields :class:`UndecodableMessage`; both keep the raw bytes and
are logged, so the caller can degrade gracefully (ygo-agent aborts instead).

Location encodings used by the core:

* ``loc_info``: ``u8 controller, u8 location, u32 sequence, u32 position``
  (for overlay units ``position`` is the overlay index).
* "loc32": ``u8 controller, u8 location, u32 sequence`` (no position).
* "loc8": ``u8 controller, u8 location, u8 sequence``.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

from ygorl.engine import constants as C

log = logging.getLogger(__name__)


class MessageDecodeError(ValueError):
    pass


# --------------------------------------------------------------------------- reader


class Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf = buf
        self.pos = pos

    def _take(self, fmt: str, size: int):
        if self.pos + size > len(self.buf):
            raise MessageDecodeError(
                f"truncated payload: need {size} bytes at offset {self.pos}, have {len(self.buf) - self.pos}"
            )
        (v,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return v

    def u8(self) -> int:
        return self._take("<B", 1)

    def i8(self) -> int:
        return self._take("<b", 1)

    def u16(self) -> int:
        return self._take("<H", 2)

    def u32(self) -> int:
        return self._take("<I", 4)

    def i32(self) -> int:
        return self._take("<i", 4)

    def u64(self) -> int:
        return self._take("<Q", 8)

    def raw(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise MessageDecodeError(f"truncated payload: need {n} raw bytes at offset {self.pos}")
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def loc_info(self) -> Location:
        return Location(self.u8(), self.u8(), self.u32(), self.u32())

    def loc32(self) -> Location:
        return Location(self.u8(), self.u8(), self.u32(), 0)

    def loc8(self) -> Location:
        return Location(self.u8(), self.u8(), self.u8(), 0)

    def remaining(self) -> int:
        return len(self.buf) - self.pos


@dataclass(frozen=True, slots=True)
class Location:
    controller: int
    location: int
    sequence: int
    position: int = 0


@dataclass(frozen=True, slots=True)
class CardInfo:
    """A card as referenced by a message: its code (0 if hidden) and where it is."""

    code: int
    loc: Location


# --------------------------------------------------------------------------- base


@dataclass(frozen=True, slots=True)
class Message:
    """Base class; ``TYPE`` is the ``MSG_*`` id."""

    TYPE = -1

    @property
    def name(self) -> str:
        return MESSAGE_NAMES.get(self.TYPE, f"MSG_{self.TYPE}")


@dataclass(frozen=True, slots=True)
class Decision(Message):
    """A message that requires ``player`` to answer via ``set_response``."""

    player: int


@dataclass(frozen=True, slots=True)
class UnknownMessage(Message):
    msg_type: int
    raw: bytes

    @property
    def name(self) -> str:
        return MESSAGE_NAMES.get(self.msg_type, f"MSG_{self.msg_type}")


@dataclass(frozen=True, slots=True)
class UndecodableMessage(Message):
    msg_type: int
    raw: bytes
    error: str

    @property
    def name(self) -> str:
        return MESSAGE_NAMES.get(self.msg_type, f"MSG_{self.msg_type}")


MESSAGE_NAMES: dict[int, str] = {v: k for k, v in vars(C).items() if k.startswith("MSG_") and isinstance(v, int)}

_DECODERS: dict[int, Callable[[Reader], Message]] = {}


def _decoder(msg_type: int):
    def register(fn: Callable[[Reader], Message]):
        _DECODERS[msg_type] = fn
        return fn

    return register


def _simple(msg_type: int, cls: type[Message]) -> None:
    cls.TYPE = msg_type
    _DECODERS[msg_type] = lambda r: cls()


# ================================================================ decision messages


@dataclass(frozen=True, slots=True)
class ChainOption:
    """An activatable effect: handler card, effect description and client mode."""

    code: int
    loc: Location
    description: int
    client_mode: int


@dataclass(frozen=True, slots=True)
class AttackOption:
    code: int
    loc: Location
    direct_attackable: bool


@dataclass(frozen=True, slots=True)
class SelectBattleCmd(Decision):
    TYPE = C.MSG_SELECT_BATTLECMD
    activatable: tuple[ChainOption, ...]
    attackable: tuple[AttackOption, ...]
    can_main2: bool
    can_end_phase: bool


@_decoder(C.MSG_SELECT_BATTLECMD)
def _(r: Reader) -> Message:
    player = r.u8()
    act = tuple(ChainOption(r.u32(), r.loc32(), r.u64(), r.u8()) for _ in range(r.u32()))
    atk = tuple(AttackOption(r.u32(), r.loc8(), bool(r.u8())) for _ in range(r.u32()))
    return SelectBattleCmd(player, act, atk, bool(r.u8()), bool(r.u8()))


@dataclass(frozen=True, slots=True)
class SelectIdleCmd(Decision):
    TYPE = C.MSG_SELECT_IDLECMD
    summonable: tuple[CardInfo, ...]
    spsummonable: tuple[CardInfo, ...]
    repositionable: tuple[CardInfo, ...]
    msetable: tuple[CardInfo, ...]
    ssetable: tuple[CardInfo, ...]
    activatable: tuple[ChainOption, ...]
    can_battle_phase: bool
    can_end_phase: bool
    can_shuffle: bool


@_decoder(C.MSG_SELECT_IDLECMD)
def _(r: Reader) -> Message:
    player = r.u8()
    summ = tuple(CardInfo(r.u32(), r.loc32()) for _ in range(r.u32()))
    spsumm = tuple(CardInfo(r.u32(), r.loc32()) for _ in range(r.u32()))
    repos = tuple(CardInfo(r.u32(), r.loc8()) for _ in range(r.u32()))
    mset = tuple(CardInfo(r.u32(), r.loc32()) for _ in range(r.u32()))
    sset = tuple(CardInfo(r.u32(), r.loc32()) for _ in range(r.u32()))
    act = tuple(ChainOption(r.u32(), r.loc32(), r.u64(), r.u8()) for _ in range(r.u32()))
    return SelectIdleCmd(player, summ, spsumm, repos, mset, sset, act, bool(r.u8()), bool(r.u8()), bool(r.u8()))


@dataclass(frozen=True, slots=True)
class SelectEffectYn(Decision):
    TYPE = C.MSG_SELECT_EFFECTYN
    card: CardInfo
    description: int


@_decoder(C.MSG_SELECT_EFFECTYN)
def _(r: Reader) -> Message:
    return SelectEffectYn(r.u8(), CardInfo(r.u32(), r.loc_info()), r.u64())


@dataclass(frozen=True, slots=True)
class SelectYesNo(Decision):
    TYPE = C.MSG_SELECT_YESNO
    description: int


@_decoder(C.MSG_SELECT_YESNO)
def _(r: Reader) -> Message:
    return SelectYesNo(r.u8(), r.u64())


@dataclass(frozen=True, slots=True)
class SelectOption(Decision):
    TYPE = C.MSG_SELECT_OPTION
    options: tuple[int, ...]  # effect descriptions


@_decoder(C.MSG_SELECT_OPTION)
def _(r: Reader) -> Message:
    player = r.u8()
    return SelectOption(player, tuple(r.u64() for _ in range(r.u8())))


@dataclass(frozen=True, slots=True)
class SelectCard(Decision):
    """Choose between ``min`` and ``max`` cards (``cancelable``: may answer -1)."""

    TYPE = C.MSG_SELECT_CARD
    cancelable: bool
    min: int
    max: int
    cards: tuple[CardInfo, ...]


@_decoder(C.MSG_SELECT_CARD)
def _(r: Reader) -> Message:
    player, cancelable, mn, mx = r.u8(), bool(r.u8()), r.u32(), r.u32()
    return SelectCard(player, cancelable, mn, mx, tuple(CardInfo(r.u32(), r.loc_info()) for _ in range(r.u32())))


@dataclass(frozen=True, slots=True)
class SelectUnselectCard(Decision):
    """Pick one card to (un)select, or finish/cancel with -1."""

    TYPE = C.MSG_SELECT_UNSELECT_CARD
    finishable: bool
    cancelable: bool
    min: int
    max: int
    selectable: tuple[CardInfo, ...]
    unselectable: tuple[CardInfo, ...]


@_decoder(C.MSG_SELECT_UNSELECT_CARD)
def _(r: Reader) -> Message:
    player, fin, can, mn, mx = r.u8(), bool(r.u8()), bool(r.u8()), r.u32(), r.u32()
    sel = tuple(CardInfo(r.u32(), r.loc_info()) for _ in range(r.u32()))
    unsel = tuple(CardInfo(r.u32(), r.loc_info()) for _ in range(r.u32()))
    return SelectUnselectCard(player, fin, can, mn, mx, sel, unsel)


@dataclass(frozen=True, slots=True)
class SelectChain(Decision):
    """Chain an effect or pass (-1, only when not ``forced``)."""

    TYPE = C.MSG_SELECT_CHAIN
    special_count: int
    forced: bool
    hint_timing: int
    hint_timing_opponent: int
    chains: tuple[ChainOption, ...]


@_decoder(C.MSG_SELECT_CHAIN)
def _(r: Reader) -> Message:
    player, spe, forced, ht, hto = r.u8(), r.u8(), bool(r.u8()), r.u32(), r.u32()
    chains = tuple(ChainOption(r.u32(), r.loc_info(), r.u64(), r.u8()) for _ in range(r.u32()))
    return SelectChain(player, spe, forced, ht, hto, chains)


@dataclass(frozen=True, slots=True)
class SelectPlace(Decision):
    """Choose ``count`` zones. ``flag`` bits set = zone NOT selectable.

    Bits 0-6 own monster zones, 8-15 own spell/trap zones (13 = field zone),
    16-22 / 24-31 the same for the opponent.
    """

    TYPE = C.MSG_SELECT_PLACE
    count: int
    flag: int


@_decoder(C.MSG_SELECT_PLACE)
def _(r: Reader) -> Message:
    return SelectPlace(r.u8(), r.u8(), r.u32())


@dataclass(frozen=True, slots=True)
class SelectDisfield(SelectPlace):
    """Same layout as :class:`SelectPlace`; used to choose zones to disable."""

    TYPE = C.MSG_SELECT_DISFIELD


@_decoder(C.MSG_SELECT_DISFIELD)
def _(r: Reader) -> Message:
    return SelectDisfield(r.u8(), r.u8(), r.u32())


@dataclass(frozen=True, slots=True)
class SelectPosition(Decision):
    TYPE = C.MSG_SELECT_POSITION
    code: int
    positions: int  # POS_* bitmask of allowed positions


@_decoder(C.MSG_SELECT_POSITION)
def _(r: Reader) -> Message:
    return SelectPosition(r.u8(), r.u32(), r.u8())


@dataclass(frozen=True, slots=True)
class TributeOption:
    code: int
    loc: Location
    release_param: int  # how many tributes this card counts as


@dataclass(frozen=True, slots=True)
class SelectTribute(Decision):
    TYPE = C.MSG_SELECT_TRIBUTE
    cancelable: bool
    min: int
    max: int
    cards: tuple[TributeOption, ...]


@_decoder(C.MSG_SELECT_TRIBUTE)
def _(r: Reader) -> Message:
    player, cancelable, mn, mx = r.u8(), bool(r.u8()), r.u32(), r.u32()
    cards = tuple(TributeOption(r.u32(), r.loc32(), r.u8()) for _ in range(r.u32()))
    return SelectTribute(player, cancelable, mn, mx, cards)


@dataclass(frozen=True, slots=True)
class CounterOption:
    code: int
    loc: Location
    counters: int


@dataclass(frozen=True, slots=True)
class SelectCounter(Decision):
    """Remove exactly ``count`` counters of ``counter_type`` spread over ``cards``."""

    TYPE = C.MSG_SELECT_COUNTER
    counter_type: int
    count: int
    cards: tuple[CounterOption, ...]


@_decoder(C.MSG_SELECT_COUNTER)
def _(r: Reader) -> Message:
    player, ctype, count = r.u8(), r.u16(), r.u16()
    return SelectCounter(player, ctype, count, tuple(CounterOption(r.u32(), r.loc8(), r.u16()) for _ in range(r.u32())))


@dataclass(frozen=True, slots=True)
class SumOption:
    code: int
    loc: Location
    param: int  # low 16 bits: value; high 16 bits: alternative value (0 = none)

    @property
    def values(self) -> tuple[int, ...]:
        lo, hi = self.param & 0xFFFF, self.param >> 16
        return (lo, hi) if hi else (lo,)


@dataclass(frozen=True, slots=True)
class SelectSum(Decision):
    """Select cards whose values sum to ``target``.

    ``exact``: the sum must equal ``target`` using between ``min`` and ``max``
    selected cards. Otherwise (``SelectWithSumGreater``) the sum must reach
    ``target`` and dropping the smallest card must fall below it. ``must``
    cards are always included and not part of the response.
    """

    TYPE = C.MSG_SELECT_SUM
    exact: bool
    target: int
    min: int
    max: int
    must: tuple[SumOption, ...]
    cards: tuple[SumOption, ...]


@_decoder(C.MSG_SELECT_SUM)
def _(r: Reader) -> Message:
    player, mode, target, mn, mx = r.u8(), r.u8(), r.u32(), r.u32(), r.u32()
    must = tuple(SumOption(r.u32(), r.loc_info(), r.u32()) for _ in range(r.u32()))
    cards = tuple(SumOption(r.u32(), r.loc_info(), r.u32()) for _ in range(r.u32()))
    return SelectSum(player, mode == 0, target, mn, mx, must, cards)


@dataclass(frozen=True, slots=True)
class SortCard(Decision):
    TYPE = C.MSG_SORT_CARD
    cards: tuple[CardInfo, ...]


@dataclass(frozen=True, slots=True)
class SortChain(SortCard):
    TYPE = C.MSG_SORT_CHAIN


def _sort(cls):
    def dec(r: Reader) -> Message:
        player = r.u8()
        return cls(player, tuple(CardInfo(r.u32(), Location(r.u8(), r.u32(), r.u32())) for _ in range(r.u32())))

    return dec


_DECODERS[C.MSG_SORT_CARD] = _sort(SortCard)
_DECODERS[C.MSG_SORT_CHAIN] = _sort(SortChain)


@dataclass(frozen=True, slots=True)
class AnnounceRace(Decision):
    TYPE = C.MSG_ANNOUNCE_RACE
    count: int
    available: int


@_decoder(C.MSG_ANNOUNCE_RACE)
def _(r: Reader) -> Message:
    return AnnounceRace(r.u8(), r.u8(), r.u64())


@dataclass(frozen=True, slots=True)
class AnnounceAttrib(Decision):
    TYPE = C.MSG_ANNOUNCE_ATTRIB
    count: int
    available: int


@_decoder(C.MSG_ANNOUNCE_ATTRIB)
def _(r: Reader) -> Message:
    return AnnounceAttrib(r.u8(), r.u8(), r.u32())


@dataclass(frozen=True, slots=True)
class AnnounceCard(Decision):
    """Declare a card name. ``opcodes`` is an RPN filter program (``OPCODE_*``)."""

    TYPE = C.MSG_ANNOUNCE_CARD
    opcodes: tuple[int, ...]


@_decoder(C.MSG_ANNOUNCE_CARD)
def _(r: Reader) -> Message:
    player = r.u8()
    return AnnounceCard(player, tuple(r.u64() for _ in range(r.u8())))


@dataclass(frozen=True, slots=True)
class AnnounceNumber(Decision):
    TYPE = C.MSG_ANNOUNCE_NUMBER
    options: tuple[int, ...]


@_decoder(C.MSG_ANNOUNCE_NUMBER)
def _(r: Reader) -> Message:
    player = r.u8()
    return AnnounceNumber(player, tuple(r.u64() for _ in range(r.u8())))


@dataclass(frozen=True, slots=True)
class RockPaperScissors(Decision):
    TYPE = C.MSG_ROCK_PAPER_SCISSORS


@_decoder(C.MSG_ROCK_PAPER_SCISSORS)
def _(r: Reader) -> Message:
    return RockPaperScissors(r.u8())


DECISION_TYPES: frozenset[int] = frozenset(
    {
        C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_IDLECMD, C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_YESNO,
        C.MSG_SELECT_OPTION, C.MSG_SELECT_CARD, C.MSG_SELECT_CHAIN, C.MSG_SELECT_PLACE,
        C.MSG_SELECT_POSITION, C.MSG_SELECT_TRIBUTE, C.MSG_SORT_CHAIN, C.MSG_SELECT_COUNTER,
        C.MSG_SELECT_SUM, C.MSG_SELECT_DISFIELD, C.MSG_SORT_CARD, C.MSG_SELECT_UNSELECT_CARD,
        C.MSG_ROCK_PAPER_SCISSORS, C.MSG_ANNOUNCE_RACE, C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_CARD,
        C.MSG_ANNOUNCE_NUMBER,
    }
)  # fmt: skip


# =================================================================== event messages


@dataclass(frozen=True, slots=True)
class Retry(Message):
    """The previous response was invalid; the same decision is asked again."""


_simple(C.MSG_RETRY, Retry)


@dataclass(frozen=True, slots=True)
class Hint(Message):
    TYPE = C.MSG_HINT
    hint_type: int
    player: int
    data: int


@_decoder(C.MSG_HINT)
def _(r: Reader) -> Message:
    return Hint(r.u8(), r.u8(), r.u64())


@dataclass(frozen=True, slots=True)
class Waiting(Message):
    pass


_simple(C.MSG_WAITING, Waiting)


@dataclass(frozen=True, slots=True)
class Win(Message):
    TYPE = C.MSG_WIN
    player: int  # 0, 1, or 2 (PLAYER_NONE) for a draw
    reason: int


@_decoder(C.MSG_WIN)
def _(r: Reader) -> Message:
    return Win(r.u8(), r.u8())


@dataclass(frozen=True, slots=True)
class ConfirmCards(Message):
    """``player`` is shown ``cards`` (CONFIRM_CARDS / CONFIRM_DECKTOP / CONFIRM_EXTRATOP)."""

    TYPE = C.MSG_CONFIRM_CARDS
    player: int
    cards: tuple[CardInfo, ...]


@dataclass(frozen=True, slots=True)
class ConfirmDeckTop(ConfirmCards):
    TYPE = C.MSG_CONFIRM_DECKTOP


@dataclass(frozen=True, slots=True)
class ConfirmExtraTop(ConfirmCards):
    TYPE = C.MSG_CONFIRM_EXTRATOP


def _confirm(cls):
    def dec(r: Reader) -> Message:
        player = r.u8()
        return cls(player, tuple(CardInfo(r.u32(), r.loc32()) for _ in range(r.u32())))

    return dec


for _cls in (ConfirmCards, ConfirmDeckTop, ConfirmExtraTop):
    _DECODERS[_cls.TYPE] = _confirm(_cls)


@dataclass(frozen=True, slots=True)
class ShuffleDeck(Message):
    TYPE = C.MSG_SHUFFLE_DECK
    player: int


@_decoder(C.MSG_SHUFFLE_DECK)
def _(r: Reader) -> Message:
    return ShuffleDeck(r.u8())


@dataclass(frozen=True, slots=True)
class ShuffleHand(Message):
    TYPE = C.MSG_SHUFFLE_HAND
    player: int
    codes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ShuffleExtra(ShuffleHand):
    TYPE = C.MSG_SHUFFLE_EXTRA


def _shuffle(cls):
    def dec(r: Reader) -> Message:
        player = r.u8()
        return cls(player, tuple(r.u32() for _ in range(r.u32())))

    return dec


_DECODERS[C.MSG_SHUFFLE_HAND] = _shuffle(ShuffleHand)
_DECODERS[C.MSG_SHUFFLE_EXTRA] = _shuffle(ShuffleExtra)


@dataclass(frozen=True, slots=True)
class SwapGraveDeck(Message):
    TYPE = C.MSG_SWAP_GRAVE_DECK
    player: int
    extra_count: int  # main-deck-side extra deck size before inserting returned monsters
    extra_mask: bytes  # bit i set: i-th card of the new deck went to the extra deck


@_decoder(C.MSG_SWAP_GRAVE_DECK)
def _(r: Reader) -> Message:
    player, extra = r.u8(), r.u32()
    return SwapGraveDeck(player, extra, r.raw(r.u32()))


@dataclass(frozen=True, slots=True)
class ShuffleSetCard(Message):
    TYPE = C.MSG_SHUFFLE_SET_CARD
    location: int
    new_locations: tuple[Location, ...]
    overlay_locations: tuple[Location, ...]


@_decoder(C.MSG_SHUFFLE_SET_CARD)
def _(r: Reader) -> Message:
    loc, ct = r.u8(), r.u8()
    new = tuple(r.loc_info() for _ in range(ct))
    return ShuffleSetCard(loc, new, tuple(r.loc_info() for _ in range(ct)))


@dataclass(frozen=True, slots=True)
class ReverseDeck(Message):
    pass


_simple(C.MSG_REVERSE_DECK, ReverseDeck)


@dataclass(frozen=True, slots=True)
class DeckTop(Message):
    TYPE = C.MSG_DECK_TOP
    player: int
    sequence: int
    code: int
    position: int


@_decoder(C.MSG_DECK_TOP)
def _(r: Reader) -> Message:
    return DeckTop(r.u8(), r.u32(), r.u32(), r.u32())


@dataclass(frozen=True, slots=True)
class NewTurn(Message):
    TYPE = C.MSG_NEW_TURN
    player: int


@_decoder(C.MSG_NEW_TURN)
def _(r: Reader) -> Message:
    return NewTurn(r.u8())


@dataclass(frozen=True, slots=True)
class NewPhase(Message):
    TYPE = C.MSG_NEW_PHASE
    phase: int


@_decoder(C.MSG_NEW_PHASE)
def _(r: Reader) -> Message:
    return NewPhase(r.u16())


@dataclass(frozen=True, slots=True)
class Move(Message):
    TYPE = C.MSG_MOVE
    code: int  # 0 when hidden from the viewer
    previous: Location
    current: Location
    reason: int


@_decoder(C.MSG_MOVE)
def _(r: Reader) -> Message:
    return Move(r.u32(), r.loc_info(), r.loc_info(), r.u32())


@dataclass(frozen=True, slots=True)
class PosChange(Message):
    TYPE = C.MSG_POS_CHANGE
    code: int
    loc: Location
    previous_position: int
    current_position: int


@_decoder(C.MSG_POS_CHANGE)
def _(r: Reader) -> Message:
    code, loc = r.u32(), r.loc8()
    return PosChange(code, loc, r.u8(), r.u8())


@dataclass(frozen=True, slots=True)
class CardEvent(Message):
    """Base for messages carrying one ``(code, loc_info)`` pair."""

    code: int
    loc: Location


@dataclass(frozen=True, slots=True)
class SetCard(CardEvent):
    TYPE = C.MSG_SET


@dataclass(frozen=True, slots=True)
class Summoning(CardEvent):
    TYPE = C.MSG_SUMMONING


@dataclass(frozen=True, slots=True)
class SpSummoning(CardEvent):
    TYPE = C.MSG_SPSUMMONING


@dataclass(frozen=True, slots=True)
class FlipSummoning(CardEvent):
    TYPE = C.MSG_FLIPSUMMONING


for _cls in (SetCard, Summoning, SpSummoning, FlipSummoning):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: cls(r.u32(), r.loc_info()))(_cls)


@dataclass(frozen=True, slots=True)
class Summoned(Message):
    pass


@dataclass(frozen=True, slots=True)
class SpSummoned(Message):
    pass


@dataclass(frozen=True, slots=True)
class FlipSummoned(Message):
    pass


_simple(C.MSG_SUMMONED, Summoned)
_simple(C.MSG_SPSUMMONED, SpSummoned)
_simple(C.MSG_FLIPSUMMONED, FlipSummoned)


@dataclass(frozen=True, slots=True)
class Swap(Message):
    TYPE = C.MSG_SWAP
    code1: int
    loc1: Location
    code2: int
    loc2: Location


@_decoder(C.MSG_SWAP)
def _(r: Reader) -> Message:
    return Swap(r.u32(), r.loc_info(), r.u32(), r.loc_info())


@dataclass(frozen=True, slots=True)
class FieldDisabled(Message):
    TYPE = C.MSG_FIELD_DISABLED
    flag: int


@_decoder(C.MSG_FIELD_DISABLED)
def _(r: Reader) -> Message:
    return FieldDisabled(r.u32())


@dataclass(frozen=True, slots=True)
class Chaining(Message):
    TYPE = C.MSG_CHAINING
    code: int
    loc: Location
    triggering_controller: int
    triggering_location: int
    triggering_sequence: int
    description: int
    chain_count: int


@_decoder(C.MSG_CHAINING)
def _(r: Reader) -> Message:
    return Chaining(r.u32(), r.loc_info(), r.u8(), r.u8(), r.u32(), r.u64(), r.u32())


@dataclass(frozen=True, slots=True)
class ChainStep(Message):
    """Base for messages carrying only a chain link number."""

    chain_count: int


@dataclass(frozen=True, slots=True)
class Chained(ChainStep):
    TYPE = C.MSG_CHAINED


@dataclass(frozen=True, slots=True)
class ChainSolving(ChainStep):
    TYPE = C.MSG_CHAIN_SOLVING


@dataclass(frozen=True, slots=True)
class ChainSolved(ChainStep):
    TYPE = C.MSG_CHAIN_SOLVED


@dataclass(frozen=True, slots=True)
class ChainNegated(ChainStep):
    TYPE = C.MSG_CHAIN_NEGATED


@dataclass(frozen=True, slots=True)
class ChainDisabled(ChainStep):
    TYPE = C.MSG_CHAIN_DISABLED


for _cls in (Chained, ChainSolving, ChainSolved, ChainNegated, ChainDisabled):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: cls(r.u8()))(_cls)


@dataclass(frozen=True, slots=True)
class ChainEnd(Message):
    pass


_simple(C.MSG_CHAIN_END, ChainEnd)


@dataclass(frozen=True, slots=True)
class CardList(Message):
    """Base for messages carrying a list of card locations."""

    locations: tuple[Location, ...]


@dataclass(frozen=True, slots=True)
class CardSelected(CardList):
    TYPE = C.MSG_CARD_SELECTED


@dataclass(frozen=True, slots=True)
class BecomeTarget(CardList):
    TYPE = C.MSG_BECOME_TARGET


@dataclass(frozen=True, slots=True)
class RemoveCards(CardList):
    TYPE = C.MSG_REMOVE_CARDS


for _cls in (CardSelected, BecomeTarget, RemoveCards):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: cls(tuple(r.loc_info() for _ in range(r.u32()))))(_cls)


@dataclass(frozen=True, slots=True)
class RandomSelected(Message):
    TYPE = C.MSG_RANDOM_SELECTED
    player: int
    locations: tuple[Location, ...]


@_decoder(C.MSG_RANDOM_SELECTED)
def _(r: Reader) -> Message:
    player = r.u8()
    return RandomSelected(player, tuple(r.loc_info() for _ in range(r.u32())))


@dataclass(frozen=True, slots=True)
class Draw(Message):
    TYPE = C.MSG_DRAW
    player: int
    cards: tuple[tuple[int, int], ...]  # (code, position); code is 0 if hidden


@_decoder(C.MSG_DRAW)
def _(r: Reader) -> Message:
    player = r.u8()
    return Draw(player, tuple((r.u32(), r.u32()) for _ in range(r.u32())))


@dataclass(frozen=True, slots=True)
class LpEvent(Message):
    """Base for ``(player, amount)`` messages."""

    player: int
    amount: int


@dataclass(frozen=True, slots=True)
class Damage(LpEvent):
    TYPE = C.MSG_DAMAGE


@dataclass(frozen=True, slots=True)
class Recover(LpEvent):
    TYPE = C.MSG_RECOVER


@dataclass(frozen=True, slots=True)
class LpUpdate(LpEvent):
    """``amount`` is the new LP value."""

    TYPE = C.MSG_LPUPDATE


@dataclass(frozen=True, slots=True)
class PayLpCost(LpEvent):
    TYPE = C.MSG_PAY_LPCOST


for _cls in (Damage, Recover, LpUpdate, PayLpCost):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: cls(r.u8(), r.u32()))(_cls)


@dataclass(frozen=True, slots=True)
class CardPair(Message):
    """Base for ``(loc_info, loc_info)`` messages."""

    card: Location
    target: Location


@dataclass(frozen=True, slots=True)
class Equip(CardPair):
    TYPE = C.MSG_EQUIP


@dataclass(frozen=True, slots=True)
class CardTarget(CardPair):
    TYPE = C.MSG_CARD_TARGET


@dataclass(frozen=True, slots=True)
class CancelTarget(CardPair):
    TYPE = C.MSG_CANCEL_TARGET


@dataclass(frozen=True, slots=True)
class Attack(CardPair):
    """``target.location == 0`` means a direct attack."""

    TYPE = C.MSG_ATTACK


for _cls in (Equip, CardTarget, CancelTarget, Attack):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: cls(r.loc_info(), r.loc_info()))(_cls)


@dataclass(frozen=True, slots=True)
class Unequip(Message):
    TYPE = C.MSG_UNEQUIP
    card: Location


@_decoder(C.MSG_UNEQUIP)
def _(r: Reader) -> Message:
    return Unequip(r.loc_info())


@dataclass(frozen=True, slots=True)
class CounterChange(Message):
    counter_type: int
    loc: Location
    count: int


@dataclass(frozen=True, slots=True)
class AddCounter(CounterChange):
    TYPE = C.MSG_ADD_COUNTER


@dataclass(frozen=True, slots=True)
class RemoveCounter(CounterChange):
    TYPE = C.MSG_REMOVE_COUNTER


for _cls in (AddCounter, RemoveCounter):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: cls(r.u16(), r.loc8(), r.u16()))(_cls)


@dataclass(frozen=True, slots=True)
class Battle(Message):
    TYPE = C.MSG_BATTLE
    attacker: Location
    attacker_atk: int
    attacker_def: int
    attacker_destroyed: int
    target: Location
    target_atk: int
    target_def: int
    target_destroyed: int


@_decoder(C.MSG_BATTLE)
def _(r: Reader) -> Message:
    return Battle(r.loc_info(), r.i32(), r.i32(), r.u8(), r.loc_info(), r.i32(), r.i32(), r.u8())


@dataclass(frozen=True, slots=True)
class AttackDisabled(Message):
    pass


@dataclass(frozen=True, slots=True)
class DamageStepStart(Message):
    pass


@dataclass(frozen=True, slots=True)
class DamageStepEnd(Message):
    pass


_simple(C.MSG_ATTACK_DISABLED, AttackDisabled)
_simple(C.MSG_DAMAGE_STEP_START, DamageStepStart)
_simple(C.MSG_DAMAGE_STEP_END, DamageStepEnd)


@dataclass(frozen=True, slots=True)
class MissedEffect(Message):
    TYPE = C.MSG_MISSED_EFFECT
    loc: Location
    code: int


@_decoder(C.MSG_MISSED_EFFECT)
def _(r: Reader) -> Message:
    return MissedEffect(r.loc_info(), r.u32())


@dataclass(frozen=True, slots=True)
class Toss(Message):
    player: int
    results: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TossCoin(Toss):
    TYPE = C.MSG_TOSS_COIN


@dataclass(frozen=True, slots=True)
class TossDice(Toss):
    TYPE = C.MSG_TOSS_DICE


for _cls in (TossCoin, TossDice):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: (lambda p: cls(p, tuple(r.u8() for _ in range(r.u8()))))(r.u8()))(
        _cls
    )


@dataclass(frozen=True, slots=True)
class HandResult(Message):
    TYPE = C.MSG_HAND_RES
    hand0: int  # 1 scissors? (value as sent by the core: 1..3)
    hand1: int


@_decoder(C.MSG_HAND_RES)
def _(r: Reader) -> Message:
    v = r.u8()
    return HandResult(v & 0x3, (v >> 2) & 0x3)


@dataclass(frozen=True, slots=True)
class CardHint(Message):
    TYPE = C.MSG_CARD_HINT
    loc: Location
    hint_type: int
    value: int


@_decoder(C.MSG_CARD_HINT)
def _(r: Reader) -> Message:
    return CardHint(r.loc_info(), r.u8(), r.u64())


@dataclass(frozen=True, slots=True)
class PlayerHint(Message):
    TYPE = C.MSG_PLAYER_HINT
    player: int
    hint_type: int
    value: int


@_decoder(C.MSG_PLAYER_HINT)
def _(r: Reader) -> Message:
    return PlayerHint(r.u8(), r.u8(), r.u64())


@dataclass(frozen=True, slots=True)
class TagSwap(Message):
    TYPE = C.MSG_TAG_SWAP
    player: int
    main_count: int
    extra_count: int
    extra_p_count: int
    hand_count: int
    deck_top_code: int
    hand: tuple[tuple[int, int], ...]
    extra: tuple[tuple[int, int], ...]


@_decoder(C.MSG_TAG_SWAP)
def _(r: Reader) -> Message:
    player, main, extra, extra_p, hand, top = r.u8(), r.u32(), r.u32(), r.u32(), r.u32(), r.u32()
    hand_cards = tuple((r.u32(), r.u32()) for _ in range(hand))
    extra_cards = tuple((r.u32(), r.u32()) for _ in range(extra))
    return TagSwap(player, main, extra, extra_p, hand, top, hand_cards, extra_cards)


@dataclass(frozen=True, slots=True)
class ReloadField(Message):
    TYPE = C.MSG_RELOAD_FIELD
    duel_options: int
    players: tuple[dict, dict]
    chain: tuple[dict, ...]


@_decoder(C.MSG_RELOAD_FIELD)
def _(r: Reader) -> Message:
    opts = r.u32()
    players = []
    for _ in range(2):
        p: dict = {"lp": r.i32()}
        for zone, n in (("mzone", 7), ("szone", 8)):
            slots = []
            for _ in range(n):
                slots.append((r.u8(), r.u32()) if r.u8() else None)
            p[zone] = slots
        for key in ("main", "hand", "grave", "removed", "extra", "extra_p"):
            p[key] = r.u32()
        players.append(p)
    chain = tuple(
        {
            "code": r.u32(),
            "loc": r.loc_info(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "description": r.u64(),
        }
        for _ in range(r.u32())
    )
    return ReloadField(opts, (players[0], players[1]), chain)


@dataclass(frozen=True, slots=True)
class StringMessage(Message):
    text: str


@dataclass(frozen=True, slots=True)
class AiName(StringMessage):
    TYPE = C.MSG_AI_NAME


@dataclass(frozen=True, slots=True)
class ShowHint(StringMessage):
    TYPE = C.MSG_SHOW_HINT


for _cls in (AiName, ShowHint):
    _DECODERS[_cls.TYPE] = (lambda cls: lambda r: cls(r.raw(r.u16()).decode("utf-8", "replace")))(_cls)


@dataclass(frozen=True, slots=True)
class MatchKill(Message):
    TYPE = C.MSG_MATCH_KILL
    code: int


@_decoder(C.MSG_MATCH_KILL)
def _(r: Reader) -> Message:
    return MatchKill(r.u32())


# Messages that the core never emits (client/server-side in EDOPro) are kept as
# UnknownMessage when they appear: MSG_START, MSG_UPDATE_DATA, MSG_UPDATE_CARD,
# MSG_REQUEST_DECK, MSG_REFRESH_DECK, MSG_BE_CHAIN_TARGET, MSG_CREATE_RELATION,
# MSG_RELEASE_RELATION, MSG_CUSTOM_MSG.
NOT_EMITTED_BY_CORE: frozenset[int] = frozenset(
    {
        C.MSG_START, C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD, C.MSG_REQUEST_DECK, C.MSG_REFRESH_DECK,
        C.MSG_BE_CHAIN_TARGET, C.MSG_CREATE_RELATION, C.MSG_RELEASE_RELATION, C.MSG_CUSTOM_MSG,
    }
)  # fmt: skip


# =================================================================== entry points


def split_messages(buf: bytes) -> Iterator[bytes]:
    """Yield each ``[msg_type][payload]`` record of an ``OCG_DuelGetMessage`` buffer."""
    pos = 0
    n = len(buf)
    while pos < n:
        if pos + 4 > n:
            raise MessageDecodeError(f"truncated length prefix at offset {pos}")
        (length,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if length == 0 or pos + length > n:
            raise MessageDecodeError(f"bad record length {length} at offset {pos - 4}")
        yield buf[pos : pos + length]
        pos += length


def decode_message(record: bytes) -> Message:
    """Decode one record; never raises."""
    msg_type = record[0]
    decoder = _DECODERS.get(msg_type)
    if decoder is None:
        log.warning("unknown message type %d (%d payload bytes)", msg_type, len(record) - 1)
        return UnknownMessage(msg_type, bytes(record[1:]))
    reader = Reader(record, 1)
    try:
        msg = decoder(reader)
    except (MessageDecodeError, struct.error) as exc:
        log.warning("cannot decode %s: %s", MESSAGE_NAMES.get(msg_type, msg_type), exc)
        return UndecodableMessage(msg_type, bytes(record[1:]), str(exc))
    if reader.remaining():
        log.warning("%s: %d trailing bytes ignored", MESSAGE_NAMES.get(msg_type, msg_type), reader.remaining())
    return msg


def is_hidden_from(viewer: int, loc: Location) -> bool:
    """A card at ``loc`` is hidden from ``viewer`` (EDOPro's public test for moves, generic_duel.cpp):
    it belongs to the other player, is not in the GY or an overlay, and is in the deck / hand or face-down.
    A position of 0 means the message did not say (SELECT_TRIBUTE) and counts as hidden."""
    if loc.controller == viewer or loc.location & (C.LOCATION_GRAVE | C.LOCATION_OVERLAY):
        return False
    return bool(loc.location & (C.LOCATION_DECK | C.LOCATION_HAND) or loc.position & C.POS_FACEDOWN or not loc.position)


def _mask(cards: tuple, viewer: int) -> tuple:
    return tuple(replace(c, code=0) if c.code and is_hidden_from(viewer, c.loc) else c for c in cards)


def hide_private(decision: Message) -> Message:
    """The decision as the deciding player may see it: hidden cards' codes set to 0.

    The core writes real passwords into SELECT_CARD / SELECT_TRIBUTE / SELECT_UNSELECT_CARD even for
    the opponent's face-down or hand cards; EDOPro's server strips opponent codes from exactly these
    messages before sending them (generic_duel.cpp ``Sending``). We strip the codes of cards that are
    hidden from the decider (:func:`is_hidden_from`) and keep those of public cards, which the player
    sees on the field anyway. Other decisions are returned unchanged.
    """
    if isinstance(decision, (SelectCard, SelectTribute)):
        return replace(decision, cards=_mask(decision.cards, decision.player))
    if isinstance(decision, SelectUnselectCard):
        return replace(decision, selectable=_mask(decision.selectable, decision.player),
                       unselectable=_mask(decision.unselectable, decision.player))  # fmt: skip
    return decision


def decode_buffer(buf: bytes) -> list[Message]:
    """Decode a whole buffer; a corrupt framing yields one UndecodableMessage for the rest."""
    out: list[Message] = []
    try:
        for record in split_messages(buf):
            out.append(decode_message(record))
    except MessageDecodeError as exc:
        log.warning("corrupt message buffer: %s", exc)
        out.append(UndecodableMessage(-1, bytes(buf), str(exc)))
    return out


def decoded_types() -> frozenset[int]:
    """Message types with a typed decoder (used by the cross-check test)."""
    return frozenset(_DECODERS)


__all__ = [name for name, obj in list(globals().items()) if isinstance(obj, type) and issubclass(obj, Message)] + [
    "Location",
    "CardInfo",
    "ChainOption",
    "AttackOption",
    "TributeOption",
    "CounterOption",
    "SumOption",
    "MessageDecodeError",
    "DECISION_TYPES",
    "NOT_EMITTED_BY_CORE",
    "MESSAGE_NAMES",
    "split_messages",
    "decode_message",
    "decode_buffer",
    "decoded_types",
]
