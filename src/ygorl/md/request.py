"""What Master Duel is asking the client right now, read from memory (design 10, 自博弈的架构).

The engine's input state says which kind of input it waits for and from whom; the selection list
(``DLL_DuelList*``) holds the candidates of a card choice; the dialog block (``DLL_DuelDlg*``) holds the choices of
position / yes-no / option dialogs. Candidates are located through their unique id on the board read by
:mod:`ygorl.md.memory` (the list's own "from" word does not hold the location for every list kind).

Read-only, and only what is offered to the client: the list never holds the opponent's hidden cards.

Kinds (``Request.kind``) seen in Solo duels on DLL 97bd4d13 (input mode -> kind):
1 draw (MD waits for a click to go on), 2 idle (Main Phase command), 3 battle (Battle Phase command),
4 open (an activate-or-pass prompt at an open timing), 5 chain (activate-or-pass in a chain), 6 summon_pick
(which monster to summon), 7 position, 8 select_card. Unknown modes come back as ``"mode<N>"``.

Prompts (4, 5) list the cards that can be activated, like a card choice. The list keeps its last items after a
selection ends, so candidates are only read while a selection kind is pending. The dialog's "mix" entries are
typed parameters of the message shown (kind 1 message id, kind 5 a card id such as the card being resolved).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ygorl.md.memory import FIELD_SLOTS, PILES, Memory, RawDuel, decode_uid
from ygorl.md.offsets import Offsets

KINDS = {1: "draw", 2: "idle", 3: "battle", 4: "open", 5: "chain", 6: "summon_pick", 7: "position", 8: "select_card"}
MAX_ITEMS = 300  # the list holds at most 0x12c items (DLL_DuelListGetItemID checks it)


@dataclass(frozen=True)
class Candidate:
    cid: int  # MD card id
    uid: int  # unique id in the duel (0: not a card, e.g. a text option)
    player: int  # MD player that controls / owns it (-1 unknown)
    location: str  # "hand", "mzone", "szone", "grave", "removed", "extra", "deck", "" unknown
    sequence: int | None  # field slot sequence (mzone 0-6, szone 0-5) or pile index


@dataclass(frozen=True)
class Request:
    kind: str
    mode: int
    accepting: bool
    player: int  # MD player the engine waits for
    candidates: tuple[Candidate, ...] = ()
    min: int = 0
    max: int = 0
    list_type: int = 0
    list_marker: int = 0
    dialog_type: int = 0
    choices: tuple[int, ...] = ()  # dialog choices: raw u32 (enabled << 31 | default << 30 | text id)
    params: tuple[tuple[int, int], ...] = ()  # dialog mix entries (kind, value): 1 message id, 5 card id, 2/8 numbers
    source_cid: int = 0  # the card the selection is for (the list marker unless it is 0xfffa / 0xfffb), 0 unknown
    raw: dict = field(default_factory=dict, compare=False)

    def for_client(self, me: int) -> bool:
        return self.accepting and self.player == me


def _locate(raw: RawDuel, uid: int) -> tuple[int, str, int | None]:
    for p, pl in enumerate(raw.players):
        for i, c in enumerate(pl.slots[:FIELD_SLOTS]):
            if c is not None and c.uid == uid:
                return (p, "mzone", i) if i < 7 else (p, "szone", i - 7)  # slot 12, the Field Zone, is szone 5
        for zone in PILES:
            for k, c in enumerate(pl.piles[zone]):
                if c.uid == uid:
                    return p, zone, k
    return -1, "", None


def read_request(mem: Memory, base: int, off: Offsets, raw: RawDuel, *, list_shown: bool = False) -> Request:
    """MD's pending request; ``raw`` is a board read at the same moment (to locate the candidates). ``list_shown``:
    a card list is on screen, so the list is current even with the engine's input closed (MD's own dialogs)."""
    state = struct.unpack("<Q", mem.read(base + off.state_ptr, 8))[0]
    lst = struct.unpack("<Q", mem.read(base + off.list_ptr, 8))[0]

    def u32(addr: int) -> int:
        return struct.unpack("<I", mem.read(addr, 4))[0]

    mode, accepting, player = (
        u32(state + off.input_mode),
        u32(state + off.input_accepting),
        u32(state + off.input_player),
    )
    kind = KINDS.get(mode, f"mode{mode}")
    head = mem.read(lst, 0x10) if lst else bytes(0x10)
    count = max(0, min(struct.unpack_from("<h", head, off.list_count)[0], MAX_ITEMS))
    list_type = struct.unpack_from("<H", head, off.list_type)[0]
    marker = struct.unpack_from("<H", head, off.list_marker)[0]
    most = head[off.list_max]
    candidates = []
    if (list_shown or kind in ("select_card", "summon_pick", "open", "chain")) and count:  # stale otherwise
        items = mem.read(lst + off.list_item_cid, 4 * count)
        sources = mem.read(lst + off.list_item_from, 4 * count)
        for i in range(count):
            cid, word = struct.unpack_from("<HH", items, 4 * i)
            uid = decode_uid(word)
            p, loc, seq = _locate(raw, uid) if cid else (-1, "", None)
            if p < 0:
                p = struct.unpack_from("<H", sources, 4 * i)[0] & 1
            candidates.append(Candidate(cid, uid, p, loc, seq))
    dialog_type = u32(state + off.dialog_type)
    n_choices = min(u32(state + off.dialog_choice_count), 16)
    choices = tuple(u32(state + off.dialog_choices + 4 * i) for i in range(n_choices))
    if kind == "position" and not choices:
        choices = (u32(state + off.dialog_choices),)  # the position mask (DLL_DuelDlgGetPosMaskOfThisSummon)
    n_mix = min(u32(state + off.dialog_mix_count), 8)
    params = tuple((v >> 28, v & 0xFFFFFFF) for v in (u32(state + off.dialog_mix + 4 * i) for i in range(n_mix)))
    least = 0 if list_type <= 5 else 1
    return Request(kind=kind, mode=mode, accepting=bool(accepting), player=player & 1, candidates=tuple(candidates),
                   min=least if candidates else 0, max=min(most, count) if candidates else 0, list_type=list_type,
                   list_marker=marker, dialog_type=dialog_type, choices=choices, params=params,
                   source_cid=marker if candidates and marker not in (0xFFFA, 0xFFFB) else 0,
                   raw={"list_count": count, "list_max": most})  # fmt: skip


__all__ = ["KINDS", "Candidate", "Request", "read_request"]
