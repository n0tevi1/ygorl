"""Parsing of ``OCG_DuelQueryLocation`` / ``OCG_DuelQuery`` results.

A location query is ``u32 total size`` followed by one record per slot: an
empty slot is ``i16 0``; a card is a list of ``[u16 len][u32 QUERY_* flag][value]``
chunks ending with ``QUERY_END`` (layout from ``card::get_infos`` in card.cpp).
``QUERY_IS_PUBLIC`` is always present regardless of the requested flags.
"""

from __future__ import annotations

import struct

from ygorl.engine import constants as C

CARD_QUERY_FLAGS = (
    C.QUERY_CODE | C.QUERY_POSITION | C.QUERY_TYPE | C.QUERY_LEVEL | C.QUERY_RANK | C.QUERY_ATTRIBUTE | C.QUERY_RACE
    | C.QUERY_ATTACK | C.QUERY_DEFENSE | C.QUERY_OVERLAY_CARD | C.QUERY_COUNTERS | C.QUERY_OWNER | C.QUERY_STATUS
    | C.QUERY_LSCALE | C.QUERY_RSCALE | C.QUERY_LINK
)  # fmt: skip

_U32 = {
    C.QUERY_CODE: "code", C.QUERY_POSITION: "position", C.QUERY_ALIAS: "alias", C.QUERY_TYPE: "type",
    C.QUERY_LEVEL: "level", C.QUERY_RANK: "rank", C.QUERY_ATTRIBUTE: "attribute", C.QUERY_BASE_ATTACK: "base_attack",
    C.QUERY_BASE_DEFENSE: "base_defense", C.QUERY_REASON: "reason", C.QUERY_COVER: "cover", C.QUERY_STATUS: "status",
    C.QUERY_LSCALE: "lscale", C.QUERY_RSCALE: "rscale",
}  # fmt: skip
_I32 = {C.QUERY_ATTACK: "attack", C.QUERY_DEFENSE: "defense"}


def _parse_card(buf: bytes, pos: int) -> tuple[dict, int]:
    card: dict = {"public": 0, "overlay": [], "counters": []}
    while True:
        (length,) = struct.unpack_from("<H", buf, pos)
        (flag,) = struct.unpack_from("<I", buf, pos + 2)
        body = pos + 6
        if flag == C.QUERY_END:
            return card, pos + 2 + length
        if flag in _U32:
            card[_U32[flag]] = struct.unpack_from("<I", buf, body)[0]
        elif flag in _I32:
            card[_I32[flag]] = struct.unpack_from("<i", buf, body)[0]
        elif flag == C.QUERY_RACE:
            card["race"] = struct.unpack_from("<Q", buf, body)[0]
        elif flag == C.QUERY_OWNER:
            card["owner"] = buf[body]
        elif flag == C.QUERY_IS_PUBLIC:
            card["public"] = buf[body]
        elif flag == C.QUERY_IS_HIDDEN:
            card["hidden"] = buf[body]
        elif flag == C.QUERY_LINK:
            card["link"], card["link_marker"] = struct.unpack_from("<II", buf, body)
        elif flag == C.QUERY_OVERLAY_CARD:
            (n,) = struct.unpack_from("<I", buf, body)
            card["overlay"] = list(struct.unpack_from(f"<{n}I", buf, body + 4))
        elif flag == C.QUERY_COUNTERS:
            (n,) = struct.unpack_from("<I", buf, body)
            card["counters"] = list(struct.unpack_from(f"<{n}I", buf, body + 4))
        # other chunks (reason/equip/target card locations) are skipped by length
        pos += 2 + length


def parse_query_location(buf: bytes) -> list[dict | None]:
    """One entry per slot of the queried location: a dict of fields, or None for an empty slot."""
    if len(buf) < 4:
        return []
    out: list[dict | None] = []
    pos = 4
    while pos < len(buf):
        (length,) = struct.unpack_from("<H", buf, pos)
        if length == 0:
            out.append(None)
            pos += 2
            continue
        card, pos = _parse_card(buf, pos)
        out.append(card)
    return out
