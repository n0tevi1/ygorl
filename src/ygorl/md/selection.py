"""Which card MD's UI has open, read from memory (design 10, 「识别与执行的系统化」): no panel image recognition.

Opening a card (a hand card, a field card) stores it in a UI object of the game (IL2CPP heap, not ``duel.dll``) as
two adjacent u32: the card id and the duel uid of that very card. The object's address is not fixed, so it is
found at run time and cached (:class:`Selection`):

- scan the writable mappings for adjacent (card id, uid) pairs of the player's own hand cards: about ten hits, in
  1.6 s (the card objects themselves, a few copies, and the open card);
- after the next opening of a different card, the open card's address is the one hit whose pair changed; the
  others are card objects that do not move. The object also keeps the card opened before (a slot 0x78 further):
  it changes too, to the open slot's old pair, and is left out.

Once known, every opening is told by one read; a read that is no card of the player's (the object was freed, a
new duel) drops the address and the next openings calibrate again. Read-only, like :mod:`ygorl.md.memory`.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable

import numpy as np

from ygorl.md.memory import Memory, MemoryReadError, RawDuel

Pair = tuple[int, int]  # (card id, uid)
PREVIOUS = 0x78  # the open-card object keeps the card opened before this far after the open one
CHUNK = 1 << 24


def writable_regions(pid: int) -> list[tuple[int, int]]:
    out = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split()
            if parts[1].startswith("rw"):
                a, b = (int(v, 16) for v in parts[0].split("-"))
                out.append((a, b))
    return out


def scan_pairs(memory: Memory, regions: Iterable[tuple[int, int]], pairs: Iterable[Pair]) -> dict[int, Pair]:
    """Every 4-aligned address holding one of ``pairs`` as two adjacent little-endian u32."""
    keys = np.array([cid | (uid << 32) for cid, uid in pairs], dtype=np.uint64)
    hits: dict[int, Pair] = {}
    if not len(keys):
        return hits
    for a, b in regions:
        for x in range(a, b, CHUNK):
            try:
                buf = memory.read(x, min(CHUNK, b - x))
            except (MemoryReadError, OSError):
                continue
            for off in (0, 4):
                n = (len(buf) - off) // 8 * 8
                v = np.frombuffer(buf, dtype=np.uint64, count=n // 8, offset=off)
                for i in np.flatnonzero(np.isin(v, keys)):
                    hits[x + off + 8 * int(i)] = (int(v[i] & 0xFFFFFFFF), int(v[i] >> 32))
    return hits


def own_cards(raw: RawDuel) -> dict[int, int]:
    """uid -> card id of the player's own cards (piles and field): what an open card can be told as."""
    me = raw.players[raw.me]
    cards = {c.uid: c.cid for pile in me.piles.values() for c in pile if c.uid}
    cards.update({c.uid: c.cid for c in me.slots if c is not None and c.uid})
    return cards


class Selection:
    """The open card, read from memory; see the module docstring."""

    def __init__(self, memory: Memory, regions: Callable[[], Iterable[tuple[int, int]]]) -> None:
        self.memory = memory
        self.regions = regions
        self.address: int | None = None
        self.candidates: dict[int, Pair] = {}
        self.last: dict = {}  # the last calibration scan, for the trace

    def read(self, raw: RawDuel) -> Pair | None:
        """The open card (card id, uid) when the address is known and holds one of the player's cards, else None
        (and the address is forgotten)."""
        if self.address is None:
            return None
        try:
            cid, uid = struct.unpack("<II", self.memory.read(self.address, 8))
        except (MemoryReadError, OSError):
            cid, uid = 0, 0
        if own_cards(raw).get(uid) == cid and cid:
            return cid, uid
        self.address = None
        return None

    def opened(self, raw: RawDuel) -> Pair | None:
        """Call after a hand card was opened: the open card, or None while still calibrating (the first opening
        of a duel, or one that showed the same card as the last)."""
        got = self.read(raw)
        if got is not None:
            return got
        hand = {(c.cid, c.uid) for c in raw.players[raw.me].piles["hand"] if c.uid}
        found = scan_pairs(self.memory, self.regions(), hand)
        changed = [a for a, p in found.items() if a in self.candidates and self.candidates[a] != p]
        # the object keeps the card opened before as well: that slot's new value is the open slot's old one
        before = {self.candidates[a] for a in changed}
        kept = [a for a in changed if found[a] not in before] or changed
        # both slots can hold the same card (one card opened twice): the open slot is the one 0x78 below the other
        kept = [a for a in kept if a - PREVIOUS not in kept]
        self.last = {"hits": {hex(a): list(p) for a, p in found.items()},
                     "changed": [hex(a) for a in changed], "kept": [hex(a) for a in kept]}  # fmt: skip
        changed = kept
        self.candidates = found
        if len(changed) == 1:
            self.address = changed[0]
            self.candidates = {}
            return found[self.address]
        return None

    def calibrated(self) -> bool:
        return self.address is not None


def for_process(memory: Memory, pid: int) -> Selection:
    return Selection(memory, lambda: writable_regions(pid))


__all__ = ["Selection", "for_process", "own_cards", "scan_pairs", "writable_regions"]
