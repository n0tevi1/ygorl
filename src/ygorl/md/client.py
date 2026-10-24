"""What MD's own client knows about legal commands, read from its IL2CPP heap (design 10, 「客户端的命令掩码」).

``duel.dll`` stores no command mask: ``DLL_DuelComGetCommandMask`` runs the card's own condition code on every call.
The C# client calls it and keeps the results in its UI objects, and those can be read:

- ``CommandOperation``: the card opened right now (player, location, index, card id), its exact command mask and
  its commands in icon order (:meth:`Client.open_command`);
- ``CardRoot`` -> ``CardPlane``: every card MD highlights while it waits for input. The highlight is set from
  ``mask & HIGHLIGHT_MASK``; its "aura" names which of three command groups the mask holds (:meth:`Client.highlighted`,
  :func:`aura_kinds`). In an open / chain prompt only the cards that can be activated are lit;
- ``CardSelectionList``: the cards an open / chain prompt offers, in list order (:meth:`Client.prompt_cards`);
- ``CardRoot.m_Atk`` / ``m_Def``: a face-up monster's current ATK / DEF as the board shows them, equips and other
  continuous effects included (:meth:`Client.field_stats`). ``duel.dll`` stores no current value either: its
  ``DLL_DuelGetCardBasicVal`` recomputes it from the effect lists on every call, and the client keeps the result.

Views of earlier duels stay on the heap and may carry the same unique id and card id as a live one (uids repeat when
the same decks meet again); a view is live only while its ``CardPlane`` points back at it and its ``CardInstancePool``
is a live pool (:meth:`Client.live_root`).

Managed field offsets are not hard-coded. Class names live in the mapped ``global-metadata.dat``; the runtime
``Il2CppClass`` is the one whose name / namespace pointers point at them, and its ``FieldInfo`` array gives every
field's offset by name. Only the layout of the IL2CPP runtime structs themselves (metadata version 31) is fixed here.
Classes are found once per process; instances are cached, checked on every read and searched again when a check
fails (a new duel, a freed object). Read-only, like :mod:`ygorl.md.memory`. Opponent cards are never returned.
"""

from __future__ import annotations

import argparse
import struct
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import numpy as np

from ygorl.md.memory import FIELD_SLOTS, PILES, Memory, MemoryReadError, RawDuel
from ygorl.md.selection import own_cards, writable_regions

# --- IL2CPP runtime layout (metadata version 31, Unity 2022) --------------------------------------------------------
METADATA_NAME = "global-metadata.dat"
META_STRINGS = 0x18  # header: (offset, size) of the string table
OBJECT_HEADER = 0x10  # klass*, monitor*; a boxed value type's fields start after it too
CLASS_NAME = 0x10  # Il2CppClass: const char* name
CLASS_NAMESPACE = 0x18  # const char* namespaze
CLASS_FIELDS = 0x80  # FieldInfo* fields
FIELD_INFO_SIZE = 0x20  # FieldInfo: name*, type*, parent*, i32 offset, u32 token
FIELD_TYPE, FIELD_PARENT, FIELD_OFFSET = 0x8, 0x10, 0x18
TYPE_ATTRS = 0x8  # Il2CppType: u32 attrs (low 16 bits: field attributes)
FIELD_STATIC, FIELD_LITERAL = 0x10, 0x40
MAX_FIELDS = 512
LIST_ITEMS, LIST_SIZE = 0x10, 0x18  # System.Collections.Generic.List<T>: T[] _items, int _size
ARRAY_DATA = 0x20  # T[]: klass, monitor, bounds, max_length, then the elements
CHUNK = 1 << 24
DUEL = "YgomGame.Duel"


class CommandBit(IntFlag):
    """MD's ``YgomGame.Duel.CommandBit`` (bit n = ``Command`` n), the bits of ``DLL_DuelComGetCommandMask``."""

    ATTACK = 0x1
    LOOK = 0x2
    SPSUMMON = 0x4  # Special Summon by the card's own procedure
    ACTION = 0x8  # activate (a card or an effect)
    SUMMON = 0x10  # Normal / Tribute Summon
    REVERSE = 0x20  # Flip Summon
    SETMONST = 0x40  # Set a monster
    SET = 0x80  # Set a Spell / Trap
    PENDULUM = 0x100  # activate as a Pendulum Scale
    TURNATK = 0x200  # change to Attack Position
    TURNDEF = 0x400  # change to Defense Position
    SURRENDER = 0x800
    DECIDE = 0x1000  # select (a card choice without a list, input mode 8)
    DRAW = 0x2000


# EffectTaskWaitInput.cmdBitHighlight (a literal in global-metadata): the bits that light a card
HIGHLIGHT_MASK = 0x7FD
# command kinds (the names the player uses) per bit
KIND = {CommandBit.ATTACK: "attack", CommandBit.SPSUMMON: "spsummon", CommandBit.ACTION: "activate",
        CommandBit.SUMMON: "summon", CommandBit.REVERSE: "flip", CommandBit.SETMONST: "mset", CommandBit.SET: "sset",
        CommandBit.PENDULUM: "activate", CommandBit.TURNATK: "reposition", CommandBit.TURNDEF: "reposition"}  # fmt: skip
# CardPlace.commandPowerList (read live, the same on every place): each command's group, 0 = never lights
POWER = {CommandBit.ATTACK: 2, CommandBit.SPSUMMON: 3, CommandBit.ACTION: 3, CommandBit.SUMMON: 2,
         CommandBit.REVERSE: 2, CommandBit.SETMONST: 2, CommandBit.SET: 1, CommandBit.PENDULUM: 3,
         CommandBit.TURNATK: 1, CommandBit.TURNDEF: 1}  # fmt: skip
# CardPlane.auraEffType (an effect id) -> the groups present. CardPlace.GetActivateAura picks 1 (only group 1),
# 3 (group 2, maybe 1), 5 (group 3, maybe 1) or 7 (2 and 3), +1 for the open card; CardPlane.ReqHighlight maps
# them to these ids (5 / 6 have two ids each)
# (a group-3 aura can hide group 2 too: Apprentice Illusion Magician offered Special Summon, Summon and Set,
# mask 0x54, under aura 0x7e, run s30; so 0x7e / 0x7f / 0x88 / 0x89 rule out nothing but are kept as group 3)
AURA_GROUPS = {0x7C: {1}, 0x7D: {1}, 0x18: {1, 2}, 0x19: {1, 2}, 0x7E: {1, 2, 3}, 0x7F: {1, 2, 3}, 0x88: {1, 2, 3},
               0x89: {1, 2, 3}, 0xED: {1, 2, 3}, 0xEE: {1, 2, 3}}  # fmt: skip
# the group a card certainly has: the one that set the aura (group 1 may come along with 2 or 3)
AURA_SURE = {0x7C: 1, 0x7D: 1, 0x18: 2, 0x19: 2, 0x7E: 3, 0x7F: 3, 0x88: 3, 0x89: 3}
# CardSelectionList.m_ListType of the prompts that offer activations (CreateActivableList)
LIST_CHAIN, LIST_CHECK_TIMING = 6, 7
FIRST_PILE = 13  # MD locations: 0-12 field slots, 13 hand, then PILES
POOLED = 18  # CardRoot.position of a view in the pool (no card)
MONSTER_ZONES = 7  # MD locations 0-6: Main and Extra Monster Zones


def mask_kinds(mask: int) -> set[str]:
    """The command kinds a command mask allows."""
    return {kind for bit, kind in KIND.items() if mask & bit}


def aura_kinds(aura: int) -> set[str]:
    """The command kinds a lit card may have (a superset: the aura only names groups); empty for an unknown aura."""
    groups = AURA_GROUPS.get(aura, set())
    return {KIND[bit] for bit, g in POWER.items() if g in groups}


# --- metadata and runtime classes -----------------------------------------------------------------------------------


class Metadata:
    """``global-metadata.dat`` as mapped in the game: a class name's address is where the runtime points at it."""

    def __init__(self, data: bytes, base: int) -> None:
        self.base = base
        self.strings, size = struct.unpack_from("<ii", data, META_STRINGS)
        self.table = data[self.strings : self.strings + size]

    def address(self, name: str) -> int:
        """Address of ``name`` in the string table (the first exact match)."""
        at = (b"\0" + self.table).find(b"\0" + name.encode() + b"\0")
        if at < 0:
            raise MemoryReadError(2, f"{name!r} is not in {METADATA_NAME}")
        return self.base + self.strings + at

    @classmethod
    def for_process(cls, pid: int) -> Metadata:
        for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].endswith("/" + METADATA_NAME) and int(parts[2], 16) == 0:
                return cls(Path(parts[5]).read_bytes(), int(parts[0].split("-")[0], 16))
        raise MemoryReadError(2, f"{METADATA_NAME} is not mapped in process {pid}")


def scan_qwords(memory: Memory, regions: Iterable[tuple[int, int]], values: Iterable[int]) -> list[tuple[int, int]]:
    """(address, value) of every 8-aligned u64 equal to one of ``values``."""
    keys = np.array(sorted(set(values)), dtype=np.uint64)
    hits: list[tuple[int, int]] = []
    for a, b in regions:
        for x in range(a, b, CHUNK):
            try:
                buf = memory.read(x, min(CHUNK, b - x))
            except (MemoryReadError, OSError):
                continue
            v = np.frombuffer(buf, dtype=np.uint64, count=len(buf) // 8)
            hits += [(x + 8 * int(i), int(v[i])) for i in np.flatnonzero(np.isin(v, keys))]
    return hits


@dataclass(frozen=True)
class Class:
    address: int  # the Il2CppClass
    name: str
    fields: dict[str, int]  # instance field -> offset in the object (OBJECT_HEADER included)


class Heap:
    """Typed reads of the managed heap."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def u64(self, a: int) -> int:
        return struct.unpack("<Q", self.memory.read(a, 8))[0]

    def i32(self, a: int) -> int:
        return struct.unpack("<i", self.memory.read(a, 4))[0]

    def u32(self, a: int) -> int:
        return struct.unpack("<I", self.memory.read(a, 4))[0]

    def cstr(self, a: int) -> str:
        for n in (128, 32, 8, 1):
            try:
                raw = self.memory.read(a, n)
            except (MemoryReadError, OSError):
                continue
            if b"\0" in raw or n == 1:
                return raw.split(b"\0")[0].decode("latin1")
        return ""

    def class_name(self, obj: int) -> str:
        """Name of an object's class ('' when ``obj`` is no object)."""
        try:
            return self.cstr(self.u64(self.u64(obj) + CLASS_NAME)) if obj else ""
        except (MemoryReadError, OSError):
            return ""

    def list_items(self, lst: int, limit: int = 400) -> list[int]:
        """The u64 elements of a ``List<T>`` of references."""
        items, n = self.u64(lst + LIST_ITEMS), self.i32(lst + LIST_SIZE)
        if not items or n <= 0:
            return []
        data = self.memory.read(items + ARRAY_DATA, 8 * min(n, limit))
        return list(struct.unpack(f"<{len(data) // 8}Q", data))

    def int_list(self, lst: int, limit: int = 64) -> list[int]:
        """The elements of a ``List<int>`` (or of an int enum)."""
        items, n = self.u64(lst + LIST_ITEMS), self.i32(lst + LIST_SIZE)
        if not items or n <= 0:
            return []
        data = self.memory.read(items + ARRAY_DATA, 4 * min(n, limit))
        return list(struct.unpack(f"<{len(data) // 4}i", data))

    def fields(self, klass: int) -> dict[str, int]:
        """Instance fields of a runtime class, by name (static and literal fields left out)."""
        array = self.u64(klass + CLASS_FIELDS)
        out: dict[str, int] = {}
        for i in range(MAX_FIELDS):
            info = array + i * FIELD_INFO_SIZE
            try:
                if self.u64(info + FIELD_PARENT) != klass:
                    break
                attrs = self.u32(self.u64(info + FIELD_TYPE) + TYPE_ATTRS)
                if not attrs & (FIELD_STATIC | FIELD_LITERAL):
                    out[self.cstr(self.u64(info))] = self.i32(info + FIELD_OFFSET)
            except (MemoryReadError, OSError):
                break
        return out


def find_classes(heap: Heap, metadata: Metadata, regions: Iterable[tuple[int, int]],
                 names: Iterable[tuple[str, str]]) -> dict[str, Class]:  # fmt: skip
    """Runtime classes by (namespace, name), keyed by name; a class not in memory (not loaded yet) is left out."""
    wanted: dict[int, dict[str, str]] = {}  # name address -> namespace -> name
    for ns, name in names:
        try:
            wanted.setdefault(metadata.address(name), {})[ns] = name
        except MemoryReadError:
            continue  # not in this build's metadata
    found: dict[str, Class] = {}
    for at, value in scan_qwords(heap.memory, regions, wanted):
        klass = at - CLASS_NAME
        try:
            name = wanted[value].get(heap.cstr(heap.u64(klass + CLASS_NAMESPACE)))
        except (MemoryReadError, OSError):
            continue
        if name is None or name in found:
            continue
        try:
            fields = heap.fields(klass)
        except (MemoryReadError, OSError):
            continue
        if fields:
            found[name] = Class(klass, name, fields)
    return found


# --- what the client holds ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenCommand:
    """The card MD has open (its command icons shown) and what it may do."""

    player: int
    location: int  # MD location: 0-12 field slot, 13 hand, 14 Extra Deck, 15 Deck, 16 GY, 17 banished
    index: int
    cid: int  # MD card id
    mask: int  # CommandBit
    commands: tuple[int, ...]  # Command (bit numbers) in icon order

    @property
    def kinds(self) -> set[str]:
        return mask_kinds(self.mask)


@dataclass(frozen=True)
class PromptCard:
    """A card an open / chain prompt offers to activate, in the prompt list's order."""

    cid: int
    player: int
    location: int  # MD location, as in OpenCommand
    index: int
    uid: int  # 0 when the board read does not hold that place


def decode_status(heap: Heap, at: int, status: Class) -> dict[str, int]:
    """A ``CardStatus`` value at ``at``: BitVector32 sections (i16 mask, i16 offset) named player / position / index /
    face / turn, followed by the packed data ``bitVec``."""
    rel = {k: v - OBJECT_HEADER for k, v in status.fields.items()}
    bits = heap.u32(at + rel["bitVec"])
    out = {}
    for name in ("player", "position", "index", "face", "turn"):
        mask, shift = struct.unpack("<hh", heap.memory.read(at + rel[name], 4))
        out[name] = (bits >> shift) & mask
    return out


def uid_at(raw: RawDuel, player: int, location: int, index: int) -> int:
    pl = raw.players[player & 1]
    if 0 <= location < FIELD_SLOTS:
        card = pl.slots[location]
        return card.uid if card is not None else 0
    if FIRST_PILE <= location < FIRST_PILE + len(PILES):
        pile = pl.piles[PILES[location - FIRST_PILE]]
        return pile[index].uid if 0 <= index < len(pile) else 0
    return 0


CLASSES = (
    (DUEL, "CommandOperation"),
    (DUEL, "RunEffectWorker"),
    (DUEL, "CardRoot"),
    (DUEL, "CardPlane"),
    (DUEL, "CardInstancePool"),
    (DUEL, "CardSelectionList"),
    (DUEL, "ListCardData"),
    ("", "CardStatus"),
)


class Client:
    """MD's client-side view of legal commands; see the module docstring."""

    def __init__(self, memory: Memory, metadata: Metadata, regions: Callable[[], Iterable[tuple[int, int]]],
                 rescan_every: float = 5.0) -> None:  # fmt: skip
        self.heap = Heap(memory)
        self.metadata = metadata
        self.regions = regions
        self.rescan_every = rescan_every
        self.classes: dict[str, Class] = {}
        self.instances: dict[str, list[int]] = {}
        self.scanned: dict[str, float] = {}

    @classmethod
    def for_process(cls, memory: Memory, pid: int) -> Client:
        return cls(memory, Metadata.for_process(pid), lambda: writable_regions(pid))

    # -- discovery --

    def cls(self, name: str) -> Class | None:
        if name not in self.classes:
            self.classes.update(find_classes(self.heap, self.metadata, self.regions(), CLASSES))
        return self.classes.get(name)

    def _valid(self, name: str, obj: int) -> bool:
        k = self.classes[name]
        f, h = k.fields, self.heap
        try:
            if h.u64(obj) != k.address:
                return False
            if name == "CommandOperation":
                return h.class_name(h.u64(obj + f["worker"])) == "RunEffectWorker"
            if name == "CardSelectionList":
                return h.class_name(h.u64(obj + f["m_DuelClient"])) == "DuelClient"
            if name == "CardRoot":
                return h.class_name(h.u64(obj + f["<cardPlane>k__BackingField"])) == "CardPlane"
        except (MemoryReadError, OSError):
            return False
        return True

    def objects(self, name: str, refresh: bool = False) -> list[int]:
        """Live instances of a class (cached; searched again when one fails its check, or on ``refresh`` at most
        every ``rescan_every`` seconds)."""
        if self.cls(name) is None:
            return []
        cached = self.instances.get(name)
        stale = cached is None or not all(self._valid(name, a) for a in cached)
        now = time.monotonic()
        if stale or (refresh and now - self.scanned.get(name, 0.0) >= self.rescan_every):
            hits = scan_qwords(self.heap.memory, self.regions(), [self.classes[name].address])
            self.instances[name] = [a for a, _ in hits if self._valid(name, a)]
            self.scanned[name] = now
        return self.instances[name]

    def live_root(self, root: int) -> bool:
        """A ``CardRoot`` of the running duel: its ``CardPlane`` points back at it and its ``CardInstancePool`` is
        live (not terminated). A view of an earlier duel fails one of them (its plane was given to another view, or
        its pool was freed and the memory reused)."""
        if self.cls("CardPlane") is None or self.cls("CardInstancePool") is None:
            return False
        r, p = self.classes["CardRoot"].fields, self.classes["CardPlane"].fields
        pool_class = self.classes["CardInstancePool"]
        back, terminated = p.get("<cardRoot>k__BackingField"), pool_class.fields.get("<isTerminated>k__BackingField")
        if back is None or terminated is None or "<cardPool>k__BackingField" not in r:
            return False
        h = self.heap
        try:
            plane = h.u64(root + r["<cardPlane>k__BackingField"])
            if not plane or h.u64(plane) != self.classes["CardPlane"].address or h.u64(plane + back) != root:
                return False
            pool = h.u64(root + r["<cardPool>k__BackingField"])
            return bool(pool) and h.u64(pool) == pool_class.address and not h.memory.read(pool + terminated, 1)[0]
        except (MemoryReadError, OSError):
            return False

    # -- reads --

    def open_command(self, me: int) -> OpenCommand | None:
        """The open card and its exact command mask; None when no card is open. An opponent card is reported
        without its card id (it may be face-down)."""
        objs = self.objects("CommandOperation")
        if not objs:
            return None
        f, h, op = self.classes["CommandOperation"].fields, self.heap, objs[0]
        player, cid = h.i32(op + f["<player>k__BackingField"]), h.i32(op + f["cardID"])
        if player < 0 or not cid:
            return None  # closed (-1), or not opened yet in this duel (all zero)
        commands = h.u64(op + f["commandList"])
        return OpenCommand(player=player, location=h.i32(op + f["<position>k__BackingField"]),
                           index=h.i32(op + f["<index>k__BackingField"]),
                           cid=cid if player == me else 0, mask=h.u32(op + f["commandMask"]),
                           commands=tuple(h.int_list(commands)) if commands else ())  # fmt: skip

    def highlighted(self, raw: RawDuel) -> dict[int, int]:
        """uid -> aura (``CardPlane.auraEffType``) of every own card MD highlights now (see :func:`aura_kinds`).
        Only cards whose (uid, card id) is one of the player's own on the board read ``raw`` count."""
        if self.cls("CardRoot") is None or self.cls("CardPlane") is None:
            return {}
        own = own_cards(raw)
        lit: dict[int, int] = {}
        for refresh in (False, True):
            roots = self.objects("CardRoot", refresh=refresh)
            seen = set()
            r, p = self.classes["CardRoot"].fields, self.classes["CardPlane"].fields
            for root in roots:
                try:
                    uid = self.heap.i32(root + r["<uniqueId>k__BackingField"])
                    if not 0 <= self.heap.i32(root + r["position"]) < POOLED:
                        continue  # a pooled view, holding no card
                    if own.get(uid) != self.heap.i32(root + r["<cardId>k__BackingField"]) or not uid:
                        continue
                    if not self.live_root(root):
                        continue  # a view of an earlier duel with the same uid and card id
                    seen.add(uid)
                    plane = self.heap.u64(root + r["<cardPlane>k__BackingField"])
                    if self.heap.u64(plane + p["highlightEff"]):
                        lit[uid] = self.heap.i32(plane + p["auraEffType"])
                except (MemoryReadError, OSError):
                    continue
            me = raw.players[raw.me]
            shown = {c.uid for c in me.piles["hand"]} | {c.uid for c in me.slots if c is not None}
            if shown <= seen:
                break  # every own hand / field card has its view (else a new one was pooled: search again)
        return lit

    def field_stats(self, raw: RawDuel) -> dict[int, tuple[int, int]]:
        """uid -> (ATK, DEF) MD shows for every face-up monster in a Monster Zone, both players' (public
        information). Only live views (:meth:`live_root`) whose player, zone, unique id and card id match the board
        read ``raw`` count; a face-down card is never read. The views follow the duel through the client's effect
        queue: read while MD waits for input, when the queue is drained."""
        if self.cls("CardRoot") is None:
            return {}
        r, h = self.classes["CardRoot"].fields, self.heap
        if "m_Atk" not in r or "m_Def" not in r:
            return {}
        faceup = raw.offsets.semantics["faceup"] if raw.offsets is not None else 1
        wanted = {}
        for p, pl in enumerate(raw.players):
            for i, card in enumerate(pl.slots[:MONSTER_ZONES]):
                if card is not None and card.uid and card.face & faceup:
                    wanted[card.uid] = (p, i, card.cid)
        out: dict[int, tuple[int, int]] = {}
        for refresh in (False, True):
            for root in self.objects("CardRoot", refresh=refresh):
                try:
                    uid = h.i32(root + r["<uniqueId>k__BackingField"])
                    if uid not in wanted or uid in out:
                        continue
                    team, position, cid = wanted[uid]
                    if (h.i32(root + r["<team>k__BackingField"]), h.i32(root + r["position"]),
                            h.i32(root + r["<cardId>k__BackingField"])) != (team, position, cid):  # fmt: skip
                        continue
                    if not self.live_root(root):
                        continue
                    out[uid] = (h.i32(root + r["m_Atk"]), h.i32(root + r["m_Def"]))
                except (MemoryReadError, OSError):
                    continue
            if set(out) == set(wanted):
                break  # every face-up monster has its view (else a new one was made: search again)
        return out

    def prompt_cards(self, raw: RawDuel) -> list[PromptCard] | None:
        """The cards of the open / chain prompt list in list order (own cards only); None when the list is no
        activation list. The list keeps its last content after the prompt: only meaningful while MD's request is
        an open / chain prompt (``ygorl.md.request`` kinds "open" / "chain")."""
        if self.cls("ListCardData") is None or self.cls("CardStatus") is None:
            return None
        h, f = self.heap, self.classes["CardSelectionList"].fields
        for sel in self.objects("CardSelectionList"):
            if h.i32(sel + f["m_ListType"]) not in (LIST_CHAIN, LIST_CHECK_TIMING):
                continue
            grouped = h.u64(sel + f["m_GroupedDataList"])
            return self._cards(self._data(grouped), raw) if grouped else []
        return None

    def list_cards(self, raw: RawDuel, places: bool = False) -> tuple[int, list[PromptCard]] | None:
        """(list type, cards) of the card list MD's client holds now, any list type (a select list of a search,
        a cost, a target), own cards in list order; None when there is none. Observed alongside the player's own
        tick-and-read identification until its order is known to be the screen's (design 10)."""
        if self.cls("ListCardData") is None or self.cls("CardStatus") is None:
            return None
        h, f = self.heap, self.classes["CardSelectionList"].fields
        for sel in self.objects("CardSelectionList"):
            grouped = h.u64(sel + f["m_GroupedDataList"])
            return h.i32(sel + f["m_ListType"]), self._cards(self._data(grouped), raw, places) if grouped else []
        return None

    def _data(self, lst: int, depth: int = 0) -> list[int]:
        """ListCardData objects of a list, nested lists (groups) flattened in order."""
        out = []
        for item in self.heap.list_items(lst):
            name = self.heap.class_name(item)
            if name == "ListCardData":
                out.append(item)
            elif name.startswith("List`") and depth < 2:
                out += self._data(item, depth + 1)
        return out

    def _cards(self, items: list[int], raw: RawDuel, places: bool = False) -> list[PromptCard]:
        """Own cards of a list; ``places``: the opponent's entries kept too, as places only (card id and uid 0:
        their card ids are never read), so positions in the list stay the screen's."""
        d, status = self.classes["ListCardData"].fields, self.classes["CardStatus"]
        out = []
        for item in items:
            s = decode_status(self.heap, item + d["cardstatus"], status)
            if s["player"] != raw.me:
                if places:
                    out.append(PromptCard(0, s["player"], s["position"], s["index"], 0))
                continue
            uid = uid_at(raw, s["player"], s["position"], s["index"])
            out.append(PromptCard(self.heap.i32(item + d["cardid"]), s["player"], s["position"], s["index"], uid))
        return out


# --- CLI ------------------------------------------------------------------------------------------------------------


def _describe(client: Client, raw: RawDuel, kind: str) -> list[str]:
    own = own_cards(raw)
    lines = [f"request {kind}  me {raw.me}"]
    for uid, aura in sorted(client.highlighted(raw).items()):
        lines.append(f"  lit uid {uid:3d} cid {own[uid]:5d} aura {aura:#x} kinds {sorted(aura_kinds(aura))}")
    for uid, (attack, defense) in sorted(client.field_stats(raw).items()):
        lines.append(f"  face-up monster uid {uid:3d} ATK {attack} DEF {defense}")
    op = client.open_command(raw.me)
    if op is not None:
        lines.append(f"  open p{op.player} loc {op.location} idx {op.index} cid {op.cid} mask {op.mask:#x} "
                     f"{sorted(op.kinds)} icons {list(op.commands)}")  # fmt: skip
    cards = client.prompt_cards(raw) if kind in ("open", "chain") else None
    if cards is not None:
        lines.append("  prompt " + ", ".join(f"cid {c.cid} loc {c.location}/{c.index} uid {c.uid}" for c in cards))
    return lines


def main(argv: list[str] | None = None) -> None:
    from ygorl.md.memory import attach
    from ygorl.md.request import read_request

    ap = argparse.ArgumentParser(description="Print MD's client-side commands (own side only, read-only).")
    ap.add_argument("--watch", type=float, default=0.0, help="keep printing changes for this many seconds")
    ap.add_argument("--interval", type=float, default=0.3)
    args = ap.parse_args(argv)
    game = attach()
    client = Client.for_process(game.memory, game.pid)
    end, last = time.monotonic() + args.watch, None
    while True:
        try:
            raw = game.read()
            req = read_request(game.memory, game.module.base, game.offsets, raw)
            lines = _describe(client, raw, req.kind if req.accepting else "none")
        except (MemoryReadError, OSError) as exc:
            lines = [f"no read: {exc}"]
        if lines != last:
            print(f"[{time.strftime('%H:%M:%S')}] " + "\n".join(lines), flush=True)
            last = lines
        if time.monotonic() >= end:
            break
        time.sleep(args.interval)


__all__ = ["AURA_GROUPS", "HIGHLIGHT_MASK", "Client", "CommandBit", "Metadata", "OpenCommand", "PromptCard",
           "aura_kinds", "decode_status", "find_classes", "mask_kinds", "scan_qwords"]  # fmt: skip

if __name__ == "__main__":
    main()
