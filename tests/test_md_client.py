"""MD client-side command reads (ygorl.md.client) on a synthetic IL2CPP heap laid out like metadata version 31."""

import struct

import pytest

from ygorl.md.client import (
    AURA_GROUPS,
    HIGHLIGHT_MASK,
    Client,
    CommandBit,
    Heap,
    Metadata,
    aura_kinds,
    decode_status,
    find_classes,
    mask_kinds,
)
from ygorl.md.memory import FIELD_SLOTS, PILES, BufferMemory, RawCard, RawDuel, RawPlayer

META_BASE, HEAP_BASE, HEAP_SIZE = 0x10000000, 0x20000000, 0x40000
STATIC = 0x10
# the live CardStatus of a chain candidate in S/T slot 7 (2026-10-23): five BitVector32 sections, then the data
LIVE_STATUS = bytes.fromhex("010000001f000100ff00060001000e0001000f000e000000")


class FakeGame:
    """global-metadata strings + a heap of Il2CppClass / FieldInfo / objects, readable through BufferMemory."""

    def __init__(self) -> None:
        self.names: list[str] = [""]
        self.heap = bytearray(HEAP_SIZE)
        self.top = 0x100
        self.klass: dict[str, int] = {}

    def string(self, s: str) -> None:
        if s not in self.names:
            self.names.append(s)

    def alloc(self, size: int) -> int:
        at = HEAP_BASE + self.top
        self.top += (size + 0xF) & ~0xF
        return at

    def put(self, addr: int, fmt: str, *values) -> None:
        struct.pack_into(fmt, self.heap, addr - HEAP_BASE, *values)

    def define(self, ns: str, name: str, fields: list[tuple[str, int, int]]) -> None:
        for s in (ns, name, *(f for f, _, _ in fields)):
            self.string(s)
        self.klass[(ns, name)] = (self.alloc(0x100), fields)

    def metadata(self) -> bytes:
        table = b"\0".join(n.encode() for n in self.names) + b"\0"
        head = bytearray(0x100)
        struct.pack_into("<ii", head, 0x18, 0x100, len(table))
        return bytes(head) + table

    def address(self, s: str) -> int:
        return Metadata(self.metadata(), META_BASE).address(s)

    def finish(self) -> BufferMemory:
        for (ns, name), (k, fields) in self.klass.items():
            self.put(k + 0x10, "<QQ", self.address(name), self.address(ns))
            array = self.alloc(0x20 * (len(fields) + 1))
            self.put(k + 0x80, "<Q", array)
            for i, (fname, off, attrs) in enumerate(fields):
                t = self.alloc(0x10)
                self.put(t + 8, "<I", attrs)
                self.put(array + 0x20 * i, "<QQQi", self.address(fname), t, k, off)
        return BufferMemory({META_BASE: self.metadata(), HEAP_BASE: bytes(self.heap)})

    def obj(self, ns: str, name: str, size: int = 0x100) -> int:
        at = self.alloc(size)
        self.put(at, "<Q", self.klass[(ns, name)][0])
        return at

    def int_list(self, values: list[int], ptr_size: int = 4) -> int:
        lst, arr = self.obj("System.Collections.Generic", "List`1"), self.alloc(0x20 + ptr_size * len(values))
        self.put(lst + 0x10, "<Qi", arr, len(values))
        self.put(arr + 0x20, f"<{len(values)}{'i' if ptr_size == 4 else 'Q'}", *values)
        return lst


D = "YgomGame.Duel"


def build_game():
    g = FakeGame()
    g.define(D, "CommandOperation", [("worker", 0x58, 0), ("<player>k__BackingField", 0x60, 0),
                                      ("<position>k__BackingField", 0x64, 0), ("<index>k__BackingField", 0x68, 0),
                                      ("commandMask", 0x6C, 0), ("cardID", 0x88, 0), ("commandList", 0xA0, 0)])  # fmt: skip
    g.define(D, "RunEffectWorker", [("step", 0x10, 0)])
    g.define(D, "DuelClient", [("step", 0x10, 0)])
    g.define(D, "CardRoot", [("<cardPool>k__BackingField", 0x68, 0), ("<cardPlane>k__BackingField", 0x78, 0),
                              ("<team>k__BackingField", 0x90, 0), ("position", 0x94, 0),
                              ("<index>k__BackingField", 0x98, 0), ("<cardId>k__BackingField", 0x9C, 0),
                              ("<uniqueId>k__BackingField", 0xA8, 0), ("m_Atk", 0xB4, 0), ("m_Def", 0xB8, 0),
                              ("<orgAtk>k__BackingField", 0xBC, 0)])  # fmt: skip
    g.define(D, "CardPlane", [("highlightEff", 0x88, 0), ("auraEffType", 0xB8, 0), ("cmdBitHighlight", 0, STATIC),
                               ("<cardRoot>k__BackingField", 0x140, 0)])  # fmt: skip
    g.define(D, "CardInstancePool", [("list", 0x38, 0), ("<isTerminated>k__BackingField", 0x49, 0)])
    g.define(D, "Dictionary`2", [("_buckets", 0x10, 0)])
    g.define(D, "CardSelectionList", [("m_DuelClient", 0x50, 0), ("m_GroupedDataList", 0xE8, 0),
                                       ("m_ListType", 0x248, 0)])  # fmt: skip
    g.define(D, "ListCardData", [("cardid", 0x18, 0), ("cardstatus", 0x30, 0)])
    g.define(D, "CardStatus", [("dynamicAnchorEffs", 0x10, 0)])  # the UI class of the same name: not this one
    g.define("", "CardStatus", [("player", 0x10, 0), ("position", 0x14, 0), ("index", 0x18, 0), ("face", 0x1C, 0),
                                ("turn", 0x20, 0), ("bitVec", 0x24, 0)])  # fmt: skip
    g.define("System.Collections.Generic", "List`1", [("_items", 0x10, 0), ("_size", 0x18, 0)])
    return g


def board(me=0, theirs_down=()):
    """Own hand 15836 (uid 68), 4998 (uid 4); own Monster Zone 1: 11169 (uid 30), S/T slot 7: 15839 (uid 78);
    the opponent has 11170 (uid 46) in Monster Zone 1 and the Set cards ``theirs_down`` (slot, cid, uid)."""
    empty = {z: () for z in PILES}

    def player(slots, hand, down=()):
        s = [None] * FIELD_SLOTS
        for i, cid, uid in slots:
            s[i] = RawCard(cid, uid, 0, face=1)
        for i, cid, uid in down:
            s[i] = RawCard(cid, uid, 0, face=0)
        piles = dict(empty, hand=tuple(RawCard(cid, uid, 0) for cid, uid in hand))
        return RawPlayer(lp=8000, slots=tuple(s), piles=piles, counts={z: len(piles[z]) for z in PILES})

    mine = player([(0, 11169, 30), (7, 15839, 78)], [(15836, 68), (4998, 4)])
    theirs = player([(0, 11170, 46)], [], theirs_down)
    players = (mine, theirs) if me == 0 else (theirs, mine)
    return RawDuel(turn=3, phase=2, turn_player=me, me=me, players=players, offsets=None)


def client_for(g: FakeGame) -> Client:
    mem = g.finish()
    return Client(mem, Metadata(g.metadata(), META_BASE), lambda: [(HEAP_BASE, HEAP_BASE + HEAP_SIZE)])


def test_command_bits_and_kinds():
    assert CommandBit.ACTION == 0x8 and CommandBit.SET == 0x80 and CommandBit.DECIDE == 0x1000
    assert mask_kinds(0x50) == {"summon", "mset"}
    assert mask_kinds(0x88) == {"activate", "sset"}
    assert mask_kinds(0x1) == {"attack"} and mask_kinds(0x600) == {"reposition"}
    assert HIGHLIGHT_MASK & CommandBit.LOOK == 0
    assert aura_kinds(0x7C) == {"sset", "reposition"}  # only group 1
    assert aura_kinds(0x18) == {"attack", "summon", "flip", "mset", "sset", "reposition"}
    # a group-3 aura may hide group 2 (mask 0x54 under aura 0x7e, run s30): it rules out nothing
    assert aura_kinds(0x88) == aura_kinds(0x7E) == aura_kinds(0xED)
    assert "activate" not in aura_kinds(0x19) and aura_kinds(0xED) >= {"activate", "attack"}
    assert aura_kinds(0) == set() and set(AURA_GROUPS) >= {0x7C, 0x7D, 0x18, 0x19, 0x7E, 0x7F, 0x88, 0x89, 0xED, 0xEE}


def test_classes_and_fields_from_metadata():
    g = build_game()
    mem = g.finish()
    found = find_classes(Heap(mem), Metadata(g.metadata(), META_BASE), [(HEAP_BASE, HEAP_BASE + HEAP_SIZE)],
                         [(D, "CardPlane"), ("", "CardStatus"), (D, "Missing")])  # fmt: skip
    assert set(found) == {"CardPlane", "CardStatus"}
    assert found["CardPlane"].fields == {
        "highlightEff": 0x88,
        "auraEffType": 0xB8,
        "<cardRoot>k__BackingField": 0x140,
    }  # the static one left out
    assert found["CardStatus"].address == g.klass[("", "CardStatus")][0]  # not YgomGame.Duel.CardStatus
    assert found["CardStatus"].fields["bitVec"] == 0x24


def test_metadata_needs_the_name():
    with pytest.raises(OSError):
        Metadata(build_game().metadata(), META_BASE).address("NoSuchClass")


def test_card_status_sections():
    g = build_game()
    at = g.alloc(0x20)
    g.heap[at - HEAP_BASE : at - HEAP_BASE + len(LIVE_STATUS)] = LIVE_STATUS
    c = client_for(g)
    assert decode_status(c.heap, at, c.cls("CardStatus")) == {"player": 0, "position": 7, "index": 0, "face": 0,
                                                               "turn": 0}  # fmt: skip


def put_open(g, player, location, index, cid, mask, commands):
    op = g.obj(D, "CommandOperation")
    g.put(op + 0x58, "<Q", g.obj(D, "RunEffectWorker"))
    g.put(op + 0x60, "<iiiI", player, location, index, mask)
    g.put(op + 0x88, "<i", cid)
    g.put(op + 0xA0, "<Q", g.int_list(commands))
    return op


def test_open_command():
    g = build_game()
    put_open(g, 0, 13, 1, 11169, 0x50, [4, 6])
    op = client_for(g).open_command(0)
    assert (op.player, op.location, op.index, op.cid, op.mask, op.commands) == (0, 13, 1, 11169, 0x50, (4, 6))
    assert op.kinds == {"summon", "mset"}


def test_open_command_closed_and_opponent():
    g = build_game()
    put_open(g, -1, -1, -1, 4998, 0x1, [0])
    assert client_for(g).open_command(0) is None
    g = build_game()
    put_open(g, 0, 0, 0, 0, 0, [])  # a new duel, nothing opened yet
    assert client_for(g).open_command(0) is None
    g = build_game()
    put_open(g, 1, 0, 0, 11170, 0x2, [1])
    op = client_for(g).open_command(0)
    assert op.player == 1 and op.cid == 0  # an opponent card: no card id


def test_a_failed_check_finds_the_object_again():
    g = build_game()
    old = put_open(g, 0, 13, 0, 15836, 0x80, [7])
    c = client_for(g)
    assert c.open_command(0).cid == 15836
    g.put(old, "<Q", 0)  # freed
    put_open(g, 0, 13, 1, 4998, 0x10, [4])
    c.heap.memory = c.heap.memory.__class__({META_BASE: g.metadata(), HEAP_BASE: bytes(g.heap)})
    assert c.open_command(0).cid == 4998


def live_pool(g):
    if not hasattr(g, "pool"):
        g.pool = g.obj(D, "CardInstancePool")
    return g.pool


def put_root(g, team, position, index, cid, uid, lit=False, aura=0, attack=0, defense=0, pool=None, back=True):
    """A card view; ``pool``: its pool (default the duel's live one); ``back``: its plane points back at it."""
    root, plane = g.obj(D, "CardRoot"), g.obj(D, "CardPlane", 0x150)
    g.put(root + 0x68, "<Q", pool if pool is not None else live_pool(g))
    g.put(root + 0x78, "<Q", plane)
    g.put(root + 0x90, "<iiii", team, position, index, cid)
    g.put(root + 0xA8, "<i", uid)
    g.put(root + 0xB4, "<iii", attack, defense, attack)
    g.put(plane + 0x88, "<Q", g.alloc(0x10) if lit else 0)
    g.put(plane + 0xB8, "<i", aura)
    g.put(plane + 0x140, "<Q", root if back else g.alloc(0x10))
    return root


def dead_pool(g):
    """The pool of an earlier duel: freed, its memory now another object."""
    return g.obj(D, "Dictionary`2")


def test_highlighted_own_cards_only():
    g = build_game()
    put_root(g, 0, 13, 0, 15836, 68, True, 0x7C)
    put_root(g, 0, 13, 1, 4998, 4, False, 0x18)  # not lit: its stale aura is not reported
    put_root(g, 0, 0, 0, 11169, 30, True, 0x18)
    put_root(g, 0, 7, 0, 15839, 78, True, 0x88)
    put_root(g, 1, 0, 0, 11170, 46, True, 0x18)  # the opponent's
    put_root(g, 0, 18, 0, 15836, 0, True, 0x7C)  # a pooled view
    put_root(g, 0, 13, 0, 12345, 68, True, 0x7C)  # a stale view: uid of another card
    put_root(g, 0, 13, 1, 4998, 4, True, 0x7E, pool=dead_pool(g))  # an earlier duel's view of the same card
    assert client_for(g).highlighted(board()) == {68: 0x7C, 30: 0x18, 78: 0x88}


def test_live_root_needs_its_plane_and_a_live_pool():
    g = build_game()
    live = put_root(g, 0, 0, 0, 11169, 30)
    stolen = put_root(g, 0, 0, 0, 11169, 30, back=False)  # its plane now belongs to another view
    freed = put_root(g, 0, 0, 0, 11169, 30, pool=dead_pool(g))
    ended = g.obj(D, "CardInstancePool")
    g.put(ended + 0x49, "<B", 1)
    terminated = put_root(g, 0, 0, 0, 11169, 30, pool=ended)
    c = client_for(g)
    c.cls("CardRoot")
    assert [c.live_root(r) for r in (live, stolen, freed, terminated)] == [True, False, False, False]


def test_field_stats_reads_face_up_monsters_of_both_players():
    g = build_game()
    put_root(g, 1, 0, 0, 11170, 46, attack=3000, defense=2400, pool=dead_pool(g))  # an earlier duel's view first
    put_root(g, 0, 0, 0, 11169, 30, attack=1100, defense=600)
    put_root(g, 1, 0, 0, 11170, 46, attack=2800, defense=2400)
    put_root(g, 0, 7, 0, 15839, 78, attack=0, defense=0)  # a Spell / Trap: not a monster
    put_root(g, 1, 2, 0, 11171, 50, attack=1900, defense=1200)  # the opponent's Set monster: never read
    put_root(g, 1, 3, 0, 11172, 52, attack=1000, defense=1000)  # a view at a place the board has no such card
    put_root(g, 0, 13, 0, 15836, 68, attack=1500, defense=500)  # a hand card
    stats = client_for(g).field_stats(board(theirs_down=((2, 11171, 50),)))
    assert stats == {30: (1100, 600), 46: (2800, 2400)}


def test_field_stats_skip_a_view_of_another_card():
    g = build_game()
    put_root(g, 1, 0, 0, 12345, 46, attack=500, defense=500)  # same uid, another card id
    put_root(g, 1, 1, 0, 11170, 46, attack=500, defense=500)  # same card, another zone
    assert client_for(g).field_stats(board()) == {}


def put_list_card(g, cid, player, position, index):
    item = g.obj(D, "ListCardData")
    g.put(item + 0x18, "<i", cid)
    bits = player | position << 1 | index << 6 | 1 << 14
    g.heap[item + 0x30 - HEAP_BASE : item + 0x48 - HEAP_BASE] = LIVE_STATUS[:20] + struct.pack("<I", bits)
    return item


def put_selection(g, list_type, items):
    sel = g.obj(D, "CardSelectionList", 0x260)
    g.put(sel + 0x50, "<Q", g.obj(D, "DuelClient"))
    g.put(sel + 0x248, "<i", list_type)
    g.put(sel + 0xE8, "<Q", g.int_list(items, ptr_size=8))
    return sel


def test_prompt_cards_in_list_order():
    g = build_game()
    group = g.int_list([put_list_card(g, 15836, 0, 13, 0)], ptr_size=8)  # a nested group
    put_selection(g, 6, [put_list_card(g, 15839, 0, 7, 0), group, put_list_card(g, 11170, 1, 0, 0)])
    cards = client_for(g).prompt_cards(board())
    assert [(c.cid, c.player, c.location, c.index, c.uid) for c in cards] == [(15839, 0, 7, 0, 78),
                                                                               (15836, 0, 13, 0, 68)]  # fmt: skip


def test_prompt_cards_need_an_activation_list():
    g = build_game()
    put_selection(g, 3, [put_list_card(g, 15839, 0, 7, 0)])
    assert client_for(g).prompt_cards(board()) is None
    g = build_game()
    put_selection(g, 7, [])
    assert client_for(g).prompt_cards(board()) == []
