"""Master Duel memory reader (design 10, T7.6): offset derivation, reading, masking, snapshot conversion.

The game binary is not in the repository: derivation is checked against the installed client when there is one.
Everything else runs on synthetic memory laid out by the derived offsets of build 23020425.
"""

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ygorl.advisor.mdmem import BufferMemory, decode_uid, find_module, format_raw, read_duel, steam_buildid, to_snapshot
from ygorl.advisor.mdoffsets import Offsets, OffsetError, derive, parse_objdump
from ygorl.advisor.snapshot import UNKNOWN, SnapshotError

INSTALLED = Path.home() / ".steam/steam/steamapps/common/Yu-Gi-Oh!  Master Duel/masterduel_Data/Plugins/x86_64/duel.dll"
BUILD_23020425 = "97bd4d136e39b0872e4a8a9171632f1f0bd9bb04d69af08e836c56e684d43c44"

# derived from build 23020425 (checked by hand against the disassembly, see design 10)
OFF = Offsets(state_ptr=0x11F9C50, key_ptr=0x11F9C48, my_player=0x4, player_type=0x8, duel_finish=0x334, duel_result=0x330, turn=0x3C14, turn_player=0x3C08, phase=0x3C1C,
              player_stride=0xDDC, lp=0x0,
              count={"hand": 0xC, "extra": 0x18, "deck": 0x10, "grave": 0x14, "removed": 0x1C},
              pile={"hand": 0x1E4, "extra": 0x5A4, "deck": 0x3C4, "grave": 0x7FC, "removed": 0xA54}, pile_stride=4,
              slot_base=0x5C, slot_stride=0x1C, slot_uid=0x2, slot_turn=0x6, slot_face=0x7, slot_chain=0xA,
              removed_face=0xCAC, removed_facedown=0x80, uid_table=0x1BEC, uid_mask=0x1FF, effect_table=0x2BEC, effect_stride=0x10, effect_kind=0x2,
              effect_uid=0x4, effect_next=0xC, overlay_code=0x3FB, overlay_kind=0x5)  # fmt: skip
# read off the screen in a Solo duel on this build (design 10)
SEM = {
    "faceup": 0x01,
    "defense": 0x01,
    "locked_byte": 0x14,
    "locked": 0x40,
    "cpu_type": 1,
    "turn_base": 1,
    "normal_summons": 0x30,
}

BEWD, DM, ASH, ZEUS, MAT = 4007, 4041, 13587, 16210, 4844  # MD card ids
PASSWORDS = {BEWD: 89631139, DM: 46986414, ASH: 14558127, ZEUS: 90448279, MAT: 70095154}


def test_parse_objdump_resolves_rip_and_branch_targets():
    text = """
   180ea9d30:	mov    rax,QWORD PTR [rip+0x34ff19]        # 0x1811f9c50
   180ea9d37:	mov    eax,DWORD PTR [rax+0x3c14]
   18059a341:	je     0x18059a3d2
   180ea9d3d:	ret
"""
    a, b, c, d = parse_objdump(text)
    assert (a.op, a.target) == ("mov", 0x1811F9C50)
    assert b.args == "eax,dword ptr [rax+0x3c14]" and b.target is None
    assert (c.op, c.target) == ("je", 0x18059A3D2)
    assert d.op == "ret"


def test_offsets_json_round_trip_and_verified_needs_semantics():
    off = replace(OFF, dll_sha256="ab" * 32, verified=True, semantics=SEM)
    assert Offsets.from_json(off.to_json()) == off
    with pytest.raises(OffsetError, match="semantics"):
        Offsets.from_json({**off.to_json(), "semantics": {"faceup": "0x1"}})


def test_recorded_offsets_match_the_derivation():
    """The committed file for build 23020425 is the derivation plus the screen-checked semantics."""
    from ygorl.advisor.mdoffsets import offsets_file

    rec = Offsets.from_json(__import__("json").loads(offsets_file(BUILD_23020425).read_text()))
    assert rec.verified and rec.semantics == SEM
    assert replace(rec, dll_sha256="", buildid="", verified=False, semantics={}, notes=[]) == OFF


@pytest.mark.skipif(not INSTALLED.is_file(), reason="Master Duel is not installed")
def test_derive_matches_the_hand_checked_layout():
    off = derive(INSTALLED)
    if off.dll_sha256 != BUILD_23020425:
        pytest.skip("the installed client is another build: derivation ran, layout may legitimately differ")
    assert replace(off, dll_sha256="") == OFF


def test_decode_uid():
    assert decode_uid(0x0000) == 0 and decode_uid(0x0001) == 1
    assert decode_uid(0x0300) == 6 and decode_uid(0x0341) == 7  # 0x40 is a flag, not part of the uid


# --- synthetic memory -------------------------------------------------------------------------------------------

BASE, STATE, KEY = 0x7F0000000000, 0x10000000, 0x20000000
XOR = 0x5A5A


def uid_word(uid: int, flags: int = 0) -> int:
    return (uid & 1) | flags | ((uid >> 1) << 8)


def build_memory(*, me=0, turn=3, turn_player=0, phase=2, types=(0, 1), summons=0):
    s = bytearray(0x3C40)
    struct.pack_into("<III", s, 0, 0, 0, 0)
    struct.pack_into("<I", s, OFF.turn, turn)
    struct.pack_into("<I", s, OFF.turn_player, turn_player)
    struct.pack_into("<I", s, OFF.phase, phase)
    uids = iter(range(1, 500))

    def put_uid(cid):
        uid = next(uids)
        struct.pack_into("<H", s, OFF.uid_table + uid * 8, cid)
        return uid

    def slot(p, i, cid, face, sideways=0, chain=0, locked=False):
        at = p * OFF.player_stride + OFF.slot_base + i * OFF.slot_stride
        struct.pack_into("<HH", s, at, cid, uid_word(put_uid(cid)))
        s[at + OFF.slot_face] = face
        s[at + OFF.slot_turn] = sideways
        s[at + SEM["locked_byte"]] = SEM["locked"] | 0x0A if locked else 0
        struct.pack_into("<H", s, at + OFF.slot_chain, chain)

    def pile(p, zone, cards):
        blk = p * OFF.player_stride
        struct.pack_into("<I", s, blk + OFF.count[zone], len(cards))
        for i, (cid, facedown) in enumerate(cards):
            struct.pack_into("<HH", s, blk + OFF.pile[zone] + 4 * i, cid, uid_word(put_uid(cid)))
            if facedown:
                s[blk + OFF.removed_face + 2 * i] = OFF.removed_facedown

    def node(n, code, kind, uid, nxt):
        at = OFF.effect_table + n * OFF.effect_stride
        struct.pack_into("<HB", s, at, code, kind)
        struct.pack_into("<I", s, at + OFF.effect_uid, uid)
        struct.pack_into("<H", s, at + OFF.effect_next, nxt)

    adv, opp = me, 1 - me
    struct.pack_into("<I", s, adv * OFF.player_stride + SEM["normal_summons"], summons)
    for p, lp in ((adv, 8000), (opp, 6100)):
        struct.pack_into("<I", s, p * OFF.player_stride + OFF.lp, lp ^ XOR)
    # advisor: Zeus with one material (node 1 -> unrelated node 2), a face-down Set S/T, two hand cards
    mat = put_uid(MAT)
    node(1, OFF.overlay_code, 0x15, mat, 2)
    node(2, 0x123, 0x05, 0, 0)
    slot(adv, 5, ZEUS, SEM["faceup"], chain=1, locked=True)
    slot(adv, 7, DM, 0)
    pile(adv, "hand", [(BEWD, 0), (ASH, 0)])
    pile(adv, "deck", [(DM, 0)] * 30)
    pile(adv, "grave", [(ASH, 0)])
    pile(adv, "extra", [(ZEUS, 0)] * 14)
    # opponent: face-up attack Blue-Eyes, face-down defense monster, Set S/T, 4 hand cards whose ids are present
    slot(opp, 0, BEWD, SEM["faceup"])
    slot(opp, 1, DM, 0, sideways=1)
    slot(opp, 8, ASH, 0)
    pile(opp, "hand", [(DM, 0)] * 4)
    pile(opp, "deck", [(BEWD, 0)] * 28)
    pile(opp, "grave", [(ASH, 0)])
    pile(opp, "removed", [(BEWD, False), (DM, True)])
    key = struct.pack("<HHIII", XOR, 0, me, *(types if me == 0 else types[::-1])).ljust(0x340, b"\0")
    regions = {BASE + OFF.state_ptr: struct.pack("<Q", STATE), BASE + OFF.key_ptr: struct.pack("<Q", KEY),
               STATE: bytes(s), KEY: key}  # fmt: skip
    return BufferMemory(regions)


@pytest.mark.parametrize("me", [0, 1])
def test_read_duel(me):
    raw = read_duel(build_memory(me=me), BASE, OFF)
    assert (raw.turn, raw.phase, raw.turn_player, raw.me) == (3, 2, 0, me)
    mine, theirs = raw.players[me], raw.players[1 - me]
    assert (mine.lp, theirs.lp) == (8000, 6100)
    assert mine.slots[5].cid == ZEUS and mine.slots[5].materials == (MAT,)
    assert [c.cid for c in mine.piles["hand"]] == [BEWD, ASH]
    assert theirs.counts == {"hand": 4, "extra": 0, "deck": 28, "grave": 1, "removed": 2}
    assert [c.facedown for c in theirs.piles["removed"]] == [False, True]


def test_to_snapshot_needs_verified_offsets():
    raw = read_duel(build_memory(), BASE, OFF)
    with pytest.raises(SnapshotError, match="not verified"):
        to_snapshot(raw, PASSWORDS)


@pytest.mark.parametrize("me", [0, 1])
def test_to_snapshot_shows_only_what_the_advisor_can_see(me):
    off = replace(OFF, verified=True, semantics=SEM, dll_sha256="ab" * 32)
    raw = read_duel(build_memory(me=me, turn_player=me), BASE, off)
    snap = to_snapshot(raw, PASSWORDS)
    assert snap.turn_player == 0 and snap.phase == "main1" and snap.source == "mdmem:" + "ab" * 8
    assert snap.turn == 4 and snap.went_first is False  # counter 3 = turn 4 on screen, the second player's
    adv, opp = snap.sides
    assert (adv.lp, opp.lp) == (8000, 6100)
    zeus = next(c for c in adv.cards if c.location == "mzone")
    assert (zeus.sequence, zeus.password, zeus.materials) == (5, PASSWORDS[ZEUS], (PASSWORDS[MAT],))
    assert zeus.position_locked and not any(c.position_locked for c in opp_monsters(snap))
    assert any(c.location == "szone" and c.password == PASSWORDS[DM] and c.pos == "facedown" for c in adv.cards)
    assert [c.password for c in adv.at("hand")] == [PASSWORDS[BEWD], PASSWORDS[ASH]]
    assert (adv.deck_count, adv.extra_count) == (30, 14)
    # opponent: hand ids exist in memory but stay hidden; face-down cards are unknown
    assert [c.password for c in opp.at("hand")] == [UNKNOWN] * 4
    monsters = {c.sequence: c for c in opp.at("mzone")}
    assert (monsters[0].password, monsters[0].pos) == (PASSWORDS[BEWD], "faceup_attack")
    assert (monsters[1].password, monsters[1].pos) == (UNKNOWN, "facedown_defense")
    assert [c.password for c in opp.at("szone")] == [UNKNOWN]
    assert [c.password for c in opp.at("removed")] == [PASSWORDS[BEWD], UNKNOWN]
    assert opp.deck_count == 28


def opp_monsters(snap):
    return snap.sides[1].at("mzone")


def test_to_snapshot_only_on_the_advisors_main_phase():
    off = replace(OFF, verified=True, semantics=SEM)
    with pytest.raises(SnapshotError, match="Main and Battle"):
        to_snapshot(read_duel(build_memory(turn_player=1), BASE, off), PASSWORDS)
    with pytest.raises(SnapshotError, match="Main and Battle"):
        to_snapshot(read_duel(build_memory(phase=5), BASE, off), PASSWORDS)
    battle = to_snapshot(read_duel(build_memory(phase=3), BASE, off), PASSWORDS, attacked=frozenset({1}))
    assert battle.phase == "battle"


def test_format_raw_hides_the_opponents_cards():
    text = format_raw(read_duel(build_memory(), BASE, OFF), {BEWD: "Blue-Eyes White Dragon"})
    me, opp = text.split("OPPONENT")
    assert "4007 Blue-Eyes White Dragon" in me and "hand" in me
    cards = [line.split() for line in opp.splitlines() if line.startswith("  ")]
    assert [c[0] for c in cards] == ["M1", "M2", "S2", "grave"]  # no hand, deck or banished entries
    assert all(c[1] == "(hidden)" for c in cards[:3]) and cards[3][1] == str(ASH)


def test_find_module_and_buildid(tmp_path):
    lib = tmp_path / "steamapps"
    dll = lib / "common" / "Yu-Gi-Oh!  Master Duel" / "masterduel_Data" / "Plugins" / "x86_64" / "duel.dll"
    dll.parent.mkdir(parents=True)
    (lib / "appmanifest_1449850.acf").write_text('"AppState"\n{\n\t"buildid"\t\t"23020425"\n}\n')
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "maps").write_text(
        f"6ffff0000000-6ffff0001000 r--p 00000000 103:02 1 {lib}/common/other.dll\n"
        f"6ffff1000000-6ffff1001000 r--p 00000000 103:02 2 {dll}\n"
        f"6ffff1001000-6ffff2000000 r-xp 00001000 103:02 2 {dll}\n"
    )  # fmt: skip
    module = find_module(42, proc=tmp_path)
    assert module.base == 0x6FFFF1000000 and module.path == dll
    assert steam_buildid(module.path) == "23020425"


@pytest.mark.parametrize("types,ok", [((0, 1), True), ((0, 0), False), ((0, 2), False), ((1, 1), False)])
def test_solo_gate_acts_only_against_the_computer(types, ok):
    from ygorl.advisor.mdmem import GateError, solo_gate

    verified = replace(OFF, verified=True, semantics=SEM)
    for me in (0, 1):
        raw = read_duel(build_memory(me=me, types=types), BASE, verified)
        if ok:
            solo_gate(raw)
        else:
            with pytest.raises(GateError):
                solo_gate(raw)
    with pytest.raises(GateError, match="not verified"):
        solo_gate(read_duel(build_memory(), BASE, OFF))


def test_parse_tree_and_xwd():
    from ygorl.advisor.mdscreen import parse_tree, parse_xwd

    tree = ('     0x6000003 "masterduel": ("steam_app_1449850" "steam_app_1449850")  2054x1184+2776+128  +2776+128\n'
            '     0x6000010 "x": ("steam_app_1449850" "steam_app_1449850")  10x10+0+0  +0+0\n'
            '     0x1200001 "other": ("other" "Other")  4000x4000+0+0  +0+0\n')  # fmt: skip
    w = parse_tree(tree)
    assert (w.id, w.x, w.y, w.width, w.height) == (0x6000003, 2776, 128, 2054, 1184)
    # a 2x1 LSBFirst TrueColor dump: red, then blue
    # xwd header: size, version, format, depth, width, height, xoffset, byte order, unit, bit order, pad,
    # bits per pixel, bytes per line, visual class, red / green / blue masks, bits per rgb, ...
    header = [100, 7, 2, 24, 2, 1, 0, 0, 32, 0, 32, 32, 8, 4, 0xFF0000, 0xFF00, 0xFF, 8, 0, 0, 0, 0, 0, 0, 0]
    data = struct.pack(">25I", *header) + bytes([0, 0, 255, 0, 255, 0, 0, 0])
    assert parse_xwd(data).tolist() == [[[255, 0, 0], [0, 0, 255]]]


def test_normal_summoned_from_the_counter():
    off = replace(OFF, verified=True, semantics=SEM)
    assert not to_snapshot(read_duel(build_memory(), BASE, off), PASSWORDS).normal_summoned
    assert to_snapshot(read_duel(build_memory(summons=1), BASE, off), PASSWORDS).normal_summoned


def test_layout_hand_order_and_positions():
    from ygorl.advisor.mdlayout import hand_card, hand_order, my_zone
    from ygorl.engine.duel import default_cards

    from ygorl.advisor.mdmem import load_md_cid

    cards = default_cards()
    pw = load_md_cid(Path(__file__).parents[1] / "environments/md-2026-09/artifacts/md_cid.json")
    # memory order -> screen order, seen in the Solo duel (MD card ids)
    seen = {(5558, 4887, 4837, 16480, 4999): (5558, 4999, 4837, 16480, 4887),
            (16480, 4999, 16482, 6511): (4999, 6511, 16480, 16482),
            (16480, 6511, 16482, 4755, 16479): (16479, 4755, 6511, 16480, 16482)}  # fmt: skip
    for hand, screen in seen.items():
        assert tuple(hand[i] for i in hand_order(hand, pw.get, cards.get)) == screen
    assert hand_card(4, 5) == (1300, 1100) and hand_card(0, 5) == (740, 1100)
    assert my_zone("szone", 0) == (650, 930)


def test_client_area_and_reference_frame():
    from ygorl.advisor.mdlayout import REF_CLIENT, reference_image, to_client
    from ygorl.advisor.mdscreen import parse_children

    kids = ('xwininfo: Window id: 0x6000003 "masterduel"\n\n  Root window id: 0x416 (the root window) (has no name)\n'
            "     1 child:\n     0x5e0002f (has no name): ()  2048x1152+3+29  +2779+157\n")  # fmt: skip
    c = parse_children(kids)
    assert (c.id, c.x, c.y, c.width, c.height) == (0x5E0002F, 2779, 157, 2048, 1152)
    assert parse_children("     0 children.\n") is None
    # the reference client area maps to itself; a half-size client halves the offsets
    assert to_client(1580, 560, 2048, 1152) == (1577, 531)
    assert to_client(1580, 560, 1024, 576) == (788, 266)
    # a half-size capture comes back at reference size, content where the reference client was
    small = np.zeros((576, 1024, 3), np.uint8)
    small[266, 788] = 255
    ref = reference_image(small)
    assert ref.shape == (1184, 2054, 3)
    cx, cy, _, _ = REF_CLIENT
    assert ref[cy + 532, cx + 1576].tolist() == [255, 255, 255]


def test_find_icons_on_a_drawn_board():
    """Two command icons (dark discs with a glyph) above a hand card; a dark square card art is not an icon."""
    from ygorl.advisor.mdlayout import find_icons

    img = np.full((1184, 2054, 3), 180, np.uint8)
    yy, xx = np.mgrid[0:1184, 0:2054]
    for cx, cy, glyph in ((668, 886, 0.3), (842, 890, 0.85)):  # glyph: share of the radius filled by the glyph
        d = np.hypot(xx - cx, yy - cy)
        img[d <= 44] = (15, 20, 25)
        img[d <= 44 * glyph] = (36, 128, 251)
    found = find_icons(img, 360, 790, 1120, 975)
    assert len(found) == 2  # within a few pixels of the centres: well inside the 44 px icon
    assert all(abs(fx - x) <= 8 and abs(fy - y) <= 8 for (fx, fy), (x, y) in zip(found, [(668, 886), (842, 890)]))


def test_new_icons_ignore_card_art_that_was_already_there():
    from ygorl.advisor.mdlayout import new_icons

    before = np.full((1184, 2054, 3), 180, np.uint8)
    before[700:780, 600:680] = (20, 20, 20)  # dark card art: looks like an icon to the shape test
    after = before.copy()
    yy, xx = np.mgrid[0:1184, 0:2054]
    after[np.hypot(xx - 840, yy - 600) <= 44] = (15, 20, 25)  # the icon the click opened
    found = new_icons(before, after, 520, 500, 980, 840)
    assert len(found) == 1 and abs(found[0][0] - 840) <= 8 and abs(found[0][1] - 600) <= 8
