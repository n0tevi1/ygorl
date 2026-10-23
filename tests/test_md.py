"""Master Duel memory reader (design 10, T7.6): offset derivation, reading, masking, snapshot conversion.

The game binary is not in the repository: derivation is checked against the installed client when there is one.
Everything else runs on synthetic memory laid out by the derived offsets of build 23020425.
"""

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ygorl.md.memory import BufferMemory, decode_uid, find_module, format_raw, read_duel, steam_buildid, to_snapshot
from ygorl.md.offsets import Offsets, OffsetError, derive, parse_objdump
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
              effect_uid=0x4, effect_next=0xC, overlay_code=0x3FB, overlay_kind=0x5, player_effects=0x1D0,
              input_mode=0x3CD0, input_accepting=0x3CDC, input_player=0x3D00, list_ptr=0x11F9C30, list_max=0x1,
              list_type=0x2, list_count=0xA, list_marker=0xC, list_item_cid=0x18, list_item_from=0x4C8,
              dialog_type=0x3C94, dialog_mix_count=0x3CA0, dialog_mix=0x3CA4, dialog_choice_count=0x3CB4,
              dialog_choices=0x3CB8)  # fmt: skip
# read off the screen in a Solo duel on this build (design 10)
SEM = {
    "faceup": 0x01,
    "defense": 0x01,
    "locked_byte": 0x14,
    "locked": 0x40,
    "cpu_type": 1,
    "turn_base": 1,
    "normal_summons": 0x30,
    "set_turn": 0x10,
    # per-turn state, decoded from the recorded Solo duels of 2026-10-22 (offsets_data notes)
    "attacked": 0x80,
    "effect_used": 0x80000,
    "card_activated": 0x1000,
    "summon_normal": 0x1,
    "summon_special": 0x4,
    "summon_flip": 0xA,
    "special_summons": 0x34,
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
    from ygorl.md.offsets import offsets_file

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


def build_memory(*, me=0, turn=3, turn_player=0, phase=2, types=(0, 1), summons=0, specials=0, zeus_flags=None,
                 player_nodes=()):  # fmt: skip
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

    def slot(p, i, cid, face, sideways=0, chain=0, locked=False, flags=None):
        at = p * OFF.player_stride + OFF.slot_base + i * OFF.slot_stride
        struct.pack_into("<HH", s, at, cid, uid_word(put_uid(cid)))
        s[at + OFF.slot_face] = face
        s[at + OFF.slot_turn] = sideways
        if flags is None:
            flags = SEM["locked"] | 0x0A if locked else 0
        struct.pack_into("<I", s, at + SEM["locked_byte"], flags)
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
    struct.pack_into("<I", s, adv * OFF.player_stride + SEM["special_summons"], specials)
    for p, lp in ((adv, 8000), (opp, 6100)):
        struct.pack_into("<I", s, p * OFF.player_stride + OFF.lp, lp ^ XOR)
    # advisor: Zeus with one material (node 1 -> unrelated node 2), a face-down Set S/T, two hand cards
    mat = put_uid(MAT)
    node(1, OFF.overlay_code, 0x15, mat, 2)
    node(2, 0x123, 0x05, 0, 0)
    slot(adv, 5, ZEUS, SEM["faceup"], chain=1, locked=True, flags=zeus_flags)
    # the advisor's player effect list: (code, kind, value) nodes from index 10 on, chained in order
    for k, (code, kind, value) in enumerate(player_nodes):
        node(10 + k, code, kind, value, 11 + k if k + 1 < len(player_nodes) else 0)
    if player_nodes:
        struct.pack_into("<H", s, adv * OFF.player_stride + OFF.player_effects, 10)
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


RECORDED = Path(__file__).resolve().parent / "data" / "md_seeker_once_per_turn.npz"
MD_CID = Path(__file__).resolve().parents[1] / "environments" / "md-2026-09" / "artifacts" / "md_cid.json"
SEEKER = 5130393  # Millennium Seeker: its field effect once per turn


def recorded_duel(seq: int):
    """A recorded Solo step (run 20261022-154920): the client's own player block as read then, the opponent and
    the turn restored from the trace's public view (the recording never held the opponent's block)."""
    import json

    from ygorl.md.memory import PHASES
    from ygorl.md.offsets import offsets_file

    z = np.load(RECORDED)
    public = json.loads(z["public"].tobytes())[str(seq)]
    off = Offsets.from_json(json.loads(offsets_file(BUILD_23020425).read_text()))
    s = bytearray(0x3C40)
    s[: off.player_stride] = z[f"me{seq}"].tobytes()
    key = z["key"].tobytes().ljust(0x340, b"\0")
    opp, o = off.player_stride, public["opp"]
    struct.pack_into("<I", s, opp + off.lp, o["lp"] ^ struct.unpack_from("<H", key)[0])
    for i, f in enumerate(o["field"]):
        if f:
            at = opp + off.slot_base + i * off.slot_stride
            struct.pack_into("<H", s, at, f[0] or 0xFFFF)  # face-down: some id, never shown
            s[at + off.slot_face], s[at + off.slot_turn] = f[1], f[2]
    for zone, n in o["counts"].items():
        struct.pack_into("<I", s, opp + off.count[zone], n)
    for i, cid in enumerate(o["grave"]):
        struct.pack_into("<H", s, opp + off.pile["grave"] + 4 * i, cid)
    struct.pack_into("<I", s, off.turn, public["turn"] - off.semantics["turn_base"])
    struct.pack_into("<I", s, off.phase, PHASES.index(public["phase"]))
    regions = {BASE + off.state_ptr: struct.pack("<Q", STATE), BASE + off.key_ptr: struct.pack("<Q", KEY),
               STATE: bytes(s), KEY: key}  # fmt: skip
    return read_duel(BufferMemory(regions), BASE, off), public


def test_recorded_once_per_turn_reaches_the_reconstruction():
    """Millennium Seeker before (seq 39) and after (seq 41, MD refused it) its field effect: the slot flag says the
    effect was used, and the rebuilt duel no longer offers it."""
    import random

    from ygorl.advisor.reconstruct import OpponentPrior, build, complete
    from ygorl.cards.ydk import Deck
    from ygorl.engine import constants as C
    from ygorl.engine.duel import default_cards
    from ygorl.md.memory import load_md_cid, own_deck

    cards, passwords = default_cards(), load_md_cid(MD_CID)
    prior = OpponentPrior((Deck(main=(89631139,) * 40),), (1.0,))
    for seq, used in ((39, False), (41, True)):
        raw, public = recorded_duel(seq)
        assert (public["refusal"] is not None) == used
        snap = to_snapshot(raw, passwords)
        seeker = next(c for c in snap.sides[0].cards if c.password == SEEKER and c.location == "mzone")
        assert (seeker.summoned, seeker.position_locked, seeker.effect_used) == ("normal", True, used)
        assert snap.normal_summons == 1 and snap.normal_summoned and not snap.normal_summon_left
        board = complete(snap, own_deck(raw, passwords, cards), prior, cards, random.Random(0))
        acts = build(board, C.DUEL_MODE_MR5, cards=cards).point.actions
        offered = any(a.kind == "activate" and a.card.code == SEEKER and a.card.loc.location == C.LOCATION_MZONE
                      for a in acts)  # fmt: skip
        assert offered != used, seq
        assert not any(a.kind in ("summon", "mset") for a in acts)


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
    from ygorl.md.memory import GateError, solo_gate

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
    from ygorl.md.screen import parse_tree, parse_xwd

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
    # +0x30 counts every summon, +0x34 the Special Summons: a Special Summon alone leaves the Normal Summon
    only_special = to_snapshot(read_duel(build_memory(summons=1, specials=1), BASE, off), PASSWORDS)
    assert not only_special.normal_summoned and only_special.normal_summons == 0
    both = to_snapshot(read_duel(build_memory(summons=3, specials=2), BASE, off), PASSWORDS)
    assert both.normal_summoned and both.normal_summons == 1


@pytest.mark.parametrize("me", [0, 1])
def test_turn_flags_of_the_advisors_cards(me):
    """The slot u32 at +0x14 carries the turn: how the monster came, whether it attacked, an effect used."""
    off = replace(OFF, verified=True, semantics=SEM)
    zeus = lambda flags: next(c for c in to_snapshot(read_duel(build_memory(me=me, turn_player=me, zeus_flags=flags), BASE, off),  # noqa: E731
                                                       PASSWORDS).sides[0].cards if c.password == PASSWORDS[ZEUS])  # fmt: skip
    fresh = zeus(0x00040044)  # Special Summoned this turn, nothing else yet
    assert (fresh.summoned, fresh.position_locked, fresh.attacked, fresh.effect_used) == ("special", True, False, False)
    used = zeus(0x000C0044)  # ... and an effect of it activated
    assert used.effect_used and not used.attacked
    hit = zeus(0x020220C0)  # summoned an earlier turn, attacked this turn
    assert (hit.summoned, hit.attacked, hit.position_locked) == ("", True, True)
    assert zeus(0x0000004A).summoned == "flip" and zeus(0x00000041).summoned == "normal"
    old = zeus(0)
    assert (old.summoned, old.attacked, old.effect_used, old.position_locked) == ("", False, False, False)


def test_extra_tribute_summons_from_the_player_effect_list():
    """Effect nodes are coded by the MD card id of their card: Ancient Chant's value-0 node grants the additional
    Tribute Summon, its value-1 node (the GY effect) and inactive nodes do not; the opponent's list is not read."""
    from ygorl.md.memory import EXTRA_TRIBUTE_CODES

    off = replace(OFF, verified=True, semantics=SEM)
    chant, advance = 0x3BE5, 0x2CA1
    assert chant in EXTRA_TRIBUTE_CODES and advance in EXTRA_TRIBUTE_CODES
    cases = {(): 0, ((chant, 0x01, 0),): 1, ((chant, 0x01, 1),): 0, ((chant, 0x06, 0),): 0,
             ((chant, 0x01, 0), (0x123, 0x01, 0), (advance, 0x02, 7)): 2}  # fmt: skip
    for nodes, extra in cases.items():
        raw = read_duel(build_memory(player_nodes=nodes, summons=1), BASE, off)
        assert [e.code for e in raw.players[0].effects] == [n[0] for n in nodes]
        assert raw.players[1].effects == ()
        snap = to_snapshot(raw, PASSWORDS)
        assert snap.extra_tribute_summons == extra
        assert snap.normal_summon_left == (extra > 0)


def test_layout_hand_order_and_positions():
    from ygorl.md.layout import hand_card, hand_order, my_zone
    from ygorl.engine.duel import default_cards

    from ygorl.md.memory import load_md_cid

    cards = default_cards()
    pw = load_md_cid(Path(__file__).parents[1] / "environments/md-2026-09/artifacts/md_cid.json")
    # memory order -> screen order, seen in the Solo duel (MD card ids)
    seen = {(5558, 4887, 4837, 16480, 4999): (5558, 4999, 4837, 16480, 4887),
            (16480, 4999, 16482, 6511): (4999, 6511, 16480, 16482),
            (16480, 6511, 16482, 4755, 16479): (16479, 4755, 6511, 16480, 16482),
            (6044, 4041, 5559, 15839, 5559, 4343): (5559, 5559, 4041, 6044, 4343, 15839)}  # fmt: skip
    for hand, screen in seen.items():
        assert tuple(hand[i] for i in hand_order(hand, pw.get, cards.get)) == screen
    assert hand_card(4, 5) == (1300, 1100) and hand_card(0, 5) == (740, 1100)
    assert my_zone("szone", 0) == (650, 930)


def test_client_area_and_reference_frame():
    from ygorl.md.layout import REF_CLIENT, reference_image, to_client
    from ygorl.md.screen import parse_children

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
    from ygorl.md.layout import find_icons

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
    from ygorl.md.layout import new_icons

    before = np.full((1184, 2054, 3), 180, np.uint8)
    before[700:780, 600:680] = (20, 20, 20)  # dark card art: looks like an icon to the shape test
    after = before.copy()
    yy, xx = np.mgrid[0:1184, 0:2054]
    after[np.hypot(xx - 840, yy - 600) <= 44] = (15, 20, 25)  # the icon the click opened
    found = new_icons(before, after, 520, 500, 980, 840)
    assert len(found) == 1 and abs(found[0][0] - 840) <= 8 and abs(found[0][1] - 600) <= 8


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_list_cards_from_the_tray_width(n):
    from ygorl.md.layout import CARD_STEP, TRAY_RGB, list_cards

    img = np.full((1184, 2054, 3), 15, np.uint8)  # the panel's dark background
    width = CARD_STEP * n - 20
    left = 1026 - width // 2
    img[820:1060, left : left + width] = TRAY_RGB
    centres = [round(left + (width - (n - 1) * CARD_STEP) / 2 + i * CARD_STEP) for i in range(n)]
    for x in centres:
        img[890:1030, x - 60 : x + 60] = (200, 160, 90)  # card art, with some tray-blue in it
        img[930:950, x - 10 : x + 10] = TRAY_RGB
    assert list_cards(img) == [(x, 905) for x in centres]


def test_expectation_checks_the_policys_action_against_memory():
    from types import SimpleNamespace

    from ygorl.selfplay.loop import Expectation, check_expectation
    from ygorl.engine import constants as C

    off = replace(OFF, verified=True, semantics=SEM)
    before = read_duel(build_memory(), BASE, off)
    me = before.players[before.me]
    hand_bewd = next(c for c in me.piles["hand"] if c.cid == BEWD)
    card = SimpleNamespace(code=PASSWORDS[BEWD], loc=SimpleNamespace(location=C.LOCATION_HAND, sequence=0))

    def moved(face, sideways):  # Blue-Eyes left the hand for Main Monster Zone 1
        slots = list(me.slots)
        slots[1] = replace(hand_bewd, face=face, sideways=sideways)
        piles = dict(me.piles, hand=tuple(c for c in me.piles["hand"] if c is not hand_bewd))
        players = list(before.players)
        players[before.me] = replace(me, slots=tuple(slots), piles=piles)
        return replace(before, players=tuple(players))

    summon = Expectation(SimpleNamespace(kind="summon", card=card), before, "main1")
    mset = Expectation(SimpleNamespace(kind="mset", card=card), before, "main1")
    assert check_expectation(summon, moved(1, 0), PASSWORDS) is None
    assert "summon" in check_expectation(summon, moved(0, 1), PASSWORDS)  # Set instead of Summon
    assert check_expectation(mset, moved(0, 1), PASSWORDS) is None
    assert "summon" in check_expectation(summon, before, PASSWORDS)  # nothing happened
    battle = Expectation(SimpleNamespace(kind="battle_phase", card=None), before, "main1")
    assert "phase" in check_expectation(battle, before, PASSWORDS)
    assert check_expectation(battle, replace(before, phase=3), PASSWORDS) is None


def test_read_request_locates_the_candidates_through_their_uid():
    from ygorl.md.request import read_request

    mem = build_memory()
    raw = read_duel(mem, BASE, OFF)
    me = raw.players[raw.me]
    zeus, bewd = me.slots[5], me.piles["hand"][0]  # a card on the field and one in hand
    lst = 0x30000000
    tail = bytearray(0x200)  # S + 0x3c40 .. : the input state and the dialog block
    struct.pack_into("<I", tail, OFF.input_mode - 0x3C40, 8)  # select_card
    struct.pack_into("<I", tail, OFF.input_accepting - 0x3C40, 1)
    struct.pack_into("<I", tail, OFF.input_player - 0x3C40, 0)
    head = bytearray(0x600)
    head[OFF.list_max] = 1
    struct.pack_into("<HhH", head, OFF.list_type, 6, 0, 0)
    struct.pack_into("<h", head, OFF.list_count, 2)
    struct.pack_into("<H", head, OFF.list_marker, 0x12E5)
    for i, c in enumerate((zeus, bewd)):
        struct.pack_into("<HH", head, OFF.list_item_cid + 4 * i, c.cid, uid_word(c.uid))
        struct.pack_into("<H", head, OFF.list_item_from + 4 * i, 14 << 1)  # not the location for this list kind
    mem.regions.update({STATE + 0x3C40: bytes(tail), BASE + OFF.list_ptr: struct.pack("<Q", lst), lst: bytes(head)})
    req = read_request(mem, BASE, OFF, raw)
    assert (req.kind, req.accepting, req.player, req.min, req.max) == ("select_card", True, 0, 1, 1)
    assert [(c.cid, c.location, c.sequence, c.player) for c in req.candidates] == [
        (ZEUS, "mzone", 5, 0),
        (BEWD, "hand", 0, 0),
    ]
    assert req.for_client(raw.me)
    # a position dialog: its choices are the position mask
    struct.pack_into("<I", tail, OFF.input_mode - 0x3C40, 7)
    struct.pack_into("<I", tail, OFF.dialog_choices - 0x3C40, 0x7F)
    mem.regions[STATE + 0x3C40] = bytes(tail)
    pos = read_request(mem, BASE, OFF, raw)
    assert (pos.kind, pos.candidates, pos.choices) == ("position", (), (0x7F,))


def test_recording_keeps_only_the_clients_own_effect_nodes():
    """The recorded state blanks the effect table but keeps the nodes our lists reach: the player list and the
    lists of our field cards (Zeus's material); a node only the opponent's list reaches stays out."""
    from ygorl.selfplay.loop import own_effect_nodes

    mem = build_memory(player_nodes=((0x3BE5, 0x01, 0), (0x2CA1, 0x01, 0)))
    state = bytearray(mem.regions[STATE])
    at = OFF.effect_table + 40 * OFF.effect_stride  # an opponent node
    struct.pack_into("<HB", state, at, 0x1368, 1)
    struct.pack_into("<H", state, OFF.player_stride + OFF.player_effects, 40)
    kept = own_effect_nodes(bytes(state), 0, OFF)
    index = sorted((a - OFF.effect_table) // OFF.effect_stride for a in kept)
    assert index == [1, 2, 10, 11]
    assert all(kept[a] == bytes(state[a : a + OFF.effect_stride]) for a in kept)
