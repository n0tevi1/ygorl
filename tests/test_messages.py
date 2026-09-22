"""Tests for typed MSG_* decoding (T1.4)."""

import struct

import pytest

from ygorl.engine import constants as C
from ygorl.engine import messages as M


def rec(msg_type: int, fmt: str = "", *values) -> bytes:
    return bytes([msg_type]) + struct.pack("<" + fmt, *values)


def framed(*records: bytes) -> bytes:
    return b"".join(struct.pack("<I", len(r)) + r for r in records)


LOC = ("BBII", 0, C.LOCATION_HAND, 2, C.POS_FACEDOWN_DEFENSE)


def test_every_header_message_is_decoded_or_documented():
    """Scripted cross-check against ocgapi_constants.h (via the generated constants)."""
    header = {v for k, v in vars(C).items() if k.startswith("MSG_")}
    decoded = M.decoded_types()
    assert header == decoded | M.NOT_EMITTED_BY_CORE
    assert not decoded & M.NOT_EMITTED_BY_CORE
    assert M.DECISION_TYPES <= decoded


def test_split_and_decode_buffer():
    buf = framed(rec(C.MSG_NEW_TURN, "B", 1), rec(C.MSG_NEW_PHASE, "H", C.PHASE_MAIN1))
    msgs = M.decode_buffer(buf)
    assert msgs == [M.NewTurn(1), M.NewPhase(C.PHASE_MAIN1)]
    assert msgs[0].name == "MSG_NEW_TURN"


def test_unknown_message_is_not_an_exception(caplog):
    msg = M.decode_message(bytes([250, 1, 2, 3]))
    assert isinstance(msg, M.UnknownMessage)
    assert msg.msg_type == 250 and msg.raw == b"\x01\x02\x03"
    assert "unknown message type 250" in caplog.text


def test_not_emitted_message_is_unknown():
    msg = M.decode_message(rec(C.MSG_START, "B", 0))
    assert isinstance(msg, M.UnknownMessage) and msg.name == "MSG_START"


def test_truncated_payload_is_undecodable():
    msg = M.decode_message(rec(C.MSG_WIN, "B", 1))  # missing reason byte
    assert isinstance(msg, M.UndecodableMessage)
    assert "truncated" in msg.error


def test_corrupt_framing():
    msgs = M.decode_buffer(struct.pack("<I", 100) + b"\x05")
    assert len(msgs) == 1 and isinstance(msgs[0], M.UndecodableMessage)


def test_select_idlecmd():
    payload = struct.pack("<B", 0)
    payload += struct.pack("<I", 1) + struct.pack("<IBBI", 111, 0, C.LOCATION_HAND, 3)  # summonable
    payload += struct.pack("<I", 0)  # spsummonable
    payload += struct.pack("<I", 1) + struct.pack("<IBBB", 222, 0, C.LOCATION_MZONE, 1)  # repos (u8 seq)
    payload += struct.pack("<I", 0) + struct.pack("<I", 0)  # mset, sset
    payload += struct.pack("<I", 1) + struct.pack("<IBBIQB", 333, 0, C.LOCATION_SZONE, 2, 5, 0)
    payload += bytes([1, 1, 0])
    msg = M.decode_message(bytes([C.MSG_SELECT_IDLECMD]) + payload)
    assert isinstance(msg, M.SelectIdleCmd) and isinstance(msg, M.Decision)
    assert msg.player == 0
    assert msg.summonable == (M.CardInfo(111, M.Location(0, C.LOCATION_HAND, 3)),)
    assert msg.repositionable[0].loc.sequence == 1
    assert msg.activatable == (M.ChainOption(333, M.Location(0, C.LOCATION_SZONE, 2), 5, 0),)
    assert msg.can_battle_phase and msg.can_end_phase and not msg.can_shuffle


def test_select_battlecmd():
    payload = struct.pack("<BI", 1, 0) + struct.pack("<I", 1) + struct.pack("<IBBBB", 9, 1, C.LOCATION_MZONE, 2, 1)
    msg = M.decode_message(bytes([C.MSG_SELECT_BATTLECMD]) + payload + bytes([1, 0]))
    assert msg == M.SelectBattleCmd(1, (), (M.AttackOption(9, M.Location(1, C.LOCATION_MZONE, 2), True),), True, False)


def test_select_card_and_chain():
    card = struct.pack("<I", 42) + struct.pack("<" + LOC[0], *LOC[1:])
    msg = M.decode_message(rec(C.MSG_SELECT_CARD, "BBIII", 0, 1, 1, 2, 1) + card)
    assert msg == M.SelectCard(0, True, 1, 2, (M.CardInfo(42, M.Location(0, C.LOCATION_HAND, 2, C.POS_FACEDOWN_DEFENSE)),))
    chain = rec(C.MSG_SELECT_CHAIN, "BBBIII", 1, 0, 0, 0x10, 0x20, 1) + card + struct.pack("<QB", 77, 1)
    msg = M.decode_message(chain)
    assert isinstance(msg, M.SelectChain) and not msg.forced and msg.chains[0].description == 77


def test_select_sum_modes():
    opt = struct.pack("<I", 5) + struct.pack("<" + LOC[0], *LOC[1:]) + struct.pack("<I", (6 << 16) | 4)
    msg = M.decode_message(rec(C.MSG_SELECT_SUM, "BBIIII", 0, 0, 8, 1, 3, 0) + struct.pack("<I", 1) + opt)
    assert msg.exact and msg.target == 8 and msg.cards[0].values == (4, 6)
    msg = M.decode_message(rec(C.MSG_SELECT_SUM, "BBIIII", 0, 1, 8, 1, 0, 0) + struct.pack("<I", 0))
    assert not msg.exact and msg.cards == ()


def test_place_disfield_position_announce():
    assert M.decode_message(rec(C.MSG_SELECT_PLACE, "BBI", 0, 1, 0xFFFFFF00)) == M.SelectPlace(0, 1, 0xFFFFFF00)
    assert isinstance(M.decode_message(rec(C.MSG_SELECT_DISFIELD, "BBI", 0, 1, 0)), M.SelectDisfield)
    assert M.decode_message(rec(C.MSG_SELECT_POSITION, "BIB", 1, 99, 0x5)) == M.SelectPosition(1, 99, 0x5)
    assert M.decode_message(rec(C.MSG_ANNOUNCE_RACE, "BBQ", 0, 1, C.RACE_ALL)) == M.AnnounceRace(0, 1, C.RACE_ALL)
    assert M.decode_message(rec(C.MSG_ANNOUNCE_ATTRIB, "BBI", 0, 2, 0x7F)) == M.AnnounceAttrib(0, 2, 0x7F)
    assert M.decode_message(rec(C.MSG_ANNOUNCE_CARD, "BBQQ", 0, 2, 1, C.OPCODE_ISCODE)).opcodes == (1, C.OPCODE_ISCODE)
    assert M.decode_message(rec(C.MSG_ANNOUNCE_NUMBER, "BBQQ", 0, 2, 3, 4)).options == (3, 4)
    assert M.decode_message(rec(C.MSG_ROCK_PAPER_SCISSORS, "B", 1)) == M.RockPaperScissors(1)


def test_tribute_counter_sort_unselect():
    t = M.decode_message(rec(C.MSG_SELECT_TRIBUTE, "BBIII", 0, 0, 1, 2, 1) + struct.pack("<IBBIB", 7, 0, C.LOCATION_MZONE, 0, 2))
    assert t.cards[0].release_param == 2 and not t.cancelable
    c = M.decode_message(rec(C.MSG_SELECT_COUNTER, "BHHI", 0, 0x1, 2, 1) + struct.pack("<IBBBH", 7, 0, C.LOCATION_SZONE, 1, 3))
    assert c.count == 2 and c.cards[0].counters == 3
    s = M.decode_message(rec(C.MSG_SORT_CHAIN, "BI", 1, 1) + struct.pack("<IBII", 7, 1, C.LOCATION_MZONE, 0))
    assert isinstance(s, M.SortChain) and s.cards[0].loc.location == C.LOCATION_MZONE
    card = struct.pack("<I", 42) + struct.pack("<" + LOC[0], *LOC[1:])
    u = M.decode_message(rec(C.MSG_SELECT_UNSELECT_CARD, "BBBIII", 0, 1, 0, 1, 3, 1) + card + struct.pack("<I", 1) + card)
    assert u.finishable and not u.cancelable and len(u.selectable) == len(u.unselectable) == 1


def test_events():
    loc = struct.pack("<" + LOC[0], *LOC[1:])
    mv = M.decode_message(rec(C.MSG_MOVE, "I", 5) + loc + loc + struct.pack("<I", C.REASON_EFFECT))
    assert mv.code == 5 and mv.reason == C.REASON_EFFECT
    d = M.decode_message(rec(C.MSG_DRAW, "BIIIII", 0, 2, 1, 10, 2, 10))
    assert d == M.Draw(0, ((1, 10), (2, 10)))
    ch = M.decode_message(rec(C.MSG_CHAINING, "I", 5) + loc + struct.pack("<BBIQI", 0, C.LOCATION_HAND, 0, 80, 1))
    assert ch.chain_count == 1 and ch.description == 80
    assert M.decode_message(rec(C.MSG_CHAIN_SOLVED, "B", 1)) == M.ChainSolved(1)
    assert M.decode_message(rec(C.MSG_DAMAGE, "BI", 1, 500)) == M.Damage(1, 500)
    assert M.decode_message(rec(C.MSG_WIN, "BB", 0, 1)) == M.Win(0, 1)
    assert M.decode_message(rec(C.MSG_TOSS_DICE, "BBBB", 0, 2, 3, 6)) == M.TossDice(0, (3, 6))
    assert M.decode_message(rec(C.MSG_HAND_RES, "B", 1 | (3 << 2))) == M.HandResult(1, 3)
    b = M.decode_message(rec(C.MSG_BATTLE, "") + loc + struct.pack("<iiB", 3000, 2500, 0) + b"\0" * 10 + struct.pack("<iiB", 0, 0, 0))
    assert b.attacker_atk == 3000 and b.target.location == 0
    assert M.decode_message(rec(C.MSG_SHOW_HINT, "H", 2) + b"hi\0") == M.ShowHint("hi")
    assert M.decode_message(rec(C.MSG_SUMMONED)) == M.Summoned()
    assert M.decode_message(rec(C.MSG_CONFIRM_CARDS, "BIIBBI", 1, 1, 9, 0, C.LOCATION_HAND, 0)).cards[0].code == 9
