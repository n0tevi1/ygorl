"""Tests for decision -> legal actions -> response encoding (T1.4 / T1.5).

Multi-selects are split into one decision per card plus Finish (design §5.2);
each step only offers actions that can still be completed into a response the
core accepts.
"""

import struct

import pytest

from ygorl.cards.cdb import Card
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action, is_declarable, make_decision


def i32(*v):
    return struct.pack(f"<{len(v)}i", *v)


def card(code, loc=C.LOCATION_HAND, seq=0, con=0):
    return M.CardInfo(code, M.Location(con, loc, seq, 0))


def kinds(state):
    return [a.kind for a in state.actions()]


def play(state, *indices):
    """Apply a sequence of action indices; return the final response bytes."""
    resp = None
    for i in indices:
        assert resp is None, "decision already complete"
        resp = state.step(i)
    return resp


def find(state, kind, **attrs):
    for i, a in enumerate(state.actions()):
        if a.kind == kind and all(getattr(a, k) == v for k, v in attrs.items()):
            return i
    raise AssertionError(f"no {kind} {attrs} in {state.actions()}")


# --------------------------------------------------------------- single-step decisions


def test_idlecmd():
    msg = M.SelectIdleCmd(
        0,
        summonable=(card(1),),
        spsummonable=(card(2),),
        repositionable=(card(3, C.LOCATION_MZONE),),
        msetable=(card(4),),
        ssetable=(card(5), card(6)),
        activatable=(M.ChainOption(7, M.Location(0, C.LOCATION_HAND, 0), 112, 0),),
        can_battle_phase=True,
        can_end_phase=True,
        can_shuffle=False,
    )
    st = make_decision(msg)
    assert st.player == 0
    assert kinds(st) == ["summon", "spsummon", "reposition", "mset", "sset", "sset", "activate", "battle_phase", "end_phase"]
    assert st.actions()[6].description == 112 and st.actions()[6].card.code == 7
    assert make_decision(msg).step(0) == i32(0)
    assert make_decision(msg).step(5) == i32(4 | (1 << 16))
    assert make_decision(msg).step(6) == i32(5)
    assert make_decision(msg).step(7) == i32(6)
    assert make_decision(msg).step(8) == i32(7)


def test_battlecmd():
    msg = M.SelectBattleCmd(
        1,
        activatable=(M.ChainOption(7, M.Location(1, C.LOCATION_MZONE, 0), 1, 0),),
        attackable=(M.AttackOption(8, M.Location(1, C.LOCATION_MZONE, 1), True), M.AttackOption(9, M.Location(1, C.LOCATION_MZONE, 2), False)),
        can_main2=True,
        can_end_phase=False,
    )
    st = make_decision(msg)
    assert kinds(st) == ["activate", "attack", "attack", "main2"]
    assert make_decision(msg).step(2) == i32(1 | (1 << 16))
    assert make_decision(msg).step(3) == i32(2)


@pytest.mark.parametrize("msg", [M.SelectYesNo(0, 5), M.SelectEffectYn(0, card(1), 5)])
def test_yes_no(msg):
    assert kinds(make_decision(msg)) == ["yes", "no"]
    assert make_decision(msg).step(0) == i32(1)
    assert make_decision(msg).step(1) == i32(0)


def test_option_and_number_and_rps():
    st = make_decision(M.SelectOption(0, (10, 20, 30)))
    assert [a.description for a in st.actions()] == [10, 20, 30]
    assert st.step(2) == i32(2)
    st = make_decision(M.AnnounceNumber(0, (3, 5)))
    assert [a.value for a in st.actions()] == [3, 5] and st.step(1) == i32(1)
    st = make_decision(M.RockPaperScissors(1))
    assert st.player == 1 and [a.value for a in st.actions()] == [1, 2, 3] and st.step(0) == i32(1)


def test_chain_pass_only_when_not_forced():
    opt = M.ChainOption(7, M.Location(0, C.LOCATION_HAND, 0, 0), 99, 0)
    st = make_decision(M.SelectChain(0, 0, False, 0, 0, (opt,)))
    assert kinds(st) == ["chain", "pass"]
    assert st.step(1) == i32(-1)
    st = make_decision(M.SelectChain(0, 0, True, 0, 0, (opt,)))
    assert kinds(st) == ["chain"] and st.step(0) == i32(0)


def test_position():
    st = make_decision(M.SelectPosition(0, 5, C.POS_FACEUP_ATTACK | C.POS_FACEDOWN_DEFENSE))
    assert [a.value for a in st.actions()] == [C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_DEFENSE]
    assert st.step(1) == i32(C.POS_FACEDOWN_DEFENSE)


# ------------------------------------------------------------------ multi-step selects


def cards_response(*indices):
    return struct.pack(f"<iI{len(indices)}I", 0, len(indices), *indices)


def test_select_card_steps():
    msg = M.SelectCard(0, False, 1, 2, (card(1), card(2), card(3)))
    st = make_decision(msg)
    assert kinds(st) == ["select"] * 3  # nothing chosen yet: no finish, no cancel
    assert st.step(2) is None
    assert kinds(st) == ["select", "select", "finish"]
    assert [a.card.code for a in st.actions()[:2]] == [1, 2]
    assert st.step(0) == cards_response(2, 0)  # max reached -> complete


def test_select_card_finish_and_cancel():
    st = make_decision(M.SelectCard(0, True, 1, 3, (card(1), card(2))))
    assert kinds(st) == ["select", "select", "cancel"]
    assert st.step(2) == i32(-1)
    st = make_decision(M.SelectCard(0, False, 1, 3, (card(1), card(2))))
    assert play(st, 1, 0) == cards_response(1, 0)  # no card left to add -> complete


def test_select_card_min_zero_finish_empty():
    st = make_decision(M.SelectCard(0, True, 0, 1, (card(1),)))
    assert kinds(st) == ["select", "finish"]
    assert st.step(1) == i32(-1)


def test_tribute_counts_release_param():
    opts = (M.TributeOption(1, M.Location(0, C.LOCATION_MZONE, 0), 2), M.TributeOption(2, M.Location(0, C.LOCATION_MZONE, 1), 1),
            M.TributeOption(3, M.Location(0, C.LOCATION_MZONE, 2), 1))  # fmt: skip
    st = make_decision(M.SelectTribute(0, False, 2, 2, opts))
    assert kinds(st) == ["select"] * 3
    assert st.step(0) is None  # a double tribute already covers min=2
    assert "finish" in kinds(st)
    assert st.step(find(st, "finish")) == cards_response(0)
    st = make_decision(M.SelectTribute(0, False, 2, 2, opts))
    assert st.step(1) is None and "finish" not in kinds(st)
    assert st.step(find(st, "select", index=2)) == cards_response(1, 2)


def sums(*values, must=()):
    def opt(i, v):
        lo, hi = (v, 0) if isinstance(v, int) else v
        return M.SumOption(100 + i, M.Location(0, C.LOCATION_HAND, i), lo | (hi << 16))

    return tuple(opt(i, v) for i, v in enumerate(must)), tuple(opt(i, v) for i, v in enumerate(values))


def test_select_sum_exact_prunes_dead_ends():
    must, cards = sums(3, 4, 5)
    st = make_decision(M.SelectSum(0, True, 8, 1, 2, must, cards))
    assert [a.index for a in st.actions() if a.kind == "select"] == [0, 2]  # 4 cannot reach 8
    assert st.step(find(st, "select", index=0)) is None
    assert kinds(st) == ["select"] and st.actions()[0].index == 2
    assert st.step(0) == cards_response(0, 2)


def test_select_sum_exact_with_must_and_alternative_values():
    must, cards = sums(3, (1, 6), must=(2,))
    st = make_decision(M.SelectSum(0, True, 8, 1, 1, must, cards))
    assert [a.index for a in st.actions() if a.kind == "select"] == [1]  # 2 + 6 = 8
    assert st.step(0) == cards_response(1)


def test_select_sum_at_least_is_minimal():
    _, cards = sums(5, 4, 3)
    st = make_decision(M.SelectSum(0, False, 8, 0, 0, (), cards))
    assert sorted(a.index for a in st.actions() if a.kind == "select") == [0, 1, 2]
    st.step(find(st, "select", index=1))  # 4
    assert [a.index for a in st.actions() if a.kind == "select"] == [0]  # 4+3 too small, 4+3+5 not minimal
    st.step(0)
    assert st.response == cards_response(1, 0)


def test_unselect_card():
    st = make_decision(M.SelectUnselectCard(0, True, False, 1, 3, (card(1), card(2)), (card(3),)))
    assert kinds(st) == ["select", "select", "unselect", "finish"]
    assert make_decision(st.decision).step(2) == i32(1, 2)
    assert make_decision(st.decision).step(3) == i32(-1)


def test_counter_distribution():
    opts = (M.CounterOption(1, M.Location(0, C.LOCATION_SZONE, 0), 3), M.CounterOption(2, M.Location(0, C.LOCATION_SZONE, 1), 1))
    st = make_decision(M.SelectCounter(0, 0x1, 2, opts))
    assert kinds(st) == ["counter", "counter"]
    st.step(1)
    assert kinds(st) == ["counter"]  # card 2 has no counters left
    assert st.step(0) == struct.pack("<2h", 1, 1)


def test_sort_cards():
    st = make_decision(M.SortCard(0, (card(1), card(2), card(3))))
    assert kinds(st) == ["sort", "sort", "sort", "default"]
    assert make_decision(st.decision).step(3) == struct.pack("<b", -1)
    # pick card 3 first, then card 1; card 2 is placed last automatically.
    # The response gives each card's new position.
    assert play(st, 2, 0) == struct.pack("<3b", 1, 2, 0)


def test_select_place_only_free_zones():
    free = (1 << 2) | (1 << 9) | (1 << 17)  # own mzone 2, own szone 1, opponent mzone 1
    st = make_decision(M.SelectPlace(0, 2, ~free & 0xFFFFFFFF))
    zones = [(a.value >> 16, a.value >> 8 & 0xFF, a.value & 0xFF) for a in st.actions()]
    assert zones == [(0, C.LOCATION_MZONE, 2), (0, C.LOCATION_SZONE, 1), (1, C.LOCATION_MZONE, 1)]
    st.step(0)
    assert len(st.actions()) == 2
    assert st.step(1) == bytes([0, C.LOCATION_MZONE, 2, 1, C.LOCATION_MZONE, 1])


def test_select_place_for_player_one_is_relative():
    st = make_decision(M.SelectPlace(1, 1, ~(1 << 0) & 0xFFFFFFFF))
    assert st.step(0) == bytes([1, C.LOCATION_MZONE, 0])


def test_announce_race_and_attribute():
    st = make_decision(M.AnnounceRace(0, 2, C.RACE_DRAGON | C.RACE_ZOMBIE | C.RACE_FIEND))
    assert [a.value for a in st.actions()] == [C.RACE_FIEND, C.RACE_ZOMBIE, C.RACE_DRAGON]
    st.step(2)
    assert [a.value for a in st.actions()] == [C.RACE_FIEND, C.RACE_ZOMBIE]
    assert st.step(0) == struct.pack("<Q", C.RACE_DRAGON | C.RACE_FIEND)
    st = make_decision(M.AnnounceAttrib(0, 1, C.ATTRIBUTE_LIGHT | C.ATTRIBUTE_DARK))
    assert st.step(1) == struct.pack("<I", C.ATTRIBUTE_DARK)


# ------------------------------------------------------------------ announce card


def mk(password, type_=C.TYPE_MONSTER | C.TYPE_EFFECT, alias=0, setcodes=(), race=C.RACE_DRAGON, attr=C.ATTRIBUTE_LIGHT):
    return Card(password, f"c{password}", "", ("",) * 16, alias, 3, tuple(setcodes), type_, 0, 0, 4, 0, 0, race, attr, 0, 0)


POOL = {
    c.password: c
    for c in [
        mk(1),
        mk(2, type_=C.TYPE_SPELL),
        mk(3, alias=1),
        mk(4, type_=C.TYPE_MONSTER | C.TYPE_TOKEN),
        mk(5, setcodes=(0x10DD,)),
        mk(6, setcodes=(0xDD,), race=C.RACE_FIEND),
    ]
}


def test_is_declarable():
    assert is_declarable(POOL[1], (1, C.OPCODE_ISCODE))
    assert not is_declarable(POOL[2], (1, C.OPCODE_ISCODE))
    assert is_declarable(POOL[2], (C.TYPE_SPELL, C.OPCODE_ISTYPE))
    both = (C.TYPE_MONSTER, C.OPCODE_ISTYPE, C.RACE_DRAGON, C.OPCODE_ISRACE, C.OPCODE_AND)
    assert is_declarable(POOL[1], both) and not is_declarable(POOL[6], both)
    anything = (C.TYPE_MONSTER | C.TYPE_SPELL | C.TYPE_TRAP, C.OPCODE_ISTYPE)
    assert not is_declarable(POOL[3], anything)  # aliases need OPCODE_ALLOW_ALIASES
    assert is_declarable(POOL[3], anything + (C.OPCODE_ALLOW_ALIASES,))
    assert not is_declarable(POOL[4], anything)  # tokens need OPCODE_ALLOW_TOKENS
    assert is_declarable(POOL[4], anything + (C.OPCODE_ALLOW_TOKENS,))
    # setcode 0xDD matches both 0xDD and its sub-archetype 0x10DD; 0x10DD only the latter
    assert is_declarable(POOL[5], (0xDD, C.OPCODE_ISSETCARD)) and is_declarable(POOL[6], (0xDD, C.OPCODE_ISSETCARD))
    assert is_declarable(POOL[5], (0x10DD, C.OPCODE_ISSETCARD)) and not is_declarable(POOL[6], (0x10DD, C.OPCODE_ISSETCARD))
    assert not is_declarable(POOL[1], ())  # empty program -> stack size != 1


def test_announce_card_candidates():
    st = make_decision(M.AnnounceCard(0, (C.TYPE_MONSTER, C.OPCODE_ISTYPE)), cards=POOL)
    assert sorted(a.value for a in st.actions()) == [1, 5, 6]
    assert st.step(find(st, "declare", value=5)) == i32(5)


def test_announce_card_requires_card_pool():
    with pytest.raises(ValueError, match="card"):
        make_decision(M.AnnounceCard(0, (1, C.OPCODE_ISCODE)))


# ------------------------------------------------------------------------ misc


def test_step_after_completion_and_bad_index():
    st = make_decision(M.SelectYesNo(0, 1))
    with pytest.raises(IndexError):
        st.step(5)
    st.step(0)
    assert st.done
    with pytest.raises(RuntimeError, match="complete"):
        st.step(0)


def test_every_decision_type_is_supported():
    from ygorl.engine.actions import SUPPORTED_DECISIONS

    assert {cls.TYPE for cls in SUPPORTED_DECISIONS} == M.DECISION_TYPES


def test_action_is_hashable_value_object():
    a = Action("yes", value=1)
    assert a == Action("yes", value=1) and hash(a) == hash(Action("yes", value=1))
