"""Solo self-play (design 10 「自博弈的架构」): the ygopro shadow duel and the MD request bridge, offline."""

import pytest

from ygorl.advisor.reconstruct import OpponentPrior
from ygorl.advisor.snapshot import BoardSnapshot, SnapshotCard, SnapshotSide
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.duel import default_cards
from ygorl.engine.tracker import DecisionPoint
from ygorl.selfplay.bridge import Option, Request, action_key, align, find
from ygorl.selfplay.shadow import ShadowDuel, ShadowError, auto_answer, opponent_answer

KING = 64788463  # King's Knight
QUEEN = 25652259  # Queen's Knight
JACK = 90876561  # Jack's Knight
JOKER = 29284413  # Joker's Knight
FACE_CARD_FUSION = 29062925
ARCANA_KNIGHT_JOKER = 6150044  # Fusion: Queen's Knight + Jack's Knight + King's Knight
BLUE_EYES = 89631139
SOLEMN = 41420027  # Solemn Judgment: the sampled opponent card that can always answer


@pytest.fixture(scope="module")
def cards():
    return default_cards()


def shadow(me, deck, extra=(), deck_count=6, cards=None):
    """The advisor's Main Phase 1 against a Set card and two hand cards, all sampled as Solemn Judgment."""
    op = (SnapshotCard("szone", 0, 2, "facedown"), SnapshotCard("hand"), SnapshotCard("hand"))
    snap = BoardSnapshot(turn=3, phase="main1", sides=(SnapshotSide(8000, tuple(me), deck_count, len(extra)),
                                                       SnapshotSide(8000, op, 20, 0))).check()  # fmt: skip
    own = Deck(tuple(c.password for c in me) + tuple(deck), tuple(extra))
    prior = OpponentPrior((Deck((SOLEMN,) * 40),), (1.0,))
    return ShadowDuel.from_snapshot(snap, own, prior, C.DUEL_MODE_MR5, cards=cards, rng=0)


def kinds(point):
    return [(a.kind, a.card.code if a.card is not None else None) for a in point.actions]


# -- shadow duel ---------------------------------------------------------------------------------------------------


def test_face_card_fusion_flow_matches_master_duel(cards):
    """The live case of design 10: Face Card Fusion with King's Knight on the field, Jack's Knight in hand and the
    Queen's Knight in the Deck. The shadow stops exactly where MD asked."""
    me = [SnapshotCard("mzone", KING, 0), SnapshotCard("mzone", JOKER, 1), SnapshotCard("hand", JACK),
          SnapshotCard("hand", FACE_CARD_FUSION), SnapshotCard("hand", BLUE_EYES)]  # fmt: skip
    sh = shadow(me, (QUEEN,) * 3 + (BLUE_EYES,) * 3, (ARCANA_KNIGHT_JOKER,) * 3, cards=cards)
    assert isinstance(sh.point.decision, M.SelectIdleCmd) and not sh.log

    # activate: a Spell & Trap Zone (MD asks the zone)
    assert sh.pick(Option(FACE_CARD_FUSION, "hand"), "activate") == []
    assert isinstance(sh.point.decision, M.SelectPlace) and len(sh.point.actions) == 5

    # the opponent's chain window (the sampled Solemn Judgment could answer) and the advisor's (only pass): not asked
    auto = sh.step(0)
    assert [(s.player, s.decision, s.reason, s.action.kind, s.could_respond) for s in auto] == [
        (1, "SelectChain", "opponent", "pass", True),
        (0, "SelectChain", "single", "pass", False),
    ]

    # the Fusion Monster: three copies in the Extra Deck, still a list in MD
    point = sh.point
    assert isinstance(point.decision, M.SelectCard)
    assert kinds(point) == [("select", ARCANA_KNIGHT_JOKER)] * 3
    assert sh.step(0) == []

    # materials: one SelectUnselectCard per pick, as MD asks them; the Deck's Queen's Knight is filled in
    point = sh.point
    assert isinstance(point.decision, M.SelectUnselectCard)
    assert sorted(c for _, c in kinds(point) if c) == sorted([QUEEN] * 3 + [JACK, KING])
    assert sh.pick(Option(JACK, "hand")) == []
    assert isinstance(sh.point.decision, M.SelectUnselectCard)
    auto = sh.pick(Option(KING, "mzone", 0))
    assert [(s.decision, s.reason, s.action.card.code) for s in auto] == [("SelectUnselectCard", "equivalent", QUEEN)]

    # the Fusion Monster's zone and position are asked; the chain windows after the summon are not
    assert isinstance(sh.point.decision, M.SelectPlace)
    assert sh.step(0) == []
    point = sh.point
    assert isinstance(point.decision, M.SelectPosition) and len(point.actions) == 2
    auto = sh.step(0)
    assert {(s.player, s.reason, s.action.kind) for s in auto} == {(0, "single", "pass"), (1, "opponent", "pass")}
    point = sh.point
    assert isinstance(point.decision, M.SelectIdleCmd)
    assert ("activate", FACE_CARD_FUSION) not in kinds(point)
    assert [s.reason for s in sh.log].count("opponent") == 2


def test_normal_summon_into_the_last_zone_is_not_asked(cards):
    """Four Monster Zones taken: the summon's zone is the only one left and the chain windows only offer pass."""
    me = [SnapshotCard("mzone", BLUE_EYES, s) for s in range(4)] + [SnapshotCard("hand", QUEEN)]
    sh = shadow(me, (BLUE_EYES,) * 6, cards=cards)
    assert sh.pick(Option(QUEEN, "hand"), "summon")[0].decision == "SelectPlace"
    auto = sh.log
    assert (auto[0].reason, auto[0].action.value) == ("single", C.LOCATION_MZONE << 8 | 4)
    assert all(s.decision == "SelectChain" and s.action.kind == "pass" for s in auto[1:])
    opponent = [s for s in auto if s.player == 1]
    assert opponent and any(s.could_respond for s in opponent)  # Solemn Judgment could negate the summon
    point = sh.point
    assert isinstance(point.decision, M.SelectIdleCmd)
    assert ("summon", QUEEN) not in kinds(point)


def test_shadow_stops_when_the_opponent_turn_starts(cards):
    sh = shadow([SnapshotCard("hand", BLUE_EYES)], (BLUE_EYES,) * 6, cards=cards)
    end = next(i for i, a in enumerate(sh.point.actions) if a.kind == "end_phase")
    sh.step(end)
    assert sh.done and sh.stopped == "opponent_turn" and sh.point is None
    with pytest.raises(ShadowError, match="stopped"):
        sh.step(0)


# -- auto-answer rules on hand-built decisions ---------------------------------------------------------------------


def loc(location, sequence=0, controller=0):
    return M.Location(controller, location, sequence)


def point_of(decision, picks=()):
    state = make_decision(decision)
    for i in picks:
        state.step(i)
    return DecisionPoint(0, decision.player, 3, C.PHASE_MAIN1, (8000, 8000), decision, state.actions(), state, ())


def test_auto_answer_rules():
    deck = [M.CardInfo(QUEEN, loc(C.LOCATION_DECK, s)) for s in (0, 4, 5)]
    king = M.CardInfo(KING, loc(C.LOCATION_MZONE, 0))
    # a fresh decision between copies is asked (MD listed the three Arcana Knight Joker)
    assert auto_answer(point_of(M.SelectCard(0, False, 1, 1, tuple(deck)))) is None
    # ... but copies left in a multi-select under way are filled in
    assert auto_answer(point_of(M.SelectCard(0, False, 2, 2, (king, *deck)), picks=[0])) == ("equivalent", 0)
    assert auto_answer(point_of(M.SelectUnselectCard(0, False, False, 3, 3, tuple(deck), (king,)))) == (
        "equivalent", 0)  # fmt: skip
    # one option besides cancel
    assert auto_answer(point_of(M.SelectCard(0, True, 1, 1, (king,)))) == ("single", 0)
    # finishing early is a choice of its own
    assert auto_answer(point_of(M.SelectCard(0, False, 1, 2, (king, *deck)), picks=[0])) is None
    # command points are always asked
    idle = M.SelectIdleCmd(0, (), (), (), (), (), (), False, True, False)
    assert auto_answer(point_of(idle)) is None


def test_opponent_answer_declines_and_reports_a_swallowed_response():
    chain = M.SelectChain(1, 0, False, 0, 0, (M.ChainOption(SOLEMN, loc(C.LOCATION_SZONE, 2, 1), 0, 0),))
    assert opponent_answer(point_of(chain)) == (1, True)
    assert opponent_answer(point_of(M.SelectChain(1, 0, False, 0, 0, ()))) == (0, False)
    forced = M.SelectCard(1, False, 1, 1, (M.CardInfo(KING, loc(C.LOCATION_MZONE, 0, 1)),))
    assert opponent_answer(point_of(forced)) == (0, False)


# -- bridge --------------------------------------------------------------------------------------------------------


def materials():
    deck = tuple(M.CardInfo(QUEEN, loc(C.LOCATION_DECK, s)) for s in (5, 4, 0))
    return point_of(M.SelectUnselectCard(0, False, True, 3, 3, (*deck, M.CardInfo(JACK, loc(C.LOCATION_HAND, 0)),
                                                                  M.CardInfo(KING, loc(C.LOCATION_MZONE, 0))), ()))  # fmt: skip


def test_align_pairs_md_options_with_actions():
    point = materials()
    req = Request("select_card", [Option(JACK, "hand"), Option(KING, "mzone", 0)] + [Option(QUEEN, "deck")] * 3, 3, 3)
    al = align(req, point)
    assert al.kind_match and al.consistent
    assert [o for o, _ in al.pairs] == [0, 1, 2, 3, 4]
    assert {point.actions[i].card.code for _, idx in al.pairs for i in idx} == {JACK, KING, QUEEN}
    assert len({idx for _, idx in al.pairs}) == 5  # each copy paired once
    assert [point.actions[i].kind for i in al.controls] == ["cancel"]


def test_align_reports_both_differences_with_multiplicity():
    point = materials()
    # MD lists two Queen's Knights, the King's Knight on another zone and a card ygopro does not offer
    req = Request("select_card", [Option(QUEEN, "deck"), Option(QUEEN, "deck"), Option(KING, "mzone", 1),
                                  Option(BLUE_EYES, "hand")])  # fmt: skip
    al = align(req, point)
    assert al.kind_match and not al.consistent
    assert [o for o, _ in al.pairs] == [0, 1]
    assert al.md_only == [2, 3]
    left = sorted(point.actions[idx[0]].card.code for idx in al.ygo_only)
    assert left == sorted([QUEEN, JACK, KING])


def test_align_exact_sequence_before_wildcards():
    point = materials()
    al = align(Request("select_card", [Option(QUEEN, "deck"), Option(QUEEN, "deck", 0)]), point)
    by_option = {o: point.actions[idx[0]].card.loc.sequence for o, idx in al.pairs}
    assert by_option[1] == 0 and by_option[0] != 0


def test_align_kind_mismatch_and_value_options():
    position = point_of(M.SelectPosition(0, ARCANA_KNIGHT_JOKER, C.POS_FACEUP_ATTACK | C.POS_FACEUP_DEFENSE))
    al = align(Request("position", [Option(0, "", value=C.POS_FACEUP_ATTACK), Option(0, "", value=C.POS_FACEUP_DEFENSE)]),
               position)  # fmt: skip
    assert al.consistent and len(al.pairs) == 2
    al = align(Request("select_card", [Option(JACK, "hand")]), position)
    assert not al.kind_match and al.md_only == [0] and len(al.ygo_only) == 2
    with pytest.raises(ValueError, match="unknown MD request kind"):
        align(Request("dance"), position)


def test_place_options_and_commands_by_card():
    place = point_of(M.SelectPlace(0, 1, ~(1 << 8 | 1 << 9) & 0xFFFFFFFF))
    assert [action_key(a) for a in place.actions] == [("card", 0, 0, "szone", 0), ("card", 0, 0, "szone", 1)]
    assert find(place, Option(0, "szone", 1)) == 1
    hand = (M.CardInfo(JACK, loc(C.LOCATION_HAND, 0)),)
    idle = point_of(M.SelectIdleCmd(0, hand, (), (), hand, (), (), True, True, False))
    al = align(Request("idle", [Option(JACK, "hand")]), idle)
    assert al.consistent and [idle.actions[i].kind for i in al.pairs[0][1]] == ["summon", "mset"]
    assert find(idle, Option(JACK, "hand"), "mset") == 1
    assert [idle.actions[i].kind for i in al.controls] == ["battle_phase", "end_phase"]


def test_md_requests_map_to_bridge_requests():
    from ygorl.md.request import Candidate
    from ygorl.md.request import Request as MDRequest
    from ygorl.selfplay.observe import POSITION_DIALOG, to_bridge

    pw = {5558: 64788463, 5559: 90876561}
    req = MDRequest(kind="select_card", mode=8, accepting=True, player=0, min=1, max=1,
                    candidates=(Candidate(5558, 32, 0, "mzone", 0), Candidate(5559, 12, 1, "hand", 2)))  # fmt: skip
    ask = to_bridge(req, 0, pw)
    assert ask.kind == "select_card" and (ask.min, ask.max) == (1, 1)
    assert [(o.password, o.location, o.sequence, o.controller) for o in ask.options] == [
        (64788463, "mzone", 0, 0), (90876561, "hand", 2, 1)]  # fmt: skip
    assert to_bridge(MDRequest(kind="chain", mode=5, accepting=True, player=0), 0, pw).kind == "chain"
    assert to_bridge(MDRequest(kind="position", mode=7, accepting=True, player=0, dialog_type=POSITION_DIALOG),
                     0, pw).kind == "position"  # fmt: skip
    assert to_bridge(MDRequest(kind="position", mode=7, accepting=True, player=0), 0, pw).kind == "place"
    assert to_bridge(MDRequest(kind="draw", mode=1, accepting=True, player=0), 0, pw) is None


def test_lockstep_answers_md_dialogs_through_the_shadow(cards):
    """design 10 「边角情况由 player 自己学会」: MD's dialogs inside our action are answered by the policy on the
    shadow's decision point and mapped back to MD's options; a dialog the shadow does not ask drops it."""
    import numpy as np

    from ygorl.md.request import Candidate
    from ygorl.md.request import Request as MDRequest
    from ygorl.selfplay.observe import ShadowObserver

    me = [SnapshotCard("mzone", KING, 0), SnapshotCard("mzone", JOKER, 1), SnapshotCard("hand", JACK),
          SnapshotCard("hand", FACE_CARD_FUSION), SnapshotCard("hand", BLUE_EYES)]  # fmt: skip
    sh = shadow(me, (QUEEN,) * 3 + (BLUE_EYES,) * 3, (ARCANA_KNIGHT_JOKER,) * 3, cards=cards)
    sh.pick(Option(FACE_CARD_FUSION, "hand"), "activate")
    sh.step(0)  # the zone
    sh.step(0)  # the Fusion Monster
    assert isinstance(sh.point.decision, M.SelectUnselectCard)

    def prefer(code=None, value=None):  # a scorer that likes one card / one value
        def score(shadow_duel, point):
            return np.array([1.0 if (a.card is not None and a.card.code == code and a.kind == "select")
                             or (value is not None and a.value == value) else 0.0 for a in point.actions])  # fmt: skip

        return score

    pw = {1: JACK, 2: KING}
    obs = ShadowObserver(C.DUEL_MODE_MR5, cards, None, prefer(KING))
    obs.shadow = sh
    req = MDRequest(kind="mode0", mode=0, accepting=False, player=0,
                    candidates=(Candidate(1, 10, 0, "hand", 0), Candidate(2, 11, 0, "mzone", 0)))  # fmt: skip
    pick, record = obs.choose("list", req, 0, pw)
    assert (
        pick == 1 and record["action"][1] == KING and record["options"] == 2
    )  # the King's Knight, second in MD's list
    assert isinstance(sh.point.decision, M.SelectUnselectCard)  # the next material
    obs.scorer = prefer(JACK)
    pick, _ = obs.choose("list", req, 0, pw)
    assert pick == 0
    assert isinstance(sh.point.decision, M.SelectPlace)  # the Queen's Knight was filled in, then the Fusion's zone
    sh.step(0)

    # the position dialog: the policy wants face-up Defense -> the right button
    obs.scorer = prefer(value=C.POS_FACEUP_DEFENSE)
    pick, record = obs.choose("position", None, 0, pw)
    assert pick == 1 and record["ygo"] == "SelectPosition"

    # a yes / no the shadow does not ask: the shadow is dropped and the caller's rule decides
    obs.shadow = shadow(me, (QUEEN,) * 3, cards=cards)
    pick, record = obs.choose("yesno", None, 0, pw)
    assert pick is None and record["desync"] and obs.shadow is None
    assert obs.choose("yesno", None, 0, pw) == (None, {})


def test_md_option_descriptions_pair_with_ygopro_options():
    from ygorl.selfplay.observe import md_option_desc, option_pairs

    pw = {0x4064: 1234}
    assert md_option_desc(0x800A4064, pw) == 1234 << 20 | 0  # card string 10 -> aux.Stringid(1234, 0)
    assert md_option_desc(0x800B4064, pw) == 1234 << 20 | 1
    assert md_option_desc(0xC00001AC, pw) is None  # a system string
    assert option_pairs([1234 << 20 | 1, 1234 << 20], [1234 << 20, 1234 << 20 | 1]) == {0: 1, 1: 0}
    assert option_pairs([None, None], [7, 8]) == {0: 0, 1: 1}  # by order when the counts agree
    assert option_pairs([None, None, None], [7, 8]) is None
