"""In-action decisions of the Solo player re-anchored on the current board (design 10 「识别与执行的系统化」), offline:
rebuilt attack targets, Summon positions and activate-or-pass prompts, and the tolerant lockstep of the shadow."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ygorl.advisor.reconstruct import OpponentPrior
from ygorl.advisor.snapshot import BoardSnapshot, SnapshotCard, SnapshotSide
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import default_cards
from ygorl.md.request import Candidate
from ygorl.md.request import Request as MDRequest
from ygorl.selfplay.audit import load_offsets, raw_from_digest
from ygorl.selfplay.decide import (
    Clock,
    Decider,
    attack_target,
    in_limbo,
    respond,
    situation,
    summon_position,
    summoned_card,
)
from ygorl.selfplay.observe import ShadowObserver, md_option_desc, option_of_description
from ygorl.selfplay.shadow import ShadowDuel

KING, QUEEN, JACK, BLUE_EYES = 64788463, 25652259, 90876561, 89631139
RAS_DISCIPLE = 74875003  # only Ra's Disciples' own effects may Special Summon while it is on the field
COURT_OF_CARDS = 28340377  # two quick effects: aux.Stringid(id, 0) and (id, 1)
MIRROR_FORCE = 44095762
MST = 5318639  # Mystical Space Typhoon
KNIGHTS = OpponentPrior((Deck((KING,) * 20 + (QUEEN,) * 20),), (1.0,))


@pytest.fixture(scope="module")
def cards():
    return default_cards()


def board(me, op, phase="main1", turn=3, deck=6, extra=0):
    return BoardSnapshot(turn=turn, phase=phase, sides=(SnapshotSide(8000, tuple(me), deck, extra),
                                                        SnapshotSide(8000, tuple(op), 20, 0))).check()  # fmt: skip


def prefer(code=None, value=None, kinds=()):
    """A scorer (core, point, clock) -> logits that likes one card, one value or some action kinds."""

    def score(core, point, clock):
        return np.array([5.0 if (a.card is not None and a.card.code == code) or (value is not None and a.value == value)
                         or a.kind in kinds else 0.0 for a in point.actions])  # fmt: skip

    return score


OPP_MONSTERS = (SnapshotCard("mzone", KING, 1), SnapshotCard("mzone", QUEEN, 3, "faceup_defense"),
                SnapshotCard("mzone", 0, 2, "facedown_defense"), SnapshotCard("hand"))  # fmt: skip


# -- rebuilt decisions ---------------------------------------------------------------------------------------------


def test_attack_target_is_the_policys_and_the_shadow_goes_on(cards):
    snap = board([SnapshotCard("mzone", BLUE_EYES, 0), SnapshotCard("hand", JACK)], OPP_MONSTERS, phase="battle")
    own = Deck((BLUE_EYES,) * 8 + (JACK,), ())
    for want, zone in ((QUEEN, 3), (KING, 1)):
        choice = attack_target(snap, own, KNIGHTS, C.DUEL_MODE_MR5, 0, cards=cards, scorer=prefer(want), samples=3)
        assert choice.pick == zone and choice.reached == 3
        assert set(choice.probs) == {1, 2, 3} and abs(sum(choice.probs.values()) - 1) < 1e-6
        # the first sample's shadow took the target: MD's next question (the next battle command) is its next
        point = choice.shadow.point
        assert point is None or isinstance(point.decision, M.SelectBattleCmd)
    choice = attack_target(snap, own, KNIGHTS, C.DUEL_MODE_MR5, 4, cards=cards, samples=1)
    assert choice.pick is None and "no attack from Monster Zone 4" in choice.notes[0]


def test_direct_attack_when_the_opponent_has_no_monster(cards):
    snap = board([SnapshotCard("mzone", BLUE_EYES, 0)], (SnapshotCard("hand"),), phase="battle")
    choice = attack_target(snap, Deck((BLUE_EYES,) * 8, ()), KNIGHTS, C.DUEL_MODE_MR5, 0, cards=cards, samples=1)
    assert choice.pick == "direct"


def test_summon_position_asks_the_core(cards):
    snap = board([SnapshotCard("mzone", BLUE_EYES, 0), SnapshotCard("hand", KING)], OPP_MONSTERS)
    own = Deck((BLUE_EYES,) * 6 + (JACK,) * 3 + (KING,), ())
    defense = summon_position(snap, own, KNIGHTS, C.DUEL_MODE_MR5, JACK, cards=cards,
                              scorer=prefer(value=C.POS_FACEUP_DEFENSE), samples=2)  # fmt: skip
    assert (defense.pick, defense.reached, defense.detail["card"]) == (1, 2, JACK)
    attack = summon_position(snap, own, KNIGHTS, C.DUEL_MODE_MR5, JACK, cards=cards,
                             scorer=prefer(value=C.POS_FACEUP_ATTACK), samples=2)  # fmt: skip
    assert attack.pick == 0
    # a lock that lets only the card's own effects Special Summon (the live audit: 3 Disciples left in Attack)
    snap = board([SnapshotCard("mzone", RAS_DISCIPLE, 0)], OPP_MONSTERS)
    disciple = summon_position(snap, Deck((RAS_DISCIPLE,) * 3 + (BLUE_EYES,) * 6, ()), KNIGHTS, C.DUEL_MODE_MR5,
                               RAS_DISCIPLE, cards=cards, scorer=prefer(value=C.POS_FACEUP_DEFENSE), samples=1)  # fmt: skip
    assert disciple.pick == 1 and not disciple.notes
    # no free Monster Zone: the core asks nothing, the rule decides
    full = board([SnapshotCard("mzone", BLUE_EYES, s) for s in range(5)], OPP_MONSTERS)
    none = summon_position(full, Deck((BLUE_EYES,) * 8 + (JACK,), ()), KNIGHTS, C.DUEL_MODE_MR5, JACK, cards=cards,
                           samples=1)  # fmt: skip
    assert none.pick is None and "asked no position" in none.notes[0]


TRAPS = [SnapshotCard("mzone", BLUE_EYES, 0), SnapshotCard("szone", MIRROR_FORCE, 0, "facedown"),
         SnapshotCard("szone", MST, 1, "facedown")]  # fmt: skip
TRAPS_DECK = Deck((BLUE_EYES,) * 6 + (MIRROR_FORCE, MST), ())
TWO_ATTACKERS = (SnapshotCard("mzone", KING, 1), SnapshotCard("mzone", QUEEN, 2), SnapshotCard("hand"))


def test_respond_reaches_the_opponents_windows(cards):
    snap = board(TRAPS, TWO_ATTACKERS)
    clock = Clock(4, True, 0)
    # the opponent's Battle Phase: the attack declaration offers Mirror Force (the window offering the most)
    got = respond(snap, TRAPS_DECK, KNIGHTS, C.DUEL_MODE_MR5, theirs=True, phase="battle", clock=clock, cards=cards,
                  scorer=prefer(MIRROR_FORCE), samples=2)  # fmt: skip
    assert got.pick == 0 and got.detail["card"] == MIRROR_FORCE and got.probs[0] > 0.9
    # MD offers MST only and the policy would rather pass
    got = respond(snap, TRAPS_DECK, KNIGHTS, C.DUEL_MODE_MR5, theirs=True, phase="battle", clock=clock, offered=(MST,),
                  cards=cards, scorer=prefer(kinds=("pass",)), samples=1)  # fmt: skip
    assert got.pick == 1 and got.detail["card"] == MST
    for phase in ("draw", "standby", "main1", "main2", "end"):  # every phase of the opponent's turn has a window
        got = respond(snap, TRAPS_DECK, KNIGHTS, C.DUEL_MODE_MR5, theirs=True, phase=phase, clock=clock, cards=cards,
                      scorer=prefer(MST), samples=1)  # fmt: skip
        assert got.pick == 0 and got.detail["card"] == MST, phase
    # our own End Phase is reachable; a prompt inside our own Main Phase action is not
    got = respond(snap, TRAPS_DECK, KNIGHTS, C.DUEL_MODE_MR5, theirs=False, phase="end", clock=clock, cards=cards,
                  scorer=prefer(MST), samples=1)  # fmt: skip
    assert got.pick == 0
    got = respond(snap, TRAPS_DECK, KNIGHTS, C.DUEL_MODE_MR5, theirs=False, phase="main1", clock=clock, cards=cards)
    assert got.pick is None and "inside an action" in got.notes[0]


def test_md_tabs_name_the_policys_activate_actions(cards):
    """MD's effect-choice tabs are card strings 10 + n of the card: aux.Stringid(password, n) = password << 20 | n,
    the description of ygopro's activate action for that effect."""
    me = [SnapshotCard("szone", COURT_OF_CARDS, 0, "faceup"), SnapshotCard("hand", QUEEN), SnapshotCard("grave", JACK)]
    sh = ShadowDuel.from_snapshot(board(me, (SnapshotCard("hand"),)), Deck((BLUE_EYES,) * 6 + (COURT_OF_CARDS, QUEEN,
                                  JACK), ()), KNIGHTS, C.DUEL_MODE_MR5, cards=cards)  # fmt: skip
    descs = {a.description for a in sh.point.actions if a.kind == "activate" and a.card.code == COURT_OF_CARDS}
    assert descs and descs <= {COURT_OF_CARDS << 20, COURT_OF_CARDS << 20 | 1}
    pw = {0x4064: COURT_OF_CARDS}
    tabs = (0x800A4064, 0x800B4064, 0xC00001AC)  # a live dialog: the two effects and Cancel
    assert [md_option_desc(c, pw) for c in tabs] == [COURT_OF_CARDS << 20, COURT_OF_CARDS << 20 | 1, None]
    for d in descs:
        assert option_of_description(tabs, d, pw) == d & 0xF
    assert option_of_description(tabs, 0, pw) is None


# -- the current board in any turn and phase -----------------------------------------------------------------------

PW = {1: BLUE_EYES, 2: MIRROR_FORCE, 3: MST, 4: KING, 5: QUEEN, 6: JACK}


def digest(turn_player="opp", phase="battle", turn=6, hand=(6,), opp_hand=3):
    """A trace step's memory digest (MD card ids of :data:`PW`)."""
    field = [None] * 13
    field[0], field[7], field[8] = [1, 1, 0], [2, 0, 0], [3, 0, 0]
    opp = [None] * 13
    opp[1], opp[2] = [4, 1, 0], [5, 1, 0]
    return {"turn": turn, "phase": phase, "turn_player": turn_player, "input": {"mode": 4},
            "memory": {"me": {"lp": 8000, "field": field, "hand": list(hand), "grave": [], "counts":
                              {"hand": len(hand), "deck": 30, "extra": 0, "grave": 0, "removed": 0}},
                       "opp": {"lp": 7000, "field": opp, "grave": [], "counts":
                               {"hand": opp_hand, "deck": 28, "extra": 0, "grave": 0, "removed": 0}}}}  # fmt: skip


def test_situation_reads_the_opponents_turn_as_the_turn_before():
    raw = raw_from_digest(digest(), load_offsets(), [1, 1, 6])
    sit = situation(raw, PW)
    assert (sit.theirs, sit.phase, sit.clock.turn, sit.clock.went_first) == (True, "battle", 6, True)
    snap = sit.snapshot
    assert (snap.turn, snap.phase, snap.turn_player, snap.went_first) == (5, "main1", 0, True)
    op = snap.sides[1]
    assert (op.hand_count, op.deck_count) == (2, 29)  # the rebuild's draw gives the card back
    assert {(c.location, c.password) for c in snap.sides[0].cards} >= {("mzone", BLUE_EYES), ("szone", MIRROR_FORCE)}
    mine = situation(raw_from_digest(digest("me", "end", 5), load_offsets(), [1]), PW)
    assert (mine.theirs, mine.phase, mine.snapshot.turn, mine.snapshot.went_first) == (False, "end", 5, True)
    first = situation(raw_from_digest(digest("opp", "main1", 1), load_offsets(), [1]), PW)
    assert (first.snapshot.turn, first.clock.went_first) == (1, False)  # the opponent went first


def test_decider_respond_on_a_recorded_step(cards):
    raw = raw_from_digest(digest(), load_offsets(), [1, 1, 1, 6])
    decider = Decider(C.DUEL_MODE_MR5, cards, PW, KNIGHTS, scorer=prefer(MIRROR_FORCE), samples=1)
    got = decider.respond(raw, SimpleNamespace(candidates=()))
    assert got.pick == 0 and got.detail["card"] == MIRROR_FORCE and got.record()["question"] == "respond"
    got = decider.respond(raw, SimpleNamespace(candidates=(SimpleNamespace(cid=3),)))  # MD offers MST only
    assert got.detail["card"] == MST
    target = decider.attack_target(raw_from_digest(digest("me", "battle", 5), load_offsets(), [1]), 0)
    assert target.pick in (1, 2)
    broken = decider.position(raw, SimpleNamespace(params=((5, 99),)))  # an id without a password: unknown card
    assert broken.pick is None and "not known" in broken.notes[0]


def test_the_summoned_monster_is_told_from_memory(cards):
    action = SimpleNamespace(kind="spsummon", card=SimpleNamespace(code=JACK))
    assert summoned_card(SimpleNamespace(params=((1, 599), (5, 5))), action, PW) == QUEEN  # the dialog names it
    assert summoned_card(SimpleNamespace(params=((1, 593), (8, 2))), action, PW) == JACK  # the policy's spsummon
    off = load_offsets()
    before = raw_from_digest(digest("me", "main1", hand=(6, 4)), off, [])
    now = raw_from_digest(digest("me", "main1", hand=(4,)), off, [])
    assert in_limbo(before, now, PW, cards) == {JACK}
    assert summoned_card(None, None, PW, before=before, now=now, cards=cards) == JACK
    # summoned from the hand to the field already: not in limbo
    field = digest("me", "main1", hand=(4,))
    field["memory"]["me"]["field"][1] = [6, 1, 0]
    assert in_limbo(before, raw_from_digest(field, off, []), PW, cards) == set()
    # ... but the one monster that arrived on the field since the action (a Fusion Summon asked in its zone)
    assert summoned_card(None, None, PW, before=before, now=raw_from_digest(field, off, []), cards=cards) == JACK
    assert summoned_card(None, None, PW, cards=cards, listed=(6, 6)) == JACK  # MD listed only Jack's Knights
    assert summoned_card(None, None, PW, cards=cards, listed=(6, 2)) == 0
    # a monster's own effect from the hand that summons it (asked while it is still in the hand)
    from ygorl.engine import constants as C

    own = SimpleNamespace(
        kind="activate", card=SimpleNamespace(code=JACK, loc=SimpleNamespace(location=C.LOCATION_HAND))
    )
    assert summoned_card(None, own, PW, cards=cards) == JACK


# -- the tolerant lockstep -----------------------------------------------------------------------------------------


def fusion_shadow(cards):
    """Face Card Fusion (the live case of design 10) answered up to the Fusion Monster's zone."""
    from ygorl.selfplay.bridge import Option

    joker, fusion, arcana = 29284413, 29062925, 6150044
    me = [SnapshotCard("mzone", KING, 0), SnapshotCard("mzone", joker, 1), SnapshotCard("hand", JACK),
          SnapshotCard("hand", fusion), SnapshotCard("hand", BLUE_EYES)]  # fmt: skip
    op = (SnapshotCard("szone", 0, 2, "facedown"), SnapshotCard("hand"), SnapshotCard("hand"))
    own = Deck(tuple(c.password for c in me) + (QUEEN,) * 3 + (BLUE_EYES,) * 3, (arcana,) * 3)
    sh = ShadowDuel.from_snapshot(board(me, op, extra=3), own, OpponentPrior((Deck((41420027,) * 40),), (1.0,)),
                                  C.DUEL_MODE_MR5, cards=cards, rng=0)  # fmt: skip
    sh.pick(Option(fusion, "hand"), "activate")
    sh.step(0)  # the Spell's zone
    sh.step(0)  # the Fusion Monster
    sh.pick(Option(JACK, "hand"))
    sh.pick(Option(KING, "mzone", 0))  # the Queen's Knight is filled in
    return sh


def test_position_before_zone_and_md_only_questions(cards):
    obs = ShadowObserver(C.DUEL_MODE_MR5, cards, None, lambda shadow, point: prefer(value=C.POS_FACEUP_DEFENSE)(
        None, point, None))  # fmt: skip
    obs.shadow = fusion_shadow(cards)
    assert isinstance(obs.shadow.point.decision, M.SelectPlace)  # ygopro: zone first; MD: position first
    pick, record = obs.choose("position", None, 0, {}, zone=("mzone", 2))
    assert pick == 1 and record["placed_first"] == "mzone 2" and record["ygo"] == "SelectPosition"
    assert obs.shadow is not None and isinstance(obs.shadow.point.decision, M.SelectIdleCmd)

    # back at the command point: MD lists one card the shadow never asked about -> nothing to decide, shadow kept
    one = MDRequest(kind="mode0", mode=0, accepting=False, player=0, candidates=(Candidate(9, 1, 0, "deck", 3),))
    pick, record = obs.choose("list", one, 0, {9: QUEEN})
    assert (pick, record["by"]) == (0, "forced") and obs.shadow is not None
    # the shadow filled a pick in by itself: MD asks it, the shadow's answer is the pick
    obs.pending = [{"card": JACK, "location": "deck", "decision": "SelectCard"}]
    two = MDRequest(kind="mode0", mode=0, accepting=False, player=0,
                    candidates=(Candidate(9, 1, 0, "deck", 3), Candidate(8, 2, 0, "deck", 5)))  # fmt: skip
    pick, record = obs.choose("list", two, 0, {9: QUEEN, 8: JACK})
    assert (pick, record["by"]) == (1, "auto") and obs.pending == []
    # the effect choice: the policy's activate action named the tab
    obs.activated = COURT_OF_CARDS << 20 | 1
    tabs = MDRequest(kind="mode0", mode=0, accepting=False, player=0, choices=(0x800A4064, 0x800B4064, 0xC00001AC))
    pick, record = obs.choose("option", tabs, 0, {0x4064: COURT_OF_CARDS})
    assert (pick, record["by"]) == (1, "activate") and obs.shadow is not None
    # anything else still drops the shadow
    pick, record = obs.choose("yesno", None, 0, {})
    assert pick is None and record["desync"] and obs.shadow is None


def test_an_optional_trigger_is_a_prompt(cards):
    """Ra's Disciple's trigger after its Normal Summon: ygopro asks yes / no, MD an activate-or-pass prompt."""
    from ygorl.selfplay.bridge import Option

    me = [SnapshotCard("hand", RAS_DISCIPLE)]
    sh = ShadowDuel.from_snapshot(board(me, (SnapshotCard("hand"),)), Deck((RAS_DISCIPLE,) * 3 + (BLUE_EYES,) * 6, ()),
                                  KNIGHTS, C.DUEL_MODE_MR5, cards=cards, rng=0)  # fmt: skip
    sh.pick(Option(RAS_DISCIPLE, "hand"), "summon")
    if isinstance(sh.point.decision, M.SelectPlace):
        sh.step(0)
    assert isinstance(sh.point.decision, M.SelectEffectYn)
    obs = ShadowObserver(C.DUEL_MODE_MR5, cards, None, lambda shadow, point: prefer(kinds=("no",))(None, point, None))
    obs.shadow = sh
    prompt = MDRequest(kind="open", mode=4, accepting=True, player=0, candidates=(Candidate(7, 1, 0, "mzone", 0),))
    pick, record = obs.choose("prompt", prompt, 0, {7: RAS_DISCIPLE})
    assert pick == 1 and record["action"][0] == "no"


def test_md_merges_the_trigger_into_its_list(cards):
    from ygorl.selfplay.bridge import Option

    me = [SnapshotCard("hand", RAS_DISCIPLE)]
    sh = ShadowDuel.from_snapshot(board(me, (SnapshotCard("hand"),)), Deck((RAS_DISCIPLE,) * 3 + (BLUE_EYES,) * 6, ()),
                                  KNIGHTS, C.DUEL_MODE_MR5, cards=cards, rng=0)  # fmt: skip
    sh.pick(Option(RAS_DISCIPLE, "hand"), "summon")
    if isinstance(sh.point.decision, M.SelectPlace):
        sh.step(0)
    obs = ShadowObserver(C.DUEL_MODE_MR5, cards, None, lambda shadow, point: prefer(RAS_DISCIPLE)(None, point, None))
    obs.shadow = sh
    deck = MDRequest(kind="mode0", mode=0, accepting=False, player=0,
                     candidates=(Candidate(7, 2, 0, "deck", 0), Candidate(7, 3, 0, "deck", 1)))  # fmt: skip
    pick, record = obs.choose("list", deck, 0, {7: RAS_DISCIPLE})
    assert pick is not None and record["ygo"] != "SelectEffectYn" and "policy_yes" in record


def test_trace_records_rebuilt_decisions(tmp_path):
    from ygorl.selfplay.trace import Trace, read, summarize

    t = Trace(tmp_path, run="r")
    raw = raw_from_digest(digest("me", "battle", 5), load_offsets(), [1])
    t.begin(0, raw, SimpleNamespace(mode=3, accepting=True, player=0), None)
    t.policy(SimpleNamespace(kind="attack", card=SimpleNamespace(code=BLUE_EYES)), 0.9, "battle")
    t.sub("target", by="rebuild", question="target", pick=1, probs={"1": 0.7, "2": 0.3})
    t.begin(1, raw, SimpleNamespace(mode=4, accepting=True, player=0), "list")
    t.rebuilt(question="respond", pick=0, probs={"0": 0.8, "1": 0.2})
    t.close()
    steps = [r for r in read(tmp_path / "r.jsonl") if r["kind"] == "step"]
    assert steps[0]["decision"]["by"] == "policy" and steps[0]["decision"]["target"]["by"] == "rebuild"
    assert steps[1]["decision"] == {"by": "rebuild", "question": "respond", "pick": 0, "probs": {"0": 0.8, "1": 0.2}}
    s = summarize(read(tmp_path / "r.jsonl"))["decisions"]
    assert s == {"policy:attack": 1, "target:rebuild": 1, "rebuild:respond": 1}
    json.dumps(steps)
