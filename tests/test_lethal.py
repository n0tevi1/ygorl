"""Lethal search on top of an agent (T4e.1 stage A, #62)."""

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import random

import pytest

from ygorl.agents import RandomAgent, make_agent
from ygorl.agents.lethal import LethalAgent, _signature
from ygorl.agents.greedy import GreedyAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.data.environment import PlayerRules
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, DecisionPoint
from ygorl.engine import constants as C
from ygorl.engine.actions import make_decision

CELTIC = 91152256  # Celtic Guardian, Level 4, 1400 ATK
DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


class Passive:
    """Never summons or attacks: ends every turn at the first chance."""

    def act(self, point):
        kinds = [a.kind for a in point.actions]
        return next((kinds.index(k) for k in ("end_phase", "pass", "no", "cancel", "finish") if k in kinds), 0)


def duel(db, lp=1000):
    cfg = DuelConfig(max_turns=4, shuffle_decks=False, player=PlayerRules(starting_lp=lp))
    return Duel(3, None, Deck(main=(CELTIC,) * 40), Deck(main=(CELTIC,) * 40), cards=db, config=cfg)


@pytest.mark.parametrize("policy_mask", [False, True])
def test_the_search_finds_the_lethal_the_wrapped_agent_would_not_play(db, policy_mask):
    """1000 LP and an empty board on the other side: summoning Celtic Guardian and attacking wins on turn 2."""
    plain = duel(db).run(Passive(), Passive())
    assert plain.reason == "turn_limit"
    agent = LethalAgent(Passive(), seed=0, policy_mask=policy_mask)
    result = duel(db).run(Passive(), agent)  # deck b moves second: its first turn is turn 2
    assert result.reason == "win" and result.winner == 1 and result.turns == 2
    s = agent.stats
    assert s.found >= 1 and s.overrides >= 3 and s.desync == 0  # summon, battle phase, attack, ...
    # with the lethal out of reach (8000 LP), it searches but finds nothing and changes nothing
    agent = LethalAgent(Passive(), seed=0, policy_mask=policy_mask)
    assert duel(db, lp=8000).run(Passive(), agent).reason == "turn_limit"
    assert agent.stats.searches > 0 and agent.stats.found == 0 and agent.stats.overrides == 0


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("mask", [(0, 1), (0, 0), (1,)])
def test_rollout_filters_before_choosing_and_preserves_engine_indices(db, greedy, mask):
    decision = M.SelectYesNo(0, 1)
    state = make_decision(decision, db)
    point = DecisionPoint(0, 0, 1, C.PHASE_MAIN1, (8000, 1000), decision, state.actions(), state, ())
    shadow = SimpleNamespace(point=point, done=False, core=None,
                             tracker=SimpleNamespace(events=(), turn=1, engine_winner=0,
                                                     result=SimpleNamespace(reason="win")))  # fmt: skip
    applied = []

    def advance(index):
        applied.append(index)
        shadow.done, shadow.point = True, None

    shadow.act = advance
    agent = LethalAgent(Passive(), policy_mask=True)
    agent.shadow = shadow
    agent._encoder = SimpleNamespace(encode=lambda p, core: {"action_mask": mask})
    chooser = GreedyAgent(0, cards=db) if greedy else random.Random(0)
    rng = chooser.rng if greedy else chooser
    before = rng.getstate()
    line, clean = agent._rollout(0, 1, chooser)
    assert clean
    if any(mask):
        index = mask.index(1)
        assert applied == [index]
        assert line == [(_signature(point), index)]
    else:
        assert line is None and not applied and rng.getstate() == before


def test_saved_line_rechecks_policy_support_before_applying_a_response(db):
    decision = M.SelectYesNo(0, 1)
    state = make_decision(decision, db)
    point = DecisionPoint(0, 0, 1, C.PHASE_MAIN1, (8000, 1000), decision, state.actions(), state, ())
    agent = LethalAgent(Passive(), policy_mask=True)
    agent.shadow = SimpleNamespace(core=None)
    agent._encoder = SimpleNamespace(encode=lambda p, core: {"action_mask": (0, 1)})
    agent.line = [(_signature(point), 0)]
    assert agent._from_line(point) is None
    assert not agent.line and agent.stats.abandoned == 1
    agent.line = [(_signature(point), 1)]
    assert agent._from_line(point) == 1


def test_strict_mode_rejects_lines_that_use_hidden_information():
    me = 0
    assert LethalAgent._clean([M.Draw(1, ((CELTIC, 10),))], me)  # the opponent's draw is public enough
    assert not LethalAgent._clean([M.Draw(0, ((CELTIC, 10),))], me)  # our own draw: the shadow knows the order
    assert LethalAgent._clean([], me)


@pytest.mark.parametrize("change", ["code", "instance", "effect", "value", "lp", "turn", "phase", "undo"])
def test_a_saved_line_is_abandoned_when_its_actions_or_context_change(change, db):
    cards = tuple(M.CardInfo(CELTIC, M.Location(1, C.LOCATION_MZONE, i, C.POS_FACEUP_ATTACK)) for i in (0, 1))
    decision = M.SelectCard(0, True, 1, 1, cards)
    state = make_decision(decision, db)
    point = DecisionPoint(7, 0, 2, C.PHASE_BATTLE_STEP, (8000, 1000), decision, state.actions(), state, ())
    actions = list(point.actions)
    if change == "code":
        actions[0] = replace(actions[0], card=replace(cards[0], code=46986414))
    elif change == "instance":
        actions[0] = replace(actions[0], card=cards[1])
        actions[1] = replace(actions[1], card=cards[0])
    elif change == "effect":
        actions[0] = replace(actions[0], description=123)
    elif change == "value":
        actions[0] = replace(actions[0], value=1)
    changed = replace(point, actions=actions)
    if change == "lp":
        changed = replace(point, lp=(8000, 2000))
    elif change == "turn":
        changed = replace(point, turn=3)
    elif change == "phase":
        changed = replace(point, phase=C.PHASE_MAIN2)
    elif change == "undo":
        changed = replace(point, undo=(0,))
    agent = LethalAgent(Passive())
    agent.line = [(_signature(point), 0)]
    assert agent._from_line(point) == 0
    assert [a.kind for a in changed.actions] == [a.kind for a in point.actions]
    assert agent.act(changed) == Passive().act(changed)
    assert not agent.line
    assert agent.stats.abandoned == 1 and agent.stats.overrides == 0


def test_opponent_deckout_after_turn_change_is_not_this_turn_lethal(db):
    class EndTurn(GreedyAgent):
        def act(self, point):
            return Passive().act(point)

    # B draws its entire five-card deck at setup, then loses at its first Draw Phase.
    game = Duel(3, None, Deck(main=(CELTIC,) * 40), Deck(main=(CELTIC,) * 5), cards=db,
                config=DuelConfig(max_turns=4, shuffle_decks=False), first=0)  # fmt: skip
    session = DuelSession(game)
    try:
        agent = LethalAgent(Passive())
        agent.shadow = session
        assert session.point.turn == 1
        line, _ = agent._rollout(0, 1, EndTurn(0, cards=db))
        assert line is None and session.point.turn == 2
        while session.point is not None:
            session.act(Passive().act(session.point))
        result = session.result()
        assert result.reason == "win" and result.winner == 0 and result.turns == 2
        assert result.win_reason == 2 and not result.error and not result.script_errors
        assert line is None
    finally:
        session.close()


def test_terminal_result_crossing_turn_boundary_without_another_prompt_is_rejected(db):
    # Simulate an atomic engine advance with no intervening decision to check.
    decision = M.SelectYesNo(0, 1)
    state = make_decision(decision, db)
    point = DecisionPoint(0, 0, 1, C.PHASE_END, (8000, 8000), decision, state.actions(), state, ())
    shadow = SimpleNamespace(
        point=point,
        done=False,
        tracker=SimpleNamespace(events=(), turn=1, engine_winner=0, result=SimpleNamespace(reason="win")),
    )

    def advance(index):
        shadow.point = None
        shadow.done = True
        shadow.tracker.turn = 2

    shadow.act = advance
    agent = LethalAgent(Passive())
    agent.shadow = shadow
    line, _ = agent._rollout(0, 1, GreedyAgent(0, cards=db))
    assert line is None


def test_shadow_mismatch_disables_search_and_never_forwards_the_response(db):
    option = M.AttackOption(CELTIC, M.Location(0, C.LOCATION_MZONE, 0), False)
    decision = M.SelectBattleCmd(0, (), (option,), True, True)
    state = make_decision(decision, db)
    point = DecisionPoint(7, 0, 2, C.PHASE_BATTLE_STEP, (8000, 1000), decision, state.actions(), state, ())
    wrong = replace(point, actions=[replace(point.actions[0], value=1), *point.actions[1:]])
    closed, forwarded = [], []
    shadow = SimpleNamespace(point=wrong, close=lambda: closed.append(True), act=forwarded.append)
    agent = LethalAgent(Passive())
    agent.shadow = shadow
    assert not agent._should_search(point)
    agent.on_decision(point, 0)
    assert closed == [True] and not forwarded
    assert agent.shadow is None and agent.stats.desync == 1


@pytest.mark.parametrize("seed, a, b", [(1, "snake_eye", "kashtira"), (2, "labrynth", "tenpai")])
def test_the_shadow_stays_in_lockstep_with_the_real_duel(db, seed, a, b):
    agent = make_agent("lethal:random", seed)
    result = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=2000)).run(
        agent, RandomAgent(seed + 1)
    )
    assert result.reason in ("win", "decision_limit", "turn_limit")
    assert agent.stats.desync == 0 and agent.stats.searches > 0
