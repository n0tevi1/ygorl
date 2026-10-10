"""Diagnostic regressions: forced-action context, dynamic stats and hidden cards."""

from types import SimpleNamespace

import pytest

from combat_probe import CombatProbe, masked_action
from ygorl.agents.greedy import GreedyAgent
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.duel import DecisionPoint, default_cards

ATTACKER, TARGET = 89631139, 46986414


def point(decision, events=()):
    state = make_decision(decision, default_cards())
    return DecisionPoint(0, 0, 7, C.PHASE_BATTLE_STEP, (8000, 8000), decision, state.actions(), state, events)


def battle(code=ATTACKER):
    attack = M.AttackOption(code, M.Location(0, C.LOCATION_MZONE, 0), False)
    return point(M.SelectBattleCmd(0, (), (attack,), True, True))


def target(position=C.POS_FACEUP_DEFENSE, code=TARGET):
    card = M.CardInfo(code, M.Location(1, C.LOCATION_MZONE, 0, position))
    return point(M.SelectCard(0, True, 1, 1, (card,)), (M.Hint(C.HINT_SELECTMSG, 0, 549),))


def test_forced_attack_context_matches_naturally_chosen_attack():
    cards = default_cards()
    natural, forced = GreedyAgent(0, cards=cards), CombatProbe(0, cards=cards)
    p = battle(91152256)  # 1400 ATK cannot beat the target's printed2100 DEF
    idx = natural.act(p)
    assert p.actions[idx].kind == "attack"
    forced.prime(p, idx)
    q = target()
    chosen = forced.act(q)
    assert chosen == natural.act(q)
    assert q.actions[chosen].kind == "cancel"
    cold = GreedyAgent(0, cards=cards)
    assert q.actions[cold.act(q)].kind == "select"  # lost attacker context enters generic selection
    with pytest.raises(ValueError, match="fresh"):
        forced.prime(p, idx)


def test_current_stats_use_instance_location_and_change_target_decision():
    board = [[{"code": ATTACKER, "position": 1, "attack": 1800, "defense": 2500}],
             [{"code": TARGET, "position": 4, "attack": 2500, "defense": 2200}]]  # fmt: skip
    current = CombatProbe(0, cards=default_cards(), board=lambda: board)
    static = CombatProbe(0, cards=default_cards())
    p, q = battle(), target()
    for agent in (current, static):
        agent.prime(p, 0)
    assert q.actions[static.act(q)].kind == "select"  # printed 3000 beats printed 2100
    assert q.actions[current.act(q)].kind == "cancel"  # current 1800 does not beat current 2200


@pytest.mark.parametrize("hidden_defense", [0, 9999])
def test_facedown_stats_are_not_used_even_with_unfiltered_provider(hidden_defense):
    board = [[{"code": ATTACKER, "position": 1, "attack": 1400, "defense": 2500}],
             [{"code": TARGET, "position": 8, "attack": 2500, "defense": hidden_defense}]]  # fmt: skip
    agent = CombatProbe(0, cards=default_cards(), board=lambda: board)
    p, q = battle(), target(C.POS_FACEDOWN_DEFENSE, 0)
    agent.prime(p, 0)
    assert q.actions[agent.act(q)].kind == "cancel"  # unknown target uses fixed1500, never secret DEF


def test_equal_card_codes_do_not_mix_current_stats_between_instances():
    board = [[{"code": ATTACKER, "position": 1, "attack": 1000},
              {"code": ATTACKER, "position": 1, "attack": 4000}], []]  # fmt: skip
    agent = CombatProbe(0, cards=default_cards(), board=lambda: board)
    actions = [SimpleNamespace(card=M.CardInfo(ATTACKER, M.Location(0, C.LOCATION_MZONE, i))) for i in (0, 1)]
    assert [agent._power(a) for a in actions] == [1000, 4000]


def test_mask_blocks_cancel_and_preserves_original_engine_indices():
    agent = CombatProbe(0, cards=default_cards())
    p, q = battle(91152256), target()
    agent.prime(p, 0)
    assert q.actions[masked_action(agent, q, [True])].kind == "select"
    agent = CombatProbe(0, cards=default_cards())
    agent.prime(p, 0)
    cancel = next(i for i, a in enumerate(q.actions) if a.kind == "cancel")
    mask = [i == cancel for i in range(len(q.actions))]
    assert masked_action(agent, q, mask) == cancel


def test_empty_mask_fails_before_mutating_chooser_context():
    agent = CombatProbe(0, cards=default_cards())
    p, q = battle(), target()
    agent.prime(p, 0)
    attacker = agent._attacker
    with pytest.raises(ValueError, match="no policy-supported"):
        masked_action(agent, q, [False] * len(q.actions))
    assert agent._attacker is attacker
