"""Tests for the Agent protocol, GreedyAgent and PolicyAgent (T3.1)."""

import functools
import math
from pathlib import Path

import pytest

from ygorl.agents import AGENTS, Agent, GreedyAgent, PolicyAgent, RandomAgent, agent_name
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.duel import DecisionPoint, Duel, default_cards

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}

BLUE_EYES = 89631139  # ATK 3000 / DEF 2500
DARK_MAGICIAN = 46986414  # ATK 2500 / DEF 2100
CELTIC_GUARDIAN = 91152256  # ATK 1400 / DEF 1200
KURIBOH = 40640057  # ATK 300 / DEF 200
BIG_SHIELD_GARDNA = 65240384  # ATK 100 / DEF 2600

ME, OPP = 0, 1


def loc(controller=ME, location=C.LOCATION_HAND, sequence=0, position=0):
    return M.Location(controller, location, sequence, position)


def card(code, controller=ME, location=C.LOCATION_HAND, sequence=0, position=0):
    return M.CardInfo(code, loc(controller, location, sequence, position))


def effect(code, description=0, location=C.LOCATION_HAND, sequence=0):
    return M.ChainOption(code, loc(ME, location, sequence), description, 0)


def point(decision, *, turn=1, phase=C.PHASE_MAIN1, events=(), index=0):
    state = make_decision(decision, default_cards())
    return DecisionPoint(index, decision.player, turn, phase, (8000, 8000), decision, state.actions(), state, tuple(events))


def idle(summonable=(), spsummonable=(), msetable=(), ssetable=(), activatable=(), battle=True, end=True):
    return M.SelectIdleCmd(ME, tuple(summonable), tuple(spsummonable), (), tuple(msetable), tuple(ssetable),
                           tuple(activatable), battle, end, False)  # fmt: skip


def battle(attackable=(), activatable=(), main2=True, end=True):
    return M.SelectBattleCmd(ME, tuple(activatable), tuple(attackable), main2, end)


def attacker(code, sequence=0, direct=False):
    return M.AttackOption(code, loc(ME, C.LOCATION_MZONE, sequence), direct)


ATTACK_TARGET_HINT = M.Hint(C.HINT_SELECTMSG, ME, 549)  # HINTMSG_ATTACKTARGET


def chosen(agent, decision, **kw):
    p = point(decision, **kw)
    return p.actions[agent.act(p)]


# ------------------------------------------------------------------ protocol


def test_all_agents_satisfy_the_protocol():
    for agent in (RandomAgent(0), GreedyAgent(0), PolicyAgent(lambda p: [0.0] * len(p.actions))):
        assert isinstance(agent, Agent)


def test_agent_registry_and_names():
    for name, factory in AGENTS.items():
        assert isinstance(factory(0), Agent) and agent_name(factory) == name
    assert agent_name(GreedyAgent(0)) == "greedy"
    assert agent_name(functools.partial(GreedyAgent, max_repeats=1)) == "greedy"

    def my_agent(seed):
        return RandomAgent(seed)

    assert agent_name(my_agent) == "test_agent_registry_and_names.<locals>.my_agent"


# ------------------------------------------------------------------ greedy: main phase


def test_greedy_main_phase_priorities():
    g = GreedyAgent(0)
    full = dict(summonable=[card(CELTIC_GUARDIAN)], spsummonable=[card(DARK_MAGICIAN)], msetable=[card(KURIBOH)],
                ssetable=[card(1)], activatable=[effect(DARK_MAGICIAN)])  # fmt: skip
    assert chosen(g, idle(**full)).kind == "activate"
    del full["activatable"]
    assert chosen(g, idle(**full)).kind == "spsummon"
    del full["spsummonable"]
    assert chosen(g, idle(**full)).kind == "summon"
    del full["summonable"]
    assert chosen(g, idle(**full)).kind in ("mset", "sset")
    assert chosen(g, idle(), turn=2).kind == "battle_phase"
    assert chosen(g, idle(battle=False), phase=C.PHASE_MAIN2).kind == "end_phase"


def test_greedy_normal_summons_the_strongest_monster():
    g = GreedyAgent(0)
    a = chosen(g, idle(summonable=[card(KURIBOH, sequence=0), card(BLUE_EYES, sequence=1), card(CELTIC_GUARDIAN, sequence=2)]))
    assert a.kind == "summon" and a.card.code == BLUE_EYES


def test_greedy_does_not_repeat_the_same_activation_forever():
    g = GreedyAgent(0)
    decision = idle(activatable=[effect(DARK_MAGICIAN, description=7)])
    kinds = [chosen(g, decision, turn=3, index=i).kind for i in range(20)]
    assert kinds[0] == "activate"
    assert "battle_phase" in kinds  # gives up on an activation that keeps coming back
    assert chosen(g, decision, turn=4).kind == "activate"  # new turn, new budget


# ------------------------------------------------------------------ greedy: battle


def test_greedy_attacks_directly_first_then_with_the_strongest():
    g = GreedyAgent(0)
    d = battle([attacker(CELTIC_GUARDIAN, 0), attacker(KURIBOH, 1, direct=True), attacker(BLUE_EYES, 2)])
    a = chosen(g, d, phase=C.PHASE_BATTLE_STEP)
    assert a.kind == "attack" and a.card.code == KURIBOH
    d = battle([attacker(CELTIC_GUARDIAN, 0), attacker(BLUE_EYES, 2)], activatable=[effect(DARK_MAGICIAN)])
    a = chosen(g, d, phase=C.PHASE_BATTLE_STEP)
    assert a.kind == "attack" and a.card.code == BLUE_EYES


def test_greedy_leaves_battle_when_no_attackers():
    g = GreedyAgent(0)
    assert chosen(g, battle(), phase=C.PHASE_BATTLE_STEP).kind == "main2"
    assert chosen(g, battle(main2=False), phase=C.PHASE_BATTLE_STEP).kind == "end_phase"


def _target_selection(targets, cancelable=True):
    return M.SelectCard(ME, cancelable, 1, 1, tuple(targets))


def _attack_then_target(g, attacker_code, targets, cancelable=True):
    a = chosen(g, battle([attacker(attacker_code)]), phase=C.PHASE_BATTLE_STEP)
    assert a.kind == "attack"
    return chosen(g, _target_selection(targets, cancelable), phase=C.PHASE_BATTLE_STEP, events=[ATTACK_TARGET_HINT])


def test_greedy_attacks_the_strongest_target_it_beats():
    atk_pos = dict(controller=OPP, location=C.LOCATION_MZONE, position=C.POS_FACEUP_ATTACK)
    def_pos = dict(controller=OPP, location=C.LOCATION_MZONE, position=C.POS_FACEDOWN_DEFENSE)
    targets = [card(KURIBOH, sequence=0, **atk_pos), card(DARK_MAGICIAN, sequence=1, **atk_pos)]
    assert _attack_then_target(GreedyAgent(0), BLUE_EYES, targets).card.code == DARK_MAGICIAN
    assert _attack_then_target(GreedyAgent(0), CELTIC_GUARDIAN, targets).card.code == KURIBOH
    # Defense-position targets are compared by DEF: 1400 does not beat Big Shield Gardna's 2600 DEF.
    up_def = dict(def_pos, position=C.POS_FACEUP_DEFENSE)
    targets = [card(BIG_SHIELD_GARDNA, sequence=0, **up_def), card(KURIBOH, sequence=1, **atk_pos)]
    assert _attack_then_target(GreedyAgent(0), CELTIC_GUARDIAN, targets).card.code == KURIBOH


def test_greedy_never_peeks_at_face_down_targets():
    """A face-down target arrives with code 0 (messages.hide_private); Greedy assumes HIDDEN_STAT for it."""
    from ygorl.agents.greedy import HIDDEN_STAT

    hidden = card(0, OPP, C.LOCATION_MZONE, 0, C.POS_FACEDOWN_DEFENSE)
    kuriboh = card(KURIBOH, OPP, C.LOCATION_MZONE, 1, C.POS_FACEUP_ATTACK)
    assert HIDDEN_STAT == 1500
    # 2500 beats the assumed 1500: the stronger (hidden) target is attacked.
    assert _attack_then_target(GreedyAgent(0), DARK_MAGICIAN, [hidden, kuriboh]).card.loc == hidden.loc
    # 1400 does not: Kuriboh is attacked instead.
    assert _attack_then_target(GreedyAgent(0), CELTIC_GUARDIAN, [hidden, kuriboh]).card.code == KURIBOH


def test_greedy_cancels_a_losing_attack_and_does_not_retry_it():
    g = GreedyAgent(0)
    targets = [card(DARK_MAGICIAN, OPP, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)]
    assert _attack_then_target(g, KURIBOH, targets).kind == "cancel"
    # Back at the battle command the same attacker is no longer chosen.
    assert chosen(g, battle([attacker(KURIBOH)]), phase=C.PHASE_BATTLE_STEP).kind == "main2"
    # Without the option to cancel it still picks the least bad target.
    t = [card(DARK_MAGICIAN, OPP, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK),
         card(CELTIC_GUARDIAN, OPP, C.LOCATION_MZONE, 1, C.POS_FACEUP_ATTACK)]  # fmt: skip
    assert _attack_then_target(GreedyAgent(0), KURIBOH, t, cancelable=False).card.code == CELTIC_GUARDIAN


# ------------------------------------------------------------------ greedy: other decisions


def test_greedy_chains_and_says_yes():
    g = GreedyAgent(0)
    chain = M.SelectChain(ME, 0, False, 0, 0, (effect(DARK_MAGICIAN),))
    assert chosen(g, chain).kind == "chain"
    assert chosen(g, M.SelectEffectYn(ME, card(DARK_MAGICIAN), 0)).kind == "yes"
    assert chosen(g, M.SelectYesNo(ME, 0)).kind == "yes"


def test_greedy_selects_before_finishing_and_never_unselects():
    g = GreedyAgent(0)
    d = M.SelectCard(ME, False, 1, 2, (card(KURIBOH, sequence=0), card(BLUE_EYES, sequence=1)))
    p = point(d)
    first = p.actions[g.act(p)]
    assert first.kind == "select"
    p.state.step(p.actions.index(first))
    p2 = DecisionPoint(1, ME, 1, C.PHASE_MAIN1, (8000, 8000), d, p.state.actions(), p.state, ())
    assert p2.actions[g.act(p2)].kind == "select"  # "finish" only when nothing better
    u = M.SelectUnselectCard(ME, True, False, 1, 1, (), (card(KURIBOH),))
    assert chosen(g, u).kind == "finish"


def test_greedy_positions_by_attack_and_defense():
    g = GreedyAgent(0)
    all_pos = C.POS_FACEUP_ATTACK | C.POS_FACEUP_DEFENSE | C.POS_FACEDOWN_DEFENSE
    assert chosen(g, M.SelectPosition(ME, BLUE_EYES, all_pos)).value == C.POS_FACEUP_ATTACK
    assert chosen(g, M.SelectPosition(ME, BIG_SHIELD_GARDNA, all_pos)).value == C.POS_FACEUP_DEFENSE


def test_greedy_fallback_is_seeded():
    d = M.SelectOption(ME, tuple(range(8)))

    def picks(seed):
        g = GreedyAgent(seed)
        return [chosen(g, d).index for _ in range(10)]

    assert picks(1) == picks(1)
    assert picks(1) != picks(2)


# ------------------------------------------------------------------ greedy: full games


@pytest.mark.parametrize("i,name", list(enumerate(DECKS)))
def test_greedy_completes_games_on_every_deck(i, name):
    names = list(DECKS)
    opponent = names[(i + 1) % len(names)]
    r = Duel(100 + i, None, DECKS[name], DECKS[opponent], first=i % 2).run(GreedyAgent(i), RandomAgent(i))
    assert r.reason in ("win", "turn_limit"), r.summary()
    assert r.retries == 0 and r.unknown_messages == 0 and r.undecodable_messages == 0, r.summary()


def test_greedy_mirror_is_deterministic():
    def play():
        return Duel(7, None, DECKS["snake_eye"], DECKS["yubel"]).run(GreedyAgent(1), GreedyAgent(2))

    r1, r2 = play(), play()
    assert r1.reason in ("win", "turn_limit"), r1.summary()
    assert r1.responses == r2.responses and r1.winner == r2.winner


# ------------------------------------------------------------------ policy agent


def test_policy_agent_argmax_and_probs():
    scores = lambda p: [float(i == 2) for i in range(len(p.actions))]  # noqa: E731
    agent = PolicyAgent(scores, greedy=True)
    p = point(M.SelectOption(ME, (10, 11, 12, 13)))
    assert agent.act(p) == 2
    assert len(agent.last_probs) == 4 and math.isclose(sum(agent.last_probs), 1.0)
    assert max(range(4), key=lambda i: agent.last_probs[i]) == 2


def test_policy_agent_sampling_is_seeded_and_follows_the_scores():
    p = point(M.SelectOption(ME, (10, 11, 12)))
    scores = lambda _: [0.0, 5.0, -50.0]  # noqa: E731

    def run(seed):
        agent = PolicyAgent(scores, seed=seed)
        return [agent.act(p) for _ in range(200)]

    assert run(3) == run(3)
    picks = run(3)
    assert 2 not in picks and picks.count(1) > picks.count(0)
    cold = PolicyAgent(scores, seed=0, temperature=1e-6)
    assert all(cold.act(p) == 1 for _ in range(20))


def test_policy_agent_rejects_wrong_length_scores():
    agent = PolicyAgent(lambda p: [0.0])
    with pytest.raises(ValueError, match="scores"):
        agent.act(point(M.SelectOption(ME, (1, 2))))
