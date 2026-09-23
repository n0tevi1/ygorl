"""Optional and free-timing effects reach the agent as real choices: hand traps and GY quick effects are offered
in every response window the rules open, each offer comes with a pass, and activating them does what the card says."""

import pytest

from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig

# The Tricky (special summons itself from the hand by discarding 1 card), Branded Retribution (GY: banish itself,
# target another "Branded" Spell/Trap in the GY; add it to the hand, Quick Effect), Branded Fusion, Celtic Guardian,
# Maxx "C" (hand, Quick Effect: this turn, draw 1 each time the opponent Special Summons; once per turn)
TRICKY, RETRIBUTION, BRANDED_FUSION, CELTIC, MAXX = 14778250, 17751597, 44362883, 91152256, 23434538


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


class Scripted:
    """A special summons two The Tricky on turn 1, discarding the two Branded cards; B may chain Maxx "C" once."""

    def __init__(self, chain_maxx: bool):
        self.chain_maxx, self.summons, self.points = chain_maxx, 0, []

    def act(self, p):
        self.points.append(p)
        rows = [(a.kind, a.card.code if a.card else 0) for a in p.actions]
        if p.player == 1 and self.chain_maxx and p.phase == C.PHASE_MAIN1 and ("chain", MAXX) in rows:
            self.chain_maxx = False
            return rows.index(("chain", MAXX))
        if p.player == 0 and p.turn == 1 and self.summons < 2 and ("spsummon", TRICKY) in rows:
            self.summons += 1
            return rows.index(("spsummon", TRICKY))
        if p.player == 0 and isinstance(p.decision, M.SelectUnselectCard):
            for code in (RETRIBUTION, BRANDED_FUSION):
                if ("select", code) in rows:
                    return rows.index(("select", code))
        kinds = [k for k, _ in rows]
        for kind in ("finish", "pass", "no", "cancel", "end_phase"):
            if kind in kinds:
                return kinds.index(kind)
        return 0


def play(db, chain_maxx):
    deck_a = Deck(main=(CELTIC,) * 36 + (BRANDED_FUSION, RETRIBUTION, TRICKY, TRICKY))  # unshuffled: last 5 drawn
    deck_b = Deck(main=(MAXX,) * 40)
    agent = Scripted(chain_maxx)
    Duel(1, None, deck_a, deck_b, cards=db, config=DuelConfig(max_turns=3, shuffle_decks=False)).run(agent, agent)
    return agent.points


def offers(points, player, code):
    """Decision points where ``player`` could activate ``code``, as (turn, phase, decision type)."""
    out = []
    for p in points:
        if p.player == player and any(a.card and a.card.code == code and a.kind in ("chain", "activate")
                                      for a in p.actions):  # fmt: skip
            out.append((p.turn, p.phase, type(p.decision).__name__))
    return out


def test_a_hand_trap_is_offered_in_every_window_of_the_opponents_turn_with_a_pass(db):
    points = play(db, chain_maxx=False)
    maxx = [p for p in points if p.player == 1 and any(a.card and a.card.code == MAXX for a in p.actions)
            and isinstance(p.decision, M.SelectChain)]  # fmt: skip
    turn1 = [p for p in maxx if p.turn == 1]
    # draw phase, standby phase, after each special summon, before A leaves the main phase, in the end phase
    phases = [p.phase for p in turn1]
    assert phases.count(C.PHASE_DRAW) == 1 and phases.count(C.PHASE_STANDBY) == 1 and phases.count(C.PHASE_END) == 1
    assert phases.count(C.PHASE_MAIN1) == 3
    for p in maxx:
        kinds = [a.kind for a in p.actions]
        # every copy in the hand is its own row, and not activating is always a choice
        assert kinds.count("chain") == sum(1 for a in p.actions if a.card and a.card.code == MAXX) >= 5
        assert kinds[-1] == "pass" and not p.decision.forced


def test_activating_the_hand_trap_draws_on_each_special_summon_once_per_turn(db):
    points = play(db, chain_maxx=True)
    draws_b = [(p.turn, len(e.cards)) for p in points for e in p.events if isinstance(e, M.Draw) and e.player == 1]
    # opening hand, one draw for A's second The Tricky (the first one resolved before Maxx "C"), B's normal draw
    assert draws_b == [(1, 5), (1, 1), (2, 1)]
    chained = [p for p in points if any(isinstance(e, M.Chaining) and e.code == MAXX for e in p.events)]
    first = chained[0].index
    assert not [o for p in points if p.index > first and p.turn == 1 for o in offers([p], 1, MAXX)]  # once per turn
    assert offers([p for p in points if p.turn == 2], 1, MAXX)  # available again on the next turn


def test_a_gy_quick_effect_is_offered_in_the_open_state_and_in_every_window_of_both_turns(db):
    points = play(db, chain_maxx=False)
    gy = offers(points, 0, RETRIBUTION)
    for p in points:
        for a in p.actions:
            if a.card and a.card.code == RETRIBUTION and a.kind in ("chain", "activate"):
                assert a.card.loc.location == C.LOCATION_GRAVE
                assert "pass" in [b.kind for b in p.actions] or isinstance(p.decision, M.SelectIdleCmd)
    assert (1, C.PHASE_MAIN1, "SelectIdleCmd") in gy  # A's own open state
    assert (1, C.PHASE_END, "SelectChain") in gy  # the End Phase window
    assert {(ph, t) for turn, ph, t in gy if turn == 2} >= {  # B's turn: every window
        (C.PHASE_DRAW, "SelectChain"), (C.PHASE_STANDBY, "SelectChain"), (C.PHASE_MAIN1, "SelectChain"),
        (C.PHASE_END, "SelectChain")}  # fmt: skip
