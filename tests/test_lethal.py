"""Lethal search on top of an agent (T4e.1 stage A, #62)."""

from pathlib import Path

import pytest

from ygorl.agents import RandomAgent, make_agent
from ygorl.agents.lethal import LethalAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.data.environment import PlayerRules
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig

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


def test_the_search_finds_the_lethal_the_wrapped_agent_would_not_play(db):
    """1000 LP and an empty board on the other side: summoning Celtic Guardian and attacking wins on turn 2."""
    plain = duel(db).run(Passive(), Passive())
    assert plain.reason == "turn_limit"
    agent = LethalAgent(Passive(), seed=0)
    result = duel(db).run(Passive(), agent)  # deck b moves second: its first turn is turn 2
    assert result.reason == "win" and result.winner == 1 and result.turns == 2
    s = agent.stats
    assert s.found >= 1 and s.overrides >= 3 and s.desync == 0  # summon, battle phase, attack, ...
    # with the lethal out of reach (8000 LP), it searches but finds nothing and changes nothing
    agent = LethalAgent(Passive(), seed=0)
    assert duel(db, lp=8000).run(Passive(), agent).reason == "turn_limit"
    assert agent.stats.searches > 0 and agent.stats.found == 0 and agent.stats.overrides == 0


def test_strict_mode_rejects_lines_that_use_hidden_information():
    me = 0
    assert LethalAgent._clean([M.Draw(1, ((CELTIC, 10),))], me)  # the opponent's draw is public enough
    assert not LethalAgent._clean([M.Draw(0, ((CELTIC, 10),))], me)  # our own draw: the shadow knows the order
    assert LethalAgent._clean([], me)


@pytest.mark.parametrize("seed, a, b", [(1, "snake_eye", "kashtira"), (2, "labrynth", "tenpai")])
def test_the_shadow_stays_in_lockstep_with_the_real_duel(db, seed, a, b):
    agent = make_agent("lethal:random", seed)
    result = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=2000)).run(
        agent, RandomAgent(seed + 1)
    )
    assert result.reason in ("win", "decision_limit", "turn_limit")
    assert agent.stats.desync == 0 and agent.stats.searches > 0
