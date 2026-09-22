"""Tests for the single-duel API Duel(...).run(agent_a, agent_b) and RandomAgent (T1.5)."""

from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.legality import IllegalDeck
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import WIN_REASON_LP, DecisionPoint, Duel, DuelConfig, DuelResult, expand_seed

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


class Recorder:
    """Wraps an agent and remembers every decision point it was shown."""

    def __init__(self, inner):
        self.inner, self.points = inner, []

    def act(self, point):
        self.points.append(point)
        return self.inner.act(point)


def test_expand_seed_is_deterministic_and_nonzero():
    assert expand_seed(0) == expand_seed(0)
    assert expand_seed(0) != expand_seed(1)
    assert len(expand_seed(7)) == 4 and any(expand_seed(0))
    assert all(0 <= x < 2**64 for x in expand_seed(2**70))


def test_random_vs_random_completes(db):
    a, b = Recorder(RandomAgent(1)), Recorder(RandomAgent(2))
    result = Duel(11, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db).run(a, b)
    assert isinstance(result, DuelResult)
    assert result.winner in (0, 1, None)
    assert result.reason
    assert result.decisions > 0 and result.turns >= 1
    assert result.retries == 0 and result.unknown_messages == 0
    assert a.points and b.points
    p = a.points[0]
    assert isinstance(p, DecisionPoint) and isinstance(p.decision, M.Decision)
    assert p.player == 0 and p.actions and p.turn >= 1
    assert all(pt.player == 0 for pt in a.points) and all(pt.player == 1 for pt in b.points)


def test_same_seed_same_game(db):
    def play():
        return Duel(5, None, DECKS["labrynth"], DECKS["purrely"], cards=db).run(RandomAgent(3), RandomAgent(4))

    r1, r2 = play(), play()
    assert r1.responses == r2.responses and r1.winner == r2.winner and r1.turns == r2.turns


def test_first_player_swap(db):
    a, b = Recorder(RandomAgent(1)), Recorder(RandomAgent(2))
    Duel(3, None, DECKS["yubel"], DECKS["tenpai"], cards=db, first=1).run(a, b)
    first = min((pt.index, "a") for pt in a.points), min((pt.index, "b") for pt in b.points)
    assert min(first)[1] == "b"
    assert b.points[0].player == 0  # engine player 0 always moves first


def test_step_limit_decides_by_lp(db):
    result = Duel(8, None, DECKS["branded_despia"], DECKS["tearlaments"], cards=db,
                  config=DuelConfig(max_decisions=20)).run(RandomAgent(0), RandomAgent(0))  # fmt: skip
    assert result.reason == "decision_limit"
    assert result.decisions <= 20
    lp_a, lp_b = result.lp
    assert result.winner == (None if lp_a == lp_b else (0 if lp_a > lp_b else 1))


def test_turn_limit(db):
    result = Duel(8, None, DECKS["voiceless_voice"], DECKS["fiendsmith_ryzeal"], cards=db,
                  config=DuelConfig(max_turns=2)).run(RandomAgent(0), RandomAgent(0))  # fmt: skip
    assert result.reason in ("turn_limit", "win")
    assert result.turns <= 3


def test_invalid_agent_output_is_rejected(db):
    class Bad:
        def act(self, point):
            return len(point.actions)

    with pytest.raises(ValueError, match="out of range"):
        Duel(1, None, DECKS["snake_eye"], DECKS["yubel"], cards=db).run(Bad(), RandomAgent(0))


def test_illegal_deck_rejected_when_validating(db):
    short = Deck(DECKS["snake_eye"].main[:39], DECKS["snake_eye"].extra)
    with pytest.raises(IllegalDeck, match="minimum 40"):
        Duel(1, None, short, DECKS["yubel"], cards=db, validate=True)


def test_lp_tracking_matches_engine(db):
    """LP tracked from MSG_DAMAGE/RECOVER/PAY_LPCOST/LPUPDATE equals the engine's field query."""
    duel = Duel(21, None, DECKS["purrely"], DECKS["snake_eye"], cards=db)
    mismatches = []

    class Checker:
        def __init__(self, seed):
            self.inner = RandomAgent(seed)

        def act(self, point):
            engine_lp = tuple(p["lp"] for p in duel.field_state().players)
            if engine_lp != point.lp:
                mismatches.append((point.index, point.lp, engine_lp))
            return self.inner.act(point)

    result = duel.run(Checker(9), Checker(8))
    assert mismatches == []
    if result.reason == "win" and result.win_reason == WIN_REASON_LP:
        assert result.lp[1 - result.winner] <= 0


@pytest.mark.parametrize("name", sorted(DECKS))
def test_every_deck_plays_without_errors(db, name):
    names = sorted(DECKS)
    opponent = names[(names.index(name) + 1) % len(names)]
    for seed in range(2):
        r = Duel(seed, None, DECKS[name], DECKS[opponent], cards=db).run(RandomAgent(seed), RandomAgent(seed + 100))
        assert r.retries == 0 and r.unknown_messages == 0 and r.undecodable_messages == 0, r.summary()
        assert r.reason in ("win", "turn_limit", "decision_limit")


def test_duel_stops_at_first_win(db):
    """The core keeps processing after MSG_WIN; the host must end the duel there."""
    result = Duel(0, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db).run(RandomAgent(1), RandomAgent(2))
    assert result.reason == "win", result.summary()
    assert result.turns < 150
    loser = 1 - result.winner
    if result.win_reason == WIN_REASON_LP:
        assert result.lp[loser] <= 0
