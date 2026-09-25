"""A core that keeps processing without asking for a decision stops as an engine loop instead of hanging, on every
host (C++ HostDuel / HostPool, the C++ DuelPool, the Python tracker); a rollout that stops receiving events raises."""

from pathlib import Path

import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import duel as duel_module
from ygorl.engine.duel import Duel, DuelConfig
from ygorl.env import GameSpec, run_games
from ygorl.env.encoded import EncodedVecEnv, chooser

DECKS = Path(__file__).parent / "decks"


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture
def one_step():
    """Every duel needs more than one engine step before its first decision: a limit of 1 always trips."""
    old = _core.set_max_engine_steps(1)
    yield
    _core.set_max_engine_steps(old)


def specs():
    a, b = load_ydk(DECKS / "snake_eye.ydk"), load_ydk(DECKS / "kashtira.ydk")
    return [GameSpec(seed=s, deck_a=a, deck_b=b, first=s % 2, config=DuelConfig(max_decisions=200)) for s in (1, 2)]


def test_cpp_host_stops_an_engine_loop(db, one_step):
    for res in EncodedVecEnv(2, 1, cards=db).play(specs(), chooser):
        assert res["reason"] == "error" and "engine loop" in res["error"] and res["winner"] is None


def test_cpp_duel_pool_stops_an_engine_loop(db, one_step):
    results = run_games(specs(), lambda i, s: (RandomAgent(0), RandomAgent(1)), num_envs=2, num_threads=1, cards=db)
    assert all(r.reason == "error" and "engine loop" in r.error for r in results)


def test_python_tracker_stops_an_engine_loop(db, monkeypatch):
    monkeypatch.setattr(duel_module, "MAX_ENGINE_STEPS", 1)
    s = specs()[0]
    r = Duel(s.seed, None, s.deck_a, s.deck_b, cards=db, config=s.config, first=s.first).run(RandomAgent(0), RandomAgent(1))
    assert r.reason == "error" and "engine loop" in r.error and r.winner is None


def test_the_default_limit_leaves_real_games_alone(db):
    assert _core.set_max_engine_steps(100_000) == 100_000 and duel_module.MAX_ENGINE_STEPS == 100_000
    for res in EncodedVecEnv(2, 1, cards=db).play(specs(), chooser):
        assert res["reason"] != "error"


class SilentEnv:
    """Two environments that never produce an event (their engines are 'stuck')."""

    num_envs = 2

    def reset(self, env_id, spec):
        pass

    def step(self, env_id, action):
        pass

    def recv(self, n=1, timeout=None):
        return []


def test_a_rollout_that_stops_receiving_events_raises():
    torch = pytest.importorskip("torch")
    from ygorl.train.rollout import Assignment, RolloutCollector

    class Model(torch.nn.Module):
        pass

    c = RolloutCollector(SilentEnv(), Model(), lambda: Assignment(spec=None, info={"deck_a": "x"}), 4,
                         stall_timeout=0.05)  # fmt: skip
    with pytest.raises(RuntimeError, match="no environment event") as err:
        c.collect()
    assert "env 0" in str(err.value) and "'deck_a': 'x'" in str(err.value)
