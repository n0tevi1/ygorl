"""A core that keeps processing without asking for a decision stops as an engine loop instead of hanging, on every
host (C++ HostDuel / HostPool, the C++ DuelPool, the Python tracker); so does a single core call whose scripts run
past their instruction budget; a rollout that stops receiving events raises."""

import json
import time
from pathlib import Path

import pytest

from ygorl import _core, paths
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine import tracker as tracker_module
from ygorl.engine.duel import Duel, DuelConfig, default_cards, expand_seed
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


@pytest.fixture
def tiny_script_budget():
    """Loading a deck runs thousands of script instructions: a budget of 1000 always runs out in the first call."""
    old = _core.set_max_script_steps(1)
    yield
    _core.set_max_script_steps(old)


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
    monkeypatch.setattr(tracker_module, "MAX_ENGINE_STEPS", 1)
    s = specs()[0]
    r = Duel(s.seed, None, s.deck_a, s.deck_b, cards=db, config=s.config, first=s.first).run(
        RandomAgent(0), RandomAgent(1)
    )
    assert r.reason == "error" and "engine loop" in r.error and r.winner is None


def test_the_default_limits_leave_real_games_alone(db):
    assert _core.set_max_engine_steps(100_000) == 100_000 and tracker_module.MAX_ENGINE_STEPS == 100_000
    assert _core.set_max_script_steps(100_000) == 100_000
    for res in EncodedVecEnv(2, 1, cards=db).play(specs(), chooser):
        assert res["reason"] != "error"
    assert 0 < _core.script_steps_peak() <= 100_000


def test_cpp_host_stops_a_script_past_its_budget(db, tiny_script_budget):
    for res in EncodedVecEnv(2, 1, cards=db).play(specs(), chooser):
        assert res["reason"] == "error" and "script budget" in res["error"] and res["winner"] is None


def test_cpp_duel_pool_stops_a_script_past_its_budget(db, tiny_script_budget):
    results = run_games(specs(), lambda i, s: (RandomAgent(0), RandomAgent(1)), num_envs=2, num_threads=1, cards=db)
    assert all(r.reason == "error" and "script budget" in r.error for r in results)


def test_python_host_stops_a_script_past_its_budget(db, tiny_script_budget):
    s = specs()[0]
    r = Duel(s.seed, None, s.deck_a, s.deck_b, cards=db, config=s.config, first=s.first).run(
        RandomAgent(0), RandomAgent(1)
    )
    assert r.reason == "error" and "script budget" in r.error and r.winner is None
    assert issubclass(_core.ScriptBudgetExceeded, RuntimeError)


NESTED = 46986414  # a card whose script is replaced by a nested search (IsExists -> filter -> IsExists ...)
NESTED_LUA = """
local s,id=GetID()
function s.f(c,g,d)
  if d==0 then local x=0 for i=1,40 do x=x+i end return false end
  return g:IsExists(s.f,1,nil,g,d-1)
end
function s.initial_effect(c)
  local e=Effect.CreateEffect(c)
  e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
  e:SetCode(EVENT_ADJUST)
  e:SetRange(LOCATION_MZONE)
  e:SetCountLimit(1)
  e:SetCondition(function() local g=Duel.GetFieldGroup(0,LOCATION_DECK,0) return g:IsExists(s.f,1,nil,g,7) end)
  e:SetOperation(function() end)
  c:RegisterEffect(e)
end
"""


def test_a_nested_script_search_stops_at_once_past_its_budget(tiny_script_budget):
    """Each filter runs in its own pcall and a failed one only returns false: past the budget every instruction must
    fail, or the search goes on through the whole tree (8^7 leaves here), just slower and logging every failure."""
    dirs = [Path(p) for p in paths.script_directories()]

    def scripts(name):
        if name == f"c{NESTED}.lua":
            return NESTED_LUA
        for d in dirs:
            for p in (d / name, *d.glob(f"*/{name}")):
                if p.exists():
                    return p.read_text(errors="replace")
        return None

    core = _core.Duel(expand_seed(1), C.DUEL_MODE_MR5, (8000, 0, 0), (8000, 0, 0), default_cards().to_core(), scripts)
    assert core.load_script("constant.lua") and core.load_script("utility.lua")
    core.new_card(0, 0, NESTED, 0, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)
    for _ in range(8):
        core.new_card(0, 0, NESTED, 0, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
    core.new_card(1, 0, NESTED, 1, C.LOCATION_DECK, 0, C.POS_FACEDOWN_DEFENSE)
    core.start()
    t = time.time()
    with pytest.raises(_core.ScriptBudgetExceeded) as error:
        for _ in range(50):
            core.process()
    assert "stack traceback:" in str(error.value)
    assert f"c{NESTED}.lua" in str(error.value)
    assert len(str(error.value)) < 8400
    assert time.time() - t < 5 and len(core.pop_logs()) < 1000
    core.close()


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent / "data/material-budget-replays.json").read_text())["cases"],
                         ids=lambda case: case["name"])  # fmt: skip
def test_real_material_searches_report_the_original_lua_trace(case, db):
    """#169: real MD failures replay without a checkpoint, inference, or stochastic policy sampling."""
    old = _core.set_max_script_steps(100000)
    try:
        original = _core.ScriptDirectory([str(p) for p in paths.script_directories()])
        host = _core.HostDuel(db.to_core(), original, [])
        host.start(case["core_seed"], case["rule_flags"], (8000, 5, 1), (8000, 5, 1),
                   case["loaded_decks"], 200, 4000)  # fmt: skip
        for action in case["actions"]:
            assert not host.done()
            host.act(action)
        result = host.result()
        assert result["reason"] == "error" and result["winner"] is None
        assert (result["turns"], result["decisions"]) == (case["turns"], case["decisions"])
        assert "stack traceback:" in result["error"] and case["source"] in result["error"]
        assert len(result["error"]) < 8400
    finally:
        _core.set_max_script_steps(old)


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
    from ygorl.train.rollout import Assignment, RolloutCollector, RolloutStalled

    class Model(torch.nn.Module):
        pass

    c = RolloutCollector(SilentEnv(), Model(), lambda: Assignment(spec=None, info={"deck_a": "x"}), 4,
                         stall_timeout=0.05)  # fmt: skip
    with pytest.raises(RolloutStalled, match="no environment event") as err:
        c.collect()
    assert "env 0" in str(err.value) and "'deck_a': 'x'" in str(err.value)
