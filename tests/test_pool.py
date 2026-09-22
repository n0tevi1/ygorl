"""C++ DuelPool + Python VecDuelEnv / DuelEnv (T2.1)."""

import itertools
from pathlib import Path

import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env import DuelEnv, GameSpec, VecDuelEnv, run_games

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def specs(n, config=DuelConfig()):
    out = []
    for i in range(n):
        a, b = NAMES[i % 10], NAMES[(i * 3 + 1) % 10]
        out.append(GameSpec(seed=1000 + i, deck_a=DECKS[a], deck_b=DECKS[b], first=i % 2, config=config))
    return out


def sequential(db, spec):
    duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=db, first=spec.first, config=spec.config)
    return duel.run(RandomAgent(spec.seed), RandomAgent(spec.seed + 1))


def key(r):
    return (r.winner, r.reason, r.win_reason, r.turns, r.lp, r.decisions, r.responses)


def agents(i, spec):
    return RandomAgent(spec.seed), RandomAgent(spec.seed + 1)


def test_core_pool_start_returns_first_decision(db):
    pool = _core.DuelPool(2, 2, db.to_core(), default_scripts())
    duel = Duel(1, None, DECKS["snake_eye"], DECKS["yubel"], cards=db)
    decks = [list(d) for d in duel.loaded_decks()]
    pool.start(0, expand_seed(1), duel.config.rule_flags, (8000, 5, 1), (8000, 5, 1), decks)
    assert pool.pending() == 1
    (env_id, status, buf, logs, error), = pool.recv(1, -1)
    assert env_id == 0 and status == _core.DUEL_STATUS_AWAITING and error == ""
    names = [m.name for m in M.decode_buffer(buf)]
    assert names.count("MSG_DRAW") == 2 and names[-1].startswith("MSG_SELECT")
    assert pool.pending() == 0
    assert pool.recv(1, 10) == []  # nothing in flight: times out empty


def test_core_pool_rejects_misuse(db):
    pool = _core.DuelPool(1, 1, db.to_core(), default_scripts())
    with pytest.raises(IndexError):
        pool.respond(5, b"\0\0\0\0")
    with pytest.raises(RuntimeError, match="not started"):
        pool.respond(0, b"\0\0\0\0")


def test_run_games_matches_sequential_duels(db):
    games = specs(12)
    pooled = run_games(games, agents, num_envs=5, num_threads=3, cards=db)
    assert [key(r) for r in pooled] == [key(sequential(db, s)) for s in games]


def test_thread_count_does_not_change_results(db):
    games = specs(8)
    one = run_games(games, agents, num_envs=8, num_threads=1, cards=db)
    four = run_games(games, agents, num_envs=8, num_threads=4, cards=db)
    assert [key(r) for r in one] == [key(r) for r in four]


def test_limits_are_applied_in_pool(db):
    games = specs(3, DuelConfig(max_decisions=15))
    pooled = run_games(games, agents, num_envs=3, num_threads=2, cards=db)
    assert all(r.reason == "decision_limit" and r.decisions <= 15 for r in pooled)
    assert [key(r) for r in pooled] == [key(sequential(db, s)) for s in games]


def test_vec_env_events_and_errors(db):
    env = VecDuelEnv(num_envs=2, num_threads=2, cards=db)
    spec = specs(1)[0]
    env.start(0, spec)
    with pytest.raises(RuntimeError, match="busy"):
        env.start(0, spec)
    (event,) = env.recv(min_events=1)
    assert event.env_id == 0 and event.point is not None and event.result is None
    with pytest.raises(RuntimeError, match="no decision"):
        env.act(1, 0)
    with pytest.raises(ValueError, match="out of range"):
        env.act(0, len(event.point.actions))
    env.close()


def test_single_env_matches_duel(db):
    spec = specs(1)[0]
    agent_a, agent_b = agents(0, spec)
    seats = (agent_a, agent_b) if spec.first == 0 else (agent_b, agent_a)
    env = DuelEnv(cards=db)
    point = env.reset(spec)
    result = None
    while result is None:
        agent = seats[point.player]
        point, result = env.step(agent.act(point))
    assert key(result) == key(sequential(db, spec))
