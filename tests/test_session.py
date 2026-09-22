"""DuelSession: a duel advanced one agent step at a time, with snapshot/restore (T2.8 / T2.9)."""

from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel, DuelConfig, DuelSession

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def make(db, snapshots=True, **config):
    return Duel(4, None, DECKS["tenpai"], DECKS["yubel"], cards=db, first=1,
                config=DuelConfig(max_turns=6, **config), snapshots=snapshots)  # fmt: skip


def play(session, agents):
    while not session.done:
        point = session.point
        if point is None:
            break
        session.act(agents[session.duel.deck_of(point.player)].act(point))
    return session.result()


class ByIndex:
    """Plays actions[point.index] (restartable, unlike ScriptedAgent)."""

    def __init__(self, actions):
        self.actions = list(actions)

    def act(self, point):
        return self.actions[point.index]


def end(r):
    return r.winner, r.reason, r.turns, r.lp, r.decisions, r.responses, r.actions


def test_session_plays_the_same_game_as_run(db):
    ref = make(db, snapshots=False).run(RandomAgent(1), RandomAgent(2))
    got = play(DuelSession(make(db)), (RandomAgent(1), RandomAgent(2)))
    assert end(got) == end(ref)


def test_restore_and_replay_the_same_actions(db):
    ref = make(db, snapshots=False).run(RandomAgent(1), RandomAgent(2))
    script = ByIndex(ref.actions)
    session = DuelSession(make(db))
    for _ in range(len(ref.actions) // 2):
        session.act(script.act(session.point))
    snap = session.snapshot()
    first = play(session, (script, script))
    assert end(first) == end(ref)
    for _ in range(2):  # the snapshot is not changed by playing on from it
        session.restore(snap)
        assert end(play(session, (script, script))) == end(ref)


def test_branches_from_a_snapshot_match_fresh_games(db):
    ref = make(db, snapshots=False).run(RandomAgent(1), RandomAgent(2))
    t = len(ref.actions) // 3
    session = DuelSession(make(db))
    for i in range(t):
        session.act(ref.actions[i])
    snap = session.snapshot()
    n = len(session.point.actions)
    for choice in range(min(n, 4)):
        session.restore(snap)
        session.act(choice)
        got = play(session, (RandomAgent(7), RandomAgent(8)))
        prefix = ByIndex(ref.actions[:t] + [choice])

        class Then:
            def __init__(self, agent):
                self.agent = agent

            def act(self, point):
                return prefix.act(point) if point.index <= t else self.agent.act(point)

        fresh = make(db, snapshots=False).run(Then(RandomAgent(7)), Then(RandomAgent(8)))
        assert end(got) == end(fresh), f"choice={choice}"


def test_curriculum_host_answers_work_in_sessions(db):
    ref = make(db, snapshots=False, curriculum="solo").run(RandomAgent(1), RandomAgent(2))
    got = play(DuelSession(make(db, curriculum="solo")), (RandomAgent(1), RandomAgent(2)))
    assert end(got) == end(ref) and got.auto_decisions == ref.auto_decisions


def test_snapshot_needs_a_snapshot_enabled_duel(db):
    session = DuelSession(make(db, snapshots=False))
    with pytest.raises(RuntimeError, match="snapshots"):
        session.snapshot()


def test_a_duel_runs_once(db):
    duel = make(db)
    DuelSession(duel)
    with pytest.raises(RuntimeError, match="once"):
        duel.run(RandomAgent(1), RandomAgent(2))
