"""Core snapshots (T2.8): a duel's whole mutable state lives in its own arena; restore == replay."""

import time
from pathlib import Path

import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.duel import Duel, DuelConfig

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def game(db):
    """A recorded random game: (seed, deck names, first, responses)."""
    duel = Duel(11, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db, config=DuelConfig(max_turns=8))
    result = duel.run(RandomAgent(1), RandomAgent(2))
    assert len(result.responses) > 60
    return (11, "snake_eye", "kashtira", 0, result.responses)


def fresh(db, game, snapshots=True):
    seed, a, b, first, _ = game
    return Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, snapshots=snapshots)._setup()


def drive(core, responses, start, stop=None, awaiting=None):
    """Answer responses[start:stop] as the core asks; return (message buffers, next index).

    Returns once the core asks for response ``stop`` (left unanswered), or when the log runs out / the duel ends.
    ``awaiting``: the core is already waiting for response ``start`` (default: ``start > 0``).
    """
    log, k = [], start
    stop = len(responses) if stop is None else stop
    awaiting = start > 0 if awaiting is None else awaiting
    while True:
        if awaiting:
            if k >= stop:
                return log, k
            core.set_response(responses[k])
            k += 1
        status = core.process()
        log.append(core.get_message())
        if status == _core.DUEL_STATUS_END:
            return log, k
        awaiting = status == _core.DUEL_STATUS_AWAITING


def test_restore_then_continue_equals_the_uninterrupted_stream(db, game):
    responses = game[4]
    reference, _ = drive(fresh(db, game, snapshots=False), responses, 0)
    for cut in (0, len(responses) // 3, len(responses) - 5):
        core = fresh(db, game)
        head, k = drive(core, responses, 0, cut)
        assert k == cut
        snap = core.snapshot()
        tail1, _ = drive(core, responses, cut, awaiting=True)
        core.restore(snap)
        tail2, _ = drive(core, responses, cut, awaiting=True)
        assert tail1 == tail2, f"cut={cut}"
        assert head + tail1 == reference, f"cut={cut}"
        assert core.arena_escapes() == 0


def expected_tail(db, game, responses, p):
    core = fresh(db, game, snapshots=False)
    drive(core, responses, 0, p)
    return drive(core, responses, p)[0]


def test_many_restores_and_out_of_order_snapshots(db, game):
    responses = game[4]
    points = [10, 25, 40]
    core = fresh(db, game)
    snaps, k = {}, 0
    for p in points:
        _, k = drive(core, responses, k, p)
        snaps[p] = core.snapshot()
    expected = {p: expected_tail(db, game, responses, p) for p in points}
    for p in (40, 10, 25, 40, 10, 10, 25):
        core.restore(snaps[p])
        assert drive(core, responses, p)[0] == expected[p], f"p={p}"
    assert core.arena_escapes() == 0


def alternative_response(db, buffers):
    """A legal response to the pending decision other than the first action's (None if there is no choice)."""
    decision = [m for buf in buffers for m in M.decode_buffer(buf) if isinstance(m, M.Decision)][-1]
    state = make_decision(decision, db)
    if len(state.actions()) < 2:
        return None
    out = state.step(1)
    while out is None:
        out = state.step(0)
    return out


def test_branching_from_a_snapshot_matches_a_fresh_replay(db, game):
    """After restoring, answer differently; the branch must equal a fresh replay of prefix + new answer."""
    responses = list(game[4])
    core = fresh(db, game)
    branched = 0
    k = 0
    for cut in range(5, len(responses) - 5, 7):
        log, k = drive(core, responses, k, cut)
        alt = alternative_response(db, log)
        if alt is None or alt == responses[cut]:
            continue
        snap = core.snapshot()
        branch = responses[:cut] + [alt] + responses[cut + 1 :]
        tail, _ = drive(core, branch, cut, cut + 3)
        ref = fresh(db, game, snapshots=False)
        drive(ref, branch, 0, cut)
        assert tail == drive(ref, branch, cut, cut + 3)[0], f"cut={cut}"
        core.restore(snap)
        branched += 1
    assert branched >= 3


def test_snapshot_belongs_to_its_duel(db, game):
    a, b = fresh(db, game), fresh(db, game)
    snap = a.snapshot()
    with pytest.raises(ValueError, match="another duel"):
        b.restore(snap)
    assert snap.nbytes > 0


def test_duels_without_snapshots_refuse(db, game):
    core = fresh(db, game, snapshots=False)
    with pytest.raises(RuntimeError, match="snapshots"):
        core.snapshot()


def test_restore_is_much_faster_than_replaying(db, game):
    responses = game[4]
    cut = len(responses) - 5
    t0 = time.perf_counter()
    for _ in range(3):
        core = fresh(db, game)
        drive(core, responses, 0, cut)
    replay = (time.perf_counter() - t0) / 3
    snap = core.snapshot()
    t0 = time.perf_counter()
    for _ in range(10):
        core.restore(snap)
    restore = (time.perf_counter() - t0) / 10
    assert restore < replay / 5, (restore, replay)
