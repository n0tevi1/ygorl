"""Prefix start and response export (T2): EncodedVecEnv.reset(..., prefix), EncodedVecEnv.responses and the position
library (ygorl.train.positions).

A prefix start plays the given action indices on the worker exactly as successive step() calls would, so its first
observation must equal, byte for byte, the observation the step-wise game reaches (events and privileged tensors
included); the exported responses must rebuild a Replay that re-runs to the same decision.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine.branch import recorded_actions
from ygorl.engine.duel import DuelConfig, PlayerRules
from ygorl.engine.replay import Replay
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv, chooser
from ygorl.train.positions import Position, PositionLibrary, board_key, board_keys, read_positions, write_positions

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)
WALK = 1_000_003  # seed offset of the walk that picks the prefix lengths


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def specs(n, base=7000, config=DuelConfig()):
    return [GameSpec(seed=base + i, deck_a=DECKS[NAMES[i % len(NAMES)]],
                     deck_b=DECKS[NAMES[(i * 3 + 1) % len(NAMES)]], first=i % 2, config=config)
            for i in range(n)]  # fmt: skip


def frozen(*dicts):
    """Every array of the given observation dicts as (key, dtype, shape, bytes): equal iff byte-identical."""
    out = []
    for i, d in enumerate(dicts):
        for k in sorted(d or {}):
            a = np.asarray(d[k])
            out.append((i, k, a.dtype.str, a.shape, a.tobytes()))
    return tuple(out)


def key(r):
    return (r["winner"], r["reason"], r["turns"], tuple(r["lp"]), r["decisions"], tuple(r["responses"]))


def pick(seed, step, mask):
    legal = np.flatnonzero(mask)
    return int(legal[chooser(seed, step, len(legal))])


def walk(env, spec, n, prefix=None):
    """Reset env 0 to ``spec`` (with ``prefix``) and play ``n`` chooser steps; returns (actions, last event)."""
    env.reset(0, spec, prefix)
    (ev,) = env.recv(1)
    actions = list(prefix or [])
    for _ in range(n):
        if ev.result is not None:
            break
        a = pick(spec.seed, len(actions), ev.obs["action_mask"])
        actions.append(a)
        env.step(0, a)
        (ev,) = env.recv(1)
    return actions, ev


@pytest.mark.parametrize("skip_forced", [False, True])
def test_prefix_start_matches_stepwise_game(db, vocab, skip_forced):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, privileged=True, skip_forced=skip_forced)
    rng = random.Random(3 + skip_forced)
    checked = 0
    for spec in specs(8):
        actions, ev = walk(env, spec, rng.randint(1, 150))
        if ev.result is not None:
            continue
        expected = (ev.player, frozen(ev.obs, ev.privileged))
        responses = env.responses(0)
        env.reset(0, spec, actions)
        (got,) = env.recv(1)
        assert got.result is None
        assert (got.player, frozen(got.obs, got.privileged)) == expected
        assert env.responses(0) == responses
        # and the game goes on identically from there
        env.step(0, pick(spec.seed, len(actions), got.obs["action_mask"]))
        (a,) = env.recv(1)
        _, b = walk(env, spec, 1, actions)
        assert frozen(a.obs, a.privileged) == frozen(b.obs, b.privileged) and a.result == b.result
        checked += 1
    assert checked >= 6


def test_prefix_starts_in_any_env_on_any_thread(db, vocab):
    one = EncodedVecEnv(1, 1, cards=db, vocab=vocab, skip_forced=True)
    plans = []
    for s in specs(6, base=7100):
        actions, ev = walk(one, s, 40)
        if ev.result is None:
            plans.append((s, actions, frozen(ev.obs)))
    assert len(plans) >= 4
    env = EncodedVecEnv(4, 3, cards=db, vocab=vocab, skip_forced=True)
    for start in range(0, len(plans), 4):
        batch = plans[start : start + 4]
        for e, (s, actions, _) in enumerate(reversed(batch)):  # other envs (and threads) than in the walk
            env.reset(e, s, actions)
        events = env.recv(len(batch))
        assert sorted(ev.env_id for ev in events) == list(range(len(batch)))
        for ev in events:
            assert frozen(ev.obs) == batch[::-1][ev.env_id][2]


def test_prefix_that_ends_the_game_returns_the_result(db, vocab):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, skip_forced=True, record_games=True)
    env.trace_model_update = 5
    spec = specs(1, base=7200)[0]
    actions, ev = walk(env, spec, 1 << 30)
    assert ev.result is not None and len(actions) > 10
    final = ev.result
    assert final["action_indices"] == actions and set(final["action_model_updates"]) == {5}
    for prefix in (actions, actions + [0, 3, 1]):  # exactly to the end; past the end
        env.reset(0, spec, prefix)
        (ev,) = env.recv(1)
        assert ev.obs is None and key(ev.result) == key(final)
        assert ev.result["prefix_applied"] == len(actions)
        assert ev.result["action_indices"] == actions  # only what was played
        assert ev.result["action_model_updates"] == [None] * len(actions)  # prefix actions come from no model
        assert env.responses(0) == final["responses"]
    env.reset(0, spec)  # a plain reset carries no prefix fields
    (ev,) = env.recv(1)
    assert ev.result is None


def test_illegal_prefix_raises_value_error(db, vocab):
    env = EncodedVecEnv(2, 2, cards=db, vocab=vocab, skip_forced=True)
    spec, other = specs(2, base=7300)
    actions, ev = walk(env, spec, 30)
    assert ev.result is None
    masked = np.flatnonzero(ev.obs["action_mask"] == 0)
    for bad in (int(masked[0]), 5000):
        env.reset(1, other)
        env.reset(0, spec, actions + [bad])
        with pytest.raises(ValueError, match=f"prefix action {len(actions)} \\(row {bad}\\)"):
            env.recv(2)
        held = env.recv(1)  # the other env's event is not lost
        assert [e.env_id for e in held] == [1] and held[0].result is None
        assert env.pending() == 0
        with pytest.raises(RuntimeError, match="no game"):
            env.responses(0)
    env.reset(0, spec, actions)  # the env is reusable
    (got,) = env.recv(1)
    assert frozen(got.obs) == frozen(ev.obs)
    with pytest.raises(ValueError, match="non-negative"):
        env.reset(0, spec, [-1])


@pytest.mark.parametrize("snapshots", [False, True])
def test_responses_rebuild_a_replay_of_the_same_position(db, vocab, snapshots):
    """The exported responses make a Replay that Replay.play re-runs to the same decision; its recorded actions, as a
    prefix (skip_forced off: every decision comes back), reach the same observation."""
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, snapshots=snapshots, event_length=0)
    rng = random.Random(11)
    checked = 0
    for spec in specs(6, base=7400):
        env.reset(0, spec)
        (ev,) = env.recv(1)
        stop, sent, actions = rng.randint(5, 200), 0, []
        while ev.result is None:
            responses = env.responses(0)
            if len(actions) >= stop and len(responses) > sent:  # at the first step of a decision
                break
            sent = len(responses)
            actions.append(pick(spec.seed, len(actions), ev.obs["action_mask"]))
            env.step(0, actions[-1])
            (ev,) = env.recv(1)
        responses = env.responses(0)
        replay = Replay.from_spec(spec, responses)
        result = replay.play(cards=db)
        if ev.result is not None:  # the game ended first: the replay re-runs the whole game
            assert responses == ev.result["responses"] and result.reason == ev.result["reason"]
            assert result.winner == ev.result["winner"]
            continue
        assert result.reason == "log_exhausted" and result.responses == responses
        again = Replay.from_json(replay.to_json())
        assert again.responses == responses
        prefix = recorded_actions(again, cards=db)
        env.reset(0, spec, prefix)
        (got,) = env.recv(1)
        assert frozen(got.obs) == frozen(ev.obs)
        checked += 1
    assert checked >= 4


# ----------------------------------------------------------------------------- positions


def test_position_jsonl_round_trip_and_start(db, vocab, tmp_path):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, skip_forced=True, cancel_budget=4)
    cfg = DuelConfig(player=PlayerRules(starting_lp=4000), max_decisions=3000, shuffle_decks=False)
    spec = specs(1, base=7500, config=cfg)[0]
    spec = GameSpec(spec.seed, spec.deck_a, spec.deck_b, first=1, config=cfg, core_seed=(1, 2, 3, 4))
    actions, ev = walk(env, spec, 25)
    assert ev.result is None
    board = {"players": [{"mzone": [{"code": 2, "position": 0x1}, {"code": 1, "position": 0x8}],
                          "szone": [{"code": 3, "position": 0xA}], "grave": [9], "banished": []},
                         {"mzone": [], "szone": [], "grave": [], "banished": []}]}  # fmt: skip
    pos = Position.from_spec(spec, actions, skip_forced=True, cancel_budget=4, learner_seat=0,
                             board_key=board_key(board), origin={"source": "test", "update": 3})  # fmt: skip
    other = pos.with_prefix(actions[:10], learner_seat=1, board_key=None, origin={})
    path = tmp_path / "positions.jsonl"
    write_positions(path, [pos])
    write_positions(path, [other], append=True)
    back = read_positions(path)
    assert back == [pos, other] and back[0].origin == pos.origin and back[0].config == cfg
    lib = PositionLibrary.load(path)
    assert len(lib) == 2 and lib[0] == pos
    lib.save(tmp_path / "again.jsonl")
    assert (tmp_path / "again.jsonl").read_text() == path.read_text()
    back[0].start(env, 0)
    (got,) = env.recv(1)
    assert frozen(got.obs) == frozen(ev.obs)
    with pytest.raises(ValueError, match="skip_forced"):
        pos.check(EncodedVecEnv(1, 1, cards=db, vocab=vocab, cancel_budget=4))
    with pytest.raises(ValueError, match="cancel_budget"):
        pos.check(EncodedVecEnv(1, 1, cards=db, vocab=vocab, skip_forced=True))
    bound = Position.from_json({**pos.to_json(), "environment": "md-2026-09"})
    with pytest.raises(ValueError, match="environment"):
        bound.check(env)
    path.write_text(path.read_text().replace('"format_version":1', '"format_version":9', 1))
    with pytest.raises(ValueError, match=":1: unsupported"):
        read_positions(path)


def test_position_library_sampling_is_deterministic():
    deck = DECKS[NAMES[0]]
    positions = [Position(environment=None, seed=i, first=i % 2, deck_a=deck, deck_b=deck, prefix=(i,),
                          skip_forced=True, learner_seat=i % 2) for i in range(20)]  # fmt: skip
    lib = PositionLibrary(positions)
    draws = [[lib.sample(np.random.default_rng(42)).seed for _ in range(3)] for _ in range(2)]
    assert draws[0] == draws[1]
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    seq_a = [lib.sample(rng_a).seed for _ in range(50)]
    assert seq_a == [lib.sample(rng_b).seed for _ in range(50)]
    assert len(set(seq_a)) > 10
    seated = [lib.sample(rng_a, learner_seat=1) for _ in range(30)]
    assert {p.learner_seat for p in seated} == {1}
    with pytest.raises(ValueError, match="learner_seat=0"):
        PositionLibrary(positions[1::2]).sample(rng_a, learner_seat=0)
    with pytest.raises(ValueError, match="empty"):
        PositionLibrary().sample(rng_a)
    with pytest.raises(ValueError, match="prefix"):
        Position(environment=None, seed=0, first=0, deck_a=deck, deck_b=deck, prefix=(-1,), skip_forced=False)


def test_board_key_is_the_field_only_key():
    side = {"mzone": [{"code": 2, "position": 0x1}, {"code": 1, "position": 0x8}],
            "szone": [{"code": 3, "position": 0xA}], "grave": [9, 8], "banished": [7]}  # fmt: skip
    empty = {"mzone": [], "szone": [], "grave": [], "banished": []}
    keys = board_keys({"players": [side, empty]})
    assert keys["field"] == (("mzone", 1, True), ("mzone", 2, False), ("szone", 3, True))
    assert keys["field_grave"] == (keys["field"], (("grave", 8), ("grave", 9), ("banished", 7)))
    shuffled = {**side, "mzone": side["mzone"][::-1], "grave": []}  # same field, other resources spent
    assert board_key({"players": [shuffled, empty]}) == board_key({"players": [side, empty]})
    assert board_key({"players": [empty, side]}, player=1) == board_key({"players": [side, empty]})
    assert board_key({"players": [empty, side]}) != board_key({"players": [side, empty]})
