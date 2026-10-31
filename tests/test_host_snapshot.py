"""Step-wise environment snapshots (T1): HostDuel / HostPool / EncodedVecEnv snapshot, restore and release.

A snapshot is the core's arena image plus deep copies of the C++ tracker and event history, so restoring one must
reproduce the observation it was taken at (events and privileged tensors included) byte for byte, and the game
played on from it must equal the uninterrupted game (result and response log).
"""

import ctypes
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv, chooser
from ygorl.env.events import DEFAULT_EVENT_LENGTH

pytestmark = pytest.mark.skipif(not _core.ARENA_AVAILABLE, reason="built without arenas (-DYGORL_ARENA=OFF)")

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)
DETOUR = 1_000_003  # seed offset of the random detour walked between snapshot and restore


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def specs(n, base=5000, config=DuelConfig()):
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


# ----------------------------------------------------------------------------- HostDuel


def host_duel(db, vocab, spec, snapshots=True):
    duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=db, first=spec.first, config=spec.config)
    host = _core.HostDuel(db.to_core(), default_scripts(), vocab.passwords(), DEFAULT_EVENT_LENGTH, snapshots=snapshots)
    cfg = spec.config
    host.start(expand_seed(spec.seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
               [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
    return host


def state(host):
    """Everything an agent (and the critic) sees at the current decision, plus the raw action rows and undo mask."""
    obs = host.observe()
    return frozen(obs, host.observe_privileged()), repr(host.actions()), tuple(host.undo())


def advance(host, seed, step, n):
    """Play up to n chooser(seed, ...) steps from step ``step``; returns the next step index."""
    for _ in range(n):
        if host.done() or not host.actions():
            break
        host.act(pick(seed, step, host.observe()["action_mask"]))
        step += 1
    return step


def finish(host, seed, step):
    advance(host, seed, step, 1 << 30)
    return host.result()


def test_host_duel_restore_reproduces_observation_and_game(db, vocab):
    rng = random.Random(1)
    for spec in specs(8):
        plain = finish(host_duel(db, vocab, spec, snapshots=False), spec.seed, 0)
        assert plain["decisions"] > 20
        host = host_duel(db, vocab, spec)
        t = advance(host, spec.seed, 0, rng.randrange(0, plain["decisions"] - 1))
        if host.done():
            continue
        before = state(host)
        snap = host.snapshot()
        assert snap.nbytes > 1 << 20
        advance(host, spec.seed + DETOUR, 0, rng.randint(1, 40))
        assert host.done() or state(host) != before
        host.restore(snap)
        assert state(host) == before
        assert key(finish(host, spec.seed, t)) == key(plain)
        assert host.arena_escapes() == 0


def test_host_duel_snapshots_restore_in_any_order(db, vocab):
    spec = specs(1, base=5101)[0]
    plain = finish(host_duel(db, vocab, spec, snapshots=False), spec.seed, 0)
    host = host_duel(db, vocab, spec)
    taken, step = [], 0
    for n in (3, 10, 15):
        step = advance(host, spec.seed, step, n)
        assert not host.done()
        taken.append((step, host.snapshot(), state(host)))
    advance(host, spec.seed + DETOUR, 0, 25)
    for i in (1, 0, 2, 0, 2, 1):
        t, snap, expected = taken[i]
        host.restore(snap)
        assert state(host) == expected, i
        advance(host, spec.seed + DETOUR + i, 0, 1 + 7 * i)  # wander off differently each time
    t, snap, _ = taken[0]
    host.restore(snap)
    assert key(finish(host, spec.seed, t)) == key(plain)
    host.restore(taken[2][1])  # restoring into a finished game works too
    assert key(finish(host, spec.seed, taken[2][0])) == key(plain)


def test_host_duel_rejects_other_games_snapshots(db, vocab):
    a, b = specs(2, base=5201)
    one, two = host_duel(db, vocab, a), host_duel(db, vocab, a)  # same game, different duels
    snap = one.snapshot()
    expected = state(two)
    with pytest.raises(ValueError, match="another"):
        two.restore(snap)
    other = host_duel(db, vocab, b)
    with pytest.raises(ValueError, match="another"):
        other.restore(snap)
    assert state(two) == expected  # a rejected restore changes nothing
    # A restarted HostDuel is another game.
    duel = Duel(b.seed, None, b.deck_a, b.deck_b, cards=db, first=b.first)
    one.start(expand_seed(b.seed), b.config.rule_flags, (8000, 5, 1), (8000, 5, 1),
              [(list(m), list(e)) for m, e in duel.loaded_decks()], b.config.max_turns, b.config.max_decisions)  # fmt: skip
    with pytest.raises(ValueError, match="another"):
        one.restore(snap)
    with pytest.raises(RuntimeError, match="without snapshots"):
        host_duel(db, vocab, a, snapshots=False).snapshot()
    finish(two, a.seed, 0)
    with pytest.raises(RuntimeError, match="decision"):
        two.snapshot()


def test_host_duel_restore_is_fast(db, vocab):
    spec = specs(1, base=5301)[0]
    host = host_duel(db, vocab, spec)
    advance(host, spec.seed, 0, 40)
    assert not host.done()
    snap = host.snapshot()
    times = []
    for i in range(200):
        advance(host, spec.seed + DETOUR + i, 0, 3)
        start = time.perf_counter()
        host.restore(snap)
        times.append(time.perf_counter() - start)
    median = statistics.median(times)
    print(f"HostDuel.restore median {median * 1e3:.3f} ms (snapshot {snap.nbytes / 2**20:.2f} MiB)")
    assert median < 1e-3


# ----------------------------------------------------------------------------- HostPool / EncodedVecEnv


def detour_plan(games, seed=7):
    rng = random.Random(seed)
    return {s.seed: (rng.randrange(0, 120), rng.randint(1, 40)) for s in games}


def play_with_detours(db, vocab, games, num_envs, threads, plan=None, **kwargs):
    """Play every game with the chooser; with ``plan`` ({seed: (t, k)}), snapshot at step t, walk k random steps (or
    to the end of the game), restore and check the observation, then play on. Returns (results, observation log)."""
    env = EncodedVecEnv(num_envs, threads, cards=db, vocab=vocab, privileged=True, snapshots=plan is not None,
                        **kwargs)  # fmt: skip
    queue, results, log = list(enumerate(games)), {}, {}
    slot = {}  # env -> dict(game state)

    def start(e):
        if not queue:
            return
        i, spec = queue.pop(0)
        slot[e] = {"i": i, "spec": spec, "step": 0, "phase": "main", "detour": 0}
        log[i] = []
        env.reset(e, spec)

    for e in range(num_envs):
        start(e)
    while env.pending():
        for ev in env.recv(1):
            g = slot[ev.env_id]
            spec, e = g["spec"], ev.env_id
            seen = None if ev.obs is None else frozen(ev.obs, ev.privileged)
            if g["phase"] == "snapshot":  # the snapshot job returns the unchanged observation
                assert seen == g["expected"]
                g["phase"] = "detour"
            elif g["phase"] == "restore":
                assert seen == g["expected"], f"game {g['i']}: restored observation differs"
                env.release(e, g["sid"])
                g["phase"] = "after"  # already logged when the snapshot was taken: play on from step t
                env.step(e, pick(spec.seed, g["step"], ev.obs["action_mask"]))
                g["step"] += 1
                continue
            elif g["phase"] == "detour" and (ev.obs is None or g["detour"] >= plan[spec.seed][1]):
                g["phase"] = "restore"
                env.restore(e, g["sid"])
                continue
            if ev.obs is None:
                results[g["i"]] = ev.result
                start(e)
                continue
            if g["phase"] == "detour":
                env.step(e, pick(spec.seed + DETOUR, g["detour"], ev.obs["action_mask"]))
                g["detour"] += 1
                continue
            log[g["i"]].append(hash(seen))
            if g["phase"] == "main" and plan is not None and g["step"] == plan[spec.seed][0]:
                g["phase"], g["expected"] = "snapshot", seen
                g["sid"] = env.snapshot(e)
                continue
            env.step(e, pick(spec.seed, g["step"], ev.obs["action_mask"]))
            g["step"] += 1
    if plan is not None:
        assert env.arena_escapes() == 0
        assert env.snapshot_count() == 0
    return [results[i] for i in range(len(games))], [log[i] for i in range(len(games))]


@pytest.mark.parametrize("skip_forced", [False, True])
def test_pool_detours_leave_games_unchanged(db, vocab, skip_forced):
    games = specs(8, base=5400)
    plain, plain_log = play_with_detours(db, vocab, games, 4, 3, skip_forced=skip_forced)
    plan = detour_plan(games)
    branched, branched_log = play_with_detours(db, vocab, games, 4, 3, plan, skip_forced=skip_forced)
    assert [key(r) for r in branched] == [key(r) for r in plain]
    assert branched_log == plain_log
    assert sum(len(g) > plan[s.seed][0] for g, s in zip(plain_log, games)) >= 6  # most games did take a detour


def test_pool_thread_count_does_not_change_detours(db, vocab):
    games = specs(6, base=5500)
    plan = detour_plan(games, seed=8)
    one = play_with_detours(db, vocab, games, 3, 1, plan)
    three = play_with_detours(db, vocab, games, 3, 3, plan)
    assert [key(r) for r in one[0]] == [key(r) for r in three[0]] and one[1] == three[1]


def first_event(env, e):
    (ev,) = env.recv(1)
    assert ev.env_id == e and ev.obs is not None
    return ev


def walk(env, e, ev, seed, n):
    for j in range(n):
        env.step(e, pick(seed, j, ev.obs["action_mask"]))
        (ev,) = env.recv(1)
        if ev.obs is None:
            break
    return ev


def test_pool_snapshot_errors_and_cross_game_restore(db, vocab):
    a, b = specs(2, base=5600)
    env = EncodedVecEnv(2, 2, cards=db, vocab=vocab, snapshots=True)
    env.reset(0, a)
    ev0 = first_event(env, 0)
    env.reset(1, a)
    first_event(env, 1)
    sid = env.snapshot(0)
    with pytest.raises(RuntimeError, match="busy"):
        env.step(0, 0)
    assert first_event(env, 0).obs.keys() == ev0.obs.keys()
    with pytest.raises(ValueError, match="env 0"):  # same game, other env
        env.restore(1, sid)
    with pytest.raises(ValueError, match="unknown"):
        env.restore(0, sid + 1000)
    ev = walk(env, 0, ev0, a.seed + DETOUR, 10**6)  # to the end of the game
    assert ev.obs is None
    with pytest.raises(RuntimeError, match="no running game"):
        env.snapshot(0)
    env.restore(0, sid)  # a finished game is still restorable
    assert frozen(first_event(env, 0).obs) == frozen(ev0.obs)
    env.reset(0, b)
    first_event(env, 0)
    with pytest.raises(ValueError, match="earlier game"):
        env.restore(0, sid)
    env.release(0, sid)
    with pytest.raises(ValueError, match="no snapshot"):
        env.release(0, sid)
    plain = EncodedVecEnv(1, 1, cards=db, vocab=vocab)
    plain.reset(0, a)
    first_event(plain, 0)
    with pytest.raises(RuntimeError, match="without snapshots"):
        plain.snapshot(0)
    assert env.arena_escapes() == 0


def test_pool_restore_rewinds_recorded_action_trace(db, vocab):
    spec = specs(1, base=5700, config=DuelConfig(max_decisions=60))[0]
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, snapshots=True, record_games=True)
    env.reset(0, spec)
    ev = walk(env, 0, first_event(env, 0), spec.seed, 10)
    sid = env.snapshot(0)
    first_event(env, 0)
    walk(env, 0, ev, spec.seed + DETOUR, 7)
    env.restore(0, sid)
    ev = first_event(env, 0)
    result = walk(env, 0, ev, spec.seed + 1, 10**6).result
    straight = EncodedVecEnv(1, 1, cards=db, vocab=vocab, record_games=True)
    straight.reset(0, spec)
    straight_ev = walk(straight, 0, first_event(straight, 0), spec.seed, 10)
    expected = walk(straight, 0, straight_ev, spec.seed + 1, 10**6).result
    assert result["action_indices"] == expected["action_indices"]
    assert key(result) == key(expected)


def rss_bytes():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096


def test_release_frees_snapshot_memory(db, vocab):
    if sys.platform != "linux":  # RSS is read from /proc and trimmed via glibc's malloc_trim
        pytest.skip("reads RSS from /proc and calls malloc_trim (Linux-only)")
    spec = specs(1, base=5800)[0]
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, snapshots=True)
    env.reset(0, spec)
    ev = walk(env, 0, first_event(env, 0), spec.seed, 20)
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    before = rss_bytes()
    sids = []
    for _ in range(40):
        sids.append(env.snapshot(0))
        first_event(env, 0)
    held = env.snapshot_bytes()
    assert env.snapshot_count() == 40 and held > 40 << 20
    grown = rss_bytes()
    assert grown - before > held // 2
    for sid in sids:
        env.release(0, sid)
    libc.malloc_trim(0)
    assert env.snapshot_count() == 0 and env.snapshot_bytes() == 0
    assert grown - rss_bytes() > held // 2
    env.step(0, int(np.flatnonzero(ev.obs["action_mask"])[0]))  # the env plays on
    env.recv(1)
