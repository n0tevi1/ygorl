"""C++ HostPool: whole step loop (actions, encoding) on worker threads (T2.2 integration / T2.7)."""

from pathlib import Path

import numpy as np
import pytest

from ygorl import _core, paths
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv, chooser

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def specs(n, config=DuelConfig()):
    return [GameSpec(seed=3000 + i, deck_a=DECKS[NAMES[i % 10]], deck_b=DECKS[NAMES[(i * 3 + 1) % 10]], first=i % 2, config=config)
            for i in range(n)]  # fmt: skip


def sequential(db, vocab, spec, scripts=None):
    """HostDuel played alone with the same deterministic chooser."""
    duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=db, first=spec.first, config=spec.config)
    host = _core.HostDuel(db.to_core(), scripts if scripts is not None else default_scripts(),
                          [vocab.password(i) for i in range(2, len(vocab))])  # fmt: skip
    cfg = spec.config
    host.start(expand_seed(spec.seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
               [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
    step = 0
    while not host.done():
        obs = host.observe()
        if not host.actions():
            break
        legal = np.flatnonzero(obs["action_mask"])  # equivalent copies are masked (docs/encoding.md)
        host.act(int(legal[chooser(spec.seed, step, len(legal))]))
        step += 1
    return host.result()


def key(r):
    return (r["winner"], r["reason"], r["turns"], tuple(r["lp"]), r["decisions"], tuple(r["responses"]))


def play_pool(db, vocab, games, num_envs, threads):
    env = EncodedVecEnv(num_envs, threads, cards=db, vocab=vocab)
    return env.play(games, chooser)


def test_pool_matches_sequential_host(db, vocab):
    games = specs(6)
    pooled = play_pool(db, vocab, games, num_envs=4, threads=3)
    assert [key(r) for r in pooled] == [key(sequential(db, vocab, s)) for s in games]


def test_thread_count_does_not_change_results(db, vocab):
    games = specs(5)
    one = play_pool(db, vocab, games, num_envs=5, threads=1)
    three = play_pool(db, vocab, games, num_envs=5, threads=3)
    assert [key(r) for r in one] == [key(r) for r in three]


def test_limits_apply(db, vocab):
    games = specs(3, DuelConfig(max_decisions=12))
    for r in play_pool(db, vocab, games, num_envs=3, threads=2):
        assert r["reason"] == "decision_limit" and r["decisions"] == 12


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("mode", ["initial", "runtime", "debug"])
def test_native_lua_health_is_visible_and_errors_stop_the_game(db, vocab, pooled, mode):
    base = default_scripts()
    body = {
        "initial": 'error("native health regression")',
        "debug": 'Debug.Message("native health regression")',
        "runtime": """local e=Effect.CreateEffect(c)
            e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
            e:SetCode(EVENT_PHASE_START+PHASE_DRAW)
            e:SetOperation(function() if Duel.GetTurnCount()==2 then error("native health regression") end end)
            Duel.RegisterEffect(e,0)""",
    }[mode]
    suffix = (
        f"\nlocal old_initial_effect=s.initial_effect\nfunction s.initial_effect(c) old_initial_effect(c) {body} end\n"
    )
    scripts = _core.ScriptDirectory([str(p) for p in paths.script_directories()],
                                   {"c14558127.lua": base.read("c14558127.lua") + suffix.encode(),
                                    **{n: base.read(n) for n in ("proc_fusion.lua", "proc_synchro.lua")}})  # fmt: skip
    spec = GameSpec(0, DECKS["branded_despia"], DECKS["branded_despia"])
    if pooled:
        env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, scripts=scripts)
        (result,) = env.play([spec], chooser)
    else:
        result = sequential(db, vocab, spec, scripts)
    assert result["retries"] == result["unknown_messages"] == 0
    if mode == "debug":
        assert not result["script_errors"] and not result["error"]
        assert key(result) == key(sequential(db, vocab, spec))
    else:
        assert result["reason"] == "error" and result["winner"] is None
        assert result["script_errors"] and all("native health regression" in e for e in result["script_errors"])
        assert "native health regression" in result["error"]
        assert (result["decisions"] == 0) == (mode == "initial")


def test_observations_have_fixed_shapes(db, vocab):
    env = EncodedVecEnv(2, 2, cards=db, vocab=vocab)
    env.reset(0, specs(1)[0])
    with pytest.raises(RuntimeError, match="busy"):  # a job is in flight until recv returns it
        env.reset(0, specs(1)[0])
    (ev,) = env.recv(1)
    assert ev.env_id == 0 and ev.result is None and ev.player in (0, 1)
    assert ev.obs["cards"].shape == (160, 23) and ev.obs["actions"].shape == (128, 10)
    assert ev.obs["cards"].dtype == np.int32 and ev.obs["action_mask"].sum() > 0
    env.step(0, 0)
    (ev,) = env.recv(1)
    assert ev.env_id == 0


def test_chooser_is_deterministic_and_in_range():
    assert chooser(5, 7, 10) == chooser(5, 7, 10)
    assert all(0 <= chooser(s, t, 3) < 3 for s in range(20) for t in range(20))


@pytest.mark.parametrize("config", [DuelConfig(curriculum="solo"), DuelConfig(curriculum="handtrap"),
                                    DuelConfig(augmented_start=True)])  # fmt: skip
def test_unsupported_curriculum_settings_are_rejected(db, vocab, config):
    """The C++ step loop has no curriculum filtering yet (T2.6 lives in the Python tracker): refuse, don't ignore."""
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab)
    with pytest.raises(NotImplementedError, match="curriculum|augmented_start"):
        env.reset(0, specs(1, config)[0])
    assert env.pending() == 0


def test_skip_forced_plays_the_same_games_without_single_action_points(db, vocab):
    """skip_forced auto-plays decisions with exactly one choosable row (one legal action, or several that are
    equivalent copies) inside the C++ loop: the games are unchanged (the only choice is taken either way), but no
    such point reaches Python (performance)."""

    def by_state(obs):  # depends on the observation only, so both runs choose alike at every real decision
        legal = np.flatnonzero(obs["action_mask"])
        return int(legal[int(obs["globals"].sum() + obs["actions"][legal].sum()) % len(legal)])

    def run(skip):
        env = EncodedVecEnv(4, 2, cards=db, vocab=vocab, skip_forced=skip)
        games, results, seen, forced, copies = specs(5), {}, 0, 0, 0
        pending = iter(enumerate(games))
        for e in range(4):
            i, spec = next(pending)
            env.reset(e, spec)
            results[e] = [i]
        done = {}
        while len(done) < len(games):
            for ev in env.recv(1):
                if ev.result is not None:
                    done[results[ev.env_id][0]] = ev.result
                    nxt = next(pending, None)
                    if nxt is not None:
                        results[ev.env_id] = [nxt[0]]
                        env.reset(ev.env_id, nxt[1])
                    continue
                seen += 1
                forced += int(ev.obs["action_mask"].sum()) == 1
                copies += int(ev.obs["action_mask"].sum()) == 1 and ev.obs["globals"][20] > 1
                env.step(ev.env_id, by_state(ev.obs))
        return [done[i] for i in range(len(games))], seen, forced, copies

    plain, seen_plain, forced_plain, copies_plain = run(False)
    skipped, seen_skip, forced_skip, _ = run(True)
    assert [key(r) for r in skipped] == [key(r) for r in plain]
    assert forced_plain > copies_plain > 0 and forced_skip == 0
    assert seen_skip == seen_plain - forced_plain
