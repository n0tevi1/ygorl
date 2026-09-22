"""C++ HostPool: whole step loop (actions, encoding) on worker threads (T2.2 integration / T2.7)."""

from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
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


def sequential(db, vocab, spec):
    """HostDuel played alone with the same deterministic chooser."""
    duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=db, first=spec.first, config=spec.config)
    host = _core.HostDuel(db.to_core(), default_scripts(), [vocab.password(i) for i in range(2, len(vocab))])
    cfg = spec.config
    host.start(expand_seed(spec.seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
               [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
    step = 0
    while not host.done():
        obs = host.observe()
        if not host.actions():
            break
        host.act(chooser(spec.seed, step, int(obs["action_mask"].sum())))
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
