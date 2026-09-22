"""Tests for the paired-seed Arena and its report (T3.2)."""

import json
import math
from pathlib import Path

import pytest

from ygorl.agents import GreedyAgent, RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.data import load_environment
from ygorl.engine.duel import DuelConfig
from ygorl.eval.arena import Arena, ArenaReport, GameRecord, derive_seed, merge, summarize, wilson_interval
from tests.test_environment import make_env

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
A, B = DECKS["snake_eye"], DECKS["kashtira"]
SHORT = DuelConfig(max_turns=4)  # keeps the tests that only check bookkeeping fast


class BrokenAgent:
    def __init__(self, seed):
        pass

    def act(self, point):
        return -1


# ------------------------------------------------------------------ statistics


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(8, 10)
    assert math.isclose(lo, 0.4902, abs_tol=1e-4) and math.isclose(hi, 0.9433, abs_tol=1e-4)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo99, hi99 = wilson_interval(8, 10, confidence=0.99)
    assert lo99 < 0.4902 and hi99 > 0.9433
    assert wilson_interval(500, 1000)[0] < 0.5 < wilson_interval(500, 1000)[1]


def test_derive_seed_is_stable_and_spreads():
    assert derive_seed(1, 2) == derive_seed(1, 2)
    seeds = {derive_seed(0, i) for i in range(1000)}
    assert len(seeds) == 1000
    assert derive_seed(0, 1, 2) != derive_seed(0, 2, 1)
    assert all(0 <= s < 2**63 for s in seeds)


def _record(pair, first, winner):
    return GameRecord(pair=pair, seed=pair, first=first, winner=winner, reason="win", turns=5, decisions=50)


def test_summarize_counts_sides_and_draws():
    records = [_record(0, 0, 0), _record(0, 1, 0), _record(1, 0, 1), _record(1, 1, None)]
    rep = summarize(records, agent_a="x", agent_b="y", deck_a="d1", deck_b="d2", seed=0)
    assert (rep.games, rep.wins, rep.losses, rep.draws) == (4, 2, 1, 1)
    assert rep.win_rate == pytest.approx(2.5 / 4)  # a draw counts half
    assert rep.ci[0] < rep.win_rate < rep.ci[1]
    assert (rep.as_first.games, rep.as_first.wins, rep.as_first.losses) == (2, 1, 1)
    assert (rep.as_second.games, rep.as_second.wins, rep.as_second.draws) == (2, 1, 1)
    assert rep.first_player_win_rate == pytest.approx(1.5 / 4)  # only game 1 went to the first player; the draw counts half
    assert rep.reasons == {"win": 4}
    assert not rep.significant()
    json.dumps(rep.to_dict())
    assert "win rate" in rep.summary()


# ------------------------------------------------------------------ pairing


def test_pairs_share_seed_and_opening_decks_and_swap_first():
    arena = Arena(GreedyAgent, RandomAgent)
    specs = arena.game_specs(A, B, pairs=3, seed=11)
    assert [(s.pair, s.first) for s in specs] == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    for g0, g1 in zip(specs[::2], specs[1::2]):
        assert g0.seed == g1.seed and g0.agent_seeds == g1.agent_seeds
        d0, d1 = g0.duel(), g1.duel()
        # deck a sits in engine seat 0 when a goes first, seat 1 otherwise; same order both games
        assert d0.loaded_decks()[0] == d1.loaded_decks()[1]
        assert d0.loaded_decks()[1] == d1.loaded_decks()[0]
        assert sorted(d0.loaded_decks()[0][0]) == sorted(A.main)
    assert specs[0].duel().loaded_decks() != specs[2].duel().loaded_decks()
    assert specs[0].seed != specs[2].seed


# ------------------------------------------------------------------ running


def test_arena_report_is_consistent():
    rep = Arena(GreedyAgent, RandomAgent).run(A, B, pairs=2, seed=1)
    assert isinstance(rep, ArenaReport)
    assert rep.agent_a == "greedy" and rep.agent_b == "random" and rep.deck_a == "snake_eye"
    assert rep.games == 4 and rep.wins + rep.losses + rep.draws == 4
    assert rep.as_first.games == rep.as_second.games == 2
    assert sum(rep.reasons.values()) == 4
    assert rep.retries == 0 and rep.unknown_messages == 0 and rep.errors == 0
    assert rep.ci[0] <= rep.win_rate <= rep.ci[1]
    assert len(rep.records) == 4 and rep.environment is None


def test_results_do_not_depend_on_worker_count():
    serial = Arena(GreedyAgent, RandomAgent, workers=1, config=SHORT).run(A, B, pairs=3, seed=5)
    parallel = Arena(GreedyAgent, RandomAgent, workers=2, config=SHORT).run(A, B, pairs=3, seed=5)
    assert serial.records == parallel.records
    assert serial.to_dict() == parallel.to_dict()


def test_run_many_matches_individual_runs():
    arena = Arena(GreedyAgent, RandomAgent, workers=2, config=SHORT)
    reps = arena.run_many([(A, B), (B, A, 9)], pairs=1, seed=3)
    assert [(r.deck_a, r.seed) for r in reps] == [("snake_eye", 3), ("kashtira", 9)]
    assert reps[0].records == Arena(GreedyAgent, RandomAgent, config=SHORT).run(A, B, pairs=1, seed=3).records
    assert reps[1].records == Arena(GreedyAgent, RandomAgent, config=SHORT).run(B, A, pairs=1, seed=9).records
    pooled = merge(reps)
    assert pooled.games == 4 and pooled.deck_a == "*" and pooled.records == reps[0].records + reps[1].records


def test_errors_are_recorded_not_raised():
    rep = Arena(BrokenAgent, RandomAgent).run(A, B, pairs=1, seed=0)
    assert rep.errors == 2 and rep.reasons == {"exception": 2}
    assert all(r.winner is None and "out of range" in r.error for r in rep.records)


def test_environment_is_stamped(tmp_path):
    env = load_environment(make_env(tmp_path))
    rep = Arena(RandomAgent, RandomAgent, env=env, config=SHORT).run(A, B, pairs=1, seed=0)
    assert rep.environment == env.stamp()
    assert rep.to_dict()["environment"] == env.stamp()
