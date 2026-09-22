"""Tests for the matchup matrix, Nash mixture and alpha-rank (T3.3)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ygorl.agents import GreedyAgent
from ygorl.cards.ydk import load_ydk
from ygorl.data import EnvironmentConfigError, load_environment
from ygorl.engine.duel import DuelConfig
from ygorl.eval.matchup import (
    MatchupMatrix,
    MetaGame,
    alpha_rank,
    analyze,
    build_matrix,
    fixation_probability,
    nash_mixture,
)
from tests.test_environment import make_env

DECK_DIR = Path(__file__).parent / "decks"
THREE = [load_ydk(DECK_DIR / f"{n}.ydk") for n in ("snake_eye", "kashtira", "yubel")]
SHORT = DuelConfig(max_turns=4)  # tiny games: these tests check reproducibility, not deck strength

RPS = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
# Weighted RPS: rock beats scissors by a lot. Antisymmetric payoff A = 4 * (M - 0.5) has Nash (1/4, 1/2, 1/4).
WEIGHTED_RPS = np.array([[0.5, 0.25, 1.0], [0.75, 0.5, 0.25], [0.0, 0.75, 0.5]])
DOMINANT = np.array([[0.5, 0.9, 0.9], [0.1, 0.5, 0.5], [0.1, 0.5, 0.5]])


# ------------------------------------------------------------------ Nash


def test_nash_of_rock_paper_scissors_is_uniform():
    assert np.allclose(nash_mixture(RPS), [1 / 3] * 3, atol=1e-6)


def test_nash_of_weighted_rps():
    assert np.allclose(nash_mixture(WEIGHTED_RPS), [0.25, 0.5, 0.25], atol=1e-6)


def test_nash_of_a_dominant_strategy_is_pure():
    assert np.allclose(nash_mixture(DOMINANT), [1, 0, 0], atol=1e-6)


def test_nash_is_unexploitable():
    rng = np.random.default_rng(0)
    upper = rng.uniform(size=(5, 5))
    m = np.triu(upper, 1) + np.tril(1 - upper.T, -1) + 0.5 * np.eye(5)  # m + m.T == 1
    x = nash_mixture(m)
    assert x.min() >= 0 and math.isclose(x.sum(), 1.0)
    assert (x @ m).min() >= 0.5 - 1e-6  # no pure deck beats the mixture


# ------------------------------------------------------------------ alpha-rank


def test_fixation_probability_closed_forms():
    m = np.full((2, 2), 0.5)
    assert math.isclose(fixation_probability(m, 1, 0, alpha=5.0, population_size=20), 1 / 20)
    # Constant fitness advantage d: rho = (1 - e^{-a d}) / (1 - e^{-m a d}).
    # Mutant 1 in resident 0 with k mutants out of 10: f_1(k) - f_0(k) = 0.2 * 10 / 9 for every k.
    m = np.array([[0.5, 0.3], [0.7, 0.5]])
    rho = fixation_probability(m, 1, 0, alpha=2.0, population_size=10)
    d = 0.2 * 10 / 9
    assert math.isclose(rho, (1 - math.exp(-2 * d)) / (1 - math.exp(-10 * 2 * d)), rel_tol=1e-9)
    assert fixation_probability(m, 0, 1, alpha=2.0, population_size=10) < 1 / 10 < rho


def test_alpha_rank_of_rps_is_uniform():
    pi = alpha_rank(RPS, alpha=10.0, population_size=50)
    assert np.allclose(pi, [1 / 3] * 3, atol=1e-9)


def test_alpha_rank_concentrates_on_a_dominant_strategy():
    pi = alpha_rank(DOMINANT, alpha=10.0, population_size=50)
    assert pi[0] > 0.99 and math.isclose(pi.sum(), 1.0)
    assert math.isclose(pi[1], pi[2], rel_tol=1e-9)


def test_alpha_rank_neutral_game_and_large_alpha_are_stable():
    assert np.allclose(alpha_rank(np.full((4, 4), 0.5)), [0.25] * 4)
    pi = alpha_rank(DOMINANT, alpha=1e4, population_size=200)  # no overflow
    assert np.all(np.isfinite(pi)) and pi[0] > 1 - 1e-9


# ------------------------------------------------------------------ matrix from games


@pytest.fixture(scope="module")
def matrix():
    return build_matrix(THREE, GreedyAgent, pairs=1, seed=7, config=SHORT)


def test_matrix_shape_and_symmetry(matrix):
    assert isinstance(matrix, MatchupMatrix)
    assert matrix.decks == ("snake_eye", "kashtira", "yubel")
    m = matrix.array()
    assert m.shape == (3, 3)
    assert np.allclose(m + m.T, 1.0) and np.allclose(np.diag(m), 0.5)
    g = np.array(matrix.games)
    assert (g == g.T).all() and (np.diag(g) == 0).all() and (g[~np.eye(3, dtype=bool)] == 2).all()
    lo, hi = np.array(matrix.ci_low), np.array(matrix.ci_high)
    assert (lo <= m + 1e-12).all() and (m <= hi + 1e-12).all()


def test_matrix_and_nash_are_reproducible(matrix):
    again = build_matrix(THREE, GreedyAgent, pairs=1, seed=7, config=SHORT, workers=2)
    assert again == matrix
    assert np.array_equal(analyze(again).nash, analyze(matrix).nash)
    assert analyze(again).to_dict() == analyze(matrix).to_dict()


def test_matrix_does_not_depend_on_deck_order(matrix):
    order = [2, 0, 1]
    shuffled = build_matrix([THREE[i] for i in order], GreedyAgent, pairs=1, seed=7, config=SHORT)
    assert shuffled.decks == tuple(matrix.decks[i] for i in order)
    assert np.array_equal(shuffled.array(), matrix.array()[np.ix_(order, order)])


def test_duplicate_deck_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_matrix([THREE[0], THREE[0]], GreedyAgent, pairs=1, config=SHORT)


def test_analyze_reports_mixtures_by_deck(matrix):
    meta = analyze(matrix, alpha=5.0, population_size=20)
    assert isinstance(meta, MetaGame)
    assert set(meta.nash_by_deck()) == set(matrix.decks)
    assert math.isclose(sum(meta.nash_by_deck().values()), 1.0)
    assert math.isclose(sum(meta.alpha_rank_by_deck().values()), 1.0)
    assert meta.alpha == 5.0 and meta.population_size == 20


# ------------------------------------------------------------------ artifacts


def test_save_without_environment_needs_a_path(matrix, tmp_path):
    meta = analyze(matrix)
    with pytest.raises(ValueError, match="path"):
        meta.save()
    path = meta.save(tmp_path / "out" / "meta.json")
    data = json.loads(path.read_text())
    assert data["environment"] is None and data["decks"] == list(matrix.decks)
    assert MetaGame.load(path) == meta


def test_save_into_environment_artifacts_embeds_the_stamp(tmp_path):
    env = load_environment(make_env(tmp_path))
    matrix = build_matrix(THREE[:2], GreedyAgent, pairs=1, seed=1, env=env, config=SHORT)
    meta = analyze(matrix)
    path = meta.save(env=env, name="greedy")
    assert path == env.root / "artifacts" / "matrix" / "greedy.json"
    data = json.loads(path.read_text())
    assert data["environment"] == env.stamp()
    assert MetaGame.load(path, env=env) == meta
    # A matrix built under another environment cannot be filed under this one.
    with pytest.raises(EnvironmentConfigError):
        analyze(build_matrix(THREE[:2], GreedyAgent, pairs=1, seed=1, config=SHORT)).save(env=env)
    # Loading checks the stamp against the given environment.
    other = load_environment(make_env(tmp_path / "o", version="other-2026-10"))
    with pytest.raises(EnvironmentConfigError, match="belongs to environment"):
        MetaGame.load(path, env=other)
