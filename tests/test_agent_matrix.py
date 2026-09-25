"""Tests for the agent matchup matrix (T4b.8, #85): how strong each agent is on a fixed sample of deck pairings."""

import json
from pathlib import Path

import numpy as np
import pytest

from ygorl.agents import AgentSpec
from ygorl.cards.ydk import load_ydk
from ygorl.data import EnvironmentConfigError, load_environment
from ygorl.engine.duel import DuelConfig
from ygorl.eval.agent_matrix import AgentMatrix, build_agent_matrix, cell_specs, pairing_slots, sample_pairings, score
from ygorl.eval.arena import GameRecord
from tests.test_environment import make_env

DECK_DIR = Path(__file__).parent / "decks"
THREE = [load_ydk(DECK_DIR / f"{n}.ydk") for n in ("snake_eye", "kashtira", "yubel")]
SHORT = DuelConfig(max_turns=4)  # tiny games: these tests check the matrix, not deck strength
AGENTS = {"random": AgentSpec("random"), "random2": AgentSpec("random"), "greedy": AgentSpec("greedy")}


@pytest.fixture(scope="module")
def matrix():
    return build_agent_matrix(AGENTS, THREE, pairings=6, seed=3, config=SHORT, workers=2)


def test_shape_symmetry_and_counts(matrix):
    assert isinstance(matrix, AgentMatrix)
    assert matrix.agents == ("greedy", "random", "random2")  # name order, whatever the input order
    assert matrix.specs == ("greedy", "random", "random")
    m = matrix.array()
    np.testing.assert_allclose(m + m.T, 1.0)
    assert np.all(np.diag(m) == 0.5)
    for i in range(3):
        for j in range(3):
            if i != j:  # 6 pairings x 2 deck assignments x 2 seats = 24 games, errors counted apart
                assert matrix.games[i][j] + matrix.errors[i][j] == 24
                assert matrix.ci_low[i][j] <= m[i, j] <= matrix.ci_high[i][j]
    assert len(matrix.pairings) == 6 and all(i != j for i, j in matrix.pairings)
    assert matrix.decks == tuple(d.name for d in THREE) and len(matrix.deck_hashes) == 3


def test_a_copy_of_an_agent_is_even_and_greedy_dominates_random(matrix):
    r, r2 = matrix.index("random"), matrix.index("random2")
    assert matrix.ci_low[r][r2] <= 0.5 <= matrix.ci_high[r][r2]
    assert matrix.against("random")["greedy"] > 0.5 and matrix.against("random2")["greedy"] > 0.5
    assert matrix.ranking()[0] == "greedy"
    assert matrix.alpha_rank[matrix.index("greedy")] > 0.9
    assert sum(matrix.nash) == pytest.approx(1.0) and sum(matrix.alpha_rank) == pytest.approx(1.0)


def test_reproducible_and_independent_of_agent_order(matrix):
    again = build_agent_matrix(dict(reversed(list(AGENTS.items()))), THREE, pairings=6, seed=3, config=SHORT)
    assert again == matrix  # same seed: bit for bit, whatever the order of the agents or the worker count


def test_every_cell_plays_the_same_games_so_a_smaller_matrix_is_a_sub_matrix(matrix):
    two = build_agent_matrix({"greedy": AgentSpec("greedy"), "random": AgentSpec("random")}, THREE, pairings=6,
                             seed=3, config=SHORT)  # fmt: skip
    g, r = matrix.index("greedy"), matrix.index("random")
    assert two.win_rate[0][1] == matrix.win_rate[g][r] and two.games[0][1] == matrix.games[g][r]


def test_each_pairing_is_played_with_both_deck_slots_and_both_first_players():
    pairs = sample_pairings(3, 4, seed=1)
    slots = pairing_slots(THREE, pairs, 1, SHORT)
    specs = cell_specs(AGENTS["greedy"], AGENTS["random"], slots, None, SHORT)
    assert len(specs) == 16 and not any(sp.config.shuffle_decks for sp in specs)
    for k, (i, j) in enumerate(pairs):
        four = specs[4 * k : 4 * k + 4]
        held = {(sp.deck_a.name, sp.deck_b.name) for sp in four}
        assert held == {(THREE[i].name, THREE[j].name), (THREE[j].name, THREE[i].name)}
        assert sorted(sp.first for sp in four) == [0, 0, 1, 1] and len({sp.seed for sp in four}) == 1
    assert sample_pairings(3, 4, seed=1) == pairs and sample_pairings(3, 4, seed=2) != pairs


def test_common_random_numbers_do_not_depend_on_the_seat():
    """The seat (a / b) follows name order; hands and agent seeds follow the deck slot. So two copies of a
    deterministic agent named before and after their opponent play exactly the same games against it."""
    m = build_agent_matrix({"a": AgentSpec("greedy"), "m": AgentSpec("random"), "z": AgentSpec("greedy")}, THREE,
                           pairings=4, seed=2, config=SHORT)  # fmt: skip
    a, mid, z = m.index("a"), m.index("m"), m.index("z")
    assert m.win_rate[a][mid] == m.win_rate[z][mid] and m.games[a][mid] == m.games[z][mid]
    assert m.win_rate[a][z] == 0.5  # the same agent in both seats of every game: every result mirrored


def test_error_games_are_counted_apart():
    rec = [GameRecord(pair=0, first=0, seed=0, winner=w, reason=reason) for w, reason in
           [(0, "win"), (1, "win"), (None, "decision_limit"), (None, "error"), (None, "exception")]]  # fmt: skip
    cell = score(rec)
    assert (cell.wins, cell.losses, cell.draws, cell.errors, cell.games) == (1, 1, 1, 2, 3)
    assert cell.win_rate == 0.5


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="at least two agents"):
        build_agent_matrix({"greedy": AgentSpec("greedy")}, THREE, pairings=1, config=SHORT)
    with pytest.raises(ValueError, match="at least two decks"):
        build_agent_matrix(AGENTS, THREE[:1], pairings=1, config=SHORT)
    with pytest.raises(ValueError, match="at least one deck pairing"):
        build_agent_matrix(AGENTS, THREE, pairings=0, config=SHORT)


def test_save_and_load_round_trip_bound_to_the_environment(matrix, tmp_path):
    path = matrix.save(tmp_path / "m.json")
    assert AgentMatrix.load(path) == matrix
    assert json.loads(path.read_text())["format"] == "ygorl-agent-matrix"
    with pytest.raises(ValueError, match="path is required"):
        matrix.save()

    env = load_environment(make_env(tmp_path / "e"))
    stamped = build_agent_matrix({"greedy": AgentSpec("greedy"), "random": AgentSpec("random")}, THREE, pairings=1,
                                 seed=1, env=env, config=SHORT)  # fmt: skip
    saved = stamped.save(env=env, name="baseline")
    assert saved.parent.name == "agent-matrix" and AgentMatrix.load(saved, env=env) == stamped
    with pytest.raises(EnvironmentConfigError):
        matrix.save(env=env)  # built without an environment
    other = load_environment(make_env(tmp_path / "o", version="other-2026-10"))
    with pytest.raises(EnvironmentConfigError, match="belongs to environment"):
        AgentMatrix.load(saved, env=other)
