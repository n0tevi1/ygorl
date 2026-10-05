import json

import numpy as np
import pytest

from tools.compare_checkpoints import read_cell
from tools.diagnose_losses import report
from ygorl.eval.paired_panel import compare_panel


def test_paired_difference_retains_shared_noise_and_fixed_opponent_weights():
    # Perfectly shared noise cancels in the difference, despite each row's uncertain absolute score.
    control = np.tile(np.linspace(0, 0.8, 20)[:, None, None], (1, 3, 4))
    result = compare_panel({"ctrl": control, "candidate": control + 0.1}, "ctrl")
    candidate = result["candidates"]["candidate"]
    assert candidate["score"] == pytest.approx(0.5)
    assert candidate["difference_vs_control"] == pytest.approx(0.1)
    assert candidate["ci95"] == pytest.approx([0.1, 0.1])
    assert result["games_per_candidate_used"] == 240


def test_bootstrap_does_not_count_replicated_games_as_independent():
    ctrl = np.full((40, 1, 4), 0.5)
    candidate = np.repeat((np.arange(40) % 2)[:, None, None], 4, axis=2)
    one = compare_panel({"ctrl": ctrl, "candidate": candidate}, "ctrl", seed=19)
    many = compare_panel({"ctrl": np.tile(ctrl, (1, 20, 1)),
                          "candidate": np.tile(candidate, (1, 20, 1))}, "ctrl", seed=19)  # fmt: skip
    assert one["candidates"]["candidate"]["ci95"] == many["candidates"]["candidate"]["ci95"]
    assert one["candidates"]["candidate"]["ci95"][1] > 0.1


def test_errors_drop_common_clusters_without_different_denominators():
    ctrl = np.zeros((5, 2, 4))
    candidate = np.ones_like(ctrl)
    ctrl[0, 0, 0] = np.nan
    candidate[1, 1, 1] = np.nan
    result = compare_panel({"ctrl": ctrl, "candidate": candidate}, "ctrl")
    assert result["pairings_used"] == 3
    assert result["pairings_excluded"] == 2
    assert result["candidates"]["candidate"]["difference_vs_control"] == 1
    for c in result["candidates"].values():
        assert c["error_games"] == 1


@pytest.mark.parametrize("bad", [np.zeros((3, 4)), np.zeros((3, 2, 2)), np.full((3, 1, 4), np.inf),
                                np.full((3, 1, 4), np.nan), np.full((3, 1, 4), 2)])  # fmt: skip
def test_invalid_or_insufficient_data_rejected(bad):
    with pytest.raises(ValueError):
        compare_panel({"ctrl": bad, "candidate": bad}, "ctrl")


def game(pairing, first, score, reason="win"):
    return dict(pairing=pairing, first=first, score=score, reason=reason, turns=3, win_reason=1,
                decisions=10, deck="a", opp_deck="b", lp=[8000, 0])  # fmt: skip


def test_diagnosis_scores_both_first_players_and_excludes_host_errors(capsys):
    report([game(0, True, 1), game(0, False, 0), game(1, True, 0.5, "error")], "self")
    output = capsys.readouterr().out
    assert "2 games (errors 1)" in output
    assert "pooled first-player score 1.000 (n=2)" in output
    assert "deck pairing explains" not in output


def test_coin_flip_null_no_longer_claims_deck_variance_explanation(capsys):
    rng = np.random.default_rng(20261002)
    games = [game(k, first, float(rng.integers(2))) for k in range(1000) for first in (True, False)]
    report(games, "self")
    output = capsys.readouterr().out
    assert "not variance explained by decks" in output
    assert "deck pairing explains" not in output


@pytest.mark.parametrize("games", [[], [game(0, True, None, "exception")], [game(0, True, 0.5)]])
def test_diagnosis_handles_empty_and_incomplete_pairs(games, capsys):
    report(games, "self")
    assert "nan" not in capsys.readouterr().out.lower()


def test_saved_game_order_is_checked_before_resume(tmp_path):
    path = tmp_path / "cell.json"
    records = [dict(pair=0, seed=17, first=f, reason="win", winner=0) for f in (0, 1, 1, 0)]
    cell = dict(candidate="a", opponent="b", records=records)
    path.write_text(json.dumps(cell))
    assert read_cell(path, "a", "b", [(17, None, None)]).shape == (1, 4)
    records[1]["seed"] = 18
    path.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="game order"):
        read_cell(path, "a", "b", [(17, None, None)])


@pytest.mark.parametrize(
    "health",
    [
        {"script_errors": 1},
        {"undecodable_messages": 1},
        {"retries": 1},
        {"unknown_messages": 1},
        {"error": "engine diagnostic"},
    ],
)
def test_saved_cell_health_failure_is_missing_not_a_win(tmp_path, health):
    path = tmp_path / "cell.json"
    records = [dict(pair=0, seed=17, first=f, reason="win", winner=0) for f in (0, 1, 1, 0)]
    records[1].update(health)
    path.write_text(json.dumps(dict(candidate="a", opponent="b", records=records)))
    scores = read_cell(path, "a", "b", [(17, None, None)])
    assert np.isnan(scores[0, 1]) and scores[0, [0, 2, 3]].tolist() == [1, 1, 1]
