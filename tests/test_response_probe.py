import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.eval.response_probe import ResponseRidge, bootstrap_clusters, common_valid, public_features  # noqa: E402


def test_error_continuations_are_removed_for_every_action():
    scores = np.array([[1, 0, np.nan, 1], [0, np.nan, 1, 1]])
    np.testing.assert_array_equal(common_valid(scores), [True, False, False, True])


def test_ridge_learns_action_differences_and_pass_stays_zero():
    x = np.array([[1.0, 0], [0, 1], [1, 1], [-1, 0]])
    y = x @ np.array([0.2, -0.3])
    teacher = ResponseRidge.fit(x, y, regularization=0.0001)
    np.testing.assert_allclose(teacher.scores(x), y, atol=0.0001)
    assert teacher.scores(np.zeros(2)) == 0
    with pytest.raises(ValueError):
        ResponseRidge.fit(x, [1, np.nan, 1, 0])


def test_bootstrap_preserves_repeated_games_in_a_pairing():
    a = bootstrap_clusters([0, 1, 0, 1], [0, 1, 2, 3], seed=3)
    b = bootstrap_clusters(np.repeat([0, 1, 0, 1], 4), np.repeat([0, 1, 2, 3], 4), seed=3)
    assert a == b
    with pytest.raises(ValueError):
        bootstrap_clusters([0, 1, 1], [0, 1, 1])


def test_public_features_drop_privileged_data_before_actor():
    class Actor:
        def features(self, obs):
            from types import SimpleNamespace

            assert set(obs) == {"globals", "actions", "action_mask"}
            return SimpleNamespace(context=obs["globals"].float(), actions=obs["actions"].float())

    obs = dict(globals=np.array([1, 2]), actions=np.array([[1, 2], [3, 4]]), action_mask=np.array([1, 1]))
    a = public_features(Actor(), [dict(obs, privileged=np.array([7, 8]))], "cpu")
    b = public_features(Actor(), [dict(obs, privileged=np.array([999, -15]))], "cpu")
    np.testing.assert_array_equal(a, b)
    assert not torch.is_tensor(a)


def test_response_window_matches_deployed_turn_and_legal_constraints():
    from ygorl.engine import constants as C
    from ygorl.env.encoding import ACTION_KINDS
    from ygorl.eval.response_probe import response_window

    g = np.zeros(22, dtype=int)
    g[3], g[18] = 4, C.MSG_SELECT_CHAIN
    obs = {"globals": g, "actions": np.zeros((10, 10), dtype=int), "action_mask": np.zeros(10, dtype=bool)}
    obs["action_mask"][[2, 7]] = True
    obs["actions"][7, 0] = ACTION_KINDS.index("pass") + 1
    legal, passed = response_window(obs)
    assert legal.tolist() == [2, 7] and passed == 7
    g[2] = 1
    assert response_window(obs) is None
    g[2] = 0
    g[3] = 5
    assert response_window(obs) is None
    g[3] = 4
    obs["action_mask"][:] = True
    assert response_window(obs) is None
    obs["action_mask"][:] = False
    obs["action_mask"][7] = True
    assert response_window(obs) is None


def test_simple_response_respects_noncontiguous_legality_and_bias():
    from ygorl.eval.response_probe import simple_response

    probs = np.array([0.9, 0.01, 0.02, 0.001, 0.079])
    legal = [1, 2, 4]
    assert simple_response(probs, legal, 4) == 2
    assert simple_response(probs, legal, 4, pass_bias=0) == 4
    assert simple_response(probs, legal, 4, pass_bias=-4) == 2


def test_response_contrasts_use_one_common_pairing_mask_and_family():
    from ygorl.eval.response_probe import response_contrasts

    scores = {
        k: np.full((4, 3, 4), v)
        for k, v in [("control", 0.4), ("reranked", 0.6), ("pass_bias", 0.5), ("always_activate", 0.3)]
    }
    scores["always_activate"][0, 1, 2] = np.nan
    result = response_contrasts(scores)
    assert len(result["contrasts"]) == 5
    for c in result["contrasts"].values():
        assert c["clusters"] == 3
    np.testing.assert_allclose(result["contrasts"]["reranked-control"]["ci99"], [0.2, 0.2])


def test_collection_excludes_a_pairing_with_less_than_half_common_continuations(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    from collect_response_panel import fit

    opponent = tmp_path / "opponent"
    opponent.mkdir()
    roots = {}
    for pair in range(3):
        game = 4 * pair
        roots[str(game)] = dict(
            game=game, pairing=pair, player=0, features=[[0, 0], [1, pair + 1]], probs=[0.5, 0.5], **{"pass": 0}
        )
        records = [
            dict(reason="error" if pair == 0 and rep >= 2 else "win", winner=1 - action)
            for action in range(2)
            for rep in range(5)
        ]
        (opponent / f"rollouts-{game}.json").write_text(json.dumps(dict(records=records)))
    (opponent / "states.json").write_text(
        json.dumps(dict(roots=roots, outcomes={key: {"status": "root"} for key in roots}))
    )
    fit(SimpleNamespace(out=tmp_path, pairings=3, continuations=5), ["opponent"])
    summary = json.loads((tmp_path / "training-summary.json").read_text())
    assert summary["excluded_pairings"] == [0]
    assert summary["training_roots"] == 2
