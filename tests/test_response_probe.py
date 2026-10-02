import numpy as np
import pytest
import torch

from ygorl.eval.response_probe import ResponseRidge, bootstrap_clusters, common_valid, public_features


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
