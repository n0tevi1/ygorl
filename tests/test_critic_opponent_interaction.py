# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from critic_opponent_interaction import interaction_features


def test_interactions_can_express_distinct_resource_slopes():
    resources = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    features = interaction_features(resources, torch.tensor([0, 1]), ["old-256x2", "mirror-u528"])
    # Shared and interaction slopes combine as w-u for old, w+u for mirror.
    weights = torch.tensor([1.0, 2.0, 0.0, 0.5, -1.0])
    torch.testing.assert_close(features @ weights, torch.tensor([10.0, 6.0]))
    torch.testing.assert_close(resources, torch.tensor([[2.0, 3.0], [2.0, 3.0]]))


def test_interactions_preserve_game_mapping_and_reject_unknown_opponents():
    resources = torch.randn(3, 13)
    ids = torch.tensor([1, 1, 0])
    a = interaction_features(resources, ids, ["old-256x2", "mirror-u528"])
    b = interaction_features(resources, 1 - ids, ["mirror-u528", "old-256x2"])
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="unknown diagnostic opponent"):
        interaction_features(resources, ids, ["new", "mirror-u528"])
