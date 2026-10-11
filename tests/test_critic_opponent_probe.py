# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from critic_opponent_probe import opponent_features


def test_indicator_follows_game_mapping_under_metadata_reordering():
    resources = torch.randn(4, 13)
    before = resources.clone()
    ids = torch.tensor([1, 0, 1, 0])
    a = opponent_features(resources, ids, ["old-256x2", "mirror-u528"], conditioned=True)
    b = opponent_features(resources, 1 - ids, ["mirror-u528", "old-256x2"], conditioned=True)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[:, -1], torch.tensor([1.0, -1.0, 1.0, -1.0]))
    torch.testing.assert_close(resources, before)
    torch.testing.assert_close(a[:, :-1], resources)


def test_control_does_not_read_opponent_metadata():
    resources = torch.randn(2, 13)
    assert opponent_features(resources, None, None, conditioned=False) is resources


def test_unknown_opponent_is_not_silently_mapped_to_old_policy():
    with pytest.raises(ValueError, match="unknown diagnostic opponent"):
        opponent_features(torch.zeros(1, 13), torch.tensor([0]), ["unseen"], conditioned=True)


@pytest.mark.parametrize("ids", [torch.tensor([-1]), torch.tensor([1]), torch.tensor([0.0])])
def test_invalid_game_mapping_is_rejected(ids):
    with pytest.raises(ValueError):
        opponent_features(torch.zeros(1, 13), ids, ["old-256x2"], conditioned=True)
