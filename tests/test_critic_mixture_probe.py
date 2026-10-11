# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from critic_mixture_probe import mixture_weights


def test_equal_game_weights_preserve_registered_corpus_mass():
    ids = torch.tensor([0, 0, 1, 2, 2, 2])
    w = mixture_weights(ids, ["broad", "broad", "new"], {"broad": 0.75, "new": 0.25})
    torch.testing.assert_close(
        torch.zeros(3, dtype=w.dtype).index_add_(0, ids, w),
        torch.tensor([0.375, 0.375, 0.25], dtype=w.dtype),
    )


def test_duplicating_samples_does_not_change_game_mass():
    ids = torch.tensor([1, 0, 1])
    labels, masses = ["a", "b"], {"a": 0.5, "b": 0.5}
    w = mixture_weights(ids, labels, masses)
    doubled = mixture_weights(ids.repeat_interleave(2), labels, masses)
    torch.testing.assert_close(doubled.reshape(-1, 2).sum(1), w)


@pytest.mark.parametrize("masses", [{"a": 1.0}, {"a": -0.1, "b": 1.1}, {"a": 0.5, "b": 0.6}])
def test_unregistered_or_invalid_masses_fail(masses):
    with pytest.raises(ValueError):
        mixture_weights(torch.tensor([0, 1]), ["a", "b"], masses)


def test_zero_sample_game_cannot_silently_take_training_mass():
    with pytest.raises(ValueError, match="eligible games"):
        mixture_weights(torch.tensor([0]), ["a", "b"], {"a": 0.5, "b": 0.5})
