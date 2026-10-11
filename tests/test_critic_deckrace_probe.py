# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from critic_deckrace_probe import deckrace_features


def test_raw_count_boundary_and_both_sides_exhaustion():
    resources = torch.zeros(4, 13)
    resources[:, 3] = torch.tensor([40, 6, 5, 0]) / 40
    resources[:, 8] = torch.tensor([0, 5, 6, 0]) / 40
    torch.testing.assert_close(
        deckrace_features(resources),
        torch.tensor([[0, 1, 0], [0, 1 / 6, 0], [1 / 6, 0, 0], [1, 1, 1]]),
    )


def test_swapping_players_swaps_pressure_and_preserves_interaction():
    resources = torch.rand(20, 13)
    resources[:, [3, 8]] /= 5
    before = resources.clone()
    swapped = resources.clone()
    swapped[:, [3, 8]] = resources[:, [8, 3]]
    a, b = deckrace_features(resources), deckrace_features(swapped)
    torch.testing.assert_close(a[:, [1, 0, 2]], b)
    torch.testing.assert_close(resources, before)


@pytest.mark.parametrize("count", [-1.0, float("nan"), float("inf")])
def test_invalid_deck_counts_are_rejected(count):
    resources = torch.zeros(1, 13)
    resources[0, 8] = count
    with pytest.raises(ValueError):
        deckrace_features(resources)
