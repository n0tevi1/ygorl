"""Host-side deck shuffle (the core does not shuffle decks at start; EDOPro's host does)."""

from pathlib import Path

from ygorl.cards.ydk import load_ydk
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, shuffle_deck

DECK = load_ydk(Path(__file__).parent / "decks" / "snake_eye.ydk")


class FirstDecision:
    """Captures the events before the first decision, then stops the duel."""

    def __init__(self):
        self.events = None

    def act(self, point):
        if self.events is None:
            self.events = point.events
        return 0


def opening_hand(seed, **kwargs):
    agent = FirstDecision()
    Duel(seed, None, DECK, DECK, config=DuelConfig(max_decisions=1, **kwargs)).run(agent, agent)
    return next(m for m in agent.events if isinstance(m, M.Draw) and m.player == 0).cards


def test_shuffle_is_a_deterministic_permutation():
    cards = list(range(40))
    s = shuffle_deck(cards, 123, 0)
    assert sorted(s) == cards and s != cards
    assert s == shuffle_deck(cards, 123, 0)
    assert s != shuffle_deck(cards, 123, 1)  # each player has an independent stream
    assert s != shuffle_deck(cards, 124, 0)
    assert shuffle_deck([], 1, 0) == [] and shuffle_deck([7], 1, 0) == [7]


def test_shuffle_golden_vector():
    """Pins the algorithm (splitmix64 + unbiased Fisher-Yates) for the C++ port in M2."""
    assert shuffle_deck(list(range(10)), 2026, 0) == GOLDEN


def test_seed_changes_opening_hand():
    hands = {opening_hand(seed) for seed in range(8)}
    assert len(hands) >= 6
    assert opening_hand(3) == opening_hand(3)


def test_shuffle_can_be_disabled():
    assert opening_hand(1, shuffle_decks=False) == opening_hand(2, shuffle_decks=False)


GOLDEN = [8, 7, 3, 4, 5, 1, 0, 6, 9, 2]  # frozen: changing it changes every seeded game
