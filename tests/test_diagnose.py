"""Per-card deck signals (#107): the opening hand from a spec's deck order, and the opening-hand effect."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from ygorl.build.diagnose import OPENING_HAND, opening_effects, opening_hand
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, deck_of_seat, shuffle_deck
from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location

DECKS = Path(__file__).parent / "decks"


def test_the_opening_hand_is_the_end_of_the_loaded_main_deck():
    """Locks the engine's draw order that the opening-hand effect relies on: with the host shuffle already applied
    (shuffle_decks False, as in arena / batched game specs) both players' first hands are the last 5 cards."""
    a, b = load_ydk(DECKS / "snake_eye.ydk"), load_ydk(DECKS / "kashtira.ydk")
    a = replace(a, main=tuple(shuffle_deck(a.main, 3, 0)))
    b = replace(b, main=tuple(shuffle_deck(b.main, 3, 1)))
    seen = {}

    class Probe:
        name = "probe"

        def observe(self, point, core):
            if not seen:
                for p in (0, 1):
                    got = parse_query_location(core.query_location(CARD_QUERY_FLAGS, p, C.LOCATION_HAND))
                    seen[p] = sorted(c.get("code") for c in got if c)

        def act(self, point):
            return 0

    for first in (0, 1):
        seen.clear()
        Duel(5, None, a, b, config=DuelConfig(shuffle_decks=False, max_decisions=2), first=first).run(Probe(), Probe())
        decks = (a, b)
        for p in (0, 1):
            assert seen[p] == sorted(opening_hand(decks[deck_of_seat(first, p)].main)), (first, p)
    assert OPENING_HAND == 5


def test_opening_effects_find_a_card_that_wins_when_held_and_ignore_one_that_does_not():
    rng = np.random.default_rng(0)
    deck = [1] * 3 + [2] * 3 + list(range(10, 44))  # 40 cards: card 1 wins when held, card 2 does nothing
    hands, scores = [], []
    for _ in range(4000):
        order = rng.permutation(deck)
        hand = opening_hand(order.tolist())
        hands.append(hand)
        scores.append(1.0 if 1 in hand else float(rng.random() < 0.4))
    eff = {e.card: e for e in opening_effects(hands, scores, deck)}
    assert eff[1].win_in == 1.0 and eff[1].effect > 0.4 and eff[1].copies == 3
    assert abs(eff[2].effect) < 3 * eff[2].stderr  # no effect: within noise
    assert eff[1].games_in + eff[1].games_out == 4000
    assert opening_effects(hands, scores, deck)[0].card == 1  # sorted by effect


def test_games_without_a_result_are_left_out():
    hands = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (1, 6, 7, 8, 9)]
    eff = {e.card: e for e in opening_effects(hands, [1.0, 0.0, math.nan], [1, 6])}
    assert eff[1].games_in == 1 and eff[1].games_out == 1 and eff[1].effect == 1.0
    assert eff[1].stderr == math.inf  # too few games for an interval
