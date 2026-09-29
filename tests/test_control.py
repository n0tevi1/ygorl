"""Critic control variate (ygorl.build.control) on a synthetic game: the opening hand moves the win chance and the
"critic" is a noisy function of the opening hand (no engine, no network)."""

from dataclasses import replace

import numpy as np

from ygorl.build.control import CriticControlVariate
from ygorl.build.diagnose import opening_hand
from ygorl.build.tuner import Edit, apply
from ygorl.cards.ydk import Deck
from ygorl.engine.duel import shuffle_deck
from ygorl.env import GameSpec

DECK = Deck(main=tuple(range(1, 41)), name="d")
OPP = Deck(main=tuple(range(101, 141)), name="o")
GOOD = set(range(1, 41, 4))  # 10 good cards: each one in the opening hand raises the win chance


def win_chance(spec):
    good = sum(c in GOOD for c in opening_hand(spec.deck_a.main))
    return float(np.clip(0.1 + 0.25 * good, 0, 1))


def specs(n, seed=0):
    out = []
    for i in range(n):
        s = seed * 1_000_003 + i
        a = replace(DECK, main=tuple(shuffle_deck(DECK.main, s, 0)))
        out.append(GameSpec(seed=s, deck_a=a, deck_b=OPP, first=i % 2))
    return out


def games(sp, rng):
    return np.array([1.0 if rng.random() < win_chance(x) else 0.0 for x in sp])


def noisy_critic(noise, bias=0.0, seed=0):
    rng = np.random.default_rng(seed)

    def values(sp):  # +1 / -1 scale, like the value head
        return np.array([2 * win_chance(x) - 1 + bias + noise * rng.standard_normal() for x in sp])

    return values


def test_control_variate_keeps_the_mean_and_lowers_the_variance():
    sp = specs(4000)
    score = games(sp, np.random.default_rng(1))
    truth = np.mean([win_chance(x) for x in specs(8000, seed=7)])  # E[score] over the shuffle
    for noise, bias in ((0.2, 0.0), (0.6, 0.3)):  # a fair critic and a noisy, biased one
        cv = CriticControlVariate(noisy_critic(noise, bias), k=8)
        adj = score - cv(sp)
        assert cv.readings == len(sp) * 9
        se = adj.std(ddof=1) / np.sqrt(len(adj))
        assert abs(adj.mean() - truth) < 3 * se and abs(score.mean() - truth) < 3 * score.std() / np.sqrt(len(sp))
        if noise == 0.2:
            assert adj.var() < 0.9 * score.var()
        else:  # a critic noisier than the luck it reads adds its noise: still unbiased, but no help (hence M5)
            assert adj.var() > score.var()
    # only the luck of the opening is removed: an exact critic removes the share of variance the opening explains
    exact = score - CriticControlVariate(noisy_critic(0.0), k=16)(sp)
    explained = np.var([win_chance(x) for x in sp]) / score.var()
    assert 1 - exact.var() / score.var() > 0.8 * explained


def test_luck_has_mean_zero_over_repeated_runs():
    """The unbiasedness itself, independent of any one sample: the luck averages to 0 over many games even with a
    critic that is wrong (it ignores most of what matters) and biased."""
    sp = specs(3000, seed=3)

    def wrong(x):
        return np.array([0.7 + 0.5 * (x_.deck_a.main[-1] in GOOD) for x_ in x])

    luck = CriticControlVariate(wrong, k=4).luck(sp)
    assert abs(luck.mean()) < 3 * luck.std() / np.sqrt(len(luck))


def test_parent_and_child_get_matching_alternatives():
    child = apply(DECK, Edit(DECK.main[7], 999, "main"))
    s = specs(1)[0]
    s_child = replace(s, deck_a=replace(child, main=tuple(shuffle_deck(child.main, s.seed, 0))))
    cv = CriticControlVariate(noisy_critic(0.0), k=5, seed=11)
    for a, b in zip(cv.alternatives([s]), cv.alternatives([s_child]), strict=True):
        diff = [(x, y) for x, y in zip(a.deck_a.main, b.deck_a.main, strict=True) if x != y]
        assert diff == [(DECK.main[7], 999)]
        assert a.deck_b == s.deck_b and a.first == s.first and a.seed == s.seed
    assert len({alt.deck_a.main for alt in cv.alternatives([s])}) == 5
    nan = CriticControlVariate(lambda x: np.full(len(x), np.nan), k=2)
    assert (nan([s]) == 0).all() and nan([]).shape == (0,)
