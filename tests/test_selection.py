"""Evaluation budgets for deck evolution (ygorl.build.selection) with synthetic paired games (no engine)."""

import zlib

import numpy as np
import pytest

from ygorl.build.selection import obrien_fleming_bounds, sequential_validate, top_two_thompson
from ygorl.cards.ydk import Deck


def _normals(seed, stream, ks):
    """Standard normals per pair index, the same for a (seed, stream, k) on every call."""
    chunks = {k // 100 for k in ks}
    table = {c: np.random.default_rng([seed, stream, c]).standard_normal(100) for c in chunks}
    return np.array([table[k // 100][k % 100] for k in ks])


def _stream(name):
    return 1 + zlib.crc32(name.encode()) % 997


class PairedGames:
    """Stand-in evaluator with continuous scores: on pair ``k`` every deck shares ``0.5 + 0.3 z_k`` (common random
    numbers); the parent adds its own ``parent_noise`` term and each child ``effect + noise`` of its own. The parent's
    term enters every child's paired difference, so the children's differences are correlated, as in real games."""

    def __init__(self, effects, noise=0.18, parent_noise=0.1, seed=0):
        self.effects, self.noise, self.parent_noise, self.seed = dict(effects), noise, parent_noise, seed
        self.games, self.calls = 0, []

    def play(self, jobs):
        self.calls.append([(d.name, r.start, r.stop) for d, r in jobs])
        out = []
        for d, pairs in jobs:
            ks = list(pairs)
            shared = 0.5 + 0.3 * _normals(self.seed, 0, ks)
            if d.name == "base":
                out.append(shared + self.parent_noise * _normals(self.seed, 1, ks))
            else:
                own = _normals(self.seed, 1 + _stream(d.name), ks)
                out.append(shared + self.effects.get(d.name, 0.0) + self.noise * own)
            self.games += 2 * len(ks)
        return out


class DiscretePairs:
    """Stand-in evaluator with real game results: a pair is two games scored 1 / 0 (pair score 0, 0.5 or 1). Each of
    a child's games repeats the parent's result on the same seed with probability ``copy`` (common random numbers;
    M1 measured a per-game correlation of 0.80) and is otherwise an independent game won with ``p + effect /
    (1 - copy)``, so the child's win rate is ``p + effect``."""

    def __init__(self, effects, p=0.45, copy=0.8, seed=0):
        self.effects, self.p, self.copy, self.seed = dict(effects), p, copy, seed
        self.games = 0

    def _uniform(self, stream, ks):
        """Uniforms per (pair, game), the same for a (seed, stream, k) on every call."""
        chunks = {k // 100 for k in ks}
        table = {c: np.random.default_rng([self.seed, stream, c]).random((100, 2)) for c in chunks}
        return np.array([table[k // 100][k % 100] for k in ks]).reshape(len(ks), 2)

    def play(self, jobs):
        out = []
        for d, pairs in jobs:
            ks = list(pairs)
            won = (self._uniform(0, ks) < self.p).astype(float)
            if d.name != "base":
                s = _stream(d.name)
                q = self.p + self.effects.get(d.name, 0.0) / (1 - self.copy)
                own = (self._uniform(2 * s, ks) < q).astype(float)
                won = np.where(self._uniform(2 * s + 1, ks) < self.copy, won, own)
            out.append(won.mean(-1))
            self.games += 2 * len(ks)
        return out


BASE = Deck(main=(1,) * 40, name="base")


def decks(n):
    return [Deck(main=(1,) * 40, name=f"c{i}") for i in range(n)]


def test_thompson_picks_the_truly_best_child_over_many_seeds():
    # most edits are neutral or worse (M1); one child is clearly better
    effects = {"c0": 0.0, "c1": -0.02, "c2": 0.01, "c3": 0.08, "c4": 0.0, "c5": -0.04, "c6": -0.01, "c7": 0.01}
    hits, pairs = 0, []
    for seed in range(100):
        ev = PairedGames(effects, seed=seed)
        race = top_two_thompson(BASE, decks(8), ev, rng=np.random.default_rng(seed))
        hits += race.best == 3
        pairs.append(race.pairs)
        assert race.pairs <= 600 and all(a.pairs % 25 == 0 and a.pairs >= 25 for a in race.arms)
        assert len(race.base_scores) == max(a.pairs for a in race.arms)
    # the misses are early confident stops on a null child (8 children at 25 pairs each): validation catches those
    assert hits >= 85 and np.mean(pairs) < 400


def test_thompson_spends_on_the_leaders_and_stops_when_confident():
    effects = {"c0": 0.15, "c1": 0.0, "c2": 0.0, "c3": 0.0}
    ev = PairedGames(effects, seed=1)
    race = top_two_thompson(BASE, decks(4), ev, rng=np.random.default_rng(1), log=print)
    assert race.best == 0 and race.stop == "confident" and race.arms[0].p_positive > 0.95
    assert race.pairs < 600 and race.rounds <= 3
    # a first batch each, all in one call with the parent; later rounds also play in one call per round
    assert sorted(ev.calls[0]) == sorted([("c0", 0, 25), ("c1", 0, 25), ("c2", 0, 25), ("c3", 0, 25), ("base", 0, 25)])
    assert abs(sum(a.p_best for a in race.arms) - 1) < 1e-9


def test_thompson_prior_breaks_ties_and_null_children_run_out_the_budget():
    effects = dict.fromkeys(["c0", "c1", "c2"], 0.0)
    race = top_two_thompson(BASE, decks(3), PairedGames(effects), rng=np.random.default_rng(0),
                            prior=[(0.0, 0.01), (0.0, 0.01), (0.0, 0.01)])  # fmt: skip
    assert race.stop == "budget" and race.pairs == 600
    race = top_two_thompson(BASE, decks(2), PairedGames({}), max_pairs=50, prior=[(0.0, 0.01), (0.2, 0.01)])
    assert race.best == 1 and race.arms[1].mean > 0.1  # a tight prior dominates 25 pairs
    assert top_two_thompson(BASE, [], PairedGames({})).stop == "empty"
    with pytest.raises(ValueError):
        top_two_thompson(BASE, decks(2), PairedGames({}), prior=[(0.0, 0.0), (0.0, 1.0)])


def test_obrien_fleming_bounds_match_the_textbook_and_hold_alpha():
    # Lan-DeMets O'Brien-Fleming, 5 equal looks, one-sided 0.025 (two-sided 0.05): 4.877 3.357 2.680 2.290 2.031
    b = obrien_fleming_bounds((0.2, 0.4, 0.6, 0.8, 1.0), 0.025)
    assert np.allclose(b, (4.877, 3.357, 2.680, 2.290, 2.031), atol=0.01)
    ten = obrien_fleming_bounds(tuple(k / 10 for k in range(1, 11)), 0.05)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((200_000, 10)).cumsum(-1) / np.sqrt(10)  # W(k/10)
    z = w / np.sqrt(np.arange(1, 11) / 10)
    assert abs((z > np.array(ten)).any(-1).mean() - 0.05) < 0.003
    with pytest.raises(ValueError):
        obrien_fleming_bounds((0.5, 0.4), 0.05)


@pytest.mark.parametrize("alpha", [0.05, 0.025])
def test_sequential_false_positive_rate_under_no_effect_is_at_most_alpha(alpha):
    runs = 2000
    accepted, pairs = 0, []
    for seed in range(runs):
        v = sequential_validate(BASE, decks(1)[0], PairedGames({"c0": 0.0}, seed=seed), alpha=alpha, look=100, cap=1000)
        accepted += v.accepted
        pairs.append(v.pairs)
    sd = np.sqrt(alpha * (1 - alpha) / runs)
    assert accepted / runs <= alpha + 2 * sd
    assert np.mean(pairs) < 500  # the non-binding futility bound stops most null runs well before the cap


def test_sequential_false_positive_rate_with_discrete_correlated_games():
    """Pair scores 0 / 0.5 / 1, parent and child results correlated game by game as under common random numbers,
    default alpha 0.025: the false-positive rate stays at most alpha and futility still cuts the null runs short."""
    alpha, runs = 0.025, 1000
    accepted, pairs = 0, []
    for seed in range(runs):
        v = sequential_validate(BASE, decks(1)[0], DiscretePairs({"c0": 0.0}, seed=seed))
        accepted += v.accepted
        pairs.append(v.pairs)
    assert accepted / runs <= alpha + 2 * np.sqrt(alpha * (1 - alpha) / runs)
    assert np.mean(pairs) < 500
    # and a real +5 pp is usually accepted within the cap
    hits = sum(sequential_validate(BASE, decks(1)[0], DiscretePairs({"c0": 0.05}, seed=s)).accepted for s in range(100))
    assert hits >= 70


def test_sequential_accepts_a_large_effect_early_and_uses_fresh_pairs():
    ev = PairedGames({"c0": 0.08})
    v = sequential_validate(BASE, decks(1)[0], ev, offset=5000, log=print)
    assert v.accepted and v.pairs < 1000 and v.looks[-1].lower > 0
    assert ev.calls[0] == [("c0", 5000, 5100), ("base", 5000, 5100)]
    v = sequential_validate(BASE, decks(1)[0], PairedGames({"c0": -0.05}), min_effect=0.01)
    assert v.decision == "futile" and not v.accepted


def test_sequential_never_decides_on_a_zero_variance_run():
    class Ties:
        games = 0

        def play(self, jobs):
            return [np.full(len(r), 0.5) for _, r in jobs]

    v = sequential_validate(BASE, decks(1)[0], Ties(), look=4, cap=8)
    assert v.decision == "cap" and v.looks[-1].upper > 0.05
