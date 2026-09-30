"""Fractional-factorial evaluation of edit groups (ygorl.build.factorial, #152) on a synthetic evaluator."""

import zlib

import numpy as np
import pytest

from ygorl.build.factorial import (GENERATORS, LETTERS, design, estimate, evaluate_edits, games_to_detect,
                                   one_at_a_time, place, plan, terms_of, variant)  # fmt: skip
from ygorl.build.signals import CardValueModel
from ygorl.build.tuner import Edit
from ygorl.cards.ydk import Deck

PARENT = Deck(main=tuple([1] * 3 + [2] * 2 + list(range(10, 45))), extra=(900,), name="parent")
EDITS = [Edit(10 + j, 50 + j, "main") for j in range(8)]  # edit j puts in card 50 + j


def legal(deck):
    return len(deck.main) == 40 and max(deck.counts().values()) <= 3


class Synthetic:
    """Paired games with known effects: a game is won when a uniform is below 0.45 + the deck's card effects +
    interactions (both cards present) + 0.05 going first. The uniform is shared by every deck on the pair (common
    random numbers) with probability ``copy``, else the deck's own."""

    def __init__(self, seed, main, inter=None, copy=0.8):
        self.seed, self.main, self.inter, self.copy, self.games = seed, main, inter or {}, copy, 0

    def p(self, deck):
        cards = set(deck.main)
        return (0.45 + sum(v for c, v in self.main.items() if c in cards)
                + sum(v for (a, b), v in self.inter.items() if a in cards and b in cards))  # fmt: skip

    def play(self, jobs, per_game=False):
        top = max(r.stop for _, r in jobs)
        shared = np.random.default_rng([self.seed, 0]).random((top, 2))
        out = []
        for deck, pairs in jobs:
            own = np.random.default_rng([self.seed, zlib.crc32(repr(deck.main).encode())]).random((top, 3))
            idx = np.arange(pairs.start, pairs.stop)
            u = np.where(own[idx, :1] < self.copy, shared[idx], own[idx, 1:])
            g = (u < self.p(deck) + np.array([0.05, 0.0])).astype(float)
            self.games += g.size
            out.append(g if per_game else g.mean(1))
        return out


def expected(d, main, inter):
    """The effects the design estimates: the truth's ±1 regression over the full 2^k factorial, each term summed over
    its alias chain (with signs)."""
    k = d.k
    full = np.array([[1.0 if r >> j & 1 else -1.0 for j in range(k)] for r in range(2**k)])
    ind = (full + 1) / 2
    mu = sum(main.get(50 + j, 0.0) * ind[:, j] for j in range(k)) + sum(
        v * ind[:, a - 50] * ind[:, b - 50] for (a, b), v in inter.items())  # fmt: skip
    beta = {}
    for m in range(1, 2**k):
        col = np.prod(full[:, [j for j in range(k) if m >> j & 1]], axis=1)
        beta[m] = float(col @ mu) / 2**k
    out = []
    for term in terms_of(d):
        m = sum(1 << j for j in term)
        out.append(2 * (beta[m] + sum(s * beta.get(a, 0.0) for s, a in d.aliases(m) if a)))
    return np.array(out)


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_main_and_interaction_estimates_are_unbiased(k):
    main = {50: 0.06, 51: -0.04, 52: 0.03, 53: 0.0, 54: 0.02, 55: -0.02}
    inter = {(50, 51): 0.04, (51, 52): -0.03}
    d = design(k)
    truth = expected(d, main, inter)
    est, ses = [], []
    for seed in range(300):
        r = evaluate_edits(PARENT, EDITS[:k], Synthetic(seed, main, inter), legal=legal, pairs=200)
        est.append([e.effect for e in r.effects])
        ses.append([e.stderr for e in r.effects])
    est, ses = np.array(est), np.array(ses)
    bias = est.mean(0) - truth
    assert np.all(np.abs(bias) < 4 * est.std(0) / np.sqrt(len(est)) + 1e-4), (bias, truth)
    # the cluster-robust standard error matches the spread of the estimates
    assert np.all(np.abs(ses.mean(0) / est.std(0) - 1) < 0.15)
    if k == 4:  # by hand: D = ABC; main effect = gain averaged over the others; AB = CD is half the extra of both
        assert np.allclose(truth[:4], [0.06 + 0.02, -0.04 + 0.02 - 0.015, 0.03 - 0.015, 0.0])
        assert np.isclose(truth[4], 0.02) and r.effects[4].label == "AB = CD"


def test_equal_games_variance_advantage_over_one_at_a_time():
    """k = 4: 8 variants × 100 pairs against the parent + 4 single edits × 160 pairs (1,600 games each)."""
    main = {50: 0.03, 51: -0.02, 52: 0.01, 53: 0.0}
    fact, oat = [], []
    for seed in range(400):
        f = evaluate_edits(PARENT, EDITS[:4], Synthetic(seed, main), legal=legal, pairs=100)
        assert f.games == 1600
        fact.append([e.effect for e in f.main()])
        o = one_at_a_time(PARENT, EDITS[:4], Synthetic(10_000 + seed, main), pairs=160)
        oat.append([m for m, _ in o])
    fact, oat = np.array(fact), np.array(oat)
    assert np.allclose(fact.mean(0), [0.03, -0.02, 0.01, 0.0], atol=0.004)
    assert np.allclose(oat.mean(0), [0.03, -0.02, 0.01, 0.0], atol=0.004)
    ratio = float(oat.var(0).mean() / fact.var(0).mean())
    print(f"variance ratio one-at-a-time / factorial at equal games (k = 4): {ratio:.2f} (theory (k + 1) / 2 = 2.5)")
    assert 2.0 < ratio < 3.1
    # games per detected +2 pp effect scale with the variance
    g_f = games_to_detect(float(fact.std(0).mean()), 1600)
    g_o = games_to_detect(float(oat.std(0).mean()), 1600)
    assert 1.5 < g_o / g_f < 3.5


@pytest.mark.parametrize("k,p", [(k, 0) for k in range(1, 7)] + sorted(GENERATORS))
def test_the_alias_structure_matches_the_design(k, p):
    d = design(k, p)
    x = d.matrix
    assert d.runs == 2 ** (k - p) and len({tuple(r) for r in x}) == d.runs
    assert np.all(x[0] == -1)  # run 0 is the parent
    assert np.allclose(x.T @ x, d.runs * np.eye(k))  # main effects orthogonal
    assert len(d.defining) == 2**p - 1

    def col(m):
        return np.prod(x[:, [j for j in range(k) if m >> j & 1]], axis=1)

    for w, s in d.defining:
        assert np.all(col(w) == s)  # I = s · word on every run
    for m in range(1, 2**k):  # every effect's column is ± its aliases' columns, and orthogonal to the rest
        aliased = {a for _, a in d.aliases(m)} | {m}
        for s, a in d.aliases(m):
            assert np.all(col(m) == s * col(a))
        for other in range(1, 2**k):
            if other not in aliased:
                assert col(m) @ col(other) == 0
    if d.resolution >= 4:  # no main effect aliased with a two-way interaction, every chain estimable
        for j in range(k):
            assert all(bin(a).count("1") >= 3 for _, a in d.aliases(1 << j))
        members = [m for first, chain in d.interactions() for m in [first, *(pair for _, pair in chain)]]
        assert sorted(members) == [(i, j) for i in range(k) for j in range(i + 1, k)]
    # the estimated columns are mutually orthogonal (distinct alias classes)
    z = np.stack([np.prod(x[:, list(t)], axis=1) for t in terms_of(d)], 1)
    assert np.allclose(z.T @ z, d.runs * np.eye(z.shape[1]))


def test_default_designs_are_resolution_iv_where_the_runs_allow():
    assert [(design(k).p, design(k).runs) for k in range(3, 9)] == [(0, 8), (1, 8), (1, 16), (2, 16), (3, 16),
                                                                     (4, 16)]  # fmt: skip
    assert all(design(k).resolution >= 4 for k in range(3, 9))
    assert design(4).generators == ("D = ABC",) and design(6).generators == ("E = ABC", "F = BCD")
    assert design(4).label((0, 1)) == "AB = CD" and LETTERS[:4] == "ABCD"


def test_legality_repair_drops_edits_and_reports_them():
    parent = Deck(main=tuple([1] * 3 + [5] * 2 + list(range(10, 45))))
    edits = [Edit(10, 5, "main"), Edit(11, 5, "main"), Edit(12, 60, "main"), Edit(12, 61, "main"),
             Edit(1, 62, "main"), Edit(10, 5, "main"), Edit(13, 1, "main")]  # fmt: skip
    pl = plan(parent, edits, legal=legal)
    # 12 has one copy (the second edit of it has none), the repeat is a duplicate, 13 -> 1 makes four copies of 1
    assert [(x["edit"]["out"], x["edit"]["into"], x["reason"]) for x in pl.dropped] == [
        (12, 61, "no_copy"), (10, 5, "duplicate"), (13, 1, "illegal")]  # fmt: skip
    assert [pl_.edit for pl_ in pl.placed] == [edits[0], edits[1], edits[2], edits[4]]
    assert pl.design.k == 4
    # each edit alone is legal, but A and B together put four copies of card 5 in: B is left out of those runs
    both = [r for r in range(pl.design.runs) if pl.design.matrix[r, 0] > 0 and pl.design.matrix[r, 1] > 0]
    assert both and pl.repairs == [{"run": r, "edit": 1, "letter": "B"} for r in both]
    assert all(legal(d) for d in pl.decks) and all(pl.levels[r, 1] == -1 for r in both)
    # estimation uses the levels played: the A main effect stays estimable, the result reports the repairs
    main = {5: 0.0, 60: 0.05, 62: -0.03}
    r = evaluate_edits(parent, edits, Synthetic(1, main), legal=legal, pairs=300)
    assert r.plan.repairs == pl.repairs and [e.label for e in r.main()] == ["A", "B", "C", "D"]
    assert abs(r.main()[2].effect - 0.05) < 4 * r.main()[2].stderr
    assert r.to_dict()["repairs"] and r.to_dict()["dropped"][2]["reason"] == "illegal"


def test_edits_touch_distinct_copies_in_place():
    placed, dropped = place(PARENT, [Edit(1, 50, "main"), Edit(1, 51, "main"), Edit(900, 901, "extra")])
    assert not dropped and [p.index for p in placed] == [0, 1, 0]
    deck = variant(PARENT, placed)
    assert deck.main[:3] == (50, 51, 1) and deck.main[3:] == PARENT.main[3:] and deck.extra == (901,)


def test_main_effects_are_single_edit_observations_for_the_card_value_model():
    main = {50: 0.08, 51: -0.05, 52: 0.0, 53: 0.02}
    r = evaluate_edits(PARENT, EDITS[:4], Synthetic(3, main), legal=legal, pairs=400)
    obs = r.observations("T")
    assert [(o.into, o.out) for o in obs] == [((50 + j,), (10 + j,)) for j in range(4)]
    assert all(o.stderr > 0 for o in obs)
    model = CardValueModel()
    for o in obs:
        model.add(o)
    assert model.gain([50], [10], "T")[0] > model.gain([51], [11], "T")[0]
    # A alone, the others off: its main effect minus the chains AB, AC, AD (each −1 · −1 → +1 → −1 · +1)
    assert r.predict([]) == 0
    assert r.predict([0]) == pytest.approx(r.main()[0].effect - sum(e.effect for e in r.interactions()))


def test_missing_games_are_dropped_per_pair():
    scores = np.array([[0.0, 1.0, 0.5, np.nan], [1.0, 1.0, np.nan, 0.5], [0.5, 0.0, 1.0, 1.0], [1.0, np.nan, 1.0, 0.5]])
    levels = design(2).matrix
    eff, se, keep = estimate(levels, scores, [(0,), (1,), (0, 1)])
    assert keep == [0, 1, 2] and np.all(np.isfinite(eff)) and np.all(se >= 0)
    with pytest.raises(ValueError):
        estimate(levels, scores[:, :1], [(0,)])
