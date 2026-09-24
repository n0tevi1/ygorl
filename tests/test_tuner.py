"""Deck tuning search (ygorl.build.tuner) with a stand-in evaluator (no games)."""

import numpy as np

from ygorl.build.tuner import Edit, apply, neighbors, paired_difference, successive_halving, tech_pool
from ygorl.cards.ydk import Deck

EXTRA = {900, 901}
BASE = Deck(main=tuple([1] * 3 + [2] * 3 + list(range(10, 44))), extra=(900,), name="base")


def legal(deck):  # stand-in: at most 3 copies, 40 main cards
    c = deck.counts()
    return len(deck.main) == 40 and max(c.values()) <= 3


def test_apply_and_neighbors_keep_sections_and_legality():
    d = apply(BASE, Edit(1, 50, "main"))
    assert d.counts()[1] == 2 and d.counts()[50] == 1 and len(d.main) == 40 and d.extra == BASE.extra
    edits = neighbors(BASE, [50, 2, 901], lambda pw: pw in EXTRA, legal)
    kinds = {(e.section, e.into) for e, _ in edits}
    assert ("extra", 901) in kinds and ("main", 50) in kinds and ("main", 901) not in kinds
    assert all(e.into != 2 or e.out != 2 for e, _ in edits)
    assert not any(e.into == 2 for e, _ in edits)  # 2 is already at 3 copies: never legal to add
    assert all(legal(d) for _, d in edits)
    some = neighbors(BASE, [50, 51], lambda pw: pw in EXTRA, legal, limit=5, rng=np.random.default_rng(0))
    assert len(some) == 5


def test_tech_pool_takes_same_type_cards_then_common_meta_cards():
    same = [Deck(main=(1, 60, 61), extra=(900,))]
    meta = [Deck(main=(70, 71)), Deck(main=(70, 72)), Deck(main=(70,))]
    pool = tech_pool(BASE, same, meta, meta_top=2)
    assert pool[:4] == [1, 60, 61, 900] and 70 in pool and len(pool) == 6


def test_paired_difference():
    m, lo, hi = paired_difference([1, 1, 0.5, 1], [0.5, 0.5, 0.5, 0.5])
    assert m == 0.375 and lo < m < hi
    assert paired_difference([1.0], [0.0])[1] == -np.inf


class FakeEvaluator:
    """Deck with card 50 wins 0.8, card 51 wins 0.6, anything else 0.5 (plus pair noise shared by all decks)."""

    def __init__(self):
        self.games, self.seconds, self.calls = 0, 0.0, []

    def scores(self, decks, pairs):
        self.calls.append((len(decks), pairs.start, pairs.stop))
        out = np.zeros((len(decks), len(pairs)))
        for i, d in enumerate(decks):
            p = 0.8 if 50 in d.main else 0.6 if 51 in d.main else 0.5
            for j, k in enumerate(pairs):
                u = np.random.default_rng([k, hash(d.main) % 1000]).random()
                out[i, j] = 1.0 if u < p else 0.0
        self.games += 2 * out.size
        return out


def test_successive_halving_finds_the_better_card_and_doubles_pairs():
    edits = neighbors(BASE, [50, 51, 52, 53, 54, 55, 56, 57], lambda pw: pw in EXTRA, legal)
    edits = [e for e in edits if e[0].out == 10]  # 8 swaps of card 10
    ev = FakeEvaluator()
    base, finals = successive_halving(BASE, edits, ev, first_pairs=100, finalists=2)
    assert finals[0].edit.into == 50 and len(finals) == 2
    assert len(base.scores) == len(finals[0].scores) == 400  # 100 -> 200 -> 400 pairs (8 -> 4 -> 2 candidates)
    assert [c[0] for c in ev.calls] == [9, 5, 3] and ev.calls[1][1:] == (100, 200)
    mean, lo, _ = paired_difference(finals[0].scores, base.scores)
    assert mean > 0.15 and lo > 0
