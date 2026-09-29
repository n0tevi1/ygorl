"""Crossover of archive elites as a child generator (ygorl.build.crossover); its hook in the evolution step is tested
in test_evolve.py."""

from collections import Counter

import numpy as np

from ygorl.build.archive import DeckArchive
from ygorl.build.crossover import cell_distance, cross_decks, crossover_children, swaps
from ygorl.cards.ydk import Deck

GRID = {"hand_traps": (5, (0.0, 15.0)), "brick_rate": (5, (0.0, 1.0))}
BASE = Deck(main=tuple([1] * 3 + [2] * 3 + list(range(10, 44))), extra=(900, 901), name="base")
# a second style: 12 cards of the base replaced (two of them by 3-ofs), one extra monster swapped
OTHER = Deck(main=tuple([1] * 3 + [2] * 3 + list(range(10, 32)) + [60] * 3 + [61] * 3 + list(range(62, 68))),
             extra=(900, 902), name="other")  # fmt: skip


def legal(deck):
    c = deck.counts()
    return len(deck.main) == 40 and max(c.values()) <= 3


def distance(a: Deck, b: Deck) -> int:
    """Copies that differ (count-based, both sections)."""
    x, y = Counter((*a.main, *(("x", c) for c in a.extra))), Counter((*b.main, *(("x", c) for c in b.extra)))
    return sum(((x - y) + (y - x)).values())


def archive(*entries):
    a = DeckArchive(GRID)
    for i, (deck, desc) in enumerate(entries):
        assert a.add(f"e{i}", deck, 0.5, desc).admitted
    return a


def test_children_are_legal_differ_from_both_parents_and_stay_closer_to_the_reference():
    rng = np.random.default_rng(0)
    for _ in range(300):
        edits, child = cross_decks(BASE, OTHER, rng, legal=legal)
        assert legal(child) and len(child.main) == len(BASE.main) and len(child.extra) == len(BASE.extra)
        assert child.counts() != BASE.counts() and child.counts() != OTHER.counts()
        assert 1 <= len(edits) <= swaps(BASE, OTHER) // 2
        assert distance(child, BASE) <= distance(child, OTHER)
        # every card put in comes from the mate, every card taken out is one the mate lacks
        assert all(e.into in (*OTHER.main, *OTHER.extra) and e.out not in (*OTHER.main, *OTHER.extra) for e in edits)
        # in-place edits: cards the two parents share keep their positions (common random numbers)
        assert all(c == b for c, b in zip(child.main, BASE.main, strict=True) if b in OTHER.main)


def test_repair_drops_swaps_that_would_break_legality_and_protected_cards_stay():
    no_60 = lambda d: legal(d) and 60 not in d.main  # noqa: E731
    rng = np.random.default_rng(1)
    for _ in range(200):
        made = cross_decks(BASE, OTHER, rng, legal=no_60, protected={10, 20})
        if made is None:  # every drawn swap brought a 60
            continue
        edits, child = made
        assert no_60(child) and {10, 20} <= set(child.main)
        assert all(e.into != 60 and e.out not in (10, 20) for e in edits)
    # a mate one swap away cannot be crossed (the child would be a mutation, or the mate itself)
    near = Deck(main=(*BASE.main[:-1], 70), extra=BASE.extra)
    assert cross_decks(BASE, near, rng, legal=legal) is None


def test_mates_favour_far_cells():
    near = Deck(main=(*BASE.main[:-4], 70, 71, 72, 73), extra=BASE.extra)
    a = archive((BASE, {"hand_traps": 0.0, "brick_rate": 0.0}), (near, {"hand_traps": 3.0, "brick_rate": 0.0}),
                (OTHER, {"hand_traps": 14.0, "brick_rate": 0.9}))  # fmt: skip
    assert cell_distance(a.grid, {"hand_traps": 0.0, "brick_rate": 0.0}, {"hand_traps": 14.0, "brick_rate": 0.9}) == 8
    rng = np.random.default_rng(2)
    picks = Counter(m.id for _ in range(400) for _, m in crossover_children(BASE, a, descriptors={}, legal=legal,
                                                                             rng=rng, n=1))  # fmt: skip
    assert set(picks) == {"e1", "e2"}  # never the parent's own list
    assert picks["e2"] > 4 * picks["e1"]  # weights 8 : 1
    # the parent is not an elite: its deck-list descriptors place it (here next to the far elite)
    outsider = Deck(main=(*BASE.main[:-2], 80, 81), extra=BASE.extra)
    picks = Counter(m.id for _ in range(400) for _, m in crossover_children(
        outsider, a, descriptors={"hand_traps": 14.0, "brick_rate": 0.9}, legal=legal, rng=rng, n=1))  # fmt: skip
    assert picks["e0"] > picks["e2"] and picks["e1"] > picks["e2"]


def test_children_are_distinct_and_record_their_mate():
    a = archive((BASE, {"hand_traps": 0.0, "brick_rate": 0.0}), (OTHER, {"hand_traps": 14.0, "brick_rate": 0.9}))
    out = crossover_children(BASE, a, descriptors={}, legal=legal, rng=np.random.default_rng(3), n=4,
                             exclude=[BASE])  # fmt: skip
    assert len(out) == 4 and len({c.deck.counts().__repr__() for c, _ in out}) == 4
    for c, m in out:
        assert c.kind == "crossover" and c.edits and m.id == "e1" and m.deck == OTHER and m.distance == 8


def test_fewer_than_two_elites_propose_nothing():
    rng = np.random.default_rng(4)
    assert crossover_children(BASE, DeckArchive(GRID), descriptors={}, legal=legal, rng=rng, n=3) == []
    one = archive((OTHER, {"hand_traps": 14.0, "brick_rate": 0.9}))
    assert crossover_children(BASE, one, descriptors={}, legal=legal, rng=rng, n=3) == []
    # two elites, both with the parent's list: no mate
    two = archive((BASE, {"hand_traps": 0.0, "brick_rate": 0.0}), (BASE, {"hand_traps": 14.0, "brick_rate": 0.9}))
    assert crossover_children(BASE, two, descriptors={}, legal=legal, rng=rng, n=3) == []
