"""Deck corpus selection (ygorl.data.corpus) and the deck-list source label."""

from ygorl.data import masterduelmeta as mdm
from ygorl.data.corpus import select_lists


def rec(t, created, main, extra=(), unmapped=()):
    return mdm.DeckRecord(t, f"https://x/{t}/{created}", created, 10.0, tuple(main), tuple(extra), tuple(unmapped))


BASE = [1] * 3 + [2] * 3 + list(range(10, 44))  # 40 cards


def swap(n, start=100):
    """BASE with its last ``n`` cards replaced by new ones."""
    return BASE[: 40 - n] + list(range(start, start + n))


def test_medoid_then_farthest_lists_skipping_duplicates_near_duplicates_and_illegal():
    records = [
        rec("A", "2026-01-01", BASE),
        rec("A", "2026-01-02", BASE),  # duplicate of the newest-first list set: dropped
        rec("A", "2026-01-03", swap(1)),  # 2 cards from BASE
        rec("A", "2026-01-04", swap(20, 200)),  # far: 40 cards from BASE
        rec("A", "2026-01-05", swap(2, 300)),  # near BASE
        rec("A", "2026-01-06", swap(30, 400)),  # far from both
        rec("A", "2026-01-07", [99] * 40),  # illegal (stand-in rule below)
        rec("B", "2026-02-01", BASE, unmapped=("Mystery",)),  # unmapped only: type left out
    ]

    def problems(r):
        return ["over_limit"] if r.counts()[99] > 3 else []

    out = select_lists(records, problems, per_type=4, min_distance=8)
    assert {c.type for c in out} == {"A"}
    roles = [(c.role, c.record.created) for c in out]
    assert roles[0][0] == "medoid" and roles[0][1] in {"2026-01-02", "2026-01-03", "2026-01-05"}  # the BASE cluster
    far = {c.record.created for c in out if c.role == "diverse"}
    assert far == {"2026-01-04", "2026-01-06"}  # the near-duplicates of BASE are skipped (< 8 cards apart)
    assert all(c.distance >= 8 for c in out if c.role == "diverse")
    assert all(c.legal_lists == 5 for c in out)  # 6 distinct lists minus the illegal one
    assert len(select_lists(records, problems, per_type=1)) == 1


def test_deck_source_is_the_ranked_or_tournament_name():
    assert mdm.deck_source({"rankedType": {"name": "Master I"}}) == "Master I"
    assert mdm.deck_source({"tournamentType": {"name": "Dice Rally"}}) == "Dice Rally"
    assert mdm.deck_source({}) == ""
