"""Signal library for deck evolution (#109): card values from paired evaluations, signal calibration, children."""

from collections import Counter

import numpy as np
import pytest

from ygorl.build.signals import (
    Calibration,
    CardValueModel,
    Observation,
    combine_prior,
    informed_children,
    spearman,
)
from ygorl.cards.ydk import Deck


def test_without_data_a_value_is_its_prior_and_with_data_it_approaches_the_evaluations():
    m = CardValueModel(card_scale=0.03, type_scale=0.02, prior={1: 0.01})
    assert m.value(1, "A") == pytest.approx((0.01, (0.03**2 + 0.02**2) ** 0.5))
    assert m.value(2, "A")[0] == 0.0
    rng = np.random.default_rng(0)
    truth = {10: 0.04, 11: 0.0, 12: -0.02, 13: 0.01}
    for _ in range(400):
        into, out = rng.choice(list(truth), 2, replace=False)
        d = truth[into] - truth[out]
        m.add(Observation("A", (int(into),), (int(out),), d + rng.normal(0, 0.01), 0.01))
    got = {c: m.value(c, "A")[0] for c in truth}
    # values are identified up to a constant: compare differences
    for c in truth:
        assert got[c] - got[11] == pytest.approx(truth[c] - truth[11], abs=0.004)
    assert m.gain([10], [11], "A")[1] < 0.003  # differences are far tighter than the prior
    assert m.value(10, "A")[1] > 5 * m.gain([10], [11], "A")[1]  # the common level stays near the prior
    draws = [m.sample([10, 11], "A", np.random.default_rng(s)) for s in range(200)]
    assert np.mean([d[10] - d[11] for d in draws]) == pytest.approx(0.04, abs=0.004)
    assert np.std([d[10] - d[11] for d in draws]) < 0.005  # joint draws keep the correlation


def test_a_card_measured_in_one_type_informs_another_type_less_than_its_own():
    m = CardValueModel(card_scale=0.03, type_scale=0.02)
    for _ in range(50):
        m.add(Observation("A", (5,), (6,), 0.05, 0.01))
    in_a, in_b = m.gain([5], [6], "A")[0], m.gain([5], [6], "B")[0]
    assert in_a == pytest.approx(0.05, abs=0.005)
    assert 0 < in_b < in_a  # shared through the card effect, shrunk
    with pytest.raises(ValueError):
        m.add(Observation("A", (5,), (6,), 0.0, 0.0))


def test_calibration_switches_off_an_unrelated_signal_and_weights_a_related_one():
    rng = np.random.default_rng(1)
    cal = Calibration({"opening": 0.0, "prior_good": 0.3}, min_pairs=20)
    assert cal.weight("prior_good") == 0.3 and cal.weight("opening") == 0.0 and cal.weight("unknown") == 0.0
    for _ in range(200):
        g = rng.normal()
        cal.record("related", g + rng.normal(0, 0.5), g)
        cal.record("prior_good", rng.normal(), g)  # its default no longer applies once there is evidence
    assert cal.weight("related") > 0.7 and cal.enabled("related")
    assert cal.weight("prior_good") == 0.0 and not cal.enabled("prior_good")
    assert set(cal.report()) == {"opening", "prior_good", "related"}


def test_the_prior_uses_only_calibrated_signals():
    cal = Calibration({"good": 0.5, "bad": 0.0})
    prior = combine_prior({"good": {1: 1.0, 2: 0.0, 3: -1.0}, "bad": {1: -5.0, 2: 5.0}}, cal, scale=0.01)
    assert prior[1] > prior[2] > prior[3]
    assert prior[1] == pytest.approx(0.5 * 0.01 * 1.2247, rel=1e-3)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0) and spearman([1, 1], [2, 3]) == 0.0


def deck(main, extra=()):
    return Deck(main=tuple(main), extra=tuple(extra), name="d")


def legal(d):
    c = Counter((*d.main, *d.extra))
    return max(c.values()) <= 3 and len(d.main) == 40


def is_extra(pw):
    return pw >= 900


def test_informed_children_are_legal_bundled_same_direction_and_explore_as_configured():
    base = deck([1] * 3 + [2] * 3 + [3] * 3 + list(range(10, 41)), [900, 901])
    pool = [100, 101, 102, 1, 902]
    m = CardValueModel(prior={100: 0.05, 101: 0.04, 102: 0.03, 902: 0.02, 3: -0.05, 2: -0.04, 10: -0.03})
    kids = informed_children(base, "T", m, pool, legal=legal, is_extra=is_extra, rng=np.random.default_rng(0),
                             informed=5, explore=2, max_bundle=3, protected=[3])  # fmt: skip
    informed = [k for k in kids if k.kind == "informed"]
    assert len(informed) == 5 and sum(k.kind == "explore" for k in kids) == 2
    assert len({(tuple(sorted(k.deck.main)), tuple(sorted(k.deck.extra))) for k in kids}) == len(kids)
    for k in kids:
        assert legal(k.deck)
        assert all(e.out != 3 for e in k.edits)  # protected
        assert all(is_extra(e.into) == (e.section == "extra") for e in k.edits)
    assert max(len(k.edits) for k in informed) > 1
    # with a confident model, informed edits go from low-valued cards to high-valued ones
    for k in informed:
        assert k.predicted > 0


def test_win_conditions_and_search_targets_are_protected():
    from pathlib import Path

    from ygorl.build.signals import protected_cards
    from ygorl.build.synergy_graph import Edge, SynergyGraph
    from ygorl.cards.ydk import load_ydk

    exodia = load_ydk(Path(__file__).parents[1] / "environments/md-2026-09/artifacts/decks/exodia.ydk")
    pieces = {33396948, 7902349, 70903634, 44519536, 8124921}  # Exodia, its arms and legs
    assert pieces <= protected_cards(exodia)
    base = deck([1] * 3 + [2] * 3 + list(range(10, 44)))
    graph = SynergyGraph({1: {}, 2: {}, 10: {}}, [Edge(1, 2, "search", 0x1, 5, "filter"),
                                                   Edge(1, 10, "search", 0x1, 500, "category")])  # fmt: skip
    got = protected_cards(base, graph, scripts_dir=Path("/nonexistent"))
    assert got == {2}  # 10 is behind a query that matches 500 cards: not a real search target
