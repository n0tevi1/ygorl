"""Per-head belief report, baseline predictors and synthetic data (T3.5)."""

import math

import numpy as np
import pytest

from ygorl.eval import beliefs as B
from ygorl.eval import calibration as cal

def tiny_batch():
    """N=3 samples, K=3 deck types, C=2 candidate cards, S=2 set zones, R=2 role bits."""
    return B.BeliefBatch(
        deck_type=B.Head(np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]), np.array([0, 2, 2])),
        remaining_copies=B.Head(
            np.array(
                [
                    [[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]],
                    [[0.25, 0.25, 0.25, 0.25], [0.0, 0.0, 1.0, 0.0]],
                    [[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]],
                ]
            ),
            np.array([[3, 0], [1, 2], [0, 0]]),
            mask=np.array([[True, True], [True, False], [True, True]]),  # public copies are excluded
        ),
        hand=B.Head(np.array([[0.9, 0.2], [0.6, 0.4], [0.1, 0.7]]), np.array([[1, 0], [0, 1], [0, 1]])),
        hand_roles=B.Head(np.array([[0.8, 0.1], [0.3, 0.6], [0.5, 0.5]]), np.array([[1, 0], [0, 0], [1, 1]])),
        set_cards=B.Head(
            np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.2, 0.8], [0.3, 0.7]], [[0.6, 0.4], [1.0, 0.0]]]),
            np.array([[0, -1], [1, 1], [1, -1]]),
            mask=np.array([[True, False], [True, True], [True, False]]),  # empty zones are masked
        ),
        responded=B.Head(np.array([0.8, 0.3, 0.6]), np.array([1, 0, 0])),
    )


def test_report_matches_the_underlying_metrics():
    b = tiny_batch()
    r = B.evaluate_beliefs(b, n_bins=5)
    d, rc, h, s, resp = b.deck_type, b.remaining_copies, b.hand, b.set_cards, b.responded
    assert r["deck_type/top1"] == pytest.approx(cal.top_k_accuracy(d.probs, d.targets, k=1))
    assert r["deck_type/top1"] == pytest.approx(2 / 3)
    assert r["deck_type/top3"] == 1.0
    assert r["deck_type/ece"] == pytest.approx(cal.multiclass_ece(d.probs, d.targets, n_bins=5))
    assert r["remaining_copies/accuracy"] == pytest.approx(cal.top_k_accuracy(rc.probs, rc.targets, mask=rc.mask))
    assert r["remaining_copies/n"] == 5
    assert r["remaining_copies/ece"] == pytest.approx(cal.multiclass_ece(rc.probs, rc.targets, mask=rc.mask, n_bins=5))
    assert r["hand/auc"] == pytest.approx(cal.roc_auc(h.probs, h.targets))
    assert r["hand/auc_macro"] == pytest.approx(cal.macro_roc_auc(h.probs, h.targets))
    assert r["hand_roles/auc_macro"] == pytest.approx(cal.macro_roc_auc(b.hand_roles.probs, b.hand_roles.targets))
    assert "responded/auc_macro" not in r  # scalar head: no columns
    assert r["hand/ece"] == pytest.approx(cal.ece(h.probs, h.targets, n_bins=5))
    assert r["hand/brier"] == pytest.approx(cal.brier_score(h.probs, h.targets))
    assert r["set_cards/top3"] == 1.0  # only 2 candidate classes
    assert r["set_cards/top1"] == pytest.approx(cal.top_k_accuracy(s.probs, s.targets, mask=s.mask))
    assert r["set_cards/n"] == 4
    assert r["responded/auc"] == pytest.approx(1.0)
    assert r["responded/ece"] == pytest.approx(cal.ece(resp.probs, resp.targets, n_bins=5))
    assert "hand_roles/auc" in r
    assert all(isinstance(v, float) for v in r.values())


def test_remaining_copies_accuracy_hand_computed():
    # argmax per included card: 3 (t=3 ok), 0 (t=0 ok), tie over all 4 (t=1: 1/4), 0 (t=0 ok), 3 (t=0 miss)
    r = B.evaluate_beliefs(tiny_batch())
    assert r["remaining_copies/accuracy"] == pytest.approx((1 + 1 + 0.25 + 1 + 0) / 5)


def test_missing_heads_are_skipped():
    b = tiny_batch()
    r = B.evaluate_beliefs(B.BeliefBatch(hand=b.hand))
    assert r and all(k.startswith("hand/") for k in r)
    assert B.evaluate_beliefs(B.BeliefBatch()) == {}


def test_batch_size_property():
    assert tiny_batch().n == 3
    assert B.BeliefBatch().n == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"deck_type": B.Head(np.full(3, 0.5), np.zeros(3, dtype=int))}, r"deck_type.*\[N, K\]"),
        ({"deck_type": B.Head(np.full((3, 2), 0.5), np.zeros((3, 1), dtype=int))}, r"deck_type.*targets"),
        ({"remaining_copies": B.Head(np.full((3, 2, 3), 1 / 3), np.zeros((3, 2), dtype=int))}, r"remaining_copies.*4"),
        ({"hand": B.Head(np.full((3, 2), 0.5), np.zeros(3, dtype=int))}, r"hand.*targets"),
        ({"set_cards": B.Head(np.full((3, 2, 4), 0.25), np.zeros((3, 4), dtype=int))}, r"set_cards.*targets"),
        ({"responded": B.Head(np.full((3, 1), 0.5), np.zeros((3, 1), dtype=int))}, r"responded.*\[N\]"),
        ({"hand": B.Head(np.full((3, 2), 0.5), np.zeros((3, 2), dtype=int), mask=np.ones(3, dtype=bool))}, r"hand.*mask"),
    ],
)
def test_shape_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        B.BeliefBatch(**kwargs)


def test_heads_must_share_batch_size():
    b = tiny_batch()
    with pytest.raises(ValueError, match="N"):
        B.BeliefBatch(deck_type=b.deck_type, responded=B.Head(np.full(2, 0.5), np.zeros(2, dtype=int)))


def test_value_errors_name_the_head():
    b = tiny_batch()
    bad = B.Head(b.hand.probs, np.full_like(b.hand.targets, 2))
    with pytest.raises(ValueError, match="hand"):
        B.evaluate_beliefs(B.BeliefBatch(hand=bad))


# --- baselines --------------------------------------------------------------------------------


def test_uniform_predictor_closed_forms():
    b = B.uniform_predictor(tiny_batch())
    np.testing.assert_allclose(b.deck_type.probs, 1 / 3)
    np.testing.assert_allclose(b.remaining_copies.probs, 0.25)
    np.testing.assert_allclose(b.hand.probs, 0.5)
    np.testing.assert_allclose(b.set_cards.probs, 0.5)
    np.testing.assert_array_equal(b.set_cards.mask, tiny_batch().set_cards.mask)  # targets/masks kept
    r = B.evaluate_beliefs(b)
    assert r["deck_type/top1"] == pytest.approx(1 / 3)
    assert r["remaining_copies/accuracy"] == pytest.approx(0.25)
    assert r["hand/auc"] == 0.5 and r["responded/auc"] == 0.5
    assert r["deck_type/ece"] == pytest.approx(0.0, abs=1e-12)


def test_prior_predictor_uses_masked_frequencies():
    b = tiny_batch()
    p = B.prior_predictor(b)
    np.testing.assert_allclose(p.deck_type.probs, np.tile([1 / 3, 0, 2 / 3], (3, 1)))
    np.testing.assert_allclose(p.hand.probs, np.tile([1 / 3, 2 / 3], (3, 1)))  # per-card base rates
    np.testing.assert_allclose(p.responded.probs, 1 / 3)
    # per-card copy distribution from included entries only: card 1 -> t=0 twice (sample 1 masked)
    np.testing.assert_allclose(p.remaining_copies.probs[:, 1], np.tile([1, 0, 0, 0], (3, 1)))
    np.testing.assert_allclose(p.set_cards.probs, np.tile([1 / 4, 3 / 4], (3, 2, 1)))


def test_prior_predictor_can_be_fitted_on_another_batch():
    b = tiny_batch()
    fit = B.BeliefBatch(responded=B.Head(np.full(4, 0.5), np.array([1, 1, 1, 0])))
    p = B.prior_predictor(B.BeliefBatch(responded=b.responded), fit=fit)
    np.testing.assert_allclose(p.responded.probs, 0.75)


def test_prior_smoothing_keeps_unseen_classes_nonzero():
    p = B.prior_predictor(tiny_batch(), smoothing=1.0)
    assert (p.deck_type.probs > 0).all()
    np.testing.assert_allclose(p.deck_type.probs.sum(-1), 1.0)


def test_synthetic_batch_shapes_and_determinism():
    b = B.synthetic_batch(50, n_deck_types=6, n_cards=10, n_set_zones=3, n_roles=4, seed=7)
    assert b.deck_type.probs.shape == (50, 6)
    assert b.remaining_copies.probs.shape == (50, 10, 4)
    assert b.hand.probs.shape == (50, 10) and b.hand_roles.probs.shape == (50, 4)
    assert b.set_cards.probs.shape == (50, 3, 10)
    assert b.responded.probs.shape == (50,)
    assert (~b.hand.mask).any() and (~b.set_cards.mask).any()  # public cards / empty zones exist
    again = B.synthetic_batch(50, n_deck_types=6, n_cards=10, n_set_zones=3, n_roles=4, seed=7)
    np.testing.assert_array_equal(b.hand.targets, again.hand.targets)


def test_baseline_report_orders_predictors_sensibly():
    rep = B.baseline_report(n=5000, seed=0)
    assert set(rep) == {"uniform", "random", "prior", "oracle"}
    u, rnd, pri, orc = rep["uniform"], rep["random"], rep["prior"], rep["oracle"]
    k = B.SYNTHETIC_DEFAULTS["n_deck_types"]
    assert u["deck_type/top1"] == pytest.approx(1 / k)
    assert rnd["deck_type/top1"] == pytest.approx(1 / k, abs=0.03)
    for head in ("hand", "responded"):
        assert u[f"{head}/auc"] == 0.5
        assert rnd[f"{head}/auc"] == pytest.approx(0.5, abs=0.05)
        assert orc[f"{head}/auc"] > pri[f"{head}/auc"]
    # pooled hand AUC credits per-card base rates; the per-card AUC does not
    assert pri["hand/auc"] > 0.6 and pri["hand/auc_macro"] == 0.5
    assert orc["hand/auc_macro"] > 0.7
    assert rnd["hand/ece"] > 0.15  # uninformed but confident: badly calibrated
    for key in ("hand/ece", "responded/ece", "deck_type/ece", "remaining_copies/ece"):
        assert pri[key] < 0.05 and orc[key] < 0.05  # frequencies and true posteriors are calibrated
    assert orc["deck_type/top1"] > pri["deck_type/top1"] >= u["deck_type/top1"]
    assert orc["set_cards/top3"] > u["set_cards/top3"]
    assert orc["hand/brier"] < pri["hand/brier"] < u["hand/brier"] < rnd["hand/brier"]


def test_format_report_is_a_markdown_table():
    text = B.format_baselines(B.baseline_report(n=200, seed=1))
    lines = text.splitlines()
    assert lines[0].startswith("| metric |") and "uniform" in lines[0] and "oracle" in lines[0]
    assert any(line.startswith("| hand/auc |") for line in lines)
    assert set(lines[1]) <= set("|-: ")
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert len(cells) == 5
        assert all(math.isfinite(float(c)) for c in cells[1:])
