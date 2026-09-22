"""Calibration / ranking metrics for belief heads (T3.5): hand-computed and closed-form cases."""

import math

import numpy as np
import pytest

from ygorl.eval import calibration as cal

P4 = np.array([0.1, 0.4, 0.6, 0.9])
Y4 = np.array([0, 0, 1, 1])


# --- binary: ECE / MCE / reliability diagram -------------------------------------------------


def test_ece_hand_computed_two_bins():
    # bin [0, .5): conf .25 acc 0; bin [.5, 1]: conf .75 acc 1 -> both gaps .25
    assert cal.ece(P4, Y4, n_bins=2) == pytest.approx(0.25)
    assert cal.mce(P4, Y4, n_bins=2) == pytest.approx(0.25)


def test_ece_hand_computed_weighted_bins():
    p = np.array([0.2, 0.2, 0.8, 0.8, 0.8])
    y = np.array([0, 1, 1, 1, 0])
    # bin [.2,.4): conf .2 acc .5 (n=2); bin [.8,1]: conf .8 acc 2/3 (n=3)
    assert cal.ece(p, y, n_bins=5) == pytest.approx(2 / 5 * 0.3 + 3 / 5 * (0.8 - 2 / 3))
    assert cal.mce(p, y, n_bins=5) == pytest.approx(0.3)


def test_reliability_diagram_fields_and_empty_bins():
    p = np.array([0.1, 0.2, 0.3, 0.9])
    y = np.array([0, 0, 1, 1])
    d = cal.reliability_diagram(p, y, n_bins=4)
    np.testing.assert_array_equal(d.count, [2, 1, 0, 1])
    np.testing.assert_allclose(d.confidence[[0, 1, 3]], [0.15, 0.3, 0.9])
    np.testing.assert_allclose(d.accuracy[[0, 1, 3]], [0.0, 1.0, 1.0])
    assert np.isnan(d.confidence[2]) and np.isnan(d.accuracy[2])
    np.testing.assert_allclose(d.lower, [0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(d.upper, [0.25, 0.5, 0.75, 1.0])
    assert d.ece == pytest.approx(cal.ece(p, y, n_bins=4))
    assert d.mce == pytest.approx(0.7)


def test_bin_edges_include_zero_and_one():
    d = cal.reliability_diagram(np.array([0.0, 0.5, 1.0]), np.array([0, 1, 1]), n_bins=2)
    np.testing.assert_array_equal(d.count, [1, 2])  # 0.5 opens the upper bin, 1.0 closes it


def test_adaptive_bins_have_equal_mass():
    p = np.array([0.1, 0.2, 0.3, 0.9])
    y = np.array([0, 0, 1, 1])
    # uniform: {.1,.2,.3} conf .2 acc 1/3, {.9} conf .9 acc 1 -> .75*(2/15) + .25*.1
    assert cal.ece(p, y, n_bins=2) == pytest.approx(0.125)
    # adaptive: {.1,.2} conf .15 acc 0, {.3,.9} conf .6 acc 1 -> .5*.15 + .5*.4
    assert cal.ece(p, y, n_bins=2, strategy="adaptive") == pytest.approx(0.275)
    d = cal.reliability_diagram(p, y, n_bins=2, strategy="adaptive")
    np.testing.assert_array_equal(d.count, [2, 2])
    np.testing.assert_allclose(d.lower, [0.1, 0.3])
    np.testing.assert_allclose(d.upper, [0.2, 0.9])


def test_adaptive_bins_uneven_and_more_bins_than_samples():
    d = cal.reliability_diagram(np.linspace(0, 1, 5), np.array([0, 0, 1, 1, 1]), n_bins=2, strategy="adaptive")
    np.testing.assert_array_equal(d.count, [3, 2])
    d = cal.reliability_diagram(np.array([0.3, 0.7]), np.array([0, 1]), n_bins=4, strategy="adaptive")
    assert d.count.sum() == 2 and (d.count <= 1).all()


def test_perfectly_calibrated_predictor_has_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.random(100_000)
    y = (rng.random(p.size) < p).astype(int)
    assert cal.ece(p, y, n_bins=10) < 0.01
    assert cal.ece(p, y, n_bins=10, strategy="adaptive") < 0.01


def test_overconfident_predictor_ece_is_the_gap():
    rng = np.random.default_rng(1)
    y = (rng.random(50_000) < 0.6).astype(int)
    assert cal.ece(np.full(y.size, 0.9), y) == pytest.approx(0.3, abs=0.01)


def test_unknown_strategy_and_bad_bin_count():
    with pytest.raises(ValueError, match="strategy"):
        cal.ece(P4, Y4, strategy="quantiles")
    with pytest.raises(ValueError, match="n_bins"):
        cal.ece(P4, Y4, n_bins=0)


# --- binary: Brier / log-loss / accuracy ------------------------------------------------------


def test_brier_hand_computed():
    assert cal.brier_score(P4, Y4) == pytest.approx((0.01 + 0.16 + 0.16 + 0.01) / 4)


def test_log_loss_hand_computed_and_clipped():
    assert cal.log_loss(np.full(7, 0.5), np.array([0, 1, 1, 0, 1, 0, 0])) == pytest.approx(math.log(2))
    assert cal.log_loss(P4, Y4) == pytest.approx(-(2 * math.log(0.9) + 2 * math.log(0.6)) / 4)
    # confidently wrong predictions stay finite: -log(eps)
    assert cal.log_loss(np.array([0.0, 1.0]), np.array([1, 0]), eps=1e-3) == pytest.approx(-math.log(1e-3))
    assert math.isfinite(cal.log_loss(np.array([0.0]), np.array([1])))


def test_accuracy_at_threshold():
    p, y = np.array([0.1, 0.5, 0.7]), np.array([0, 1, 0])
    assert cal.accuracy_at_threshold(p, y) == pytest.approx(2 / 3)  # p >= .5 predicts positive
    assert cal.accuracy_at_threshold(p, y, threshold=0.6) == pytest.approx(1 / 3)


# --- binary: ROC-AUC --------------------------------------------------------------------------


def brute_auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def test_auc_closed_forms():
    assert cal.roc_auc(P4, Y4) == 1.0
    assert cal.roc_auc(1 - P4, Y4) == 0.0
    assert cal.roc_auc(np.full(4, 0.3), Y4) == 0.5
    # ties count half: pairs (.4,.1)=1 (.4,.4)=.5 (.8,.1)=1 (.8,.4)=1
    assert cal.roc_auc(np.array([0.1, 0.4, 0.4, 0.8]), Y4) == pytest.approx(0.875)


def test_auc_matches_brute_force_on_random_data_with_ties():
    rng = np.random.default_rng(2)
    for _ in range(20):
        n = int(rng.integers(2, 60))
        s = np.round(rng.random(n), 1)  # many ties
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        assert cal.roc_auc(s, y) == pytest.approx(brute_auc(s, y))


def test_auc_accepts_any_real_scores():
    assert cal.roc_auc(np.array([-3.0, 10.0]), np.array([0, 1])) == 1.0


def test_auc_single_class_is_nan():
    assert math.isnan(cal.roc_auc(P4, np.ones(4, dtype=int)))


def test_macro_auc_averages_columns_with_both_classes():
    rng = np.random.default_rng(7)
    s = rng.random((40, 3))
    y = rng.integers(0, 2, (40, 3))
    y[:, 2] = 1  # single-class column is skipped
    expected = np.mean([cal.roc_auc(s[:, j], y[:, j]) for j in range(2)])
    assert cal.macro_roc_auc(s, y) == pytest.approx(expected)
    # a per-column constant (a base-rate prior) has no within-column discrimination
    assert cal.macro_roc_auc(np.tile([0.1, 0.7, 0.3], (40, 1)), y) == 0.5
    assert cal.macro_roc_auc(P4, Y4) == 1.0  # 1-D: a single column
    assert math.isnan(cal.macro_roc_auc(s, np.ones_like(y)))


def test_macro_auc_mask_is_per_column():
    rng = np.random.default_rng(8)
    s = rng.random((50, 2, 3))  # leading dims flatten, last axis = columns
    y = rng.integers(0, 2, (50, 2, 3))
    m = rng.random((50, 2, 3)) < 0.7
    s2, y2, m2 = s.reshape(-1, 3), y.reshape(-1, 3), m.reshape(-1, 3)
    expected = np.mean([cal.roc_auc(s2[m2[:, j], j], y2[m2[:, j], j]) for j in range(3)])
    assert cal.macro_roc_auc(s, y, mask=m) == pytest.approx(expected)


# --- multiclass -------------------------------------------------------------------------------

P3 = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
T3 = np.array([1, 1, 0])


def test_top_k_accuracy_hand_computed_with_fractional_ties():
    assert cal.top_k_accuracy(P3, T3, k=1) == pytest.approx(1 / 3)
    # row 2: class 2 beats the target, class 1 ties it for the last top-2 slot -> 1/2
    assert cal.top_k_accuracy(P3, T3, k=2) == pytest.approx((1 + 1 + 0.5) / 3)
    assert cal.top_k_accuracy(P3, T3, k=3) == 1.0
    assert cal.top_k_accuracy(P3, T3, k=10) == 1.0


def test_uniform_predictor_top_k_is_exactly_k_over_classes():
    rng = np.random.default_rng(3)
    t = rng.integers(0, 7, 100)
    u = np.full((100, 7), 1 / 7)
    for k in (1, 3, 7):
        assert cal.top_k_accuracy(u, t, k=k) == pytest.approx(k / 7)
    assert cal.multiclass_ece(u, t) == pytest.approx(0.0, abs=1e-12)


def test_multiclass_ece_on_top_label():
    p = np.array([[0.7, 0.3], [0.6, 0.4], [0.2, 0.8]])
    t = np.array([0, 1, 1])
    # confidences .7 .6 .8 all in the upper bin, correctness 1 0 1
    assert cal.multiclass_ece(p, t, n_bins=2) == pytest.approx(abs(0.7 - 2 / 3))
    assert cal.multiclass_mce(p, t, n_bins=2) == pytest.approx(abs(0.7 - 2 / 3))
    d = cal.top_label_reliability_diagram(p, t, n_bins=2)
    np.testing.assert_array_equal(d.count, [0, 3])


def test_multiclass_calibrated_predictor_has_near_zero_ece():
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.ones(5), size=50_000)
    t = np.minimum((rng.random((p.shape[0], 1)) > p.cumsum(1)).sum(1), 4)
    assert cal.multiclass_ece(p, t, n_bins=10) < 0.01


def test_multiclass_nll_and_brier():
    p = np.array([[0.5, 0.5], [0.25, 0.75]])
    t = np.array([0, 1])
    assert cal.multiclass_nll(p, t) == pytest.approx((math.log(2) + math.log(4 / 3)) / 2)
    assert cal.multiclass_brier(p, t) == pytest.approx((0.5 + 0.125) / 2)
    assert math.isfinite(cal.multiclass_nll(np.array([[1.0, 0.0]]), np.array([1])))


def test_multiclass_leading_dims_are_flattened():
    p = np.stack([P3, P3[::-1]])  # [2, 3, 3]
    t = np.stack([T3, T3[::-1]])
    assert cal.top_k_accuracy(p, t, k=2) == pytest.approx(cal.top_k_accuracy(P3, T3, k=2))
    assert cal.multiclass_nll(p, t) == pytest.approx(cal.multiclass_nll(P3, T3))


# --- masking ----------------------------------------------------------------------------------

BINARY = [cal.ece, cal.mce, cal.brier_score, cal.log_loss, cal.roc_auc, cal.accuracy_at_threshold]
MULTI = [cal.top_k_accuracy, cal.multiclass_ece, cal.multiclass_mce, cal.multiclass_nll, cal.multiclass_brier]


@pytest.mark.parametrize("fn", BINARY)
def test_binary_mask_equals_subset(fn):
    rng = np.random.default_rng(5)
    p = rng.random((30, 4))
    y = rng.integers(0, 2, (30, 4))
    m = rng.random((30, 4)) < 0.6
    assert fn(p, y, mask=m) == pytest.approx(fn(p[m], y[m]))
    # masked-out entries may hold anything, e.g. NaN predictions or -1 padding targets
    p2, y2 = p.copy(), y.copy()
    p2[~m], y2[~m] = np.nan, -1
    assert fn(p2, y2, mask=m) == pytest.approx(fn(p[m], y[m]))
    assert math.isnan(fn(p, y, mask=np.zeros_like(m)))


@pytest.mark.parametrize("fn", MULTI)
def test_multiclass_mask_equals_subset(fn):
    rng = np.random.default_rng(6)
    p = rng.dirichlet(np.ones(5), size=(20, 3))
    t = rng.integers(0, 5, (20, 3))
    m = rng.random((20, 3)) < 0.6
    assert fn(p, t, mask=m) == pytest.approx(fn(p[m], t[m]))
    t2 = t.copy()
    t2[~m] = -1
    assert fn(p, t2, mask=m) == pytest.approx(fn(p[m], t[m]))
    assert math.isnan(fn(p, t, mask=np.zeros_like(m)))


def test_empty_input_is_nan():
    assert math.isnan(cal.ece(np.array([]), np.array([], dtype=int)))
    assert math.isnan(cal.top_k_accuracy(np.zeros((0, 3)), np.zeros(0, dtype=int)))


# --- validation -------------------------------------------------------------------------------


def test_binary_validation_errors():
    with pytest.raises(ValueError, match="shape"):
        cal.ece(P4, Y4[:3])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cal.brier_score(np.array([0.2, 1.2]), np.array([0, 1]))
    with pytest.raises(ValueError, match="finite"):
        cal.log_loss(np.array([0.2, np.nan]), np.array([0, 1]))
    with pytest.raises(ValueError, match="0 or 1"):
        cal.roc_auc(P4, np.array([0, 1, 2, 1]))
    with pytest.raises(ValueError, match="mask.*shape"):
        cal.ece(P4, Y4, mask=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="bool"):
        cal.ece(P4, Y4, mask=np.ones(4))


def test_multiclass_validation_errors():
    with pytest.raises(ValueError, match="shape"):
        cal.top_k_accuracy(P3, np.array([0, 1]))
    with pytest.raises(ValueError, match="sum to 1"):
        cal.multiclass_nll(np.array([[0.5, 0.2]]), np.array([0]))
    with pytest.raises(ValueError, match="range"):
        cal.multiclass_brier(P3, np.array([0, 1, 3]))
    with pytest.raises(ValueError, match="integer"):
        cal.multiclass_ece(P3, np.array([0.0, 1.5, 2.0]))
    with pytest.raises(ValueError, match="k"):
        cal.top_k_accuracy(P3, T3, k=0)
