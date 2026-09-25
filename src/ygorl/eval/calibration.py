"""Calibration and ranking metrics for belief heads (T3.5). Pure numpy.

Binary metrics take ``probs`` (probability of the positive class) and ``labels`` (0/1) of the same
shape; multiclass metrics take ``probs`` of shape ``[..., K]`` (rows sum to 1) and integer
``targets`` of shape ``[...]``. Leading dimensions are flattened, so a hand head ``[N, C]`` or a
set-card head ``[N, S, C]`` can be passed directly.

Every metric accepts an optional boolean ``mask`` shaped like ``labels`` / ``targets``: ``True``
means "evaluate this entry", ``False`` excludes it (already-public cards, empty zones, padding).
Excluded entries are not validated, so they may hold NaN predictions or ``-1`` targets. A metric
over zero included entries is NaN, as is ROC-AUC when only one class is present.

Definitions and formulas: docs/belief-eval.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-7
"""Default clipping for log-loss / NLL: a confidently wrong prediction costs -log(EPS) ≈ 16.1."""

STRATEGIES = ("uniform", "adaptive")

_SUM_ATOL = 1e-3


# --- input handling ---------------------------------------------------------------------------


def _mask(mask, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype != bool:
        raise ValueError(f"mask must be a bool array (True = evaluate), got dtype {m.dtype}; use .astype(bool)")
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} does not match labels/targets shape {shape}")
    return m


def _select(a: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    return a.reshape(-1) if m is None else a[m]


def _binary(probs, labels, mask, *, scores: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Flattened, masked, validated (probs, labels as float 0/1)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels)
    if p.shape != y.shape:
        raise ValueError(f"probs shape {p.shape} does not match labels shape {y.shape}")
    m = _mask(mask, y.shape)
    p, y = _select(p, m), _select(y, m)
    what = "scores" if scores else "probs"
    if not np.isfinite(p).all():
        raise ValueError(f"{what} must be finite")
    if not scores and ((p < 0) | (p > 1)).any():
        raise ValueError(f"probs must lie in [0, 1] (min {p.min():.4g}, max {p.max():.4g})")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return p, y.astype(float)


def _multiclass(probs, targets, mask) -> tuple[np.ndarray, np.ndarray]:
    """Flattened, masked, validated (probs [M, K], targets int [M])."""
    p = np.asarray(probs, dtype=float)
    t = np.asarray(targets)
    if p.ndim < 1 or p.shape[:-1] != t.shape:
        raise ValueError(f"probs shape {p.shape} must be targets shape {t.shape} + (K,)")
    k = p.shape[-1]
    m = _mask(mask, t.shape)
    p = p.reshape(-1, k) if m is None else p[m]
    t = _select(t, m)
    if not np.isfinite(p).all():
        raise ValueError("probs must be finite")
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"probs must lie in [0, 1] (min {p.min():.4g}, max {p.max():.4g})")
    if p.size and not np.allclose(p.sum(-1), 1.0, rtol=0, atol=_SUM_ATOL):
        raise ValueError(
            f"probs rows must sum to 1 (atol {_SUM_ATOL}); max deviation {np.abs(p.sum(-1) - 1).max():.4g}"
        )
    if t.size and (not np.isfinite(t.astype(float)).all() or (t != np.round(t)).any()):
        raise ValueError("targets must be integer class indices")
    t = t.astype(np.int64)
    if t.size and (t.min() < 0 or t.max() >= k):
        raise ValueError(f"targets out of range [0, {k}): min {t.min()}, max {t.max()}")
    return p, t


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


# --- reliability diagram / ECE ----------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityDiagram:
    """Per-bin statistics; empty bins have count 0 and NaN confidence/accuracy.

    ``lower`` / ``upper`` are the bin edges for ``uniform`` bins and the smallest / largest
    confidence inside the bin for ``adaptive`` bins. For binary metrics ``accuracy`` is the
    observed positive rate; for top-label (multiclass) metrics it is the top-1 accuracy.
    """

    lower: np.ndarray
    upper: np.ndarray
    confidence: np.ndarray
    accuracy: np.ndarray
    count: np.ndarray

    @property
    def ece(self) -> float:
        """Σ_b (n_b / n) · |acc_b − conf_b|."""
        total = self.count.sum()
        if total == 0:
            return float("nan")
        ne = self.count > 0
        return float(np.sum(self.count[ne] / total * np.abs(self.accuracy[ne] - self.confidence[ne])))

    @property
    def mce(self) -> float:
        """max_b |acc_b − conf_b| over non-empty bins."""
        ne = self.count > 0
        return float(np.abs(self.accuracy[ne] - self.confidence[ne]).max()) if ne.any() else float("nan")


def _reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int, strategy: str) -> ReliabilityDiagram:
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strategy == "uniform":
        ids = np.clip(np.floor(conf * n_bins).astype(np.int64), 0, n_bins - 1)
        lower = np.arange(n_bins) / n_bins
        upper = np.arange(1, n_bins + 1) / n_bins
    else:  # equal-mass bins over the sorted confidences
        order = np.argsort(conf, kind="stable")
        ids = np.empty(conf.size, dtype=np.int64)
        lower = np.full(n_bins, np.nan)
        upper = np.full(n_bins, np.nan)
        for b, chunk in enumerate(np.array_split(order, n_bins)):
            ids[chunk] = b
            if chunk.size:
                lower[b], upper[b] = conf[chunk[0]], conf[chunk[-1]]
    count = np.bincount(ids, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.bincount(ids, weights=conf, minlength=n_bins) / count
        accuracy = np.bincount(ids, weights=correct, minlength=n_bins) / count
    return ReliabilityDiagram(lower, upper, confidence, accuracy, count)


def reliability_diagram(probs, labels, *, n_bins: int = 15, strategy: str = "uniform", mask=None) -> ReliabilityDiagram:
    """Binary reliability diagram: bins over P(positive), positive rate per bin.

    ``strategy="uniform"``: equal-width bins [i/B, (i+1)/B), the last bin closed at 1.
    ``strategy="adaptive"``: equal-mass bins (sorted predictions split into B near-equal groups).
    """
    p, y = _binary(probs, labels, mask)
    return _reliability(p, y, n_bins, strategy)


def ece(probs, labels, *, n_bins: int = 15, strategy: str = "uniform", mask=None) -> float:
    """Binary expected calibration error (see `reliability_diagram`)."""
    return reliability_diagram(probs, labels, n_bins=n_bins, strategy=strategy, mask=mask).ece


def mce(probs, labels, *, n_bins: int = 15, strategy: str = "uniform", mask=None) -> float:
    """Binary maximum calibration error (see `reliability_diagram`)."""
    return reliability_diagram(probs, labels, n_bins=n_bins, strategy=strategy, mask=mask).mce


# --- binary scores ----------------------------------------------------------------------------


def brier_score(probs, labels, *, mask=None) -> float:
    """mean (p − y)²."""
    p, y = _binary(probs, labels, mask)
    return _mean((p - y) ** 2)


def log_loss(probs, labels, *, eps: float = EPS, mask=None) -> float:
    """mean −[y·log p + (1 − y)·log(1 − p)] with p clipped to [eps, 1 − eps]."""
    p, y = _binary(probs, labels, mask)
    p = np.clip(p, eps, 1 - eps)
    return _mean(-(y * np.log(p) + (1 - y) * np.log1p(-p)))


def accuracy_at_threshold(probs, labels, *, threshold: float = 0.5, mask=None) -> float:
    """Accuracy of predicting positive iff p ≥ threshold."""
    p, y = _binary(probs, labels, mask)
    return _mean(((p >= threshold) == (y == 1)).astype(float))


def roc_auc(scores, labels, *, mask=None) -> float:
    """ROC-AUC via the Mann–Whitney U statistic with average ranks (a tied pos/neg pair counts ½).

    AUC = (Σ rank(pos) − n₊(n₊ + 1)/2) / (n₊ · n₋) = P(score₊ > score₋) + ½ P(score₊ = score₋).
    Scores may be any real numbers (probabilities or logits). NaN when a class is absent.
    """
    s, y = _binary(scores, labels, mask, scores=True)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts  # ranks are 1-based: group g spans start+1 .. start+count
    ranks = (start + (counts + 1) / 2)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_roc_auc(scores, labels, *, mask=None) -> float:
    """Mean of per-column ROC-AUCs; the last axis indexes columns (e.g. candidate cards).

    Leading dimensions are flattened into samples; columns where only one class is present are
    skipped (NaN if none remain). Unlike the pooled `roc_auc`, a predictor that only knows each
    column's base rate scores exactly 0.5. A 1-D input is a single column.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    if s.shape != y.shape:
        raise ValueError(f"scores shape {s.shape} does not match labels shape {y.shape}")
    m = _mask(mask, y.shape)
    if s.ndim <= 1:
        return roc_auc(s, y, mask=m)
    cols = s.shape[-1]
    s, y = s.reshape(-1, cols), y.reshape(-1, cols)
    m = None if m is None else m.reshape(-1, cols)
    aucs = np.array([roc_auc(s[:, j], y[:, j], mask=None if m is None else m[:, j]) for j in range(cols)])
    aucs = aucs[~np.isnan(aucs)]
    return _mean(aucs)


# --- multiclass -------------------------------------------------------------------------------


def _fractional_top_k(p: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
    """Per-row P(target in top-k) with ties broken uniformly at random (expected value)."""
    pt = p[np.arange(t.size), t][:, None]
    greater = (p > pt).sum(-1)
    equal = (p == pt).sum(-1)  # includes the target itself
    return np.clip((k - greater) / equal, 0.0, 1.0)


def top_k_accuracy(probs, targets, *, k: int = 1, mask=None) -> float:
    """Fraction of targets among the k most probable classes.

    Ties are resolved in expectation: if g classes score strictly higher than the target and e
    others tie with it, the hit counts (k − g)/(e + 1), clipped to [0, 1]. A uniform predictor thus
    scores exactly k/K instead of depending on index order.
    """
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    p, t = _multiclass(probs, targets, mask)
    return _mean(_fractional_top_k(p, t, k)) if t.size else float("nan")


def top_label_reliability_diagram(
    probs, targets, *, n_bins: int = 15, strategy: str = "uniform", mask=None
) -> ReliabilityDiagram:
    """Reliability of the top label: confidence = max_k p_k, correctness = top-1 hit (ties fractional)."""
    p, t = _multiclass(probs, targets, mask)
    correct = _fractional_top_k(p, t, 1) if t.size else np.zeros(0)
    return _reliability(p.max(-1) if t.size else np.zeros(0), correct, n_bins, strategy)


def multiclass_ece(probs, targets, *, n_bins: int = 15, strategy: str = "uniform", mask=None) -> float:
    """Top-label expected calibration error."""
    return top_label_reliability_diagram(probs, targets, n_bins=n_bins, strategy=strategy, mask=mask).ece


def multiclass_mce(probs, targets, *, n_bins: int = 15, strategy: str = "uniform", mask=None) -> float:
    """Top-label maximum calibration error."""
    return top_label_reliability_diagram(probs, targets, n_bins=n_bins, strategy=strategy, mask=mask).mce


def multiclass_nll(probs, targets, *, eps: float = EPS, mask=None) -> float:
    """mean −log p_target with p clipped below at eps."""
    p, t = _multiclass(probs, targets, mask)
    return _mean(-np.log(np.clip(p[np.arange(t.size), t], eps, 1.0)))


def multiclass_brier(probs, targets, *, mask=None) -> float:
    """mean Σ_k (p_k − 1[k = target])², in [0, 2]."""
    p, t = _multiclass(probs, targets, mask)
    onehot = np.zeros_like(p)
    onehot[np.arange(t.size), t] = 1.0
    return _mean(((p - onehot) ** 2).sum(-1))
