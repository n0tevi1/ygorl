"""Per-head belief report, baseline predictors and synthetic data (T3.5).

The opponent model (docs/design/04-opponent-model.md) has these heads; shapes use N = samples
(decision points), K = deck types (meta types + "other"), C = candidate cards (meta union +
generic cards, indexed by a fixed card-`password` vocabulary), S = opponent set-card zones,
R = hand role bits:

=================  ============  ===========  ==========================================
head               probs         targets      meaning
=================  ============  ===========  ==========================================
deck_type          [N, K]        [N] int      softmax over deck types
remaining_copies   [N, C, 4]     [N, C] int   remaining copies 0..3 per card (multi-head)
hand               [N, C]        [N, C] 0/1   P(≥ 1 copy in hand) per card
hand_roles         [N, R]        [N, R] 0/1   role bits (hand trap, Ash, Maxx "C", ...)
set_cards          [N, S, C]     [N, S] int   class of each face-down card
responded          [N]           [N] 0/1      P(my next search/special summon is responded to)
=================  ============  ===========  ==========================================

Each head is optional and carries an optional bool ``mask`` shaped like its targets (True =
evaluate). Mask out public information (cards the loss masks set by construction, fully revealed
copies) and empty set zones; masked targets may be padding such as -1.

Metric formulas: `ygorl.eval.calibration` and docs/belief-eval.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from ygorl.eval import calibration as cal

N_COPY_CLASSES = 4  # 0..3 remaining copies


@dataclass(frozen=True)
class Head:
    """Predictions, targets and an optional bool mask (True = evaluate) for one belief head."""

    probs: np.ndarray
    targets: np.ndarray
    mask: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "probs", np.asarray(self.probs, dtype=float))
        object.__setattr__(self, "targets", np.asarray(self.targets))
        if self.mask is not None:
            object.__setattr__(self, "mask", np.asarray(self.mask))

    def included(self) -> np.ndarray:
        """Bool array shaped like targets: the mask, or all-True."""
        return np.ones(self.targets.shape, dtype=bool) if self.mask is None else self.mask.astype(bool)


# name -> (binary?, probs shape as documented)
HEAD_SPECS: dict[str, tuple[bool, str]] = {
    "deck_type": (False, "[N, K]"),
    "remaining_copies": (False, "[N, C, 4]"),
    "hand": (True, "[N, C]"),
    "hand_roles": (True, "[N, R]"),
    "set_cards": (False, "[N, S, C]"),
    "responded": (True, "[N]"),
}


@dataclass(frozen=True)
class BeliefBatch:
    """One evaluation batch of belief-head outputs; shapes are validated on construction."""

    deck_type: Head | None = None
    remaining_copies: Head | None = None
    hand: Head | None = None
    hand_roles: Head | None = None
    set_cards: Head | None = None
    responded: Head | None = None

    def __post_init__(self):
        sizes = {}
        for name, head in self.heads().items():
            _check_shapes(name, head)
            sizes[name] = head.probs.shape[0]
        if len(set(sizes.values())) > 1:
            raise ValueError(f"all heads must share the batch size N, got {sizes}")

    def heads(self) -> dict[str, Head]:
        """The heads that are present, in HEAD_SPECS order."""
        return {name: getattr(self, name) for name in HEAD_SPECS if getattr(self, name) is not None}

    @property
    def n(self) -> int:
        """Batch size N (0 for an empty batch)."""
        return next((h.probs.shape[0] for h in self.heads().values()), 0)


def _check_shapes(name: str, head: Head) -> None:
    binary, spec = HEAD_SPECS[name]
    ndim = spec.count(",") + 1
    p, t = head.probs, head.targets
    if p.ndim != ndim:
        raise ValueError(f"{name}: probs must have shape {spec}, got {p.shape}")
    if name == "remaining_copies" and p.shape[-1] != N_COPY_CLASSES:
        raise ValueError(f"{name}: probs last dim must be {N_COPY_CLASSES} (0..3 copies), got {p.shape}")
    want = p.shape if binary else p.shape[:-1]
    if t.shape != want:
        raise ValueError(f"{name}: targets must have shape {want} for probs {p.shape}, got {t.shape}")
    if head.mask is not None and head.mask.shape != want:
        raise ValueError(f"{name}: mask must have shape {want} (like targets), got {head.mask.shape}")


# --- report -----------------------------------------------------------------------------------


def _head_metrics(name: str, h: Head, n_bins: int, strategy: str) -> dict[str, float]:
    kw = {"mask": h.mask}
    bins = {"n_bins": n_bins, "strategy": strategy, **kw}
    if HEAD_SPECS[name][0]:
        out = {"auc": cal.roc_auc(h.probs, h.targets, **kw)}
        if h.probs.ndim > 1:  # per-card / per-role AUC, blind to column base rates
            out["auc_macro"] = cal.macro_roc_auc(h.probs, h.targets, **kw)
        return out | {
            "ece": cal.ece(h.probs, h.targets, **bins),
            "brier": cal.brier_score(h.probs, h.targets, **kw),
            "log_loss": cal.log_loss(h.probs, h.targets, **kw),
            "accuracy": cal.accuracy_at_threshold(h.probs, h.targets, **kw),
        }
    out = {"top1": cal.top_k_accuracy(h.probs, h.targets, k=1, **kw)}
    if name == "remaining_copies":
        out = {"accuracy": out["top1"]}
    else:
        out["top3"] = cal.top_k_accuracy(h.probs, h.targets, k=3, **kw)
    out["ece"] = cal.multiclass_ece(h.probs, h.targets, **bins)
    out["nll"] = cal.multiclass_nll(h.probs, h.targets, **kw)
    out["brier"] = cal.multiclass_brier(h.probs, h.targets, **kw)
    return out


def evaluate_beliefs(batch: BeliefBatch, *, n_bins: int = 15, strategy: str = "uniform") -> dict[str, float]:
    """Flat ``{"<head>/<metric>": value}`` report over the heads present in ``batch``.

    Binary heads (hand, hand_roles, responded): auc (pooled), ece, brier, log_loss, accuracy
    (p ≥ 0.5); hand and hand_roles also auc_macro (mean per-card / per-role AUC).
    deck_type, set_cards: top1, top3, ece (top label), nll, brier. remaining_copies: accuracy
    (top-1 over 0..3 copies), ece, nll, brier. Every head also reports ``n``, the number of
    evaluated (unmasked) entries. ECE uses ``n_bins`` bins of the given ``strategy``.
    """
    report: dict[str, float] = {}
    for name, head in batch.heads().items():
        try:
            metrics = _head_metrics(name, head, n_bins, strategy)
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from e
        metrics["n"] = float(head.included().sum())
        report.update({f"{name}/{k}": float(v) for k, v in metrics.items()})
    return report


# --- baseline predictors ----------------------------------------------------------------------


def _map_heads(batch: BeliefBatch, fn) -> BeliefBatch:
    """Replace each head's probs with fn(name, head); targets and masks are kept."""
    return BeliefBatch(**{name: dataclasses.replace(h, probs=fn(name, h)) for name, h in batch.heads().items()})


def uniform_predictor(batch: BeliefBatch) -> BeliefBatch:
    """Constant maximum-entropy predictions: 0.5 for binary heads, 1/K for multiclass heads."""
    return _map_heads(
        batch, lambda name, h: np.full(h.probs.shape, 0.5 if HEAD_SPECS[name][0] else 1 / h.probs.shape[-1])
    )


def random_predictor(batch: BeliefBatch, *, seed: int = 0) -> BeliefBatch:
    """Uninformed but confident predictions: U(0, 1) for binary heads, Dirichlet(1) for multiclass."""
    rng = np.random.default_rng(seed)

    def draw(name, h):
        if HEAD_SPECS[name][0]:
            return rng.random(h.probs.shape)
        return rng.dirichlet(np.ones(h.probs.shape[-1]), size=h.probs.shape[:-1])

    return _map_heads(batch, draw)


def prior_predictor(batch: BeliefBatch, *, fit: BeliefBatch | None = None, smoothing: float = 0.0) -> BeliefBatch:
    """Frequency prior estimated from the (unmasked) targets of ``fit`` (default: ``batch``).

    Ignores the observation: deck_type → deck-type shares; remaining_copies → per-card copy
    distribution; hand / hand_roles → per-column base rate; set_cards → class shares pooled over
    zones; responded → base rate. ``smoothing`` is an additive (Laplace) pseudo-count per class.
    Classes / columns with no data fall back to uniform.
    """
    fit = batch if fit is None else fit
    fit_heads = fit.heads()

    def estimate(name, h):
        if name not in fit_heads:
            raise ValueError(f"{name}: fit batch has no such head")
        f = fit_heads[name]
        if f.probs.shape[1:] != h.probs.shape[1:]:
            raise ValueError(f"{name}: fit probs shape {f.probs.shape} incompatible with {h.probs.shape}")
        m = f.included()
        if HEAD_SPECS[name][0]:  # binary: per-column rate, columns = everything after N
            y = np.where(m, f.targets, 0).astype(float)
            pos, cnt = y.sum(0) + smoothing, m.sum(0) + 2 * smoothing
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(cnt > 0, pos / np.maximum(cnt, 1e-300), 0.5)
            return np.broadcast_to(rate, h.probs.shape).copy()
        k = h.probs.shape[-1]
        t = f.targets[m].astype(np.int64)
        if name == "remaining_copies":  # per card
            cards = np.nonzero(m)[1]
            counts = np.zeros((h.probs.shape[1], k))
            np.add.at(counts, (cards, t), 1.0)
        else:  # deck_type, set_cards: one distribution shared by all rows / zones
            counts = np.bincount(t, minlength=k).astype(float)
        counts += smoothing
        total = counts.sum(-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = np.where(total > 0, counts / np.maximum(total, 1e-300), 1 / k)
        return np.broadcast_to(dist, h.probs.shape).copy()

    return _map_heads(batch, estimate)


# --- synthetic data ---------------------------------------------------------------------------

SYNTHETIC_DEFAULTS: dict[str, Any] = {
    "n_deck_types": 8,
    "n_cards": 40,
    "n_set_zones": 5,
    "n_roles": 5,
    "public_fraction": 0.2,
    "signal": 1.5,
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _categorical(rng: np.random.Generator, q: np.ndarray) -> np.ndarray:
    u = rng.random(q.shape[:-1] + (1,))
    return np.minimum((u > q.cumsum(-1)).sum(-1), q.shape[-1] - 1)


def synthetic_batch(n: int, *, seed: int = 0, **kwargs) -> BeliefBatch:
    """A synthetic world whose ``probs`` are the true generating posteriors (an oracle predictor).

    Every target is sampled from its ``probs``, so the returned batch is perfectly calibrated by
    construction. World parameters (Zipf-like deck-type and set-card shares, per-card base rates,
    per-card copy distributions) and per-sample evidence (Gaussian logit noise of scale
    ``signal``) are drawn from ``seed``. A ``public_fraction`` of card entries is masked out as
    public information; each set zone is empty (masked, target -1) with probability 0.4.
    Keyword arguments override SYNTHETIC_DEFAULTS.
    """
    unknown = set(kwargs) - set(SYNTHETIC_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown synthetic_batch arguments: {sorted(unknown)}")
    o = {**SYNTHETIC_DEFAULTS, **kwargs}
    k, c, s, r, pub, sig = (
        o[x] for x in ("n_deck_types", "n_cards", "n_set_zones", "n_roles", "public_fraction", "signal")
    )
    rng = np.random.default_rng(seed)

    def zipf(m):
        w = 1 / np.arange(1, m + 1)
        return rng.permutation(w / w.sum())

    def public_mask(shape):
        return rng.random(shape) >= pub

    q = _softmax(np.log(zipf(k)) + sig * rng.standard_normal((n, k)))
    deck_type = Head(q, _categorical(rng, q))

    q = _softmax(rng.standard_normal((c, N_COPY_CLASSES)) + sig * rng.standard_normal((n, c, N_COPY_CLASSES)))
    remaining = Head(q, _categorical(rng, q), public_mask((n, c)))

    def binary(cols, a, b, mask):
        base = rng.beta(a, b, cols)
        p = _sigmoid(np.log(base / (1 - base)) + sig * rng.standard_normal((n, cols)))
        return Head(p, (rng.random(p.shape) < p).astype(np.int64), mask)

    hand = binary(c, 1.0, 4.0, public_mask((n, c)))
    roles = binary(r, 2.0, 3.0, None)

    q = _softmax(np.log(zipf(c)) + sig * rng.standard_normal((n, s, c)))
    occupied = rng.random((n, s)) < 0.6
    set_cards = Head(q, np.where(occupied, _categorical(rng, q), -1), occupied)

    p = _sigmoid(-0.5 + sig * rng.standard_normal(n))
    responded = Head(p, (rng.random(n) < p).astype(np.int64))

    return BeliefBatch(deck_type, remaining, hand, roles, set_cards, responded)


BASELINES = ("uniform", "random", "prior", "oracle")


def baseline_report(n: int = 20_000, *, seed: int = 0, n_bins: int = 15, **synthetic) -> dict[str, dict[str, float]]:
    """`evaluate_beliefs` for the baseline predictors on one synthetic batch.

    uniform / random: uninformed; prior: in-sample frequency prior (ignores the observation);
    oracle: the true generating posteriors (an upper bound for the synthetic world).
    """
    batch = synthetic_batch(n, seed=seed, **synthetic)
    preds = {
        "uniform": uniform_predictor(batch),
        "random": random_predictor(batch, seed=seed + 1),
        "prior": prior_predictor(batch),
        "oracle": batch,
    }
    return {name: evaluate_beliefs(b, n_bins=n_bins) for name, b in preds.items()}


def format_baselines(report: dict[str, dict[str, float]]) -> str:
    """Markdown table: one row per metric, one column per predictor."""
    names = list(report)
    keys = list(next(iter(report.values()), {}))
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key in keys:
        fmt = "{:.0f}" if key.endswith("/n") else "{:.3f}"
        lines.append(f"| {key} | " + " | ".join(fmt.format(report[m][key]) for m in names) + " |")
    return "\n".join(lines)
