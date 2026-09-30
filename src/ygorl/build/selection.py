"""Game budgets for deck evolution: which child to play, and when a child is confirmed (#110,
docs/spikes/deck-evolution.md §4, docs/tuning.md).

Both work on common-random-number paired games (:class:`ygorl.build.tuner.PairedEvaluator`): pair ``k`` of a child
and pair ``k`` of its parent share the seed, the opponent and both first players, and a child is judged by the
per-pair difference child minus parent.

- :func:`top_two_thompson` (Russo, COLT 2016) replaces the fixed successive halving (kept as the control,
  :func:`ygorl.build.tuner.successive_halving`): every child starts with one batch of pairs, then each round draws
  batches by top-two Thompson sampling from the children's Gaussian posteriors on their difference (prior: each
  child's predicted gain, e.g. :meth:`ygorl.build.signals.CardValueModel.gain`; likelihood: the observed paired
  differences with the pooled empirical per-pair variance). It stops once the child with the highest posterior mean
  has ``P(diff > 0) > 1 - (1 - confidence) / multiplicity`` and at least ``min_pairs`` pairs (#145: without the two
  guards a null child among 8 stopped the search after one batch about 10% of the time), or when the pair budget is
  spent.
- :func:`screen`: cold-start breadth (#145): many single swaps play one batch each, the best few by paired
  difference go on to the search.
- :func:`sequential_validate`: the chosen child and its parent on fresh pairs (never used by the search), a look
  every ``look`` pairs up to ``cap``, with one-sided repeated confidence bounds from Lan-DeMets O'Brien-Fleming alpha
  spending (:func:`obrien_fleming_bounds`). Accept when the lower bound is above 0; stop as futile when a looser
  (fixed ``futility_z``, non-binding) upper bound is below ``min_effect``; otherwise reject at the cap. The futility
  stop never accepts, so the chance of accepting a child that is no better is at most ``alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from statistics import NormalDist

import numpy as np

from ygorl.cards.ydk import Deck

_N = NormalDist()
SD_PAIR = 0.212  # per-pair standard deviation of a paired difference measured by M1 (benchmarks.md)


@dataclass(frozen=True)
class Arm:
    """One child's posterior on its paired difference to the parent after the search."""

    index: int  # position in the children given to the search
    pairs: int
    observed: float  # raw mean paired difference (NaN before any pair)
    prior_mean: float
    prior_sd: float
    mean: float  # posterior
    sd: float
    p_positive: float  # posterior P(diff > 0)
    p_best: float  # posterior P(this child has the largest difference)


@dataclass
class Race:
    """Result of :func:`top_two_thompson`: ``best`` is the index of the child with the highest posterior mean."""

    best: int | None
    arms: list[Arm]
    scores: list[list[float]]  # per child, one score per pair 0 .. n-1
    base_scores: list[float]
    pairs: int  # child pairs played (the parent's are extra)
    rounds: int
    stop: str  # "confident", "budget" or "empty"
    sd_pair: float  # the pooled per-pair standard deviation used by the last posterior
    threshold: float = 0.95  # the P(diff > 0) the stop needed (confidence corrected for multiplicity)


def _differences(child: Sequence[float], base: Sequence[float]) -> np.ndarray:
    n = min(len(child), len(base))
    d = np.asarray(child[:n], dtype=float) - np.asarray(base[:n], dtype=float)
    return d[np.isfinite(d)]


def _posterior(diffs: Sequence[np.ndarray], prior: Sequence[tuple[float, float]], sd_pair: float,
               prior_pairs: float) -> tuple[np.ndarray, np.ndarray, float]:  # fmt: skip
    """Gaussian posterior means and sds per arm, and the pooled per-pair sd: ``prior_pairs`` pseudo-pairs of
    variance ``sd_pair²`` plus every arm's within-arm squares."""
    ss = prior_pairs * sd_pair**2 + sum(float(((d - d.mean()) ** 2).sum()) for d in diffs if len(d) > 1)
    df = prior_pairs + sum(len(d) - 1 for d in diffs if len(d) > 1)
    var = ss / df
    means, sds = [], []
    for d, (m0, s0) in zip(diffs, prior, strict=True):
        prec = 1 / s0**2 + len(d) / var
        means.append((m0 / s0**2 + float(d.sum()) / var) / prec)
        sds.append(math.sqrt(1 / prec))
    return np.array(means), np.array(sds), math.sqrt(var)


def top_two_thompson(base: Deck, children: Sequence[Deck], evaluator, *,
                     prior: Sequence[tuple[float, float]] | None = None, batch: int = 25, max_pairs: int = 600,
                     batches_per_round: int = 4, confidence: float = 0.95, top_two: float = 0.5,
                     sd_pair: float = SD_PAIR, prior_pairs: float = 10.0, base_scores: Sequence[float] = (),
                     min_pairs: int = 0, multiplicity: int = 1, rng: np.random.Generator | None = None,
                     log: Callable[[str], None] | None = None) -> Race:  # fmt: skip
    """Allocate ``batch``-pair batches of paired games among ``children`` of ``base`` by top-two Thompson sampling.

    ``prior[i]`` = (mean, sd) of child ``i``'s difference to the parent (default N(0, 0.05²)). Every child first
    plays one batch; each later round draws ``batches_per_round`` batches (one child per draw: the top of a
    posterior sample with probability ``top_two``, else the top of a resample that differs) and plays them in one
    ``evaluator.play`` call, together with the parent's pairs the children need (``base_scores``: parent pairs
    already played, e.g. by the diagnosis, with the same evaluator settings). Stops when the child with the highest
    posterior mean has ``P(diff > 0) > 1 - (1 - confidence) / multiplicity`` (Bonferroni over the ``multiplicity``
    candidates the search chose among, e.g. every child a screen looked at) and at least ``min_pairs`` pairs, or
    ``max_pairs`` child pairs have been played (the first batch of every child is always played)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(children)
    prior = list(prior) if prior is not None else [(0.0, 0.05)] * n
    if len(prior) != n or any(not s > 0 for _, s in prior):
        raise ValueError("one prior (mean, positive sd) per child")
    if multiplicity < 1:
        raise ValueError("multiplicity must be at least 1")
    threshold = 1 - (1 - confidence) / multiplicity
    base_s = list(base_scores)
    scores: list[list[float]] = [[] for _ in range(n)]
    if n == 0:
        return Race(None, [], scores, base_s, 0, 0, "empty", sd_pair, threshold)

    def play(counts: dict[int, int]) -> None:
        jobs = [(children[i], range(len(scores[i]), len(scores[i]) + c * batch)) for i, c in counts.items()]
        need = max(r.stop for _, r in jobs)
        if need > len(base_s):
            jobs.append((base, range(len(base_s), need)))
        out = evaluator.play(jobs)
        for i, s in zip(counts, out, strict=False):
            scores[i].extend(np.asarray(s, dtype=float).tolist())
        if need > len(base_s):
            base_s.extend(np.asarray(out[-1], dtype=float).tolist())

    counts = dict.fromkeys(range(n), 1)
    rounds = 0
    while True:
        play(counts)
        rounds += 1
        mean, sd, pooled = _posterior([_differences(s, base_s) for s in scores], prior, sd_pair, prior_pairs)
        best = int(np.argmax(mean))
        p_pos = _N.cdf(mean[best] / sd[best])
        played = sum(len(s) for s in scores)
        if log:
            log(f"round {rounds}: {played} child pairs, best child {best} {mean[best]:+.3f} ± {sd[best]:.3f} "
                f"(P>0 {p_pos:.3f}), sd/pair {pooled:.3f}, {evaluator.games} games")  # fmt: skip
        sure = p_pos > threshold and len(scores[best]) >= min_pairs
        stop = "confident" if sure else "budget" if played + batch > max_pairs else None
        if stop:
            break
        counts = {}
        for _ in range(min(batches_per_round, (max_pairs - played) // batch)):
            i = _top_two(mean, sd, top_two, rng)
            counts[i] = counts.get(i, 0) + 1
    draws = rng.normal(mean, sd, size=(4000, n)).argmax(-1)
    p_best = np.bincount(draws, minlength=n) / len(draws)
    arms = []
    for i in range(n):
        d = _differences(scores[i], base_s)
        arms.append(Arm(i, len(scores[i]), float(d.mean()) if len(d) else math.nan, prior[i][0], prior[i][1],
                        float(mean[i]), float(sd[i]), _N.cdf(mean[i] / sd[i]), float(p_best[i])))  # fmt: skip
    return Race(best, arms, scores, base_s, played, rounds, stop, pooled, threshold)


@dataclass
class Screen:
    """Result of :func:`screen`: ``kept`` are indices into the screened children, best first."""

    kept: list[int]
    diffs: list[float]  # per child, the mean paired difference on the screen's pairs (NaN when none is finite)
    pairs: int
    base_scores: list[float]


def screen(base: Deck, children: Sequence[Deck], evaluator, *, pairs: int = 25, keep: int = 4,
           base_scores: Sequence[float] = (), tiebreak: Sequence[float] | None = None,
           rng: np.random.Generator | None = None) -> Screen:  # fmt: skip
    """Cold-start breadth (#145): every child plays pairs ``0 .. pairs - 1`` (one call, with the parent's pairs it
    still lacks), and the ``keep`` children with the highest mean paired difference go on, best first. Ties (25 pairs
    give few distinct values) go to the higher ``tiebreak`` (e.g. the predicted gain), then to a random order.
    The pairs are the search's first batch: a search that follows reuses them (a game log plays nothing twice), and
    every screened child's first batch is non-adaptive data for the signal library."""
    rng = rng if rng is not None else np.random.default_rng(0)
    base_s = list(base_scores)
    jobs = [(c, range(pairs)) for c in children]
    if len(base_s) < pairs:
        jobs.append((base, range(len(base_s), pairs)))
    out = evaluator.play(jobs) if jobs else []
    if len(base_s) < pairs:
        base_s.extend(np.asarray(out[-1], dtype=float).tolist())
    diffs = []
    for i in range(len(children)):
        d = _differences(np.asarray(out[i], dtype=float), base_s[:pairs])
        diffs.append(float(d.mean()) if len(d) else math.nan)
    tb = list(tiebreak) if tiebreak is not None else [0.0] * len(children)
    noise = rng.random(len(children))
    order = sorted(range(len(children)),
                   key=lambda i: (-(diffs[i] if math.isfinite(diffs[i]) else -math.inf), -tb[i], noise[i]))  # fmt: skip
    return Screen(order[:keep], diffs, pairs, base_s)


def _top_two(mean: np.ndarray, sd: np.ndarray, beta: float, rng: np.random.Generator, tries: int = 100) -> int:
    """Russo's top-two Thompson draw: the leader of a posterior sample with probability ``beta``, else the leader of
    the first resample led by another arm (the second of the first sample if none within ``tries``)."""
    theta = rng.normal(mean, sd)
    lead = int(np.argmax(theta))
    if len(mean) == 1 or rng.random() < beta:
        return lead
    for _ in range(tries):
        other = int(np.argmax(rng.normal(mean, sd)))
        if other != lead:
            return other
    return int(np.argsort(theta)[-2])


def _spent(t: float, alpha: float) -> float:
    """Lan-DeMets O'Brien-Fleming spending function (one-sided level ``alpha``) at information fraction ``t``."""
    return 2 * (1 - _N.cdf(_N.inv_cdf(1 - alpha / 2) / math.sqrt(t)))


@cache
def obrien_fleming_bounds(fractions: tuple[float, ...], alpha: float, step: float = 0.002) -> tuple[float, ...]:
    """One-sided critical z per look at information ``fractions`` (increasing, the last usually 1): the
    probability under no effect that the standardized statistic first exceeds the bound at look ``k`` is the
    O'Brien-Fleming spending ``alpha(t_k) - alpha(t_(k-1))``. Computed exactly (to the grid) by propagating the
    density of the Brownian score ``W(t)`` over the not-yet-stopped region (Armitage-McPherson-Rowe recursion)."""
    if not fractions or any(b <= a for a, b in zip((0.0, *fractions), fractions, strict=False)) or fractions[-1] > 1:
        raise ValueError("information fractions must increase within (0, 1]")
    x = np.arange(-10.0, 10.0 + step / 2, step)

    def pdf(z: np.ndarray, sd: float) -> np.ndarray:
        return np.exp(-0.5 * (z / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

    f, prev_t, prev_spent, out = None, 0.0, 0.0, []
    for t in fractions:
        spend = _spent(t, alpha) - prev_spent
        if f is None:
            g = pdf(x, math.sqrt(t))
        else:
            half = int(math.ceil(9 * math.sqrt(t - prev_t) / step))
            kernel = pdf(np.arange(-half, half + 1) * step, math.sqrt(t - prev_t)) * step
            g = np.convolve(f, kernel, mode="same")
        tail = np.cumsum((g * step)[::-1])[::-1]  # P(W(t) >= x_i, not stopped before)
        i = int(np.searchsorted(-tail, -spend, side="left"))  # first grid point whose tail is <= spend
        if i >= len(x):
            b = x[-1]
        elif i == 0:
            b = x[0]
        else:  # interpolate between x[i-1] (tail > spend) and x[i]
            w = (tail[i - 1] - spend) / (tail[i - 1] - tail[i])
            b = x[i - 1] + w * step
        out.append(float(b / math.sqrt(t)))
        f = np.where(x < b, g, 0.0)
        prev_t, prev_spent = t, _spent(t, alpha)
    return tuple(out)


@dataclass(frozen=True)
class Look:
    pairs: int
    mean: float
    lower: float  # efficacy bound: mean - z_bound * se
    upper: float  # futility bound: mean + futility_z * se
    z_bound: float


@dataclass
class Validation:
    """Result of :func:`sequential_validate`: ``decision`` is "accept", "futile" or "cap"."""

    decision: str
    looks: list[Look]
    scores: list[float] = field(default_factory=list)  # the child's fresh pairs
    base_scores: list[float] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"

    @property
    def pairs(self) -> int:
        return len(self.scores)


def sequential_validate(base: Deck, child: Deck, evaluator, *, look: int = 100, cap: int = 1000,
                        alpha: float = 0.025, min_effect: float = 0.02, offset: int = 1_000_000,
                        futility_z: float = 1.645, sd_pair: float = SD_PAIR, prior_pairs: float = 10.0,
                        log: Callable[[str], None] | None = None) -> Validation:  # fmt: skip
    """Group-sequential check of ``child`` against ``base`` on fresh pairs from ``offset`` (never used by a search).

    At each look (every ``look`` pairs, at most ``cap``), accept when the lower bound ``mean - z_k * sd / sqrt(n)``
    is above 0, with ``z_k`` from :func:`obrien_fleming_bounds` at information ``pairs / cap``; stop as futile when
    the upper bound ``mean + futility_z * sd / sqrt(n)`` is below ``min_effect``. The futility bound is non-binding
    (stopping only ever rejects), so it can be much looser than the efficacy bound without raising the false-positive
    rate above ``alpha``; with the efficacy z (7.0 at the first of 10 looks) it would almost never fire. The per-pair
    variance ``s²`` counts ``prior_pairs`` pseudo-pairs of ``sd_pair²`` (as in :func:`top_two_thompson`), so a short
    run of identical pairs (common under common random numbers) cannot give a zero-width interval."""
    if look <= 0 or cap < look:
        raise ValueError("need 0 < look <= cap")
    stops = list(range(look, cap + 1, look))
    if stops[-1] != cap:
        stops.append(cap)
    bounds = obrien_fleming_bounds(tuple(s / cap for s in stops), alpha)
    out = Validation("cap", [])
    done = 0
    for stop, z in zip(stops, bounds, strict=True):
        s = evaluator.play([(child, range(offset + done, offset + stop)), (base, range(offset + done, offset + stop))])
        out.scores.extend(np.asarray(s[0], dtype=float).tolist())
        out.base_scores.extend(np.asarray(s[1], dtype=float).tolist())
        done = stop
        d = _differences(out.scores, out.base_scores)
        if len(d) < 2:
            continue
        var = (prior_pairs * sd_pair**2 + float(((d - d.mean()) ** 2).sum())) / (prior_pairs + len(d) - 1)
        mean, se = float(d.mean()), math.sqrt(var / len(d))
        lower, upper = mean - z * se, mean + futility_z * se
        out.looks.append(Look(stop, mean, lower, upper, z))
        if log:
            log(f"validation {stop} pairs: {mean:+.3f} (accept if {lower:+.3f} > 0, futile if {upper:+.3f} < "
                f"{min_effect:+.3f}), z {z:.2f}")  # fmt: skip
        if lower > 0:
            out.decision = "accept"
            break
        if upper < min_effect:
            out.decision = "futile"
            break
    return out


__all__ = ["SD_PAIR", "Arm", "Look", "Race", "Screen", "Validation", "obrien_fleming_bounds", "screen",
           "sequential_validate", "top_two_thompson"]  # fmt: skip
