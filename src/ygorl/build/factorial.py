"""Fractional-factorial evaluation of edit groups (#152, docs/tuning.md「析因评估」).

One-at-a-time screening plays every candidate edit of a parent against the parent, so each game informs one edit.
Here ``k`` compatible edits of one parent are evaluated together as a two-level fractional factorial design
(2^(k−p) variants): variant ``r`` is the parent with the edits at level +1 in run ``r`` applied in place (every other
card keeps its position, so all variants share the parent's common random numbers), every variant plays the same
pairs, and one regression on the per-pair scores gives every edit's main effect and the estimable two-way
interactions. Every game informs every main effect: at equal game counts the variance of a main effect is about
``2 / (k + 1)`` of a one-at-a-time difference (``k / 2`` when the parent's pairs are already played).

- :func:`design`: the 2^(k−p) design with minimum-aberration generators (:data:`GENERATORS`), resolution IV where the
  run count allows (main effects not aliased with two-way interactions), signs chosen so run 0 is all −1 (the
  parent). Its defining relation gives the alias structure (:meth:`Design.aliases`) and the two-way interaction
  columns that can be estimated apart from the main effects (:meth:`Design.interactions`).
- :func:`plan`: resolves each edit to its own copy of the card it takes out (no two edits touch the same copy; an edit
  with no copy left, an illegal single edit or a repeated one is dropped before the design), builds the design over
  the remaining edits and repairs each illegal variant by leaving out the fewest of its edits (later ones first).
  Every dropped edit and every repair is reported; estimation uses the levels actually played.
- :func:`estimate`: least squares with a fixed effect per pair (the pair's common random numbers cancel) and
  cluster-robust (by pair) standard errors; with complete, unrepaired data it equals the mean of the per-pair
  contrasts and their standard error over pairs.
- :func:`evaluate_edits`: plan, play every variant on the same pairs through the evaluator's ``play``, estimate.
  :meth:`FactorialResult.observations` gives each main effect as a single-edit observation for
  :class:`ygorl.build.signals.CardValueModel`.
- :func:`one_at_a_time`: the parent and each single edit on the same pairs (the comparison arm).

Effects are in pair-score units (win rate), in the usual two-level convention: a main effect is the mean score with
the edit minus without it, averaged over the design's settings of the other edits; an interaction ``AB`` is half the
difference between A's effect with B and without B. Main effects are unbiased when interactions of three or more
edits are negligible (resolution IV aliases them with main effects), interaction chains (``AB = CD``) estimate the
signed sum of their members.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from statistics import NormalDist

import numpy as np

from ygorl.build.signals import Observation
from ygorl.build.tuner import Edit
from ygorl.cards.ydk import Deck

LETTERS = "ABCDEFGHJKLMNOPQ"  # factor names; I is the identity of the defining relation
# minimum-aberration generators (Box, Hunter & Hunter 2005, Table 6.22; Montgomery 2017, Table 8.14): per (k, p), the
# words of base factors that the added factors (k − p .. k − 1) are the product of, up to the sign
GENERATORS: dict[tuple[int, int], tuple[str, ...]] = {
    (3, 1): ("AB",),  # III, 4 runs
    (4, 1): ("ABC",),  # IV, 8 runs
    (5, 1): ("ABCD",),  # V, 16 runs
    (5, 2): ("AB", "AC"),  # III, 8 runs
    (6, 1): ("ABCDE",),  # VI, 32 runs
    (6, 2): ("ABC", "BCD"),  # IV, 16 runs
    (6, 3): ("AB", "AC", "BC"),  # III, 8 runs
    (7, 2): ("ABCD", "ABDE"),  # IV, 32 runs
    (7, 3): ("ABC", "BCD", "ACD"),  # IV, 16 runs
    (7, 4): ("AB", "AC", "BC", "ABC"),  # III, 8 runs
    (8, 3): ("ABC", "ABD", "BCDE"),  # IV, 32 runs
    (8, 4): ("BCD", "ACD", "ABC", "ABD"),  # IV, 16 runs
}
MAX_K = 8


def _bits(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


def default_p(k: int, min_resolution: int = 4) -> int:
    """The largest tabulated fraction of ``k`` factors with at least ``min_resolution`` (0: the full factorial)."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be 1..{MAX_K}")
    best = 0
    for (kk, p), _ in GENERATORS.items():
        if kk == k and p > best and design(k, p).resolution >= min_resolution:
            best = p
    return best


@dataclass(frozen=True, eq=False)
class Design:
    """A two-level 2^(k−p) design: ``matrix[run, factor]`` in ±1, run 0 all −1 (the parent)."""

    k: int
    p: int
    matrix: np.ndarray
    defining: tuple[tuple[int, int], ...]  # (word bitmask, sign): I = sign · word, identity excluded
    generators: tuple[str, ...] = ()  # e.g. "D = ABC", "E = −ABCD"

    @property
    def runs(self) -> int:
        return len(self.matrix)

    @property
    def resolution(self) -> float:
        """Length of the shortest word of the defining relation (inf for the full factorial)."""
        return min((len(_bits(w)) for w, _ in self.defining), default=math.inf)

    def name(self, mask: int) -> str:
        return "".join(LETTERS[i] for i in _bits(mask)) or "I"

    def aliases(self, mask: int) -> list[tuple[int, int]]:
        """``(sign, effect bitmask)`` of every other effect whose column equals ``sign`` times ``mask``'s column."""
        return [(s, mask ^ w) for w, s in self.defining]

    def interactions(self) -> list[tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]]]:
        """The two-way interaction columns estimable apart from the main effects: per alias class (not containing a
        main effect), its first pair and the class's other two-way members with their signs relative to it."""
        seen: set[int] = set()
        out = []
        for i, j in combinations(range(self.k), 2):
            m = 1 << i | 1 << j
            if m in seen:
                continue
            chain = self.aliases(m)
            if any(len(_bits(a)) < 2 for _, a in chain):
                continue  # aliased with a main effect (resolution III): not estimable apart from it
            seen.add(m)
            members = []
            for s, a in chain:
                if len(_bits(a)) == 2:
                    seen.add(a)
                    members.append((s, tuple(_bits(a))))
            out.append(((i, j), members))
        return out

    def label(self, term: Sequence[int]) -> str:
        """``"AB = CD = −EF"``: the term and the two-way members of its alias chain (main effects: just the name)."""
        m = sum(1 << i for i in term)
        text = self.name(m)
        if len(term) == 2:
            for s, a in self.aliases(m):
                if len(_bits(a)) == 2:
                    text += f" = {'' if s > 0 else '−'}{self.name(a)}"
        return text

    def to_dict(self) -> dict:
        res = self.resolution
        return {"k": self.k, "p": self.p, "runs": self.runs, "generators": list(self.generators),
                "resolution": None if math.isinf(res) else int(res),
                "defining": [f"I = {'' if s > 0 else '−'}{self.name(w)}" for w, s in self.defining],
                "matrix": self.matrix.astype(int).tolist()}  # fmt: skip


def design(k: int, p: int | None = None) -> Design:
    """The 2^(k−p) design of :data:`GENERATORS` (``p`` default: :func:`default_p`, resolution IV where possible).
    Base factors run in standard order; an added factor ``j = s · word`` takes ``s = (−1)^(|word| + 1)``, so the
    all −1 run (the parent) is in the fraction and is run 0."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be 1..{MAX_K}")
    p = default_p(k) if p is None else p
    gens = GENERATORS.get((k, p), ()) if p else ()
    if p and not gens:
        raise ValueError(f"no tabulated 2^({k}-{p}) design; have {sorted(x for x in GENERATORS if x[0] == k)}")
    b = k - p
    rows = np.arange(2**b)
    x = np.empty((2**b, k))
    for i in range(b):
        x[:, i] = np.where(rows >> i & 1, 1.0, -1.0)
    group = {0: 1}
    text = []
    for q, w in enumerate(gens):
        idx = [LETTERS.index(c) for c in w]
        sign = (-1) ** (len(idx) + 1)
        x[:, b + q] = sign * np.prod(x[:, idx], axis=1)
        text.append(f"{LETTERS[b + q]} = {'' if sign > 0 else '−'}{w}")
        word = sum(1 << i for i in idx) | 1 << (b + q)
        group.update({a ^ word: s * sign for a, s in list(group.items())})
    defining = tuple(sorted(((w, s) for w, s in group.items() if w), key=lambda t: (len(_bits(t[0])), t[0])))
    return Design(k, p, x, defining, tuple(text))


# ------------------------------------------------------------------ edits and variants


@dataclass(frozen=True)
class Placed:
    """An edit resolved to one copy: position ``index`` of its section in the parent."""

    edit: Edit
    index: int


def place(parent: Deck, edits: Sequence[Edit]) -> tuple[list[Placed], list[dict]]:
    """Each edit takes its own copy of ``edit.out`` (the first copy no earlier edit took), so no two edits touch the
    same card copy; an edit with no copy left, or repeating an earlier edit, is dropped (``{"edit", "reason"}``)."""
    taken: set[tuple[str, int]] = set()
    seen: set[Edit] = set()
    placed, dropped = [], []
    for e in edits:
        if e in seen:
            dropped.append({"edit": _edit_json(e), "reason": "duplicate"})
            continue
        cards = getattr(parent, e.section)
        idx = next((i for i, c in enumerate(cards) if c == e.out and (e.section, i) not in taken), None)
        if idx is None:
            dropped.append({"edit": _edit_json(e), "reason": "no_copy"})
            continue
        taken.add((e.section, idx))
        seen.add(e)
        placed.append(Placed(e, idx))
    return placed, dropped


def variant(parent: Deck, placed: Sequence[Placed]) -> Deck:
    """``parent`` with every placed edit applied in place (the others keep their positions)."""
    main, extra = list(parent.main), list(parent.extra)
    for pl in placed:
        (main if pl.edit.section == "main" else extra)[pl.index] = pl.edit.into
    return Deck(main=tuple(main), extra=tuple(extra), side=parent.side, name=parent.name)


@dataclass
class Plan:
    """What :func:`evaluate_edits` plays: the design over ``placed``, one deck per run with the levels actually
    applied (``levels`` differs from ``design.matrix`` where a variant was repaired)."""

    design: Design
    placed: list[Placed]
    dropped: list[dict]  # edits left out of the design: {"edit", "reason": "duplicate" | "no_copy" | "illegal"}
    repairs: list[dict]  # {"run", "edit" (factor), "letter"}: edits left out of one variant to make it legal
    levels: np.ndarray
    decks: list[Deck]

    @property
    def edits(self) -> list[Edit]:
        return [pl.edit for pl in self.placed]


def plan(parent: Deck, edits: Sequence[Edit], *, legal: Callable[[Deck], bool], p: int | None = None) -> Plan:
    """Place the edits (:func:`place`), drop the ones that are illegal on their own, build the design over the rest
    (``p`` default: resolution IV where possible) and repair each illegal variant by leaving out the fewest of its
    edits that make it legal (later edits first)."""
    placed, dropped = place(parent, edits)
    ok = []
    for pl in placed:
        if legal(variant(parent, [pl])):
            ok.append(pl)
        else:
            dropped.append({"edit": _edit_json(pl.edit), "reason": "illegal"})
    if not ok:
        raise ValueError("no compatible legal edit to evaluate")
    d = design(len(ok), p)
    levels = d.matrix.copy()
    decks, repairs = [], []
    for r in range(d.runs):
        on = [j for j in range(d.k) if levels[r, j] > 0]
        deck = variant(parent, [ok[j] for j in on])
        if not legal(deck):
            for out in _removals(on):  # the fewest edits, later ones first
                rest = [j for j in on if j not in out]
                if legal(cand := variant(parent, [ok[j] for j in rest])):
                    deck = cand
                    for j in sorted(out):
                        levels[r, j] = -1.0
                        repairs.append({"run": r, "edit": j, "letter": LETTERS[j]})
                    break
        decks.append(deck)
    return Plan(d, ok, dropped, repairs, levels, decks)


def _removals(on: Sequence[int]):
    """Subsets of ``on`` to leave out, fewest first, later edits first (all of them last: the parent is legal)."""
    back = list(reversed(on))
    for size in range(1, len(on) + 1):
        yield from (set(c) for c in combinations(back, size))


# ------------------------------------------------------------------ estimation


@dataclass(frozen=True)
class Effect:
    term: tuple[int, ...]  # (j,) a main effect, (i, j) a two-way interaction (its alias chain's first pair)
    label: str  # "A", "AB = CD"
    effect: float  # two-level convention (module docstring): 2 × the ±1 regression coefficient
    stderr: float  # cluster-robust by pair

    def to_dict(self) -> dict:
        return {"term": list(self.term), "label": self.label, "effect": _finite(self.effect),
                "stderr": _finite(self.stderr)}  # fmt: skip


def estimate(levels: np.ndarray, scores: np.ndarray,
             terms: Sequence[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray, list[int]]:  # fmt: skip
    """Effects (2β) and cluster-robust standard errors of ``terms`` (each a tuple of factor indices; the column is
    the product of their levels) from ``scores[run, pair]`` (NaN: missing), with a fixed effect per pair.

    Within each pair the columns and scores are centred on the pair's observed runs (the pair effect, i.e. the
    common random numbers, drops out); β = (Σ Z̃ᵀZ̃)⁻¹ Σ Z̃ᵀỹ; V = G/(G − 1) · A⁻¹ (Σ_pairs s sᵀ) A⁻¹ with s the pair's
    score vector Z̃ᵀ(ỹ − Z̃β) (CR1, G pairs with at least two observed runs). Returns (effects, standard errors, the
    indices of the terms kept: an interaction whose column is collinear with earlier ones after repairs is dropped,
    a main effect raises)."""
    levels = np.asarray(levels, dtype=float)
    y = np.asarray(scores, dtype=float)
    obs = np.isfinite(y)
    y = np.where(obs, y, 0.0)
    w = obs.astype(float)  # [run, pair]
    n = w.sum(0)
    use = n >= 2
    w, y, n = w[:, use], y[:, use], n[use]
    g = int(use.sum())
    if g < 2:
        raise ValueError("need at least two pairs with two or more observed runs")
    keep: list[int] = []
    for t, term in enumerate(terms):
        z = np.stack([np.prod(levels[:, list(terms[i])], axis=1) for i in [*keep, t]], 1)
        if _rank(_gram(z, w, n)) == len(keep) + 1:
            keep.append(t)
        elif len(term) == 1:
            raise ValueError(f"main effect of factor {term[0]} is not estimable (collinear after repairs)")
    z = np.stack([np.prod(levels[:, list(terms[i])], axis=1) for i in keep], 1)  # [run, term]
    zbar = (w.T @ z) / n[:, None]  # [pair, term]
    ybar = (w * y).sum(0) / n  # [pair]
    zc = z[:, None, :] - zbar[None, :, :]  # [run, pair, term]
    yc = (y - ybar[None, :]) * w
    a = np.einsum("rp,rpi,rpj->ij", w, zc, zc)
    beta = np.linalg.solve(a, np.einsum("rpi,rp->i", zc, yc))
    resid = (yc - np.einsum("rpi,i->rp", zc, beta)) * w
    s = np.einsum("rpi,rp->pi", zc, resid)  # [pair, term]
    ainv = np.linalg.inv(a)
    v = ainv @ (s.T @ s) @ ainv * g / (g - 1)
    return 2 * beta, 2 * np.sqrt(np.clip(np.diag(v), 0, None)), keep


def _gram(z: np.ndarray, w: np.ndarray, n: np.ndarray) -> np.ndarray:
    zbar = (w.T @ z) / n[:, None]
    zc = z[:, None, :] - zbar[None, :, :]
    return np.einsum("rp,rpi,rpj->ij", w, zc, zc)


def _rank(a: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(a, tol=1e-9 * max(1.0, float(np.abs(a).max()))))


@dataclass
class FactorialResult:
    plan: Plan
    scores: np.ndarray  # [run, pair]
    effects: list[Effect]  # main effects (factor order), then the estimable interactions
    pairs: range
    games: int = field(default=0)

    @property
    def design(self) -> Design:
        return self.plan.design

    @property
    def edits(self) -> list[Edit]:
        return self.plan.edits

    def main(self) -> list[Effect]:
        return [e for e in self.effects if len(e.term) == 1]

    def interactions(self) -> list[Effect]:
        return [e for e in self.effects if len(e.term) == 2]

    def observations(self, deck_type: str) -> list[Observation]:
        """Each main effect as a single-edit observation (one copy in, one out) for ``CardValueModel``."""
        return [Observation(deck_type, (self.edits[e.term[0]].into,), (self.edits[e.term[0]].out,), e.effect, e.stderr)
                for e in self.main() if math.isfinite(e.effect) and e.stderr > 0]  # fmt: skip

    def predict(self, on: Sequence[int]) -> float:
        """Predicted difference to the parent (all −1) of applying the edits ``on`` (main effects and the estimated
        interaction chains, each chain attributed to its first pair)."""
        x = -np.ones(self.design.k)
        x[list(on)] = 1.0
        return float(sum(e.effect / 2 * (np.prod(x[list(e.term)]) - (-1) ** len(e.term)) for e in self.effects))

    def to_dict(self) -> dict:
        return {"design": self.design.to_dict(), "pairs": [self.pairs.start, self.pairs.stop], "games": self.games,
                "edits": [{**_edit_json(pl.edit), "letter": LETTERS[j], "index": pl.index}
                          for j, pl in enumerate(self.plan.placed)],
                "dropped": self.plan.dropped, "repairs": self.plan.repairs,
                "levels": self.plan.levels.astype(int).tolist(),
                "effects": [e.to_dict() for e in self.effects]}  # fmt: skip


def terms_of(d: Design) -> list[tuple[int, ...]]:
    """Main effects, then one column per estimable two-way interaction chain."""
    return [(j,) for j in range(d.k)] + [pair for pair, _ in d.interactions()]


def evaluate_edits(parent: Deck, edits: Sequence[Edit], evaluator, *, legal: Callable[[Deck], bool],
                   pairs: range | int = 200, p: int | None = None) -> FactorialResult:  # fmt: skip
    """Evaluate ``edits`` of ``parent`` as one fractional factorial (:func:`plan`): every variant plays ``pairs`` in
    one ``evaluator.play`` call (common random numbers; run 0 is the parent, so its pairs can come from an earlier
    diagnosis through a game log), then :func:`estimate` gives the main effects and the estimable interactions."""
    pairs = range(pairs) if isinstance(pairs, int) else pairs
    pl = plan(parent, edits, legal=legal, p=p)
    before = getattr(evaluator, "games", 0)
    out = evaluator.play([(deck, pairs) for deck in pl.decks])
    scores = np.stack([np.asarray(s, dtype=float) for s in out])
    terms = terms_of(pl.design)
    eff, se, keep = estimate(pl.levels, scores, terms)
    effects = [Effect(terms[t], pl.design.label(terms[t]), float(e), float(s))
               for t, e, s in zip(keep, eff, se, strict=True)]  # fmt: skip
    games = getattr(evaluator, "games", 0) - before
    return FactorialResult(pl, scores, effects, pairs, games if games > 0 else 2 * len(pl.decks) * len(pairs))


def one_at_a_time(parent: Deck, edits: Sequence[Edit], evaluator, *, pairs: range | int) -> list[tuple[float, float]]:
    """The comparison arm: the parent and each single edit (placed as in :func:`place`) on the same ``pairs``;
    per edit the mean paired difference and its standard error (per-pair sd / √n)."""
    pairs = range(pairs) if isinstance(pairs, int) else pairs
    placed, dropped = place(parent, edits)
    if dropped:
        raise ValueError(f"incompatible edits: {dropped}")
    out = evaluator.play([(parent, pairs), *((variant(parent, [pl]), pairs) for pl in placed)])
    base = np.asarray(out[0], dtype=float)
    res = []
    for s in out[1:]:
        d = np.asarray(s, dtype=float) - base
        d = d[np.isfinite(d)]
        res.append((float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.inf))
    return res


def games_to_detect(stderr: float, games: float, *, effect: float = 0.02, alpha: float = 0.025,
                    power: float = 0.8) -> float:  # fmt: skip
    """Games needed for a standard error at which a true ``effect`` is detected (one-sided ``alpha``) with ``power``,
    scaling the observed ``stderr`` at ``games`` games as 1 / √games."""
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return games * (stderr * z / effect) ** 2


def _edit_json(e: Edit) -> dict:
    return {"out": e.out, "into": e.into, "section": e.section}


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def summarize(result: FactorialResult, names: Mapping[int, str] | None = None) -> str:
    """One line per effect: label, edit (main effects), effect ± standard error."""
    lines = [f"2^({result.design.k}-{result.design.p}) design, {result.design.runs} runs, resolution "
             f"{result.design.to_dict()['resolution'] or 'full'}, {len(result.pairs)} pairs"]  # fmt: skip
    for e in result.effects:
        what = f" ({result.edits[e.term[0]].describe(names)})" if len(e.term) == 1 else ""
        lines.append(f"  {e.label}{what}: {e.effect:+.4f} ± {e.stderr:.4f}")
    return "\n".join(lines)


__all__ = ["GENERATORS", "LETTERS", "Design", "Effect", "FactorialResult", "Placed", "Plan", "default_p", "design",
           "estimate", "evaluate_edits", "games_to_detect", "one_at_a_time", "place", "plan", "summarize", "terms_of",
           "variant"]  # fmt: skip
