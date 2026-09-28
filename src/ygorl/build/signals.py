"""Signal library for deck evolution (#109, docs/spikes/deck-evolution.md §3 and 「测量结果」).

M3 showed that no cheap per-card signal (opening-hand effect, critic opening value, dead / idle / use rate, advantage
after use) predicts a card's leave-one-out value well enough to rank cards. So the per-card values here come from the
evaluations themselves, and the cheap signals only enter as a calibrated prior:

- :class:`CardValueModel`: every paired evaluation of a child against its parent (``diff`` = child minus parent, with
  its standard error) is a linear observation of the cards swapped in minus the cards swapped out. Each card's value
  in a deck type is a card effect plus a card-by-type effect, both shrunk toward a prior (Gaussian, solved as one
  weighted ridge): with no data a value is its prior, with a lot of data it approaches what the evaluations say, and
  a card measured in one type informs the same card in another through the shared card effect.
- :class:`Calibration`: per signal, the (predicted, measured) pairs seen so far; a signal's weight is its Spearman
  correlation with the measurements once there are enough of them and the correlation's 95% interval excludes 0,
  otherwise its default weight, which is 0 for every signal that failed M3.
- :func:`informed_children`: children of a parent deck. An informed child is a bundle of 1 to ``max_bundle`` edits
  in the same direction (each takes out a card the model rates low and puts in one it rates higher), chosen on a
  Thompson draw from the model so repeated calls spread over the plausible edits; explore children are random legal
  swaps. Protected cards (win conditions, searched targets) are never taken out.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ygorl import paths
from ygorl.build.tuner import Edit, apply, neighbors
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C


@dataclass(frozen=True)
class Observation:
    """A measured paired difference: the child (``into`` swapped in for ``out``, one copy each) minus its parent."""

    deck_type: str
    into: tuple[int, ...]
    out: tuple[int, ...]
    diff: float
    stderr: float


class CardValueModel:
    """value(card, type) = card effect + card-by-type effect; priors N(prior[card], card_scale²) and
    N(0, type_scale²); observations weighted by 1 / stderr²."""

    def __init__(self, *, card_scale: float = 0.03, type_scale: float = 0.02,
                 prior: Mapping[int, float] | None = None) -> None:  # fmt: skip
        if card_scale <= 0 or type_scale <= 0:
            raise ValueError("the prior scales must be positive")
        self.card_scale, self.type_scale = card_scale, type_scale
        self.prior = dict(prior or {})
        self.observations: list[Observation] = []
        self._fit: tuple[dict, np.ndarray, np.ndarray] | None = None

    def add(self, obs: Observation) -> None:
        if not obs.stderr > 0:
            raise ValueError("an observation needs a positive standard error")
        if len(obs.into) != len(obs.out):
            raise ValueError("an observation swaps as many copies in as out")
        self.observations.append(obs)
        self._fit = None

    def set_prior(self, prior: Mapping[int, float]) -> None:
        self.prior = dict(prior)
        self._fit = None

    def _solve(self) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """(term index, posterior mean, posterior covariance, inverse Cholesky factor of the precision), cached until
        the next ``add`` / ``set_prior``. Precision = diag(prior precision) + Xᵀ W X, built from the stacked rows."""
        if self._fit is not None:
            return self._fit
        index: dict[tuple, int] = {}
        for o in self.observations:
            for c in (*o.into, *o.out):
                index.setdefault(("card", c), len(index))
                index.setdefault(("type", c, o.deck_type), len(index))
        n = len(index)
        if n == 0:
            self._fit = (index, np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0)))
            return self._fit
        prec = np.zeros(n)
        mu = np.zeros(n)
        for key, i in index.items():
            if key[0] == "card":
                prec[i], mu[i] = 1 / self.card_scale**2, self.prior.get(key[1], 0.0)
            else:
                prec[i] = 1 / self.type_scale**2
        x = np.zeros((len(self.observations), n))
        for r, o in enumerate(self.observations):
            for sign, cards in ((1.0, o.into), (-1.0, o.out)):
                for c in cards:
                    x[r, index[("card", c)]] += sign
                    x[r, index[("type", c, o.deck_type)]] += sign
        w = np.array([1 / o.stderr**2 for o in self.observations])
        y = np.array([o.diff for o in self.observations])
        a = np.diag(prec) + (x * w[:, None]).T @ x
        linv = np.linalg.inv(np.linalg.cholesky(a))  # A = L Lᵀ, so A⁻¹ = L⁻ᵀ L⁻¹
        cov = linv.T @ linv
        self._fit = (index, cov @ (prec * mu + x.T @ (w * y)), cov, linv)
        return self._fit

    def _terms(self, card: int, deck_type: str) -> list[tuple[tuple, float, float]]:
        """(term key, prior mean, prior scale) of one copy of ``card`` in ``deck_type``."""
        return [(("card", card), self.prior.get(card, 0.0), self.card_scale),
                (("type", card, deck_type), 0.0, self.type_scale)]  # fmt: skip

    def gain(self, into: Sequence[int], out: Sequence[int], deck_type: str) -> tuple[float, float]:
        """Posterior mean and standard deviation of swapping ``out`` for ``into`` (one copy each) in ``deck_type``,
        from the joint posterior: evaluations pin down differences between cards much better than single values
        (only differences are observed), so this is the quantity to rank edits by. Terms never observed keep their
        independent priors (their coefficients summed first, so a card both in and out cancels)."""
        index, mean, cov, _ = self._solve()
        x = np.zeros(len(mean))
        free: dict[tuple, list[float]] = {}  # unseen term -> [coefficient, prior mean, prior scale]
        for sign, cards in ((1.0, into), (-1.0, out)):
            for c in cards:
                for key, prior_mean, scale in self._terms(c, deck_type):
                    i = index.get(key)
                    if i is None:
                        free.setdefault(key, [0.0, prior_mean, scale])[0] += sign
                    else:
                        x[i] += sign
        m = float(x @ mean) + sum(k * pm for k, pm, _ in free.values())
        var = float(x @ cov @ x) + sum((k * sd) ** 2 for k, _, sd in free.values())
        return m, float(math.sqrt(max(var, 0.0)))

    def value(self, card: int, deck_type: str) -> tuple[float, float]:
        """Posterior mean and standard deviation of one copy of ``card`` in a deck of ``deck_type`` (absolute values
        are only as sure as the prior; compare cards with :meth:`gain`)."""
        return self.gain([card], [], deck_type)

    def sample(self, cards: Iterable[int], deck_type: str, rng: np.random.Generator) -> dict[int, float]:
        """One joint Thompson draw of the values of ``cards``: mean + L⁻ᵀ z for the observed terms, each unseen term
        drawn once from its prior."""
        index, mean, _, linv = self._solve()
        theta = mean + linv.T @ rng.standard_normal(len(mean))
        free: dict[tuple, float] = {}
        out = {}
        for c in dict.fromkeys(cards):
            v = 0.0
            for key, prior_mean, scale in self._terms(c, deck_type):
                i = index.get(key)
                if i is not None:
                    v += theta[i]
                else:
                    if key not in free:
                        free[key] = rng.normal(prior_mean, scale)
                    v += free[key]
            out[c] = float(v)
        return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation with average ranks for ties (0 when either side is constant)."""

    def ranks(x):
        x = np.asarray(x, dtype=float)
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        for v in np.unique(x):
            r[x == v] = r[x == v].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


class Calibration:
    """Signal -> measured gain. ``defaults`` are the weights before there is evidence (M3: 0 for every signal that
    failed the gate); after ``min_pairs`` pairs a signal's weight is its Spearman correlation with the measurements
    when the 95% interval (Fisher z) excludes 0, else 0 (switched off)."""

    def __init__(self, defaults: Mapping[str, float] | None = None, *, min_pairs: int = 20) -> None:
        self.defaults = dict(defaults or {})
        self.min_pairs = min_pairs
        self.pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def record(self, signal: str, predicted: float, measured: float) -> None:
        if math.isfinite(predicted) and math.isfinite(measured):
            self.pairs[signal].append((float(predicted), float(measured)))

    def correlation(self, signal: str) -> tuple[float, float, float]:
        """Spearman r and its 95% interval (Fisher z; nan with fewer than 4 pairs)."""
        p = self.pairs.get(signal, [])
        if len(p) < 4:
            return math.nan, math.nan, math.nan
        r = spearman([x for x, _ in p], [y for _, y in p])
        # Fieller et al. (1957): the Fisher-z standard error of a Spearman correlation is sqrt(1.06 / (n - 3))
        z, h = math.atanh(max(min(r, 0.999999), -0.999999)), 1.96 * math.sqrt(1.06 / (len(p) - 3))
        return r, math.tanh(z - h), math.tanh(z + h)

    def weight(self, signal: str) -> float:
        if len(self.pairs.get(signal, [])) < self.min_pairs:
            return self.defaults.get(signal, 0.0)
        r, lo, _ = self.correlation(signal)
        return r if lo > 0 else 0.0

    def enabled(self, signal: str) -> bool:
        return self.weight(signal) > 0

    def report(self) -> dict[str, dict]:
        out = {}
        for s in sorted(set(self.defaults) | set(self.pairs)):
            r, lo, hi = self.correlation(s)
            out[s] = {"pairs": len(self.pairs.get(s, [])), "spearman": r, "ci": (lo, hi), "weight": self.weight(s)}
        return out


def combine_prior(signals: Mapping[str, Mapping[int, float]], calibration: Calibration, *,
                  scale: float = 0.01) -> dict[int, float]:  # fmt: skip
    """A prior value per card: the calibrated weights times each signal's z-score over the cards it covers, times
    ``scale`` (a card's typical value spread). Signals with weight 0 contribute nothing."""
    prior: dict[int, float] = defaultdict(float)
    for name, values in signals.items():
        w = calibration.weight(name)
        vals = {c: v for c, v in values.items() if math.isfinite(v)}
        if w <= 0 or len(vals) < 2:
            continue
        x = np.array(list(vals.values()))
        sd = x.std()
        if sd == 0:
            continue
        for c, v in vals.items():
            prior[c] += w * scale * (v - x.mean()) / sd
    return dict(prior)


@dataclass(frozen=True)
class Child:
    edits: tuple[Edit, ...]
    deck: Deck
    kind: str  # "informed" or "explore"
    predicted: float  # the model's mean gain of the edits (0 for explore children)


def informed_children(base: Deck, deck_type: str, model: CardValueModel, pool: Iterable[int], *,
                      legal: Callable[[Deck], bool], is_extra: Callable[[int], bool], rng: np.random.Generator,
                      informed: int = 6, explore: int = 2, max_bundle: int = 3,
                      protected: Iterable[int] = ()) -> list[Child]:  # fmt: skip
    """Up to ``informed`` bundled children and ``explore`` random-swap children of ``base``, all legal and distinct.
    Each informed child draws the card values once (Thompson), then pairs the lowest-drawn cards of the deck with
    the highest-drawn candidates of the same section, keeping only pairs with a positive drawn gain and a legal
    result, until the bundle (1..``max_bundle``, drawn) is full; a bundle never takes out a card it put in or puts
    back one it took out. May return fewer children when the deck has few legal improving edits."""
    pool = [c for c in dict.fromkeys(pool)]
    protected = set(protected)
    seen: set[tuple] = {_key(base)}
    out: list[Child] = []
    for _ in range(informed * 4):
        if sum(c.kind == "informed" for c in out) >= informed:
            break
        draw = model.sample([*base.main, *base.extra, *pool], deck_type, rng)
        size = int(rng.integers(1, max_bundle + 1))
        deck, edits = base, []
        outs = sorted(
            (c for c in dict.fromkeys((*base.main, *base.extra)) if c not in protected), key=lambda c: draw[c]
        )
        for o in outs:
            if len(edits) >= size:
                break
            section = "extra" if o in base.extra else "main"
            ins = sorted((c for c in pool if is_extra(c) == (section == "extra") and draw[c] > draw[o] and c != o),
                         key=lambda c: -draw[c])  # fmt: skip
            for i in ins:
                e = Edit(o, i, section)
                cand = apply(deck, e)
                if legal(cand):
                    deck, edits = cand, [*edits, e]
                    break
        if edits and _key(deck) not in seen:
            seen.add(_key(deck))
            gain = model.gain([e.into for e in edits], [e.out for e in edits], deck_type)[0]
            out.append(Child(tuple(edits), deck, "informed", gain))
    for e, deck in neighbors(base, pool, is_extra, legal, rng=rng, limit=explore * 20):
        if sum(c.kind == "explore" for c in out) >= explore:
            break
        if e.out in protected or _key(deck) in seen:
            continue
        seen.add(_key(deck))
        out.append(Child((e,), deck, "explore", 0.0))
    return out


def protected_cards(deck: Deck, graph=None, *, scripts_dir=None, max_fanout: int = 30) -> set[int]:
    """Cards of ``deck`` that per-card signals misjudge, so evolution never takes them out on a signal's word:

    - **win conditions**: cards whose script calls ``Duel.Win`` and the deck's cards that script names (Exodia and its
      pieces: dead in hand, yet the deck's reason to exist);
    - **search targets**: cards another card of the deck brings from the deck to the hand or field (a ``search`` /
      ``special_summon`` edge of the synergy graph with fanout at most ``max_fanout``), whose "in hand" statistics
      measure the searcher, not the card.
    """
    root = Path(scripts_dir) if scripts_dir is not None else paths.card_scripts() / "official"
    named = _card_constants(root.parent)
    cards = set(deck.main) | set(deck.extra)
    out: set[int] = set()
    for pw in cards:
        f = root / f"c{pw}.lua"
        if f.is_file():
            text = f.read_text(errors="replace")
            if "Duel.Win(" in text:
                out.add(pw)
                ids = {int(n) for n in re.findall(r"\b\d{5,9}\b", text)}
                for name in re.findall(r"\bCARDS?_[A-Z0-9_]+\b", text):  # named constants, e.g. CARDS_SPIRIT_MESSAGE
                    ids.update(named.get(name, ()))
                out.update(ids & cards)
    if graph is not None:
        for src in cards:
            for e in graph.out_edges(src, ("search", "special_summon")):
                if e.dst in cards and e.dst != src and e.fanout <= max_fanout and e.locations & C.LOCATION_DECK:
                    out.add(e.dst)
    return out


def _card_constants(scripts_root: Path) -> dict[str, tuple[int, ...]]:
    """``CARD_X = 123`` / ``CARDS_X = {1, 2}`` from the CardScripts constant files."""
    out: dict[str, tuple[int, ...]] = {}
    for f in ("card_counter_constants.lua", "constant.lua"):
        path = scripts_root / f
        if not path.is_file():
            continue
        for name, rhs in re.findall(r"^\s*(CARDS?_[A-Z0-9_]+)\s*=\s*([^\n]+)", path.read_text(errors="replace"), re.M):
            ids = tuple(int(n) for n in re.findall(r"\b\d{5,9}\b", rhs.split("--")[0]))
            if ids:
                out[name] = ids
    return out


def _key(deck: Deck) -> tuple:
    return tuple(sorted(deck.main)), tuple(sorted(deck.extra))


__all__ = ["Calibration", "CardValueModel", "Child", "Observation", "combine_prior", "informed_children", "protected_cards",
           "spearman"]  # fmt: skip
