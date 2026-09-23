"""Surrogate model of deck strength for the deck-building funnel (T5.7).

The second layer of the evaluation funnel (docs/design/05-deck-building.md,
06-architecture.md): a cheap regression from a deck's genotype to its expected
win rate against the meta pool and to the behaviour descriptors of the QD
search, updated online as real games come in (the DSA-ME loop). Pieces:

* :func:`card_feature_matrix` -- per-card structural features from the card
  database (card type, level buckets, attribute, race, spell / trap subtype,
  ATK / DEF, cdb search-category bits);
* :class:`TextEmbeddings` -- an optional ``[V, D]`` card-text embedding table
  aligned with :class:`~ygorl.cards.cdb.CardVocab` (the T5.2 artefact: ``.npy``
  + vocab JSON); absent -> the text feature is off;
* :class:`FeatureMap` -- genotype / deck / count vector -> feature vector, as named
  groups: ``counts`` (the genotype's count vector), ``packages`` (copies per
  engine package), ``structure`` (copy-weighted Main / Extra Deck composition
  from the card features, deck sizes), ``roles`` (copies per generic role, e.g.
  hand traps) and ``text`` (copy-weighted mean text embedding);
* :class:`RidgeEnsemble` -- multi-target ridge regression, penalty picked per
  target by generalised cross-validation, bagged into a bootstrap ensemble whose
  spread is the epistemic uncertainty used for acquisition; missing targets
  (NaN) and per-row weights (games played) are supported;
* :class:`Surrogate` -- the online model: labelled observations keyed by deck
  (repeated evaluations of one deck are merged, weighted by games), refit,
  predictions with uncertainty, QD descriptors (predicted ones plus exact ones
  such as the hand-trap count) and :meth:`Surrogate.select`;
* :func:`acquire` -- the acquisition rule: which candidates get real games.

Everything here is numpy; real-game labels come from :mod:`ygorl.build.labels`.
See docs/surrogate.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ygorl.build.genotype import Genotype, GenotypeSpace
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

DEFAULT_TARGETS = ("win_rate", "win_rate_first", "win_rate_second")
DEFAULT_DESCRIPTORS = ("win_rate_first", "win_rate_second", "hand_traps")
DEFAULT_GROUPS = ("counts", "packages", "structure", "roles")
GROUPS = (*DEFAULT_GROUPS, "text")

# ------------------------------------------------------------------ per-card features


def _flags(prefix: str, skip: tuple[str, ...]) -> list[tuple[str, int]]:
    """Single-bit ``prefix*`` constants as ``(lower-case name, bit)``, by bit."""
    flags = ((n[len(prefix) :].lower(), v) for n, v in vars(C).items() if n.startswith(prefix) and n not in skip)
    return sorted(((n, v) for n, v in flags if isinstance(v, int) and v > 0 and v & (v - 1) == 0), key=lambda nv: nv[1])


_ATTRIBUTES = _flags("ATTRIBUTE_", ("ATTRIBUTE_ALL",))
_RACES = _flags("RACE_", ("RACE_ALL", "RACE_MAX"))
_EXTRA = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK


def _card_row(card: Any) -> dict[str, float]:
    t = card.type
    mon, spell, trap = bool(t & C.TYPE_MONSTER), bool(t & C.TYPE_SPELL), bool(t & C.TYPE_TRAP)
    extra = mon and bool(t & _EXTRA)
    lv = card.level if mon else 0
    row = {
        "monster": mon, "spell": spell, "trap": trap,
        "normal_monster": mon and bool(t & C.TYPE_NORMAL), "tuner": mon and bool(t & C.TYPE_TUNER),
        "pendulum": mon and bool(t & C.TYPE_PENDULUM), "flip": mon and bool(t & C.TYPE_FLIP),
        "ritual_monster": mon and bool(t & C.TYPE_RITUAL), "nomi": mon and bool(t & C.TYPE_SPSUMMON),
        "level_1_4": mon and not extra and 1 <= lv <= 4, "level_5_6": mon and not extra and 5 <= lv <= 6,
        "level_7_plus": mon and not extra and lv >= 7,
        "fusion": mon and bool(t & C.TYPE_FUSION), "synchro": mon and bool(t & C.TYPE_SYNCHRO),
        "xyz": mon and bool(t & C.TYPE_XYZ), "link": mon and bool(t & C.TYPE_LINK),
        "rating": lv / 4.0,  # level / rank / link rating
        "attack": max(card.attack, 0) / 1000.0 if mon else 0.0,
        "defense": max(card.defense, 0) / 1000.0 if mon else 0.0,
        "normal_spell": spell and not t & (C.TYPE_QUICKPLAY | C.TYPE_CONTINUOUS | C.TYPE_FIELD | C.TYPE_EQUIP | C.TYPE_RITUAL),
        "quickplay": spell and bool(t & C.TYPE_QUICKPLAY), "continuous_spell": spell and bool(t & C.TYPE_CONTINUOUS),
        "field": spell and bool(t & C.TYPE_FIELD), "equip": spell and bool(t & C.TYPE_EQUIP),
        "ritual_spell": spell and bool(t & C.TYPE_RITUAL),
        "normal_trap": trap and not t & (C.TYPE_CONTINUOUS | C.TYPE_COUNTER),
        "continuous_trap": trap and bool(t & C.TYPE_CONTINUOUS), "counter_trap": trap and bool(t & C.TYPE_COUNTER),
    }  # fmt: skip
    for name, bit in _ATTRIBUTES:
        row[f"attribute_{name}"] = mon and bool(card.attribute & bit)
    for name, bit in _RACES:
        row[f"race_{name}"] = mon and bool(card.race & bit)
    return {k: float(v) for k, v in row.items()}


_CARD_FEATURES = tuple(_card_row(SimpleNamespace(type=0, level=0, attack=0, defense=0, attribute=0, race=0)))


def card_feature_matrix(passwords: Sequence[int], cards: Mapping[int, Any]) -> tuple[np.ndarray, list[str]]:
    """Structural features of each card: ``(matrix [len(passwords), F], feature names)``.

    Binary type / subtype / level-bucket / attribute / race flags, level-rank-link
    rating / 4, ATK / 1000, DEF / 1000, plus one flag per cdb ``category`` bit
    (the database's card-search categories) set on any of the cards. A card
    missing from ``cards`` gets an all-zero row.
    """
    found = [cards.get(int(p)) for p in passwords]
    names = list(_CARD_FEATURES)
    bits = sorted({b for c in found if c is not None for b in range(64) if c.category >> b & 1})
    names += [f"category_bit{b}" for b in bits]
    m = np.zeros((len(passwords), len(names)))
    for i, c in enumerate(found):
        if c is None:
            continue
        m[i, : len(_CARD_FEATURES)] = list(_card_row(c).values())
        m[i, len(_CARD_FEATURES) :] = [float(c.category >> b & 1) for b in bits]
    return m, names


# ------------------------------------------------------------------ text embeddings


class TextEmbeddings:
    """A card-text embedding table ``[len(vocab), D]`` whose rows follow :class:`CardVocab` indices.

    Rows of cards in ``missing`` (no text / placeholder rows) and cards the vocab
    does not know are left out of deck means. Alternate artworks resolve through
    the feature map's genotype space (it indexes originals).
    """

    def __init__(self, table: Any, vocab: CardVocab, missing: Iterable[int] = ()) -> None:
        table = np.asarray(table, dtype=np.float64)
        if table.ndim != 2 or table.shape[0] != len(vocab):
            raise ValueError(f"embedding table has {table.shape[0] if table.ndim else 0} rows, vocab needs {len(vocab)}")
        self.table, self.vocab, self.missing = table, vocab, frozenset(int(p) for p in missing)

    @classmethod
    def load(cls, table_path: str | Path, vocab_path: str | Path, missing: Iterable[int] = ()) -> TextEmbeddings:
        """Load the T5.2 artefact: an ``.npy`` table and the :meth:`CardVocab.save` JSON it is aligned with."""
        return cls(np.load(table_path), CardVocab.load(vocab_path), missing)

    @property
    def dim(self) -> int:
        return int(self.table.shape[1])

    def lookup(self, passwords: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """``(rows [len(passwords), D], covered mask)``; uncovered rows are zero."""
        out = np.zeros((len(passwords), self.dim))
        covered = np.zeros(len(passwords), dtype=bool)
        for i, p in enumerate(passwords):
            p = int(p)
            if p in self.vocab and p not in self.missing:
                out[i] = self.table[self.vocab.index(p)]
                covered[i] = True
        return out, covered

    def coverage(self, passwords: Sequence[int]) -> float:
        return float(self.lookup(passwords)[1].mean()) if len(passwords) else 1.0


# ------------------------------------------------------------------ feature map


Item = Genotype | Deck | np.ndarray


class FeatureMap:
    """Genotypes, decks or count vectors of one :class:`GenotypeSpace` -> feature matrix.

    ``groups`` picks feature groups from :data:`GROUPS` (default
    :data:`DEFAULT_GROUPS`, plus ``text`` when ``text`` is given); ``structure``
    needs ``cards``. ``groups`` maps each group to its column slice, ``names``
    names every column. Every feature is a function of the deck only (the count
    vector), so two genotypes that decode to the same deck get the same features.
    """

    def __init__(self, space: GenotypeSpace, cards: Mapping[int, Any] | None = None, *,
                 groups: Sequence[str] | None = None, text: TextEmbeddings | None = None) -> None:  # fmt: skip
        groups = list(DEFAULT_GROUPS if groups is None else groups)
        if text is not None and "text" not in groups:
            groups.append("text")
        unknown = [g for g in groups if g not in GROUPS]
        if unknown:
            raise ValueError(f"unknown feature groups {unknown}; expected a subset of {GROUPS}")
        if "structure" in groups and cards is None:
            raise ValueError("the structure features need the card database (cards=...)")
        if "text" in groups and text is None:
            raise ValueError("the text feature needs a TextEmbeddings table (text=...)")
        self.space, self.text = space, text
        v = len(space)
        main = ~space.is_extra
        self._main = main.astype(np.float64)
        names: list[str] = []
        self.groups: dict[str, slice] = {}

        def add(group: str, cols: list[str]) -> None:
            self.groups[group] = slice(len(names), len(names) + len(cols))
            names.extend(cols)

        for group in groups:
            if group == "counts":
                add(group, [f"count:{p}" for p in space.passwords])
            elif group == "packages":
                a = np.zeros((v, len(space.packages)))
                for p in range(len(space.packages)):
                    a[space.package_indices(p), p] = 1.0
                self._pkg = a
                add(group, [f"package:{p}" for p in range(len(space.packages))])
            elif group == "roles":
                roles = sorted(set(space.roles.values()))
                a = np.zeros((v, len(roles)))
                for i, r in space.roles.items():
                    a[i, roles.index(r)] = 1.0
                self._roles = a
                add(group, [f"role:{r}" for r in roles])
            elif group == "structure":
                f, fnames = card_feature_matrix(space.passwords, cards)
                keep_main = np.flatnonzero(np.abs(f[main]).sum(0) > 0)
                keep_extra = np.flatnonzero(np.abs(f[~main]).sum(0) > 0)
                self._fmain = f[:, keep_main] * main[:, None]
                self._fextra = f[:, keep_extra] * ~main[:, None]
                add(group, [f"main:{fnames[j]}" for j in keep_main] + [f"extra:{fnames[j]}" for j in keep_extra]
                    + ["main_size", "extra_size", "main_distinct"])  # fmt: skip
            elif group == "text":
                self._emb, covered = text.lookup(space.passwords)
                self._covered = covered.astype(np.float64)
                add(group, [f"text:{j}" for j in range(text.dim)])
        self.names: list[str] = names
        self.dim = len(names)

    # -- inputs ------------------------------------------------------------------
    def deck_counts(self, decks: Sequence[Deck]) -> np.ndarray:
        """Count vectors of decks (Main + Extra Deck, no repair; cards outside the space are ignored)."""
        out = np.zeros((len(decks), len(self.space)), dtype=np.int16)
        for n, deck in enumerate(decks):
            for pw in (*deck.main, *deck.extra):
                try:
                    out[n, self.space.index_of(pw)] += 1
                except KeyError:
                    continue
        return out

    def as_counts(self, items: Sequence[Item]) -> np.ndarray:
        """Count matrix ``[n, len(space)]`` of genotypes, decks or count vectors."""
        rows = []
        for it in items:
            if isinstance(it, Genotype):
                rows.append(it.counts.astype(np.int16))
            elif isinstance(it, Deck):
                rows.append(self.deck_counts([it])[0])
            else:
                rows.append(np.asarray(it, dtype=np.int16))
        if not rows:
            return np.zeros((0, len(self.space)), dtype=np.int16)
        out = np.stack(rows)
        if out.shape[1] != len(self.space):
            raise ValueError(f"count vectors have length {out.shape[1]}, the space has {len(self.space)} cards")
        return out

    # -- features ----------------------------------------------------------------
    def transform(self, counts: Any) -> np.ndarray:
        """Features of a count matrix ``[n, len(space)]``."""
        c = np.asarray(counts, dtype=np.float64)
        if c.ndim == 1:
            c = c[None, :]
        n = c.shape[0]
        out = np.empty((n, self.dim))
        main_size = c @ self._main
        extra_size = c.sum(1) - main_size
        for group, sl in self.groups.items():
            if group == "counts":
                block = c
            elif group == "packages":
                block = c @ self._pkg
            elif group == "roles":
                block = c @ self._roles
            elif group == "structure":
                block = np.hstack([
                    (c @ self._fmain) / np.maximum(main_size, 1)[:, None],
                    (c @ self._fextra) / np.maximum(extra_size, 1)[:, None],
                    np.stack([main_size, extra_size, ((c > 0) * self._main).sum(1)], axis=1),
                ])  # fmt: skip
            else:  # text
                w = c * self._covered
                block = (w @ self._emb) / np.maximum(w.sum(1), 1e-12)[:, None]
            out[:, sl] = block
        return out

    def __call__(self, items: Sequence[Item]) -> np.ndarray:
        return self.transform(self.as_counts(items))


# ------------------------------------------------------------------ ridge ensemble


class RidgeEnsemble:
    """Bootstrap ensemble of ridge regressions, one penalty per target chosen by GCV.

    Inputs are standardised (constant columns drop out). For each target, rows
    whose label is finite and weight positive are used; a target with fewer than
    ``min_rows`` such rows is not fitted and predicts NaN. The ridge penalty is
    picked from ``alphas`` (multiplied by the row count) by generalised
    cross-validation on all rows; ``residual_std_`` is the matching estimate of
    the residual standard deviation (label noise + misfit) at mean weight. Each
    of ``n_members`` members is fitted on a bootstrap resample (with
    ``bootstrap=False`` all members are identical and the spread is 0);
    :meth:`predict` returns the member mean and standard deviation.
    """

    def __init__(self, n_members: int = 16, alphas: Sequence[float] | None = None, *, bootstrap: bool = True,
                 min_rows: int = 10, seed: int = 0) -> None:  # fmt: skip
        self.n_members = int(n_members)
        self.alphas = np.asarray(np.logspace(-4, 2, 25) if alphas is None else alphas, dtype=np.float64)
        self.bootstrap = bootstrap
        self.min_rows = int(min_rows)
        self.seed = seed
        self._fitted = False

    @staticmethod
    def _centre(x: np.ndarray, y: np.ndarray, w: np.ndarray):
        sw = np.sqrt(w)
        xm = w @ x / w.sum()
        ym = float(w @ y / w.sum())
        return sw[:, None] * (x - xm), sw * (y - ym), xm, ym

    def _solve(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
        xw, yw, xm, ym = self._centre(x, y, w)
        n, d = xw.shape
        if d <= n:
            coef = np.linalg.solve(xw.T @ xw + alpha * np.eye(d), xw.T @ yw)
        else:
            coef = xw.T @ np.linalg.solve(xw @ xw.T + alpha * np.eye(n), yw)
        return coef, ym - float(xm @ coef)

    def _gcv(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
        xw, yw, _, _ = self._centre(x, y, w)
        n = len(yw)
        u, s, _ = np.linalg.svd(xw, full_matrices=False)
        uty = u.T @ yw
        base = float(yw @ yw - uty @ uty)  # part of y outside the column space
        best = (np.inf, float(self.alphas[0] * n), 0.0)
        for a in self.alphas * n:
            h = s**2 / (s**2 + a)
            rss = base + float(((1 - h) * uty) @ ((1 - h) * uty))
            dof = n - 1 - h.sum()  # -1: the intercept
            if dof <= 0:
                continue
            score = n * rss / dof**2
            if score < best[0]:
                best = (score, float(a), float(np.sqrt(rss / dof)))
        return best[1], best[2]

    def fit(self, x: Any, y: Any, weights: Any = None) -> RidgeEnsemble:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y = y[:, None] if y.ndim == 1 else y
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError(f"x has {len(x)} rows, y has {len(y)} rows")
        n, t = y.shape
        w = np.ones((n, t)) if weights is None else np.asarray(weights, dtype=np.float64)
        w = np.broadcast_to(w[:, None] if w.ndim == 1 else w, (n, t))
        self.mu_ = x.mean(0)
        sd = x.std(0)
        self._live = sd > 1e-12
        self.sd_ = np.where(self._live, sd, 1.0)
        xs = ((x - self.mu_) / self.sd_)[:, self._live]
        rng = np.random.default_rng(self.seed)
        self.coef_: list[np.ndarray | None] = []
        self.intercept_: list[np.ndarray | None] = []
        self.alpha_ = np.full(t, np.nan)
        self.residual_std_ = np.full(t, np.nan)
        for j in range(t):
            rows = np.flatnonzero(np.isfinite(y[:, j]) & (w[:, j] > 0))
            if len(rows) < max(self.min_rows, 2):
                self.coef_.append(None)
                self.intercept_.append(None)
                continue
            xj, yj, wj = xs[rows], y[rows, j], w[rows, j] / w[rows, j].mean()
            alpha, resid = self._gcv(xj, yj, wj)
            self.alpha_[j], self.residual_std_[j] = alpha, resid
            coefs, icepts = [], []
            for _ in range(self.n_members):
                idx = rng.integers(0, len(rows), len(rows)) if self.bootstrap else np.arange(len(rows))
                cf, ic = self._solve(xj[idx], yj[idx], wj[idx], alpha)
                coefs.append(cf)
                icepts.append(ic)
            self.coef_.append(np.stack(coefs, axis=1))  # [d_live, M]
            self.intercept_.append(np.asarray(icepts))
        self._fitted = True
        return self

    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble ``(mean, std)``, each ``[n, targets]``."""
        if not self._fitted:
            raise RuntimeError("RidgeEnsemble.predict called before fit")
        x = np.asarray(x, dtype=np.float64)
        xs = ((x - self.mu_) / self.sd_)[:, self._live]
        t = len(self.coef_)
        mean = np.full((len(x), t), np.nan)
        std = np.full((len(x), t), np.nan)
        for j, (cf, ic) in enumerate(zip(self.coef_, self.intercept_)):
            if cf is None:
                continue
            p = xs @ cf + ic  # [n, M]
            mean[:, j], std[:, j] = p.mean(1), p.std(1)
        return mean, std


# ------------------------------------------------------------------ acquisition


def acquire(mean: Any, std: Any, k: int, *, beta: float = 1.0, cells: Any = None, explore: float = 0.0,
            rng: np.random.Generator | None = None) -> np.ndarray:  # fmt: skip
    """Indices of the ``k`` candidates to evaluate for real, in priority order.

    Upper confidence bound ``mean + beta * std`` (NaN counts as worst). With
    ``cells`` (one archive cell id per candidate, e.g. the predicted descriptor
    bin), the best candidate of every cell comes before any cell's second one, so
    a batch spreads over the archive (DSA-ME evaluates the surrogate archive's
    elites). A share ``explore`` of the batch is instead given to the candidates
    with the largest ``std`` among those not already chosen (pure uncertainty
    sampling), taken after the exploitation picks. ``rng`` is unused (ties
    keep index order); it is accepted for interface symmetry with emitters.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if not 0.0 <= explore <= 1.0:
        raise ValueError(f"explore must be in [0, 1], got {explore}")
    n = len(mean)
    k = max(0, min(int(k), n))
    n_explore = int(round(k * explore))
    score = np.nan_to_num(mean + beta * std, nan=-np.inf)
    order = [int(i) for i in np.argsort(-score, kind="stable")]
    if cells is not None:
        cells = np.asarray(cells)
        seen: set = set()
        first, rest = [], []
        for i in order:
            key = cells[i].item() if hasattr(cells[i], "item") else cells[i]
            (rest if key in seen else first).append(i)
            seen.add(key)
        order = first + rest
    chosen = order[: k - n_explore]
    taken = set(chosen)
    by_std = [int(i) for i in np.argsort(-np.nan_to_num(std, nan=-np.inf), kind="stable") if int(i) not in taken]
    return np.asarray(chosen + by_std[:n_explore], dtype=np.int64)


# ------------------------------------------------------------------ online surrogate


def _hand_traps(space: GenotypeSpace) -> Callable[[np.ndarray], np.ndarray]:
    mask = np.zeros(len(space))
    for i, role in space.roles.items():
        if role == "hand_trap":
            mask[i] = 1.0
    return lambda counts: (np.asarray(counts, dtype=np.float64) @ mask).astype(np.int64)


EXACT_DESCRIPTORS: dict[str, Callable[[GenotypeSpace], Callable[[np.ndarray], np.ndarray]]] = {"hand_traps": _hand_traps}


@dataclass(frozen=True)
class Prediction:
    """Per-target ensemble mean (clipped to the target's bounds) and standard deviation."""

    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]


class Surrogate:
    """Online surrogate: labelled decks in, predictions with uncertainty out (DSA-ME).

    ``targets`` are the label names regressed (default :data:`DEFAULT_TARGETS`;
    later layers add e.g. ``combo_length`` / ``brick_rate`` from T5.6 -- a label
    may miss any target). ``descriptors`` are the QD descriptors reported by
    :meth:`descriptors`: each is a target (predicted) or a key of
    :data:`EXACT_DESCRIPTORS` (computed from the deck, e.g. ``hand_traps``).
    Targets whose name starts with ``win_rate`` are clipped to [0, 1];
    ``bounds`` overrides per target. ``model`` is a factory for the regressor
    (``fit(x, y, weights)`` / ``predict(x) -> (mean, std)``), default a
    :class:`RidgeEnsemble` with ``n_members`` and ``seed``.

    Observations are keyed by deck (count vector): adding a deck again merges the
    labels as a weighted mean (weights = games, default 1), so repeated
    evaluations of an elite sharpen its label instead of duplicating rows.
    """

    def __init__(self, features: FeatureMap, targets: Sequence[str] = DEFAULT_TARGETS, *,
                 descriptors: Sequence[str] = DEFAULT_DESCRIPTORS,
                 bounds: Mapping[str, tuple[float, float] | None] | None = None, n_members: int = 16,
                 seed: int = 0, model: Callable[[], Any] | None = None) -> None:  # fmt: skip
        self.features = features
        self.targets = tuple(targets)
        unknown = [d for d in descriptors if d not in self.targets and d not in EXACT_DESCRIPTORS]
        if unknown:
            raise ValueError(f"unknown descriptor(s) {unknown}: neither a target nor one of {sorted(EXACT_DESCRIPTORS)}")
        self.descriptor_names = tuple(descriptors)
        self._exact = {d: EXACT_DESCRIPTORS[d](features.space) for d in descriptors if d not in self.targets}
        b = {t: (0.0, 1.0) if t.startswith("win_rate") else None for t in self.targets}
        b.update(bounds or {})
        self.bounds = b
        self._model_factory = model or (lambda: RidgeEnsemble(n_members=n_members, seed=seed))
        self.model: Any = None
        self._index: dict[bytes, int] = {}
        self._counts: list[np.ndarray] = []
        self._sum = np.zeros((0, len(self.targets)))
        self._w = np.zeros((0, len(self.targets)))
        self._total = np.zeros(0)

    # -- data --------------------------------------------------------------------
    @property
    def n_observations(self) -> int:
        return len(self._counts)

    def add(self, items: Sequence[Item], labels: Sequence[Mapping[str, float]], weights: Sequence[float] | None = None) -> None:
        """Record labelled decks (genotypes, decks or count vectors); does not refit."""
        counts = self.features.as_counts(items)
        if len(labels) != len(counts) or (weights is not None and len(weights) != len(counts)):
            raise ValueError(f"{len(counts)} items, {len(labels)} labels, {None if weights is None else len(weights)} weights")
        new = {c.tobytes() for c in counts} - self._index.keys()  # a deck repeated in this call is one new row
        if new:
            t = len(self.targets)
            self._sum = np.vstack([self._sum, np.zeros((len(new), t))])
            self._w = np.vstack([self._w, np.zeros((len(new), t))])
            self._total = np.concatenate([self._total, np.zeros(len(new))])
        for c, lab, wt in zip(counts, labels, weights if weights is not None else [1.0] * len(counts)):
            key = c.tobytes()
            if key not in self._index:
                self._index[key] = len(self._counts)
                self._counts.append(c)
            i = self._index[key]
            self._total[i] += wt
            for j, name in enumerate(self.targets):
                v = lab.get(name)
                if v is not None and np.isfinite(v):
                    self._sum[i, j] += wt * float(v)
                    self._w[i, j] += wt

    def dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(features, labels, weights)`` of the merged observations; missing labels are NaN with weight 0."""
        x = self.features.transform(np.stack(self._counts)) if self._counts else np.zeros((0, self.features.dim))
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.where(self._w > 0, self._sum / np.where(self._w > 0, self._w, 1), np.nan)
        return x, y, self._w.copy()

    def observation(self, item: Item) -> dict[str, float]:
        """Merged labels of one deck (NaN for targets never labelled) and its total ``weight``."""
        key = self.features.as_counts([item])[0].tobytes()
        i = self._index[key]
        _, y, _ = self.dataset()
        return {**{t: float(y[i, j]) for j, t in enumerate(self.targets)}, "weight": float(self._total[i])}

    # -- model -------------------------------------------------------------------
    def fit(self) -> Surrogate:
        if not self._counts:
            raise RuntimeError("the surrogate has no labelled observations yet")
        x, y, w = self.dataset()
        self.model = self._model_factory().fit(x, y, w)
        return self

    def update(self, items: Sequence[Item], labels: Sequence[Mapping[str, float]],
               weights: Sequence[float] | None = None) -> Surrogate:  # fmt: skip
        """Online step: add the new labelled decks, then refit."""
        self.add(items, labels, weights)
        return self.fit()

    def predict(self, items: Sequence[Item]) -> Prediction:
        if self.model is None:
            raise RuntimeError("the surrogate has no labelled observations yet (call update first)")
        mean, std = self.model.predict(self.features(items))
        out_mean, out_std = {}, {}
        for j, t in enumerate(self.targets):
            m = mean[:, j]
            if self.bounds.get(t) is not None:
                lo, hi = self.bounds[t]
                m = np.clip(m, lo, hi)
            out_mean[t], out_std[t] = m, std[:, j]
        return Prediction(out_mean, out_std)

    def descriptors(self, items: Sequence[Item], prediction: Prediction | None = None) -> dict[str, np.ndarray]:
        """QD descriptors: predicted means for target descriptors, exact values for the rest."""
        counts = self.features.as_counts(items)
        need_pred = any(d in self.targets for d in self.descriptor_names)
        pred = prediction if prediction is not None or not need_pred else self.predict(counts)
        return {d: pred.mean[d] if d in self.targets else self._exact[d](counts) for d in self.descriptor_names}

    def select(self, candidates: Sequence[Item], k: int, *, objective: str = "win_rate", beta: float = 1.0,
               cells: Any = None, explore: float = 0.0) -> np.ndarray:  # fmt: skip
        """Which ``candidates`` to evaluate for real: :func:`acquire` on the ``objective`` target."""
        pred = self.predict(candidates)
        return acquire(pred.mean[objective], pred.std[objective], k, beta=beta, cells=cells, explore=explore)
