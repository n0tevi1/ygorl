"""Crossover of MAP-Elites archive elites as a child generator (#141, spec #105; docs/tuning.md「交叉子代」).

The evolution step's children are one-card edits of the round's parent (``ygorl.build.signals.informed_children``);
:func:`crossover_children` adds children that take a block of cards from a second deck, an archive elite, so two play
styles meet in one list. It works on deck lists, not on genotypes: the archive holds deck lists (corpus decks and
their evolved children), and :meth:`ygorl.build.genotype.GenotypeSpace.from_deck` would drop every card outside the
space's packages and generic pool.

- **Mates**: the first mate is the round's parent, which is also the reference of the paired games; the second is an
  archive elite with another list, drawn with probability proportional to its **cell distance** from the parent (L1
  distance in grid bins over the descriptors both have: the parent's own archive entry when it is an elite, else its
  deck-list descriptors), so elites of far-away cells, other play styles, are preferred. Two distinct elites always
  sit in different cells (one elite per cell); when every candidate is at distance 0 the draw is uniform. An archive
  with fewer than 2 elites proposes nothing.
- **Crossing** (:func:`cross_decks`): per section, the copies the mate has and the parent lacks go in, in whole-card
  groups (all missing copies of a card together) in random order, and as many copies the parent has and the mate
  lacks come out (never a protected card), in random whole-card groups too. The child takes ``k`` such swaps,
  1 ≤ ``k`` ≤ half the swaps that would turn the parent into the mate, so it is never farther from the parent than
  from the mate: the parent is the elite it is closer to, and the paired-game reference. Each swap is an in-place
  :class:`ygorl.build.tuner.Edit` (every other card keeps its position: common random numbers with the parent) and
  **repair** keeps only the swaps whose result is legal, so a legal parent always gives a legal child.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ygorl.build.signals import Child
from ygorl.build.tuner import Edit, apply
from ygorl.cards.ydk import Deck

GENERATOR = "crossover"


@dataclass(frozen=True)
class Mate:
    """The second parent of a crossover child: an archive elite."""

    id: str
    deck: Deck
    cell: int
    distance: float  # cell distance from the round's parent

    def to_dict(self) -> dict:
        return {"id": self.id, "cell": self.cell, "distance": self.distance}


def _counts(deck: Deck) -> tuple:
    return tuple(sorted(Counter(deck.main).items())), tuple(sorted(Counter(deck.extra).items()))


def _bins(grid: Mapping[str, tuple[int, tuple[float, float]]], descriptors: Mapping[str, float | None]) -> dict:
    """Grid bin per descriptor with a finite value (values beyond the range fall in the edge bins, as in the archive)."""
    out = {}
    for name, (bins, (lo, hi)) in grid.items():
        v = descriptors.get(name)
        if v is None or not math.isfinite(float(v)):
            continue
        out[name] = min(max(int((float(v) - lo) / (hi - lo) * bins), 0), bins - 1)
    return out


def cell_distance(grid: Mapping[str, tuple[int, tuple[float, float]]], a: Mapping[str, float | None],
                  b: Mapping[str, float | None]) -> float:  # fmt: skip
    """L1 distance in grid bins between two descriptor sets, over the descriptors both have."""
    x, y = _bins(grid, a), _bins(grid, b)
    return float(sum(abs(x[k] - y[k]) for k in x.keys() & y.keys()))


def swaps(base: Deck, mate: Deck, protected: Iterable[int] = ()) -> int:
    """Swaps (one copy out, one in, same section) available to move ``base`` toward ``mate``."""
    keep = set(protected)
    n = 0
    for section in ("main", "extra"):
        b, m = Counter(getattr(base, section)), Counter(getattr(mate, section))
        n += min(sum((m - b).values()), sum(v for c, v in (b - m).items() if c not in keep))
    return n


def cross_decks(base: Deck, mate: Deck, rng: np.random.Generator, *, legal: Callable[[Deck], bool],
                protected: Iterable[int] = ()) -> tuple[tuple[Edit, ...], Deck] | None:  # fmt: skip
    """A child of ``base`` with a block of ``mate``'s cards (module docstring): its edits and deck, or None when
    ``mate`` is less than two swaps away or no swap is legal."""
    keep = set(protected)
    total = swaps(base, mate, keep)
    if total < 2:
        return None
    k = int(rng.integers(1, total // 2 + 1))
    groups, outs = [], {}
    for section in ("main", "extra"):
        b, m = Counter(getattr(base, section)), Counter(getattr(mate, section))
        groups += [(section, c, n) for c, n in sorted((m - b).items())]
        lose = [(c, n) for c, n in sorted((b - m).items()) if c not in keep]
        outs[section] = [c for j in rng.permutation(len(lose)) for c in [lose[j][0]] * lose[j][1]]
    room = {s: min(len(outs[s]), sum(n for t, _, n in groups if t == s)) for s in outs}
    into = {"main": [], "extra": []}
    for j in rng.permutation(len(groups)):
        section, card, n = groups[j]
        take = min(n, k - len(into["main"]) - len(into["extra"]), room[section] - len(into[section]))
        into[section] += [card] * max(take, 0)
    deck, edits = base, []
    for section in ("main", "extra"):
        for o, i in zip(outs[section], into[section]):  # outs is at least as long: room
            e = Edit(o, i, section)
            cand = apply(deck, e)
            if legal(cand):  # repair: drop a swap that would make the deck illegal
                deck, edits = cand, [*edits, e]
    return (tuple(edits), deck) if edits else None


def mate_distances(archive, base: Deck, base_descriptors: Mapping[str, float | None]) -> list[tuple[dict, float]]:
    """Archive elites other than ``base``'s list with their cell distance from ``base`` (the draw weight). ``base``'s
    descriptors are its archive entry's when it is an elite, else ``base_descriptors``."""
    key = _counts(base)
    own = next((e for e in archive.elites() if _counts(e["deck"]) == key), None)
    where = own["descriptors"] if own is not None else base_descriptors
    others = [e for e in archive.elites() if _counts(e["deck"]) != key]
    return [(e, cell_distance(archive.grid, where, e["descriptors"])) for e in others]


def crossover_children(base: Deck, archive, *, descriptors: Mapping[str, float | None], legal: Callable[[Deck], bool],
                       rng: np.random.Generator, n: int, protected: Iterable[int] = (),
                       exclude: Iterable[Deck] = ()) -> list[tuple[Child, Mate]]:  # fmt: skip
    """Up to ``n`` distinct legal crossover children of ``base`` and an archive elite each (module docstring), none
    with the list of ``base``, of an ``exclude`` deck or of its mate. ``descriptors``: ``base``'s descriptors (used
    when it is not an elite). Nothing when the archive holds fewer than 2 elites."""
    if n <= 0 or len(archive) < 2:
        return []
    keep = set(protected)
    cands = [(e, d) for e, d in mate_distances(archive, base, descriptors) if swaps(base, e["deck"], keep) >= 2]
    if not cands:
        return []
    w = np.array([d for _, d in cands], dtype=float)
    if not w.sum():  # every candidate in the parent's cell (on the descriptors both have): uniform
        w[:] = 1.0
    seen = {_counts(base), *(_counts(d) for d in exclude)}
    out: list[tuple[Child, Mate]] = []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        e, d = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        made = cross_decks(base, e["deck"], rng, legal=legal, protected=keep)
        if made is None or _counts(made[1]) in seen or _counts(made[1]) == _counts(e["deck"]):
            continue
        seen.add(_counts(made[1]))
        out.append((Child(made[0], made[1], GENERATOR, 0.0), Mate(e["id"], e["deck"], int(e["cell"]), d)))
    return out


__all__ = ["GENERATOR", "Mate", "cell_distance", "cross_decks", "crossover_children", "mate_distances", "swaps"]
