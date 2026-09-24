"""Deck corpus: several legal lists per deck type from the whole masterduelmeta history (docs/data.md「牌组语料」).

The environment's ``meta/`` holds one representative list for each type with a share in the current window. For
training on many decks (T4d.1) and as starting points for deck tuning, the corpus keeps every deck type that has a
list legal in the environment -- ranked, tournament and event lists alike, so rogue and fun decks are in -- with
up to ``per_type`` lists each: the type's medoid, then the lists farthest (by card-count L1 distance) from the ones
already chosen, skipping near duplicates. Lists are real lists, never averaged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ygorl.data.masterduelmeta import DeckRecord


@dataclass(frozen=True)
class CorpusDeck:
    type: str
    role: str  # "medoid" or "diverse"
    record: DeckRecord
    legal_lists: int  # legal lists of this type in the history
    distance: int  # card-count L1 distance to the nearest list chosen before it (0 for the medoid)


def _vectors(records: Sequence[DeckRecord]) -> np.ndarray:
    index: dict[int, int] = {}
    rows = []
    for r in records:
        c = r.counts()
        rows.append({index.setdefault(pw, len(index)): n for pw, n in c.items()})
    out = np.zeros((len(records), len(index)), dtype=np.int16)
    for i, row in enumerate(rows):
        for j, n in row.items():
            out[i, j] = n
    return out


def _pairwise_l1(v: np.ndarray) -> np.ndarray:
    return np.abs(v[:, None, :].astype(np.int32) - v[None, :, :]).sum(-1)


def select_lists(records: Iterable[DeckRecord], validate: Callable[[DeckRecord], list[str]], *, per_type: int = 3,
                 min_distance: int = 8, max_candidates: int = 400) -> list[CorpusDeck]:  # fmt: skip
    """Up to ``per_type`` legal lists per deck type: the medoid of the type's newest ``max_candidates`` distinct legal
    lists, then farthest-point picks at least ``min_distance`` cards away from every list already chosen."""
    by_type: dict[str, list[DeckRecord]] = defaultdict(list)
    for r in records:
        by_type[r.type].append(r)
    out: list[CorpusDeck] = []
    for name in sorted(by_type):
        seen, legal = set(), []
        for r in sorted(by_type[name], key=lambda r: r.created, reverse=True):
            key = (tuple(sorted(r.main)), tuple(sorted(r.extra)))
            if key in seen or r.unmapped or validate(r):
                continue
            seen.add(key)
            legal.append(r)
        if not legal:
            continue
        cand = legal[:max_candidates]
        d = _pairwise_l1(_vectors(cand))
        first = int(np.argmin(d.sum(1)))  # ties: the newest (lowest index)
        chosen = [first]
        out.append(CorpusDeck(name, "medoid", cand[first], len(legal), 0))
        nearest = d[first].copy()
        while len(chosen) < per_type:
            nxt = int(np.argmax(nearest))
            if nearest[nxt] < min_distance:
                break
            chosen.append(nxt)
            out.append(CorpusDeck(name, "diverse", cand[nxt], len(legal), int(nearest[nxt])))
            nearest = np.minimum(nearest, d[nxt])
    return out


__all__ = ["CorpusDeck", "select_lists"]
