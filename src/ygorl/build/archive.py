"""MAP-Elites archive of decks (#111, design 05, eng-plan T5.8), on pyribs' :class:`ribs.archives.GridArchive`.

A deck lands in the grid cell of its five behaviour descriptors (design 05): win rate going first, win rate going
second, hand-trap count, combo length and brick rate. Each cell keeps one elite, the deck with the highest objective
(its win rate against the evolution's opponent mix). A deck is **admitted** when it fills an empty cell or beats the
cell's elite; a replaced elite is remembered by id (a later history opponent, #105 user story 28).

Two descriptors need solver runs for their real value (``ygorl.build.funnel``: 120 solver process seconds per deck),
so :func:`deck_descriptors` gives cheap proxies from the deck list and the synergy graph instead (docs/tuning.md
「进化步骤」):

- ``combo_length``: the deck's search-chain depth: the longest shortest path, in edges, among the deck's own cards
  along ``search`` / ``special_summon`` edges that take a card from the deck (fanout at most ``max_fanout``);
- ``brick_rate``: the chance that a 5-card opening hand holds no starter, a starter being a main-deck card with such
  an edge to another card of the deck (hypergeometric).

``hand_traps`` is exact (copies of cards whose generic role is ``hand_trap``); the two win rates come from games.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

FORMAT = "ygorl-deck-archive"
VERSION = 1
DESCRIPTORS = ("win_rate_first", "win_rate_second", "hand_traps", "combo_length", "brick_rate")
# descriptor -> (bins, (low, high)); values outside the range fall in the edge cells
DEFAULT_GRID: dict[str, tuple[int, tuple[float, float]]] = {
    "win_rate_first": (5, (0.0, 1.0)),
    "win_rate_second": (5, (0.0, 1.0)),
    "hand_traps": (5, (0.0, 15.0)),
    "combo_length": (4, (0.0, 8.0)),
    "brick_rate": (5, (0.0, 1.0)),
}
REACH = ("search", "special_summon")


@dataclass(frozen=True)
class Admission:
    """Outcome of :meth:`DeckArchive.add`: ``status`` is ``new`` (filled an empty cell), ``improved`` (beat the
    cell's elite, whose id is ``replaced``), ``rejected`` (the cell's elite is at least as good) or ``invalid`` (a
    descriptor or the objective is not finite)."""

    status: str
    cell: int | None = None
    replaced: str | None = None

    @property
    def admitted(self) -> bool:
        return self.status in ("new", "improved")


class DeckArchive:
    """One elite deck per descriptor cell. ``grid`` maps each descriptor to (bins, (low, high))."""

    def __init__(self, grid: Mapping[str, tuple[int, tuple[float, float]]] | None = None, *,
                 stamp: Mapping | None = None) -> None:  # fmt: skip
        from ribs.archives import GridArchive

        self.grid = {k: (int(b), (float(r[0]), float(r[1]))) for k, (b, r) in (grid or DEFAULT_GRID).items()}
        self.stamp = dict(stamp) if stamp else None
        self.names = tuple(self.grid)
        self._ribs = GridArchive(solution_dim=1, dims=[b for b, _ in self.grid.values()],
                                 ranges=[r for _, r in self.grid.values()])  # fmt: skip
        self._cells: dict[int, dict] = {}  # cell -> {"id", "deck", "objective", "descriptors"}
        self.replaced: list[str] = []

    def _measures(self, descriptors: Mapping[str, float]) -> np.ndarray:
        """Descriptor values in grid order, clipped into the grid (the edge cells take everything beyond)."""
        out = []
        for name, (_, (lo, hi)) in self.grid.items():
            v = float(descriptors[name])
            out.append(min(max(v, lo), hi - 1e-9 * max(1.0, hi - lo)))
        return np.array(out)

    def cell(self, descriptors: Mapping[str, float]) -> int:
        return int(self._ribs.index_of_single(self._measures(descriptors)))

    def add(self, deck_id: str, deck: Deck, objective: float, descriptors: Mapping[str, float]) -> Admission:
        values = [float(descriptors[n]) for n in self.names]
        if not math.isfinite(objective) or not all(math.isfinite(v) for v in values):
            return Admission("invalid")
        m = self._measures(descriptors)
        cell = int(self._ribs.index_of_single(m))
        status = int(
            self._ribs.add_single([float(len(self.replaced) + len(self._cells))], float(objective), m)["status"]
        )
        if status == 0:
            return Admission("rejected", cell)
        old = self._cells.get(cell)
        self._cells[cell] = {"id": deck_id, "deck": deck, "objective": float(objective),
                             "descriptors": dict(zip(self.names, values))}  # fmt: skip
        if old is not None:
            self.replaced.append(old["id"])
            return Admission("improved", cell, old["id"])
        return Admission("new", cell)

    def elite(self, cell: int) -> dict | None:
        return self._cells.get(cell)

    def elites(self) -> list[dict]:
        return [{"cell": c, **e} for c, e in sorted(self._cells.items())]

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> int:
        return int(self._ribs.cells)

    @property
    def coverage(self) -> float:
        return len(self._cells) / self.cells

    @property
    def qd_score(self) -> float:
        return float(sum(e["objective"] for e in self._cells.values()))

    def summary(self) -> dict:
        return {"elites": len(self), "cells": self.cells, "coverage": self.coverage, "qd_score": self.qd_score,
                "replaced": len(self.replaced)}  # fmt: skip

    def to_dict(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "environment": self.stamp,
                "grid": {k: [b, list(r)] for k, (b, r) in self.grid.items()},
                "elites": [{"id": e["id"], "main": list(e["deck"].main), "extra": list(e["deck"].extra),
                            "objective": e["objective"], "descriptors": e["descriptors"]} for e in self.elites()],
                "replaced": list(self.replaced)}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping) -> DeckArchive:
        if d.get("format") != FORMAT or d.get("version") != VERSION:
            raise ValueError(f"not a {FORMAT} v{VERSION} file")
        a = cls({k: (b, tuple(r)) for k, (b, r) in d["grid"].items()}, stamp=d.get("environment"))
        for e in d.get("elites", []):
            deck = Deck(main=tuple(e["main"]), extra=tuple(e["extra"]), name=e["id"])
            if not a.add(e["id"], deck, e["objective"], e["descriptors"]).admitted:
                raise ValueError(f"archive elite {e['id']} does not fit its own grid")
        a.replaced = list(d.get("replaced", []))
        return a


def _deck_edges(deck: Deck, graph, max_fanout: int) -> dict[int, set[int]]:
    cards = set(deck.main) | set(deck.extra)
    out: dict[int, set[int]] = {}
    for src in cards:
        for e in graph.out_edges(src, REACH):
            if e.dst in cards and e.dst != src and e.fanout <= max_fanout and e.locations & C.LOCATION_DECK:
                out.setdefault(src, set()).add(e.dst)
    return out


def deck_descriptors(deck: Deck, *, hand_traps: Iterable[int] = (), graph=None, hand: int = 5,
                     max_fanout: int = 30) -> dict[str, float]:  # fmt: skip
    """The deck-list descriptors: ``hand_traps`` (exact) and the proxies ``combo_length`` and ``brick_rate`` (module
    docstring). Without a graph there are no starters: ``combo_length`` 0 and ``brick_rate`` 1."""
    traps = set(hand_traps)
    edges = _deck_edges(deck, graph, max_fanout) if graph is not None else {}
    depth = 0
    for start in edges:
        seen, todo = {start: 0}, deque([start])
        while todo:
            c = todo.popleft()
            for n in edges.get(c, ()):
                if n not in seen:
                    seen[n] = seen[c] + 1
                    todo.append(n)
        depth = max(depth, max(seen.values()))
    n = len(deck.main)
    starters = sum(1 for c in deck.main if c in edges)
    k = min(hand, n)
    brick = math.comb(n - starters, k) / math.comb(n, k) if n else 1.0
    return {"hand_traps": float(sum(1 for c in deck.main if c in traps)), "combo_length": float(depth),
            "brick_rate": float(brick)}  # fmt: skip


__all__ = ["DEFAULT_GRID", "DESCRIPTORS", "Admission", "DeckArchive", "deck_descriptors"]
