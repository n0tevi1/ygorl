"""Engine-aware candidates for deck evolution (#145, docs/tuning.md「引擎感知的候选」).

The first evolution rounds accepted nothing: additions came from other engines (a card of another archetype does
nothing alone) and removals hit the parent's engine (with no evidence, the card-value model's draws are random).
This module says which of a deck's cards form its engine and which cards may come in:

- :func:`deck_engine`: the parent's **engine members**, its cards (generic cards excepted) with a synergy-graph edge
  to or from another card of the same deck, of any edge type (Endless Engine Argyro System and Dragon Shrine only
  *send* engine cards to the GY), with fanout at most ``max_fanout``. Searchers and starters count, not only the
  cards they fetch. Inside one deck the fanout limit can be wide (500 by default): Planet Pathfinder's "add 1 Field
  Spell" (fanout 332) or Lonefire Blossom's "Special Summon 1 Plant" (202) only land on the deck's own engine cards.
  :func:`ygorl.build.packages.score_package` on the members gives the starters.
- :func:`addition_pool`: the cards a child may put in: **generic cards** (a role in the generic pool: hand traps,
  board breakers, staples, generic Extra Deck monsters) and cards **connected** to the parent's non-generic cards
  (an edge either way, fanout at most ``max_fanout``, 30 by default: specific effects only), within the environment's
  card pool. Off-engine cards only come in as whole packages (the new-build mode, #113).
- :func:`archetypes`: the base setcodes (``setcode & 0xfff``) of a card set held by at least two distinct cards,
  the stratum key of the association rules when deck types have too few lists.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ygorl.build.packages import score_package
from ygorl.build.synergy_graph import EDGE_TYPES, SynergyGraph
from ygorl.cards.ydk import Deck

GENERIC_ROLES = ("hand_trap", "board_breaker", "staple", "extra")
ENGINE_FANOUT = 500  # edges inside one deck
ADDITION_FANOUT = 30  # edges from the deck to a card it lacks


def generic_roles(path: str | Path, roles: Iterable[str] = GENERIC_ROLES) -> dict[int, str]:
    """``password -> role`` of the generic pool file (``tests/data/generic_pool.json`` format) with a role in
    ``roles``."""
    keep = set(roles)
    cards = json.loads(Path(path).read_text())["cards"]
    return {int(c["password"]): c["role"] for c in cards if c.get("role") in keep}


@dataclass(frozen=True)
class Engine:
    """A deck's engine members, the starters among them (members nothing else in the engine reaches, greedily
    covering the engine) and the in-deck edges that made them members."""

    members: frozenset[int]
    starters: tuple[int, ...]
    edges: int

    def __contains__(self, card: int) -> bool:
        return card in self.members

    def __len__(self) -> int:
        return len(self.members)


def _cards(deck: Deck) -> set[int]:
    return set(deck.main) | set(deck.extra)


def deck_engine(deck: Deck, graph: SynergyGraph, *, generic: Iterable[int] = (), max_fanout: int = ENGINE_FANOUT,
                types: Iterable[str] = EDGE_TYPES) -> Engine:  # fmt: skip
    """The engine members of ``deck`` (module docstring): non-``generic`` cards with a ``types`` edge of fanout at most
    ``max_fanout`` to or from another non-generic card of ``deck``."""
    skip = set(generic)
    cards = _cards(deck) - skip
    types = tuple(types)
    members: set[int] = set()
    n = 0
    for c in cards:
        for e in graph.out_edges(c, types):
            if e.dst in cards and e.dst != c and e.fanout <= max_fanout:
                members |= {c, e.dst}
                n += 1
    starters: tuple[int, ...] = ()
    if members:
        starters = score_package(graph, members, reach_types=types).starters
    return Engine(frozenset(members), starters, n)


def connected(deck: Deck, graph: SynergyGraph, *, generic: Iterable[int] = (), max_fanout: int = ADDITION_FANOUT,
              types: Iterable[str] = EDGE_TYPES) -> set[int]:  # fmt: skip
    """Cards outside ``deck`` with a ``types`` edge of fanout at most ``max_fanout`` to or from one of its
    non-``generic`` cards."""
    have = _cards(deck)
    anchors = have - set(generic)
    types = tuple(types)
    out: set[int] = set()
    for c in anchors:
        out.update(e.dst for e in graph.out_edges(c, types) if e.fanout <= max_fanout)
        out.update(e.src for e in graph.in_edges(c, types) if e.fanout <= max_fanout)
    return out - have


def addition_pool(deck: Deck, graph: SynergyGraph, *, generic: Iterable[int] = (), candidates: Iterable[int] = (),
                  pool: Iterable[int] | None = None, max_fanout: int = ADDITION_FANOUT,
                  types: Iterable[str] = EDGE_TYPES) -> list[int]:  # fmt: skip
    """Cards a child of ``deck`` may put in (module docstring): ``candidates`` (e.g. ``tuner.tech_pool``) that are
    generic or connected first, in their order, then the other generic and connected cards by password; only cards of
    ``pool`` (the environment's card pool) when given. The deck's own cards stay allowed (another copy)."""
    gen = set(generic)
    link = connected(deck, graph, generic=gen, max_fanout=max_fanout, types=types)
    ok = gen | link | _cards(deck)
    allowed = set(pool) if pool is not None else None
    out: dict[int, None] = {}
    for c in (*candidates, *sorted(gen | link)):
        if c in ok and (allowed is None or c in allowed):
            out.setdefault(int(c))
    return list(out)


def archetypes(cards: Iterable[int], setcodes: Mapping[int, Iterable[int]], *, min_cards: int = 2) -> set[int]:
    """Base setcodes (``setcode & 0xfff``) that at least ``min_cards`` distinct ``cards`` carry."""
    count: dict[int, int] = {}
    for c in set(cards):
        for base in {sc & 0xFFF for sc in setcodes.get(c, ())}:
            count[base] = count.get(base, 0) + 1
    return {b for b, n in count.items() if n >= min_cards}


__all__ = ["ADDITION_FANOUT", "ENGINE_FANOUT", "GENERIC_ROLES", "Engine", "addition_pool", "archetypes", "connected",
           "deck_engine", "generic_roles"]  # fmt: skip
