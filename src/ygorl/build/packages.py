"""Engine-package enumeration on the synergy graph (T5.4).

An *engine package* is a small connected set of cards that fetch each other:
starters that, once drawn, reach the rest through ``search`` / ``special_summon``
edges. Packages are found by greedy local growth from every seed card and
ranked by *reachability density* rather than by meta popularity, so packages
that share a race / level / named card across archetypes (rogue seeds) rank
alongside archetype cores (design doc 5.2).

Growth (:func:`grow_package`). Edges are weighted by specificity,
``w = TYPE_WEIGHT[type] / sqrt(fanout)`` (a named-card search weighs 1, an
archetype search matching 16 cards 0.25), and summed in both directions into an
undirected affinity. From the seed, repeatedly add the candidate ``v`` with the
largest ``aff(v, P)**2 / deg(v)`` -- strongly tied to the package *and* with most
of its own weight inside it -- as long as ``aff(v, P) >= min_affinity`` and
``aff(v, P) / deg(v) >= min_share``; generic hubs (searchable by everything)
fail the share test. Every package is therefore weakly connected.

Scoring (:func:`score_package`), on the subgraph induced by the package with
``reach_types`` edges only:

* ``sources(v)`` = members other than ``v`` from which ``v`` is reachable;
* ``starters`` = a greedy minimum set of members whose reach closures cover the
  package (members nothing reaches are always starters);
* ``reach_mass`` = sum over members of ``min(sources(v), k) / k`` (a member
  reachable along ``k`` distinct sources counts fully -- "k paths");
* ``score`` = ``reach_mass / len(starters)``: robustly reachable members per
  starter, the ranking key;
* ``density`` = share of ordered member pairs ``(u, v)`` with ``v`` reachable from ``u``
  (tie-breaker: ``score`` saturates at the package size when one starter reaches all).

Cross-archetype packages: ``cross_edges`` counts internal edges (any type)
between members that share no base setcode (``setcode & 0xfff``; a card without
setcodes is its own group); ``archetypes`` are the base setcodes held by >= 2
members. A package is ``cross_archetype`` when it has a cross edge and either
two archetypes or >= 2 members outside its main archetype.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from ygorl.build.synergy_graph import EDGE_TYPES, REACH_TYPES, SynergyGraph

TYPE_WEIGHT = {"search": 1.0, "special_summon": 1.0, "send_to_grave": 0.7, "recover": 0.5, "material": 0.5}
DEFAULTS = {"max_size": 15, "min_size": 3, "min_affinity": 0.25, "min_share": 0.15, "k": 2, "max_overlap": 0.6}


@dataclass(frozen=True)
class Package:
    members: tuple[int, ...]
    starters: tuple[int, ...]
    score: float
    reach_mass: float
    density: float
    archetypes: tuple[int, ...] = ()
    cross_edges: int = 0
    cross_archetype: bool = False
    seed: int | None = None

    def __len__(self) -> int:
        return len(self.members)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping) -> Package:
        d = dict(data)
        for key in ("members", "starters", "archetypes"):
            d[key] = tuple(d[key])
        return cls(**d)


def setcodes_from_db(db: Mapping) -> dict[int, tuple[int, ...]]:
    """``password -> setcodes`` for :func:`enumerate_packages` (from a :class:`~ygorl.cards.cdb.CardDB`)."""
    return {pw: tuple(card.setcodes) for pw, card in db.items()}


class Affinity:
    """Undirected specificity-weighted adjacency of a synergy graph."""

    def __init__(self, graph: SynergyGraph, types: Iterable[str] = EDGE_TYPES, weights: Mapping[str, float] = TYPE_WEIGHT) -> None:
        adj: dict[int, dict[int, float]] = {}
        for e in graph.edges(types):
            w = weights.get(e.type, 0.5) / math.sqrt(max(e.fanout, 1))
            a = adj.setdefault(e.src, {})
            a[e.dst] = a.get(e.dst, 0.0) + w
            b = adj.setdefault(e.dst, {})
            b[e.src] = b.get(e.src, 0.0) + w
        self.adj = adj
        self.degree = {v: sum(nb.values()) for v, nb in adj.items()}


def grow_package(
    graph: SynergyGraph,
    seed: int,
    *,
    max_size: int = DEFAULTS["max_size"],
    min_affinity: float = DEFAULTS["min_affinity"],
    min_share: float = DEFAULTS["min_share"],
    affinity: Affinity | None = None,
) -> list[int]:
    """Members grown greedily from ``seed``, in insertion order (seed first)."""
    aff_index = affinity if affinity is not None else Affinity(graph)
    adj, deg = aff_index.adj, aff_index.degree
    members = [seed]
    inside = {seed}
    aff: dict[int, float] = dict(adj.get(seed, {}))
    while len(members) < max_size:
        best, best_key = None, 0.0
        for v, a in aff.items():
            if a < min_affinity:
                continue
            share = a / deg[v]
            if share < min_share:
                continue
            key = a * share
            if key > best_key or (key == best_key and best is not None and v < best):
                best, best_key = v, key
        if best is None:
            break
        members.append(best)
        inside.add(best)
        del aff[best]
        for v, w in adj.get(best, {}).items():
            if v not in inside:
                aff[v] = aff.get(v, 0.0) + w
    return members


def _groups(password: int, setcodes: Mapping[int, Iterable[int]] | None) -> frozenset:
    codes = setcodes.get(password, ()) if setcodes is not None else ()
    bases = frozenset(sc & 0xFFF for sc in codes)
    return bases if bases else frozenset({("card", password)})


def score_package(
    graph: SynergyGraph,
    members: Iterable[int],
    *,
    k: int = DEFAULTS["k"],
    reach_types: Iterable[str] = REACH_TYPES,
    setcodes: Mapping[int, Iterable[int]] | None = None,
    seed: int | None = None,
) -> Package:
    """Reachability statistics of a member set (see the module docstring)."""
    ms = sorted(set(members))
    inside = set(ms)
    reach_types = tuple(reach_types)
    out = {u: [v for v in graph.successors(u, reach_types) if v in inside] for u in ms}
    closure: dict[int, set[int]] = {}
    for u in ms:
        seen: set[int] = set()
        stack = list(out[u])
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            stack.extend(out[v])
        seen.discard(u)
        closure[u] = seen
    sources = {v: 0 for v in ms}
    for u in ms:
        for v in closure[u]:
            sources[v] += 1
    reach_mass = sum(min(sources[v], k) / k for v in ms)
    # greedy set cover of the package by closed reach sets
    uncovered = set(ms)
    starters = []
    while uncovered:
        best = max(ms, key=lambda u: (len((closure[u] | {u}) & uncovered), -u))
        starters.append(best)
        uncovered -= closure[best] | {best}
    n = len(ms)
    density = sum(len(c) for c in closure.values()) / (n * (n - 1)) if n > 1 else 0.0
    # archetypes and cross-archetype edges
    groups = {p: _groups(p, setcodes) for p in ms}
    counts: dict[int, int] = {}
    for p in ms:
        for g in groups[p]:
            if isinstance(g, int):
                counts[g] = counts.get(g, 0) + 1
    archetypes = tuple(sorted(g for g, c in counts.items() if c >= 2))
    cross = 0
    for u in ms:
        for e in graph.out_edges(u):
            if e.dst in inside and not (groups[u] & groups[e.dst]):
                cross += 1
    main = max(archetypes, key=lambda g: (counts[g], -g)) if archetypes else None
    outside_main = sum(1 for p in ms if main not in groups[p])
    cross_archetype = cross > 0 and (len(archetypes) >= 2 or outside_main >= 2)
    return Package(
        members=tuple(ms),
        starters=tuple(sorted(starters)),
        score=reach_mass / len(starters),
        reach_mass=reach_mass,
        density=density,
        archetypes=archetypes,
        cross_edges=cross,
        cross_archetype=cross_archetype,
        seed=seed,
    )


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def enumerate_packages(
    graph: SynergyGraph,
    *,
    seeds: Iterable[int] | None = None,
    setcodes: Mapping[int, Iterable[int]] | None = None,
    max_size: int = DEFAULTS["max_size"],
    min_size: int = DEFAULTS["min_size"],
    min_affinity: float = DEFAULTS["min_affinity"],
    min_share: float = DEFAULTS["min_share"],
    k: int = DEFAULTS["k"],
    max_overlap: float = DEFAULTS["max_overlap"],
    grow_types: Iterable[str] = EDGE_TYPES,
    reach_types: Iterable[str] = REACH_TYPES,
    limit: int | None = None,
) -> list[Package]:
    """Engine packages grown from ``seeds`` (default: every card with an out-edge), best first.

    A package whose Jaccard overlap with a better-ranked one exceeds
    ``max_overlap`` is dropped as a near-duplicate.
    """
    affinity = Affinity(graph, grow_types)
    if seeds is None:
        seeds = sorted(p for p in graph.nodes if graph.out_edges(p, grow_types))
    found: dict[tuple[int, ...], Package] = {}
    for seed in seeds:
        if seed not in graph.nodes:
            continue
        grown = grow_package(graph, seed, max_size=max_size, min_affinity=min_affinity, min_share=min_share, affinity=affinity)
        if len(grown) < min_size:
            continue
        key = tuple(sorted(grown))
        if key in found:
            continue
        found[key] = score_package(graph, key, k=k, reach_types=reach_types, setcodes=setcodes, seed=seed)
    ranked = sorted(found.values(), key=lambda p: (-p.score, -p.density, -len(p.members), p.members))
    kept: list[Package] = []
    for p in ranked:
        if any(_jaccard(p.members, q.members) > max_overlap for q in kept):
            continue
        kept.append(p)
        if limit is not None and len(kept) >= limit:
            break
    return kept
