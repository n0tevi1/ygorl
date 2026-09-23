"""Yugipedia SMW relations as an optional edge source of the synergy graph (T5.3 x T5.1).

An environment may ship ``relations.json`` (``ygorl env build``, docs/data.md):
Yugipedia page name (mostly archetypes) -> pool passwords, for the properties
``archetype_support``, ``anti_support``, ``archseries_related`` and ``archseries``.
This module turns them into :class:`~ygorl.build.synergy_graph.Edge` objects
that :meth:`SynergyGraph.add_edges` merges into a (usually environment-restricted)
graph. Nothing here runs by default: the script-mined graph, its disk cache and
``load_or_build()`` are unchanged; callers opt in with :func:`add_relation_edges`.

For every archetype ``X`` with members ``M = archseries[X]``:

=======================  ==========================================================
edge type                edges
=======================  ==========================================================
``archetype_support``    ``s -> m`` for each ``s`` in ``archetype_support[X]``, ``m`` in ``M``
``archseries_related``   ``r -> m`` for each ``r`` in ``archseries_related[X]``, ``m`` in ``M``
=======================  ==========================================================

``archetype_support`` is the default. ``archseries_related`` is opt-in: besides
real support ("Maiden of White" -> Blue-Eyes) it holds pages that are not
synergy at all ("Recolored counterpart", "Signature move"), about 17k of its
~83k edges on md-2026-09.

* sources in ``M`` are skipped unless ``member_sources`` (see below);
* ``fanout = |M|`` (the number of cards the relation points at, like a script
  query's fanout), ``locations = 0``, ``evidence = "yugipedia"``.
* **Cap:** archetypes with ``|M| > max_fanout`` (default 100, the script graph's
  ``DEFAULT_MAX_FANOUT``) are skipped, as a script query matching more than 100
  cards is: "HERO" (157 members), "Number", "Chaos", "Performapal" say little
  about which cards combine and would add up to ~10^4 edges each.
* Pages without an ``archseries`` entry (card groups such as "Link Monster")
  have no members and add nothing.
* ``anti_support`` (cards that work *against* an archetype, e.g. hate cards for
  Link Monsters) is never turned into edges: the graph's edges mean positive
  synergy, and every consumer (recall, T5.4 package growth) treats an edge as
  a reason to put two cards together. Its size is reported in the stats only.

**Members are not sources** (``member_sources=False``): Yugipedia lists an
archetype's own members under its ``archetype_support`` (all 14 Dracotail cards
"support" Dracotail), which would only restate archetype membership as a dense
biclique -- 107k of the 145k ``archetype_support`` pairs on md-2026-09, and no
change in recall. With the default, the edges say what the script graph may
miss: a non-member supports the archetype ("Sage with Eyes of Blue" -> the
"Blue-Eyes" monsters, "Nadir Servant" -> the "Dogmatika" cards). ``member_sources=True`` restores the
literal reading.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from ygorl.build.synergy_graph import DEFAULT_MAX_FANOUT, Edge, SynergyGraph
from ygorl.data.environment import Environment

RELATION_TYPES = ("archetype_support", "archseries_related")
DEFAULT_RELATION_TYPES = ("archetype_support",)  # archseries_related is noisier (see below); opt in explicitly
EVIDENCE = "yugipedia"
RELATIONS_FILE = "relations.json"


def load_relations(source: str | Path | Environment) -> dict:
    """``relations.json`` of an environment (``Environment``, its directory, or the file itself)."""
    path = source.root if isinstance(source, Environment) else Path(source)
    if path is None:
        raise ValueError(f"environment {source.version} has no directory")  # type: ignore[union-attr]
    if path.is_dir():
        path = path / RELATIONS_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def relation_edges(
    relations: Mapping,
    types: Iterable[str] = RELATION_TYPES,
    max_fanout: int = DEFAULT_MAX_FANOUT,
    member_sources: bool = False,
) -> tuple[list[Edge], dict]:
    """Edges of the given relation ``types`` and statistics (see the module docstring)."""
    types = tuple(types)
    unknown = set(types) - set(RELATION_TYPES)
    if unknown:
        raise ValueError(f"unknown relation types {sorted(unknown)}; have {RELATION_TYPES}")
    archseries: Mapping[str, list[int]] = relations.get("archseries", {})
    edges: list[Edge] = []
    stats: dict = {"max_fanout": max_fanout, "anti_support_skipped": sum(len(v) for v in relations.get("anti_support", {}).values())}
    for etype in types:
        used = capped = 0
        for page, sources in sorted(relations.get(etype, {}).items()):
            members = archseries.get(page)
            if not members:
                continue
            if len(members) > max_fanout:
                capped += 1
                continue
            used += 1
            if not member_sources:
                sources = sorted(set(sources) - set(members))
            for s in sources:
                edges.extend(Edge(int(s), int(m), etype, 0, len(members), EVIDENCE) for m in members if m != s)
        stats[etype] = {"archetypes": used, "capped_archetypes": capped}
    return edges, stats


def add_relation_edges(
    graph: SynergyGraph,
    relations: Mapping,
    types: Iterable[str] = DEFAULT_RELATION_TYPES,
    max_fanout: int = DEFAULT_MAX_FANOUT,
    member_sources: bool = False,
) -> dict:
    """Merge Yugipedia relation edges into ``graph`` in place; records ``graph.meta["relations"]``.

    Edges whose endpoints are not graph nodes (cards outside a restricted pool)
    are dropped by :meth:`SynergyGraph.add_edges`. Returns the statistics, with
    the number of edges of each type actually in the graph afterwards.
    """
    types = tuple(types)
    edges, stats = relation_edges(relations, types, max_fanout, member_sources)
    graph.add_edges(edges)
    for etype in types:
        stats[etype]["edges"] = len(graph.edges((etype,)))
    stats["source"] = relations.get("source")
    stats["retrieved"] = relations.get("retrieved")
    graph.meta["relations"] = stats
    return stats
