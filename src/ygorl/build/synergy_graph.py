"""Script-mined synergy graph over card passwords (T5.3).

Nodes are the canonical, non-token cards of the card database (alternate
artworks folded into the original password). A directed, typed edge
``A -> B`` means "card A's effects can bring card B into play":

================  =====================================================================
type              meaning (``locations`` = where B is taken from, own side)
================  =====================================================================
``search``        A adds B from the Deck to the hand, or Sets / places / equips it from the Deck
``special_summon`` A Special Summons B (hand, Deck, GY, banished or Extra Deck)
``send_to_gy``    A sends B from the Deck, hand or Extra Deck to the GY
``recover``       A adds B to the hand from the GY or banishment
``material``      A can be used as material for B (Link/Xyz/Synchro/Fusion procedure of B)
================  =====================================================================

Edges come from :mod:`ygorl.build.scripts` queries: each query's filter is
evaluated against the card database (:class:`~ygorl.build.filters.CardIndex`),
restricted to the cards that can sit in the source location (e.g. no Extra
Deck monsters are searched from the Deck). A query matching more than
``max_fanout`` cards ("add 1 monster") carries no synergy signal and is
dropped; every edge records the *fanout* of the most specific query that
produced it, so consumers can weight edges by specificity.

Extension point: other relation sources (Yugipedia SMW relations, text
similarity, T5.1/T5.2) can be merged as extra edge types via
:meth:`SynergyGraph.add_edges`; they are not available offline yet.

The graph itself is environment-independent (mined from the pinned
CardScripts / BabelCDB revisions recorded in ``meta``); bind it to an
environment with :meth:`SynergyGraph.restrict` (card pool + ``Environment.stamp()``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ygorl import paths
from ygorl.build.filters import CardIndex
from ygorl.build.scripts import EVIDENCE_RANK, Query, ScriptFacts, analyze_script, load_constants
from ygorl.cards.cdb import CardDB

FORMAT = "ygorl-synergy-graph"
FORMAT_VERSION = 1
EDGE_TYPES = ("search", "special_summon", "send_to_gy", "recover", "material")
REACH_TYPES = ("search", "special_summon")  # edges that bring a card to the hand / field
DEFAULT_MAX_FANOUT = 100

GRAPH_SOURCES = ("lua.py", "filters.py", "scripts.py", "synergy_graph.py")

_TYPE_ORDER = {t: i for i, t in enumerate(EDGE_TYPES)}
_SEARCH_ACTIONS = frozenset(("to_hand", "set", "place", "equip"))


@dataclass(frozen=True, slots=True)
class Edge:
    src: int
    dst: int
    type: str
    locations: int  # own-side LOCATION_* mask B is taken from (0 for material)
    fanout: int  # number of cards the (most specific) producing query matches
    evidence: str  # best Query.evidence among the producing queries


def _merge(a: Edge, b: Edge) -> Edge:
    evidence = min(a.evidence, b.evidence, key=lambda e: EVIDENCE_RANK.get(e, 99))
    return Edge(a.src, a.dst, a.type, a.locations | b.locations, min(a.fanout, b.fanout), evidence)


class SynergyGraph:
    """Directed multigraph (at most one edge per ``(src, dst, type)``) over passwords."""

    def __init__(self, nodes: Mapping[int, dict], edges: Iterable[Edge] = (), meta: dict | None = None) -> None:
        self.nodes: dict[int, dict] = {int(p): dict(info) for p, info in nodes.items()}
        self.meta: dict = dict(meta or {})
        self._edges: dict[tuple[int, int, str], Edge] = {}
        self._out: dict[int, list[Edge]] | None = None
        self._in: dict[int, list[Edge]] | None = None
        self.add_edges(edges)

    # -- construction -------------------------------------------------------------
    def add_edges(self, edges: Iterable[Edge]) -> None:
        """Add (or merge) edges; endpoints must be nodes. New edge types are allowed."""
        table = self._edges
        for e in edges:
            if e.src == e.dst or e.src not in self.nodes or e.dst not in self.nodes:
                continue
            key = (e.src, e.dst, e.type)
            old = table.get(key)
            table[key] = e if old is None else _merge(old, e)
        self._out = self._in = None

    def restrict(self, pool: Iterable[int], stamp: Mapping | None = None) -> SynergyGraph:
        """Subgraph on ``pool`` (e.g. an environment's card pool); ``stamp`` = ``Environment.stamp()``."""
        keep = set(pool) & set(self.nodes)
        meta = dict(self.meta)
        if stamp is not None:
            meta["environment"] = dict(stamp)
        return SynergyGraph(
            {p: self.nodes[p] for p in sorted(keep)},
            (e for e in self._edges.values() if e.src in keep and e.dst in keep),
            meta,
        )

    # -- queries -----------------------------------------------------------------------
    def _index(self) -> None:
        out: dict[int, list[Edge]] = {}
        inn: dict[int, list[Edge]] = {}
        for e in sorted(self._edges.values(), key=lambda e: (_TYPE_ORDER.get(e.type, 99), e.type, e.dst, e.src)):
            out.setdefault(e.src, []).append(e)
            inn.setdefault(e.dst, []).append(e)
        self._out, self._in = out, inn

    def edges(self, types: Iterable[str] | None = None) -> list[Edge]:
        if types is None:
            return list(self._edges.values())
        ts = set(types)
        return [e for e in self._edges.values() if e.type in ts]

    def edge(self, src: int, dst: int, type: str) -> Edge | None:
        return self._edges.get((src, dst, type))

    def out_edges(self, password: int, types: Iterable[str] | None = None) -> list[Edge]:
        if self._out is None:
            self._index()
        es = self._out.get(password, [])
        if types is None:
            return list(es)
        ts = set(types)
        return [e for e in es if e.type in ts]

    def in_edges(self, password: int, types: Iterable[str] | None = None) -> list[Edge]:
        if self._in is None:
            self._index()
        es = self._in.get(password, [])
        if types is None:
            return list(es)
        ts = set(types)
        return [e for e in es if e.type in ts]

    def successors(self, password: int, types: Iterable[str] | None = None) -> set[int]:
        return {e.dst for e in self.out_edges(password, types)}

    def predecessors(self, password: int, types: Iterable[str] | None = None) -> set[int]:
        return {e.src for e in self.in_edges(password, types)}

    def neighbors(self, password: int, types: Iterable[str] | None = None) -> set[int]:
        return self.successors(password, types) | self.predecessors(password, types)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, password: int) -> bool:
        return password in self.nodes

    # -- summary -----------------------------------------------------------------------
    def summary(self) -> dict:
        by_type = {t: 0 for t in EDGE_TYPES}
        for e in self._edges.values():
            by_type[e.type] = by_type.get(e.type, 0) + 1
        if self._out is None:
            self._index()
        out = {
            "nodes": len(self.nodes),
            "edges": len(self._edges),
            "edges_by_type": by_type,
            "nodes_with_out_edges": sum(1 for p in self._out if self._out[p]),
            "nodes_with_edges": len(set(self._out) | set(self._in)),
        }
        out.update(self.meta.get("stats", {}))
        if out.get("scripts_in_db"):
            out["script_coverage"] = out.get("scripts_with_edges", 0) / out["scripts_in_db"]
        return out

    # -- persistence ---------------------------------------------------------------------
    def to_json(self) -> dict:
        types = list(EDGE_TYPES) + sorted({e.type for e in self._edges.values()} - set(EDGE_TYPES))
        tix = {t: i for i, t in enumerate(types)}
        evidences = sorted({e.evidence for e in self._edges.values()})
        eix = {t: i for i, t in enumerate(evidences)}
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "meta": self.meta,
            "edge_types": types,
            "evidence": evidences,
            "nodes": [[p, info.get("categories", 0), int(bool(info.get("has_script")))] for p, info in sorted(self.nodes.items())],
            "edges": [
                [e.src, e.dst, tix[e.type], e.locations, e.fanout, eix[e.evidence]]
                for e in sorted(self._edges.values(), key=lambda e: (e.src, e.dst, tix[e.type]))
            ],
        }

    @classmethod
    def from_json(cls, data: Mapping) -> SynergyGraph:
        if data.get("format") != FORMAT or data.get("version") != FORMAT_VERSION:
            raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} file")
        types, evidences = data["edge_types"], data["evidence"]
        nodes = {p: {"categories": cat, "has_script": bool(hs)} for p, cat, hs in data["nodes"]}
        edges = (Edge(s, d, types[t], loc, fan, evidences[ev]) for s, d, t, loc, fan, ev in data["edges"])
        return cls(nodes, edges, data.get("meta"))

    def save(self, path: str | Path) -> None:
        """Write JSON (gzip-compressed when ``path`` ends with ``.gz``)."""
        path = Path(path)
        text = json.dumps(self.to_json(), separators=(",", ":"))
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(text.encode(), mtime=0))
        else:
            path.write_text(text)

    @classmethod
    def load(cls, path: str | Path) -> SynergyGraph:
        path = Path(path)
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        return cls.from_json(json.loads(raw))


# ---------------------------------------------------------------- building


def script_files(scripts_dir: str | Path) -> dict[int, Path]:
    """``c<password>.lua`` files of a directory, by password."""
    out = {}
    for p in Path(scripts_dir).glob("c*.lua"):
        if p.stem[1:].isdigit():
            out[int(p.stem[1:])] = p
    return dict(sorted(out.items()))


_WORKER_CONSTS: dict[str, int] = {}


def _init_worker(consts: dict[str, int]) -> None:
    _WORKER_CONSTS.clear()
    _WORKER_CONSTS.update(consts)


def _analyze_file(item: tuple[int, str]) -> ScriptFacts:
    password, path = item
    return analyze_script(Path(path).read_text(encoding="utf-8", errors="replace"), password, _WORKER_CONSTS)


def analyze_scripts(scripts_dir: str | Path, consts: dict[str, int] | None = None, workers: int = 1) -> dict[int, ScriptFacts]:
    """Analyze every ``c<password>.lua`` script of ``scripts_dir``."""
    consts = consts if consts is not None else load_constants()
    items = [(pw, str(p)) for pw, p in script_files(scripts_dir).items()]
    if workers <= 1:
        _init_worker(consts)
        return {pw: _analyze_file((pw, p)) for pw, p in items}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(consts,)) as pool:
        return {f.password: f for f in pool.map(_analyze_file, items, chunksize=64)}


def _scopes(index: CardIndex, K: Mapping[str, int]) -> dict[str, int]:
    pend = index.type_mask(K["TYPE_PENDULUM"]) & index.monsters
    return {
        "main": index.main_deck,
        "main_monsters": index.main_deck & index.monsters,
        "monsters": index.monsters,
        "extra": index.extra_deck | pend,
    }


def query_targets(q: Query, K: Mapping[str, int], scopes: Mapping[str, int]) -> list[tuple[str, int, int]]:
    """``(edge_type, locations, scope_bits)`` a query can produce edges of."""
    deck, hand, grave, removed, extra = (K[f"LOCATION_{n}"] for n in ("DECK", "HAND", "GRAVE", "REMOVED", "EXTRA"))
    loc = q.locations
    out = []
    if q.action == "material":
        out.append(("material", 0, scopes["monsters"]))
    elif q.action in _SEARCH_ACTIONS and loc & deck:
        out.append(("search", deck, scopes["main"]))
    if q.action == "to_hand" and loc & (grave | removed):
        out.append(("recover", loc & (grave | removed), scopes["main"]))
    elif q.action == "special_summon":
        where = loc & (hand | deck | grave | removed | extra)
        scope = 0
        if where & (hand | deck):
            scope |= scopes["main_monsters"]
        if where & (grave | removed):
            scope |= scopes["monsters"]
        if where & extra:
            scope |= scopes["extra"]
        if where:
            out.append(("special_summon", where, scope))
    elif q.action == "to_grave":
        where = loc & (deck | hand | extra)
        scope = (scopes["main"] if where & (deck | hand) else 0) | (scopes["extra"] if where & extra else 0)
        if where:
            out.append(("send_to_gy", where, scope))
    return out


def edges_from_facts(
    facts: Mapping[int, ScriptFacts], index: CardIndex, K: Mapping[str, int], max_fanout: int, stats: dict | None = None
) -> list[Edge]:
    scopes = _scopes(index, K)
    stats = stats if stats is not None else {}
    for key in ("queries", "queries_unconstrained", "queries_over_fanout", "queries_no_target", "queries_untracked", "queries_with_edges", "scripts_with_edges"):
        stats.setdefault(key, 0)
    edges: list[Edge] = []
    db = index.db
    for pw, f in facts.items():
        if pw not in db:
            continue
        src = db.canonical(pw)
        if src not in index.pos:
            continue
        produced = False
        for q in f.queries:
            stats["queries"] += 1
            targets = query_targets(q, K, scopes)
            if not targets:
                stats["queries_untracked"] += 1
                continue
            bits, _exact = index.evaluate(q.filter)
            if bits is None:
                stats["queries_unconstrained"] += 1
                continue
            made = over = False
            for etype, locs, scope in targets:
                hit = bits & scope
                fanout = hit.bit_count()
                if fanout > max_fanout:
                    over = True
                    continue
                for dst in index.members(hit):
                    if dst == src:
                        continue
                    made = True
                    if etype == "material":  # the matched card is material *for* the script's card
                        edges.append(Edge(dst, src, etype, locs, fanout, q.evidence))
                    else:
                        edges.append(Edge(src, dst, etype, locs, fanout, q.evidence))
            if made:
                stats["queries_with_edges"] += 1
                produced = True
            elif over:
                stats["queries_over_fanout"] += 1
            else:
                stats["queries_no_target"] += 1
        stats["scripts_with_edges"] += produced
    return edges


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def build_graph(
    db: CardDB | None = None,
    scripts_dir: str | Path | None = None,
    *,
    max_fanout: int = DEFAULT_MAX_FANOUT,
    workers: int = 1,
    facts: Mapping[int, ScriptFacts] | None = None,
) -> SynergyGraph:
    """Mine ``scripts_dir`` (default: CardScripts ``official/``) into a :class:`SynergyGraph`."""
    db = db if db is not None else CardDB.load()
    scripts_dir = Path(scripts_dir) if scripts_dir is not None else paths.card_scripts() / "official"
    K = load_constants()
    if facts is None:
        facts = analyze_scripts(scripts_dir, K, workers=workers)
    index = CardIndex(db)
    in_db = {pw: f for pw, f in facts.items() if pw in db}

    def table(attr: str) -> dict[int, tuple[int, ...]]:
        return {pw: getattr(f, attr) for pw, f in in_db.items() if getattr(f, attr)}

    index.set_listings(table("listed_names"), table("listed_series"), table("material_codes"), table("material_setcodes"))
    stats: dict[str, int] = {
        "scripts": len(facts),
        "scripts_in_db": len(in_db),
        "parse_errors": sum(f.stats.get("parse_error", 0) for f in facts.values()),
        "scripts_with_categories": sum(1 for f in in_db.values() if f.categories),
        "scripts_with_queries": sum(1 for f in in_db.values() if f.queries),
    }
    for key in ("match_calls", "opponent_side", "unknown_action", "unknown_location"):
        stats[key] = sum(f.stats.get(key, 0) for f in in_db.values())
    edges = edges_from_facts(in_db, index, K, max_fanout, stats)
    nodes = {p: {"categories": 0, "has_script": False} for p in index.passwords}
    for pw, f in in_db.items():
        node = nodes.get(db.canonical(pw))
        if node is not None:
            node["categories"] |= f.categories
            node["has_script"] = True
    meta = {
        "max_fanout": max_fanout,
        "sources": {
            "card_scripts": _git_head(paths.card_scripts()) if scripts_dir.is_relative_to(paths.card_scripts()) else str(scripts_dir),
            "babel_cdb": _git_head(paths.cards_cdb().parent),
        },
        "stats": stats,
    }
    return SynergyGraph(nodes, edges, meta)


# ---------------------------------------------------------------- caching


def cache_dir() -> Path:
    return Path(os.environ.get("YGORL_CACHE_DIR") or Path.home() / ".cache" / "ygorl")


def _cache_key(scripts_dir: Path, cdb: Path, max_fanout: int) -> str:
    h = hashlib.sha256()
    for name in GRAPH_SOURCES:  # the modules the graph depends on (not packages.py etc.)
        h.update((Path(__file__).parent / name).read_bytes())
    for p in sorted(scripts_dir.glob("c*.lua")):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns};".encode())
    for name in ("constant.lua", "archetype_setcode_constants.lua", "card_counter_constants.lua"):
        st = (paths.card_scripts() / name).stat()
        h.update(f"{name}:{st.st_size}:{st.st_mtime_ns};".encode())
    st = cdb.stat()
    h.update(f"{cdb}:{st.st_size}:{st.st_mtime_ns}:{max_fanout}".encode())
    return h.hexdigest()[:20]


def load_or_build(*, max_fanout: int = DEFAULT_MAX_FANOUT, workers: int = 1, cache: Path | None = None) -> SynergyGraph:
    """The default graph (CardScripts ``official/`` x ``cards.cdb``), cached on disk.

    The cache key covers the graph modules' source, the script files, the constant
    files and the card database, so a stale graph is never returned.
    """
    scripts_dir = paths.card_scripts() / "official"
    cdb = paths.cards_cdb()
    directory = Path(cache) if cache is not None else cache_dir() / "synergy"
    path = directory / f"graph-{_cache_key(scripts_dir, cdb, max_fanout)}.json.gz"
    if path.is_file():
        try:
            return SynergyGraph.load(path)
        except (OSError, ValueError):
            pass
    graph = build_graph(CardDB.load(cdb), scripts_dir, max_fanout=max_fanout, workers=workers)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.gz")
    graph.save(tmp)
    os.replace(tmp, path)
    return graph


# ---------------------------------------------------------------- recall evaluation


def package_coverage(
    graph: SynergyGraph,
    package: Iterable[int],
    types: Iterable[str] | None = None,
    canonical: Callable[[int], int | None] | None = None,
) -> float:
    """Share of a package held together by the graph.

    Members are canonicalized (``canonical``), then the fraction of members in
    the largest weakly connected component of the subgraph induced by the
    package (edges of ``types``, default all) is returned. 1.0 = the graph
    links the whole package; 1/n = no member is linked to another.
    """
    members = set()
    for p in package:
        c = canonical(p) if canonical is not None else None
        members.add(c if c is not None else p)
    if not members:
        return 0.0
    types = tuple(types) if types is not None else None
    parent = {p: p for p in members}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in members:
        if p not in graph.nodes:
            continue
        for q in graph.successors(p, types):
            if q in members:
                parent[find(p)] = find(q)
    sizes: dict[int, int] = {}
    for p in members:
        r = find(p)
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) / len(members)


@dataclass
class RecallReport:
    threshold: float
    coverage: dict[str, float]
    recovered: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return len(self.recovered) / len(self.coverage) if self.coverage else 0.0

    @property
    def mean_coverage(self) -> float:
        return sum(self.coverage.values()) / len(self.coverage) if self.coverage else 0.0


def evaluate_recall(
    graph: SynergyGraph,
    packages: Mapping[str, Iterable[int]],
    threshold: float = 0.8,
    types: Iterable[str] | None = None,
    canonical: Callable[[int], int | None] | None = None,
) -> RecallReport:
    """Recall of engine packages: a package is *recovered* when its coverage >= ``threshold``.

    ``packages`` must come from outside the graph (meta deck lists, T5.1); they
    are only used for evaluation, never for building the graph.
    """
    cov = {name: package_coverage(graph, pkg, types, canonical) for name, pkg in packages.items()}
    rec = sorted(n for n, c in cov.items() if c >= threshold)
    miss = sorted(n for n, c in cov.items() if c < threshold)
    return RecallReport(threshold, cov, rec, miss)
