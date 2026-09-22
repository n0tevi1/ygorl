"""Tests for engine-package enumeration on the synergy graph (T5.4)."""

import pytest

from ygorl.build.packages import (
    Package,
    enumerate_packages,
    grow_package,
    score_package,
    setcodes_from_db,
)
from ygorl.build.synergy_graph import Edge, SynergyGraph


def graph(edges, nodes=None):
    es = [Edge(s, d, t, 1, f, "filter") for s, d, t, f in edges]
    ns = set(nodes or ()) | {e.src for e in es} | {e.dst for e in es}
    return SynergyGraph({n: {} for n in ns}, es)


# cluster A (setcode 0x10): 1 searches 2, 3, 4; 2 special summons 3, 4; 3 searches 4
CLUSTER_A = [
    (1, 2, "search", 3), (1, 3, "search", 3), (1, 4, "search", 3),
    (2, 3, "special_summon", 2), (2, 4, "special_summon", 2), (3, 4, "search", 1),
]  # fmt: skip
# cluster B (setcode 0x20): a chain 11 -> 12 -> 13
CLUSTER_B = [(11, 12, "search", 1), (12, 13, "special_summon", 1)]
# a generic hub: searched by 30 other cards through broad filters (fanout 90)
HUB = [(s, 99, "search", 90) for s in (1, 2, 3, 11, 12, 13, *range(21, 51))]
SETCODES = {1: (0x10,), 2: (0x10,), 3: (0x1010,), 4: (0x10,), 11: (0x20,), 12: (0x20,), 13: (0x20,), 99: ()}


def test_score_chain_example():
    g = graph([(1, 2, "search", 1), (2, 3, "search", 1)])
    p = score_package(g, [1, 2, 3], k=2)
    # sources: 1 <- none, 2 <- {1}, 3 <- {1, 2}; one starter
    assert p.starters == (1,)
    assert p.reach_mass == pytest.approx(0 + 0.5 + 1.0)
    assert p.score == pytest.approx(1.5)
    assert p.density == pytest.approx(3 / 6)
    assert p.members == (1, 2, 3)


def test_score_uses_only_reach_edges_and_counts_starters():
    g = graph([(1, 2, "search", 1), (3, 2, "material", 1), (3, 4, "send_to_grave", 1)])
    p = score_package(g, [1, 2, 3, 4], k=1)
    assert p.starters == (1, 3, 4)  # 3 and 4 are not reached through search / special_summon
    assert p.reach_mass == 1.0 and p.score == pytest.approx(1 / 3)


def test_grow_package_prefers_specific_edges_over_hubs():
    g = graph(CLUSTER_A + CLUSTER_B + HUB)
    grown = grow_package(g, 1, max_size=10)
    assert grown[0] == 1 and sorted(grown) == [1, 2, 3, 4]  # insertion order, seed first
    assert grow_package(g, 11, max_size=10) == [11, 12, 13]
    assert grow_package(g, 3, max_size=2) == [3, 4]  # the strongest tie first


def test_enumerate_ranks_and_deduplicates():
    g = graph(CLUSTER_A + CLUSTER_B + HUB)
    pkgs = enumerate_packages(g, min_size=3, setcodes=SETCODES)
    assert [p.members for p in pkgs] == [(1, 2, 3, 4), (11, 12, 13)]
    a, b = pkgs
    assert isinstance(a, Package) and a.score > b.score
    assert a.starters == (1,) and b.starters == (11,)
    assert a.archetypes == (0x10,) and not a.cross_archetype
    assert all(99 not in p.members for p in pkgs)


def test_cross_archetype_package():
    # 11 ("B") searches 1 ("A"), 1 special summons 11: an engine across two archetypes
    g = graph(CLUSTER_A + CLUSTER_B + [(12, 1, "search", 1), (1, 12, "special_summon", 1), (2, 13, "special_summon", 1)])
    pkgs = enumerate_packages(g, min_size=3, max_size=8, setcodes=SETCODES)
    top = pkgs[0]
    assert {1, 2, 12}.issubset(top.members)
    assert top.cross_archetype and top.cross_edges >= 2
    assert set(top.archetypes) == {0x10, 0x20}


def test_packages_are_connected_subgraphs():
    g = graph(CLUSTER_A + CLUSTER_B + HUB)
    for p in enumerate_packages(g, min_size=2):
        members = set(p.members)
        seen, stack = {p.members[0]}, [p.members[0]]
        while stack:
            for q in g.neighbors(stack.pop()):
                if q in members and q not in seen:
                    seen.add(q)
                    stack.append(q)
        assert seen == members


def test_seeds_and_limit():
    g = graph(CLUSTER_A + CLUSTER_B + HUB)
    pkgs = enumerate_packages(g, seeds=[11], min_size=3)
    assert [p.members for p in pkgs] == [(11, 12, 13)] and pkgs[0].seed == 11
    assert len(enumerate_packages(g, min_size=3, limit=1)) == 1


def test_package_json_roundtrip():
    g = graph(CLUSTER_A)
    p = score_package(g, [1, 2, 3, 4], setcodes=SETCODES, seed=1)
    assert Package.from_json(p.to_json()) == p


def test_setcodes_from_db():
    from tests.test_synergy_graph import CARDS
    from ygorl.cards.cdb import CardDB

    sc = setcodes_from_db(CardDB(CARDS))
    assert sc[1001] == (0x10,) and sc[3001] == ()


# ---------------------------------------------------------------- real data (smoke)


def test_real_graph_packages_smoke(real_graph):
    from ygorl.cards.cdb import CardDB

    db = CardDB.load()
    snake_eye_ash, snake_eye_oak = 9674034, 45663742
    aluber, despian_tragedy, branded_opening = 62962630, 36577931, 36637374
    seeds = [snake_eye_ash, aluber, 1225009, 60764609, 72270339, 32909498, 25550531, 91810826]
    pkgs = enumerate_packages(real_graph, seeds=seeds, setcodes=setcodes_from_db(db))
    assert len(pkgs) >= 5
    by_seed = {p.seed: p for p in pkgs}
    assert snake_eye_oak in by_seed[snake_eye_ash].members
    # Aluber grows the Branded + Despia engine: a cross-archetype package
    branded_despia = by_seed[aluber]
    assert {despian_tragedy, branded_opening} <= set(branded_despia.members)
    assert branded_despia.cross_archetype and len(branded_despia.archetypes) >= 2
    for p in pkgs:
        assert 3 <= len(p.members) <= 15 and p.score > 0
    assert [p.score for p in pkgs] == sorted((p.score for p in pkgs), reverse=True)
