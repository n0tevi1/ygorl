"""Tests for the script-mined synergy graph (T5.3).

Synthetic tests build a graph from a handful of hand-written scripts over a
hand-made card database; the real-data tests share one full build per session
(see the ``real_graph`` fixture in conftest.py).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ygorl.build.relations import RELATION_TYPES, add_relation_edges, load_relations, relation_edges
from ygorl.build.synergy_graph import (
    EDGE_TYPES,
    REACH_TYPES,
    Edge,
    SynergyGraph,
    build_graph,
    evaluate_recall,
    package_coverage,
)
from ygorl.cards.cdb import Card, CardDB
from ygorl.engine import constants as C

EFFECT = C.TYPE_MONSTER | C.TYPE_EFFECT


def mk(password, *, setcodes=(), type=EFFECT, level=4, race=C.RACE_WARRIOR, attribute=C.ATTRIBUTE_EARTH, alias=0, name=None):
    return Card(
        password=password, name=name or f"card{password}", desc="", strings=("",) * 16, alias=alias, ot=3,
        setcodes=tuple(setcodes), type=type, attack=1000, defense=1000, level=level, lscale=0, rscale=0,
        race=race, attribute=attribute, link_marker=0, category=0,
    )  # fmt: skip


CARDS = [
    mk(1001, setcodes=(0x10,)),  # "A" searcher
    mk(1002, setcodes=(0x10,), type=C.TYPE_SPELL, level=0),  # "A" spell: Special Summons "A" monsters
    mk(1003, setcodes=(0x10,)),  # "A" extender: sends a Fiend to the GY
    mk(2001, setcodes=(0x20,), type=C.TYPE_MONSTER | C.TYPE_NORMAL, level=3, race=C.RACE_FIEND),  # "B", no script
    mk(2002, setcodes=(0x20,), type=EFFECT | C.TYPE_LINK, level=2),  # "B" Link: 2 "A" monsters
    mk(3001, level=4),  # generic: search any monster (too broad)
    mk(3002, alias=1001, name="card1001"),  # alternate art of 1001
    mk(3003, level=4, race=C.RACE_FIEND),
]

SCRIPTS = {
    1001: """
local s,id=GetID()
function s.initial_effect(c)
    local e1=Effect.CreateEffect(c)
    e1:SetCategory(CATEGORY_TOHAND+CATEGORY_SEARCH)
    e1:SetTarget(s.thtg)
    c:RegisterEffect(e1)
end
function s.thfilter(c) return c:IsSetCard(0x10) and c:IsSpellTrap() and c:IsAbleToHand() end
function s.thtg(e,tp,eg,ep,ev,re,r,rp,chk)
    if chk==0 then return Duel.IsExistingMatchingCard(s.thfilter,tp,LOCATION_DECK,0,1,nil) end
end
""",
    1002: """
local s,id=GetID()
function s.initial_effect(c)
    local e1=Effect.CreateEffect(c)
    e1:SetCategory(CATEGORY_SPECIAL_SUMMON)
    e1:SetTarget(s.tg)
    c:RegisterEffect(e1)
end
function s.filter(c,e,tp) return c:IsSetCard(0x10) and c:IsCanBeSpecialSummoned(e,0,tp,false,false) end
function s.tg(e,tp,eg,ep,ev,re,r,rp,chk)
    if chk==0 then return Duel.IsExistingMatchingCard(s.filter,tp,LOCATION_DECK|LOCATION_GRAVE,0,1,nil,e,tp) end
end
""",
    1003: """
local s,id=GetID()
function s.initial_effect(c) end
function s.tgfilter(c) return c:IsRace(RACE_FIEND) and c:IsLevelBelow(3) and c:IsAbleToGrave() end
function s.op(e,tp) local g=Duel.SelectMatchingCard(tp,s.tgfilter,tp,LOCATION_DECK,0,1,1,nil) end
""",
    2002: """
local s,id=GetID()
function s.initial_effect(c)
    Link.AddProcedure(c,aux.FilterBoolFunctionEx(Card.IsSetCard,0x10),2,2)
end
""",
    3001: """
local s,id=GetID()
function s.initial_effect(c) end
function s.f(c) return c:IsMonster() and c:IsAbleToHand() end
function s.op(e,tp) local g=Duel.SelectMatchingCard(tp,s.f,tp,LOCATION_DECK,0,1,1,nil) end
""",
    9999: "function s.initial_effect(c) -- broken, and not in the database",
}


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("scripts")
    for pw, src in SCRIPTS.items():
        (root / f"c{pw}.lua").write_text(src)
    return build_graph(CardDB(CARDS), scripts_dir=root, max_fanout=3)


def triples(graph):
    return sorted((e.src, e.dst, e.type) for e in graph.edges())


def test_edges_from_scripts(mini):
    assert triples(mini) == [
        (1001, 1002, "search"),
        (1001, 2002, "material"),  # material -> the Extra Deck monster it summons
        (1002, 1001, "special_summon"),
        (1002, 1003, "special_summon"),
        (1003, 2001, "send_to_gy"),
        (1003, 2002, "material"),
    ]
    assert mini.edge(1002, 1001, "special_summon").locations == C.LOCATION_DECK | C.LOCATION_GRAVE
    assert mini.edge(1001, 1002, "search").fanout == 1
    assert mini.edge(1002, 1003, "special_summon").fanout == 2


def test_nodes_are_canonical_non_token_cards(mini):
    assert sorted(mini.nodes) == [1001, 1002, 1003, 2001, 2002, 3001, 3003]
    assert mini.nodes[1001]["has_script"] and not mini.nodes[2001]["has_script"]
    assert mini.nodes[1001]["categories"] & (1 << 17)  # CATEGORY_SEARCH


def test_adjacency_queries(mini):
    assert mini.successors(1002) == {1001, 1003}
    assert mini.successors(1001, types=("search",)) == {1002}
    assert mini.predecessors(2002) == {1001, 1003}
    assert mini.neighbors(1001) == {1002, 2002}
    assert [e.type for e in mini.out_edges(1003)] == ["send_to_gy", "material"]
    assert mini.in_edges(1002)[0].src == 1001


def test_summary_and_coverage_stats(mini):
    s = mini.summary()
    assert s["nodes"] == 7 and s["edges"] == 6
    assert s["edges_by_type"] == {"search": 1, "special_summon": 2, "send_to_gy": 1, "recover": 0, "material": 2}
    assert s["scripts"] == 6 and s["scripts_in_db"] == 5 and s["parse_errors"] == 1
    assert s["nodes_with_out_edges"] == 3
    assert s["queries_over_fanout"] == 1  # the "any monster" searcher
    assert 0 < s["script_coverage"] <= 1


def test_save_load_roundtrip(mini, tmp_path):
    path = tmp_path / "g.json"
    mini.save(path)
    data = json.loads(path.read_text())
    assert data["format"] == "ygorl-synergy-graph" and data["edge_types"] == list(EDGE_TYPES)
    back = SynergyGraph.load(path)
    assert triples(back) == triples(mini)
    assert back.nodes == mini.nodes and back.meta == mini.meta
    assert back.edge(1002, 1001, "special_summon") == mini.edge(1002, 1001, "special_summon")
    gz = tmp_path / "g.json.gz"
    mini.save(gz)
    assert triples(SynergyGraph.load(gz)) == triples(mini)


def test_restrict_to_pool_and_environment_stamp(mini):
    sub = mini.restrict({1001, 1002, 2002}, stamp={"environment": "md-test", "fingerprint": "abc"})
    assert triples(sub) == [(1001, 1002, "search"), (1001, 2002, "material"), (1002, 1001, "special_summon")]
    assert sub.meta["environment"] == {"environment": "md-test", "fingerprint": "abc"}


def test_edge_merging():
    g = SynergyGraph({1: {}, 2: {}}, [Edge(1, 2, "search", 1, 5, "filter"), Edge(1, 2, "search", 16, 3, "category")])
    e = g.edge(1, 2, "search")
    assert (e.locations, e.fanout, e.evidence) == (17, 3, "filter")


def test_package_coverage_and_recall(mini):
    assert package_coverage(mini, {1001, 1002, 1003}) == 1.0
    assert package_coverage(mini, {1001, 2001, 3001}) == pytest.approx(1 / 3)
    assert package_coverage(mini, {1001, 1003}) == 0.5  # not adjacent without 1002 / 2002
    assert package_coverage(mini, {1001, 1003, 2002}, types=("search", "special_summon")) == pytest.approx(1 / 3)
    assert package_coverage(mini, {3002, 1002}, canonical={3002: 1001}.get) == 1.0  # alt art folded
    report = evaluate_recall(mini, {"a": {1001, 1002, 1003}, "b": {1001, 2001, 3001}, "c": {1003, 2001}}, threshold=0.8)
    assert report.recall == pytest.approx(2 / 3)
    assert report.recovered == ["a", "c"] and report.missed == ["b"]
    assert report.coverage["b"] == pytest.approx(1 / 3)



def test_relation_edges():
    relations = {
        "archseries": {"A": [1, 2, 3], "Big": list(range(100, 202))},
        "archetype_support": {"A": [1, 9], "Big": [9], "Link Monster": [9]},
        "archseries_related": {"A": [8]},
        "anti_support": {"A": [7]},
    }
    edges, stats = relation_edges(relations, ("archetype_support",))
    assert sorted((e.src, e.dst, e.type, e.fanout, e.evidence) for e in edges) == [
        (9, m, "archetype_support", 3, "yugipedia") for m in (1, 2, 3)
    ]  # member 1 is no source; "Big" (102 members) is capped; "Link Monster" has no members
    assert stats["archetype_support"] == {"archetypes": 1, "capped_archetypes": 1} and stats["anti_support_skipped"] == 1
    with_members, _ = relation_edges(relations, ("archetype_support",), member_sources=True)
    assert {(e.src, e.dst) for e in with_members} == {(9, 1), (9, 2), (9, 3), (1, 2), (1, 3)}
    assert len(relation_edges(relations, ("archetype_support",), max_fanout=200)[0]) == 3 + 102
    with pytest.raises(ValueError):
        relation_edges(relations, ("anti_support",))  # anti-support never becomes a synergy edge

    g = SynergyGraph({p: {} for p in (1, 2, 3, 8, 9)})
    stats = add_relation_edges(g, relations, RELATION_TYPES)
    assert {(e.src, e.dst, e.type) for e in g.edges()} == {(9, 1, "archetype_support"), (9, 2, "archetype_support"),
                                                          (9, 3, "archetype_support"), (8, 1, "archseries_related"),
                                                          (8, 2, "archseries_related"), (8, 3, "archseries_related")}  # fmt: skip
    assert g.meta["relations"]["archseries_related"]["edges"] == 3 and stats["archetype_support"]["edges"] == 3
    assert g.summary()["edges_by_type"]["archetype_support"] == 3
    assert SynergyGraph.from_json(g.to_json()).edge(8, 2, "archseries_related").evidence == "yugipedia"


# ---------------------------------------------------------------- real data

SNAKE_EYE_ASH, SNAKE_EYE_OAK, SNAKE_EYES_POPLAR = 9674034, 45663742, 90241276
ALUBER, BRANDED_FUSION, MIRRORJADE = 62962630, 44362883, 44146295
ARIANNA, WELCOME_LABRYNTH = 1225009, 5380979
FIENDSMITH_ENGRAVER, FIENDSMITHS_TRACT = 60764609, 98567237
DIABELLSTAR, ORIGINAL_SINFUL_SPOILS = 72270339, 89023486
TENPAI_PAIDRA, SANGEN_SUMMONING = 39931513, 30336082


@pytest.mark.parametrize(
    "src,dst,etype",
    [
        (SNAKE_EYE_ASH, SNAKE_EYE_OAK, "special_summon"),  # SS 1 "Snake-Eye" monster from hand/Deck
        (SNAKE_EYE_ASH, SNAKE_EYES_POPLAR, "search"),  # add 1 Level 1 FIRE monster
        (ALUBER, BRANDED_FUSION, "search"),  # add 1 "Branded" Spell/Trap
        (BRANDED_FUSION, MIRRORJADE, "special_summon"),  # Fusion that lists "Fallen of Albaz" as material
        (ARIANNA, WELCOME_LABRYNTH, "search"),
        (FIENDSMITH_ENGRAVER, FIENDSMITHS_TRACT, "search"),
        (DIABELLSTAR, ORIGINAL_SINFUL_SPOILS, "search"),  # Set 1 "Sinful Spoils" Spell/Trap from the Deck
        (TENPAI_PAIDRA, SANGEN_SUMMONING, "search"),  # add or Set 1 "Sangen" Spell/Trap
    ],
)
def test_real_graph_known_edges(real_graph, src, dst, etype):
    assert real_graph.edge(src, dst, etype) is not None, (src, dst, etype)


def test_real_graph_summary(real_graph):
    s = real_graph.summary()
    assert s["nodes"] > 14000
    assert s["scripts"] > 13000 and s["parse_errors"] == 0
    assert s["scripts_with_categories"] / s["scripts_in_db"] > 0.8
    assert s["scripts_with_queries"] / s["scripts_in_db"] > 0.6
    assert s["script_coverage"] > 0.3  # scripts contributing at least one edge
    for etype in EDGE_TYPES:
        assert s["edges_by_type"][etype] > 1000, etype


PROXY = Path(__file__).parent / "data" / "proxy_packages.json"


def proxy_packages():
    data = json.loads(PROXY.read_text())
    return {name: [pw for pw, _name in members] for name, members in data["packages"].items()}


def test_proxy_packages_are_reproducible(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "p.json"
    subprocess.run([sys.executable, str(root / "tools" / "make_proxy_packages.py"), "--out", str(out)], check=True, capture_output=True)
    assert out.read_text() == PROXY.read_text()


def test_proxy_package_recall(real_graph):
    """Acceptance (eng plan T5.3): recall >= 80% of engine packages -- on *proxy* packages until T5.1."""
    packages = proxy_packages()
    assert len(packages) >= 10
    report = evaluate_recall(real_graph, packages, threshold=0.8)
    assert report.recall >= 0.8, report.coverage
    reach = evaluate_recall(real_graph, packages, threshold=0.8, types=("search", "special_summon"))
    assert reach.recall >= 0.8, reach.coverage


ENVIRONMENT = "md-2026-09"
META_PACKAGES = Path(__file__).resolve().parents[1] / "environments" / ENVIRONMENT / "artifacts" / "meta_packages.json"


def meta_packages():
    data = json.loads(META_PACKAGES.read_text())
    return data, {name: [pw for pw, _name in pkg["members"]] for name, pkg in data["packages"].items()}


def test_meta_packages_are_reproducible(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "p.json"
    cmd = [sys.executable, str(root / "tools" / "make_meta_packages.py"), ENVIRONMENT, "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, cwd=root)
    assert out.read_text() == META_PACKAGES.read_text()


def test_meta_package_recall(real_graph):
    """Acceptance (eng plan T5.3): recall >= 80% of the real meta engine packages (docs/synergy.md).

    Thresholds sit just below the values measured on the committed md-2026-09
    snapshot (script edges 0.935 / 0.871, + Yugipedia archetype_support 1.000 / 0.968).
    """
    from ygorl.data.environment import load_environment

    env = load_environment(ENVIRONMENT)
    data, packages = meta_packages()
    env.check_stamp(data["environment"])
    assert len(packages) >= 25
    graph = real_graph.restrict(env.card_pool, stamp=env.stamp())
    script = evaluate_recall(graph, packages)
    assert script.recall >= 0.9, script.missed
    reach = evaluate_recall(graph, packages, types=REACH_TYPES)
    assert reach.recall >= 0.85, reach.missed

    add_relation_edges(graph, load_relations(env))  # default: archetype_support, non-member sources
    both = evaluate_recall(graph, packages)
    assert both.recall >= 0.95, both.missed
    assert all(both.coverage[n] >= script.coverage[n] for n in packages)  # extra edges never lower coverage
    reach = evaluate_recall(graph, packages, types=REACH_TYPES + ("archetype_support",))
    assert reach.recall >= 0.93, reach.missed
