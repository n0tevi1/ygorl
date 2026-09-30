"""Engine-aware candidates for deck evolution (ygorl.build.deck_engine, #145), on a hand-made synergy graph."""

import json

import numpy as np

from ygorl.build.deck_engine import addition_pool, archetypes, connected, deck_engine, generic_roles
from ygorl.build.signals import CardValueModel, informed_children
from ygorl.build.synergy_graph import Edge, SynergyGraph
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

# The parent runs an engine: 1 (a field-spell searcher, "add 1 Field Spell": fanout 300) -> 2 (the field spell)
# -> 3, 4 (archetype monsters, fanout 5); 5 sends 3 to the GY (fanout 8); 6 is a generic starter that special
# summons 1 (fanout 200). 7 is a hand trap (generic), 8 an unconnected card. Outside the deck: 20 is searched by 2
# (fanout 5: connected), 21 is special summoned by 6's generic effect (fanout 200: another engine's card),
# 22 has no edge at all, 23 is a generic board breaker, 24 sends 3 to the GY from outside (fanout 10: connected).
DECK_LOC = C.LOCATION_DECK
EDGES = [
    Edge(1, 2, "search", DECK_LOC, 300, "x"),
    Edge(2, 3, "search", DECK_LOC, 5, "x"),
    Edge(2, 4, "search", DECK_LOC, 5, "x"),
    Edge(5, 3, "send_to_grave", DECK_LOC, 8, "x"),
    Edge(6, 1, "special_summon", DECK_LOC, 200, "x"),
    Edge(2, 20, "search", DECK_LOC, 5, "x"),
    Edge(6, 21, "special_summon", DECK_LOC, 200, "x"),
    Edge(24, 3, "send_to_grave", DECK_LOC, 10, "x"),
    Edge(7, 3, "special_summon", DECK_LOC, 900, "x"),  # a generic card's huge-fanout edge: not engine-making
]
GRAPH = SynergyGraph({p: {} for p in range(1, 30)}, EDGES)
GENERIC = {7: "hand_trap", 23: "board_breaker"}
PARENT = Deck(main=(1, 1, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 8, *range(100, 127)))


def test_the_engine_holds_searchers_starters_and_targets_but_not_generic_or_loose_cards():
    eng = deck_engine(PARENT, GRAPH, generic=GENERIC)
    assert eng.members == {1, 2, 3, 4, 5, 6}  # 1 and 6 only via wide-fanout edges: inside the deck they count
    assert 7 not in eng and 8 not in eng  # a hand trap never, an unconnected card neither
    assert set(eng.starters) <= eng.members and 6 in eng.starters
    # a tight in-deck fanout drops the wide edges (and with them the searcher and the starter)
    assert deck_engine(PARENT, GRAPH, generic=GENERIC, max_fanout=50).members == {2, 3, 4, 5}
    # search / special summon only: the send-to-GY card drops out
    assert 5 not in deck_engine(PARENT, GRAPH, generic=GENERIC, types=("search", "special_summon")).members


def test_the_addition_pool_excludes_off_engine_cards():
    assert connected(PARENT, GRAPH, generic=GENERIC) == {20, 24}  # 21 only via a fanout-200 edge
    candidates = [21, 22, 20, 23, 7, 8]  # e.g. the tech pool: other engines' cards 21 and 22 must not come in
    pool = addition_pool(PARENT, GRAPH, generic=GENERIC, candidates=candidates)
    assert 21 not in pool and 22 not in pool
    assert pool[:4] == [20, 23, 7, 8]  # candidates first, in their order (7, 8: another copy of a deck card)
    assert set(pool) == {20, 23, 7, 8, 24}  # then the other generic and connected cards
    # the environment's pool bounds it; a wider addition fanout lets the other engine's card in
    assert addition_pool(PARENT, GRAPH, generic=GENERIC, candidates=candidates, pool={20, 7}) == [20, 7]
    assert 21 in addition_pool(PARENT, GRAPH, generic=GENERIC, candidates=candidates, max_fanout=500)


def test_generic_roles_and_archetypes(tmp_path):
    f = tmp_path / "generic.json"
    f.write_text(json.dumps({"cards": [{"password": 7, "role": "hand_trap"}, {"password": 9, "role": "other"}]}))
    assert generic_roles(f) == {7: "hand_trap"}
    setcodes = {1: (0x10,), 2: (0x1010,), 3: (0x20,), 4: (0x10, 0x20)}
    assert archetypes([1, 2, 3], setcodes) == {0x10}  # 0x1010 is a sub-archetype of base 0x10
    assert archetypes([1, 2, 3, 4], setcodes) == {0x10, 0x20}
    assert archetypes([1, 3], setcodes) == set()


def test_informed_children_spare_engine_cards_and_add_one_copy_per_card():
    pool = [20, 23]
    model = CardValueModel(prior={20: 0.5, 23: 0.4, 1: -0.3, 2: -0.3, 3: -0.3, 4: -0.3, 5: -0.3, 6: -0.3})

    def legal(d):
        return len(d.main) == 40 and max(d.counts().values()) <= 3

    kids = informed_children(PARENT, "T", model, pool, legal=legal, is_extra=lambda pw: False,
                             rng=np.random.default_rng(0), informed=6, explore=0, max_bundle=3,
                             engine={1, 2, 3, 4, 5, 6})  # fmt: skip
    assert kids
    for k in kids:
        assert all(e.out not in {1, 2, 3, 4, 5, 6} for e in k.edits)  # engine drawn lowest, still out last
        assert len({e.into for e in k.edits}) == len(k.edits)  # one copy of a card per bundle
