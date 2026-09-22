"""Tests for target-filter compilation and evaluation against card data (T5.3)."""

import pytest

from ygorl.build import lua
from ygorl.build.filters import (
    ANY,
    UNKNOWN,
    And,
    CardIndex,
    FilterCompiler,
    Not,
    Or,
    Pred,
    hints,
    setcode_matches,
)
from ygorl.build.scripts import load_constants
from ygorl.cards.cdb import Card, CardDB
from ygorl.engine import constants as C

MONSTER = C.TYPE_MONSTER | C.TYPE_EFFECT


def mk(password, *, setcodes=(), type=MONSTER, level=4, race=C.RACE_FIEND, attribute=C.ATTRIBUTE_DARK,
       attack=1000, defense=1000, alias=0, name=None):  # fmt: skip
    return Card(
        password=password, name=name or f"card{password}", desc="", strings=("",) * 16, alias=alias, ot=3,
        setcodes=tuple(setcodes), type=type, attack=attack, defense=defense, level=level, lscale=0, rscale=0,
        race=race, attribute=attribute, link_marker=0, category=0,
    )  # fmt: skip


CARDS = [
    mk(100, setcodes=(0x205,), level=1, attribute=C.ATTRIBUTE_FIRE),  # plain "A"
    mk(101, setcodes=(0x1205,), level=1, attribute=C.ATTRIBUTE_FIRE, race=C.RACE_DRAGON),  # sub-archetype "A-sub"
    mk(102, setcodes=(0x3205, 0x10), level=7, race=C.RACE_SPELLCASTER),  # both sub-archetypes 0x1 and 0x2
    mk(103, setcodes=(0x205,), type=C.TYPE_SPELL, level=0, race=0, attribute=0),
    mk(104, setcodes=(0x205,), type=C.TYPE_SPELL | C.TYPE_QUICKPLAY, level=0, race=0, attribute=0),
    mk(105, setcodes=(0x10,), type=C.TYPE_TRAP, level=0, race=0, attribute=0),
    mk(106, setcodes=(0x205,), type=C.TYPE_MONSTER | C.TYPE_EFFECT | C.TYPE_XYZ, level=4, attack=2500),
    mk(107, setcodes=(0x205,), type=C.TYPE_MONSTER | C.TYPE_EFFECT | C.TYPE_LINK, level=2, defense=0),
    mk(108, type=C.TYPE_MONSTER | C.TYPE_NORMAL, level=3, attack=1500),
    mk(109, type=C.TYPE_MONSTER | C.TYPE_NORMAL, level=3, attack=1500, alias=108, name="card108"),  # alt art
    mk(110, type=C.TYPE_MONSTER | C.TYPE_EFFECT, level=3, alias=108, name="card108 (rule variant)"),
    mk(111, type=C.TYPE_TOKEN | C.TYPE_MONSTER, level=1),
]


@pytest.fixture(scope="module")
def index():
    idx = CardIndex(CardDB(CARDS))
    idx.set_listings(listed_names={100: {108}, 102: {101}}, listed_series={103: {0x205}},
                     material_codes={106: {100}}, material_setcodes={107: {0x1205}})  # fmt: skip
    return idx


@pytest.fixture(scope="module")
def consts():
    return load_constants()


def members(index, flt):
    bits, _exact = index.evaluate(flt)
    return None if bits is None else sorted(index.members(bits))


def test_setcode_match_semantics():
    assert setcode_matches(0x1205, 0x205)  # plain archetype query matches sub-archetype cards
    assert not setcode_matches(0x205, 0x1205)  # sub-archetype query needs the sub bits
    assert setcode_matches(0x3205, 0x1205) and setcode_matches(0x3205, 0x2205)
    assert not setcode_matches(0x1206, 0x205)


def test_universe_is_canonical_non_token(index):
    assert sorted(index.passwords) == [100, 101, 102, 103, 104, 105, 106, 107, 108, 110]


def test_predicates(index):
    assert members(index, Pred("setcard", (0x205,))) == [100, 101, 102, 103, 104, 106, 107]
    assert members(index, Pred("setcard", (0x1205,))) == [101, 102]
    assert members(index, Pred("setcard", (0x1205, 0x10))) == [101, 102, 105]
    assert members(index, Pred("code", (108,))) == [108, 110]  # alias match; alt art 109 folded into 108
    assert members(index, Pred("code", (109,))) == [108]  # an alt-art password resolves to the original
    assert members(index, Pred("type_any", (C.TYPE_SPELL | C.TYPE_TRAP,))) == [103, 104, 105]
    assert members(index, Pred("type_all", (C.TYPE_SPELL | C.TYPE_QUICKPLAY,))) == [104]
    assert members(index, Pred("type_eq", (C.TYPE_SPELL,))) == [103]
    assert members(index, Pred("race", (C.RACE_DRAGON | C.RACE_SPELLCASTER,))) == [101, 102]
    assert members(index, Pred("attribute", (C.ATTRIBUTE_FIRE,))) == [100, 101]
    # levels only exist on non-Xyz, non-Link monsters
    assert members(index, Pred("level", ("le", 4))) == [100, 101, 108, 110]
    assert members(index, Pred("level", ("eq", 1, 7))) == [100, 101, 102]
    assert members(index, Pred("level", ("ge", 5))) == [102]
    assert members(index, Pred("rank", ("eq", 4))) == [106]
    assert members(index, Pred("link", ("le", 2))) == [107]
    assert members(index, Pred("attack", ("ge", 1500))) == [106, 108]
    assert members(index, Pred("has_level", ())) == [100, 101, 102, 108, 110]
    assert members(index, Pred("lists_code", (108,))) == [100]
    assert members(index, Pred("lists_archetype", (0x205,))) == [103]
    assert members(index, Pred("lists_code_as_material", (100,))) == [106]
    # utility.lua MatchSetcode(query, listed): a plain query matches a listed sub-archetype
    assert members(index, Pred("lists_archetype_as_material", (0x205,))) == [107]
    assert members(index, Pred("lists_archetype_as_material", (0x2205,))) == []
    assert members(index, Pred("lists_archetype", (0x1205,))) == []  # ListsArchetype compares exactly


def test_boolean_combinators_and_unknowns(index):
    sc = Pred("setcard", (0x205,))
    assert members(index, And((sc, Not(Pred("code", (100,)))))) == [101, 102, 103, 104, 106, 107]
    assert members(index, And((sc, UNKNOWN))) == members(index, sc)  # unknown conjuncts are dropped
    assert members(index, Or((sc, UNKNOWN))) is None  # unknown disjunct: unconstrained
    assert members(index, Not(And((sc, UNKNOWN)))) is None  # never negate an over-approximation
    assert members(index, ANY) == sorted(index.passwords)
    assert index.evaluate(sc)[1] is True and index.evaluate(And((sc, UNKNOWN)))[1] is False


def compile_fn(src, name="s.filter", consts=None, card_id=100):
    # the expression must live in the script's own tokens (function bodies are token spans)
    script = lua.Script.parse(f"{src}\nprobe={{{name}}}\n")
    comp = FilterCompiler(script, consts or {}, card_id)
    return comp.compile_value(script.top_level_assignments()["probe"].positional()[0])


def test_compile_function_filter(consts, index):
    src = """
function s.filter(c)
    return c:IsSetCard(SET_TEST) and c:IsSpellTrap() and not c:IsCode(id) and c:IsAbleToHand()
end"""
    flt = compile_fn(src, consts={**consts, "SET_TEST": 0x205})
    assert hints(flt) == {"to_hand"}
    assert members(index, flt) == [103, 104]
    assert members(index, compile_fn(src.replace("id", "103"), consts={**consts, "SET_TEST": 0x205})) == [104]


def test_compile_inlines_helpers_and_aux_wrappers(consts, index):
    src = """
function s.base(c)
    return c:IsRace(RACE_FIEND|RACE_DRAGON) and c:IsLevelBelow(4)
end
function s.filter(c,e,tp)
    return s.base(c) and c:IsCanBeSpecialSummoned(e,0,tp,false,false)
end"""
    flt = compile_fn(src, consts=consts)
    assert hints(flt) == {"special_summon"}
    assert members(index, flt) == [100, 101, 108, 110]
    nv = compile_fn(src, "aux.NecroValleyFilter(s.filter)", consts=consts)
    assert members(index, nv) == [100, 101, 108, 110]
    fb = compile_fn("", "aux.FilterBoolFunction(Card.IsAttribute,ATTRIBUTE_FIRE)", consts=consts)
    assert members(index, fb) == [100, 101]
    anon = compile_fn("", "function(c) return c:IsSetCard({0x1205,0x10}) end", consts=consts)
    assert members(index, anon) == [101, 102, 105]
    notf = compile_fn("", "aux.NOT(aux.FilterBoolFunctionEx(Card.IsType,TYPE_SPELL))", consts=consts)
    assert 103 not in members(index, notf) and 100 in members(index, notf)
    fus = compile_fn("", "Fusion.IsMonsterFilter(Card.IsAbleToGrave)", consts=consts)
    assert hints(fus) == {"to_grave"} and members(index, fus) == [100, 101, 102, 106, 107, 108, 110]
    assert members(index, compile_fn("", "Card.IsSpellTrap", consts=consts)) == [103, 104, 105]


def test_compile_comparisons_and_unknown_calls(consts, index):
    src = """
function s.filter(c,tp)
    return c:GetLevel()<=4 and c:GetAttack()>=1500 and Duel.IsExistingMatchingCard(s.other,tp,LOCATION_DECK,0,1,c)
end
function s.f2(c)
    return 4<c:GetLevel() or c:IsLocation(LOCATION_GRAVE)
end
function s.f3(c)
    if c:IsLocation(LOCATION_GRAVE) then return c:IsSetCard(0x10) end
    return c:IsCode(108)
end"""
    assert members(index, compile_fn(src, consts=consts)) == [108]
    assert members(index, compile_fn(src, "s.f2", consts=consts)) is None  # `or <unknown>` is unconstrained
    assert members(index, compile_fn(src, "s.f3", consts=consts)) == [102, 105, 108, 110]  # union of returns
    assert members(index, compile_fn(src, "s.missing", consts=consts)) is None


def test_parameters_bound_from_call_arguments(consts, index):
    src = """
function s.base(c,lv) return c:IsLevel(lv) end
function s.filter(c,e,tp,code,set)
    return (c:IsCode(code) or c:IsSetCard(set)) and s.base(c,7)
end"""
    script = lua.Script.parse(src)
    comp = FilterCompiler(script, consts, 100)
    ref = lua.Index(lua.Name("s"), "filter")
    extra = (lua.Name("e"), lua.Name("tp"), lua.Const(102), lua.Const(0x10))
    assert members(index, comp.compile_value(ref, extra)) == [102]
    # unbound parameters stay unknown instead of being guessed
    assert members(index, comp.compile_value(ref)) == [102]  # only `level 7` is known
    fu = compile_fn("", "aux.FaceupFilter(Card.IsSetCard,0x1205)", consts=consts)
    assert members(index, fu) == [101, 102]


def test_recursive_helpers_terminate(consts):
    src = """
function s.a(c) return s.b(c) and c:IsLevel(1) end
function s.b(c) return s.a(c) end"""
    flt = compile_fn(src, "s.a", consts=consts)
    assert Pred("level", ("eq", 1)) in flt.items
