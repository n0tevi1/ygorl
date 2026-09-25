"""Tests for per-script fact extraction (effects, categories, target queries) (T5.3)."""

import pytest

from ygorl import paths
from ygorl.build.filters import ANY, Pred, hints
from ygorl.build.scripts import Query, analyze_script, load_constants


@pytest.fixture(scope="module")
def K():
    return load_constants()


def analyze(src, K, password=1000):
    return analyze_script(src, password, K)


def by_action(facts):
    out = {}
    for q in facts.queries:
        out.setdefault(q.action, []).append(q)
    return out


SEARCHER = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_TOHAND+CATEGORY_SEARCH)
	e1:SetType(EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_O)
	e1:SetCode(EVENT_SUMMON_SUCCESS)
	e1:SetTarget(s.thtg)
	e1:SetOperation(s.thop)
	c:RegisterEffect(e1)
	local e2=e1:Clone()
	e2:SetCode(EVENT_SPSUMMON_SUCCESS)
	c:RegisterEffect(e2)
	local e3=Effect.CreateEffect(c)
	e3:SetCategory(CATEGORY_SPECIAL_SUMMON)
	e3:SetTarget(s.sptg)
	e3:SetOperation(s.spop)
	c:RegisterEffect(e3)
end
s.listed_names={id,CARD_ALBAZ}
s.listed_series={SET_SNAKE_EYE}
function s.thfilter(c)
	return c:IsLevel(1) and c:IsAttribute(ATTRIBUTE_FIRE) and c:IsAbleToHand()
end
function s.thtg(e,tp,eg,ep,ev,re,r,rp,chk)
	if chk==0 then return Duel.IsExistingMatchingCard(s.thfilter,tp,LOCATION_DECK,0,1,nil) end
	Duel.SetOperationInfo(0,CATEGORY_TOHAND,nil,1,tp,LOCATION_DECK)
end
function s.thop(e,tp,eg,ep,ev,re,r,rp)
	Duel.Hint(HINT_SELECTMSG,tp,HINTMSG_ATOHAND)
	local g=Duel.SelectMatchingCard(tp,s.thfilter,tp,LOCATION_DECK,0,1,1,nil)
	if #g>0 then Duel.SendtoHand(g,nil,REASON_EFFECT) end
end
function s.spfilter(c,e,tp)
	return c:IsSetCard(SET_SNAKE_EYE) and c:IsCanBeSpecialSummoned(e,0,tp,false,false) and not c:IsCode(id)
end
function s.sptg(e,tp,eg,ep,ev,re,r,rp,chk)
	if chk==0 then return Duel.IsExistingMatchingCard(s.spfilter,tp,LOCATION_HAND|LOCATION_DECK,0,1,nil,e,tp) end
end
function s.spop(e,tp,eg,ep,ev,re,r,rp)
	local g=Duel.SelectMatchingCard(tp,s.spfilter,tp,LOCATION_HAND|LOCATION_DECK,0,1,1,nil,e,tp)
	Duel.SpecialSummon(g,0,tp,tp,false,false,POS_FACEUP)
end
"""


def test_effects_categories_and_listings(K):
    f = analyze(SEARCHER, K)
    assert f.effects == 3
    assert f.categories == K["CATEGORY_TOHAND"] | K["CATEGORY_SEARCH"] | K["CATEGORY_SPECIAL_SUMMON"]
    assert f.category_names() == ["tohand", "special_summon", "search"]
    assert f.listed_names == (1000, K["CARD_ALBAZ"])
    assert f.listed_series == (K["SET_SNAKE_EYE"],)


def test_search_and_special_summon_queries(K):
    f = analyze(SEARCHER, K)
    acts = by_action(f)
    assert set(acts) == {"to_hand", "special_summon"}
    (th,) = acts["to_hand"]  # the target-check and the selection use the same filter: deduplicated
    assert th.locations == K["LOCATION_DECK"]
    assert th.evidence == "filter"
    assert Pred("level", ("eq", 1)) in th.filter.items and Pred("attribute", (K["ATTRIBUTE_FIRE"],)) in th.filter.items
    assert th.categories & K["CATEGORY_SEARCH"]
    (sp,) = acts["special_summon"]
    assert sp.locations == K["LOCATION_HAND"] | K["LOCATION_DECK"]
    assert Pred("setcard", (K["SET_SNAKE_EYE"],)) in sp.filter.items
    assert f.stats["match_calls"] == 4 and f.stats["queries"] == 2


def test_action_from_select_hint_and_category_fallback(K):
    src = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_TOGRAVE)
	e1:SetOperation(s.op)
	c:RegisterEffect(e1)
	local e2=Effect.CreateEffect(c)
	e2:SetCategory(CATEGORY_SPECIAL_SUMMON)
	e2:SetTarget(s.sptg)
	c:RegisterEffect(e2)
end
function s.op(e,tp)
	Duel.Hint(HINT_SELECTMSG,tp,HINTMSG_TOGRAVE)
	local g=Duel.SelectMatchingCard(tp,s.cf,tp,LOCATION_DECK,0,1,1,nil)
	Duel.SendtoGrave(g,REASON_EFFECT)
end
function s.cf(c) return c:IsSetCard(0x10) end
function s.gyf(c) return c:IsRace(RACE_FIEND) end
function s.sptg(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
	if chk==0 then return Duel.IsExistingTarget(s.gyf,tp,LOCATION_GRAVE,0,1,nil) end
	local g=Duel.SelectTarget(tp,s.gyf,tp,LOCATION_GRAVE,0,1,1,nil)
end
"""
    f = analyze(src, K)
    acts = by_action(f)
    (tg,) = acts["to_grave"]
    assert tg.evidence == "hintmsg" and tg.locations == K["LOCATION_DECK"]
    (sp,) = acts["special_summon"]
    assert sp.evidence == "category" and sp.locations == K["LOCATION_GRAVE"]


def test_place_and_equip_from_deck(K):
    src = """
function s.initial_effect(c) end
function s.pf(c) return c:IsSetCard(0x10) and c:IsFieldSpell() end
function s.ef(c) return c:IsSetCard(0x20) end
function s.op(e,tp)
	Duel.Hint(HINT_SELECTMSG,tp,HINTMSG_TOFIELD)
	local g=Duel.SelectMatchingCard(tp,s.pf,tp,LOCATION_DECK,0,1,1,nil)
	Duel.Hint(HINT_SELECTMSG,tp,HINTMSG_EQUIP)
	local h=Duel.SelectMatchingCard(tp,s.ef,tp,LOCATION_DECK,0,1,1,nil)
end
"""
    assert sorted((q.action, q.evidence) for q in analyze(src, K).queries) == [
        ("equip", "hintmsg"),
        ("place", "hintmsg"),
    ]


def test_extra_call_arguments_reach_the_filter(K):
    src = """
function s.initial_effect(c) end
function s.spfilter(c,e,tp,code) return c:IsCode(code) and c:IsCanBeSpecialSummoned(e,0,tp,false,false) end
function s.tg(e,tp)
	return Duel.IsExistingMatchingCard(s.spfilter,tp,LOCATION_HAND|LOCATION_DECK,0,1,nil,e,tp,44632120)
		and Duel.IsExistingMatchingCard(s.spfilter,tp,LOCATION_HAND|LOCATION_DECK,0,1,nil,e,tp,71036835)
end
"""
    codes = sorted(q.filter.items[0].args for q in analyze(src, K).queries)
    assert codes == [(44632120,), (71036835,)]


def test_opponent_side_and_player_swap(K):
    src = """
function s.initial_effect(c) end
function s.f(c) return c:IsSetCard(0x10) and c:IsAbleToHand() end
function s.op(e,tp)
	local g=Duel.GetMatchingGroup(s.f,tp,0,LOCATION_MZONE,nil)
	local h=Duel.GetMatchingGroup(s.f,1-tp,0,LOCATION_GRAVE,nil)
end
"""
    f = analyze(src, K)
    assert [(q.action, q.locations) for q in f.queries] == [("to_hand", K["LOCATION_GRAVE"])]
    assert f.stats["opponent_side"] == 1


def test_unknown_location_falls_back_to_operation_info(K):
    src = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_SPECIAL_SUMMON)
	e1:SetTarget(s.tg)
	e1:SetOperation(s.op)
	c:RegisterEffect(e1)
end
function s.f(c,e,tp) return c:IsSetCard(0x10) and c:IsCanBeSpecialSummoned(e,0,tp,false,false) end
function s.tg(e,tp,eg,ep,ev,re,r,rp,chk)
	if chk==0 then return true end
	Duel.SetOperationInfo(0,CATEGORY_SPECIAL_SUMMON,nil,1,tp,LOCATION_DECK|LOCATION_GRAVE)
end
function s.op(e,tp)
	local loc=LOCATION_DECK
	local g=Duel.SelectMatchingCard(tp,s.f,tp,loc,0,1,1,nil,e,tp)
end
"""
    (q,) = analyze(src, K).queries
    assert q.action == "special_summon" and q.locations == K["LOCATION_DECK"] | K["LOCATION_GRAVE"]


def test_material_procedures(K):
    src = """
local s,id=GetID()
function s.initial_effect(c)
	Link.AddProcedure(c,aux.FilterBoolFunctionEx(Card.IsSetCard,SET_FIENDSMITH),2,2)
	Fusion.AddProcMix(c,true,true,CARD_ALBAZ,{11111111,22222222},aux.FilterBoolFunctionEx(Card.IsRace,RACE_DRAGON))
	Xyz.AddProcedure(c,aux.FilterBoolFunctionEx(Card.IsRace,RACE_FIEND),4,2)
	Synchro.AddProcedure(c,nil,1,1,Synchro.NonTuner(Card.IsSetCard,SET_TENPAI_DRAGON),1,99)
	Xyz.AddProcedure(c,nil,4,2)
end
s.material_setcode={SET_FIENDSMITH}
"""
    f = analyze(src, K, password=2000)
    mats = [q.filter for q in f.queries if q.action == "material"]
    assert Pred("setcard", (K["SET_FIENDSMITH"],)) in mats
    assert Pred("code", (K["CARD_ALBAZ"],)) in mats
    assert Pred("code", (11111111, 22222222)) in mats
    assert Pred("race", (K["RACE_DRAGON"],)) in mats
    assert any(
        Pred("race", (K["RACE_FIEND"],)) in getattr(m, "items", ()) and Pred("level", ("eq", 4)) in m.items
        for m in mats
    )
    assert Pred("setcard", (K["SET_TENPAI_DRAGON"],)) in mats
    assert Pred("level", ("eq", 4)) in mats  # generic Rank-4 Xyz: kept here, dropped later by fan-out
    assert set(f.material_codes) == {K["CARD_ALBAZ"], 11111111, 22222222}
    assert f.material_setcodes == (K["SET_FIENDSMITH"],)


def test_fusion_and_ritual_summon_procedures(K):
    src = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Fusion.CreateSummonEff({handler=c,fusfilter=aux.FilterBoolFunction(Card.ListsCodeAsMaterial,CARD_ALBAZ),extrafil=s.fextra})
	c:RegisterEffect(e1)
	Ritual.AddProcGreaterCode(c,4,nil,71408082)
	Ritual.AddProcGreater({handler=c,filter=aux.FilterBoolFunction(Card.IsSetCard,SET_NOUVELLES),location=LOCATION_HAND|LOCATION_GRAVE})
end
function s.fextra(e,tp,mg)
	return Duel.GetMatchingGroup(Fusion.IsMonsterFilter(Card.IsAbleToGrave),tp,LOCATION_DECK,0,nil)
end
"""
    f = analyze(src, K)
    sp = by_action(f)["special_summon"]
    fus = [q for q in sp if q.origin == "fusion_summon"]
    assert len(fus) == 1 and fus[0].locations == K["LOCATION_EXTRA"]
    assert Pred("lists_code_as_material", (K["CARD_ALBAZ"],)) in fus[0].filter.items
    rit = sorted((q for q in sp if q.origin == "ritual_summon"), key=lambda q: q.locations)
    assert rit[0].locations == K["LOCATION_HAND"] and Pred("code", (71408082,)) in rit[0].filter.items
    assert rit[1].locations == K["LOCATION_HAND"] | K["LOCATION_GRAVE"]
    assert f.categories & K["CATEGORY_FUSION_SUMMON"]
    (tg,) = by_action(f)["to_grave"]  # extra material from the Deck
    assert tg.locations == K["LOCATION_DECK"] and hints(tg.filter) == {"to_grave"}


def test_parse_failure_is_reported_not_raised(K):
    f = analyze("function s.initial_effect(c) local e1=Effect.CreateEffect(c) ", K)
    assert f.stats["parse_error"] == 1 and f.queries == ()


def test_real_script_snake_eye_ash(K):
    src = (paths.card_scripts() / "official" / "c9674034.lua").read_text()
    f = analyze(src, K, password=9674034)
    acts = by_action(f)
    assert [q.locations for q in acts["to_hand"]] == [K["LOCATION_DECK"]]
    assert [q.locations for q in acts["special_summon"]] == [K["LOCATION_HAND"] | K["LOCATION_DECK"]]
    assert f.listed_series == (K["SET_SNAKE_EYE"],)


def test_real_script_branded_fusion(K):
    src = (paths.card_scripts() / "official" / "c44362883.lua").read_text()
    f = analyze(src, K, password=44362883)
    fus = [q for q in f.queries if q.origin == "fusion_summon"]
    assert fus and Pred("lists_code_as_material", (K["CARD_ALBAZ"],)) in fus[0].filter.items


def test_query_is_hashable():
    q = Query("to_hand", 1, ANY, "call", 0, "filter")
    assert {q: 1}[Query("to_hand", 1, ANY, "call", 0, "filter")] == 1
