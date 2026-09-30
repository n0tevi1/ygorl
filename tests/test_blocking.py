"""Blocking boards (docs/solver.md「阻断场面」): interruptions from script facts, board scores, automatic targets."""

from pathlib import Path

import pytest

from ygorl.build.scripts import analyze_script, load_constants
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.solver.blocking import (
    Interruption,
    blocking_plan,
    board_interruptions,
    card_interruptions,
    deck_pieces,
    interruptions_from_facts,
)

HERE = Path(__file__).parent
ASH, IMPERM, BARONNE, SPLK, CALLED, MAXX_C = 14558127, 10045474, 84815190, 29301450, 24224830, 23434538


@pytest.fixture(scope="module")
def K():
    return load_constants()


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


QUICK_NEGATE = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_NEGATE+CATEGORY_DESTROY)
	e1:SetType(EFFECT_TYPE_QUICK_O)
	e1:SetCode(EVENT_CHAINING)
	e1:SetRange(LOCATION_MZONE)
	e1:SetTarget(s.tg)
	e1:SetOperation(s.op)
	c:RegisterEffect(e1)
	local e2=e1:Clone()
	e2:SetRange(LOCATION_GRAVE)
	c:RegisterEffect(e2)
end
function s.tg(e,tp,eg,ep,ev,re,r,rp,chk) if chk==0 then return true end end
function s.op(e,tp,eg,ep,ev,re,r,rp) Duel.NegateActivation(ev) end
"""

QUICK_SEARCH = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_TOHAND+CATEGORY_SEARCH)
	e1:SetType(EFFECT_TYPE_QUICK_O)
	e1:SetCode(EVENT_FREE_CHAIN)
	e1:SetRange(LOCATION_MZONE)
	e1:SetOperation(s.op)
	c:RegisterEffect(e1)
	local e2=Effect.CreateEffect(c)
	e2:SetCategory(CATEGORY_DESTROY)
	e2:SetType(EFFECT_TYPE_IGNITION)
	e2:SetRange(LOCATION_MZONE)
	e2:SetTarget(s.destg)
	c:RegisterEffect(e2)
end
function s.op(e,tp,eg,ep,ev,re,r,rp)
	local g=Duel.SelectMatchingCard(tp,Card.IsAbleToHand,tp,LOCATION_DECK,0,1,1,nil)
end
function s.destg(e,tp,eg,ep,ev,re,r,rp,chk)
	if chk==0 then return Duel.IsExistingTarget(nil,tp,0,LOCATION_ONFIELD,1,nil) end
	Duel.SelectTarget(tp,nil,tp,0,LOCATION_ONFIELD,1,1,nil)
end
"""

TRAP_DESTROY = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_DESTROY)
	e1:SetType(EFFECT_TYPE_ACTIVATE)
	e1:SetCode(EVENT_FREE_CHAIN)
	e1:SetTarget(s.tg)
	c:RegisterEffect(e1)
end
function s.tg(e,tp,eg,ep,ev,re,r,rp,chk)
	local g=Duel.GetFieldGroup(tp,0,LOCATION_MZONE)
	if chk==0 then return #g>0 end
end
"""


def test_effect_facts_record_type_range_and_the_opponent(K):
    facts = analyze_script(QUICK_NEGATE, 1000, K)
    assert [(e.type, e.range) for e in facts.effect_facts] == [
        (K["EFFECT_TYPE_QUICK_O"], K["LOCATION_MZONE"]),
        (K["EFFECT_TYPE_QUICK_O"], K["LOCATION_GRAVE"]),  # the clone keeps the type, overrides the range
    ]
    assert facts.effect_facts[1].categories & K["CATEGORY_NEGATE"]
    search = analyze_script(QUICK_SEARCH, 1001, K).effect_facts
    assert [e.opponent for e in search] == [False, True]
    assert analyze_script(TRAP_DESTROY, 1002, K).effect_facts[0].opponent  # Duel.GetFieldGroup(tp, 0, ...)


def test_interruptions_need_opponent_turn_timing_and_a_disruptive_effect(K):
    monster = K["TYPE_MONSTER"] | K["TYPE_EFFECT"]
    assert interruptions_from_facts(analyze_script(QUICK_NEGATE, 1000, K), monster) == (
        Interruption("field", True),
        Interruption("grave", True),
    )
    # a quick search is not an interruption, an ignition destruction is not usable on the opponent's turn
    assert interruptions_from_facts(analyze_script(QUICK_SEARCH, 1001, K), monster) == ()
    trap = analyze_script(TRAP_DESTROY, 1002, K)
    assert interruptions_from_facts(trap, K["TYPE_TRAP"]) == (Interruption("set", False),)
    assert interruptions_from_facts(trap, K["TYPE_SPELL"]) == ()  # a normal spell cannot be activated when set


def test_real_cards(db):
    assert card_interruptions(ASH, db) == (Interruption("hand", True),)
    assert card_interruptions(IMPERM, db) == (Interruption("set", True),)
    assert card_interruptions(BARONNE, db) == (Interruption("field", True),)
    assert card_interruptions(SPLK, db) == (Interruption("field", False),)
    assert card_interruptions(CALLED, db) == (Interruption("set", False),)
    assert card_interruptions(MAXX_C, db) == ()


def test_board_score_counts_each_card_where_it_sits(db):
    up, down = C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_DEFENSE
    board = {"players": [{
        "mzone": [{"code": BARONNE, "position": up}, {"code": SPLK, "position": down}],
        "szone": [{"code": IMPERM, "position": down}, {"code": CALLED, "position": up}],
        "hand": [ASH, MAXX_C], "grave": [ASH],
    }, {}]}  # fmt: skip
    score = board_interruptions(board, db)
    # Baronne (field), Impermanence (set), Ash (hand); a face-down SPLK, a face-up Called and Ash in the GY do not
    assert score.key() == (3, 3)
    assert sorted(score.pieces) == sorted([(BARONNE, "field", True), (IMPERM, "set", True), (ASH, "hand", True)])


def test_plan_sets_the_hand_traps_and_ranks_engine_pieces_first(db):
    deck = load_ydk(HERE / "decks" / "voiceless_voice.ydk")
    pieces = deck_pieces(deck.main, deck.extra, db)
    assert pieces and pieces[0].engine and pieces[0].negate  # the deck's own boss before generic Extra Deck monsters
    assert all(p.password != ASH for p in pieces)  # hand traps are kept, not summoned
    assert [p.rank() for p in pieces] == sorted(p.rank() for p in pieces)
    plan = blocking_plan(deck.main, deck.extra, [ASH, IMPERM, MAXX_C], db)
    assert plan.base == [f"{ASH}@hand", f"{IMPERM}@szone:fd"]
