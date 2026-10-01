"""Survival boards (docs/solver.md「存活场面」): the strict interruption counter and the turn-2 board evaluator."""

from pathlib import Path

import pytest

from ygorl.build.scripts import analyze_script, load_constants
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.solver.blocking import board_interruptions, card_interruptions, interruptions_from_facts

HERE = Path(__file__).parent
ASH, FUWALOS, DROLL, XYZ_REFLECT, IMPERM, SPLK, BARONNE = (
    14558127,
    42141493,
    94145021,
    2371506,
    10045474,
    29301450,
    84815190,
)
PEARL = 71594310  # Gem-Knight Pearl: a vanilla Xyz monster


@pytest.fixture(scope="module")
def K():
    return load_constants()


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


LOCK = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_DISABLE)
	e1:SetType(EFFECT_TYPE_QUICK_O)
	e1:SetCode(EVENT_FREE_CHAIN)
	e1:SetRange(LOCATION_HAND)
	e1:SetCountLimit(1,{id,1})
	e1:SetCost(s.cost)
	e1:SetOperation(s.op)
	c:RegisterEffect(e1)
end
function s.cost(e,tp,eg,ep,ev,re,r,rp,chk) Duel.SendtoGrave(e:GetHandler(),REASON_COST) end
function s.op(e,tp,eg,ep,ev,re,r,rp)
	local e1=Effect.CreateEffect(e:GetHandler())
	e1:SetType(EFFECT_TYPE_FIELD)
	e1:SetCode(EFFECT_CANNOT_TO_HAND)
	Duel.RegisterEffect(e1,tp)
end
"""

NEEDS_A_DRAGON = """
local s,id=GetID()
function s.initial_effect(c)
	local e1=Effect.CreateEffect(c)
	e1:SetCategory(CATEGORY_TOHAND)
	e1:SetType(EFFECT_TYPE_ACTIVATE)
	e1:SetCode(EVENT_FREE_CHAIN)
	e1:SetTarget(s.tg)
	e1:SetOperation(function(e,tp) Duel.SendtoHand(Duel.GetFirstTarget(),nil,REASON_EFFECT) end)
	c:RegisterEffect(e1)
end
function s.filter(c) return c:IsFaceup() and c:IsRace(RACE_DRAGON) end
function s.tg(e,tp,eg,ep,ev,re,r,rp,chk)
	if chk==0 then return Duel.IsExistingTarget(s.filter,tp,LOCATION_MZONE,0,1,nil)
		and Duel.IsExistingTarget(Card.IsAbleToHand,tp,0,LOCATION_ONFIELD,1,nil) end
end
"""


def test_effect_facts_record_limits_actions_and_requirements(K):
    lock = analyze_script(LOCK, 1000, K).effect_facts[0]
    assert lock.count_code == 1000 and not lock.acts  # {id, 1}: a hard limit; a cost does not count as acting
    tidy = analyze_script(NEEDS_A_DRAGON, 1001, K).effect_facts[0]
    assert tidy.acts and tidy.count_code == 0 and len(tidy.requires) == 1
    trap = K["TYPE_TRAP"]
    facts = analyze_script(NEEDS_A_DRAGON, 1001, K)
    assert interruptions_from_facts(facts, trap)[0].requires == ()  # the loose counter does not carry them
    assert interruptions_from_facts(facts, trap, strict=True)[0].requires == tidy.requires
    monster = K["TYPE_MONSTER"] | K["TYPE_EFFECT"]
    assert interruptions_from_facts(analyze_script(LOCK, 1000, K), monster)  # loose: a hand "negation"
    assert interruptions_from_facts(analyze_script(LOCK, 1000, K), monster, strict=True) == ()


def test_strict_counter_on_real_cards(db):
    for code in (FUWALOS, DROLL):  # a draw engine and a player lock
        assert card_interruptions(code, db) and not card_interruptions(code, db, strict=True)
    for code in (ASH, IMPERM, SPLK, BARONNE):
        assert [(i.where, i.negate) for i in card_interruptions(code, db, strict=True)] == [
            (i.where, i.negate) for i in card_interruptions(code, db)
        ]
    (reflect,) = card_interruptions(XYZ_REFLECT, db, strict=True)
    assert reflect.where == "set" and reflect.requires  # "targets an Xyz monster you control"


def test_strict_board_counts_names_once_and_checks_requirements(db):
    up, down = C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_DEFENSE

    def board(mzone, szone, hand):
        return {"players": [{"mzone": [{"code": c, "position": up} for c in mzone],
                             "szone": [{"code": c, "position": down} for c in szone], "hand": hand, "grave": []}, {}]}  # fmt: skip

    b = board([], [XYZ_REFLECT, IMPERM], [ASH, ASH, FUWALOS])
    assert board_interruptions(b, db).interruptions == 5
    assert board_interruptions(b, db, strict=True).interruptions == 2  # Impermanence + one Ash
    with_xyz = board([PEARL], [XYZ_REFLECT], [])
    assert board_interruptions(with_xyz, db, strict=True).interruptions == 1


def test_survival_score_and_board_features(db):
    from ygorl.solver.survival import PILOT_WEIGHTS, SCORE_WEIGHTS, board_features, survival_score

    up = C.POS_FACEUP_ATTACK
    side = lambda m, h, lp: {"mzone": [{"code": c, "position": up, "sequence": i} for i, c in enumerate(m)],  # noqa: E731
                             "szone": [], "hand": h, "grave": [], "lp": lp}  # fmt: skip
    t1 = {"players": [side([BARONNE, SPLK], [ASH], 8000), side([], [], 8000)]}
    t2 = {"players": [side([BARONNE], [ASH, IMPERM], 6000), side([SPLK], [], 8000)]}
    f = board_features(t2, db, t1)
    assert (f["monsters"], f["survivors"], f["hand"], f["lp"], f["opp_board"]) == (1, 1, 2, 6000, 1)
    assert survival_score({"t2": f}, PILOT_WEIGHTS) == pytest.approx(1 + 0.5 * 2 + 0.75 - 0.25 + 2)
    w = SCORE_WEIGHTS
    want = w["monsters"] + 2 * w["hand"] + 0.75 * w["lp"] + w["opp_board"] + f["interruptions"] * w["interruptions"]
    assert survival_score({"t2": f}) == pytest.approx(want + w["alive"])
    dead = {**f, "lp": 0}
    assert survival_score({"t2": dead}) == pytest.approx(w["opp_board"])


def test_boss_pieces_prefer_the_engine(db):
    from ygorl.solver.survival import boss_pieces

    deck = load_ydk(HERE / "decks" / "voiceless_voice.ydk")
    bosses = boss_pieces(deck.main, deck.extra, db, n=2)
    assert len(bosses) == 2 and all(b in {db.canonical(c) for c in deck.extra} for b in bosses)


def test_play_line_follows_a_line_against_a_real_deck(db, tmp_path):
    torch = pytest.importorskip("torch")
    from ygorl.agents.random_agent import RandomAgent
    from ygorl.cards.cdb import CardVocab
    from ygorl.engine.duel import Duel, DuelConfig
    from ygorl.engine.replay import Replay, load_yrp
    from ygorl.nets import NetConfig, PolicyNet
    from ygorl.nets.agent import save_checkpoint
    from ygorl.solver.demo import Demonstration, convert_line
    from ygorl.solver.survival import Opponent, play_line, survival_score

    snake, kash = load_ydk(HERE / "decks" / "snake_eye.ydk"), load_ydk(HERE / "decks" / "kashtira.ydk")
    duel = Duel(41, None, snake, kash, cards=db, config=DuelConfig(max_turns=1))
    result = duel.run(RandomAgent(41), RandomAgent(42))
    Replay.from_duel(duel, result).to_yrpx(tmp_path / "g.yrpX")
    yrp = load_yrp(tmp_path / "g.yrpX")
    line = convert_line(yrp, [], responses=yrp.replayable().responses[:8], cards=db)
    demo = Demonstration.start_from(yrp, deck=snake, hand=[1], hand_index=0, hand_seed=1, variant="plain",
                                    targets=[], environment=None)  # fmt: skip
    demo.lines.append(line)
    torch.manual_seed(0)
    vocab = CardVocab.from_db(db)
    net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=16, n_heads=2, board_layers=1, history_layers=1))
    ckpt = save_checkpoint(tmp_path / "p.pt", net, vocab, event_length=16)
    opp = Opponent(load_ydk(HERE / "decks" / "yubel.ydk"), 7, 8)
    play = play_line(demo, 0, opp, str(ckpt), cards=db)
    assert not play.error, play.error
    assert 0 < play.followed <= play.line_steps  # the rest answers template-opponent moves that never happen
    assert play.t1 and play.t2 and (play.t2["turn"] == 3 or play.end_turn)
    assert play_line(demo, 0, opp, str(ckpt), cards=db).to_json() == play.to_json()  # paired samples repeat
    assert isinstance(survival_score(play), float)
