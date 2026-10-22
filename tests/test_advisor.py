"""Master Duel advisor (design 10): board snapshots, reconstruction in the core, suggestions."""

import random
from dataclasses import replace
from pathlib import Path

import pytest

from ygorl.advisor.advise import action_key, advise, masked
from ygorl.advisor.reconstruct import OpponentPrior, ReconstructError, board_script, build, complete
from ygorl.advisor.snapshot import (
    BoardSnapshot,
    SnapshotCard,
    SnapshotError,
    SnapshotSide,
    load_snapshot,
    snapshot_from_core,
)
from ygorl.agents.greedy import GreedyAgent
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelSession, default_cards

ROOT = Path(__file__).resolve().parents[1]
DECKS = ROOT / "tests" / "decks"
SNAPSHOT = ROOT / "tests" / "data" / "advisor_snapshot.json"
RYZEAL = ROOT / "environments" / "md-2026-09" / "meta" / "ryzeal-mitsurugi.ydk"
MALISS = ROOT / "environments" / "md-2026-09" / "meta" / "maliss.ydk"

# cards used in hand-written snapshots
BLUE_EYES = 89631139  # Blue-Eyes White Dragon, level 8
ASH = 14558127  # Ash Blossom & Joyous Spring
ZEUS = 90448279  # Divine Arsenal AA-ZEUS - Sky Thunder (Xyz, rank 12)
MYSTIC_MINE = 76375976  # a Field Spell


@pytest.fixture(scope="module")
def cards():
    return default_cards()


def side(lp=8000, cards=(), deck=20, extra=0):
    return SnapshotSide(lp=lp, cards=tuple(cards), deck_count=deck, extra_count=extra)


def snapshot(me=(), op=(), **kw):
    kw.setdefault("turn", 3)
    kw.setdefault("phase", "main1")
    return BoardSnapshot(sides=(side(cards=me), side(cards=op, lp=kw.pop("op_lp", 8000))), **kw)


# -- snapshots ---------------------------------------------------------------------------------------------------


def test_snapshot_json_roundtrip():
    snap = load_snapshot(SNAPSHOT)
    assert snap.sides[0].hand_count == 4
    assert BoardSnapshot.from_json(snap.to_json()) == snap
    assert load_snapshot(SNAPSHOT).dumps() == snap.dumps()


@pytest.mark.parametrize(
    "snap, msg",
    [
        (snapshot(turn=0), "turn must be at least 1"),
        (snapshot(phase="draw"), "phase must be one of"),
        (snapshot(me=[SnapshotCard("hand", BLUE_EYES, attacked=True)]), "only monsters in a Monster Zone attack"),
        (snapshot(me=[SnapshotCard("hand")]), "must be known"),
        (snapshot(op=[SnapshotCard("grave")]), "give the password"),
        (snapshot(op=[SnapshotCard("mzone", BLUE_EYES, 1), SnapshotCard("mzone", ASH, 1)]), "holds two cards"),
        (snapshot(op=[SnapshotCard("mzone", 0, 1, "faceup_attack")]), "face-up card must be known"),
        (snapshot(me=[SnapshotCard("szone", MYSTIC_MINE, 6)]), "out of range"),
        (snapshot(me=[SnapshotCard("deck", BLUE_EYES)]), "unknown location"),
        (snapshot(me=[SnapshotCard("hand", BLUE_EYES, materials=(ASH,))]), "only monsters in a Monster Zone"),
        (snapshot(turn=1, went_first=False), "turn 1 is the first player's turn"),
    ],
)
def test_snapshot_rejects_impossible_boards(snap, msg):
    with pytest.raises(SnapshotError, match=msg):
        snap.check()


def test_snapshot_json_errors():
    with pytest.raises(SnapshotError, match="unknown snapshot keys"):
        BoardSnapshot.from_json({"turn": 1, "sides": [], "clock": 3})
    with pytest.raises(SnapshotError, match="missing 'sides'"):
        BoardSnapshot.from_json({"turn": 1})
    with pytest.raises(SnapshotError, match="unknown card keys"):
        SnapshotCard.from_json({"location": "hand", "name": "x"})


# -- reconstruction ----------------------------------------------------------------------------------------------


def test_opponent_prior_filters_by_public_cards():
    a, b = Deck((1, 1, 2)), Deck((3,))
    prior = OpponentPrior((a, b), (0.5, 0.5))
    from collections import Counter

    assert prior.consistent(Counter({1: 2})) == [0]
    assert prior.consistent(Counter({1: 3})) == []
    assert prior.pick(Counter({3: 1}), random.Random(0)) == (1, True)
    assert prior.pick(Counter({9: 1}), random.Random(0))[1] is False


def test_complete_fills_hidden_cards_from_the_opponent_deck(cards):
    own, opp = load_ydk(RYZEAL), load_ydk(MALISS)
    snap = load_snapshot(SNAPSHOT)
    board = complete(snap, own, OpponentPrior((opp,), (1.0,)), cards, random.Random(0))
    assert board.consistent
    op = snap.sides[1]
    assert len(board.cards[1]) == len(op.cards) and len(board.decks[1]) == op.deck_count
    assert all(p in opp.main for p in board.decks[1])
    hidden = [placed.password for placed, card in zip(board.cards[1], op.cards) if not card.known]
    assert all(p in opp.main for p in hidden)
    # the advisor's deck is its list minus the cards already out of the deck
    assert len(board.decks[0]) == snap.sides[0].deck_count
    from collections import Counter

    assert not Counter(board.decks[0]) - (Counter(own.main) - Counter(snap.sides[0].seen()))


def test_reconstruction_restores_the_board(cards):
    """Export a self-play board, rebuild it and export again: the same cards, zones and LP."""
    env_decks = [load_ydk(DECKS / f"{n}.ydk") for n in ("snake_eye", "kashtira")]
    duel = Duel(3, None, *env_decks)
    session = DuelSession(duel)
    agent = GreedyAgent(3)
    while not session.done:
        p = session.point
        if (
            p.turn >= 3
            and p.player == p.turn_player
            and isinstance(p.decision, M.SelectIdleCmd)
            and p.phase == C.PHASE_MAIN1
        ):
            break
        session.act(agent.act(p))
    assert not session.done
    snap = snapshot_from_core(session.core, p).check()
    own, opp = duel.decks[duel.deck_of(p.player)], duel.decks[duel.deck_of(1 - p.player)]
    rec = build(complete(snap, own, OpponentPrior((opp,), (1.0,)), cards, random.Random(1)), C.DUEL_MODE_MR5,
                cards=cards)  # fmt: skip
    again = snapshot_from_core(rec.core, rec.point)

    def board(s):
        return [sorted((c.location, c.password, c.sequence, c.pos, c.materials) for c in sd.cards) for sd in s.sides]

    assert board(again) == board(snap)
    assert [(s.lp, s.deck_count, s.extra_count) for s in again.sides] == [(s.lp, s.deck_count, s.extra_count)
                                                                          for s in snap.sides]  # fmt: skip
    real = {action_key(a) for a in p.actions}
    rebuilt = {action_key(a) for a in rec.point.actions}
    assert len(real & rebuilt) / len(real | rebuilt) >= 0.8


def test_xyz_materials_counters_and_field_zone(cards):
    me = [SnapshotCard("mzone", ZEUS, 5, "faceup_attack", materials=(BLUE_EYES, BLUE_EYES)),
          SnapshotCard("szone", MYSTIC_MINE, 5), SnapshotCard("hand", ASH)]  # fmt: skip
    op = [SnapshotCard("hand"), SnapshotCard("szone", 0, 2, "facedown")]
    snap = snapshot(me, op, op_lp=1234).check()
    board = complete(snap, Deck((ASH,) * 3 + (BLUE_EYES,) * 3, (ZEUS,)), OpponentPrior((load_ydk(MALISS),), (1.0,)),
                     cards, random.Random(0))  # fmt: skip
    script = board_script(board)
    assert f"Debug.AddCard({BLUE_EYES},0,0,{C.LOCATION_MZONE},5," in script  # material onto the occupied zone
    rec = build(board, C.DUEL_MODE_MR5, cards=cards)
    again = snapshot_from_core(rec.core, rec.point)
    zeus = next(c for c in again.sides[0].cards if c.password == ZEUS)
    assert (zeus.sequence, zeus.materials) == (5, (BLUE_EYES, BLUE_EYES))
    assert any(c.password == MYSTIC_MINE and c.sequence == 5 for c in again.sides[0].cards)
    assert rec.point.lp == (8000, 1234)
    assert [c.location for c in again.sides[1].cards].count("hand") == 1


def test_battle_is_allowed_after_turn_one(cards):
    me = [SnapshotCard("mzone", BLUE_EYES, 2)]
    prior = OpponentPrior((load_ydk(MALISS),), (1.0,))
    kinds = {}
    for turn in (1, 3):
        board = complete(snapshot(me, turn=turn), Deck((BLUE_EYES,) * 40), prior, cards, random.Random(0))
        kinds[turn] = {a.kind for a in build(board, C.DUEL_MODE_MR5, cards=cards).point.actions}
    assert "battle_phase" not in kinds[1]
    assert "battle_phase" in kinds[3]


def test_opponent_turn_is_not_supported(cards):
    with pytest.raises(ReconstructError, match="own turn"):
        complete(snapshot(turn_player=1), Deck(), OpponentPrior((Deck((ASH,)),), (1.0,)), cards, random.Random(0))


# -- advice ------------------------------------------------------------------------------------------------------


def test_masks_what_the_core_cannot_be_told(cards):
    me = [SnapshotCard("hand", BLUE_EYES), SnapshotCard("mzone", BLUE_EYES, 1, position_locked=True),
          SnapshotCard("mzone", BLUE_EYES, 2)]  # fmt: skip
    prior = OpponentPrior((load_ydk(MALISS),), (1.0,))
    board = complete(snapshot(me, phase="main2", normal_summoned=True), Deck((BLUE_EYES,) * 40), prior, cards,
                     random.Random(0))  # fmt: skip
    rec = build(board, C.DUEL_MODE_MR5, cards=cards)
    snap = board.snapshot
    by_kind = {}
    for a in rec.point.actions:
        by_kind.setdefault(a.kind, []).append(masked(snap, a))
    assert by_kind["battle_phase"] == [True]
    assert all(by_kind.get("summon", [True])) and all(by_kind.get("mset", [True]))
    repos = {a.card.loc.sequence: masked(snap, a) for a in rec.point.actions if a.kind == "reposition"}
    assert repos == {1: True, 2: False}


def test_battle_phase_snapshot_reaches_the_battle_command(cards):
    """A Battle Phase snapshot is rebuilt at the battle command; a monster that already attacked is masked."""
    me = [SnapshotCard("mzone", BLUE_EYES, 1, attacked=True), SnapshotCard("mzone", BLUE_EYES, 2)]
    prior = OpponentPrior((load_ydk(MALISS),), (1.0,))
    board = complete(snapshot(me, phase="battle"), Deck((BLUE_EYES,) * 40), prior, cards, random.Random(0))
    rec = build(board, C.DUEL_MODE_MR5, cards=cards)
    assert isinstance(rec.point.decision, M.SelectBattleCmd)
    attacks = {a.card.loc.sequence: masked(board.snapshot, a) for a in rec.point.actions if a.kind == "attack"}
    assert attacks == {1: True, 2: False}
    assert {a.kind for a in rec.point.actions} >= {"attack", "main2", "end_phase"}


def test_advise_without_a_policy_lists_legal_actions(cards):
    snap = load_snapshot(SNAPSHOT)
    advice = advise(snap, load_ydk(RYZEAL), None, OpponentPrior((load_ydk(MALISS),), (1.0,)), C.DUEL_MODE_MR5,
                    cards=cards, samples=2)  # fmt: skip
    assert advice.consistent == 2 and not advice.policy
    assert {s.action.kind for s in advice.suggestions} >= {"summon", "end_phase"}
    assert sum(s.prob for s in advice.suggestions) == pytest.approx(1.0)
    assert all(s.legal_in == 2 for s in advice.suggestions)


def test_advise_with_a_policy(cards, tmp_path):
    torch = pytest.importorskip("torch")
    from ygorl.cards.cdb import CardVocab
    from ygorl.nets import NetConfig, PolicyNet
    from ygorl.nets.agent import save_checkpoint
    from ygorl.train.checkpoint import load_actor

    own, opp = load_ydk(RYZEAL), load_ydk(MALISS)
    vocab = CardVocab(sorted(set(own.main + own.extra + opp.main + opp.extra)))
    torch.manual_seed(0)
    net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=32, n_heads=2, board_layers=1, history_layers=1))
    path = save_checkpoint(tmp_path / "policy.pt", net, vocab, event_length=16)
    snap = replace(load_snapshot(SNAPSHOT), went_first=False)
    advice = advise(snap, own, load_actor(path), OpponentPrior((opp,), (1.0,)), C.DUEL_MODE_MR5, cards=cards,
                    samples=2)  # fmt: skip
    assert advice.policy
    probs = [s.prob for s in advice.suggestions]
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0)
