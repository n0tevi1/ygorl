"""Curriculum modes (solo / handtrap / full) and opening balance (T2.6)."""

import dataclasses
from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.curriculum import FULL, HANDTRAP, MODES, SOLO, allowed_actions, auto_action
from ygorl.engine.duel import Duel, DuelConfig
from ygorl.engine.replay import Replay
from ygorl.env import GameSpec, paired_specs, run_games
from ygorl.env.encoding import ObservationEncoder

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


# ------------------------------------------------------------------ filtering on hand-built decisions


def opt(code, loc, con=1, seq=0):
    return M.ChainOption(code, M.Location(con, loc, seq), code << 4, 0)


def chain(*options, forced=False):
    return M.SelectChain(1, 0, forced, 0, 0, tuple(options))


def acts(decision):
    return make_decision(decision).actions()


HAND_TRAP = opt(14558127, C.LOCATION_HAND)  # Ash Blossom from the hand
FIELD_QUICK = opt(84815190, C.LOCATION_MZONE)  # a monster's quick effect on the field
SET_TRAP = opt(40605147, C.LOCATION_SZONE)
GRAVE = opt(12345678, C.LOCATION_GRAVE)


def test_full_mode_never_restricts():
    d = chain(HAND_TRAP, FIELD_QUICK)
    assert allowed_actions(FULL, d, acts(d)) == [0, 1, 2]
    assert auto_action(FULL, d, acts(d)) is None
    only_pass = chain()
    assert auto_action(FULL, only_pass, acts(only_pass)) is None


def test_solo_passes_every_optional_chain():
    for d in (chain(), chain(HAND_TRAP), chain(FIELD_QUICK, SET_TRAP, GRAVE)):
        a = acts(d)
        assert allowed_actions(SOLO, d, a) == [len(a) - 1]
        assert a[auto_action(SOLO, d, a)].kind == "pass"


def test_handtrap_keeps_only_hand_activations():
    d = chain(FIELD_QUICK, HAND_TRAP, SET_TRAP, opt(23434538, C.LOCATION_HAND, seq=3), GRAVE)
    a = acts(d)
    allowed = allowed_actions(HANDTRAP, d, a)
    assert [a[i].kind for i in allowed] == ["chain", "chain", "pass"]
    assert [a[i].card.code for i in allowed[:2]] == [14558127, 23434538]
    assert auto_action(HANDTRAP, d, a) is None  # a real choice is left: ask the agent


def test_handtrap_without_hand_options_is_auto_passed():
    d = chain(FIELD_QUICK, SET_TRAP, GRAVE)
    a = acts(d)
    assert allowed_actions(HANDTRAP, d, a) == [3]
    assert a[auto_action(HANDTRAP, d, a)].kind == "pass"


@pytest.mark.parametrize("mode", [SOLO, HANDTRAP])
def test_forced_chain_is_not_restricted(mode):
    d = chain(FIELD_QUICK, GRAVE, forced=True)  # mandatory triggers: the core needs one of them
    a = acts(d)
    assert allowed_actions(mode, d, a) == [0, 1]
    assert auto_action(mode, d, a) is None


def test_effect_yes_no_depends_on_card_location():
    hand = M.SelectEffectYn(1, M.CardInfo(14558127, M.Location(1, C.LOCATION_HAND, 0)), 99)
    field = M.SelectEffectYn(1, M.CardInfo(84815190, M.Location(1, C.LOCATION_MZONE, 0)), 99)
    for d in (hand, field):
        a = acts(d)
        assert a[auto_action(SOLO, d, a)].kind == "no"
    assert allowed_actions(HANDTRAP, hand, acts(hand)) == [0, 1]
    assert auto_action(HANDTRAP, hand, acts(hand)) is None
    a = acts(field)
    assert a[auto_action(HANDTRAP, field, a)].kind == "no"


def test_other_passive_prompts():
    yes_no = M.SelectYesNo(1, 1234)
    a = acts(yes_no)
    assert a[auto_action(SOLO, yes_no, a)].kind == "no"
    assert auto_action(HANDTRAP, yes_no, a) is None  # not an activation choice: left to the agent
    cancelable = M.SelectCard(1, True, 1, 1, (M.CardInfo(1, M.Location(1, C.LOCATION_GRAVE, 0)),))
    a = acts(cancelable)
    assert a[auto_action(SOLO, cancelable, a)].kind == "cancel"
    assert auto_action(HANDTRAP, cancelable, a) is None


@pytest.mark.parametrize("mode", MODES)
def test_decisions_without_a_passive_answer_are_never_restricted(mode):
    must_pick = M.SelectCard(
        1,
        False,
        1,
        1,
        (M.CardInfo(1, M.Location(1, C.LOCATION_HAND, 0)), M.CardInfo(2, M.Location(1, C.LOCATION_HAND, 1))),
    )
    place = M.SelectPlace(1, 1, 0xFFFFFF00)
    for d in (must_pick, place, M.SelectOption(1, (5, 6))):
        a = acts(d)
        assert allowed_actions(mode, d, a) == list(range(len(a)))
        assert auto_action(mode, d, a) is None


def test_config_validation():
    assert DuelConfig().curriculum == FULL and DuelConfig().learner == 0 and not DuelConfig().augmented_start
    with pytest.raises(ValueError, match="curriculum"):
        DuelConfig(curriculum="goldfish")
    with pytest.raises(ValueError, match="learner"):
        DuelConfig(learner=2)


# ------------------------------------------------------------------ opening balance


def test_paired_specs_play_each_seed_from_both_seats():
    base = [GameSpec(seed=s, deck_a=DECKS["yubel"], deck_b=DECKS["tenpai"], first=1, config=DuelConfig(curriculum=SOLO))
            for s in (3, 4)]  # fmt: skip
    pairs = paired_specs(base)
    assert [(s.seed, s.first) for s in pairs] == [(3, 0), (3, 1), (4, 0), (4, 1)]
    assert all(s.deck_a is DECKS["yubel"] and s.config.curriculum == SOLO for s in pairs)


# ------------------------------------------------------------------ real games


class Watcher:
    """Random agent that keeps every decision point it is shown."""

    def __init__(self, seed):
        self.inner = RandomAgent(seed)
        self.points = []

    def act(self, point):
        self.points.append(point)
        return self.inner.act(point)


# (seed, deck_a, deck_b, first): the learner (deck a) sits first and second
GAMES = [(11, "snake_eye", "kashtira", 0), (12, "tenpai", "yubel", 1), (13, "branded_despia", "fiendsmith_ryzeal", 0),
         (14, "labrynth", "purrely", 1)]  # fmt: skip


def play(db, mode, seed, a, b, first, learner=0, max_turns=4):
    config = DuelConfig(curriculum=mode, learner=learner, max_turns=max_turns)
    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, config=config, record_messages=True)
    watchers = Watcher(seed), Watcher(seed + 1)
    result = duel.run(*watchers)
    learner_seat = next(p for p in (0, 1) if duel.deck_of(p) == learner)
    opponent_points = [p for p in watchers[1 - learner].points if p.turn_player == learner_seat]
    return duel, result, learner_seat, opponent_points


def opponent_chainings(result, learner_seat):
    """(MSG_CHAINING, location activated from) of the opponent while the learner is the turn player.

    A spell/trap activated from the hand (Infinite Impermanence) is first moved
    to the field, so its origin is the ``previous`` location of that move.
    """
    out, turn_player, moved = [], None, {}
    for buf in result.message_log:
        for msg in M.decode_buffer(buf):
            if isinstance(msg, M.NewTurn):
                turn_player = msg.player
            elif isinstance(msg, M.Decision):
                moved = {}
            elif isinstance(msg, M.Move):
                moved[msg.current] = msg.previous.location
            elif isinstance(msg, M.Chaining) and turn_player == learner_seat and msg.loc.controller != learner_seat:
                out.append((msg, moved.get(msg.loc, msg.loc.location)))
    return out


def forced_picks(points):
    return sum(1 for p in points if isinstance(p.decision, M.SelectChain) and p.decision.forced)


def test_solo_opponent_never_responds_in_learner_turn(db):
    autos = 0
    for seed, a, b, first in GAMES:
        for learner in (0, 1):
            duel, result, seat, opp = play(db, SOLO, seed, a, b, first, learner)
            assert result.reason != "error" and result.retries == 0
            assert not any(x.kind in ("pass", "no", "cancel") for p in opp for x in p.actions)
            assert len(opponent_chainings(result, seat)) <= forced_picks(opp)
            autos += result.auto_decisions
    assert autos > 0


def test_handtrap_opponent_only_chains_from_hand(db):
    hand_chains = 0
    for seed, a, b, first in GAMES:
        for learner in (0, 1):
            duel, result, seat, opp = play(db, HANDTRAP, seed, a, b, first, learner)
            assert result.reason != "error" and result.retries == 0
            for p in opp:
                d = p.decision
                if isinstance(d, M.SelectChain) and not d.forced:
                    assert all(x.card.loc.location == C.LOCATION_HAND for x in p.actions if x.kind == "chain")
                    assert any(x.kind == "chain" for x in p.actions)  # pass-only windows are answered by the host
                if isinstance(d, M.SelectEffectYn):
                    assert d.card.loc.location == C.LOCATION_HAND
            origins = [origin for _, origin in opponent_chainings(result, seat)]
            assert sum(o != C.LOCATION_HAND for o in origins) <= forced_picks(opp)
            hand_chains += origins.count(C.LOCATION_HAND)
    assert hand_chains > 0  # the test decks carry hand traps and random opponents do use them


def test_full_mode_is_unchanged(db):
    seed, a, b, first = GAMES[0]
    plain = Duel(seed, None, DECKS[a], DECKS[b], cards=db, first=first, config=DuelConfig(max_turns=4))
    ref = plain.run(RandomAgent(seed), RandomAgent(seed + 1))
    _, result, _, _ = play(db, FULL, seed, a, b, first)
    assert result.responses == ref.responses and result.auto_decisions == 0


def key(r):
    return (r.winner, r.reason, r.win_reason, r.turns, r.lp, r.decisions, r.auto_decisions, r.responses, r.actions)


@pytest.mark.parametrize("mode", MODES)
def test_pool_matches_sequential_duels(db, mode):
    specs = paired_specs([GameSpec(seed=s, deck_a=DECKS[a], deck_b=DECKS[b], config=DuelConfig(curriculum=mode, max_turns=4))
                          for s, a, b, _ in GAMES[:2]])  # fmt: skip

    def agents(i, spec):
        return RandomAgent(spec.seed), RandomAgent(spec.seed + 1)

    pooled = run_games(specs, agents, num_envs=3, num_threads=2, cards=db)
    for spec, r in zip(specs, pooled, strict=True):
        duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=db, first=spec.first, config=spec.config)
        assert key(r) == key(duel.run(*agents(0, spec)))
    if mode != FULL:
        assert any(r.auto_decisions for r in pooled)


@pytest.mark.parametrize("mode", [SOLO, HANDTRAP])
def test_replay_roundtrip_keeps_the_mode(db, mode, tmp_path):
    duel, result, _, _ = play(db, mode, 21, "snake_eye", "yubel", 1, learner=1)
    assert result.auto_decisions > 0
    rep = Replay.from_duel(duel, result)
    rep.save(tmp_path / "game.json")
    loaded = Replay.load(tmp_path / "game.json")
    assert loaded == rep and loaded.config() == duel.config
    again = loaded.play(cards=db, record_messages=True)
    assert again.message_log == result.message_log and again.responses == result.responses


def test_old_replays_load_as_full_mode(db):
    duel = Duel(5, None, DECKS["snake_eye"], DECKS["labrynth"], cards=db, config=DuelConfig(max_turns=2))
    result = duel.run(RandomAgent(1), RandomAgent(2))
    data = Replay.from_duel(duel, result).to_json()
    for k in ("curriculum", "learner", "augmented_start"):
        data.pop(k)
    rep = Replay.from_json(data)
    assert rep.config() == DuelConfig(max_turns=2)
    assert rep.play(cards=db).responses == result.responses


# ------------------------------------------------------------------ augmented-start flag


def test_augmented_start_flag_reaches_the_observation(db):
    encoder = ObservationEncoder(db, CardVocab.from_db(db))
    for flag in (False, True):
        config = DuelConfig(augmented_start=flag, max_decisions=5)
        duel = Duel(3, None, DECKS["snake_eye"], DECKS["kashtira"], cards=db, config=config)
        seen = []

        class Enc(RandomAgent):
            def act(self, point):
                seen.append((point.augmented_start, int(encoder.encode(point, duel._core)["globals"][21])))
                return super().act(point)

        duel.run(Enc(1), Enc(2))
        assert seen and all(s == (flag, int(flag)) for s in seen)


def test_augmented_start_is_stored_in_replays(db):
    config = dataclasses.replace(DuelConfig(max_turns=2), augmented_start=True)
    duel = Duel(8, None, DECKS["tenpai"], DECKS["purrely"], cards=db, config=config)
    rep = Replay.from_json(Replay.from_duel(duel, duel.run(RandomAgent(1), RandomAgent(2))).to_json())
    assert rep.augmented_start and rep.config().augmented_start
