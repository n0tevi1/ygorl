"""Observation encoding: query parsing and the Python reference encoder (T2.2)."""

from pathlib import Path

import numpy as np
import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel
from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location
from ygorl.env.encoding import ACTION_KINDS, MAX_OPTIONS, N_CARDS, ObservationEncoder

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
COL = {name: i for i, name in enumerate(
    ["card_index", "location", "sequence", "overlay_index", "controller", "owner", "position", "visible", "public",
     "type", "attribute", "race", "level", "rank", "link", "lscale", "rscale", "attack", "defense", "link_marker",
     "counters", "materials", "disabled"])}  # fmt: skip
ACOL = {name: i for i, name in enumerate(
    ["kind", "card_row", "card_index", "effect_card", "effect_index", "system_string", "position", "zone", "value", "index"])}  # fmt: skip


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def encoder(db):
    return ObservationEncoder(db, CardVocab.from_db(db))


class Capture:
    """Random agent that encodes every decision point it is shown."""

    def __init__(self, seed, encoder, duel, limit=None):
        self.inner, self.encoder, self.duel, self.limit = RandomAgent(seed), encoder, duel, limit
        self.samples = []

    def act(self, point):
        if self.limit is None or len(self.samples) < self.limit:
            self.samples.append((point, self.encoder.encode(point, self.duel._core)))
        return self.inner.act(point)


def play(db, encoder, seed, a="snake_eye", b="kashtira", limit=None, max_decisions=400):
    from ygorl.engine.duel import DuelConfig

    duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=max_decisions))
    agents = Capture(seed, encoder, duel, limit), Capture(seed + 1, encoder, duel, limit)
    duel.run(*agents)
    return agents[0].samples + agents[1].samples


def test_parse_query_location(db):
    duel = Duel(1, None, DECKS["snake_eye"], DECKS["yubel"], cards=db)
    seen = {}

    class Peek(RandomAgent):
        def act(self, point):
            if not seen:
                core = duel._core
                seen["hand"] = parse_query_location(core.query_location(CARD_QUERY_FLAGS, 0, C.LOCATION_HAND))
                seen["mzone"] = parse_query_location(core.query_location(CARD_QUERY_FLAGS, 0, C.LOCATION_MZONE))
            return super().act(point)

    duel.run(Peek(1), Peek(2))
    hand, mzone = seen["hand"], seen["mzone"]
    assert len(hand) == 5 and all(c is not None for c in hand)
    assert all(c["code"] in db and c["public"] in (0, 1) for c in hand)
    assert len(mzone) == 7 and all(c is None for c in mzone)


def test_shapes_and_dtypes(db, encoder):
    point, obs = play(db, encoder, 3, limit=1)[0]
    assert obs["cards"].shape == (N_CARDS, 23) and obs["cards"].dtype == np.int32
    assert obs["globals"].shape == (22,) and obs["actions"].shape == (MAX_OPTIONS, 10)
    assert obs["action_mask"].shape == (MAX_OPTIONS,)
    assert 0 < obs["action_mask"].sum() <= min(len(point.actions), MAX_OPTIONS)  # equivalent copies are masked
    assert obs["action_mask"][0] and not obs["action_mask"][len(point.actions):].any()
    assert obs["globals"][20] == len(point.actions)


def test_first_decision_content(db, encoder):
    vocab = encoder.vocab
    point, obs = play(db, encoder, 5, limit=1)[0]
    assert point.player == 0 and point.turn == 1
    cards, g = obs["cards"], obs["globals"]
    rows = cards[cards[:, COL["card_index"]] != 0]
    own_hand = rows[(rows[:, COL["location"]] == 2) & (rows[:, COL["controller"]] == 0)]
    opp_hand = rows[(rows[:, COL["location"]] == 2) & (rows[:, COL["controller"]] == 1)]
    own_deck = rows[(rows[:, COL["location"]] == 1)]
    own_extra = rows[(rows[:, COL["location"]] == 7) & (rows[:, COL["controller"]] == 0)]
    assert len(own_hand) == 5 and own_hand[:, COL["visible"]].all() and (own_hand[:, COL["card_index"]] > 1).all()
    assert len(opp_hand) == 5 and not opp_hand[:, COL["visible"]].any() and (opp_hand[:, COL["card_index"]] == 1).all()
    assert len(own_deck) == 35 and (own_deck[:, COL["sequence"]] == 0).all() and (own_deck[:, COL["controller"]] == 0).all()
    assert list(own_deck[:, COL["card_index"]]) == sorted(own_deck[:, COL["card_index"]])
    assert len(own_extra) == 15
    assert not ((rows[:, COL["location"]] == 1) & (rows[:, COL["controller"]] == 1)).any()  # no opponent deck rows
    assert len(rows) == 5 + 5 + 35 + 15
    deck = DECKS["snake_eye"]
    assert sorted(own_hand[:, 0].tolist() + own_deck[:, 0].tolist()) == sorted(vocab.index(p) for p in deck.main)
    assert g[0] == 0 and g[1] == 1 and g[2] == 1 and g[3] == 1
    assert g[5] == g[6] == 8000
    assert list(g[7:17]) == [35, 5, 0, 0, 15, 35, 5, 0, 0, 15]
    assert g[18] == point.decision.TYPE and g[20] == len(point.actions)


def test_hidden_rows_carry_no_identity(db, encoder):
    for point, obs in play(db, encoder, 9, a="labrynth", b="tenpai"):
        cards = obs["cards"]
        hidden = cards[(cards[:, COL["card_index"]] != 0) & (cards[:, COL["visible"]] == 0)]
        assert (hidden[:, COL["card_index"]] == 1).all()
        assert not hidden[:, COL["type"]:].any()


def test_action_rows_point_at_matching_cards(db, encoder):
    vocab = encoder.vocab
    checked = 0
    for point, obs in play(db, encoder, 13, a="branded_despia", b="purrely"):
        cards, acts = obs["cards"], obs["actions"]
        for i, action in enumerate(point.actions[:MAX_OPTIONS]):
            row = acts[i]
            assert row[ACOL["kind"]] == ACTION_KINDS.index(action.kind) + 1
            if action.card is not None and action.card.code:
                assert row[ACOL["card_index"]] == vocab.index(action.card.code)
                if row[ACOL["card_row"]]:
                    target = cards[row[ACOL["card_row"]] - 1]
                    if target[COL["visible"]]:
                        assert target[COL["card_index"]] == vocab.index(action.card.code)
                        checked += 1
    assert checked > 50


def test_effect_ids(db, encoder):
    vocab = encoder.vocab
    found = 0
    for point, obs in play(db, encoder, 17, a="snake_eye", b="fiendsmith_ryzeal"):
        for i, action in enumerate(point.actions[:MAX_OPTIONS]):
            desc = action.description  # aux.Stringid(code, n) == code << 20 | n (utility.lua)
            if action.kind in ("activate", "chain") and desc >> 20 in db:
                row = obs["actions"][i]
                assert row[ACOL["effect_card"]] == vocab.index(desc >> 20)
                assert row[ACOL["effect_index"]] == (desc & 0xFFFFF) + 1 and row[ACOL["system_string"]] == 0
                found += 1
    assert found > 0


def test_encoding_is_deterministic(db, encoder):
    a = play(db, encoder, 21, limit=30)
    b = play(db, encoder, 21, limit=30)
    for (_, x), (_, y) in zip(a, b):
        for k in x:
            np.testing.assert_array_equal(x[k], y[k])


def test_turn_player_is_tracked(db, encoder):
    for point, obs in play(db, encoder, 25, limit=60):
        assert point.turn_player in (0, 1)
        assert obs["globals"][2] == int(point.turn_player == point.player)
