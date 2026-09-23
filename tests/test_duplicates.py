"""Equivalent-action deduplication in the action mask (docs/encoding.md 「等价动作去重」)."""

import numpy as np
import pytest

from ygorl import _core
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env.encoding import (
    A_ACTION,
    ACTION_KINDS,
    F_CARD,
    MAX_OPTIONS,
    N_CARDS,
    ObservationEncoder,
    action_representatives,
    canonical_action,
    mask_duplicates,
)

MAXX, CELTIC = 23434538, 91152256
DECK, HAND, MZONE, GRAVE, REMOVED = 1, 2, 3, 5, 6  # card-table location enum
CHAIN, SELECT, PASS = (ACTION_KINDS.index(k) + 1 for k in ("chain", "select", "pass"))


def card(index, location, sequence, public=0, position=0, visible=1):
    row = np.zeros(F_CARD, dtype=np.int32)
    row[[0, 1, 2, 6, 7, 8]] = index, location, sequence, position, visible, public
    return row


def action(kind, card_row, card_index, index, effect=0, value=0):
    row = np.zeros(A_ACTION, dtype=np.int32)
    row[[0, 1, 2, 3, 4, 8, 9]] = kind, card_row, card_index, card_index if effect else 0, effect, value, index + 1
    return row


def masked(cards, actions, decision=C.MSG_SELECT_CHAIN):
    table = np.zeros((N_CARDS, F_CARD), dtype=np.int32)
    table[: len(cards)] = cards
    acts = np.zeros((MAX_OPTIONS, A_ACTION), dtype=np.int32)
    acts[: len(actions)] = actions
    mask = np.zeros(MAX_OPTIONS, dtype=np.int32)
    mask[: len(actions)] = 1
    mask_duplicates(table, acts, mask, decision)
    return mask[: len(actions)].tolist()


def test_copies_in_the_hand_keep_only_the_first_row():
    cards = [card(7, HAND, 0), card(7, HAND, 1), card(9, HAND, 2), card(7, HAND, 3)]
    acts = [action(CHAIN, 1, 7, 0, effect=1), action(CHAIN, 2, 7, 1, effect=1), action(CHAIN, 3, 9, 2, effect=1),
            action(CHAIN, 4, 7, 3, effect=1), action(PASS, 0, 0, -1)]  # fmt: skip
    assert masked(cards, acts) == [1, 0, 1, 0, 1]


@pytest.mark.parametrize("second, row_change", [
    ("public", lambda c: card(7, HAND, 1, public=1)),  # one copy was revealed to the opponent
    ("field sequence", lambda c: card(7, MZONE, 1, position=1)),  # a different column
    ("hidden identity", None),  # the decider does not know what the second card is
    ("other effect", None),
])  # fmt: skip
def test_rows_that_differ_to_the_decider_are_kept(second, row_change):
    first_zone = MZONE if second == "field sequence" else HAND
    cards = [card(7, first_zone, 0, position=1 if first_zone == MZONE else 0), card(7, HAND, 1)]
    acts = [action(CHAIN, 1, 7, 0, effect=1), action(CHAIN, 2, 7, 1, effect=1)]
    if row_change is not None:
        cards[1] = row_change(None)
    if second == "hidden identity":
        acts[1] = action(CHAIN, 2, 0, 1)
        acts[0] = action(CHAIN, 1, 0, 0)
    if second == "other effect":
        acts[1] = action(CHAIN, 2, 7, 1, effect=2)
    assert masked(cards, acts) == [1, 1]


@pytest.mark.parametrize("zone", [GRAVE, REMOVED])
def test_graveyard_and_banished_copies_stay_apart(zone):
    """GY / banished order tells cards apart (newest last; the core keeps per-card state such as "sent to the GY
    this turn" that only the order reveals), so copies there are not merged."""
    cards = [card(7, zone, 0), card(7, zone, 1)]
    acts = [action(SELECT, 1, 7, 0), action(SELECT, 2, 7, 1)]
    assert masked(cards, acts, C.MSG_SELECT_CARD) == [1, 1]


def test_a_row_the_card_table_hides_is_never_merged():
    """Defence in depth: even with an identity in the action row, a card-table row not visible to the decider
    keeps its own action row (the mask must not reveal that two hidden cards are the same)."""
    cards = [card(7, HAND, 0, visible=0), card(7, HAND, 1, visible=0)]
    acts = [action(SELECT, 1, 7, 0), action(SELECT, 2, 7, 1)]
    assert masked(cards, acts, C.MSG_SELECT_CARD) == [1, 1]


def test_hand_copies_merge_but_values_that_matter_do_not():
    cards = [card(7, HAND, 0), card(7, HAND, 1), card(7, HAND, 2)]
    # SELECT_UNSELECT_CARD: value is the list index and is ignored
    acts = [action(SELECT, 1, 7, 0, value=0), action(SELECT, 2, 7, 1, value=1), action(SELECT, 3, 7, 2, value=2)]
    assert masked(cards, acts, C.MSG_SELECT_UNSELECT_CARD) == [1, 0, 0]
    # SELECT_SUM: value is the card's sum parameter (e.g. a modified level) and must match
    acts = [action(SELECT, 1, 7, 0, value=4), action(SELECT, 2, 7, 1, value=4), action(SELECT, 3, 7, 2, value=5)]
    assert masked(cards, acts, C.MSG_SELECT_SUM) == [1, 0, 1]


def test_canonical_action_maps_every_row_to_its_representative():
    cards = [card(7, HAND, 0), card(9, HAND, 1), card(7, HAND, 2)]
    table = np.zeros((N_CARDS, F_CARD), dtype=np.int32)
    table[:3] = cards
    acts = np.zeros((MAX_OPTIONS, A_ACTION), dtype=np.int32)
    acts[:4] = [action(CHAIN, 1, 7, 0), action(CHAIN, 2, 9, 1), action(CHAIN, 3, 7, 2), action(PASS, 0, 0, -1)]
    globals_ = np.zeros(22, dtype=np.int32)
    globals_[18], globals_[20] = C.MSG_SELECT_CHAIN, 4
    obs = {"cards": table, "actions": acts, "globals": globals_}
    assert [canonical_action(obs, i) for i in range(4)] == [0, 1, 0, 3]
    assert action_representatives(table, acts, 4, C.MSG_SELECT_CHAIN).tolist() == [0, 1, 0, 3]
    with pytest.raises(IndexError):
        canonical_action(obs, 4)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


class Collect:
    def __init__(self, encoder, duel, host=None):
        self.encoder, self.duel, self.host, self.seen = encoder, duel, host, []

    def act(self, point):
        obs = self.encoder.encode(point, self.duel._core)
        self.seen.append((point, obs))
        if self.host is not None:
            cpp = self.host.observe()
            for key in ("cards", "globals", "actions", "action_mask"):
                np.testing.assert_array_equal(np.asarray(cpp[key]).reshape(obs[key].shape), obs[key], err_msg=key)
        kinds = [a.kind for a in point.actions]
        choice = next((kinds.index(k) for k in ("pass", "end_phase", "no") if k in kinds), 0)
        if self.host is not None:
            self.host.act(choice)
        return choice


def test_maxx_c_in_hand_is_one_choice_in_python_and_cpp(db):
    vocab = CardVocab.from_db(db)
    deck_a, deck_b = Deck(main=(CELTIC,) * 40), Deck(main=(MAXX,) * 40)
    cfg = DuelConfig(max_turns=2, shuffle_decks=False)
    duel = Duel(1, None, deck_a, deck_b, cards=db, config=cfg)
    host = _core.HostDuel(db.to_core(), default_scripts(), [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))])
    decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
    host.start(expand_seed(1), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1), decks, cfg.max_turns, cfg.max_decisions)
    agent = Collect(ObservationEncoder(db, vocab), duel, host)
    duel.run(agent, agent)
    offers = [(p, o) for p, o in agent.seen if p.player == 1 and sum(a.card is not None and a.card.code == MAXX
                                                                    for a in p.actions) >= 5]  # fmt: skip
    assert offers
    for point, obs in offers:
        live = [point.actions[i] for i in np.flatnonzero(obs["action_mask"])]
        maxx = [a for a in live if a.card is not None and a.card.code == MAXX]
        # one row per distinct (kind, effect): Maxx "C" has a single effect, so one chain / activate / summon / set row
        assert len({(a.kind, a.description) for a in maxx}) == len(maxx) >= 1
        assert obs["globals"][20] == len(point.actions)  # the raw legal count is kept
