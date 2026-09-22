"""Hidden information in decisions (audit finding): an agent must not learn the identity of cards
it cannot see. The core puts real passwords into SELECT_CARD / SELECT_TRIBUTE / SELECT_UNSELECT_CARD;
EDOPro's server strips opponent card codes from these before a player sees them, and so does the host.
"""

from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DuelConfig, default_scripts, expand_seed
from ygorl.env.encoding import ObservationEncoder

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)
MASKED = (M.SelectCard, M.SelectTribute, M.SelectUnselectCard)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


def hidden_from(viewer: int, loc: M.Location) -> bool:
    """EDOPro's public test (generic_duel.cpp MSG_MOVE): not in GY/overlay, and in deck/hand or face-down."""
    if loc.controller == viewer or loc.location & (C.LOCATION_GRAVE | C.LOCATION_OVERLAY):
        return False
    return bool(loc.location & (C.LOCATION_DECK | C.LOCATION_HAND) or loc.position & C.POS_FACEDOWN or not loc.position)


def test_hide_private_masks_only_hidden_opponent_cards():
    own = M.CardInfo(111, M.Location(0, C.LOCATION_MZONE, 0, C.POS_FACEDOWN_DEFENSE))
    opp_down = M.CardInfo(222, M.Location(1, C.LOCATION_MZONE, 1, C.POS_FACEDOWN_DEFENSE))
    opp_up = M.CardInfo(333, M.Location(1, C.LOCATION_MZONE, 2, C.POS_FACEUP_ATTACK))
    opp_set = M.CardInfo(444, M.Location(1, C.LOCATION_SZONE, 0, C.POS_FACEDOWN))
    opp_hand = M.CardInfo(555, M.Location(1, C.LOCATION_HAND, 0, 0))
    opp_grave = M.CardInfo(666, M.Location(1, C.LOCATION_GRAVE, 0, C.POS_FACEUP))
    d = M.SelectCard(0, False, 1, 1, (own, opp_down, opp_up, opp_set, opp_hand, opp_grave))
    masked = M.hide_private(d)
    assert [c.code for c in masked.cards] == [111, 0, 333, 0, 0, 666]
    assert [c.loc for c in masked.cards] == [c.loc for c in d.cards]
    no_pos = M.Location(1, C.LOCATION_MZONE, 3)  # SELECT_TRIBUTE carries no position: treated as hidden
    tribute = M.SelectTribute(0, False, 1, 1, (M.TributeOption(222, opp_down.loc, 1), M.TributeOption(111, own.loc, 2),
                                                M.TributeOption(777, no_pos, 1)))  # fmt: skip
    assert [c.code for c in M.hide_private(tribute).cards] == [0, 111, 0]
    unselect = M.SelectUnselectCard(0, True, False, 1, 1, (opp_down,), (opp_up,))
    m = M.hide_private(unselect)
    assert (m.selectable[0].code, m.unselectable[0].code) == (0, 333)
    other = M.SelectChain(0, 0, False, 0, 0, ())
    assert M.hide_private(other) is other


class Probe:
    """Random agent that checks every point and its encoding for hidden identities."""

    def __init__(self, seed, duel, encoder, stats):
        self.rng, self.duel, self.encoder, self.stats = RandomAgent(seed), duel, encoder, stats

    def act(self, point):
        if isinstance(point.decision, MASKED):
            obs = self.encoder.encode(point, self.duel._core)
            for i, a in enumerate(point.actions[:128]):
                if a.card is None or not hidden_from(point.player, a.card.loc):
                    continue
                self.stats["hidden"] += 1
                assert a.card.code == 0, (point.decision.name, a)
                row = obs["actions"][i]
                assert row[2] == 0, row  # no identity in the action row
                assert row[3] == 0 and row[4] == 0, row  # nor through its effect description
                if row[1]:
                    assert obs["cards"][row[1] - 1][7] == 0  # it still points at the (unknown) card row
                    self.stats["referenced"] += 1
        return self.rng.act(point)


def test_agents_and_encodings_never_see_hidden_card_identities(db, vocab):
    stats = {"hidden": 0, "referenced": 0}
    encoder = ObservationEncoder(db, vocab)
    for seed in range(16):
        a, b = NAMES[seed % 10], NAMES[(seed * 3 + 1) % 10]
        duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=3000))
        duel.run(Probe(seed, duel, encoder, stats), Probe(seed + 1000, duel, encoder, stats))
    assert stats["hidden"] > 20 and stats["referenced"] > 10, stats  # the scenario actually occurs


def test_cpp_host_masks_the_same_cards(db, vocab):
    """The C++ tracker decodes decisions itself: its actions must carry the same masked codes."""
    passwords = [vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(vocab))]
    seen = 0
    for seed in range(6):
        a, b = NAMES[seed % 10], NAMES[(seed * 3 + 1) % 10]
        duel = Duel(seed, None, DECKS[a], DECKS[b], cards=db, config=DuelConfig(max_decisions=3000))
        host = _core.HostDuel(db.to_core(), default_scripts(), passwords)
        cfg = duel.config
        host.start(expand_seed(seed), cfg.rule_flags, (8000, 5, 1), (8000, 5, 1),
                   [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
        rng = np.random.default_rng(seed)
        while not host.done():
            obs = host.observe()
            n = int(obs["action_mask"].sum())
            for row in obs["actions"][:n]:
                ref = row[1]
                if ref and obs["cards"][ref - 1][7] == 0:  # the action's card is hidden from the decider
                    seen += 1
                    assert row[2] == 0 and row[3] == 0 and row[4] == 0, row
            host.act(int(rng.integers(n)))
    assert seen > 5
