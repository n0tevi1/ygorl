"""Actual training limits caused by cancellation in a second SELECT_CARD target window."""

import json
from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, PlayerRules
from ygorl.env.encoding import ObservationEncoder


@pytest.mark.parametrize("update", (385, 400))
def test_real_target_cancel_cycle_is_bounded_with_native_parity_and_snapshot(update):
    data = json.loads((Path(__file__).parent / f"data/target_cancel_{update}.json").read_text())
    cfg = DuelConfig(**{**data["config"], "player": PlayerRules(**data["config"]["player"])})
    duel = Duel(data["seed"], load_environment(data["environment"]), Deck(**data["deck_a"]),
                Deck(**data["deck_b"]), first=data["first"], config=cfg, snapshots=True)  # fmt: skip
    vocab = CardVocab.from_db(duel.cards)
    encoder = ObservationEncoder(duel.cards, vocab)
    host = _core.HostDuel(duel.cards.to_core(), duel.scripts,
                         [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))])  # fmt: skip
    p = cfg.player
    player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
    host.start(list(duel.core_seed), cfg.rule_flags, player, player,
               [(list(m), list(e)) for m, e in duel.loaded_decks()], cfg.max_turns, cfg.max_decisions)  # fmt: skip
    session = DuelSession(duel)

    def step(index):
        session.act(index)
        host.act(index)

    def masks():
        point = session.point
        py_obs, native_obs = encoder.encode(point, session.core), host.observe()
        assert host.player() == point.player
        for key, value in py_obs.items():
            np.testing.assert_array_equal(value, native_obs[key], err_msg=f"{key} at {point.index}")
        return point, py_obs["action_mask"]

    try:
        for index in data["actions"]:
            step(index)
        for count in range(33):
            point, mask = masks()
            assert point.decision.TYPE == C.MSG_SELECT_CARD
            cancel = next(i for i, a in enumerate(point.actions) if a.kind == "cancel")
            assert bool(mask[cancel]) == (count < 32)
            if count < 32:
                step(cancel)
                assert len(session.point.actions) == 1 and session.point.actions[0].kind == "select"
                step(0)
        snapshot = session.snapshot()
        forward = next(i for i, a in enumerate(point.actions) if mask[i] and a.kind == "select")
        session.act(forward)
        assert any(type(e).__name__ not in ("Hint", "SelectCard") for e in session.point.events)
        session.restore(snapshot)
        point, mask = masks()
        assert not mask[cancel]
        # A replay can still submit old legal responses; the policy restriction changes no engine semantics.
        step(cancel)
        step(0)
        point, mask = masks()
        assert not mask[cancel]
        step(forward)
        masks()
        assert host.result()["responses"] == session.tracker.result.responses
    finally:
        session.close()
