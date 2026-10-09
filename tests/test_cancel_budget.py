"""Experimental masks must retain necessary exits and preserve engine semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cancel_budget import restrict_cancel
from ygorl import _core
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, PlayerRules
from ygorl.env.encoding import ObservationEncoder


def test_only_encoded_exit_survives_and_mask_never_enables_actions():
    point = SimpleNamespace(
        decision=SimpleNamespace(TYPE=C.MSG_SELECT_CARD),
        actions=[SimpleNamespace(kind=k) for k in ("select", "cancel")],
    )
    only_exit = {"action_mask": np.array([False, True]), "other": np.array([123])}
    obs, info = restrict_cancel(point, only_exit, 10, 0)
    assert obs is only_exit and info == {"intervened": 0, "sole_exit_preserved": 1}
    alternatives = {**only_exit, "action_mask": np.array([True, True, False])}
    obs, info = restrict_cancel(point, alternatives, 0, 0)
    np.testing.assert_array_equal(obs["action_mask"], [True, False, False])
    np.testing.assert_array_equal(alternatives["action_mask"], [True, True, False])
    assert obs["other"] is alternatives["other"] and info["intervened"] == 1
    # The native all-undo fallback can re-enable cancel and another undo action.
    # Baseline32 must preserve this mask exactly, even after its usual budget.
    obs, info = restrict_cancel(point, alternatives, 40, 32)
    assert obs is alternatives and info == {"intervened": 0, "sole_exit_preserved": 0}
    point.decision.TYPE = C.MSG_SELECT_CHAIN
    assert restrict_cancel(point, alternatives, 100, 0)[0] is alternatives


@pytest.mark.parametrize("budget", (32, 4, 1, 0))
@pytest.mark.parametrize("fixture", ("target_cancel_385", "target_cancel_400", "material_cancel_loop"))
def test_real_windows_with_python_native_masks_and_snapshot(fixture, budget):
    data = json.loads((Path(__file__).parent / "data" / (fixture + ".json")).read_text())
    cfg = DuelConfig(**{**data["config"], "player": PlayerRules(**data["config"]["player"])})
    duel = Duel(data["seed"], load_environment(data["environment"]), Deck(**data["deck_a"]), Deck(**data["deck_b"]),
                config=cfg, first=data["first"], snapshots=True)  # fmt: skip
    vocab = CardVocab.from_db(duel.cards)
    encoder = ObservationEncoder(duel.cards, vocab)
    host = _core.HostDuel(
        duel.cards.to_core(), duel.scripts, [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]
    )
    p = cfg.player
    player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
    host.start(
        list(duel.core_seed),
        cfg.rule_flags,
        player,
        player,
        [(list(m), list(e)) for m, e in duel.loaded_decks()],
        cfg.max_turns,
        cfg.max_decisions,
    )
    session = DuelSession(duel)

    def step(index):
        session.act(index)
        host.act(index)

    try:
        for idx in data["actions"]:
            step(idx)
        for count in range(budget + 1):
            point = session.point
            assert session.tracker._selection_cancels == count
            py = encoder.encode(point, session.core)
            native = host.observe()
            for key in py:
                np.testing.assert_array_equal(py[key], native[key])
            altered, info = restrict_cancel(point, native, count, budget)
            py_altered, _ = restrict_cancel(point, py, count, budget)
            np.testing.assert_array_equal(altered["action_mask"], py_altered["action_mask"])
            cancel = next(i for i, a in enumerate(point.actions) if a.kind == "cancel")
            assert bool(altered["action_mask"][cancel]) == (count < budget)
            assert np.any(altered["action_mask"])
            for key in native:
                if key != "action_mask":
                    np.testing.assert_array_equal(altered[key], native[key])
            if count < budget:
                step(cancel)
                # Cancellation returns to a target window; selecting again is not game progress.
                assert session.point.decision.TYPE == C.MSG_SELECT_CARD
                step(0)
        snapshot = session.snapshot()
        forward = next(i for i, a in enumerate(point.actions) if a.kind == "select" and altered["action_mask"][i])
        session.act(forward)
        session.restore(snapshot)
        assert session.tracker._selection_cancels == budget
        after, _ = restrict_cancel(session.point, host.observe(), session.tracker._selection_cancels, budget)
        np.testing.assert_array_equal(after["action_mask"], altered["action_mask"])
        # Explicit original actions are still legal engine responses under this inference restriction.
        step(cancel)
        assert session.point.decision.TYPE == C.MSG_SELECT_CARD
        assert host.result()["responses"] == session.tracker.result.responses
    finally:
        session.close()
