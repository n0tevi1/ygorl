"""Bound repeated legal fusion-material cancellations without changing engine action semantics."""

import json
from pathlib import Path

import numpy as np

from ygorl import _core
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, PlayerRules
from ygorl.env.encoding import ObservationEncoder


def test_real_fusion_cancel_cycle_is_bounded_in_both_hosts_and_survives_snapshot():
    data = json.loads((Path(__file__).parent / "data/material_cancel_loop.json").read_text())
    env = load_environment(data["environment"])
    cfg = DuelConfig(**{**data["config"], "player": PlayerRules(**data["config"]["player"])})

    def deck(d):
        return Deck(name=d["name"], main=tuple(d["main"]), extra=tuple(d["extra"]), side=tuple(d["side"]))

    duel = Duel(data["seed"], env, deck(data["deck_a"]), deck(data["deck_b"]),
                first=data["first"], config=cfg, snapshots=True)  # fmt: skip
    vocab = CardVocab.from_db(duel.cards)
    encoder = ObservationEncoder(duel.cards, vocab)
    passwords = [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]
    host = _core.HostDuel(duel.cards.to_core(), duel.scripts, passwords)
    p = cfg.player
    player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
    decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
    host.start(list(duel.core_seed), cfg.rule_flags, player, player, decks, cfg.max_turns, cfg.max_decisions)
    session = DuelSession(duel)

    def step(index):
        session.act(index)
        host.act(index)

    def masks():
        point = session.point
        assert host.player() == point.player
        py_obs, native_obs = encoder.encode(point, session.core), host.observe()
        for key, value in py_obs.items():
            np.testing.assert_array_equal(value, native_obs[key], err_msg=f"{key} at {point.index}")
        return point, py_obs["action_mask"]

    try:
        for index in data["actions"]:
            step(index)
        for cancellations in range(33):
            point, mask = masks()
            assert point.decision.TYPE == C.MSG_SELECT_UNSELECT_CARD
            cancel = next(i for i, a in enumerate(point.actions) if a.kind == "cancel")
            assert bool(mask[cancel]) == (cancellations < 32)
            assert any(mask[i] and a.kind == "select" for i, a in enumerate(point.actions))
            if cancellations < 32:
                step(cancel)
                assert session.point.decision.TYPE == C.MSG_SELECT_CARD
                assert len(session.point.actions) == 1
                step(0)  # the only fusion target; a new message must not reset the cancellation count
        snapshot = session.snapshot()
        selected = next(i for i, a in enumerate(session.point.actions) if a.kind == "select" and mask[i])
        session.act(selected)
        session.restore(snapshot)
        point, mask = masks()
        assert not mask[next(i for i, a in enumerate(point.actions) if a.kind == "cancel")]
        # The policy mask never changes raw legal actions or replay bytes: explicit old traces still play.
        step(cancel)
        assert session.point.decision.TYPE == C.MSG_SELECT_CARD
        assert host.result()["responses"] == session.tracker.result.responses
    finally:
        session.close()


def test_material_cancel_budget_resets_on_game_progress_player_or_decision_change():
    import struct

    from ygorl.engine.duel import default_cards
    from ygorl.engine.tracker import DuelTracker

    def record(kind, payload):
        body = bytes([kind]) + payload
        return struct.pack("<I", len(body)) + body

    def card(player):
        return struct.pack("<IBBII", 89631139, player, C.LOCATION_HAND, 0, 1)

    def target(player):
        return record(C.MSG_SELECT_CARD, struct.pack("<BBIII", player, 0, 1, 1, 1) + card(player))

    def material(player, *, empty=False):
        payload = struct.pack("<BBBIII", player, 0, 1, 1, 1, 0 if empty else 1)
        return record(C.MSG_SELECT_UNSELECT_CARD, payload + (b"" if empty else card(player)) + struct.pack("<I", 0))

    def feed(tracker, buffer):
        tracker.on_buffer(buffer, _core.DUEL_STATUS_AWAITING)
        return tracker.point()

    for reset in ("game_event", "player", "other_decision"):
        tracker = DuelTracker(DuelConfig(), 0, default_cards())
        for _ in range(32):
            feed(tracker, target(0))
            tracker.act(0)
            point = feed(tracker, material(0))
            cancel = next(i for i, a in enumerate(point.actions) if a.kind == "cancel")
            assert cancel not in point.undo
            tracker.act(cancel)
        # The target prompt and its hint are not game progress.
        feed(tracker, target(0))
        tracker.act(0)
        hint = record(C.MSG_HINT, struct.pack("<BBQ", 3, 0, 0))
        point = feed(tracker, hint + material(0))
        assert cancel in point.undo
        # Even an exhausted budget cannot remove the sole legal exit.
        point = feed(tracker, material(0, empty=True))
        assert [a.kind for a in point.actions] == ["cancel"] and not point.undo
        point = feed(tracker, material(0))
        assert cancel in point.undo
        if reset == "game_event":
            point = feed(tracker, record(C.MSG_NEW_PHASE, struct.pack("<H", C.PHASE_MAIN2)) + material(0))
        elif reset == "player":
            point = feed(tracker, material(1))
        else:
            feed(tracker, record(C.MSG_SELECT_YESNO, struct.pack("<BQ", 0, 0)))
            tracker.act(0)
            point = feed(tracker, material(0))
        assert next(i for i, a in enumerate(point.actions) if a.kind == "cancel") not in point.undo
