"""Prespecified long-game and control sampling must retain its exact boundary."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from critic_continuation.run import behavior_selection


@pytest.mark.parametrize(
    "turns,decisions,seed,expected",
    [
        (20, 1000, 1, []),
        (21, 1000, 1, ["turns_over_20"]),
        (20, 1001, 1, ["decisions_over_1000"]),
        (2, 40, 64, ["seed_mod_64"]),
        (31, 1200, 128, ["turns_over_20", "decisions_over_1000", "seed_mod_64"]),
    ],
)
def test_behavior_selection(turns, decisions, seed, expected):
    game = SimpleNamespace(
        result={"turns": turns, "decisions": decisions}, assignment=SimpleNamespace(spec=SimpleNamespace(seed=seed))
    )
    assert behavior_selection(game) == expected


def test_selected_healthy_game_preserves_replayable_native_responses(tmp_path, monkeypatch):
    import gzip
    import json
    from pathlib import Path

    from critic_continuation import run
    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import Duel
    from ygorl.env import GameSpec
    from ygorl.env.encoded import EncodedVecEnv, chooser
    from ygorl.train.rollout import Assignment, FinishedGame

    decks = [load_ydk(Path(__file__).parent / "decks" / n) for n in ("snake_eye.ydk", "kashtira.ydk")]
    spec = GameSpec(64, *decks)
    result = EncodedVecEnv(1, 1, record_actions=True).play([spec], chooser)[0]
    assert result["reason"] == "win" and "action_indices" not in result
    game = FinishedGame(Assignment(spec), result["winner"], "win", False, 0, result)
    rollout = SimpleNamespace(games=[game])
    monkeypatch.setattr(run.Trainer, "_collect", lambda self: rollout)
    monkeypatch.setattr(run, "check_stop", lambda: None)
    trainer = run.AuditedTrainer.__new__(run.AuditedTrainer)
    trainer.run_dir = tmp_path
    trainer.cfg = SimpleNamespace(skip_forced=True)
    trainer.environment = None
    trainer.learner = SimpleNamespace(updates=129)
    trainer.counters = {"updates": 129}
    assert trainer._collect() is rollout
    with gzip.open(tmp_path / "behavior-traces.jsonl.gz", "rt") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 1
    record = records[0]
    assert "seed_mod_64" in record["behavior_selection"]
    assert record["training_update"] == 130 and record["replay_basis"] == "native_responses"
    assert record["action_indices"] is None
    responses = [bytes.fromhex(r) for r in record["responses"]]
    assert responses == result["responses"]
    replay = Duel(spec.seed, None, *decks, first=spec.first, config=spec.config).replay(responses)
    # Byte replay bypasses decision decoding, so its decision counter is intentionally zero.
    assert replay.responses == responses
    for key in ("winner", "reason", "turns"):
        assert getattr(replay, key) == record[key]
