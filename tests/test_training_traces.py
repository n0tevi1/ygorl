"""Exact native replay of training limits, including choices spanning multiple PPO updates."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.ydk import Deck, load_ydk  # noqa: E402
from ygorl.data.environment import PlayerRules  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.env import GameSpec  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv, chooser  # noqa: E402
from ygorl.train.ppo import PPOConfig  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer  # noqa: E402

DECKS = tuple(str(Path(__file__).parent / "decks" / n) for n in ("snake_eye.ydk", "kashtira.ydk"))


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_training_limit_retains_cross_update_actions_for_native_replay(tmp_path):
    cfg = TrainConfig(decks=DECKS, net={"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1},
                      num_envs=1, steps=1, event_length=8, max_decisions=24, eval_every=0, checkpoint_every=0,
                      ppo=PPOConfig(epochs=1, minibatch_size=1))  # fmt: skip
    trainer = Trainer(cfg, tmp_path)
    for _ in range(32):
        trainer.step()
        if trainer.counters["truncated"]:
            break
    assert trainer.counters["truncated"] > 0
    record = json.loads((tmp_path / "truncations.jsonl").read_text().splitlines()[0])
    assert record["reason"] == "decision_limit" and len(record["action_indices"]) > cfg.steps
    spec = dict(record["spec"])
    for key in ("deck_a", "deck_b"):
        d = spec[key]
        spec[key] = Deck(name=d["name"], main=tuple(d["main"]), extra=tuple(d["extra"]), side=tuple(d["side"]))
    config = dict(spec["config"])
    config["player"] = PlayerRules(**config["player"])
    spec["config"] = DuelConfig(**config)
    env = EncodedVecEnv(1, 1, vocab=trainer.vocab, event_length=cfg.event_length, skip_forced=record["skip_forced"])
    env.reset(0, GameSpec(**spec))
    index = 0
    while True:
        ev = env.recv()[0]
        if ev.result is not None:
            result = ev.result
            break
        action = record["action_indices"][index]
        assert ev.obs["action_mask"][action]
        env.step(0, action)
        index += 1
    assert index == len(record["action_indices"])
    for key in ("reason", "winner", "turns", "decisions"):
        assert result[key] == record[key]
    assert [bytes(r).hex() for r in result["responses"]] == record["responses"]


@pytest.mark.parametrize("slots", [1, 2])
def test_optional_traces_preserve_games_and_reset_between_slots(slots):
    decks = [load_ydk(p) for p in DECKS]
    normal = GameSpec(37, *decks)
    limited = replace(normal, config=DuelConfig(max_decisions=24))
    plain = EncodedVecEnv(slots, 1, skip_forced=True)
    traced = EncodedVecEnv(slots, 1, skip_forced=True, record_actions=True)
    a = plain.play([normal, limited], chooser)
    b = traced.play([normal, limited], chooser)
    assert b[0]["reason"] == "win" and "action_indices" not in b[0]
    actions = b[1].pop("action_indices")
    assert actions and a == b
    fresh = EncodedVecEnv(1, 1, skip_forced=True, record_actions=True).play([limited], chooser)[0]
    assert fresh.pop("action_indices") == actions and fresh == a[1]


def test_error_trace_includes_the_attempted_action():
    env = EncodedVecEnv(1, 1, record_actions=True)
    env.reset(0, GameSpec(37, *(load_ydk(p) for p in DECKS)))
    ev = env.recv()[0]
    valid = int(np.flatnonzero(ev.obs["action_mask"])[0])
    env.step(0, valid)
    assert env.recv()[0].result is None
    env.step(0, 99999)
    result = env.recv()[0].result
    assert result["reason"] == "error" and result["action_indices"] == [valid, 99999]


def test_unhealthy_win_is_preserved_as_a_diagnostic(tmp_path):
    from ygorl.train.rollout import Assignment, FinishedGame

    cfg = TrainConfig(decks=DECKS, net={"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1},
                      num_envs=1, steps=1, event_length=8, eval_every=0)  # fmt: skip
    trainer = Trainer(cfg, tmp_path)
    spec = GameSpec(37, *(load_ydk(p) for p in DECKS))
    result = {"script_errors": ["diagnostic regression"], "responses": [b"\x01"], "action_indices": [3]}
    bad = FinishedGame(Assignment(spec), 0, "win", False, 1, result)
    trainer._log_errors([bad])
    record = json.loads((tmp_path / "errors.jsonl").read_text())
    assert record["reason"] == "win" and not record["truncated"]
    assert record["script_errors"] == result["script_errors"] and record["action_indices"] == [3]
    assert record["spec"]["deck_a"]["main"] == list(spec.deck_a.main)
