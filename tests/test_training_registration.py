"""Actual training publication, immutable history, and matrix recovery without replaying finished cells."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ygorl.agents.registry import AgentSpec  # noqa: E402
from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.engine.duel import DuelConfig  # noqa: E402
from ygorl.eval.agent_matrix import AgentMatrix, build_agent_matrix  # noqa: E402
from ygorl.train.checkpoint import load_checkpoint  # noqa: E402
from ygorl.train.ppo import PPOConfig  # noqa: E402
from ygorl.train.registration import consume_registrations, digest, register_checkpoint  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer  # noqa: E402

DECKS = tuple(str(Path(__file__).parent / "decks" / f"{name}.ydk") for name in ("snake_eye", "kashtira"))


def config(**kw):
    return TrainConfig(
        decks=DECKS,
        num_envs=2,
        env_threads=1,
        steps=4,
        event_length=16,
        net={"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1},
        privileged_dim=8,
        critic_hidden=16,
        max_decisions=30,
        ppo=PPOConfig(epochs=1, minibatch_size=8),
        selfplay_fraction=1,
        snapshot_every=0,
        checkpoint_every=0,
        eval_every=0,
        torch_threads=1,
        collect_threads=1,
        **kw,
    )


def test_off_and_invalid_config(tmp_path):
    trainer = Trainer(config(), tmp_path / "off", log=None)
    trainer.train(max_updates=1)
    assert not (trainer.run_dir / "registrations").exists()
    assert "wall_seconds" not in trainer.counters
    with pytest.raises(ValueError, match="register_every"):
        config(register_every=1)
    with pytest.raises(ValueError, match="register_every"):
        config(register_every=-1)


def test_interval_resume_immutable_and_consumer_recovery(tmp_path, monkeypatch):
    from ygorl.eval import agent_matrix
    from ygorl.train import registration

    target = tmp_path / "matrix.json"
    decks = [load_ydk(p) for p in DECKS]
    baseline = build_agent_matrix(
        {"random": AgentSpec("random"), "greedy": AgentSpec("greedy")},
        decks,
        pairings=1,
        seed=71,
        config=DuelConfig(max_turns=2, max_decisions=40),
    )
    baseline.save(target)
    trainer = Trainer(config(register_every=2, register_matrix=str(target)), tmp_path / "run", log=None)
    trainer.train(max_updates=3)
    root = trainer.run_dir / "registrations"
    files = sorted(root.glob("update_*.json"))
    assert len(files) == 1
    first = json.loads(files[0].read_text())
    assert first["update"] == 2 and first["rows"] == 16 and first["wall_seconds"] >= first["training_seconds"] > 0
    assert digest(Path(first["checkpoint"])) == first["sha256"]
    saved = load_checkpoint(first["checkpoint"])
    assert saved["counters"]["updates"] == saved["learner"]["updates"] == 2
    resumed = Trainer.resume(trainer.run_dir / "checkpoints/latest.pt", log=None)
    resumed.train(max_updates=1)
    records = [json.loads(p.read_text()) for p in sorted(root.glob("update_*.json"))]
    assert [r["update"] for r in records] == [2, 4]
    assert records[1]["run_id"] == first["run_id"] and records[1]["wall_seconds"] > first["wall_seconds"]
    with pytest.raises(ValueError, match="already exists"):
        register_checkpoint(resumed)
    assert digest(Path(first["checkpoint"])) == first["sha256"]

    # Simulate crashing after committing all matrix cells but before publishing the derived curve.
    atomic = registration.atomic_json
    curve_path = target.with_suffix(".strength.json")

    def fail_curve(path, data):
        if path == curve_path:
            raise OSError("interrupted curve")
        atomic(path, data)

    monkeypatch.setattr(registration, "atomic_json", fail_curve)
    with pytest.raises(OSError, match="interrupted curve"):
        consume_registrations([trainer.run_dir], target, decks)
    completed = AgentMatrix.load(target)
    assert len(completed.agents) == 4
    assert completed.games[completed.index("random")][completed.index("greedy")] == 4
    monkeypatch.setattr(registration, "atomic_json", atomic)

    def no_games(*args, **kw):
        pytest.fail("an already committed cell was replayed")

    monkeypatch.setattr(agent_matrix, "_play", no_games)
    curve = consume_registrations([trainer.run_dir], target, decks)
    assert [r["update"] for r in curve["records"]] == [2, 4]
    assert all(set(r["baselines"]) == {"random", "greedy"} for r in curve["records"])
    assert curve == consume_registrations([trainer.run_dir], target, decks)
    assert json.loads(curve_path.read_text()) == curve
    assert AgentMatrix.load(target) == completed

    # Existing matrix protocol, deck content and immutable weights must still be checked on no-op calls.
    with pytest.raises(ValueError, match="deck pool differs"):
        consume_registrations([trainer.run_dir], target, list(reversed(decks)))
    replace(completed, seed=99).save(target)
    with pytest.raises(ValueError, match="protocol changed"):
        consume_registrations([trainer.run_dir], target, decks)
    completed.save(target)
    Path(first["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        consume_registrations([trainer.run_dir], target, decks)


def test_error_matrix_is_not_published(tmp_path, monkeypatch):
    from ygorl.eval import agent_matrix

    target = tmp_path / "matrix.json"
    decks = [load_ydk(p) for p in DECKS]
    matrix = build_agent_matrix(
        {"r": AgentSpec("random"), "g": AgentSpec("greedy")}, decks, pairings=1, config=DuelConfig(max_turns=1)
    )
    matrix.save(target)
    before = target.read_bytes()
    trainer = Trainer(config(register_every=1, register_matrix=str(target)), tmp_path / "run", log=None)
    trainer.train(max_updates=1)
    monkeypatch.setattr(agent_matrix, "extend_agent_matrix", lambda *a, **kw: replace(matrix, errors=((0, 1), (1, 0))))
    with pytest.raises(ValueError, match="engine errors"):
        consume_registrations([trainer.run_dir], target, decks)
    assert target.read_bytes() == before
    assert target.with_suffix(".json.rejected.json").exists()
