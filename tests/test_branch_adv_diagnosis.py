"""Smoke test of tools/branch_adv_diagnosis.py (first-seat plan §11): a tiny trainer's checkpoint with scripted lines,
one short rollout with every window branched M = 2 times, the update variants and a checkpoint comparison."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import branch_adv_diagnosis  # noqa: E402
from ygorl.train.ppo import PPOConfig  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer  # noqa: E402

HERE = Path(__file__).parent


def test_branch_adv_diagnosis_runs_end_to_end(tmp_path):
    data = str(HERE / "data" / "line_recovery.jsonl")
    pair = (str(HERE / "decks" / "snake_eye.ydk"), str(HERE / "decks" / "kashtira.ydk"))
    cfg = TrainConfig(decks=pair, num_envs=2, env_threads=1, steps=24, event_length=16,
                      net={"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}, privileged_dim=8,
                      critic_hidden=16, max_decisions=60, eval_every=0, checkpoint_every=0, snapshot_every=0,
                      torch_threads=1, collect_threads=1, seed=4, selfplay_fraction=0.0, line_demos=(data,),
                      line_share=1.0, line_recovery=(data,), ppo=PPOConfig(minibatch_size=16))  # fmt: skip
    trainer = Trainer(cfg, tmp_path / "run", log=lambda *_: None)
    trainer.pool.add(trainer.model, update=0)
    ckpt = str(trainer.save("tiny.pt"))
    out = tmp_path / "r.json"
    report = branch_adv_diagnosis.main([ckpt, "--envs", "2", "--steps", "24", "--rounds", "1", "--rollouts", "2",
                                        "--env-threads", "1", "--torch-threads", "1", "--updates", "--records",
                                        "--compare", ckpt, "--out", str(out)])  # fmt: skip
    assert json.loads(out.read_text())["rounds"] == report["rounds"]
    rnd = report["rounds"][0]
    assert rnd["windows"] >= rnd["valid_windows"] > 0 and rnd["magnitude"]["rows"] == 48
    assert report["noise"]["M"] == 2 and report["noise"]["windows"] == len(report["records"])
    assert all(len(r["v_fire"]) == len(r["v_pass"]) == 2 for r in report["records"])
    assert {"ctrl", "a1n_replace_m1_half_scale0.1"} <= set(report["updates"])
    assert report["updates"]["ctrl"]["all"]["n"] == rnd["valid_windows"]
    same = report["compare"][ckpt]["all"]  # the checkpoint itself: the probabilities the rollout recorded
    fire = [r["pi_fire"] / (r["pi_fire"] + r["pi_pass"]) for r in report["records"]]
    assert same["fire_prob"] == pytest.approx(sum(fire) / len(fire), abs=1e-4)
