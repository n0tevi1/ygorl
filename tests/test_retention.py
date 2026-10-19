"""Retention controls must preserve the PPO off path and legal action support."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.train.ppo import PPOConfig, PPOLearner
from ygorl.train.retention import FrozenReplay, MechanismDeckPool, prompt_groups
from ygorl.train.rollout import Assignment, RolloutCollector
from ygorl.train.selfplay import DeckPool
from ygorl.train.toy import NimEnv, NimModel


def make_rollout():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = NimModel()
    ro = RolloutCollector(NimEnv(2), model, lambda: Assignment(10), num_steps=8, seed=7).collect()
    replay = FrozenReplay(ro.obs, ro.probs.reshape(-1, 3), ro.obs["pile"] % 3, seed=91)
    return model, ro, replay


def test_disabled_is_exact_and_does_not_draw_replay():
    model, ro, replay = make_rollout()
    a, b = [PPOLearner(copy.deepcopy(model), PPOConfig()) for _ in range(2)]
    torch.manual_seed(9)
    sa = a.update(ro)
    torch.manual_seed(9)
    sb = b.update(ro, replay=replay)
    assert sa == sb
    assert replay.sampled == 0
    assert all(torch.equal(x, y) for x, y in zip(a.model.parameters(), b.model.parameters(), strict=True))


def test_retention_requires_bank_and_counts_only_accepted_steps():
    model, ro, replay = make_rollout()
    learner = PPOLearner(model, PPOConfig(retention_coef=0.5, retention_batch=7))
    with pytest.raises(ValueError, match="bank"):
        learner.update(ro)
    teacher = replay.probabilities.clone()
    rng = torch.get_rng_state().clone()
    replay.sample(2)
    assert torch.equal(rng, torch.get_rng_state())
    stats = learner.update(ro, replay=replay)
    assert stats["retention_rows"] == stats["minibatches"] * 7
    assert replay.sampled == stats["retention_rows"] + 2
    assert torch.equal(replay.probabilities, teacher)
    assert not replay.probabilities.requires_grad
    # Reject every minibatch by forcing the supplied behavior log probabilities away.
    ro.log_probs -= 10
    sampled = replay.sampled
    stats = learner.update(ro, replay=replay)
    assert stats["minibatches"] == stats["retention_rows"] == 0
    assert replay.sampled == sampled


def test_distillation_gradient_moves_toward_teacher_and_ignores_illegal_logits():
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([0.0, 0.0, 900.0]))

        def policy_logits(self, obs):
            return self.logits.expand(len(obs["action_mask"]), -1)

    model = Actor()
    p = torch.tensor([[0.9, 0.1, 0.0]], requires_grad=True)
    obs = {"action_mask": torch.tensor([[True, True, False]])}
    replay = FrozenReplay(obs, p, torch.tensor([0]))
    loss = replay.loss(model, 1)
    loss.backward()
    assert model.logits.grad[0] < 0 < model.logits.grad[1]
    assert model.logits.grad[2] == 0 and p.grad is None
    assert torch.isfinite(loss)
    p.data[0, 0] = 0
    assert replay.probabilities[0, 0] == pytest.approx(0.9)


def test_bank_rejects_invalid_teacher_and_restores_sampler():
    _, ro, bank = make_rollout()
    state = bank.state_dict()
    first = bank.sample(100)
    bank.load_state_dict(state)
    assert torch.equal(first, bank.sample(100))
    p = ro.probs.reshape(-1, 3).clone()
    p[0, 0] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        FrozenReplay(ro.obs, p, ro.obs["pile"])


def test_rare_groups_have_equal_sampling_mass():
    mask = torch.ones(101, 2, dtype=torch.bool)
    bank = FrozenReplay({"action_mask": mask}, mask.float() / 2, torch.tensor([0] * 100 + [1]), seed=17)
    ix = bank.sample(10000)
    assert 0.47 < float((ix == 100).float().mean()) < 0.53


def test_mechanism_sampler_preserves_support_and_zero_share_rng():
    decks = [Deck(name=str(i), main=(i,), extra=(), side=()) for i in range(3)]
    cards = {
        0: SimpleNamespace(type=C.TYPE_RITUAL),
        1: SimpleNamespace(type=C.TYPE_MONSTER),
        2: SimpleNamespace(type=C.TYPE_MONSTER),
    }
    base = DeckPool(decks)
    zero = MechanismDeckPool(decks, cards, share=0)
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    assert [base.sample(a) for _ in range(50)] == [zero.sample(b) for _ in range(50)]
    active = MechanismDeckPool(decks, cards)
    assert all(active.probabilities > 0)
    assert active.probabilities.sum() == pytest.approx(1)
    observed = sum(any(d.name == "0" for d in active.sample(a)) for _ in range(10000)) / 10000
    assert observed == pytest.approx(0.5 + 0.5 * 4 / 6, abs=0.02)


def test_groups_use_pre_action_prompt_and_phase():
    g = torch.zeros(3, 22, dtype=torch.long)
    g[:, 18] = torch.tensor([10, 10, 11])
    g[:, 4] = torch.tensor([1, 2, 1])
    assert len(prompt_groups({"globals": g}).unique()) == 3


def test_censoring_never_uses_lp_winner_and_factorial_interaction():
    from retention_factorial import factorial_contrasts, outcome_bounds

    assert outcome_bounds({"reason": "turn_limit", "winner": 0}) == (0.0, 1.0)
    assert outcome_bounds({"reason": "decision_limit", "winner": None}) == (0.0, 1.0)
    assert outcome_bounds({"reason": "win", "winner": 0}) == (1.0, 1.0)
    with pytest.raises(RuntimeError, match="unhealthy"):
        outcome_bounds({"reason": "win", "winner": 0, "retries": 1})
    # baseline=.2, coverage alone +.1, replay alone +.2; combined includes +.1 interaction
    v = np.array([[[0.2], [0.4], [0.3], [0.6]]])
    c = factorial_contrasts(v)
    np.testing.assert_allclose(c["interaction"], 0.1)
    np.testing.assert_allclose(c["coverage"], 0.15)
    np.testing.assert_allclose(c["replay"], 0.25)


def test_runner_replays_finished_native_training_traces(tmp_path):
    from pathlib import Path
    from retention_factorial import replay_training
    from ygorl.train.trainer import TrainConfig, Trainer

    decks = tuple(str(Path(__file__).parent / "decks" / n) for n in ("snake_eye.ydk", "kashtira.ydk"))
    cfg = TrainConfig(
        decks=decks,
        net={"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1},
        num_envs=1,
        steps=24,
        event_length=8,
        max_decisions=12,
        trace_games=True,
        eval_every=0,
        checkpoint_every=0,
        torch_threads=1,
        collect_threads=1,
        ppo=PPOConfig(epochs=1, minibatch_size=24),
    )
    trainer = Trainer(cfg, tmp_path)
    trainer.step()
    assert trainer.counters["games"] > 0
    assert replay_training(trainer) == trainer.counters["games"]
