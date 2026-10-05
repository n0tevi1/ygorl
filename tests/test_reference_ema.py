"""The slow reference must update shared model state once, regardless of its module aliases."""

import pytest

torch = pytest.importorskip("torch")

from ygorl.nets.actor_critic import ActorCritic  # noqa: E402
from ygorl.nets.config import NetConfig  # noqa: E402
from ygorl.train.ppo import PPOConfig, PPOLearner  # noqa: E402


@pytest.mark.parametrize("history", ["none", "transformer"])
@pytest.mark.parametrize("shared_backbone", [True, False])
def test_actual_actor_critic_reference_updates_each_shared_parameter_once(history, shared_backbone):
    model = ActorCritic(
        NetConfig(vocab_size=20, d_model=16, board_layers=1, history=history, history_layers=1),
        shared_backbone=shared_backbone,
    )
    learner = PPOLearner(model, PPOConfig(reference_ema=0.02))
    keys = list(model.state_dict())
    assert len(list(model.named_parameters(remove_duplicate=False))) > len(list(model.parameters()))
    for shift in (1.0, -0.5):
        before = {name: value.clone() for name, value in learner.reference.state_dict().items()}
        with torch.no_grad():
            for value in model.parameters():
                value.add_(shift)
        learner.update_reference()
        for name, value in learner.reference.state_dict().items():
            torch.testing.assert_close(value, before[name] * 0.98 + model.state_dict()[name] * 0.02, msg=name)
    reference = learner.reference
    assert reference.actor.identity is reference.actor.board.cards.identity
    assert reference.actor.identity is reference.actor.action_encoder.identity
    assert not reference.training and not any(p.requires_grad for p in reference.parameters())
    assert list(reference.state_dict()) == keys
    restored = PPOLearner(ActorCritic(model.cfg, shared_backbone=shared_backbone), learner.cfg)
    restored.load_state_dict(learner.state_dict())
    for name, value in restored.reference.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    assert restored.reference.actor.identity is restored.reference.actor.action_encoder.identity


def test_reference_ema_preserves_persistent_and_transient_buffer_contracts():
    model = torch.nn.Module()
    model.shared = torch.nn.Linear(2, 2)
    model.alias = model.shared
    model.shared.register_buffer("average", torch.tensor([1.0, 3.0]))
    model.shared.register_buffer("count", torch.tensor(2))
    model.shared.register_buffer("transient", torch.tensor(7.0), persistent=False)
    learner = PPOLearner(model, PPOConfig(reference_ema=0.1))
    before = {name: value.clone() for name, value in learner.reference.state_dict().items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(1)
        model.shared.average.add_(2)
        model.shared.count.add_(4)
        model.shared.transient.add_(5)
    learner.update_reference()
    for name, value in learner.reference.state_dict().items():
        current = model.state_dict()[name]
        expected = before[name] * 0.9 + current * 0.1 if value.is_floating_point() else current
        torch.testing.assert_close(value, expected, msg=name)
    assert learner.reference.shared is learner.reference.alias
    assert learner.reference.shared.transient.item() == 7
