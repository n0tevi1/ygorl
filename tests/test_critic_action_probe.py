# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from critic_action_probe import all_action_values
from critic_opponent_probe import opponent_features
from critic_residual_probe import predict_residual
from critic_resource_probe import ResourceCritic
from ygorl.train.critic import Critic


def example():
    torch.manual_seed(4)
    model = nn.Module()
    model.actor = nn.Linear(3, 3)
    model.critic_trunk = None
    model.privileged = nn.Linear(2, 2)
    model.critic = Critic(3, 3, 2, hidden=8)
    net = ResourceCritic(model, resources=True, seed=5).eval()
    batch = dict(
        context=torch.randn(3, 3, requires_grad=True),
        candidates=torch.randn(3, 4, 3, requires_grad=True),
        action_mask=torch.tensor([[True, False, True, False], [False, True, True, True], [True, False, False, False]]),
        privileged=torch.randn(3, 2),
        resources=torch.randn(3, 13),
    )
    return net, batch


@pytest.mark.parametrize("offset", [None, 0.0, 2.0])
def test_each_legal_q_matches_chosen_action_path_and_padding_stays_zero(offset):
    net, b = example()
    opponents = ["old-256x2", "mirror-u528", "old-256x2"]
    ridge = None
    if offset is not None:
        ridge = dict(mean=torch.zeros(14).double(), scale=torch.ones(14).double(), beta=torch.zeros(15, 2).double())
        ridge["beta"][-1] = torch.tensor([offset, -offset])
        ridge["beta"][0] = torch.tensor([0.2, -0.1])
    q, v = all_action_values(net, b, ridge=ridge, opponents=opponents)
    x = opponent_features(b["resources"], torch.arange(3), opponents, conditioned=True)
    for action in range(4):
        expected = net({**b, "candidate": b["candidates"][:, action]})
        if ridge is not None:
            expected = predict_residual(ridge, x, expected)
        legal = b["action_mask"][:, action]
        torch.testing.assert_close(q[legal, action], expected[legal, 0])
        torch.testing.assert_close(v, expected[:, 1])
    assert bool((q[~b["action_mask"]] == 0).all())


def test_critic_gradient_does_not_enter_reference_features():
    net, b = example()
    q, v = all_action_values(net, b)
    (q.square().sum() + v.square().sum()).backward()
    assert b["context"].grad is None and b["candidates"].grad is None
    assert all(p.grad is None for p in net.model.actor.parameters())
    assert net.model.privileged.weight.grad.abs().sum() > 0
    assert net.public[-1].weight.grad.abs().sum() > 0


def test_missing_or_unknown_opponent_and_non_boolean_mask_rejected():
    net, b = example()
    with pytest.raises(ValueError, match="registered opponent"):
        all_action_values(net, b, ridge={})
    with pytest.raises(ValueError, match="unknown diagnostic opponent"):
        all_action_values(net, b, ridge={}, opponents=["unseen"] * 3)
    with pytest.raises(ValueError, match="boolean"):
        all_action_values(net, {**b, "action_mask": b["action_mask"].float()})
