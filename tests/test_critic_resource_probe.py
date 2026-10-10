# ruff: noqa: E402 -- skip the module before torch-dependent imports in core CI
import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from critic_resource_probe import ResourceCritic, calibration_coefficients, losses, resource_features
from ygorl.train.critic import Critic


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(3, 3)
        self.critic_trunk = None
        self.privileged = nn.Linear(2, 2)
        self.critic = Critic(3, 3, 2, hidden=8)


def test_zero_init_and_private_update():
    torch.manual_seed(1)
    base = Model()
    b = dict(
        context=torch.randn(7, 3),
        candidate=torch.randn(7, 3),
        privileged=torch.randn(7, 2),
        resources=torch.randn(7, 13),
    )
    expected = base.critic(
        b["context"], b["candidate"][:, None], torch.ones(7, 1, dtype=torch.bool), base.privileged(b["privileged"])
    )
    probe = ResourceCritic(base, resources=True, seed=2)
    torch.testing.assert_close(probe(b), torch.stack((expected.q[:, 0], expected.v), -1), rtol=0, atol=0)
    actor = copy.deepcopy(base.actor.state_dict())
    opt = torch.optim.Adam([p for p in probe.parameters() if p.requires_grad], lr=0.01)
    probe(b).square().mean().backward()
    assert all(p.grad is None for p in base.actor.parameters())
    assert probe.public[-1].weight.grad.abs().sum() > 0
    opt.step()
    assert all(torch.equal(actor[k], v) for k, v in base.actor.state_dict().items())


def test_control_ignores_resource_channel_and_feature_orientation():
    base = Model()
    probe = ResourceCritic(base, resources=False, seed=4)
    with torch.no_grad():
        probe.public[-1].weight.fill_(0.1)
    b = dict(
        context=torch.randn(5, 3),
        candidate=torch.randn(5, 3),
        privileged=torch.randn(5, 2),
        resources=torch.randn(5, 13),
    )
    torch.testing.assert_close(probe(b), probe({**b, "resources": b["resources"] + 100}), rtol=0, atol=0)
    g = torch.arange(22).reshape(1, 22)
    expected = torch.tensor([[3 / 20, 5 / 8000, 6 / 8000, *[n / 40 for n in range(7, 17)]]])
    torch.testing.assert_close(resource_features(g), expected)


def test_two_pass_calibration_matches_full_autograd_with_unequal_game_lengths():
    torch.manual_seed(3)
    p = torch.randn(9, 2, requires_grad=True)
    z = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    # Unequal game lengths; the two groups also have different total weight.
    w = torch.tensor([1.0, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1])
    mse, cal, _, _ = losses(p, z, w, groups)
    exact = torch.autograd.grad(mse + 0.5 * cal, p)[0]
    coeff = calibration_coefficients(p.detach(), z, w, groups, 0.5)
    q = p.detach().clone().requires_grad_()
    for start in range(0, 9, 4):
        sl = slice(start, start + 4)
        loss = ((q[sl] - z[sl, None]).square().mean(-1) * w[sl]).sum() / w.sum() + (coeff[sl] * q[sl]).sum()
        loss.backward()
    torch.testing.assert_close(q.grad, exact)
