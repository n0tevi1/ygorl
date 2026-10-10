"""Experimental private-critic resource path; no actor or production checkpoint changes."""

from __future__ import annotations

import torch
from torch import nn


def resource_features(globals_: torch.Tensor) -> torch.Tensor:
    """Public turn, both LP totals, and both players' five zone counts."""
    g = globals_.float()
    return torch.cat((g[..., 3:4] / 20, g[..., 5:7] / 8000, g[..., 7:17] / 40), -1)


class ResourceCritic(nn.Module):
    """Identical auxiliary capacity in each arm; the control receives zero resources.

    The zero-initialized final projection preserves the original critic exactly.
    The actor/shared public context remains frozen. Outputs are chosen-action Q
    and state V; these research weights are not a resumable PPO checkpoint.
    """

    def __init__(self, model: nn.Module, *, resources: bool, seed: int) -> None:
        super().__init__()
        assert model.critic_trunk is None
        self.model = model
        self.resources = resources
        model.requires_grad_(False)
        model.privileged.requires_grad_(True)
        model.critic.requires_grad_(True)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            width = model.critic.trunk[0][0].out_features
            self.public = nn.Sequential(nn.Linear(13, 32), nn.ReLU(), nn.Linear(32, width))
            nn.init.zeros_(self.public[-1].weight)
            nn.init.zeros_(self.public[-1].bias)

    def forward(self, b: dict) -> torch.Tensor:
        c = self.model.critic
        public = b["resources"] if self.resources else torch.zeros_like(b["resources"])
        x = torch.cat((b["context"], self.model.privileged(b["privileged"])), -1)
        hidden = c.trunk[0][0](x) + self.public(public)
        hidden = c.trunk[1](c.trunk[0][2](c.trunk[0][1](hidden)))
        candidates = b["candidate"][:, None, :]
        q = c.q_head(hidden, candidates, torch.ones(candidates.shape[:2], dtype=torch.bool, device=x.device))
        return torch.stack((q[:, 0], c.v_head(hidden)), -1)


def losses(pred, target, weight, groups):
    error = pred - target[:, None]
    mse = (weight * error.square().mean(-1)).sum() / weight.sum()
    totals = torch.zeros(int(groups.max()) + 1, device=pred.device, dtype=pred.dtype).index_add_(0, groups, weight)
    sums = torch.zeros((len(totals), pred.shape[-1]), device=pred.device, dtype=pred.dtype).index_add_(
        0, groups, error * weight[:, None]
    )
    assert bool((totals > 0).all())
    means = sums / totals[:, None]
    calibration = means.square().mean()
    return mse, calibration, means, totals


def calibration_coefficients(pred, target, weight, groups, coefficient):
    """Exact derivative of group-mean squared error for memory-bounded two-pass training.

    Evaluate at the current weights before any optimizer step, then add the
    coefficient times the differentiable minibatch predictions to its MSE loss.
    No parameters may change between the two passes; dropout must be disabled.
    """
    with torch.no_grad():
        _, _, means, totals = losses(pred, target, weight, groups)
        return (2 * coefficient / means.numel()) * means[groups] * (weight / totals[groups])[:, None]
