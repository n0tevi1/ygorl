"""Experimental local-risk scoring and bounded soft filtering (no PPO integration)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ygorl.nets.heads import MASKED_LOGIT


class ActionRiskHead(nn.Module):
    """Action-conditional local preference classifier over frozen actor features.

    Fit normalization only on training candidates. Sigmoid outputs are scores, not
    calibrated probabilities of a globally bad move. No actor is owned or updated.
    """

    def __init__(self, d_model: int, hidden: int = 32) -> None:
        super().__init__()
        if d_model < 1 or hidden < 1:
            raise ValueError("feature and hidden dimensions must be positive")
        self.register_buffer("mean", torch.zeros(3 * d_model))
        self.register_buffer("scale", torch.ones(3 * d_model))
        self.net = nn.Sequential(nn.Linear(3 * d_model, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @staticmethod
    def inputs(context: Tensor, actions: Tensor) -> Tensor:
        if context.ndim != 2 or actions.ndim != 3 or context.shape != (actions.shape[0], actions.shape[2]):
            raise ValueError("expected context [B,D] and actions [B,A,D]")
        c = context[:, None, :].expand_as(actions)
        return torch.cat((c, actions, c * actions), -1)

    @torch.no_grad()
    def fit_normalization(self, training_candidates: Tensor) -> None:
        if training_candidates.ndim != 2 or training_candidates.shape[1] != self.mean.numel():
            raise ValueError("expected training candidates [N,3D]")
        if len(training_candidates) < 2 or not torch.isfinite(training_candidates).all():
            raise ValueError("normalization needs at least two finite training candidates")
        self.mean.copy_(training_candidates.mean(0))
        self.scale.copy_(training_candidates.std(0, correction=0).clamp_min(1e-3))

    def forward(self, context: Tensor, actions: Tensor) -> Tensor:
        x = self.inputs(context, actions)
        return self.net((x - self.mean) / self.scale).squeeze(-1)


@dataclass
class FilterOutput:
    logits: Tensor
    penalty: Tensor
    failed_rows: Tensor


def soft_filter(
    logits: Tensor,
    risk_scores: Tensor,
    legal: Tensor,
    supported: Tensor,
    *,
    threshold: float = 0.9,
    max_penalty: float = 4.0,
) -> FilterOutput:
    """Penalize supported scores above threshold without removing legal actions.

    Scores must lie in [0,1]. Any invalid supported score disables filtering for
    that entire row. Bad scores outside support are irrelevant. Legal logits are
    centered for numerical stability; fallback preserves their distribution. The caller must
    use returned logits for every probability calculation of a composite policy.
    This utility alone does not make an existing PPO implementation compatible.
    """
    if logits.ndim != 2 or any(x.shape != logits.shape for x in (risk_scores, legal, supported)):
        raise ValueError("all inputs must have shape [B,A]")
    if legal.dtype != torch.bool or supported.dtype != torch.bool:
        raise ValueError("legal and supported must be boolean")
    if not logits.is_floating_point() or not risk_scores.is_floating_point():
        raise ValueError("logits and risk scores must be floating point")
    if not 0 <= threshold < 1 or not math.isfinite(max_penalty) or not 0 <= max_penalty <= 20:
        raise ValueError("threshold must be in [0,1), max_penalty in [0,20]")
    if not legal.any(-1).all() or not torch.isfinite(logits.float()[legal]).all():
        raise ValueError("each row needs a legal action and finite legal actor logits")
    scope = legal & supported
    valid = torch.isfinite(risk_scores) & (risk_scores >= 0) & (risk_scores <= 1)
    failed = (scope & ~valid).any(-1)
    active = scope & ~failed[:, None]
    safe_scores = torch.where(active, risk_scores, torch.zeros_like(risk_scores)).float()
    penalty = max_penalty * ((safe_scores - threshold) / (1 - threshold)).clamp(0, 1)
    penalty = torch.where(active, penalty, torch.zeros_like(penalty))
    result = logits.float() - penalty
    # Center legal logits so the finite illegal sentinel cannot outrank very negative actors.
    result = result - result.masked_fill(~legal, -torch.inf).amax(-1, keepdim=True)
    result = result.clamp_min(torch.finfo(result.dtype).min).masked_fill(~legal, MASKED_LOGIT)
    return FilterOutput(result, penalty, failed)
