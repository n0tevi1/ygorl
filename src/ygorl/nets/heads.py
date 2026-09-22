"""Action scoring head (T4b.1): dot product between action embeddings and a context query, masked."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ygorl.nets.config import NetConfig

MASKED_LOGIT = -1e9  # finite, so entropy / KL terms stay NaN-free (0 * MASKED_LOGIT == 0 after softmax)


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    return logits.masked_fill(~mask, MASKED_LOGIT)


class ActionHead(nn.Module):
    """``logit_a = <action_a, W q> / sqrt(d)``; ``q`` = trunk context (+ detached belief features)."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.scale = 1 / math.sqrt(d)
        self.belief = nn.Linear(cfg.belief_dim, d) if cfg.belief_dim else None
        self.query = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        with torch.no_grad():  # near-uniform initial policy
            self.query[-1].weight.mul_(0.1)
            self.query[-1].bias.zero_()

    def forward(self, context: Tensor, actions: Tensor, mask: Tensor, belief: Tensor | None = None) -> Tensor:
        if self.belief is not None and belief is not None:
            context = context + self.belief(belief.detach())  # design 04: no gradient into the belief heads
        q = self.query(context)
        return masked_logits(torch.einsum("bad,bd->ba", actions, q) * self.scale, mask)
