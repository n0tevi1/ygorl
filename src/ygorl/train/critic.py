"""History-conditioned privileged critic: a Q head over candidate actions and a V head (T4b.3, design I2/I9).

The critic is independent of the network architecture: it takes generic feature tensors, so any history
encoder (the event-stream Transformer or the LSTM baseline) and any encoding of the privileged tensors
(docs/encoding.md 训练态真值) can feed it.

- ``history``: ``[..., H]`` history-conditioned features of the decision point (what the actor also sees);
  design I9 requires the critic to see history, not only the privileged state.
- ``privileged``: ``[..., P]`` features of the opponent ground truth (``EncodedEvent.privileged``). It enters
  only here and in the belief losses, never the actor.
- ``candidates``: ``[..., A, D]`` candidate-action embeddings (the rows the actor's scoring head uses) and
  ``action_mask`` ``[..., A]``.

Q is a dot product between the projected critic context and each projected candidate (the actor's
scoring-head shape) plus a per-candidate bias; values are from the perspective of the seat to act and,
with ``squash``, bounded by tanh to the [-1, 1] range of the terminal reward.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))


class CandidateQHead(nn.Module):
    """``Q(s, a) = <f(ctx), g(cand_a)> / sqrt(hidden) + b(cand_a)`` for every candidate; 0 on illegal ones."""

    def __init__(self, ctx_dim: int, action_dim: int, hidden: int = 256, squash: bool = True) -> None:
        super().__init__()
        self.ctx = _mlp(ctx_dim, hidden, hidden)
        self.cand = _mlp(action_dim, hidden, hidden)
        self.bias = nn.Linear(action_dim, 1)
        self.scale = 1.0 / math.sqrt(hidden)
        self.squash = squash

    def forward(self, ctx: Tensor, candidates: Tensor, action_mask: Tensor) -> Tensor:
        q = (self.cand(candidates) * self.ctx(ctx).unsqueeze(-2)).sum(-1) * self.scale
        q = q + self.bias(candidates).squeeze(-1)
        if self.squash:
            q = torch.tanh(q)
        return q.masked_fill(~action_mask, 0.0)


class ValueHead(nn.Module):
    def __init__(self, ctx_dim: int, hidden: int = 256, squash: bool = True) -> None:
        super().__init__()
        self.net = _mlp(ctx_dim, hidden, 1)
        self.squash = squash

    def forward(self, ctx: Tensor) -> Tensor:
        v = self.net(ctx).squeeze(-1)
        return torch.tanh(v) if self.squash else v


class CriticOutput(NamedTuple):
    q: Tensor  # [..., A], 0 on illegal candidates
    v: Tensor  # [...]


class Critic(nn.Module):
    """Critic context = MLP(history ⊕ privileged); a :class:`CandidateQHead` and a :class:`ValueHead` on top.

    ``privileged_dim = 0`` builds a non-privileged critic (for the ablation in T4c.2).
    """

    def __init__(self, history_dim: int, action_dim: int, privileged_dim: int = 0, hidden: int = 256,
                 squash: bool = True) -> None:  # fmt: skip
        super().__init__()
        self.privileged_dim = privileged_dim
        self.trunk = nn.Sequential(_mlp(history_dim + privileged_dim, hidden, hidden), nn.ReLU())
        self.q_head = CandidateQHead(hidden, action_dim, hidden, squash)
        self.v_head = ValueHead(hidden, hidden, squash)

    def forward(self, history: Tensor, candidates: Tensor, action_mask: Tensor,
                privileged: Tensor | None = None) -> CriticOutput:  # fmt: skip
        if (privileged is None) != (self.privileged_dim == 0):
            raise ValueError(f"this critic was built with privileged_dim={self.privileged_dim}; "
                             f"privileged features {'missing' if privileged is None else 'unexpected'}")  # fmt: skip
        x = history if privileged is None else torch.cat([history, privileged.to(history)], -1)
        ctx = self.trunk(x)
        return CriticOutput(self.q_head(ctx, candidates, action_mask), self.v_head(ctx))


def _masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x.mean()
    w = mask.to(x.dtype)
    return (w * x).sum() / w.sum().clamp_min(torch.finfo(x.dtype).tiny)


def q_loss(q: Tensor, actions: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """``0.5 (Q(s_t, a_t) - G_t)^2`` on the taken candidate, averaged over ``mask`` (bool, or float weights).

    ``targets`` are the Expected-SARSA(λ) returns (constants); other candidates get no gradient.
    """
    taken = q.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    return _masked_mean(0.5 * (taken - targets.detach().to(taken)) ** 2, mask)


def v_loss(v: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """``0.5 (V(s_t) - target_t)^2`` averaged over ``mask`` (bool, or float weights)."""
    return _masked_mean(0.5 * (v - targets.detach().to(v)) ** 2, mask)
