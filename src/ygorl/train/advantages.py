"""Advantage and critic-target estimators for two-player zero-sum self-play (T4b.3).

The formulas, the sign convention and the rollout layout are specified in docs/training.md; in short:

- Tensors are time-major ``[T, B]`` (``[T, B, A]`` for per-candidate quantities). Row ``t`` of column
  ``b`` is one agent decision; ``dones[t, b]`` means the episode ended after it and row ``t + 1`` (if any)
  starts a new episode. Optional ``valid`` marks padding, which may only follow a terminal step.
- Every value, Q-value, reward and return is from the perspective of ``players[t]``, the seat that acts
  at row ``t``. Crossing to a row of the other seat multiplies by ``-1`` (zero-sum), so the recursions use
  the continuation ``c_t = γ (1 - done_t) σ_t`` with ``σ_t = +1`` if the next row's seat is the same, else
  ``-1``. The next row of the last row is the bootstrap state (``bootstrap_value`` / ``bootstrap_player``).
- ``gae`` is the control (GAE(λ) on the V head); ``expected_sarsa_returns`` / ``vrpo_advantages`` are
  the design's estimator (design I2): Expected-SARSA(λ) targets for the Q head and the Q-boosted advantage.

All estimators are targets, computed without gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

ESTIMATORS = ("vrpo", "gae")
VRPO_MODES = ("return", "critic")
NORMALIZE_MODES = ("none", "standard", "scale")


def _check_layout(dones: Tensor, valid: Tensor | None, bootstrap_value: Tensor | None,
                  bootstrap_player: Tensor | None) -> None:  # fmt: skip
    if dones.dim() != 2:
        raise ValueError(f"expected [T, B] tensors, got dones of shape {tuple(dones.shape)}")
    if (bootstrap_value is None) != (bootstrap_player is None):
        raise ValueError("bootstrap_value and bootstrap_player must be given together")
    if valid is not None:
        if (valid[1:] & ~valid[:-1]).any():
            raise ValueError("padding (valid = False) must not be followed by real steps in the same column")
        if (valid[:-1] & ~dones[:-1] & ~valid[1:]).any():
            raise ValueError("padding (valid = False) may only follow a terminal step (done = True)")
    open_end = ~dones[-1] if valid is None else valid[-1] & ~dones[-1]
    if bootstrap_value is None and open_end.any():
        raise ValueError("the last row has unfinished episodes: pass bootstrap_value and bootstrap_player")


def perspective_signs(players: Tensor, bootstrap_player: Tensor | None = None) -> Tensor:
    """``σ_t``: +1 where row ``t + 1`` (the bootstrap state for the last row) is the same seat, else -1."""
    last = players[-1] if bootstrap_player is None else bootstrap_player.to(players.device)
    nxt = torch.cat([players[1:], last.unsqueeze(0)])
    return torch.where(nxt == players, 1.0, -1.0)


def _continuation(dones, players, gamma, bootstrap_player, valid, dtype) -> Tensor:
    """``c_t = γ (1 - done_t) σ_t``, zero on padding."""
    cont = gamma * perspective_signs(players, bootstrap_player).to(dtype) * (~dones).to(dtype)
    return cont if valid is None else cont * valid.to(dtype)


def _next(x: Tensor, bootstrap: Tensor | None) -> Tensor:
    """``x`` shifted up one row; the last row gets the bootstrap (zero when absent: that row is terminal)."""
    last = torch.zeros_like(x[-1]) if bootstrap is None else bootstrap.to(x)
    return torch.cat([x[1:], last.unsqueeze(0)])


def _scan(deltas: Tensor, coefs: Tensor) -> Tensor:
    """``out_t = deltas_t + coefs_t * out_{t+1}`` backwards over T, with ``out_T = 0``."""
    out = torch.empty_like(deltas)
    acc = torch.zeros_like(deltas[-1])
    for t in range(deltas.shape[0] - 1, -1, -1):
        acc = deltas[t] + coefs[t] * acc
        out[t] = acc
    return out


def _zero_padding(x: Tensor, valid: Tensor | None) -> Tensor:
    return x if valid is None else torch.where(valid, x, torch.zeros((), dtype=x.dtype, device=x.device))


@torch.no_grad()
def gae(rewards: Tensor, values: Tensor, dones: Tensor, players: Tensor, *, gamma: float = 1.0, lam: float = 0.95,
        bootstrap_value: Tensor | None = None, bootstrap_player: Tensor | None = None,
        valid: Tensor | None = None) -> tuple[Tensor, Tensor]:  # fmt: skip
    """GAE(λ) with perspective flips: ``(advantages, returns)``, both ``[T, B]``.

    ``δ_t = r_t + c_t V(s_{t+1}) - V(s_t)``, ``A_t = δ_t + λ c_t A_{t+1}``, ``returns = A + V`` (the λ-return,
    the V-head target). λ = 1 gives the Monte-Carlo return minus V, λ = 0 the TD error.
    """
    _check_layout(dones, valid, bootstrap_value, bootstrap_player)
    cont = _continuation(dones, players, gamma, bootstrap_player, valid, values.dtype)
    deltas = _zero_padding(rewards.to(values) + cont * _next(values, bootstrap_value) - values, valid)
    advantages = _scan(deltas, lam * cont)
    return advantages, _zero_padding(advantages + values, valid)


@torch.no_grad()
def expected_values(q: Tensor, probs: Tensor, action_mask: Tensor) -> Tensor:
    """``V̄(s) = Σ_a π(a|s) Q(s, a)`` over legal candidates, with π renormalized over them: ``[..., A] -> [...]``.

    Entries outside ``action_mask`` are ignored whatever their value (the Q head leaves them arbitrary).
    """
    zero = torch.zeros((), dtype=q.dtype, device=q.device)
    p = torch.where(action_mask, probs.to(q), zero)
    total = p.sum(-1)
    return torch.where(action_mask, p * q, zero).sum(-1) / torch.where(total > 0, total, torch.ones_like(total))


@torch.no_grad()
def q_advantages(q: Tensor, probs: Tensor, action_mask: Tensor) -> Tensor:
    """Per-candidate Q-boosted advantage ``A(s, a) = Q(s, a) - V̄(s)``, zero on illegal candidates: ``[..., A]``."""
    a = q - expected_values(q, probs, action_mask).unsqueeze(-1)
    return torch.where(action_mask, a, torch.zeros((), dtype=a.dtype, device=a.device))


def _taken(q: Tensor, actions: Tensor) -> Tensor:
    return q.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def expected_sarsa_returns(rewards: Tensor, q: Tensor, probs: Tensor, action_mask: Tensor, actions: Tensor,
                           dones: Tensor, players: Tensor, *, gamma: float = 1.0, lam: float = 0.95,
                           bootstrap_value: Tensor | None = None, bootstrap_player: Tensor | None = None,
                           valid: Tensor | None = None) -> Tensor:  # fmt: skip
    """Expected-SARSA(λ) returns ``G_t`` (the Q-head target for ``Q(s_t, a_t)``), ``[T, B]``.

    ``G_t = r_t + c_t [V̄(s_{t+1}) + λ (G_{t+1} - Q(s_{t+1}, a_{t+1}))]`` (Sutton & Barto 2018 ch. 12:
    the action-value λ-return with control variates, on-policy, with per-seat signs in ``c_t``);
    equivalently ``G_t = Q(s_t, a_t) + Σ_k (λ c)^k δ^Q_{t+k}`` with
    ``δ^Q_t = r_t + c_t V̄(s_{t+1}) - Q(s_t, a_t)``. ``bootstrap_value`` is ``V̄`` of the state after the
    last row (use :func:`expected_values`); the trace is cut there.
    """
    _check_layout(dones, valid, bootstrap_value, bootstrap_player)
    taken = _taken(q, actions)
    vbar = expected_values(q, probs, action_mask)
    cont = _continuation(dones, players, gamma, bootstrap_player, valid, q.dtype)
    deltas = _zero_padding(rewards.to(q) + cont * _next(vbar, bootstrap_value) - taken, valid)
    return _zero_padding(taken + _scan(deltas, lam * cont), valid)


@torch.no_grad()
def vrpo_advantages(rewards: Tensor, q: Tensor, probs: Tensor, action_mask: Tensor, actions: Tensor,
                    dones: Tensor, players: Tensor, *, gamma: float = 1.0, lam: float = 0.95,
                    bootstrap_value: Tensor | None = None, bootstrap_player: Tensor | None = None,
                    valid: Tensor | None = None, mode: str = "return") -> tuple[Tensor, Tensor]:  # fmt: skip
    """Q-boosted advantage of the taken action and the Expected-SARSA(λ) returns: ``(advantages, returns)``.

    ``mode="return"`` (default): ``A_t = G_t - V̄(s_t) = [Q(s_t, a_t) - V̄(s_t)] + Σ_k (λ c)^k δ^Q_{t+k}``.
    ``mode="critic"``: ``A_t = Q(s_t, a_t) - V̄(s_t)`` (pure critic, no sampled correction).
    With an exact Q both have conditional mean ``Q^π(s, a) - V^π(s)``; neither contains the noise of later
    sampled actions, which GAE's δ does (docs/training.md).
    """
    if mode not in VRPO_MODES:
        raise ValueError(f"mode must be one of {VRPO_MODES}, got {mode!r}")
    returns = expected_sarsa_returns(rewards, q, probs, action_mask, actions, dones, players, gamma=gamma, lam=lam,
                                     bootstrap_value=bootstrap_value, bootstrap_player=bootstrap_player,
                                     valid=valid)  # fmt: skip
    vbar = expected_values(q, probs, action_mask)
    base = returns if mode == "return" else _taken(q, actions)
    return _zero_padding(base - vbar, valid), returns


@torch.no_grad()
def normalize_advantages(advantages: Tensor, valid: Tensor | None = None, mode: str = "standard",
                         eps: float = 1e-8) -> Tensor:  # fmt: skip
    """``standard``: zero mean, unit std over valid entries; ``scale``: divide by the std only (keeps signs);
    ``none``: unchanged. Padding is set to 0."""
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"normalize mode must be one of {NORMALIZE_MODES}, got {mode!r}")
    keep = torch.ones_like(advantages, dtype=torch.bool) if valid is None else valid
    x = advantages[keep]
    if mode == "standard" and x.numel():
        advantages = (advantages - x.mean()) / (x.std(correction=0) + eps)
    elif mode == "scale" and x.numel():
        advantages = advantages / (x.std(correction=0) + eps)
    return torch.where(keep, advantages, torch.zeros((), dtype=advantages.dtype, device=advantages.device))


def terminal_rewards(players: Tensor, dones: Tensor, winner: Tensor) -> Tensor:
    """Terminal reward of each row from its acting seat: +1 win, -1 loss, 0 draw, 0 on non-terminal rows.

    ``winner`` holds the winning seat (0 / 1) or -1 for a draw; it is only read where ``dones``.
    """
    won = torch.where(winner == players, 1.0, torch.where(winner == 1 - players, -1.0, 0.0))
    return torch.where(dones, won, torch.zeros_like(won))


@dataclass
class Estimate:
    """Output of :func:`estimate`, all ``[T, B]`` and zero on padding."""

    advantages: Tensor  # policy-gradient weights for the taken actions (normalized per ``normalize``)
    v_targets: Tensor  # V-head regression targets
    q_targets: Tensor | None  # Q-head targets for Q(s_t, a_t); None when no Q head is given (GAE control)


@torch.no_grad()
def estimate(estimator: str, *, rewards: Tensor, dones: Tensor, players: Tensor, actions: Tensor | None = None,
             action_mask: Tensor | None = None, probs: Tensor | None = None, q: Tensor | None = None,
             values: Tensor | None = None, valid: Tensor | None = None, gamma: float = 1.0, lam: float = 0.95,
             bootstrap_player: Tensor | None = None, bootstrap_value: Tensor | None = None,
             bootstrap_q: Tensor | None = None, bootstrap_probs: Tensor | None = None,
             bootstrap_mask: Tensor | None = None, vrpo_mode: str = "return",
             normalize: str = "none") -> Estimate:  # fmt: skip
    """The GAE / VRPO switch (docs/training.md).

    ``"vrpo"``: advantages from :func:`vrpo_advantages`; Q and V targets are the Expected-SARSA(λ) returns.
    ``"gae"``: advantages and V targets from :func:`gae` on ``values``; Q targets are still the
    Expected-SARSA(λ) returns when a Q head is given, so the ablation trains the same critic.
    Bootstrap: ``bootstrap_value`` is the V head at the state after the last row (GAE); ``bootstrap_q`` /
    ``bootstrap_probs`` / ``bootstrap_mask`` (``[B, A]``) give ``V̄`` there for the Expected-SARSA returns.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    common = {"gamma": gamma, "lam": lam, "bootstrap_player": bootstrap_player, "valid": valid}
    has_q = q is not None
    if has_q:
        es_boot = None if bootstrap_q is None else expected_values(bootstrap_q, bootstrap_probs, bootstrap_mask)
        es_args = (rewards, q, probs, action_mask, actions, dones, players)
    if estimator == "vrpo":
        if not has_q:
            raise ValueError("the vrpo estimator needs q, probs, action_mask and actions")
        advantages, returns = vrpo_advantages(*es_args, bootstrap_value=es_boot, mode=vrpo_mode, **common)
        q_targets = v_targets = returns
    else:
        if values is None:
            raise ValueError("the gae estimator needs values (the V head)")
        advantages, v_targets = gae(rewards, values, dones, players, bootstrap_value=bootstrap_value, **common)
        q_targets = expected_sarsa_returns(*es_args, bootstrap_value=es_boot, **common) if has_q else None
    return Estimate(normalize_advantages(advantages, valid, normalize), v_targets, q_targets)
