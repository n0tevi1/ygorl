"""Actor + privileged critic for PPO self-play (T4b.4): :class:`PolicyNet` with the Q / V heads of T4b.3.

- Actor: :class:`ygorl.nets.PolicyNet` on the observation only (``EncodedEvent.obs``).
- Critic (design I2 / I9): :class:`ygorl.train.critic.Critic` on the trunk context (board + event history)
  ⊕ :class:`PrivilegedEncoder` features of the opponent ground truth (``EncodedEvent.privileged``), scoring the
  same candidate-action embeddings the actor scores. The privileged tensors never reach the actor.
- ``shared_backbone=True`` (default): the critic reads the actor's trunk (one forward pass; the critic loss
  also trains the trunk). ``False``: the critic has its own :class:`PolicyNet` trunk (twice the compute).

The model interface used by :mod:`ygorl.train.rollout` and :mod:`ygorl.train.ppo`:
``collate(obs_list)``, ``collate_privileged(priv_list)``, ``model(batch, privileged) -> ActorCriticOutput``
and ``policy_logits(batch)`` (actor only, for snapshot opponents and KL references).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from ygorl.nets.batch import Batch, collate
from ygorl.nets.config import NetConfig
from ygorl.nets.heads import MASKED_LOGIT
from ygorl.nets.policy import PolicyNet, _pad_to, trim_padding
from ygorl.nets.text import TextFeatures
from ygorl.train.critic import Critic

PRIVILEGED_LISTS = ("op_hand", "op_deck", "op_extra", "op_set", "op_removed")  # docs/encoding.md 训练态真值
PRIVILEGED_COUNTS = "counts"


class ActorCriticOutput(NamedTuple):
    logits: Tensor  # [B, A] actor logits, illegal rows = MASKED_LOGIT
    q: Tensor  # [B, A] Q head, 0 on illegal rows
    v: Tensor  # [B] V head


class PrivilegedEncoder(nn.Module):
    """Opponent ground truth -> ``[B, dim]``: per list, masked mean of card-identity + public embeddings;
    ``log1p`` of the true counts; one linear layer. Its own embedding table (nothing shared with the actor)."""

    def __init__(self, vocab_size: int, dim: int = 64) -> None:
        super().__init__()
        self.card = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.public = nn.Embedding(2, dim)
        self.out = nn.Sequential(nn.Linear(len(PRIVILEGED_LISTS) * dim + 5, dim), nn.ReLU())
        self.dim = dim

    def forward(self, priv: Mapping[str, Tensor]) -> Tensor:
        parts = []
        for key in PRIVILEGED_LISTS:
            rows = priv[key]  # [B, N, 3] = (card_index, public, sequence)
            idx = rows[..., 0].clamp(0, self.card.num_embeddings - 1)
            present = (idx > 0).unsqueeze(-1).float()
            emb = (self.card(idx) + self.public(rows[..., 1].clamp(0, 1))) * present
            parts.append(emb.sum(-2) / present.sum(-2).clamp(min=1))
        parts.append(torch.log1p(priv[PRIVILEGED_COUNTS].float()))
        return self.out(torch.cat(parts, -1))


def collate_privileged(privileged: Sequence[Mapping[str, np.ndarray] | None],
                       device: torch.device | str | None = None) -> dict[str, Tensor] | None:  # fmt: skip
    """Stack ``EncodedEvent.privileged`` dicts; None when any is missing (inference-mode environment)."""
    if not privileged or any(p is None for p in privileged):
        return None
    keys = (*PRIVILEGED_LISTS, PRIVILEGED_COUNTS)
    return {k: torch.as_tensor(np.stack([p[k] for p in privileged])).long().to(device) for k in keys}


class ActorCritic(nn.Module):
    def __init__(self, cfg: NetConfig, text: TextFeatures | None = None, *, privileged: bool = True,
                 privileged_dim: int = 64, critic_hidden: int = 128, shared_backbone: bool = True) -> None:  # fmt: skip
        super().__init__()
        self.cfg = cfg
        self.actor = PolicyNet(cfg, text)
        self.critic_trunk = None if shared_backbone else PolicyNet(cfg, text)
        self.privileged = PrivilegedEncoder(cfg.vocab_size, privileged_dim) if privileged else None
        d = cfg.d_model
        self.critic = Critic(history_dim=d, action_dim=d, privileged_dim=privileged_dim if privileged else 0,
                             hidden=critic_hidden)  # fmt: skip

    collate = staticmethod(collate)
    collate_privileged = staticmethod(collate_privileged)

    def policy_logits(self, obs: Batch) -> Tensor:
        return self.actor(obs).logits

    def forward(self, obs: Batch, privileged: Mapping[str, Tensor] | None = None) -> ActorCriticOutput:
        # Heads on the trimmed batch (the critic's Q head then scores only the used action rows, not all 128);
        # logits / Q are padded back to the batch's action width.
        width = obs["action_mask"].shape[1]
        obs = trim_padding(obs)
        out = self._forward(obs, privileged)
        return ActorCriticOutput(_pad_to(out.logits, width, value=MASKED_LOGIT), _pad_to(out.q, width), out.v)

    def _forward(self, obs: Batch, privileged: Mapping[str, Tensor] | None) -> ActorCriticOutput:
        f = self.actor.features(obs)
        logits = self.actor.logits(f)
        cf = f if self.critic_trunk is None else self.critic_trunk.features(obs)
        priv = None
        if self.privileged is not None:
            if privileged is None:
                raise ValueError("this critic is privileged: pass the opponent ground truth "
                                 "(EncodedVecEnv(privileged=True))")  # fmt: skip
            priv = self.privileged(privileged)
        crit = self.critic(cf.context, cf.actions, cf.action_mask, priv)
        return ActorCriticOutput(logits, crit.q, crit.v)


__all__ = ["ActorCritic", "ActorCriticOutput", "PrivilegedEncoder", "collate_privileged"]
