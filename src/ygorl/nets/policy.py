"""The policy network (T4b.1 + T4b.2): board encoder ⊕ event history -> trunk -> action logits.

Usage::

    net = PolicyNet(NetConfig(vocab_size=len(vocab)).with_text(text), text)
    out = net(collate([event.obs for event in events]))   # window mode (state=None)
    action = out.sample()                                  # indices into the 128 action rows

``net.features(obs)`` exposes the trunk for the critic / auxiliary heads (T4b.3, T4c.1); ``net.logits(f,
belief)`` is the actor head on top of it (belief features enter detached).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ygorl.env.events import E_EVENT
from ygorl.nets.batch import Batch
from ygorl.nets.config import NetConfig
from ygorl.nets.encoders import ActionEncoder, BoardEncoder, CardIdentity, EffectText, EventEmbedding
from ygorl.nets.heads import ActionHead
from ygorl.nets.history import HistoryEncoder, HistoryState, build_history
from ygorl.nets.text import TextFeatures


@dataclass
class Features:
    """Trunk outputs shared by the actor, the critic (T4b.3) and the belief / auxiliary heads (T4c.1)."""

    context: Tensor  # [B, d]  decision context: board + history (the actor's query input)
    globals: Tensor  # [B, d]  global token after the board Transformer
    cards: Tensor  # [B, N, d]  contextual card tokens
    card_mask: Tensor  # [B, N]
    actions: Tensor  # [B, A, d]  candidate-action embeddings (also the input of a per-action Q head)
    action_mask: Tensor  # [B, A]
    history: Tensor  # [B, d]  history summary (zeros when the history module is off)
    history_tokens: Tensor | None  # [B, T, d]  causal per-event outputs (next-token / auxiliary heads)
    history_state: HistoryState | None


@dataclass
class PolicyOutput:
    logits: Tensor  # [B, A], illegal rows = MASKED_LOGIT
    features: Features
    state: HistoryState | None  # streaming history state to pass to the next call

    @property
    def action_mask(self) -> Tensor:
        return self.features.action_mask

    def log_probs(self) -> Tensor:
        return torch.log_softmax(self.logits, -1)

    def probs(self) -> Tensor:
        return torch.softmax(self.logits, -1)

    def entropy(self) -> Tensor:
        return -(self.probs() * self.log_probs()).sum(-1)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        return torch.multinomial(self.probs(), 1, generator=generator).squeeze(-1)

    def greedy(self) -> Tensor:
        return self.logits.argmax(-1)


class PolicyNet(nn.Module):
    def __init__(self, cfg: NetConfig, text: TextFeatures | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.identity = CardIdentity(cfg, text)  # shared by cards, actions and events
        self.effect = EffectText(cfg, text) if cfg.effect_text_dim else None
        self.board = BoardEncoder(cfg, self.identity, with_history=cfg.history != "none")
        self.history = build_history(cfg, EventEmbedding(cfg, self.identity, self.effect))
        self.action_encoder = ActionEncoder(cfg, self.identity, self.effect)
        self.context = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.head = ActionHead(cfg)

    # -- trunk -------------------------------------------------------------------
    def features(self, obs: Batch, state: HistoryState | None = None) -> Features:
        cards, glob = obs["cards"], obs["globals"]
        b = cards.shape[0]
        tokens = history_state = None
        if self.history is not None:
            events = obs.get("events")
            if events is None:  # EncodedVecEnv(event_length=0): no tokens this step
                events = cards.new_zeros(b, 0, E_EVENT)
                mask = cards.new_zeros(b, 0, dtype=torch.bool)
            else:
                mask = obs["event_mask"]
            summary, history_state, tokens = self.history(events, mask, state)
        else:
            summary = cards.new_zeros(b, self.cfg.d_model, dtype=torch.float)
        g, card_out, card_mask = self.board(cards, glob, summary if self.history is not None else None)
        m = card_mask.unsqueeze(-1).to(card_out.dtype)
        pooled = (card_out * m).sum(1) / m.sum(1).clamp(min=1)
        context = self.context(torch.cat([g, summary, pooled], -1))
        actions = self.action_encoder(obs["actions"], card_out)
        return Features(context, g, card_out, card_mask, actions, obs["action_mask"], summary, tokens, history_state)

    # -- actor ---------------------------------------------------------------------
    def logits(self, features: Features, belief: Tensor | None = None) -> Tensor:
        return self.head(features.context, features.actions, features.action_mask, belief)

    def forward(self, obs: Batch, belief: Tensor | None = None, state: HistoryState | None = None) -> PolicyOutput:
        """``belief``: ``[B, belief_dim]`` detached belief-head outputs (T4c.1); ``state``: streaming history."""
        f = self.features(obs, state)
        return PolicyOutput(self.logits(f, belief), f, f.history_state)

    @torch.no_grad()
    def act(self, obs: Batch, greedy: bool = False, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """-> (action rows ``[B]``, their log-probabilities ``[B]``) for acting in ``EncodedVecEnv``."""
        out = self(obs)
        a = out.greedy() if greedy else out.sample(generator)
        return a, out.log_probs().gather(1, a.unsqueeze(1)).squeeze(1)

    # -- reporting -------------------------------------------------------------------
    def parameter_report(self) -> str:
        counts = count_parameters(self)
        width = max(map(len, counts))
        return "\n".join(f"{k:<{width}}  {v:>12,}" for k, v in counts.items())


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Trainable parameters per top-level submodule (shared modules counted once, first name wins) + total."""
    counts: dict[str, int] = defaultdict(int)
    for name, p in module.named_parameters():
        if p.requires_grad:
            counts[name.split(".", 1)[0]] += p.numel()
    counts["total"] = sum(counts.values())
    return dict(counts)


__all__ = ["Features", "HistoryEncoder", "PolicyNet", "PolicyOutput", "count_parameters"]
