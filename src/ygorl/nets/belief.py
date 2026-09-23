"""Belief heads of the opponent model and their masked losses (T4c.1, docs/design/04-opponent-model.md).

Five heads (shapes of ``ygorl.eval.beliefs``; K + 1 deck types, C candidate cards, R role bits,
S = 15 set zones, A candidate actions):

- ``deck_type`` ``[B, K+1]``: softmax over the meta types and "other";
- ``remaining_copies`` ``[B, C, 4]``: copies 0..3 left hidden in main + extra deck, one 4-way
  classifier per candidate (multi-head classification, not sigmoid, so the counts are calibrated);
- ``hand`` ``[B, C]`` + ``hand_roles`` ``[B, R]``: P(>= 1 in hand) per candidate and per role bit;
- ``set_cards`` ``[B, S, C]``: class of each face-down card;
- ``responded`` ``[B, A]`` (``[B]`` without action embeddings): P(my activation is responded to).

Every head is a residual on the HDT-style posterior of ``ygorl.env.belief_prior.hdt_prior``: the
last layers start at zero, so an untrained head outputs the meta prior / HDT filter exactly ("meta
先验初始化"). Constraints by construction: cards already public in hand are forced to 1, and
remaining-copy classes above ``3 - visible`` are masked (the copies seen are deducted).

The heads sit on the policy trunk (``PolicyNet.features``, auxiliary-loss channel) and their
probabilities enter the actor detached through ``NetConfig.belief_dim`` (:class:`BeliefPolicy`).
Specification and experiment: docs/belief-heads.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ygorl.eval.beliefs import BeliefBatch, Head
from ygorl.nets.heads import MASKED_LOGIT
from ygorl.nets.policy import PolicyNet, PolicyOutput

HEADS = ("deck_type", "remaining_copies", "hand", "hand_roles", "set_cards", "responded")
LATE_HEADS = ("hand", "hand_roles", "set_cards")  # design 04: weighted up later in training
PRIOR_KEYS = ("deck_type", "remaining_copies", "hand", "hand_roles", "set_cards", "features", "public_hand",
              "public_roles", "max_copies")  # fmt: skip
N_COPY_CLASSES = 4
FORCED_LOGIT = -MASKED_LOGIT  # "public -> 1 by construction"


@dataclass(frozen=True)
class BeliefConfig:
    """Sizes of the belief heads; build from a meta table with :meth:`from_meta`.

    ``context_dim``: width of the trunk context the heads read (``NetConfig.d_model``; 0 = the heads
    see only the HDT features). ``action_dim``: width of the candidate-action embeddings for the
    action-level ``responded`` head (0 = one probability per decision point).
    """

    n_deck_types: int  # K + 1
    n_cards: int  # C
    n_roles: int  # R
    feature_dim: int  # BeliefPrior.features width
    context_dim: int = 0
    action_dim: int = 0
    n_set_zones: int = 15
    hidden: int = 256
    card_dim: int = 32  # candidate embedding for the set-card head
    prior_eps: float = 1e-4  # probabilities are clipped to [eps, 1 - eps] before taking logs
    dropout: float = 0.1  # on the shared hidden layer (the labels are few and correlated within a game)
    detach_context: bool = False  # True: belief losses do not train the shared trunk (T4c.2 ablation)

    @property
    def policy_dim(self) -> int:
        """Width of :meth:`BeliefOutput.policy_features` (the value for ``NetConfig.belief_dim``)."""
        return self.n_deck_types + 2 * self.n_cards + self.n_roles

    @classmethod
    def from_meta(cls, meta, **kw) -> BeliefConfig:
        """Sizes from a :class:`ygorl.env.belief_prior.MetaTable`."""
        return cls(meta.n_deck_types, meta.n_cards, meta.n_roles, meta.feature_dim, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BeliefConfig:
        return cls(**data)


@dataclass
class BeliefOutput:
    """Logits of every head (constraints already applied)."""

    deck_type: Tensor  # [B, K+1]
    remaining_copies: Tensor  # [B, C, 4]
    hand: Tensor  # [B, C]
    hand_roles: Tensor  # [B, R]
    set_cards: Tensor  # [B, S, C]
    responded: Tensor  # [B, A] or [B]

    def probs(self) -> dict[str, Tensor]:
        return {
            "deck_type": torch.softmax(self.deck_type, -1),
            "remaining_copies": torch.softmax(self.remaining_copies, -1),
            "hand": torch.sigmoid(self.hand),
            "hand_roles": torch.sigmoid(self.hand_roles),
            "set_cards": torch.softmax(self.set_cards, -1),
            "responded": torch.sigmoid(self.responded),
        }

    def policy_features(self) -> Tensor:
        """``[B, K+1 + 2C + R]``: deck-type probs, hand probs, expected remaining copies / 3, role probs.

        The actor input of design 04 channel (a); ``PolicyNet`` detaches it.
        """
        p = self.probs()
        copies = (p["remaining_copies"] * torch.arange(N_COPY_CLASSES, device=self.hand.device)).sum(-1) / 3
        return torch.cat([p["deck_type"], p["hand"], copies, p["hand_roles"]], -1)


def _log_prob(p: Tensor, eps: float) -> Tensor:
    return torch.log(p.clamp(min=eps))


def _logit(p: Tensor, eps: float) -> Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


class BeliefHeads(nn.Module):
    """``h = MLP(HDT features ⊕ trunk context)``; each head = ``scale · log prior + residual(h)``."""

    def __init__(self, cfg: BeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hid, c = cfg.hidden, cfg.n_cards
        self.norm = nn.LayerNorm(cfg.feature_dim + cfg.context_dim)
        self.trunk = nn.Sequential(nn.Linear(cfg.feature_dim + cfg.context_dim, hid), nn.ReLU(), nn.Linear(hid, hid),
                                   nn.ReLU(), nn.Dropout(cfg.dropout))  # fmt: skip
        self.deck_type = nn.Linear(hid, cfg.n_deck_types)
        self.remaining_copies = nn.Linear(hid, c * N_COPY_CLASSES)
        self.hand = nn.Linear(hid, c)
        self.hand_roles = nn.Linear(hid, cfg.n_roles)
        self.zone = nn.Embedding(cfg.n_set_zones, hid)
        self.set_query = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, cfg.card_dim))
        self.card = nn.Parameter(torch.randn(c, cfg.card_dim) * 0.1)
        self.card_bias = nn.Parameter(torch.zeros(c))
        self.prior_scale = nn.Parameter(torch.ones(5))  # per head, on the log prior
        if cfg.action_dim:
            self.resp_ctx = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            self.resp_act = nn.Linear(cfg.action_dim, hid)
        self.resp_out = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        with torch.no_grad():  # residuals start at zero: the untrained heads are the HDT prior
            for layer in (self.deck_type, self.remaining_copies, self.hand, self.hand_roles, self.set_query[-1]):
                layer.weight.zero_()
                layer.bias.zero_()
            if cfg.action_dim:
                self.resp_ctx[-1].weight.mul_(0.1)

    def forward(self, prior: Mapping[str, Tensor], context: Tensor | None = None, actions: Tensor | None = None,
                action_mask: Tensor | None = None) -> BeliefOutput:  # fmt: skip
        """``prior``: tensors of :class:`ygorl.env.belief_prior.BeliefPrior` (:func:`prior_tensors`);
        ``context`` ``[B, context_dim]``; ``actions`` ``[B, A, action_dim]`` for the action-level head."""
        cfg, eps = self.cfg, self.cfg.prior_eps
        x = prior["features"].float()
        if cfg.context_dim:
            if context is None:
                raise ValueError("these heads read the trunk context (context_dim > 0)")
            x = torch.cat([x, context.detach() if cfg.detach_context else context], -1)
        h = self.trunk(self.norm(x))
        b, c = h.shape[0], cfg.n_cards
        s = self.prior_scale
        deck = s[0] * _log_prob(prior["deck_type"].float(), eps) + self.deck_type(h)
        copies = s[1] * _log_prob(prior["remaining_copies"].float(), eps)
        copies = copies + self.remaining_copies(h).view(b, c, N_COPY_CLASSES)
        too_many = torch.arange(N_COPY_CLASSES, device=h.device) > prior["max_copies"].unsqueeze(-1)
        copies = copies.masked_fill(too_many, MASKED_LOGIT)
        hand = s[2] * _logit(prior["hand"].float(), eps) + self.hand(h)
        hand = hand.masked_fill(prior["public_hand"].bool(), FORCED_LOGIT)
        roles = s[3] * _logit(prior["hand_roles"].float(), eps) + self.hand_roles(h)
        roles = roles.masked_fill(prior["public_roles"].bool(), FORCED_LOGIT)
        q = self.set_query(h.unsqueeze(1) + self.zone.weight)  # [B, S, card_dim]
        set_res = torch.einsum("bsd,cd->bsc", q, self.card) / math.sqrt(cfg.card_dim) + self.card_bias
        set_cards = s[4] * _log_prob(prior["set_cards"].float(), eps) + set_res
        if cfg.action_dim and actions is not None:
            ctx = self.resp_ctx(h).unsqueeze(1) + self.resp_act(actions)  # [B, A, hid]
            responded = self.resp_out(torch.relu(ctx)).squeeze(-1)
            if action_mask is not None:
                responded = responded.masked_fill(~action_mask.bool(), 0.0)
        else:
            responded = self.resp_out(h).squeeze(-1)
        return BeliefOutput(deck, copies, hand, roles, set_cards, responded)


def prior_tensors(prior, device: torch.device | str | None = None) -> dict[str, Tensor]:
    """A :class:`ygorl.env.belief_prior.BeliefPrior` (numpy, batched) -> the tensors :class:`BeliefHeads` reads."""
    arrays = prior.arrays() if hasattr(prior, "arrays") else prior
    out = {}
    for key in PRIOR_KEYS:
        a = np.asarray(arrays[key])
        out[key] = torch.as_tensor(a if a.dtype == bool or a.dtype.kind == "i" else a.astype(np.float32), device=device)
    return out


# --- losses -----------------------------------------------------------------------------------


def _masked_mean(loss: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(loss.dtype)
    return (loss * m).sum() / m.sum().clamp(min=1)


def chosen_responded(logits: Tensor, action: Tensor | None) -> Tensor:
    """``[B]`` responded logit of the chosen action (action-level head) or the head itself (``[B]``)."""
    if logits.dim() == 1:
        return logits
    if action is None:
        raise ValueError("the action-level responded head needs targets['responded_action']")
    return logits.gather(1, action.clamp(min=0).long().unsqueeze(1)).squeeze(1)


def belief_losses(out: BeliefOutput, targets: Mapping[str, Tensor], weights: Mapping[str, float] | None = None,
                  prior: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:  # fmt: skip
    """Per-head masked cross-entropy (mean over unmasked entries) and their weighted sum ``total``.

    ``targets``: ``<head>`` and ``<head>_mask`` (bool, True = train) for each head present
    (``ygorl.env.privileged.belief_targets`` / ``role_targets`` layouts); ``deck_type`` ``[B]``
    (-1 = unknown); ``responded`` ``[B]`` 0/1 with ``responded_action`` ``[B]`` (the chosen row)
    for the action-level head. Remaining-copy targets above ``prior["max_copies"]`` are dropped
    (impossible by construction; happens only with inconsistent evidence). Missing heads are skipped.
    """
    w = {h: 1.0 for h in HEADS} | dict(weights or {})
    losses: dict[str, Tensor] = {}

    def mask_of(name: str, like: Tensor) -> Tensor:
        m = targets.get(f"{name}_mask")
        return torch.ones_like(like, dtype=torch.bool) if m is None else m.bool()

    if "deck_type" in targets:
        t = targets["deck_type"].long()
        m = mask_of("deck_type", t) & (t >= 0)
        losses["deck_type"] = _masked_mean(F.cross_entropy(out.deck_type, t.clamp(min=0), reduction="none"), m)
    if "remaining_copies" in targets:
        t = targets["remaining_copies"].long().clamp(0, N_COPY_CLASSES - 1)
        m = mask_of("remaining_copies", t)
        if prior is not None:
            m = m & (t <= prior["max_copies"])
        ce = F.cross_entropy(out.remaining_copies.flatten(0, 1), t.flatten(), reduction="none").view_as(t)
        losses["remaining_copies"] = _masked_mean(ce, m)
    for name in ("hand", "hand_roles"):
        if name in targets:
            t = targets[name].float()
            logit = getattr(out, name)
            losses[name] = _masked_mean(F.binary_cross_entropy_with_logits(logit, t, reduction="none"), mask_of(name, t))
    if "set_cards" in targets:
        t = targets["set_cards"].long()
        m = mask_of("set_cards", t) & (t >= 0)
        ce = F.cross_entropy(out.set_cards.flatten(0, 1), t.clamp(min=0).flatten(), reduction="none").view_as(t)
        losses["set_cards"] = _masked_mean(ce, m)
    if "responded" in targets:
        t = targets["responded"].float()
        logit = chosen_responded(out.responded, targets.get("responded_action"))
        m = mask_of("responded", t) & (t >= 0)
        losses["responded"] = _masked_mean(F.binary_cross_entropy_with_logits(logit, t.clamp(min=0), reduction="none"), m)
    losses["total"] = sum(w[k] * v for k, v in losses.items())
    return losses


def loss_weights(progress: float, late: float = 2.0) -> dict[str, float]:
    """Design 04: hand / role / set-card heads weighted up over training, 1 -> ``late`` as ``progress`` 0 -> 1."""
    p = min(max(progress, 0.0), 1.0)
    return {h: 1.0 + (late - 1.0) * p if h in LATE_HEADS else 1.0 for h in HEADS}


def evaluation_batch(probs: Mapping[str, np.ndarray | Tensor], targets: Mapping[str, np.ndarray | Tensor]) -> BeliefBatch:
    """Predicted probabilities + targets -> ``ygorl.eval.beliefs.BeliefBatch`` (heads missing in either are left out).

    An action-level ``responded`` ``[N, A]`` is reduced to the chosen row (``responded_action``).
    """

    def np_(x):
        return x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)

    heads = {}
    for name in HEADS:
        if name not in probs or name not in targets:
            continue
        p, t = np_(probs[name]).astype(float), np_(targets[name])
        m = targets.get(f"{name}_mask")
        m = np.ones(t.shape, dtype=bool) if m is None else np_(m).astype(bool)
        if name in ("deck_type", "set_cards", "responded"):
            m = m & (t >= 0)
        if name == "responded" and p.ndim == 2:
            p = np.take_along_axis(p, np.clip(np_(targets["responded_action"]), 0, None)[:, None].astype(np.int64), 1)[:, 0]
        heads[name] = Head(p, np.where(m, t, 0), m)
    return BeliefBatch(**heads)


# --- wiring to the policy ---------------------------------------------------------------------


class BeliefPolicy(nn.Module):
    """``PolicyNet`` + :class:`BeliefHeads` on its trunk (design 04 channels (a) and (b)).

    The heads read ``Features.context`` (and the candidate-action embeddings for the action-level
    ``responded`` head); their losses train the shared trunk unless ``detach_context``. With
    ``feed_policy`` the belief probabilities enter the actor through ``NetConfig.belief_dim``
    (``PolicyNet`` detaches them, so the policy cannot use the heads as a bypass).
    """

    def __init__(self, net: PolicyNet, heads: BeliefHeads, feed_policy: bool = True) -> None:
        super().__init__()
        d, cfg = net.cfg.d_model, heads.cfg
        if cfg.context_dim not in (0, d) or cfg.action_dim not in (0, d):
            raise ValueError(f"belief heads read the trunk: context_dim / action_dim must be 0 or d_model={d}")
        if feed_policy and net.cfg.belief_dim != cfg.policy_dim:
            raise ValueError(f"NetConfig.belief_dim={net.cfg.belief_dim} but the heads output {cfg.policy_dim} "
                             "features; build the net with belief_dim=BeliefConfig.policy_dim or feed_policy=False")  # fmt: skip
        self.net, self.heads, self.feed_policy = net, heads, feed_policy

    def forward(self, obs, prior: Mapping[str, Tensor], state=None) -> tuple[PolicyOutput, BeliefOutput]:
        f = self.net.features(obs, state)
        cfg = self.heads.cfg
        beliefs = self.heads(prior, f.context if cfg.context_dim else None, f.actions if cfg.action_dim else None,
                             f.action_mask)  # fmt: skip
        logits = self.net.logits(f, beliefs.policy_features() if self.feed_policy else None)
        return PolicyOutput(logits, f, f.history_state), beliefs


__all__ = [
    "HEADS",
    "BeliefConfig",
    "BeliefHeads",
    "BeliefOutput",
    "BeliefPolicy",
    "belief_losses",
    "chosen_responded",
    "evaluation_batch",
    "loss_weights",
    "prior_tensors",
]
