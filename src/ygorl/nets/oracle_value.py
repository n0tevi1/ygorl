"""Offline oracle value network (docs/oracle-value.md): the win probability of the acting player from a position.

Trained off-line by supervised regression on Monte Carlo outcomes ``z`` in {-1, 0, 1} of self-play games
(``tools/vn_collect.py`` / ``tools/vn_train.py``), not inside PPO. Its inputs are the actor observation plus, by
``inputs``:

- ``"public"``: the observation only (what the actor sees);
- ``"privileged"``: + the opponent ground truth (hand, deck, extra deck, set and face-down banished cards; the PPO
  critic's privileged input, docs/encoding.md 训练态真值);
- ``"oracle"``: + the next :data:`~ygorl.env.privileged.P_NEXT` draws of both players (``my_next`` / ``op_next``).

Network: a :class:`ygorl.nets.PolicyNet` trunk (board + event history; its action head is unused) whose decision
context and global token, concatenated with the :class:`PrivilegedEncoder` features, feed an MLP with a tanh output:
``V(s) = E[z | s, inputs]`` from the acting player's side, in [-1, 1] (win probability ``(1 + V) / 2``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from ygorl.cards.cdb import CardVocab
from ygorl.nets.actor_critic import ActorCritic, PrivilegedEncoder
from ygorl.env.privileged import P_ORDER
from ygorl.nets.batch import Batch
from ygorl.nets.config import NetConfig
from ygorl.nets.policy import PolicyNet, trim_padding

ValueInputs = Literal["public", "privileged", "oracle"]
DrawEncoding = Literal["order", "identity"]
FORMAT = "ygorl-oracle-value-1"
DRAW_POOLS = (1, 3, 10)  # identity draw encoding: mean of the next k draws for each k


class DrawEncoder(nn.Module):
    """The next draws of both players (``my_next`` / ``op_next``) through the trunk's card identity encoder (pretrained
    card semantics, unlike :class:`PrivilegedEncoder`'s own table): per player, the masked mean of the next ``k``
    cards for each ``k`` in :data:`DRAW_POOLS`; one linear layer."""

    def __init__(self, d: int, out_dim: int) -> None:
        super().__init__()
        self.out = nn.Sequential(nn.Linear(len(P_ORDER) * len(DRAW_POOLS) * d, out_dim), nn.ReLU())

    def forward(self, priv: Mapping[str, Tensor], identity: nn.Module) -> Tensor:
        parts = []
        for key in P_ORDER:
            idx = priv[key][..., 0]
            present = (idx > 0).unsqueeze(-1).float()
            emb = identity(idx) * present
            for k in DRAW_POOLS:
                parts.append(emb[:, :k].sum(1) / present[:, :k].sum(1).clamp(min=1))
        return self.out(torch.cat(parts, -1))


class OracleValueNet(nn.Module):
    def __init__(self, cfg: NetConfig, *, inputs: ValueInputs = "oracle", privileged_dim: int = 64,
                 hidden: int = 256, head_layers: int = 2, globals_input: bool = True,
                 draws: DrawEncoding = "order") -> None:  # fmt: skip
        super().__init__()
        if inputs not in ("public", "privileged", "oracle"):
            raise ValueError(f"unknown value inputs {inputs!r}")
        self.cfg, self.inputs = cfg, inputs
        self.trunk = PolicyNet(cfg)
        self.privileged = (
            None if inputs == "public"
            else PrivilegedEncoder(cfg.vocab_size, privileged_dim, deck_order=inputs == "oracle" and draws == "order")
        )  # fmt: skip
        d = cfg.d_model
        self.draws = DrawEncoder(d, privileged_dim) if inputs == "oracle" and draws == "identity" else None
        width = (2 if globals_input else 1) * d + (privileged_dim if self.privileged is not None else 0)
        width += privileged_dim if self.draws is not None else 0
        layers: list[nn.Module] = []
        for _ in range(head_layers):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.head = nn.Sequential(*layers, nn.Linear(width, 1))
        self.privileged_dim, self.hidden, self.head_layers, self.globals_input = (
            privileged_dim, hidden, head_layers, globals_input)  # fmt: skip
        self.draw_encoding = draws

    @classmethod
    def from_actor_critic(cls, model: ActorCritic, inputs: ValueInputs = "oracle",
                          draws: DrawEncoding = "order") -> OracleValueNet:  # fmt: skip
        """A value network equal to the PPO critic's V head (shared-trunk privileged critic): the actor trunk, the
        privileged encoder and the critic MLP + V head are copied; the deck-order inputs a PPO critic without them
        lacks start with zero weight, so the copy computes exactly the critic's V."""
        if model.critic_trunk is not None or model.privileged is None or inputs == "public":
            raise ValueError("needs a shared-trunk privileged critic and privileged / oracle inputs")
        crit = model.critic
        hidden = crit.v_head.net[0].in_features
        net = cls(model.cfg, inputs=inputs, privileged_dim=model.privileged.dim, hidden=hidden, head_layers=3,
                  globals_input=False, draws=draws)  # fmt: skip
        net.trunk.load_state_dict(model.actor.state_dict())
        src, dst = model.privileged.state_dict(), net.privileged.state_dict()
        for k, w in src.items():
            if k == "out.0.weight" and dst[k].shape != w.shape:  # extra deck-order columns: zero
                full = torch.zeros_like(dst[k])
                full[:, : w.shape[1]] = w
                w = full
            dst[k] = w
        net.privileged.load_state_dict(dst)
        linears = [crit.trunk[0][0], crit.trunk[0][2], crit.v_head.net[0], crit.v_head.net[2]]
        for mine, theirs in zip([m for m in net.head if isinstance(m, nn.Linear)], linears, strict=True):
            w = theirs.weight.detach()
            if mine.weight.shape != w.shape:  # the draw features' columns: zero
                w = torch.cat([w, w.new_zeros(w.shape[0], mine.weight.shape[1] - w.shape[1])], 1)
            mine.load_state_dict({"weight": w, "bias": theirs.bias.detach()})
        return net

    def forward(self, obs: Batch, privileged: Mapping[str, Tensor] | None = None) -> Tensor:
        """``[B]`` value in [-1, 1] of the acting player."""
        f = self.trunk._features(trim_padding(obs))
        parts = [f.context, f.globals] if self.globals_input else [f.context]
        if self.privileged is not None:
            if privileged is None:
                raise ValueError(f"this value network reads {self.inputs} inputs: pass the privileged tensors")
            parts.append(self.privileged(privileged))
        if self.draws is not None:
            parts.append(self.draws(privileged, self.trunk.identity))
        return torch.tanh(self.head(torch.cat(parts, -1)).squeeze(-1))

    def options(self) -> dict:
        return {
            "inputs": self.inputs,
            "privileged_dim": self.privileged_dim,
            "hidden": self.hidden,
            "head_layers": self.head_layers,
            "globals_input": self.globals_input,
            "draws": self.draw_encoding,
        }


def save_value_net(net: OracleValueNet, vocab: CardVocab, event_length: int, path: str | Path, **extra) -> None:
    passwords = [vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(vocab))]
    state = {"format": FORMAT, "net_config": net.cfg.to_dict(), "options": net.options(), "passwords": passwords,
             "event_length": int(event_length), "model": net.state_dict(), **extra}  # fmt: skip
    torch.save(state, Path(path))


def load_value_net(path: str | Path) -> tuple[OracleValueNet, CardVocab, int]:
    """-> (network in eval mode on CPU, its card vocab, event window length)."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError(f"{path}: not an oracle value network ({FORMAT})")
    net = OracleValueNet(NetConfig.from_dict(state["net_config"]), **state["options"])
    net.load_state_dict(state["model"])
    net.eval().requires_grad_(False)
    return net, CardVocab(state["passwords"]), int(state["event_length"])


__all__ = [
    "DRAW_POOLS",
    "FORMAT",
    "DrawEncoder",
    "DrawEncoding",
    "OracleValueNet",
    "ValueInputs",
    "load_value_net",
    "save_value_net",
]
