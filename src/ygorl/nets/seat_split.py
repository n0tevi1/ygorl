"""Seat-split agent (#83 strength experiment): one network for the first player, another for the second player.

:class:`SeatSplit` wraps two modules of the same kind (two :class:`~ygorl.nets.ActorCritic` while training, two
:class:`~ygorl.nets.PolicyNet` when a checkpoint is loaded for play) and routes every observation row by
``globals[1]`` (``is_first``: the viewer goes first, docs/encoding.md): first-player rows go to ``nets[0]``,
second-player rows to ``nets[1]``; the outputs are put back in row order. It has the interface of the module it
wraps (``collate`` / ``collate_privileged``, ``forward`` -> ``.logits`` (and ``.q`` / ``.v``), ``policy_logits``),
so the rollout collector, the snapshot pool, batched evaluation and ``policy:`` agents use it unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from torch import Tensor, nn

from ygorl.nets.batch import Batch, collate, policy_logits

IS_FIRST = 1  # globals column: the viewer goes first (docs/encoding.md)


def first_player_rows(obs: Batch) -> Tensor:
    """Bool ``[n]``: rows decided by the first player (``globals[:, 1] == 1``)."""
    return obs["globals"][:, IS_FIRST] == 1


def _index(x, idx: Tensor):
    if x is None:
        return None
    if isinstance(x, Tensor):
        return x[idx]
    return {k: v[idx] for k, v in x.items()}


class SplitOutput(NamedTuple):
    logits: Tensor  # forward output of wrapped PolicyNets: the logits only


def _merge(n: int, parts: list[tuple[Tensor, Tensor]]) -> Tensor:
    first = parts[0][1]
    out = first.new_zeros((n, *first.shape[1:]))
    for idx, x in parts:
        out[idx] = x.to(out.dtype)
    return out


class SeatSplit(nn.Module):
    """``nets[0]`` plays the first player's decisions, ``nets[1]`` the second player's."""

    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        super().__init__()
        self.nets = nn.ModuleList([first, second])
        self.cfg = getattr(first, "cfg", None)
        self.actor_critic = hasattr(first, "critic")  # ActorCritic halves (forward -> logits, q, v)

    @staticmethod
    def collate(observations, device=None) -> Batch:
        return collate(observations, device)

    def collate_privileged(self, privileged, device=None):
        fn = getattr(self.nets[0], "collate_privileged", None)
        return fn(privileged, device) if fn is not None else None

    def _route(self, obs: Batch, call: Callable[[nn.Module, Batch, int], object], merge: Callable):
        first = first_player_rows(obs)
        groups = [(s, sel) for s, sel in ((0, first), (1, ~first)) if bool(sel.any())]
        if len(groups) <= 1:  # one seat only (or no rows): no copy
            return call(self.nets[groups[0][0] if groups else 0], obs, None)
        parts = []
        for s, sel in groups:
            idx = sel.nonzero().squeeze(1)
            parts.append((idx, call(self.nets[s], _index(obs, idx), idx)))
        return merge(obs["globals"].shape[0], parts)

    def forward(self, obs: Batch, privileged=None):
        if not self.actor_critic:
            return SplitOutput(self.policy_logits(obs))

        def call(net, o, idx):
            return net(o, privileged if privileged is None or idx is None else _index(privileged, idx))

        def merge(n, parts):
            fields = parts[0][1]._fields
            return type(parts[0][1])(**{f: _merge(n, [(i, getattr(o, f)) for i, o in parts]) for f in fields})

        return self._route(obs, call, merge)

    def policy_logits(self, obs: Batch) -> Tensor:
        return self._route(obs, lambda net, o, _idx: policy_logits(net, o), _merge)


__all__ = ["IS_FIRST", "SeatSplit", "SplitOutput", "first_player_rows"]
