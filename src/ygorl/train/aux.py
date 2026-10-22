"""Action-conditioned auxiliary targets for the critic (#440, design R4): short-horizon observables of the taken
action, computed from the rollout's own later observations, so no branching and no reward change.

For a row decided at turn ``τ`` by seat ``p``, the target row is the first later row of the same game (same column,
no episode end in between) whose turn counter exceeds ``τ``: the first decision after the current turn ended.
From ``p``'s perspective at that row:

- ``opp_monsters``: cards the opponent controls in its monster zone / 5;
- ``own_monsters``: cards ``p`` controls in its monster zone / 5;
- ``lp_swing``: (p's LP − opponent's LP there) − (the same at the decision) / 8000.

Rows without such a later row in the segment (segment end, game end) are invalid and excluded from the loss.
The observation's card table is viewer-relative (docs/encoding.md), so counts flip when the target row's seat differs.
"""

from __future__ import annotations

import torch
from torch import Tensor

AUX_TARGETS = ("opp_monsters", "own_monsters", "lp_swing")
_TURN, _MY_LP, _OP_LP = 3, 5, 6  # globals columns
_LOCATION, _CONTROLLER, _MZONE = 1, 4, 3  # card columns and the monster-zone code


@torch.no_grad()
def aux_targets(obs, players: Tensor, dones: Tensor) -> tuple[Tensor, Tensor]:
    """-> ``(targets [T, B, K], valid [T, B])`` for a rollout laid out as docs/training.md §1 (row ``t * B + b``)."""
    T, B = players.shape
    g = obs["globals"]
    cards = obs["cards"]
    mz = cards[..., _LOCATION] == _MZONE
    own = (mz & (cards[..., _CONTROLLER] == 0)).sum(-1).float().cpu().view(T, B)
    opp = (mz & (cards[..., _CONTROLLER] == 1)).sum(-1).float().cpu().view(T, B)
    turn = g[:, _TURN].long().cpu().view(T, B)
    adv = (g[:, _MY_LP] - g[:, _OP_LP]).float().cpu().view(T, B)
    players, dones = players.cpu(), dones.cpu()
    targets = torch.zeros(T, B, len(AUX_TARGETS))
    valid = torch.zeros(T, B, dtype=torch.bool)
    for b in range(B):
        start = 0
        ends = torch.nonzero(dones[:, b]).squeeze(1).tolist() + ([T - 1] if not bool(dones[T - 1, b]) else [])
        for end in ends:  # one game: rows start..end (inclusive); turn is non-decreasing inside it
            seg = turn[start : end + 1, b]
            nxt = torch.searchsorted(seg, seg, right=True)  # first index with a larger turn
            for t in range(start, end + 1):
                j = int(nxt[t - start])
                if j >= seg.shape[0]:
                    continue  # the turn did not end inside the segment
                u = start + j
                same = bool(players[u, b] == players[t, b])
                o = opp[u, b] if same else own[u, b]
                w = own[u, b] if same else opp[u, b]
                a_u = adv[u, b] if same else -adv[u, b]
                targets[t, b, 0] = o / 5.0
                targets[t, b, 1] = w / 5.0
                targets[t, b, 2] = (a_u - adv[t, b]) / 8000.0
                valid[t, b] = True
            start = end + 1
    return targets, valid


__all__ = ["AUX_TARGETS", "aux_targets"]
