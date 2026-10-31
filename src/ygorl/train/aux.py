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

from typing import NamedTuple

import torch
from torch import Tensor

AUX_TARGETS = ("opp_monsters", "own_monsters", "lp_swing", "first_hand")
_MY_TURN, _TURN, _MY_LP, _OP_LP = 2, 3, 5, 6  # globals columns
_IS_FIRST, _MY_HAND = 1, 8  # g[1]=1 when the viewer is the first player; g[8]=viewer's hand card count
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
    hand = g[:, _MY_HAND].float().cpu().view(T, B)  # viewer's hand count at each row
    is_first = g[:, _IS_FIRST].long().cpu().view(T, B) == 1  # True when this row's viewer went first
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
                # Seat-aware "hand preservation" target: only the first player's rows learn to value
                # keeping cards in hand (the second seat should NOT be penalized for spending its
                # extra card on aggression; audit R4 2026-10-31: the policy over-extends from the
                # first seat, converting handtrap resources into board pieces with no threat value).
                if bool(is_first[t, b]):
                    h_u = hand[u, b] if same else hand[t, b]  # own hand at next own turn (or current if not same)
                    targets[t, b, 3] = h_u / 6.0
                valid[t, b] = True
            start = end + 1
    return targets, valid


# -------------------------------------------------------------------------------- survival potential (§8.2)

SURVIVE_MONSTER_LOSS = 1  # "not broken": at most this many fewer own monsters at the next own turn ...
SURVIVE_LP_LOSS = 2000  # ... or at most this much LP lost


class TurnBoundaries(NamedTuple):
    """A seat's turn boundaries in a rollout (``[T, B]`` each) and the survival-potential targets at them."""

    boundary: Tensor  # bool: the seat's last decision of its own turn (confirmed: the turn ended in the rollout)
    survive: Tensor  # float 0 / 1 at valid boundaries, 0 elsewhere
    valid: Tensor  # bool: boundary whose outcome is observed (next own turn reached, or a win / loss)
    next_row: Tensor  # long: row of the same seat's next boundary in the same game, -1 if none in the rollout
    terminal: Tensor  # bool: boundary with no later one of its seat because the game ended (not truncated)


@torch.no_grad()
def turn_boundaries(obs, players: Tensor, dones: Tensor, rewards: Tensor, truncated: Tensor) -> TurnBoundaries:
    """Turn boundaries and survival targets for a rollout laid out as docs/training.md §1 (row ``t * B + b``).

    A row of seat ``p`` decided on its own turn ``τ`` (globals ``is_my_turn``) is a boundary when no later row of
    ``p`` in the same game has turn ``τ`` and the turn is known to have ended (a later row has a larger turn, or the
    game ended at a terminal row). Its target ``survive`` looks at ``p``'s first row at turn ``τ + 2`` or later (its
    next own turn): 1 if ``p``'s LP there is positive and (``p`` controls at least the monsters it had at the boundary
    minus :data:`SURVIVE_MONSTER_LOSS`, or lost at most :data:`SURVIVE_LP_LOSS` LP since the boundary). Without such a
    row: 1 / 0 when the game ended with ``p`` winning / losing; invalid on a draw, a truncation or the segment end.
    The boundary row is ``p``'s last observation of the turn: its board is the one ``p`` ends the turn with, up to
    that row's own action (often the end-phase pass).
    """
    T, B = players.shape
    g = obs["globals"]
    cards = obs["cards"]
    own = ((cards[..., _LOCATION] == _MZONE) & (cards[..., _CONTROLLER] == 0)).sum(-1).cpu().view(T, B).tolist()
    turn = g[:, _TURN].long().cpu().view(T, B).tolist()
    mine = (g[:, _MY_TURN] == 1).cpu().view(T, B).tolist()
    lp = g[:, _MY_LP].long().cpu().view(T, B).tolist()
    pl = players.cpu().tolist()
    dn, tr, rw = dones.cpu().tolist(), truncated.cpu().tolist(), rewards.cpu().tolist()
    boundary = torch.zeros(T, B, dtype=torch.bool)
    survive = torch.zeros(T, B)
    valid = torch.zeros(T, B, dtype=torch.bool)
    next_row = torch.full((T, B), -1, dtype=torch.long)
    terminal = torch.zeros(T, B, dtype=torch.bool)
    for b in range(B):
        start = 0
        ends = [t for t in range(T) if dn[t][b]] + ([T - 1] if not dn[T - 1][b] else [])
        for end in ends:  # one game: rows start..end (inclusive)
            closed = bool(dn[end][b]) and not bool(tr[end][b])
            for p in (0, 1):
                rows = [t for t in range(start, end + 1) if pl[t][b] == p]
                found = []
                for k, t in enumerate(rows):
                    tau = turn[t][b]
                    if not mine[t][b] or (k + 1 < len(rows) and turn[rows[k + 1]][b] == tau):
                        continue
                    if not (closed or turn[end][b] > tau):
                        continue  # the turn may go on past the segment
                    found.append(t)
                    u = next((r for r in rows[k + 1 :] if turn[r][b] >= tau + 2), None)
                    if u is not None:
                        ok = lp[u][b] > 0 and (
                            own[u][b] >= own[t][b] - SURVIVE_MONSTER_LOSS or lp[u][b] >= lp[t][b] - SURVIVE_LP_LOSS
                        )
                        survive[t, b], valid[t, b] = float(ok), True
                    elif closed:
                        z = rw[end][b] if pl[end][b] == p else -rw[end][b]
                        if z != 0:
                            survive[t, b], valid[t, b] = float(z > 0), True
                for t, n in zip(found, found[1:]):
                    next_row[t, b] = n
                if found and closed:
                    terminal[found[-1], b] = True
                boundary[found, b] = True
            start = end + 1
    return TurnBoundaries(boundary, survive, valid, next_row, terminal)


def phi_shaping(phi: Tensor, tb: TurnBoundaries, gamma: float, rows: Tensor | None = None) -> Tensor:
    """Potential-based shaping rewards ``[T, B]`` (Ng et al. 1999) from the potential ``phi`` ``[T, B]`` (read at
    boundary rows only): at each boundary ``s`` of a seat, ``γ^k Φ(s') − Φ(s)`` with ``s'`` the seat's next boundary
    ``k`` rows later in the column (``γ^k`` is the estimator's discount between the two rows, ``1`` at ``γ = 1``);
    ``Φ(s') = 0`` after the seat's last boundary of a game that ended, so the increments of a seat telescope to
    ``−Φ(first boundary)`` over a complete game. A last boundary whose game goes on past the rollout gets 0 (its next
    potential is unobserved). Other rows get 0. ``rows`` (bool ``[T, B]``) restricts the shaping to those rows."""
    phi = phi.float()
    T, B = phi.shape
    nxt = tb.next_row.clamp(min=0)
    phi_next = phi.gather(0, nxt)
    steps = (nxt - torch.arange(T).unsqueeze(1)).clamp(min=0).to(phi)
    linked = tb.next_row >= 0
    out = torch.where(linked, gamma**steps * phi_next - phi, torch.zeros_like(phi))
    out = torch.where(tb.terminal, -phi, out)
    keep = tb.boundary if rows is None else tb.boundary & rows
    return torch.where(keep, out, torch.zeros_like(out))


__all__ = [
    "AUX_TARGETS",
    "SURVIVE_LP_LOSS",
    "SURVIVE_MONSTER_LOSS",
    "TurnBoundaries",
    "aux_targets",
    "phi_shaping",
    "turn_boundaries",
]
