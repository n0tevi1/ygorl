"""Opt-in public attacker/target relation at adjacent attack-target prompts."""

import torch
from torch import Tensor, nn

from ygorl.engine import constants as C
from ygorl.env.encoding import ACTION_KINDS
from ygorl.env.events import ATTACK_CHOICE_EVENT


def battle_context_rows(obs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Return zero-based attacker/target rows and a per-action validity mask.

    Resolve only the LAST event, not any old attack in the window. Coordinates
    distinguish equal card identities; identity additionally rejects a replaced
    occupant. Unknown/truncated/ambiguous context abstains. No core or privileged
    data is consulted. Indices on abstaining rows are placeholders, never labels.
    """
    cards, actions, glob = obs["cards"], obs["actions"], obs["globals"]
    batch, n_cards = cards.shape[:2]
    attacker = torch.zeros(batch, dtype=torch.long, device=cards.device)
    targets = (actions[..., 1] - 1).clamp(0, n_cards - 1)
    valid = torch.zeros_like(obs["action_mask"], dtype=torch.bool)
    events = obs.get("events")
    if events is None or events.shape[1] == 0:
        return attacker, targets, valid
    emask = obs["event_mask"].bool()
    positions = torch.arange(events.shape[1], device=cards.device).expand(batch, -1)
    last = positions.masked_fill(~emask, -1).max(-1).values
    event = events[torch.arange(batch, device=cards.device), last.clamp_min(0)]
    pending = (
        (last >= 0)
        & (event[:, 0] == ATTACK_CHOICE_EVENT)
        & (event[:, 1] == 1)
        & (event[:, 2] > 1)
        & (event[:, 4] == 1)
        & (event[:, 5] == 3)
        & (event[:, 12] == ACTION_KINDS.index("attack") + 1)
        & (event[:, 13] == C.MSG_SELECT_BATTLECMD)
        & (event[:, 15] == glob[:, 3])
        & (event[:, 16] == glob[:, 4])
        & (event[:, 17] == 1)
        & (glob[:, 2] == 1)
        & ((glob[:, 4] == C.PHASE_BATTLE_START.bit_length()) | (glob[:, 4] == C.PHASE_BATTLE_STEP.bit_length()))
        & (glob[:, 17] == 0)
        & (glob[:, 18] == C.MSG_SELECT_CARD)
    )
    matches = (
        (cards[..., 0] == event[:, 2, None])
        & (cards[..., 1] == 3)
        & (cards[..., 2] == event[:, 6, None])
        & (cards[..., 4] == 0)
        & (cards[..., 6] == 1)
        & (cards[..., 7] == 1)
    )
    pending &= matches.sum(-1) == 1
    attacker = matches.long().argmax(-1)
    target_cards = cards.gather(1, targets[..., None].expand(-1, -1, cards.shape[-1]))
    valid = (
        pending[:, None]
        & obs["action_mask"].bool()
        & (actions[..., 0] == ACTION_KINDS.index("select") + 1)
        & (actions[..., 1] > 0)
        & (actions[..., 1] <= n_cards)
        & (target_cards[..., 0] > 1)
        & (target_cards[..., 0] == actions[..., 2])
        & (target_cards[..., 1] == 3)
        & (target_cards[..., 4] == 1)
        & (target_cards[..., 7] == 1)
        & ((target_cards[..., 6] == 1) | (target_cards[..., 6] == 3))
    )
    return attacker, targets, valid


class BattleContext(nn.Module):
    """Zero-initialized residual on visible target-action embeddings only."""

    def __init__(self, width: int):
        super().__init__()
        self.pair = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, width)
        )
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)

    def forward(self, obs: dict[str, Tensor], cards: Tensor) -> Tensor:
        attacker, targets, valid = battle_context_rows(obs)
        own = cards[torch.arange(cards.shape[0], device=cards.device), attacker]
        other = cards.gather(1, targets[..., None].expand(-1, -1, cards.shape[-1]))
        pair = torch.cat([own[:, None].expand_as(other), other], -1)
        return self.pair(pair).masked_fill(~valid[..., None], 0)
