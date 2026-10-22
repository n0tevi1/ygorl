"""Where things are in the Master Duel window (design 10, Solo self-play), as fractions of the window size.

Measured 2026-10-21 on a 2054x1184 window (Wine frame included) in a Solo duel; each point was clicked and the
result checked in memory. Fractions keep the layout usable when the window is resized at the same aspect ratio.

- Hand: centred, one card every 140 px; MD shows it sorted (:func:`hand_order`), not in memory order.
- Clicking a hand card lifts it and shows its commands as round buttons above it (Set for a Trap).
- Setting a Spell/Trap asks for a zone: the free zones are highlighted, CANCEL sits left of the phase badge.
- Zones are in the owner's frame (M1 = the owner's leftmost); the opponent's rows are mirrored on screen.
- Clicking a monster on the field shows its command (攻击 Attack in the Battle Phase) about 145 px above it.
- The phase badge opens a menu: Battle, Main2, End (greyed when not available).
- A response prompt on the opponent's turn ("对手宣告了战斗。要发动卡的效果吗？") lists the cards that can respond:
  click a card to tick it, then 发动效果 (Activate); 取消 (Cancel) declines.
- Memory runs ahead of the animations: wait for the input state (``S + 0x3cd0`` mode, ``+0x3cdc`` accepting,
  ``+0x3d00`` the player asked) before the next click. Seen: mode 3 = the turn player's command (Battle Phase),
  mode 4 = a response prompt, accepting 0 while animations or the opponent run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ygorl.engine import constants as C

REF_W, REF_H = 2054, 1184  # the window the measurements below come from

HAND_Y, HAND_CENTER_X, HAND_STEP = 1100, 1020, 140
COMMAND_DY = -220  # first command button, relative to the clicked hand card
MY_MZONE_Y, MY_SZONE_Y = 740, 930
ZONE_X = (650, 844, 1038, 1232, 1426)  # columns 1-5 from the advisor's left
PHASE_BADGE = (1580, 560)
PHASE_MENU = {"battle": (1145, 855), "main2": (1380, 855), "end": (1615, 855)}
FIELD_COMMAND_DY = -145
PROMPT_CARD = (1026, 934)  # the first card of a response prompt
PROMPT_CANCEL, PROMPT_ACTIVATE = (870, 1117), (1180, 1117)


def scale(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return round(point[0] * width / REF_W), round(point[1] * height / REF_H)


def hand_card(index: int, count: int) -> tuple[int, int]:
    """Screen position of the ``index``-th card from the left of a ``count``-card hand."""
    return round(HAND_CENTER_X + (index - (count - 1) / 2) * HAND_STEP), HAND_Y


def command_button(card: tuple[int, int]) -> tuple[int, int]:
    return card[0], card[1] + COMMAND_DY


def my_zone(location: str, sequence: int) -> tuple[int, int]:
    """The advisor's Main Monster Zone (sequence 0-4) or Spell & Trap Zone (0-4)."""
    if not 0 <= sequence < 5:
        raise ValueError(f"no measured position for {location} {sequence}")
    return ZONE_X[sequence], MY_MZONE_Y if location == "mzone" else MY_SZONE_Y


def _sort_key(card, cid: int) -> tuple:
    if card is None:
        return (3, 0, 0, cid)
    if card.type & C.TYPE_MONSTER:
        return (0, card.level, 0, cid)
    return (1 if card.type & C.TYPE_SPELL else 2, 0, card.type, cid)


def hand_order(cids: Sequence[int], passwords: Callable[[int], int], cards: Callable[[int], object]) -> list[int]:
    """Indices into the memory-order hand (MD card ids) in on-screen order, left to right.

    Seen in three hands: monsters by level, then spells with Normal before Quick-Play, then traps; ascending MD
    card id within a group (King's Knight 4 < Slifer 10; Imperial Bower 4 < Berfomet 5; Pot of Avarice 6511 <
    Joker's Straight 16480 < Thunderspeed Summon 16482, a Quick-Play)."""
    return sorted(range(len(cids)), key=lambda i: _sort_key(cards(passwords(cids[i])), cids[i]))


__all__ = ["command_button", "hand_card", "hand_order", "my_zone", "scale"]
