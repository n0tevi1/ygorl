"""Where things are in the Master Duel window (design 10, Solo self-play), as fractions of the window size.

Measured 2026-10-21 on a 2054x1184 window (Wine frame included) in a Solo duel; each point was clicked and the
result checked in memory. Fractions keep the layout usable when the window is resized at the same aspect ratio.

- Hand: centred, one card every 140 px; MD shows it sorted by kind (monsters, spells, traps), in memory order
  within a kind (:func:`hand_order`).
- Clicking a hand card lifts it and shows its commands as round buttons above it (Set for a Trap).
- Setting a Spell/Trap asks for a zone: the free zones are highlighted, CANCEL sits left of the phase badge.
- Zones are in the owner's frame (M1 = the owner's leftmost); the opponent's rows are mirrored on screen.
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


def _kind(card) -> int:
    if card is None or card.type & C.TYPE_MONSTER:
        return 0
    return 1 if card.type & C.TYPE_SPELL else 2


def hand_order(passwords: Sequence[int], cards: Callable[[int], object]) -> list[int]:
    """Indices into the memory-order hand ``passwords`` in on-screen order (left to right)."""
    return sorted(range(len(passwords)), key=lambda i: (_kind(cards(passwords[i])), i))


__all__ = ["command_button", "hand_card", "hand_order", "my_zone", "scale"]
