"""Where things are in the Master Duel window (design 10, Solo self-play), as fractions of the window size.

Measured 2026-10-21 in a Solo duel on a 2054x1184 window whose client area (the game's pixels, without Wine's
title bar and border) was 2048x1152 at (3, 29); each point was clicked and the result checked in memory. Positions
below stay in that *reference frame*. :func:`to_client` maps a reference point to the actual client area, and
:func:`reference_image` resamples a capture of any size back into the reference frame, so recognisers and click
positions are written once for every window size. MD's layout is fixed at 16:9; other client aspect ratios are
mapped per axis, which is exact only while the game itself letterboxes to 16:9.

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

import numpy as np

from ygorl.engine import constants as C

REF_W, REF_H = 2054, 1184  # the reference frame: the window the measurements below come from
REF_CLIENT = (3, 29, 2048, 1152)  # its client area: x, y, width, height

HAND_Y, HAND_CENTER_X, HAND_STEP = 1100, 1020, 140
COMMAND_DY = -220  # first command button, relative to the clicked hand card
MY_MZONE_Y, MY_SZONE_Y = 740, 930
ZONE_X = (650, 844, 1038, 1232, 1426)  # columns 1-5 from the advisor's left
PHASE_BADGE = (1580, 560)
PHASE_MENU = {"battle": (1144, 861), "main2": (1379, 861), "end": (1613, 861)}


def menu_button(img: np.ndarray, x: int, y: int, radius: int = 90) -> bool:
    """A phase-menu button is drawn at (x, y): a light (or, for the current phase, yellow-green) ring of radius 90.

    Measured on the open menu: 1.0 of 72 directions hit the ring at r 90 for every available button, 0 off-menu."""
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    px = np.array([img[y + round(radius * np.sin(t)), x + round(radius * np.cos(t))] for t in ang]).astype(int)
    ring = (px.min(axis=1) > 120) | ((px[:, 1] > 180) & (px[:, 2] < 100))
    return bool(ring.mean() > 0.8)


FIELD_COMMAND_DY = -145
PROMPT_CARD = (1026, 934)  # the first card of a response prompt
PROMPT_CANCEL, PROMPT_ACTIVATE = (870, 1117), (1180, 1117)


def to_client(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """A reference-frame point as pixels of a ``width`` x ``height`` client area."""
    cx, cy, cw, ch = REF_CLIENT
    return round((x - cx) * width / cw), round((y - cy) * height / ch)


def reference_image(client: np.ndarray) -> np.ndarray:
    """A client-area capture of any size, resampled (nearest) into the reference frame."""
    cx, cy, cw, ch = REF_CLIENT
    h, w = client.shape[:2]
    if (w, h) == (cw, ch):
        scaled = client
    else:
        ys = np.minimum((np.arange(ch) * h / ch).astype(int), h - 1)
        xs = np.minimum((np.arange(cw) * w / cw).astype(int), w - 1)
        scaled = client[ys][:, xs]
    out = np.zeros((REF_H, REF_W, 3), dtype=np.uint8)
    out[cy : cy + ch, cx : cx + cw] = scaled
    return out


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
        return (0, 0 if card.type & C.TYPE_NORMAL else 1, card.level, cid)
    return (1 if card.type & C.TYPE_SPELL else 2, 0, card.type, cid)


def hand_order(cids: Sequence[int], passwords: Callable[[int], int], cards: Callable[[int], object]) -> list[int]:
    """Indices into the memory-order hand (MD card ids) in on-screen order, left to right.

    Seen in four hands: Normal Monsters, then Effect Monsters, each by level; then spells with Normal before
    Quick-Play, then traps; ascending MD card id within a group (Jack's Knight 5, Dark Magician 7 before King of
    the Swamp 3, an Effect Monster; King's Knight 4 < Slifer 10; Imperial Bower 4 < Berfomet 5; Pot of Avarice 6511
    < Joker's Straight 16480 < Thunderspeed Summon 16482, a Quick-Play)."""
    return sorted(range(len(cids)), key=lambda i: _sort_key(cards(passwords(cids[i])), cids[i]))


__all__ = [
    "REF_CLIENT",
    "command_button",
    "find_icons",
    "new_icons",
    "hand_card",
    "hand_order",
    "my_zone",
    "reference_image",
    "to_client",
]


def find_icons(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Centres of MD's round command icons (Summon / Set / Activate / Attack / Confirm ...) inside a box.

    Each icon is a disc of radius 40-48 reference pixels with a near-black rim at least ~8 pixels wide; the glyph
    inside varies (it fills almost the whole Activate icon). A centre scores the share of 48 directions with a
    dark pixel in the rim band [r-8, r] and light ones at r+6 and r+12, best over the radii (a small circle Hough
    test); local maxima of at least 0.7 are returned, left to right. Dark card art can pass too: compare with a
    capture from before the click (:func:`new_icons`).
    """
    from scipy import ndimage

    box = img[y0:y1, x0:x1].astype(int)
    dark = box.max(axis=2) < 62
    h, w = dark.shape
    reach = 60
    ys, xs = np.mgrid[reach : h - reach : 2, reach : w - reach : 2]
    if ys.size == 0:
        return []
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    best = np.zeros(ys.shape)
    for r in (40, 44, 48):
        score = np.zeros(ys.shape)
        for t in ang:
            si, co = np.sin(t), np.cos(t)
            band = np.zeros(ys.shape, dtype=bool)
            for rr in (r - 8, r - 5, r - 2):
                band |= dark[ys + round(rr * si), xs + round(rr * co)]
            out1 = dark[ys + round((r + 6) * si), xs + round((r + 6) * co)]
            out2 = dark[ys + round((r + 12) * si), xs + round((r + 12) * co)]
            score += band & ~out1 & ~out2
        best = np.maximum(best, score / len(ang))
    peaks = (best >= 0.7) & (best == ndimage.maximum_filter(best, size=21))
    out: list[tuple[int, int]] = []
    for i, j in sorted(zip(*np.nonzero(peaks), strict=True), key=lambda ij: -best[ij]):
        cx, cy = x0 + int(xs[i, j]), y0 + int(ys[i, j])
        if all(abs(cx - qx) > 44 or abs(cy - qy) > 44 for qx, qy in out):
            out.append((cx, cy))
    return sorted(out)


def new_icons(before: np.ndarray, after: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Icons in ``after`` that were not there in ``before`` (the commands a click just opened).

    Shape alone does not separate the icons from dark card art, but the art does not change when a card is
    clicked: a candidate counts only if its disc changed by a mean of at least 25 per channel.
    """
    out = []
    for x, y in find_icons(after, x0, y0, x1, y1):
        a = after[y - 40 : y + 40, x - 40 : x + 40].astype(int)
        b = before[y - 40 : y + 40, x - 40 : x + 40].astype(int)
        if np.abs(a - b).mean() >= 25:
            out.append((x, y))
    return out


TRAY_RGB, CARD_STEP = (37, 66, 143), 158  # MD's card-list tray colour and card pitch (reference pixels)
TRAY_RGBS = (TRAY_RGB, (163, 26, 25))  # blue: own cards; red: the opponent's (a Monster Reborn target in their GY)
LIST_CARD_Y = 905  # the middle of a listed card's art: y 960, near its lower edge, did not always register a tick


def list_cards(img: np.ndarray) -> list[tuple[int, int]]:
    """Centres of the cards shown in MD's card-list panel, left to right.

    The cards sit on flat trays (blue for own cards, red for the opponent's) whose strip just below them (y ~1030-1060) is unbroken; a tray is
    ``158 * n - 20`` wide for n cards (lists of 2, 3, 4 and 6 cards measured), and its cards are spread evenly over
    it. Cards from different places (the field, the Deck: Fusion Materials) sit on separate trays 20 px apart.
    """
    best: list[tuple[int, int]] = []  # (left, width) of the trays on the row that shows the most tray
    for y in range(1030, 1062, 2):
        row = img[y].astype(int)
        tray = np.zeros(len(row), bool)
        for rgb in TRAY_RGBS:
            tray |= (np.abs(row - np.array(rgb)) < 22).all(axis=1)
        padded = np.concatenate([[False], tray, [False]])
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs = [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= CARD_STEP - 40]
        if sum(w for _, w in runs) > sum(w for _, w in best):
            best = runs
    cards = []
    for left, width in best:
        n = round((width + 20) / CARD_STEP)
        first = left + (width - (n - 1) * CARD_STEP) / 2
        cards += [(round(first + i * CARD_STEP), LIST_CARD_Y) for i in range(n)]
    return cards
