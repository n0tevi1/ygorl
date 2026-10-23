"""Master Duel UI actuators (design 10, Solo self-play): "do this on screen" -> clicks + screen checks.

Every click script of the Solo player lives here; the player (:mod:`ygorl.selfplay.loop`) only decides *what* to do.
Nothing in this module chooses between options or knows about the policy: each method takes the choice already made
(which card, which command, which list entries, activate or pass), clicks it through the injected ``mouse``,
checks the screen through the injected ``capture`` where MD's own UI is the only signal, and returns what it saw
(icons found, list cards, ...). Anything it cannot carry out raises :class:`Stuck`.

``mouse`` needs only ``click(x, y)`` in reference-frame coordinates (live: :class:`ygorl.md.input.GatedMouse`,
which re-checks the Solo gate before every event); ``capture()`` returns a reference-frame image
(:func:`ygorl.md.layout.reference_image`); ``sleep`` is injectable so tests run without waiting. Every click is
printed and kept in :attr:`UI.clicks` until :meth:`UI.take_clicks` hands it to the trace.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ygorl.md import layout as L

XY = tuple[int, int]


class Stuck(RuntimeError):
    """The player cannot continue on its own; the record says why."""


class Mouse(Protocol):
    def click(self, x: int, y: int) -> None: ...


# --- fixed positions (reference frame, measured on the 2054x1184 window) ---------------------------------------
POSITION_FACEUP_ATTACK = (940, 956)  # the left button of the summon-position dialog
POSITION_FACEUP_DEFENSE = (1115, 956)  # its right button (the dark disc measured at 1100-1130, 940-970)
YESNO_NO, YESNO_YES = (866, 658), (1186, 658)
PHASE_MENU_CANCEL = (1026, 1035)
MY_DECK = (1625, 985)  # the advisor's Deck pile, bottom right
LIST_SELECT, LIST_SELECT_TWO_BUTTONS = (1026, 1116), (1180, 1117)
LIST_SELECT_TEXT = (963, 1099, 1093, 1139)  # the one-button list's Select label: x0, y0, x1, y1


LIST_SELECT_TWO_TEXT = (1150, 1102, 1215, 1134)  # the two-button list's right (Select) label


def list_card_ticked(img: np.ndarray, card: XY) -> bool:
    """A list card carries MD's tick badge (a blue diamond with a check or an order number) at its top right:
    0.26-0.33 of a 24 px window white or blue on ticked cards, 0.01 without (measured on a 5-card order list)."""
    x, _ = card
    best = 0.0
    for dx in range(20, 60, 4):
        for y in range(860, 920, 4):
            r = img[y - 12 : y + 12, x + dx - 12 : x + dx + 12].reshape(-1, 3).astype(int)
            best = max(best, float(((r.min(axis=1) > 200) | ((r[:, 2] > 150) & (r[:, 0] < 80))).mean()))
    return best > 0.15


def list_select_state(img: np.ndarray, box: tuple[int, int, int, int] = LIST_SELECT_TEXT) -> str:
    """The one-button card list's Select: "enabled" (bright yellow-green label, measured 0.075 of the box),
    "disabled" (dim olive label, nothing ticked yet: 0.05 dim, 0 bright), "" (no such label: another layout)."""
    x0, y0, x1, y1 = box
    r = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
    if ((r[:, 1] > 150) & (r[:, 2] < 90)).mean() > 0.03:
        return "enabled"
    if ((r[:, 1] >= 85) & (r[:, 2] < 40) & (r[:, 1] > r[:, 0])).mean() > 0.02:
        return "disabled"
    return ""


# screen centres of field cards; the opponent's row is mirrored
MY_MZONE_Y, MY_EMZ_Y, OPP_MZONE_Y = 750, 540, 380
EMZ_X = (844, 1232)


def my_monster_xy(slot: int) -> XY:
    return (L.ZONE_X[slot], MY_MZONE_Y) if slot < 5 else (EMZ_X[slot - 5], MY_EMZ_Y)


def opp_monster_xy(slot: int) -> XY:
    return (L.ZONE_X[4 - slot], OPP_MZONE_Y) if slot < 5 else (EMZ_X[6 - slot], MY_EMZ_Y)


# --- recognisers --------------------------------------------------------------------------------------------------
def box_mean(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return img[y0:y1, x0:x1].reshape(-1, 3).mean(0)


RADIO_X = (540, 590)  # the yellow radio rings of MD's "pick one" dialog (e.g. banish 3 or 6 cards), left of the rows
RADIO_ROW_X, RADIO_CONFIRM = 1027, (1026, 840)
RADIO_CONFIRM_TEXT = (995, 826, 1060, 856)  # its 确认 label (measured 1005-1047 x 831-850)


def radio_rows(img: np.ndarray) -> list[int]:
    """The y centres of the rows of MD's radio-button dialog (rings measured 43 px tall, 77 px apart), top down."""
    x0, x1 = RADIO_X
    band = img[300:1000, x0:x1].astype(int)
    ring = ((band[:, :, 0] > 150) & (band[:, :, 1] > 150) & (band[:, :, 2] < 110)).any(axis=1)
    rows, start = [], None
    for y, on in enumerate(np.append(ring, False)):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if 30 <= y - start <= 56:
                rows.append(300 + (start + y) // 2)
            start = None
    return rows


MINIMISED_RESTORE = (1335, 1140)  # the teal ▲ button of a minimised card list / dialog


def minimised(img: np.ndarray) -> bool:
    """A card list or dialog minimised with its ▽ button: only a dark grid button and a teal ▲ (restore) button
    are left at the bottom (measured: grid box 0.81 dark, restore box 0.39-0.58 teal; nothing else above 0.3)."""
    grid = img[1100:1175, 1180:1250].astype(int)
    restore = img[1105:1175, 1290:1380].astype(int)
    dark = (grid.max(axis=2) < 45).mean()
    teal = ((restore[:, :, 1] > restore[:, :, 0] + 25) & (restore[:, :, 1] > 110)).mean()
    return bool(dark > 0.6 and teal > 0.3)


def screen_dialog(img: np.ndarray) -> str | None:
    """MD UI dialogs that are not engine input states (measured on the 2054x1184 window)."""
    if minimised(img):
        return "minimised"
    if L.menu_button(img, *L.PHASE_MENU["end"]):
        return "phase_menu"  # MD's phase menu (left open by an interrupted phase change)
    if len(radio_rows(img)) >= 2 and list_select_state(img, RADIO_CONFIRM_TEXT) != "":
        return "radio"  # its dark panel would otherwise read as a yes / no dialog (rings alone: an animation)
    pos_l, pos_r = box_mean(img, 925, 940, 955, 970), box_mean(img, 1100, 940, 1130, 970)
    above = box_mean(img, 700, 760, 1350, 790)  # the board around the position dialog, a prompt panel's title row
    labels = [float((img[1025:1065, x0 : x0 + 120].min(axis=2) > 200).mean()) for x0 in (880, 1060)]  # 表侧…
    if position_buttons(img) is not None:
        return "position"
    dark = pos_l[0] < 30 and pos_r[0] < 30
    if dark and ((pos_l[1] < 60 and pos_r[1] < 70 and above.mean() > 100) or min(labels) > 0.08):
        # the white labels under the buttons (0.15-0.19 measured) also show on a dark (lava) board, where the right
        # button's shading came out greener (74) than on the first stages
        return "position"
    band, mid = box_mean(img, 620, 520, 1430, 560), box_mean(img, 900, 560, 1150, 600)
    if band.mean() < 100 and mid.mean() < 120 and _lime(img, *YESNO_YES) <= 0.02 and notice_button(img) is not None:
        return "notice"  # one lit Confirm on a dark panel (yes / no has two)
    if band.mean() < 100 and mid.mean() < 120:
        # yes / no has lime labels on both buttons (0.06 of the label box measured); without them the dark band is
        # another panel (the effect choice, told apart in memory) or just a dark / dimmed board
        # two lit buttons, exactly at the No and Yes spots (the button finder: pills with lime labels). Lime zone
        # outlines on a dark board lit both label spots without being buttons; a text-density test on the question
        # missed a short one ("要破坏一张卡吗?", 0.0078)
        from ygorl.md.explore import find_buttons

        lit = [b.centre for b in find_buttons(img) if b.label == "enabled"]
        at = lambda spot: any(abs(x - spot[0]) <= 20 and abs(y - spot[1]) <= 20 for x, y in lit)  # noqa: E731
        return "yesno" if at(YESNO_NO) and at(YESNO_YES) else "panel"
    if box_mean(img, 950, 1100, 1100, 1130).max() < 30 and L.list_cards(img):
        return "list"  # the dark strip alone also shows during a turn change: the list's blue tray must be up too
    banner = box_mean(img, 900, 270, 1150, 300)
    text = (img[260:330, 470:1590].min(axis=2) > 200).mean()
    # MD's banner is a navy band (32 68 84 measured) with white text: a dark field (the lava stage, 53 44 48) is not
    if banner[0] < 60 and banner[2] < 110 and banner.mean() < 80 and banner[2] > banner[0] + 25 and text > 0.0015:
        return "banner"
    return None


def position_buttons(img: np.ndarray) -> tuple[XY, XY] | None:
    """The summon-position dialog's two buttons wherever MD puts them (centred under the board, or next to the zone
    of a Special Summon from the hand): a blue upright-card glyph (face-up Attack, ~54x77) and, 140-200 px to its
    right on the same row, a blue sideways-card glyph (face-up Defense, ~82x40)."""
    from scipy import ndimage

    y0, x0 = 300, 300
    box = img[y0:1100, x0:1750].astype(int)
    blue = (box[:, :, 2] > 150) & (box[:, :, 0] < 110) & (box[:, :, 2] > box[:, :, 0] + 60)
    labels, _ = ndimage.label(blue)
    upright, sideways = [], []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if (labels[sl] == i).mean() < 0.4:
            continue
        centre = (x0 + (sl[1].start + sl[1].stop) // 2, y0 + (sl[0].start + sl[0].stop) // 2)
        if 35 <= w <= 75 and 55 <= h <= 100 and h > w * 1.2:
            upright.append(centre)
        elif 60 <= w <= 105 and 28 <= h <= 60 and w > h * 1.5:
            sideways.append(centre)
    for a in upright:
        for d in sideways:
            if 140 <= d[0] - a[0] <= 200 and abs(d[1] - a[1]) <= 15:
                return a, d
    return None


def notice_button(img: np.ndarray) -> XY | None:
    """MD's notice ("no card can receive this effect", ...): a centred panel whose only button is a lit Confirm
    and nothing else to click (buttons, tabs, radio rows, list cards). Its position depends on the text."""
    from ygorl.md.explore import find_buttons, find_list_cards, find_radios, find_tabs

    found = find_buttons(img) + find_tabs(img) + find_radios(img) + find_list_cards(img)  # not screen_dialog-based
    if (
        len(found) == 1
        and found[0].kind == "button"
        and found[0].label == "enabled"
        and 600 < found[0].centre[0] < 1450
    ):
        return found[0].centre  # nothing else to pick: not the effect choice (tabs + OK) nor a list
    return None


def zone_outlined(img: np.ndarray, x: int, y: int, w: int = 190, h: int = 170, t: int = 14) -> bool:
    """A free zone MD offers for a placement: outlined in a dim yellow-green (0.42-0.52 of the ring band measured on
    the S/T row of the lava stage, at most 0.15 elsewhere, a card's own frame included)."""
    x0, x1, y0, y1 = x - w // 2, x + w // 2, y - h // 2, y + h // 2
    ring = np.concatenate([img[y0 : y0 + t, x0:x1].reshape(-1, 3), img[y1 - t : y1, x0:x1].reshape(-1, 3),
                           img[y0:y1, x0 : x0 + t].reshape(-1, 3), img[y0:y1, x1 - t : x1].reshape(-1, 3)]).astype(int)  # fmt: skip
    return bool(((ring[:, 0] > 100) & (ring[:, 1] > 110) & (ring[:, 2] < 90)).mean() > 0.3)


def _lime(img: np.ndarray, x: int, y: int) -> float:
    """The share of lime (enabled button label) pixels in a 120x28 box around (x, y)."""
    r = img[y - 14 : y + 14, x - 60 : x + 60].reshape(-1, 3).astype(int)
    return float(((r[:, 1] > 150) & (r[:, 2] < 90)).mean())


def yellow_highlight(img: np.ndarray, x: int, y: int, w: int = 150, h: int = 180, t: int = 12) -> bool:
    """A card outlined in MD's (thin, animated) yellow selection frame around (x, y): measured 0.15 of the ring band
    yellow on a highlighted monster, at most 0.04 elsewhere."""
    x0, x1, y0, y1 = x - w // 2, x + w // 2, y - h // 2, y + h // 2
    ring = np.concatenate([img[y0 : y0 + t, x0:x1].reshape(-1, 3), img[y1 - t : y1, x0:x1].reshape(-1, 3),
                           img[y0:y1, x0 : x0 + t].reshape(-1, 3), img[y0:y1, x1 - t : x1].reshape(-1, 3)])  # fmt: skip
    yellow = (ring[:, 0] > 190) & (ring[:, 1] > 170) & (ring[:, 2] < 110)
    return bool(yellow.mean() > 0.08)


def is_activate_icon(img: np.ndarray, x: int, y: int, r: int = 30) -> bool:
    """MD's Activate icon has a yellow-green glyph (0.65 of the disc measured); Summon / Set glyphs are blue (0.25-0.52
    blue, no yellow-green)."""
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disc = img[y - r : y + r + 1, x - r : x + r + 1][(yy**2 + xx**2) <= r * r].astype(int)
    yellow_green = (disc[:, 0] > 140) & (disc[:, 1] > 170) & (disc[:, 2] < 120)
    return bool(yellow_green.mean() > 0.2)


def icon_kind(img: np.ndarray, x: int, y: int, r: int = 30) -> str:
    """A command icon's kind from its glyph: "activate" (yellow-green), "summon" (blue on 0.51-0.52 of the disc),
    "set" (blue on 0.23-0.36: monster Set 0.23-0.26, Spell / Trap Set 0.32-0.36), "" else (measured on 8 icons)."""
    if is_activate_icon(img, x, y, r):
        return "activate"
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disc = img[y - r : y + r + 1, x - r : x + r + 1][(yy**2 + xx**2) <= r * r].astype(int)
    blue = float(((disc[:, 2] > 180) & (disc[:, 0] < 120)).mean())
    return "summon" if blue > 0.43 else "set" if blue > 0.15 else ""


def is_command_icon(img: np.ndarray, x: int, y: int, r: int = 30) -> bool:
    """A real command icon has a blue (Summon / Set: 0.25-0.52 of the disc) or yellow-green (Activate) glyph; a
    face-down card's back in the Spell & Trap row (which the hand's icon band overlaps) has neither."""
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disc = img[y - r : y + r + 1, x - r : x + r + 1][(yy**2 + xx**2) <= r * r].astype(int)
    blue = (disc[:, 2] > 180) & (disc[:, 0] < 120)
    return bool(blue.mean() > 0.1) or is_activate_icon(img, x, y, r)


def command_order(is_monster: bool, options: Iterable[str]) -> list[str]:
    """MD's left-to-right icon order above a hand card, restricted to the commands in ``options``.

    Summon, Set for monsters; Activate, Set for spells and traps. Where a monster's hand effect (Activate) sits
    among Summon / Set has only been seen alone, so its place in a longer row is a guess."""
    kinds = ("summon", "spsummon", "activate", "mset") if is_monster else ("activate", "sset")
    options = set(options)
    return [k for k in kinds if k in options]


@dataclass
class UI:
    """Click scripts over an injectable mouse and capture. Holds no game state besides the click record."""

    mouse: Mouse
    capture: Callable[[], np.ndarray]
    sleep: Callable[[float], None] = time.sleep
    echo: bool = True  # print every click (the live player's console log)
    clicks: list[dict] = field(default_factory=list)
    evidence: list[np.ndarray] = field(default_factory=list)  # frames a failed step saw (taken by the stuck record)

    # --- primitives -------------------------------------------------------------------------------------------
    def click(self, x: int, y: int, why: str, pause: float = 1.0) -> None:
        self.clicks.append({"t": round(time.time(), 2), "xy": [int(x), int(y)], "why": why})
        if self.echo:
            print(f"  click ({x}, {y}) {why}", flush=True)
        self.mouse.click(x, y)
        self.sleep(pause)

    def take_clicks(self) -> list[dict]:
        """The clicks since the last call (for the trace)."""
        out, self.clicks = self.clicks, []
        return out

    def dialog(self) -> str | None:
        """Which of MD's own dialogs is on screen now (:func:`screen_dialog`)."""
        return screen_dialog(self.capture())

    def highlighted(self) -> list[XY]:
        """Field cards outlined in MD's yellow selection frame (own row first, then the opponent's); a card in
        Defense Position lies sideways, so its frame is wide and short (200x130)."""
        from ygorl.md.explore import find_highlights

        img = self.capture()
        spots = [my_monster_xy(s) for s in range(7)] + [opp_monster_xy(s) for s in range(7)]
        lit = [xy for xy in spots if yellow_highlight(img, *xy) or yellow_highlight(img, *xy, w=200, h=130)]
        # frames found anywhere (MD draws the Extra Monster Zones and a tilted board a little off the measured
        # spots: a target at (1214, 556) for the spot (1232, 540)); the click goes to the frame itself
        for frame in (a.centre for a in find_highlights(img)):
            near = min(spots, key=lambda p: abs(p[0] - frame[0]) + abs(p[1] - frame[1]))
            if abs(near[0] - frame[0]) + abs(near[1] - frame[1]) < 80 and near not in lit:
                lit.append(frame)
        return lit

    # --- cards and commands -----------------------------------------------------------------------------------
    def wait_still(
        self, box: tuple[int, int, int, int] = (300, 1000, 1760, 1180), timeout: float = 8.0, tolerance: float = 4.0
    ) -> bool:
        """Wait until ``box`` (default: the hand) stops changing for three captures in a row: MD keeps animating
        after the engine accepts input (a drawn card still flying to the hand, while memory already counts it)."""
        x0, y0, x1, y1 = box
        last, still = None, 0
        for _ in range(max(1, round(timeout / 0.4))):
            now = self.capture()[y0:y1, x0:x1].astype(np.int16)
            if last is not None and float(np.abs(now - last).mean()) < tolerance:
                still += 1
                if still >= 3:
                    return True
            else:
                still = 0
            last = now
            self.sleep(0.4)
        return False

    def open_card(self, xy: XY, why: str, *, hand: bool = False) -> list[XY]:
        """Click a card and return the command icons that click opened: a row above a hand card, an icon above a
        field card (compared with a capture from just before, so card art is never taken for an icon)."""
        x, y = xy
        box = ((max(x - 380, 0), 790, min(x + 380, L.REF_W), 975) if hand
               else (max(x - 130, 0), max(y - 260, 0), min(x + 130, L.REF_W), y - 60))  # fmt: skip
        if not hand:
            before = self.capture()
            self.click(x, y, why, 1.2)
            icons = L.new_icons(before, self.capture(), *box)
            if icons:
                return icons
            # MD's Confirm icon blinks (about 2 s on, 2 s off) and may already be up from an earlier click:
            # poll for one straight above the card
            for _ in range(16):
                img = self.capture()
                found = [p for p in L.find_icons(img, *box) or L.blue_icons(img, *box) if abs(p[0] - x) <= 40]
                if found:
                    return found
                self.sleep(0.3)
            return []
        for attempt in range(2):  # the first click can land while an animation still covers the hand
            before = self.capture()
            self.click(x, y, why if not attempt else f"{why} (again)", 0.6)
            # the icons can take a few seconds (a King's Knight Summon icon came only after the check); a card left
            # open by an earlier stop keeps its icons, so icons already there count too (only command icons sit
            # above y 940 in the hand band: the lifted card's art starts lower)
            for _ in range(10):
                after = self.capture()
                icons = L.new_icons(before, after, *box) or [p for p in L.find_icons(after, *box) if p[1] < 940]
                icons = [p for p in icons if is_command_icon(after, *p)]
                # the circle test misses some icons (a Set icon next to Activate, The True Name): add the blue
                # glyphs found by colour, wherever no icon is known yet
                extra = [p for p in L.blue_icons(after, *box) if p[1] < 940 and is_command_icon(after, *p)
                         and all(abs(p[0] - q[0]) > 60 for q in icons)]  # fmt: skip
                icons = sorted(icons + extra)
                if icons:
                    return icons
                self.sleep(0.3)
        return []

    def open_hand_card(self, xy: XY, why: str) -> list[XY]:
        return self.open_card(xy, why, hand=True)

    def card_command(self, xy: XY, why_card: str, why: str, pause: float = 1.8, kind: str | None = None) -> bool:
        """Click a field card and then the command icon closest above it; False when no icon appeared. With
        ``kind`` ("activate") only an icon of that glyph (:func:`icon_kind`) counts: a monster can show Activate next
        to Change Position, and the nearest one was once the wrong one."""
        icons = self.open_card(xy, why_card)
        if kind is not None and icons:
            img = self.capture()
            icons = [p for p in icons if icon_kind(img, *p) == kind]
        if not icons:
            return False
        self.click(*min(icons, key=lambda p: abs(p[0] - xy[0])), why, pause)
        return True

    def choose_command(self, icons: Sequence[XY], kind: str, order: Sequence[str], name: str = "") -> XY | None:
        """Click the icon of command ``kind`` among the ``icons`` an opened hand card shows, each told by its glyph
        (:func:`icon_kind`): Activate yellow-green, Summon mostly blue, Set less blue. Special Summon has no
        measured glyph: it is the icon that matches none of the card's other legal commands (``order``).
        None (nothing clicked) when the card does not show the command asked for: MD refuses it."""
        if kind not in order:
            raise Stuck(f"{kind} not among the buttons {list(order)}")
        if not icons:
            return None
        img = self.capture()
        kinds = [icon_kind(img, *p) for p in icons]
        want = {"activate": "activate", "summon": "summon", "mset": "set", "sset": "set"}.get(kind)
        hits = [p for p, k in zip(icons, kinds, strict=True) if k == want] if want else []
        if want and hits:
            target = hits[0]
        elif want:
            return None
        else:  # spsummon: the icon that is none of the card's other legal commands
            rest = [p for p, k in zip(icons, kinds, strict=True) if k == ""]
            if len(rest) != 1:
                # its glyph can read as another kind (Apprentice Illusion Magician's came out "activate"): an icon
                # whose look matches no other command the card has is the Special Summon
                others = {{"mset": "set"}.get(o, o) for o in order if o != kind}
                rest = [p for p, k in zip(icons, kinds, strict=True) if k not in others]
            if len(rest) != 1:
                raise Stuck(f"cannot tell the {kind} icon of {name or 'the card'} among {kinds}")
            target = rest[0]
        self.click(*target, kind, 1.5)
        return target

    def pick_zone(self, location: str, sequence: int, why: str) -> XY:
        """One of the advisor's own Main Monster ("mzone") / Spell & Trap ("szone") zones, 0-4."""
        xy = L.my_zone(location, sequence)
        self.click(*xy, why, 2.5)
        return xy

    def pick_position(self, position: str = "faceup_attack") -> None:
        """The summon-position dialog: face-up Attack (left button) or face-up Defense (right button)."""
        found = position_buttons(self.capture())
        attack, defense = found if found is not None else (POSITION_FACEUP_ATTACK, POSITION_FACEUP_DEFENSE)
        buttons = {"faceup_attack": attack, "faceup_defense": defense}
        if position not in buttons:
            raise Stuck(f"no measured button for {position} in the position dialog")
        self.click(*buttons[position], position.replace("_", " ").replace("faceup", "face-up"), 1.5)

    def yesno(self, yes: bool, why: str = "") -> None:
        """MD's yes / no dialog (both buttons measured: Tribute Summon confirmation, "Normal Summon now?")."""
        self.click(*(YESNO_YES if yes else YESNO_NO), why or ("yes" if yes else "no"), 1.5)

    def pick_radio(self, index: int = 0) -> int:
        """MD's radio-button dialog: pick row ``index`` (top down) and Confirm; returns the number of rows."""
        rows = radio_rows(self.capture())
        if not 0 <= index < len(rows):
            raise Stuck(f"radio row {index} asked for, {len(rows)} rows shown")
        self.click(RADIO_ROW_X, rows[index], f"option {index + 1} of {len(rows)}", 0.8)
        self.click(*RADIO_CONFIRM, "confirm", 1.5)
        return len(rows)

    def pick_list_set(self, need: int, order: Sequence[int] | None = None, tries: int = 20) -> tuple[int, ...]:
        """A list that wants ``need`` cards ticked before Select (Fusion Materials): tick a combination of the cards
        (in ``order`` of preference, default left to right), wait for Select to light up and press it; a combination
        MD does not accept (Select stays dark) is unticked and the next one tried. Returns the combination."""
        from itertools import combinations, islice

        img = self.capture()
        cards = L.list_cards(img)
        if len(cards) < need:
            raise Stuck(f"a list asking for {need} cards shows {len(cards)}")
        two = list_select_state(img, LIST_SELECT_TWO_TEXT) != ""
        box, button = (LIST_SELECT_TWO_TEXT, LIST_SELECT_TWO_BUTTONS) if two else (LIST_SELECT_TEXT, LIST_SELECT)
        order = list(order) if order is not None else list(range(len(cards)))
        for combo in islice(combinations(order, need), tries):
            for i in combo:
                # one click per card: whether a card is ticked cannot be read reliably (blue card art passes for
                # the badge), and a second click takes the tick away
                self.click(*cards[i], f"#{i + 1} of {len(cards)} cards in the list", 1.0)
            for _ in range(12):  # MD can take a few seconds to light Select after the last tick
                if list_select_state(self.capture(), box) == "enabled":
                    self.click(*button, f"select {need} cards", 1.5)
                    self._list_open(4.0)
                    return combo
                self.sleep(0.3)
            for i in combo:  # MD does not take this set: untick it
                self.click(*cards[i], f"untick #{i + 1}", 0.6)
        raise Stuck(f"no set of {need} of the {len(cards)} list cards lit Select")

    def pick_list_cards(self, indices: Sequence[int], ticked: bool = False) -> list[XY]:
        """Tick ``indices`` (left to right on screen) in MD's card-list panel and press Select; returns the cards
        found in the list. A list with two buttons keeps the panel open after the first: press its right one.
        ``ticked``: the caller already ticked them (and checked them on the information panel)."""
        cards = L.list_cards(self.capture())
        if not cards:
            raise Stuck("a card list with no card found")
        bad = [i for i in indices if not 0 <= i < len(cards)]
        if bad or not indices:
            raise Stuck(f"list entries {list(indices)} asked for, {len(cards)} cards in the list")
        for again in ("", " (again: the tick did not register)"):
            for i in [] if ticked and not again else indices:
                nth = "first" if i == 0 else f"#{i + 1}"
                self.click(*cards[i], f"{nth} of {len(cards)} cards in the list{again}", 0.8)
            # Select lights up once MD has the tick; pressing it before then does nothing, and the next tick of the
            # same card would undo this one
            for _ in range(8):
                if list_select_state(self.capture()) != "disabled":
                    break
                self.sleep(0.3)
            else:
                continue
            break
        self.click(*LIST_SELECT, "select", 1.0)
        if self._list_open(4.0):  # MD can take seconds to act on Select; clicking again would undo the tick
            self.click(*LIST_SELECT_TWO_BUTTONS, "select (two-button list)", 1.5)
            self._list_open(3.0)
        return cards

    def _list_open(self, seconds: float) -> bool:
        """Whether the card-list panel is still up after waiting up to ``seconds`` for it to close."""
        for _ in range(max(1, round(seconds / 0.4))):
            if screen_dialog(self.capture()) != "list":
                return False
            self.sleep(0.4)
        return True

    def respond_prompt(self, activate: bool) -> None:
        """A chain / open-timing prompt: tick the first offered card and Activate, or Cancel."""
        if activate:
            self.click(*L.PROMPT_CARD, "tick the offered card", 0.8)
            self.click(*L.PROMPT_ACTIVATE, "activate", 2.0)
        else:
            self.click(*L.PROMPT_CANCEL, "pass", 1.5)

    # --- phases and battle ------------------------------------------------------------------------------------
    def phase(self, which: str) -> None:
        """Open the phase menu and pick ``which`` (battle / main2 / end). The menu must be seen open first: its End
        button sits where the Graveyard pile is, so a click on a menu that did not open browses the Graveyard."""
        self.evidence.clear()
        for attempt in range(3):  # a click can just close a card left open, or the drawn card still shown big
            if L.menu_button(self.capture(), *L.PHASE_MENU["end"]):
                img = self.capture()  # already open (an earlier click): do not toggle it shut
            else:
                self.click(*L.PHASE_BADGE, "phase menu" if not attempt else f"phase menu (attempt {attempt + 1})", 1.2)
                img = None
                for poll in range(16):  # the menu fades in over about a second, more when MD is busy
                    shot = self.capture()
                    if L.menu_button(shot, *L.PHASE_MENU["end"]):
                        img = shot
                        break
                    if poll in (2, 8, 15):
                        self.evidence.append(shot)  # what the screen showed instead, for the stuck record
                    self.sleep(0.3)
            if img is not None:
                if not L.menu_button(img, *L.PHASE_MENU[which]):
                    raise Stuck(f"the phase menu offers no {which}")
                self.click(*L.PHASE_MENU[which], which, 2.0)
                return
        raise Stuck("the phase menu did not open")  # self.evidence: frames seen while waiting for it

    def continue_draw(self, attempt: int = 0) -> None:
        """Draw Phase: MD sometimes waits for a click to go on. The phase badge did it once (Graveyard viewer open);
        when the badge shows only the turn, a click on the Deck did; once only a click on an empty spot of the board
        did (the drawn card shown big). Cycle through the three."""
        spot, why = [(L.PHASE_BADGE, "continue the Draw Phase"), (MY_DECK, "continue the Draw Phase: the Deck"),
                     ((1300, 640), "continue the Draw Phase: an empty spot")][attempt % 3]  # fmt: skip
        self.click(*spot, why, 1.5)

    def attack(self, attacker_xy: XY, why: str) -> bool:
        """Open an own monster and press its Attack icon; False when MD shows no Attack for it."""
        return self.card_command(attacker_xy, why, "attack")

    def pick_target(self, target_xy: XY, why: str) -> bool:
        """Open the attack target / a highlighted card and press its Confirm icon; False when none appeared."""
        return self.card_command(target_xy, why, "confirm target", 2.0)
