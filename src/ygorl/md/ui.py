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
YESNO_NO, YESNO_YES = (866, 658), (1186, 658)
PHASE_MENU_CANCEL = (1026, 1035)
LIST_SELECT, LIST_SELECT_TWO_BUTTONS = (1026, 1116), (1180, 1117)

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


def screen_dialog(img: np.ndarray) -> str | None:
    """MD UI dialogs that are not engine input states (measured on the 2054x1184 window)."""
    if L.menu_button(img, *L.PHASE_MENU["end"]):
        return "phase_menu"  # MD's phase menu (left open by an interrupted phase change)
    pos_l, pos_r = box_mean(img, 925, 940, 955, 970), box_mean(img, 1100, 940, 1130, 970)
    above = box_mean(img, 700, 760, 1350, 790)  # the board around the position dialog, a prompt panel's title row
    if pos_l[0] < 30 and pos_l[1] < 60 and pos_r[0] < 30 and pos_r[1] < 70 and above.mean() > 100:
        return "position"
    band, mid = box_mean(img, 620, 520, 1430, 560), box_mean(img, 900, 560, 1150, 600)
    if band.mean() < 100 and mid.mean() < 120:
        return "yesno"
    if box_mean(img, 950, 1100, 1100, 1130).max() < 30:
        return "list"
    banner = box_mean(img, 900, 270, 1150, 300)
    if banner[0] < 60 and banner[2] < 110 and banner.mean() < 80:
        return "banner"
    return None


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
        """Field cards outlined in MD's yellow selection frame (own row first, then the opponent's)."""
        img = self.capture()
        spots = [my_monster_xy(s) for s in range(7)] + [opp_monster_xy(s) for s in range(7)]
        return [xy for xy in spots if yellow_highlight(img, *xy)]

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
                found = [p for p in L.find_icons(self.capture(), *box) if abs(p[0] - x) <= 40]
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
                if icons:
                    return icons
                self.sleep(0.3)
        return []

    def open_hand_card(self, xy: XY, why: str) -> list[XY]:
        return self.open_card(xy, why, hand=True)

    def card_command(self, xy: XY, why_card: str, why: str, pause: float = 1.8) -> bool:
        """Click a field card and then the command icon closest above it; False when no icon appeared."""
        icons = self.open_card(xy, why_card)
        if not icons:
            return False
        self.click(*min(icons, key=lambda p: abs(p[0] - xy[0])), why, pause)
        return True

    def choose_command(self, icons: Sequence[XY], kind: str, order: Sequence[str], name: str = "") -> XY | None:
        """Click the icon of command ``kind`` among the ``icons`` an opened hand card shows, each told by its glyph
        (:func:`icon_kind`): Activate yellow-green, Summon mostly blue, Set less blue. Special Summon has no
        measured glyph yet and falls back to MD's ``order`` (:func:`command_order`) among the unclassified icons.
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
        else:  # spsummon: by position among the icons that are neither Activate nor Set nor Summon
            rest = [p for p, k in zip(icons, kinds, strict=True) if k == ""]
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
        """The summon-position dialog. Only face-up Attack Position is measured."""
        if position != "faceup_attack":
            raise Stuck(f"no measured button for {position} in the position dialog")
        self.click(*POSITION_FACEUP_ATTACK, "face-up attack", 1.5)

    def yesno(self, yes: bool, why: str = "") -> None:
        """MD's yes / no dialog (both buttons measured: Tribute Summon confirmation, "Normal Summon now?")."""
        self.click(*(YESNO_YES if yes else YESNO_NO), why or ("yes" if yes else "no"), 1.5)

    def pick_list_cards(self, indices: Sequence[int]) -> list[XY]:
        """Tick ``indices`` (left to right on screen) in MD's card-list panel and press Select; returns the cards
        found in the list. A list with two buttons keeps the panel open after the first: press its right one."""
        cards = L.list_cards(self.capture())
        if not cards:
            raise Stuck("a card list with no card found")
        bad = [i for i in indices if not 0 <= i < len(cards)]
        if bad or not indices:
            raise Stuck(f"list entries {list(indices)} asked for, {len(cards)} cards in the list")
        for i in indices:
            nth = "first" if i == 0 else f"#{i + 1}"
            self.click(*cards[i], f"{nth} of {len(cards)} cards in the list", 0.8)
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
        for attempt in range(2):  # a first click can just close a card left open
            if L.menu_button(self.capture(), *L.PHASE_MENU["end"]):
                img = self.capture()  # already open (an earlier click): do not toggle it shut
            else:
                self.click(*L.PHASE_BADGE, "phase menu" if not attempt else "phase menu (again)", 1.2)
                img = None
                for _ in range(8):  # the menu fades in over about a second
                    shot = self.capture()
                    if L.menu_button(shot, *L.PHASE_MENU["end"]):
                        img = shot
                        break
                    self.sleep(0.3)
            if img is not None:
                if not L.menu_button(img, *L.PHASE_MENU[which]):
                    raise Stuck(f"the phase menu offers no {which}")
                self.click(*L.PHASE_MENU[which], which, 2.0)
                return
        raise Stuck("the phase menu did not open")

    def continue_draw(self) -> None:
        """Draw Phase: MD sometimes waits for a click to go on (seen with the Graveyard viewer open); the phase
        badge is harmless here."""
        self.click(*L.PHASE_BADGE, "continue the Draw Phase", 1.5)

    def attack(self, attacker_xy: XY, why: str) -> bool:
        """Open an own monster and press its Attack icon; False when MD shows no Attack for it."""
        return self.card_command(attacker_xy, why, "attack")

    def pick_target(self, target_xy: XY, why: str) -> bool:
        """Open the attack target / a highlighted card and press its Confirm icon; False when none appeared."""
        return self.card_command(target_xy, why, "confirm target", 2.0)
