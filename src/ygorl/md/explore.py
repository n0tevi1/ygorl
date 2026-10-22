"""Learning how to operate Master Duel's dialogs (design 10, 边角情况由 player 自己学会): layer 2, "how to click".

The decision layer says *which* option it wants (``choice``); this module works out *how* to get it done on a dialog
it may never have seen, without per-dialog click scripts:

- :func:`find_affordances`: the clickable elements (可点元素) found in a screenshot by their look, not by fixed
  coordinates: buttons (a pure black pill with a bright label), radio rows (yellow rings), the cards of a card list
  (blue tray), effect tabs, yellow-framed cards, the two cards of the position dialog, and (on demand) command icons.
  Danger zones (the settings gear that leads to Surrender, Auto / info / log, the duellists' portraits) never yield
  one, and the left card-info panel is ignored.
- :func:`signature`: the screen signature (签名): what memory says MD is asking (input mode, accepting, dialog type,
  list type / size, number of choices) plus a coarse visual fingerprint of the dialog (its class and which elements
  are where). Selection state (a ticked card, a lit button) is deliberately left out, so ticking an option does not
  change the signature.
- :class:`Playbook`: the operation manual (操作手册), ``out/mdplay/playbook.json``: per signature, recipes (操作序列)
  made of relative steps ("pick radio row #choice, press the lit button") with success / failure counts; seeded with
  the scripts :mod:`ygorl.md.ui` hard-codes.
- :class:`Explorer`: runs the playbook's recipe for the current signature, or explores candidate recipes built from
  the affordances (option #choice, then each confirm-like button; single buttons ...) until the caller's
  ``progress()`` says MD moved on; undoes a tick that did not help, records the outcome, never clicks a danger zone.

Everything is offline-testable: the explorer only needs a :class:`ygorl.md.ui.UI`-like object (``click(x, y, why,
pause)``, ``capture()``, ``sleep(s)``) and a ``progress`` callable.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ygorl.md import layout as L
from ygorl.md import ui as U

XY = tuple[int, int]
Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (reference frame, exclusive ends)

# --- where never to click, and what to ignore ------------------------------------------------------------------
DANGER_ZONES: dict[str, Box] = {
    "settings": (0, 0, 175, 175),  # the gear (measured 30-105 x 55-130): its menu holds Surrender
    "duel_buttons": (1600, 1050, L.REF_W, L.REF_H),  # Auto, card info, duel log, notes (1625-2025 x 1075-1160)
    "my_portrait": (0, 1030, 460, L.REF_H),  # own portrait / name / LP: opens the profile and emotes
    "opp_portrait": (1600, 0, L.REF_W, 120),  # the opponent's portrait / name / LP
}
INFO_PANEL: Box = (0, 140, 520, 960)  # the left card-info panel: shown text, nothing to operate


def in_box(xy: XY, box: Box) -> bool:
    return box[0] <= xy[0] < box[2] and box[1] <= xy[1] < box[3]


def is_dangerous(xy: XY) -> bool:
    """Whether a click at ``xy`` would land in a danger zone (or outside the frame)."""
    x, y = xy
    if not (0 <= x < L.REF_W and 0 <= y < L.REF_H):
        return True
    return any(in_box(xy, b) for b in DANGER_ZONES.values())


# --- affordances --------------------------------------------------------------------------------------------------
KINDS = ("button", "radio", "tab", "list_card", "position", "highlight", "icon")
OPTION_KINDS = ("radio", "tab", "list_card", "position", "highlight")  # the "选项类" elements, in trying order


@dataclass(frozen=True)
class Affordance:
    """A clickable element. ``index`` counts within its kind, left to right (buttons, tabs, list cards) or top down
    (radio rows); ``label`` is a button's label colour ("enabled" lime, "disabled" dim olive, "white"), a tab's
    "selected" / "", else ""."""

    kind: str
    centre: XY
    bbox: Box
    index: int = 0
    label: str = ""

    def to_json(self) -> dict:
        return {"kind": self.kind, "xy": list(self.centre), "index": self.index, "label": self.label}


BLACK_MAX = 10  # a button's pill is pure black (0, 0, 0 measured); panels and card art are not
BUTTON_W, BUTTON_H = (140, 480), (36, 90)  # pills measured ~285 x 62


def _components(mask: np.ndarray):
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    return lab, ndimage.find_objects(lab), n


def label_class(img: np.ndarray, box: Box) -> str:
    """The colour class of a button label inside ``box``: "enabled" (lime ~(203, 255, 0)), "disabled" (dim olive
    ~(82, 102, 0)), "white", or "" (no label)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = img[y0 + h // 5 : y1 - h // 5, x0 + w // 6 : x1 - w // 6].reshape(-1, 3).astype(int)
    if r.size == 0:
        return ""
    lime = (r[:, 1] > 170) & (r[:, 0] > 100) & (r[:, 2] < 90)
    olive = (r[:, 1] >= 70) & (r[:, 1] <= 170) & (r[:, 2] < 50) & (r[:, 1] > r[:, 0] + 10)
    white = r.min(axis=1) > 170
    counts = {"enabled": lime.mean(), "disabled": olive.mean(), "white": white.mean()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0.004 else ""


def find_buttons(img: np.ndarray) -> list[Affordance]:
    """MD's buttons: pure black pills (a grey rim, a lime accent in one corner) holding a short label; left to right
    (then top down)."""
    from scipy import ndimage

    black = img.max(axis=2) <= BLACK_MAX
    lab, objs, _ = _components(black)
    found = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        w, h = xs.stop - xs.start, ys.stop - ys.start
        if not (BUTTON_W[0] <= w <= BUTTON_W[1] and BUTTON_H[0] <= h <= BUTTON_H[1]):
            continue
        filled = ndimage.binary_fill_holes(lab[sl] == i)
        if filled.mean() < 0.8:
            continue
        box = (xs.start, ys.start, xs.stop, ys.stop)
        centre = ((xs.start + xs.stop) // 2, (ys.start + ys.stop) // 2)
        if is_dangerous(centre) or in_box(centre, INFO_PANEL):
            continue
        label = label_class(img, box)
        if not label:
            continue  # an empty black box (a dark card art, a closed slot) is not a button
        found.append((centre, box, label))
    found.sort(key=lambda f: (round(f[0][1] / 40), f[0][0]))
    return [Affordance("button", c, b, i, lb) for i, (c, b, lb) in enumerate(found)]


def find_radios(img: np.ndarray) -> list[Affordance]:
    rows = U.radio_rows(img)
    if len(rows) < 2:
        return []
    return [
        Affordance("radio", (U.RADIO_ROW_X, y), (U.RADIO_X[0], y - 22, 1510, y + 22), i) for i, y in enumerate(rows)
    ]


TAB_FILL = (131, 173, 28)  # the selected effect tab's lime fill; the others are flat dark teal (23, 52, 70)


def _lime_runs(row: np.ndarray, least: int) -> list[tuple[int, int]]:
    r = row.astype(int)
    lime = (r[:, 1] > 140) & (r[:, 2] < 60) & (r[:, 1] > r[:, 0] + 20)
    padded = np.concatenate([[False], lime, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= least]


def find_tabs(img: np.ndarray) -> list[Affordance]:
    """The tabs of MD's effect-choice dialog, left to right: the selected one is a lime-filled parallelogram
    (measured 320 x 35), the others flat dark teal with a lime underline (330 px) on the selected tab's baseline."""
    fill = (np.abs(img.astype(int) - np.array(TAB_FILL)) < 24).all(axis=2)
    _, objs, _ = _components(fill)
    for sl in objs:
        if sl is None:
            continue
        ys, xs = sl
        w, h = xs.stop - xs.start, ys.stop - ys.start
        if not (120 <= w <= 700 and 25 <= h <= 60) or fill[sl].mean() < 0.6:
            continue
        spans = [(xs.start, xs.stop)]
        for y in range(ys.stop - 2, min(ys.stop + 8, img.shape[0])):
            for a, b in _lime_runs(img[y], 120):
                if all(min(b, d) - max(a, c) < 0.5 * (b - a) for c, d in spans):  # not one already seen
                    spans.append((a, b))
        spans.sort()
        # the tabs are about the same width and tile the bar: drop runs of another size (a long divider line)
        spans = [s for s in spans if abs((s[1] - s[0]) - w) <= 0.35 * w]
        if len(spans) < 2:
            continue
        y_mid = (ys.start + ys.stop) // 2
        return [Affordance("tab", ((a + b) // 2, y_mid), (a, ys.start, b, ys.stop), i,
                           "selected" if (a, b) == (xs.start, xs.stop) else "")
                for i, (a, b) in enumerate(spans)]  # fmt: skip
    return []


def find_list_cards(img: np.ndarray) -> list[Affordance]:
    """The cards of MD's card-list panel (:func:`ygorl.md.layout.list_cards`), kept only where the tray really is
    under each card (the strip at y 1045)."""
    cards = L.list_cards(img)
    if not cards:
        return []
    tray = np.array(L.TRAY_RGB)
    for x, _ in cards:
        for dx in (-60, 0, 60):
            if not any((np.abs(img[y, x + dx].astype(int) - tray) < 22).all() for y in range(1030, 1062, 4)):
                return []
    return [Affordance("list_card", (x, y), (x - 69, y - 100, x + 69, y + 100), i) for i, (x, y) in enumerate(cards)]


POSITION_CARDS = (U.POSITION_FACEUP_ATTACK, U.POSITION_FACEUP_DEFENSE)  # the position dialog: face-up Attack, Defense


def find_position(img: np.ndarray) -> list[Affordance]:
    if U.screen_dialog(img) != "position":
        return []
    return [Affordance("position", xy, (xy[0] - 70, xy[1] - 100, xy[0] + 70, xy[1] + 100), i)
            for i, xy in enumerate(POSITION_CARDS)]  # fmt: skip


def find_highlights(img: np.ndarray) -> list[Affordance]:
    """Cards / zones outlined in MD's thin yellow selection frame: yellow rings of card size, mostly hollow; top
    down, then left to right."""
    from scipy import ndimage

    yellow = yellow_mask(img)
    x0, y0, x1, y1 = INFO_PANEL
    yellow[y0:y1, x0:x1] = False  # the card-info panel's art can be yellow
    yellow[HIGHLIGHT_MAX_Y:] = False  # the hand's glow marks playable cards, not a selection
    grown = ndimage.binary_dilation(yellow, iterations=4)
    lab, objs, _ = _components(grown)
    found = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        w, h = xs.stop - xs.start, ys.stop - ys.start
        if not (100 <= w <= 280 and 110 <= h <= 300):
            continue
        part = yellow[sl] & (lab[sl] == i)
        if part.mean() > 0.45:
            continue  # a filled yellow area (a lit card art), not a frame
        # a frame has yellow along at least three of its four sides (bands of an eighth of its size)
        bw, bh = max(8, w // 8), max(8, h // 8)
        sides = [part[:bh].any(axis=0).mean(), part[-bh:].any(axis=0).mean(),
                 part[:, :bw].any(axis=1).mean(), part[:, -bw:].any(axis=1).mean()]  # fmt: skip
        if sum(s > 0.45 for s in sides) < 3:
            continue
        # ... and in its corners (the current phase's ring in the phase menu has none)
        corners = [part[:bh, :bw].any(), part[:bh, -bw:].any(), part[-bh:, :bw].any(), part[-bh:, -bw:].any()]
        if sum(corners) < 3:
            continue
        centre = ((xs.start + xs.stop) // 2, (ys.start + ys.stop) // 2)
        if is_dangerous(centre) or in_box(centre, PHASE_BADGE_BOX):
            continue
        found.append((centre, (xs.start, ys.start, xs.stop, ys.stop)))
    # frames of neighbouring zones touch and merge into one blob: test the zone spots one by one too
    for x, y in ZONE_SPOTS:
        if any(abs(x - c[0]) < 70 and abs(y - c[1]) < 90 for c, _ in found):
            continue
        # free-zone frames sit up to ~20 px off the card centres (perspective): look around the spot
        if any(_frame_at(yellow, x + dx, y + dy) for dx in range(-24, 25, 6) for dy in (0, -12, 12)):
            found.append(((x, y), (x - 75, y - 90, x + 75, y + 90)))  # click the card spot itself
    found.sort(key=lambda f: (round(f[0][1] / 60), f[0][0]))
    return [Affordance("highlight", c, b, i) for i, (c, b) in enumerate(found)]


HIGHLIGHT_MAX_Y = 1000  # frames are searched above the hand
PHASE_BADGE_BOX: Box = (L.PHASE_BADGE[0] - 110, L.PHASE_BADGE[1] - 110, L.PHASE_BADGE[0] + 110, L.PHASE_BADGE[1] + 110)
# field spots a selection frame can sit on: own Main Monster / Spell & Trap zones, the Extra Monster Zones, the
# opponent's Main Monster Zones (ygorl.md.ui's card centres)
ZONE_SPOTS: tuple[XY, ...] = tuple(
    [L.my_zone("mzone", s) for s in range(5)] + [L.my_zone("szone", s) for s in range(5)]
    + [(x, U.MY_EMZ_Y) for x in U.EMZ_X] + [U.opp_monster_xy(s) for s in range(5)]
)  # fmt: skip


def _frame_at(yellow: np.ndarray, x: int, y: int, w: int = 190, h: int = 200, t: int = 24) -> bool:
    """A yellow frame around the zone spot (x, y) (measured on free zones: 185-195 x 200, sides 10-25 px, shared
    with the neighbouring zone's): :func:`ygorl.md.ui.yellow_highlight` on :func:`yellow_mask`, with all four
    sides required."""
    x0, x1, y0, y1 = x - w // 2, x + w // 2, y - h // 2, y + h // 2
    sides = [
        yellow[y0 : y0 + t, x0:x1],
        yellow[y1 - t : y1, x0:x1],
        yellow[y0:y1, x0 : x0 + t],
        yellow[y0:y1, x1 - t : x1],
    ]
    if not all(s.any(axis=0 if k < 2 else 1).mean() > 0.4 for k, s in enumerate(sides)):
        return False
    return bool(yellow[y0 + t : y1 - t, x0 + t : x1 - t].mean() < 0.2)  # hollow: not a lit area between two frames


def yellow_mask(img: np.ndarray) -> np.ndarray:
    """MD's selection yellow, bright or mid-blink (measured (253, 253, 64) ... (197, 195, 74)): red about equal to
    green, blue well below. The orange glow of activatable set cards (253, 175, 59) and lime labels (203, 255, 0)
    have red and green apart."""
    a = img.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (g > 150) & (np.abs(r - g) < 30) & (b < g - 60)


ICON_BOX: Box = (520, 300, 1560, 1000)  # where command icons appear: above own / opposing field and hand cards


def find_command_icons(img: np.ndarray, box: Box = ICON_BOX) -> list[Affordance]:
    """Round command icons (:func:`ygorl.md.layout.find_icons`) whose glyph tells a command
    (:func:`ygorl.md.ui.is_command_icon`: blue Summon / Set, yellow-green Activate), left to right; ``label`` is
    :func:`ygorl.md.ui.icon_kind`. Slow (a circle Hough test, ~0.5 s over :data:`ICON_BOX`), so only on demand.
    Face-down cards pass the shape test too (5 of 22 captures), hence the glyph filter; an icon with another glyph
    (Confirm) is only found by comparing with a capture from before the click (:meth:`Explorer` does)."""
    out = [xy for xy in L.find_icons(img, *box) if not is_dangerous(xy) and U.is_command_icon(img, *xy)]
    return [Affordance("icon", xy, (xy[0] - 44, xy[1] - 44, xy[0] + 44, xy[1] + 44), i,
                       U.icon_kind(img, *xy)) for i, xy in enumerate(out)]  # fmt: skip


def find_affordances(img: np.ndarray, *, icons: bool = False) -> list[Affordance]:
    """Every clickable element found on a reference-frame screenshot (see the module doc); ``icons`` adds the
    (slow) command-icon search over :data:`ICON_BOX`."""
    out = find_buttons(img) + find_radios(img) + find_tabs(img) + find_list_cards(img) + find_position(img)
    out += find_highlights(img)
    if icons:
        out += find_command_icons(img)
    return [a for a in out if not is_dangerous(a.centre)]


def of_kind(affs: Sequence[Affordance], kind: str) -> list[Affordance]:
    return sorted((a for a in affs if a.kind == kind), key=lambda a: a.index)


# --- signature ----------------------------------------------------------------------------------------------------
def memory_key(memory: dict | None) -> str:
    """The memory half of a signature. ``memory`` keys (all optional): ``mode``, ``accepting``, ``dialog_type``,
    ``list_type``, ``list_count``, ``choices`` (the number of dialog choices; an int or a sequence)."""
    m = memory or {}
    choices = m.get("choices", m.get("n_choices", 0))
    n = len(choices) if isinstance(choices, (list, tuple)) else int(choices or 0)
    count = int(m.get("list_count", 0) or 0)
    return (f"m{int(m.get('mode', 0) or 0)}a{int(bool(m.get('accepting')))}d{int(m.get('dialog_type', 0) or 0)}"
            f"l{int(m.get('list_type', 0) or 0)}c{min(count, 2)}n{n}")  # fmt: skip


def screen_class(img: np.ndarray, affs: Sequence[Affordance]) -> str:
    """A coarse name for what is on screen: radio / tabs / position / list / prompt (a list with two buttons) /
    yesno / buttons / phase_menu / highlight / banner / board."""
    kinds = {a.kind for a in affs}
    n_buttons = sum(a.kind == "button" for a in affs)
    if "radio" in kinds:
        return "radio"
    if "tab" in kinds:
        return "tabs"
    if "position" in kinds:
        return "position"
    if "list_card" in kinds:
        return "prompt" if n_buttons >= 2 else "list"
    dialog = U.screen_dialog(img)
    if dialog == "phase_menu":
        return "phase_menu"
    if n_buttons == 2 and dialog == "yesno":
        return "yesno"
    if n_buttons:
        return "buttons"
    if "highlight" in kinds:
        return "highlight"
    if dialog == "banner":
        return "banner"
    return "board"


def visual_key(img: np.ndarray, affs: Sequence[Affordance] | None = None) -> str:
    """The visual half of a signature: the screen class, then each button's position on a coarse grid (128 x 64 px)
    and which option kinds are present. Labels (lit / dim) and counts of options are left out: they change with the
    selection and with the cards offered, not with the kind of dialog."""
    affs = find_affordances(img) if affs is None else affs
    cls = screen_class(img, affs)
    buttons = ",".join(f"{a.centre[0] // 128}.{a.centre[1] // 64}" for a in of_kind(affs, "button"))
    options = "".join(k[0] for k in OPTION_KINDS if any(a.kind == k for a in affs))
    return f"{cls}|b{buttons}|o{options}"


def signature(img: np.ndarray, memory: dict | None = None, affs: Sequence[Affordance] | None = None) -> str:
    """The screen signature: ``memory_key(memory) + "|" + visual_key(img)``."""
    return f"{memory_key(memory)}|{visual_key(img, affs)}"


def sig_visual(sig: str) -> str:
    return sig.split("|", 1)[1] if "|" in sig else sig


def sig_class(sig: str) -> str:
    return sig_visual(sig).split("|", 1)[0]


# --- playbook -----------------------------------------------------------------------------------------------------
Recipe = list[dict]
DECLINE = "#decline"  # key suffix: recipes that decline / pass instead of taking option #choice

# recipes equivalent to the scripts in ygorl.md.ui, keyed by screen class ("class:<cls>"); steps are relative:
#   {"pick": KIND}               the KIND affordance number `choice` (the decision layer's option index)
#   {"pick": KIND, "index": k}   a fixed one
#   {"press": "button", "which": "enabled" | "rightmost" | "leftmost" | "choice" | "index", "index": k}
#   {"press": "icon"}            the command icon that the previous click opened above it (Confirm a target)
SEEDS: dict[str, Recipe] = {
    "class:yesno": [{"press": "button", "which": "choice"}],  # buttons left to right: 0 = 否 No, 1 = 是 Yes
    "class:list": [{"pick": "list_card"}, {"press": "button", "which": "enabled"}],
    "class:prompt": [{"pick": "list_card"}, {"press": "button", "which": "rightmost"}],  # tick, 发动效果 Activate
    "class:prompt" + DECLINE: [{"press": "button", "which": "leftmost"}],  # 取消 Cancel
    "class:radio": [{"pick": "radio"}, {"press": "button", "which": "enabled"}],  # row, 确认 Confirm
    "class:tabs": [{"pick": "tab"}, {"press": "button", "which": "enabled"}],  # effect tab, OK
    "class:position": [{"pick": "position"}],
    "class:highlight": [{"pick": "highlight"}, {"press": "icon"}],
}


@dataclass
class Playbook:
    """Recipes per signature, with success / failure counts, persisted as JSON (re-written on every record)."""

    path: Path | str | None = Path("out/mdplay/playbook.json")
    entries: dict[str, list[dict]] = field(default_factory=dict)
    min_score: float = 0.5  # a recipe is used while (ok + 1) / (ok + fail + 2) is at least this

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            if self.path.exists():
                self.entries = json.loads(self.path.read_text()).get("entries", {})
        for key, recipe in SEEDS.items():
            if key not in self.entries:
                self.entries[key] = [{"recipe": recipe, "ok": 0, "fail": 0, "seed": True}]

    @staticmethod
    def score(e: dict) -> float:
        return (e["ok"] + 1) / (e["ok"] + e["fail"] + 2)

    def recipes(self, sig: str, decline: bool = False) -> list[tuple[str, dict]]:
        """Candidate (source, entry) pairs for ``sig``, best first: this signature's own recipes, then recipes
        learnt on the same visual layout under another memory state, then the seed of its screen class. Entries
        demoted below :attr:`min_score` are left out."""
        suffix = DECLINE if decline else ""
        out: list[tuple[str, dict]] = []
        seen: list[Recipe] = []

        def add(source: str, entries: list[dict]) -> None:
            for e in sorted(entries, key=self.score, reverse=True):
                if self.score(e) >= self.min_score and e["recipe"] not in seen:
                    seen.append(e["recipe"])
                    out.append((source, e))

        own = self.entries.get(sig + suffix, [])
        add("playbook", own)
        failed_here = [e["recipe"] for e in own if self.score(e) < self.min_score]
        seen.extend(failed_here)  # a recipe demoted on this very signature is not offered again through a fallback
        vis = sig_visual(sig)
        add("layout", [e for k, es in self.entries.items() if not k.startswith("class:") and k != sig + suffix
                       and k.endswith(suffix) and sig_visual(k.removesuffix(suffix)) == vis
                       and (suffix or not k.endswith(DECLINE)) for e in es])  # fmt: skip
        add("seed", self.entries.get(f"class:{sig_class(sig)}{suffix}", []))
        return out

    def best(self, sig: str, decline: bool = False) -> Recipe | None:
        found = self.recipes(sig, decline)
        return found[0][1]["recipe"] if found else None

    def record(self, sig: str, recipe: Recipe, ok: bool, decline: bool = False) -> dict:
        """Count one success / failure of ``recipe`` on ``sig`` and save."""
        key = sig + (DECLINE if decline else "")
        entries = self.entries.setdefault(key, [])
        entry = next((e for e in entries if e["recipe"] == recipe), None)
        if entry is None:
            entry = {"recipe": [dict(s) for s in recipe], "ok": 0, "fail": 0}
            entries.append(entry)
        entry["ok" if ok else "fail"] += 1
        entry["last"] = round(time.time(), 1)
        self.save()
        return entry

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": self.entries}, indent=1, ensure_ascii=False))
        os.replace(tmp, self.path)


# --- explorer -----------------------------------------------------------------------------------------------------
class ClickUI(Protocol):
    def click(self, x: int, y: int, why: str, pause: float = 1.0) -> None: ...
    def capture(self) -> np.ndarray: ...
    def sleep(self, seconds: float) -> None: ...


TOGGLES = ("list_card", "highlight")  # option kinds a second click un-ticks (a radio row or a tab stays selected)


@dataclass
class ExploreResult:
    ok: bool
    signature: str
    new_signature: str | None
    recipe: Recipe | None  # the steps that worked (or the last one tried)
    source: str  # "playbook" / "layout" / "seed" (a known recipe worked) / "explore" / "none"
    clicks: list[dict] = field(default_factory=list)  # {"xy", "why", "step"} per click, for the trace
    tried: list[dict] = field(default_factory=list)  # {"recipe", "source", "ok"} per recipe run

    def to_json(self) -> dict:
        return {"ok": self.ok, "signature": self.signature, "new_signature": self.new_signature,
                "recipe": self.recipe, "source": self.source, "clicks": self.clicks, "tried": self.tried}  # fmt: skip


class _Unresolved(Exception):
    """A step names an element the screen does not show (no such option, no lit button ...)."""


Memory = dict | Callable[[], dict] | None


@dataclass
class Explorer:
    """Gets MD past the dialog on screen: the playbook's recipe for its signature, else exploration.

    ``ui`` needs ``click(x, y, why, pause)``, ``capture()`` and ``sleep(s)`` (a :class:`ygorl.md.ui.UI`).
    ``max_clicks`` bounds the clicks of one :meth:`run`; ``settle`` is how long to keep asking ``progress`` after a
    step before calling it a dud; ``enable_wait`` how long to wait for a button to light up after an option click.
    """

    ui: ClickUI
    playbook: Playbook
    max_clicks: int = 12
    settle: float = 2.0
    poll: float = 0.4
    pause: float = 0.8
    enable_wait: float = 2.0

    def stable(self, memory: Memory = None, tries: int = 4) -> tuple[np.ndarray, list[Affordance], str]:
        """A capture whose signature matches the one before it (dialogs fade in); after ``tries`` the last one."""
        last = None
        for _ in range(tries):
            img = self.ui.capture()
            affs = find_affordances(img)
            sig = signature(img, _mem(memory), affs)
            if sig == last:
                return img, affs, sig
            last = sig
            self.ui.sleep(0.3)
        return img, affs, sig

    def screen_progress(self, before_sig: str, memory: Memory = None) -> Callable[[], bool]:
        """A ``progress`` callable from the screen alone: the signature changed (another dialog, or it closed)."""

        def moved_on() -> bool:
            img = self.ui.capture()
            return signature(img, _mem(memory)) != before_sig

        return moved_on

    def run(self, memory: Memory = None, progress: Callable[[], bool] | None = None, choice: int = 0,
            decline: bool = False) -> ExploreResult:  # fmt: skip
        """Answer the dialog on screen with option ``choice`` (or decline it). ``progress`` tells whether MD moved
        on (default: :meth:`screen_progress`); ``memory`` is the dict for :func:`memory_key`, or a callable that
        reads it now."""
        img, affs, sig = self.stable(memory)
        progress = progress or self.screen_progress(sig, memory)
        res = ExploreResult(False, sig, None, None, "none")
        tried: list[Recipe] = []
        for source, entry in self.playbook.recipes(sig, decline):
            if len(res.clicks) >= self.max_clicks:
                break
            recipe = entry["recipe"]
            tried.append(recipe)
            ok, done = self._attempt(recipe, choice, progress, res, source)
            if done is not None:  # a step resolved: the outcome says something about this recipe
                self.playbook.record(sig, recipe, ok, decline)
            if ok:
                return self._finish(res, recipe, source, memory)
        for recipe in self.candidates(affs, choice, decline):
            if len(res.clicks) >= self.max_clicks:
                break
            if recipe in tried:
                continue
            tried.append(recipe)
            ok, done = self._attempt(recipe, choice, progress, res, "explore")
            if ok:
                self.playbook.record(sig, done, True, decline)
                return self._finish(res, done, "explore", memory)
        res.recipe = tried[-1] if tried else None
        return res

    def _finish(self, res: ExploreResult, recipe: Recipe, source: str, memory: Memory) -> ExploreResult:
        res.ok, res.recipe, res.source = True, recipe, source
        img = self.ui.capture()
        res.new_signature = signature(img, _mem(memory))
        return res

    # --- candidates -----------------------------------------------------------------------------------------------
    @staticmethod
    def candidates(affs: Sequence[Affordance], choice: int = 0, decline: bool = False) -> list[Recipe]:
        """Recipes to try on a screen never answered before, most plausible first: the option #choice of each
        option kind present followed by a confirm-like button (the lit one, then each button right to left), the
        option alone; with no options, button #choice alone, then each other button. Declining tries each button
        alone, left to right (Cancel / No sit left)."""
        buttons = of_kind(affs, "button")
        out: list[Recipe] = []
        if decline:
            return [[{"press": "button", "which": "index", "index": b.index}] for b in buttons]

        def with_option(kind: str) -> None:
            opts = of_kind(affs, kind)
            if not opts or choice >= len(opts):
                return
            pick = {"pick": kind}
            out.append([pick, {"press": "button", "which": "enabled"}])
            out.extend([pick, {"press": "button", "which": "index", "index": b.index}] for b in reversed(buttons))
            if kind == "highlight":
                out.append([pick, {"press": "icon"}])
            out.append([pick])

        # a yes / no look (no other options, every button lit): the buttons are the options, and a framed card next
        # to them is the card the question is about; otherwise options come first, so that a lit Cancel is tried
        # only after option + Confirm (it makes "progress" too)
        dialog_options = any(a.kind in OPTION_KINDS and a.kind != "highlight" for a in affs)
        yesno_like = not dialog_options and len(buttons) >= 2 and all(b.label == "enabled" for b in buttons)
        for kind in OPTION_KINDS:
            if kind != "highlight" or not yesno_like:
                with_option(kind)
        if buttons:
            out.append([{"press": "button", "which": "choice"}])
            out += [[{"press": "button", "which": "index", "index": b.index}] for b in buttons if b.index != choice]
        if yesno_like:
            with_option("highlight")
        uniq: list[Recipe] = []
        for r in out:
            if r not in uniq:
                uniq.append(r)
        return uniq

    # --- running a recipe -----------------------------------------------------------------------------------------
    def _attempt(self, recipe: Recipe, choice: int, progress: Callable[[], bool], res: ExploreResult,
                 source: str) -> tuple[bool, Recipe | None]:  # fmt: skip
        """Run ``recipe`` step by step, checking ``progress`` after each click. Returns (moved on, the steps that
        ran: the prefix that made progress, or the whole recipe; None when its first step did not resolve)."""
        done: Recipe = []
        ticked: list[tuple[XY, str]] = []
        last: XY | None = None
        before: np.ndarray | None = None
        for k, step in enumerate(recipe):
            if len(res.clicks) >= self.max_clicks:
                break
            try:
                xy, why = self._resolve(step, choice, last, before)
                # an icon step needs the screen from before the click that opens the icon
                before = self.ui.capture() if k + 1 < len(recipe) and recipe[k + 1].get("press") == "icon" else None
            except _Unresolved as e:
                res.tried.append({"recipe": recipe, "source": source, "ok": False, "unresolved": str(e)})
                self._undo(ticked, res)
                return False, (done or None) and recipe
            self._click(xy, why, res, step)
            done.append(step)
            last = xy
            if step.get("pick") in TOGGLES:
                ticked.append((xy, why))
            # an option click that leaves the dialog up only changed the selection (MD may note a tick in memory):
            # go on to the confirm step
            only_selection = "pick" in step and k + 1 < len(recipe)
            if self._wait_progress(progress, sig_visual(res.signature) if only_selection else None):
                res.tried.append({"recipe": done, "source": source, "ok": True})
                return True, done
        res.tried.append({"recipe": recipe, "source": source, "ok": False})
        self._undo(ticked, res)
        return False, recipe

    def _undo(self, ticked: list[tuple[XY, str]], res: ExploreResult) -> None:
        for xy, why in reversed(ticked):  # un-tick what this attempt ticked before the next one
            self._click(xy, f"undo: {why}", res, {"undo": True})

    def _wait_progress(self, progress: Callable[[], bool], same_screen: str | None = None) -> bool:
        """Poll ``progress`` for :attr:`settle` seconds. With ``same_screen`` (a visual key), progress while the
        screen still shows that key is only a selection change: False at once."""
        for _ in range(max(1, round(self.settle / self.poll))):
            if progress():
                return same_screen is None or visual_key(self.ui.capture()) != same_screen
            self.ui.sleep(self.poll)
        return False

    def _click(self, xy: XY, why: str, res: ExploreResult, step: dict) -> None:
        if is_dangerous(xy):  # affordances never sit there; this is the last line
            raise U.Stuck(f"refusing to click a danger zone at {xy} ({why})")
        self.ui.click(int(xy[0]), int(xy[1]), f"explore: {why}", self.pause)
        res.clicks.append({"xy": [int(xy[0]), int(xy[1])], "why": why, "step": step})

    def _resolve(self, step: dict, choice: int, last: XY | None, before: np.ndarray | None = None) -> tuple[XY, str]:
        """The point a step clicks on the current screen."""
        if "pick" in step:
            kind = step["pick"]
            idx = int(step.get("index", choice))
            opts = of_kind(find_affordances(self.ui.capture()), kind)
            if not 0 <= idx < len(opts):
                raise _Unresolved(f"{kind} #{idx} asked for, {len(opts)} shown")
            return opts[idx].centre, f"{kind} {idx + 1} of {len(opts)}"
        if step.get("press") == "icon":
            if last is None:
                raise _Unresolved("an icon step with no click before it")
            x, y = last
            box = (max(x - 130, 0), max(y - 260, 0), min(x + 130, L.REF_W), max(y - 60, 1))
            for _ in range(max(1, round(self.enable_wait / self.poll))):  # MD's Confirm icon blinks
                now = self.ui.capture()
                icons = L.new_icons(before, now, *box) if before is not None else []
                icons = icons or [p for p in L.find_icons(now, *box) if U.is_command_icon(now, *p)]
                icons = [p for p in icons if not is_dangerous(p)]
                if icons:
                    return min(icons, key=lambda p: abs(p[0] - x)), "the icon above it"
                self.ui.sleep(self.poll)
            raise _Unresolved("no icon above the last click")
        if step.get("press") == "button":
            which = step.get("which", "enabled")
            for _ in range(max(1, round(self.enable_wait / self.poll))):
                buttons = of_kind(find_affordances(self.ui.capture()), "button")
                b = _pick_button(buttons, which, int(step.get("index", choice)))
                if b is not None:
                    return b.centre, f"button {which} ({b.index + 1} of {len(buttons)}, {b.label})"
                if which != "enabled":
                    break
                self.ui.sleep(self.poll)  # a button lights up once MD registered the option
            raise _Unresolved(f"no {which} button")
        raise _Unresolved(f"unknown step {step}")


def _pick_button(buttons: Sequence[Affordance], which: str, index: int) -> Affordance | None:
    if not buttons:
        return None
    if which == "enabled":
        lit = [b for b in buttons if b.label == "enabled"]
        return lit[-1] if lit else None  # the rightmost lit one: Confirm / Activate sit right of Cancel
    if which == "rightmost":
        return max(buttons, key=lambda b: b.centre[0])
    if which == "leftmost":
        return min(buttons, key=lambda b: b.centre[0])
    if which in ("choice", "index"):
        return buttons[index] if 0 <= index < len(buttons) else None
    return None


def _mem(memory: Memory) -> dict:
    if callable(memory):
        return memory()
    return memory or {}


__all__ = [
    "DANGER_ZONES",
    "DECLINE",
    "SEEDS",
    "Affordance",
    "ExploreResult",
    "Explorer",
    "Playbook",
    "find_affordances",
    "find_buttons",
    "is_dangerous",
    "memory_key",
    "signature",
    "visual_key",
]


def _main(argv: Sequence[str] | None = None) -> None:
    """``python -m ygorl.md.explore SCREEN.npy|.npz ...``: print the affordances and visual key of captures."""
    import sys

    for p in argv if argv is not None else sys.argv[1:]:
        img = _load(p)
        affs = find_affordances(img)
        print(p, visual_key(img, affs))
        for a in affs:
            print("  ", a.kind, a.index, a.centre, a.label)


def _load(path: str) -> np.ndarray:
    """A saved capture as a reference-frame image: ``screen.npy`` of a stuck record, or a trace's half-resolution
    ``.npz``."""
    if path.endswith(".npz"):
        half = np.load(path)["screen"]
        return np.ascontiguousarray(half.repeat(2, 0).repeat(2, 1)[: L.REF_H, : L.REF_W])
    return np.load(path)


if __name__ == "__main__":
    _main()
