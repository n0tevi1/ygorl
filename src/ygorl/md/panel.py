"""Which card MD's card-information panel shows (design 10): verify a hand click instead of trusting a sort rule.

Clicking a hand card opens the panel at the top left: the card's name, and for monsters the level, ATK and DEF in a
large white font. The hand's on-screen order is MD's own (client) sort, inferred from observations only, so the player
checks the panel before using a command:

- **name**: the white text of the name row as a bitmap, compared with names learned for card ids;
- **stats** (monsters): the level / ATK / DEF digits, read with learned digit templates, filtered against the stats
  of the cards in hand.

Both are learned while playing (:meth:`PanelBook.learn`): once memory confirms an action on card X, the panel crop
taken when X was opened becomes X's name signature and labels its digits. Unknown digits read as ``?`` and match
anything; with nothing learned yet the panel cannot tell, and the caller falls back to the sort rule.

Positions are in the reference frame of :mod:`ygorl.md.layout` (measured on two full-resolution captures).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

NAME_BAND = (40, 194, 400, 221)  # x0, y0, x1, y1
LEVEL_BAND = (264, 260, 300, 292)
ATK_BAND = (264, 306, 372, 339)
DEF_BAND = (264, 354, 372, 388)
BODY = (372, 250, 440, 420)  # right of the digits: the panel's dark body
DIGIT = (14, 20)  # template size (w, h)


def _white(img: np.ndarray, band) -> np.ndarray:
    """Text pixels: white, or blue for a modified ATK / DEF (bright in some channel on the dark panel)."""
    x0, y0, x1, y1 = band
    return img[y0:y1, x0:x1].max(axis=2) > 180


RIM_Y, RIM_RGB = 180, (57, 58, 60)  # the grey rim above the panel's name bar (Chinese client; 205 in English)


def rim_y(img: np.ndarray) -> int | None:
    """The row of the panel's grey top rim, or None (no panel)."""
    for y in range(150, 260):
        row = img[y, 80:440].astype(int)
        if (np.abs(row - np.array(RIM_RGB)).max(axis=1) < 10).mean() > 0.9:
            return y
    return None


def _shifted(band, img):
    """``band`` moved with the panel: the English client draws the whole panel (name bar, level / ATK / DEF rows)
    25 px lower than the Chinese one; the top rim tells by how much."""
    dy = (rim_y(img) or RIM_Y) - RIM_Y
    x0, y0, x1, y1 = band
    return x0, y0 + dy, x1, y1 + dy


def panel_open(img: np.ndarray) -> bool:
    """The information panel is up: its grey top rim and some white name text under it."""
    return rim_y(img) is not None and bool(_white(img, _shifted(NAME_BAND, img)).mean() > 0.02)


HEADER = (380, 175, 440, 192)  # the name bar's colour, right of the name


def panel_kind(img: np.ndarray) -> str:
    """The card kind the panel's header colour shows: "spell" (teal, measured 44 96 90), "monster" (orange / yellow:
    104 77 58 effect, 104 91 65 normal), "purple" (a Trap, or an Extra Deck monster such as Fusion: 100 58 94), "" else.
    A hand card is never an Extra Deck monster, so "purple" there means a Trap."""
    x0, y0, x1, y1 = _shifted(HEADER, img)
    x1 = min(x1, 380)  # the English name bar ends at the attribute icon (x ~390)
    x0 = min(x0, x1 - 60)
    r, g, b = img[y0 + 6 : y1 + 6, x0:x1].reshape(-1, 3).astype(int).mean(axis=0)
    if g > r + 25 and b > r + 20:
        return "spell"
    if r > b + 20 and g > b + 10:
        return "monster"
    if r > g + 20 and b > g + 15:
        return "purple"
    return ""


def name_signature(img: np.ndarray) -> np.ndarray:
    return _white(img, _shifted(NAME_BAND, img))


def same_name(a: np.ndarray, b: np.ndarray, shift: int = 2) -> float:
    """Best IoU of two name bitmaps over small horizontal shifts."""
    best = 0.0
    for dx in range(-shift, shift + 1):
        bb = np.roll(b, dx, axis=1)
        union = (a | bb).sum()
        if union:
            best = max(best, float((a & bb).sum() / union))
    return best


def digit_glyphs(img: np.ndarray, band) -> list[np.ndarray]:
    """The digits of a number field, left to right, each resized to :data:`DIGIT` (empty columns split them)."""
    mask = _white(img, band)
    cols = mask.any(axis=0)
    glyphs, start = [], None
    for x, on in enumerate(np.append(cols, False)):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= 3:
                part = mask[:, start:x]
                rows = np.nonzero(part.any(axis=1))[0]
                part = part[rows.min() : rows.max() + 1]
                ys = (np.arange(DIGIT[1]) * part.shape[0] / DIGIT[1]).astype(int)
                xs = (np.arange(DIGIT[0]) * part.shape[1] / DIGIT[0]).astype(int)
                glyphs.append(part[ys][:, xs])
            start = None
    return glyphs


DIGIT_IOU = 0.92  # measured: a digit against its own templates >= 0.975, against other digits' <= 0.84


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


NAME_TEXT_END = 340  # band columns from here on are the attribute icon (its white rim)


def name_width(sig: np.ndarray) -> int:
    """The width of the name text in a name signature (pixels from its first to its last white column). The band's
    last columns hold the attribute icon's white rim: left out."""
    cols = np.flatnonzero(sig[:, :NAME_TEXT_END].any(axis=0))
    return int(cols[-1] - cols[0] + 1) if len(cols) else 0


_NARROW, _WIDE = set("ijlI.,'!:;|()[] -ftr\"\u00b7"), set("mwMW@%")


def predicted_width(name: str) -> float:
    """A rough width of ``name`` in the panel's proportional font, in units of an average lowercase letter."""
    return sum(0.45 if c in _NARROW else 1.45 if c in _WIDE else 1.15 if c.isupper() or c.isdigit() else 1.0
               for c in name)  # fmt: skip


PX_PER_UNIT = 17.8  # measured on the English client: Raigeki 107 px (6.0 units), Blaze Cannon 200 px (11.2)
NAME_FULL = 320  # text this wide runs into the attribute icon: the name is truncated and its length hidden


def width_fits(width: int, name: str, tolerance: float = 0.15) -> bool:
    """Whether a never-learned panel whose name text is ``width`` px wide can show the card called ``name``: lets
    the player tell two unlearned cards of one kind apart (Raigeki from Blaze Cannon) without guessing."""
    want = predicted_width(name) * PX_PER_UNIT
    if width >= NAME_FULL:
        return want >= NAME_FULL * (1 - tolerance)
    return abs(width - want) <= tolerance * want


class PanelBook:
    """Learned name signatures (card id -> bitmaps) and digit templates (char -> bitmaps); saved as one ``.npz``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.names: dict[int, list[np.ndarray]] = {}
        self.digits: dict[str, list[np.ndarray]] = {}
        if path is not None and Path(path).is_file():
            data = np.load(path)
            for key in data.files:
                kind, label, _ = key.split(":")
                (self.names.setdefault(int(label), []) if kind == "n" else self.digits.setdefault(label, [])).append(
                    data[key]
                )

    def save(self) -> None:
        if self.path is None:
            return
        out = {}
        for cid, sigs in self.names.items():
            out.update({f"n:{cid}:{i}": s for i, s in enumerate(sigs[-3:])})
        for ch, glyphs in self.digits.items():
            out.update({f"d:{ch}:{i}": g for i, g in enumerate(glyphs[-5:])})
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **out)

    # --- reading ---
    def read_number(self, img: np.ndarray, band) -> str:
        """Each digit is the template with the best foreground IoU, if that reaches :data:`DIGIT_IOU`. (Pixel
        agreement was dominated by the background: a level 6 read as 5 at 0.925.)"""
        text = ""
        for g in digit_glyphs(img, _shifted(band, img)):
            best, score = "?", 0.0
            for ch, temps in self.digits.items():
                for t in temps:
                    s = _iou(g, t)
                    if s > score:
                        best, score = ch, s
            text += best if score >= DIGIT_IOU else "?"
        return text

    def identify(self, img: np.ndarray, candidates: Sequence[int], stats: Mapping[int, tuple[int, int, int] | None]
                 ) -> tuple[int | None, str]:  # fmt: skip
        """The card id among ``candidates`` the panel shows, and how it was told ("name", "stats", "closed",
        "unknown", "ambiguous"). ``stats``: card id -> (level, attack, defense) for monsters, None otherwise."""
        if not panel_open(img):
            return None, "closed"
        sig = name_signature(img)
        named = [c for c in dict.fromkeys(candidates) if any(same_name(sig, s) > 0.85 for s in self.names.get(c, ()))]
        if len(named) == 1:
            return named[0], "name"
        level, atk, dfn = (self.read_number(img, b) for b in (LEVEL_BAND, ATK_BAND, DEF_BAND))
        if not level or level.count("?") == len(level):
            return None, "unknown"

        def fits(read: str, value: int) -> bool:
            want = str(value) if value >= 0 else "?"
            return len(read) == len(want) and all(r in ("?", w) for r, w in zip(read, want, strict=True))

        fitting = [c for c in dict.fromkeys(candidates) if stats.get(c) is not None and fits(level, stats[c][0])
                   and fits(atk, stats[c][1]) and fits(dfn, stats[c][2])]  # fmt: skip
        if len(fitting) == 1 and "?" not in level + atk + dfn:
            return fitting[0], "stats"
        if not fitting:
            return None, "unknown"
        # digits that could not be read match anything: one fitting card is a guess, not an identification (two
        # positions of one list both "matched" Ra this way)
        return None, "ambiguous" if len(fitting) > 1 else "partial"

    # --- learning ---
    def learn(self, img: np.ndarray, cid: int, stats: tuple[int, int, int] | None) -> None:
        """The panel in ``img`` showed card ``cid`` (confirmed in memory afterwards)."""
        if not panel_open(img):
            return
        sigs = self.names.setdefault(cid, [])
        sig = name_signature(img)
        if not any(same_name(sig, s) > 0.95 for s in sigs):
            sigs.append(sig)
        if stats is None:
            return
        for band, value in zip((LEVEL_BAND, ATK_BAND, DEF_BAND), stats, strict=True):
            text, glyphs = str(value), digit_glyphs(img, _shifted(band, img))
            if value < 0 or len(text) != len(glyphs):
                continue  # "?" ATK, or a split that does not match the number: do not learn from it
            for ch, g in zip(text, glyphs, strict=True):
                self.digits.setdefault(ch, []).append(g)


__all__ = [
    "PanelBook",
    "digit_glyphs",
    "name_signature",
    "name_width",
    "panel_kind",
    "panel_open",
    "predicted_width",
    "same_name",
    "width_fits",
]
