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


def panel_open(img: np.ndarray) -> bool:
    """The information panel is up: a dark body right of the digits and some white name text."""
    x0, y0, x1, y1 = BODY
    return bool(img[y0:y1, x0:x1].astype(int).mean() < 45 and _white(img, NAME_BAND).mean() > 0.02)


def name_signature(img: np.ndarray) -> np.ndarray:
    return _white(img, NAME_BAND)


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
        text = ""
        for g in digit_glyphs(img, band):
            best, score = "?", 0.0
            for ch, temps in self.digits.items():
                for t in temps:
                    s = float((g == t).mean())
                    if s > score:
                        best, score = ch, s
            text += best if score >= 0.9 else "?"
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
        return (fitting[0], "stats") if len(fitting) == 1 else (None, "ambiguous")

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
            text, glyphs = str(value), digit_glyphs(img, band)
            if value < 0 or len(text) != len(glyphs):
                continue  # "?" ATK, or a split that does not match the number: do not learn from it
            for ch, g in zip(text, glyphs, strict=True):
                self.digits.setdefault(ch, []).append(g)


__all__ = ["PanelBook", "digit_glyphs", "name_signature", "panel_open", "same_name"]
