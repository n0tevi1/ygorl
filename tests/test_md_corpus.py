"""Regression corpus of labelled Master Duel screens (design 10, 「识别与执行的系统化」 item 1).

Every stuck record ``out/mdplay/stuck-*/screen.npy`` (reference frame, 2054x1184 RGB) has a hand-made ground truth in
``out/mdplay/corpus/labels.json``: ``{"<dir>": {"class", "skin", "targets", "note"}}``, with ``targets`` in reference
coordinates. This test runs the current recognisers over the whole corpus and fails with a table of every mismatch,
after a per-class / per-recogniser accuracy summary, so the output doubles as a benchmark. The corpus is not in git:
the module is skipped when ``labels.json`` is missing (``YGORL_MD_CORPUS`` may point at another corpus directory).

Run as a script (``python tests/test_md_corpus.py``) to print the report without pytest.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from ygorl.md import layout as L
from ygorl.md import ui


def _corpus_dir() -> Path | None:
    if os.environ.get("YGORL_MD_CORPUS"):  # an explicit corpus: nothing else
        candidates = [Path(os.environ["YGORL_MD_CORPUS"])]
        return candidates[0] if (candidates[0] / "labels.json").is_file() else None
    candidates = []
    repo = Path(__file__).resolve().parents[1]
    candidates.append(repo / "out" / "mdplay" / "corpus")
    dotgit = repo / ".git"
    if dotgit.is_file():  # a git worktree: the corpus lives in the main checkout's out/
        gitdir = Path(dotgit.read_text().split("gitdir:", 1)[1].strip())
        candidates.append(gitdir.parents[2] / "out" / "mdplay" / "corpus")  # <main>/.git/worktrees/<name>
    return next((c for c in candidates if (c / "labels.json").is_file()), None)


CORPUS = _corpus_dir()
if CORPUS is None and __name__ != "__main__":
    pytest.skip("no labelled Master Duel corpus (out/mdplay/corpus/labels.json)", allow_module_level=True)
TOL = 15  # px: how far a located element may be from its label
SLOT_TOL = 45  # px: a labelled outlined zone / card belongs to the zone or monster slot this close to it

# the classifier's vocabulary for each labelled class; classes it has no name for only must not get an actionable one
EXPECTED: dict[str, set[str | None]] = {
    "board": {None, "panel"},
    "yesno": {"yesno"},
    "option": {"panel"},  # memory tells the effect choice apart
    "list": {"list"},
    "prompt": {"list"},
    "radio": {"radio"},
    "position": {"position"},
    "notice": {"notice"},
    "banner": {"banner"},
    "phase_menu": {"phase_menu"},
    "minimised": {"minimised"},
}
NAMELESS = {None, "panel"}  # declare / result / menu / other


@dataclass
class Report:
    checked: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mismatches: list[tuple[str, str, str, str, str]] = field(default_factory=list)  # dir, recogniser, label, got, why

    def check(self, key: str, ok: bool, row: tuple[str, str, str, str, str]) -> None:
        self.checked[key] += 1
        self.correct[key] += ok
        if not ok:
            self.mismatches.append(row)

    def table(self) -> str:
        lines = ["accuracy:"]
        for key in sorted(self.checked):
            n, k = self.checked[key], self.correct[key]
            lines.append(f"  {key:<34} {k:>4}/{n:<4} {k / n:6.1%}")
        lines.append(f"mismatches ({len(self.mismatches)}):")
        widths = [max([len(r[i]) for r in self.mismatches] + [4]) for i in range(4)]
        for row in self.mismatches:
            lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(row[:4], widths, strict=True)) + "  " + row[4])
        return "\n".join(lines)


def _near(a, b, tol: int = TOL) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _fmt(xy) -> str:
    return "None" if xy is None else f"({int(xy[0])},{int(xy[1])})"


def _slots(points, slots: dict[str, tuple[int, int]]) -> set[str]:
    """The named slots the labelled points fall on."""
    out = set()
    for p in points:
        name, xy = min(slots.items(), key=lambda kv: abs(kv[1][0] - p[0]) + abs(kv[1][1] - p[1]))
        if _near(xy, p, SLOT_TOL):
            out.add(name)
    return out


ZONES = {f"{loc[0]}{i}": L.my_zone(loc, i) for loc in ("mzone", "szone") for i in range(5)}
MONSTERS = {f"me{i}": ui.my_monster_xy(i) for i in range(7)} | {f"op{i}": ui.opp_monster_xy(i) for i in range(7)}
MONSTERS = {k: v for k, v in MONSTERS.items() if k not in ("op5", "op6")}  # the EMZ is shared: same centres as me5/me6


def evaluate(corpus: Path) -> Report:
    labels = json.loads((corpus / "labels.json").read_text())
    rep = Report()
    for name in sorted(labels):
        lab = labels[name]
        path = corpus.parent / name / "screen.npy"
        if not path.is_file():
            rep.check("(screen missing)", False, (name, "load", lab["class"], "-", f"no {path}"))
            continue
        img = np.load(path)[:, :, :3]
        cls, t = lab["class"], lab.get("targets", {})
        tag = f"{cls}/{lab.get('skin', '?')}"

        got = ui.screen_dialog(img)
        allowed = EXPECTED.get(cls, NAMELESS)
        rep.check(f"screen_dialog [{cls}]", got in allowed,
                  (name, "screen_dialog", tag, str(got), f"want {' or '.join(map(str, sorted(allowed, key=str)))}"))  # fmt: skip

        cards = L.list_cards(img)
        if cls in ("list", "prompt"):
            want = t.get("cards", [])
            ok = len(cards) == len(want) and all(abs(c[0] - w[0]) <= TOL for c, w in zip(cards, want, strict=False))
            rep.check("list_cards (lists)", ok, (name, "list_cards", tag, " ".join(map(_fmt, cards)),
                                                  "want x " + " ".join(str(w[0]) for w in want)))  # fmt: skip
        else:
            rep.check("list_cards (no list: empty)", not cards,
                      (name, "list_cards", tag, " ".join(map(_fmt, cards)), "no card list on screen"))  # fmt: skip

        pos = ui.position_buttons(img)
        if cls == "position":
            ok = pos is not None and _near(pos[0], t["attack"]) and _near(pos[1], t["defense"])
            rep.check("position_buttons (dialog)", ok, (name, "position_buttons", tag, str(pos and tuple(map(_fmt, pos))),
                                                         f"want {_fmt(t['attack'])} {_fmt(t['defense'])}"))  # fmt: skip
        else:
            rep.check("position_buttons (none: None)", pos is None,
                      (name, "position_buttons", tag, str(tuple(map(_fmt, pos)) if pos else None), "no position dialog"))  # fmt: skip

        rows = ui.radio_rows(img)
        if cls == "radio":
            want = [r[1] for r in t["rows"]]
            ok = len(rows) == len(want) and all(abs(a - b) <= TOL for a, b in zip(rows, want, strict=True))
            rep.check("radio_rows (dialog)", ok, (name, "radio_rows", tag, str(rows), f"want y {want}"))
        else:
            rep.check(
                "radio_rows (none: <2 rows)", len(rows) < 2, (name, "radio_rows", tag, str(rows), "no radio dialog")
            )

        button = ui.notice_button(img)
        if cls == "notice":
            ok = button is not None and _near(button, t["confirm"])
            rep.check(
                "notice_button (notice)", ok, (name, "notice_button", tag, _fmt(button), f"want {_fmt(t['confirm'])}")
            )
        else:
            rep.check("notice_button (none: None)", button is None,
                      (name, "notice_button", tag, _fmt(button), "no notice on screen"))  # fmt: skip

        want_z = _slots(t.get("outlined_zones", []), ZONES)
        got_z = {k for k, xy in ZONES.items() if ui.zone_outlined(img, *xy)}
        rep.check(f"zone_outlined ({'outlined' if want_z else 'none'})", got_z == want_z,
                  (name, "zone_outlined", tag, str(sorted(got_z)), f"want {sorted(want_z)}"))  # fmt: skip

        want_h = _slots(t.get("outlined_cards", []), MONSTERS)
        skip = _slots(t.get("selected_cards", []), MONSTERS)  # already chosen: framed or glowing, either is fine
        got_h = {k for k, xy in MONSTERS.items() if ui.yellow_highlight(img, *xy)} - skip
        rep.check(f"yellow_highlight ({'outlined' if want_h else 'none'})", got_h == want_h,
                  (name, "yellow_highlight", tag, str(sorted(got_h)), f"want {sorted(want_h)}"))  # fmt: skip
    return rep


def test_recognisers_on_corpus():
    rep = evaluate(CORPUS)
    print(rep.table())
    assert not rep.mismatches, "\n" + rep.table()


if __name__ == "__main__":
    if CORPUS is None:
        raise SystemExit("no corpus: set YGORL_MD_CORPUS")
    print(evaluate(CORPUS).table())
