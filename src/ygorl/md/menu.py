"""From a won Solo duel to the next one (design 10, 「多局」): MD's screens between two duels.

A won Solo duel offers no Retry. Its Confirm leads out of the duel into the stage menu, and the next duel is started
from there. The user allowed the player to click through these screens (2026-10-22). Outside a duel the Solo gate
cannot run (MD frees the duel state), so :class:`MenuWalker` is guarded differently: it clicks only fixed targets
on screens it recognises, and nothing else.

Screens (reference frame of :mod:`ygorl.md.layout`, measured on one pass through the SP deck challenge):

- ``victory``: VICTORY with one Confirm (lime label at :data:`VICTORY_CONFIRM`);
- ``back``: the LEVEL & EXP overlay on the stage list, closed with its Back button (:data:`BACK`);
- ``popup``: a reward popup (通关奖励) with one Confirm whose height depends on the text: found with
  :func:`ygorl.md.explore.find_buttons`;
- ``detail``: a stage's page with the big lime Play button (:data:`PLAY`);
- ``turn_order``: "go first or second?" before some stages: random (the top row);
- ``stages``: the row of stage nodes. The target is the first newly unlocked node (white border), else the last
  lime node (the current stage: replaying it, or opening its reward popup).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

XY = tuple[int, int]
VICTORY_CONFIRM = (1027, 1065)
BACK = (1825, 1079)
PLAY, PLAY_BOX = (1675, 1002), (1460, 975, 1890, 1030)
NODE_EDGE_Y, NODE_CENTRE_DY = 515, 120  # the nodes' top edge row (measured); their centre is this far below it
NODE_EDGE_BAND = range(480, 600)  # the edge row moves with the layout (538 on the headless display): searched
NODE_MIN_WIDTH = 40  # a node partly scrolled off the right edge still counts


def _lime_label(img: np.ndarray, x: int, y: int) -> float:
    r = img[y - 14 : y + 14, x - 60 : x + 60].reshape(-1, 3).astype(int)
    return float(((r[:, 1] > 150) & (r[:, 2] < 90)).mean())


def _runs(mask: np.ndarray, least: int) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= least]


def _edge_nodes(img: np.ndarray) -> tuple[int, list[tuple[str, int]]]:
    """The row of node top edges with the most nodes, and those nodes."""
    best: tuple[int, list[tuple[str, int]]] = (NODE_EDGE_Y, [])
    for y in NODE_EDGE_BAND:
        row = img[y].astype(int)
        lime = (row[:, 1] > 180) & (row[:, 0] > 130) & (row[:, 2] < 80)
        white = row.min(axis=1) > 180
        found = [("lime", (a + b) // 2) for a, b in _runs(lime, NODE_MIN_WIDTH) if b - a < 260]
        found += [("white", (a + b) // 2) for a, b in _runs(white, NODE_MIN_WIDTH) if b - a < 260]
        if len(found) > len(best[1]):
            best = (y, sorted(found, key=lambda n: n[1]))
    return best


def stage_nodes(img: np.ndarray) -> list[tuple[str, int]]:
    """The stage nodes on screen, left to right: ("lime" cleared / current, or "white" newly unlocked, centre x).
    Locked nodes (grey) are left out."""
    return _edge_nodes(img)[1]


TURN_ORDER_ROWS = (571, 661, 751)  # 随机 (random) / 先攻 (first) / 后攻 (second), stacked at x 1027


def turn_order(img: np.ndarray) -> bool:
    """The "go first or second?" popup some stages open before the duel: three lime labels stacked 90 px apart
    (0.10-0.12 measured) with nothing between them (the stage list's lime nodes have no such gaps)."""
    x = VICTORY_CONFIRM[0]
    labels = [_lime_label(img, x, y) for y in TURN_ORDER_ROWS]
    gaps = [_lime_label(img, x, y) for y in (616, 706)]
    return min(labels) > 0.06 and max(gaps) < 0.01


def screen(img: np.ndarray) -> str | None:
    """Which between-duels screen ``img`` shows (see the module docstring), or None."""
    from ygorl.md.explore import find_buttons

    x0, y0, x1, y1 = PLAY_BOX
    r = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
    if ((r[:, 1] > 200) & (r[:, 0] > 150) & (r[:, 2] < 100)).mean() > 0.3:
        return "detail"
    if _lime_label(img, *VICTORY_CONFIRM) > 0.05:
        return "victory"
    if _lime_label(img, *BACK) > 0.05:
        return "back"
    if turn_order(img):
        return "turn_order"
    if [b for b in find_buttons(img) if b.label == "enabled"]:
        return "popup"
    if len(stage_nodes(img)) >= 2:
        return "stages"
    return None


def target(img: np.ndarray, kind: str) -> XY | None:
    """Where to click on a recognised screen."""
    from ygorl.md.explore import find_buttons

    if kind == "detail":
        return PLAY
    if kind == "victory":
        return VICTORY_CONFIRM
    if kind == "back":
        return BACK
    if kind == "turn_order":
        return VICTORY_CONFIRM[0], TURN_ORDER_ROWS[0]  # random: the player does not pick a side
    if kind == "popup":
        lit = [b for b in find_buttons(img) if b.label == "enabled"]
        return lit[0].centre if len(lit) == 1 else None  # one Confirm, else not a screen of ours
    if kind == "stages":
        edge, nodes = _edge_nodes(img)
        new = [x for c, x in nodes if c == "white"]
        x = new[0] if new else max(x for c, x in nodes if c == "lime")
        return min(x, 2010), edge + NODE_CENTRE_DY
    return None


@dataclass
class MenuWalker:
    """Click through the between-duels screens until ``duel_running()``; ``click(x, y)`` takes reference points."""

    click: Callable[[int, int], None]
    capture: Callable[[], np.ndarray]
    duel_running: Callable[[], bool]
    sleep: Callable[[float], None] = time.sleep
    log: list[tuple[str, XY]] = field(default_factory=list)

    def run(self, timeout: float = 240.0, step: float = 2.5, clock: Callable[[], float] = time.time) -> bool:
        end = clock() + timeout
        while clock() < end:
            if self.duel_running():
                return True
            img = self.capture()
            kind = screen(img)
            xy = target(img, kind) if kind else None
            if xy is not None:
                self.log.append((kind, xy))
                print(f"  menu: {kind} -> click {xy}", flush=True)
                self.click(*xy)
            self.sleep(step)
        return False


__all__ = ["MenuWalker", "screen", "stage_nodes", "target"]
