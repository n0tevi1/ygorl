"""The between-duels screens (ygorl.md.menu), on synthetic screens built from the measured layout."""

import numpy as np

from ygorl.md import layout as L
from ygorl.md import menu

LIME, WHITE = (200, 255, 0), (230, 230, 230)


def blank():
    return np.full((L.REF_H, L.REF_W, 3), 30, np.uint8)


def label(img, xy):
    x, y = xy
    img[y - 6 : y + 6, x - 25 : x + 25] = LIME
    return img


def nodes(img, colours):
    for i, c in enumerate(colours):
        x0 = 77 + 260 * i
        img[menu.NODE_EDGE_Y - 1 : menu.NODE_EDGE_Y + 2, x0 : x0 + 170] = c
    return img


def play(img):
    x0, y0, x1, y1 = menu.PLAY_BOX
    img[y0:y1, x0:x1] = LIME
    return img


def test_screens_and_targets():
    v = label(blank(), menu.VICTORY_CONFIRM)
    assert menu.screen(v) == "victory" and menu.target(v, "victory") == menu.VICTORY_CONFIRM
    b = label(blank(), menu.BACK)
    assert menu.screen(b) == "back"
    d = play(blank())
    assert menu.screen(d) == "detail" and menu.target(d, "detail") == menu.PLAY
    s = nodes(blank(), [LIME, LIME, LIME])
    assert menu.screen(s) == "stages"
    x, y = menu.target(s, "stages")  # the current stage, below the edge row found
    assert x == 77 + 520 + 85 and abs(y - (menu.NODE_EDGE_Y + menu.NODE_CENTRE_DY)) <= 1
    s = nodes(blank(), [LIME, LIME, WHITE])
    assert [c for c, _ in menu.stage_nodes(s)] == ["lime", "lime", "white"]
    assert menu.target(s, "stages")[0] == 77 + 520 + 85  # the newly unlocked one
    assert menu.screen(blank()) is None


def test_walker_clicks_until_the_duel_runs():
    screens = [label(blank(), menu.VICTORY_CONFIRM), blank(), nodes(blank(), [LIME, WHITE]), play(blank())]
    clicks, state = [], {"n": 0}

    def capture():
        img = screens[min(state["n"], len(screens) - 1)]
        state["n"] += 1
        return img

    walker = menu.MenuWalker(lambda x, y: clicks.append((x, y)), capture, lambda: len(clicks) >= 3,
                             sleep=lambda s: None)  # fmt: skip
    assert walker.run(timeout=60, clock=iter(range(100)).__next__)
    assert [k for k, _ in walker.log] == ["victory", "stages", "detail"]  # the blank screen is left alone
    assert clicks[-1] == menu.PLAY


def test_turn_order_popup_picks_random():
    img = blank()
    for y in menu.TURN_ORDER_ROWS:
        label(img, (menu.VICTORY_CONFIRM[0], y))
    assert menu.screen(img) == "turn_order"
    assert menu.target(img, "turn_order") == (menu.VICTORY_CONFIRM[0], menu.TURN_ORDER_ROWS[0])
