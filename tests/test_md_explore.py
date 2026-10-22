"""Learning to operate Master Duel dialogs (design 10, 边角情况由 player 自己学会), offline: synthetic captures drawn
like MD's (black pills with lime labels, yellow rings and frames, the blue list tray, lime tabs) and a fake MD that
reacts to clicks."""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ygorl.md import explore as E
from ygorl.md import layout as L
from ygorl.md import ui as U
from ygorl.md.ui import UI

H, W = L.REF_H, L.REF_W
LIME, OLIVE, WHITE = (203, 255, 0), (82, 102, 0), (240, 240, 240)


# --- drawing ------------------------------------------------------------------------------------------------------
def board(value: int = 120) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def button(img: np.ndarray, centre, label: str = "enabled", size=(285, 62)) -> np.ndarray:
    """A pure black pill with a grey rim and a short label in the middle (lime, dim olive or white)."""
    x, y = centre
    w, h = size
    img[y - h // 2 - 2 : y + h // 2 + 2, x - w // 2 - 2 : x + w // 2 + 2] = (90, 90, 90)
    img[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2] = 0
    if label:
        colour = {"enabled": LIME, "disabled": OLIVE, "white": WHITE}[label]
        img[y - 10 : y + 10, x - 24 : x + 24] = colour
    return img


def frame(img: np.ndarray, centre, w: int = 157, h: int = 188, t: int = 8, colour=(253, 253, 64)) -> np.ndarray:
    """MD's yellow selection frame around a card."""
    x, y = centre
    x0, x1, y0, y1 = x - w // 2, x + w // 2, y - h // 2, y + h // 2
    img[y0:y1, x0:x1] = colour
    img[y0 + t : y1 - t, x0 + t : x1 - t] = (150, 60, 160)  # the card inside
    return img


def tick(img: np.ndarray, centre) -> np.ndarray:
    """A ticked card: a blue mark in its middle (not a colour any recogniser looks for)."""
    x, y = centre
    img[y - 15 : y + 15, x - 15 : x + 15] = (40, 90, 230)
    return img


def radios(img: np.ndarray, ys) -> np.ndarray:
    for y in ys:
        img[y - 21 : y + 22, 545:585] = (230, 220, 60)
    return img


def tabs(img: np.ndarray, n: int = 3, selected: int = 0) -> np.ndarray:
    """The effect dialog's tab bar: the selected tab filled lime, the others dark teal with a lime underline."""
    for i in range(n):
        x0 = 520 + 343 * i
        if i == selected:
            img[431:466, x0 : x0 + 320] = E.TAB_FILL
        else:
            img[431:466, x0 : x0 + 330] = (23, 52, 70)
            img[468:470, x0 : x0 + 330] = (153, 201, 24)
    return img


def card_list(img: np.ndarray, n: int) -> np.ndarray:
    """MD's card-list panel: n cards (158 px apart) above the flat blue tray, a dark strip below."""
    width = L.CARD_STEP * n - 20
    x0 = 1027 - width // 2
    img[1020:1070, x0 : x0 + width] = L.TRAY_RGB
    for i in range(n):
        cx = x0 + 69 + L.CARD_STEP * i
        img[820:990, cx - 60 : cx + 60] = (150, 110, 60)
    return img


def detect(img):
    return [(a.kind, a.index, a.label) for a in E.find_affordances(img)]


# --- affordances --------------------------------------------------------------------------------------------------
def test_buttons_with_their_label_colour_left_to_right():
    img = board()
    for xy, label in [((1180, 1117), "disabled"), ((870, 1117), "enabled"), ((1026, 840), "white")]:
        button(img, xy, label)
    found = E.find_buttons(img)
    assert [(b.centre, b.label, b.index) for b in found] == [
        ((1026, 840), "white", 0), ((870, 1117), "enabled", 1), ((1180, 1117), "disabled", 2)]  # fmt: skip
    assert found[0].bbox == (1026 - 142, 840 - 31, 1026 + 142, 840 + 31)


def test_black_boxes_without_label_or_of_another_shape_are_not_buttons():
    img = board()
    button(img, (1026, 600), label="")  # an empty black box (dark art)
    img[300:330, 700:1500] = 0  # a long black bar
    img[500:700, 1300:1400] = 0  # a tall black block
    button(img, (300, 500), "enabled")  # inside the card-info panel
    assert E.find_buttons(img) == []


def test_danger_zones_yield_no_affordance():
    img = board()
    button(img, (90, 90), "enabled", size=(150, 60))  # the settings gear's corner
    button(img, (1800, 1110), "enabled", size=(200, 60))  # Auto / info / log
    button(img, (250, 1110), "white", size=(200, 60))  # own portrait / LP
    frame(img, (1900, 60), w=150, h=115)  # opponent portrait
    assert E.find_affordances(img) == []
    assert E.is_dangerous((60, 80)) and E.is_dangerous((1700, 1120)) and not E.is_dangerous((1026, 840))


def test_radio_tabs_list_position():
    img = button(radios(board(), [413, 489, 566]), (1026, 840), "disabled")
    assert detect(img) == [("button", 0, "disabled"), ("radio", 0, ""), ("radio", 1, ""), ("radio", 2, "")]
    assert [a.centre for a in E.find_radios(img)] == [(U.RADIO_ROW_X, 413), (U.RADIO_ROW_X, 489), (U.RADIO_ROW_X, 566)]

    img = button(tabs(board(), 3, selected=1), (1026, 818), "enabled")
    found = E.find_tabs(img)
    assert [(t.index, t.label) for t in found] == [(0, ""), (1, "selected"), (2, "")]
    assert [t.centre[0] for t in found] == [685, 1023, 1371]

    img = button(card_list(board(), 4), (1026, 1117), "disabled")
    cards = E.find_list_cards(img)
    assert [c.centre for c in cards] == [(x, L.LIST_CARD_Y) for x, _ in L.list_cards(img)] and len(cards) == 4


def test_yellow_frames_are_highlights_but_filled_yellow_and_rings_are_not():
    img = frame(frame(board(), (650, 740)), (1232, 380))
    img[300:500, 900:1050] = (253, 253, 64)  # a lit, filled area
    d = np.hypot(*np.mgrid[-100:101, -100:101])
    ring = (d >= 86) & (d <= 94)
    img[761:962, 808:1009][ring] = (253, 253, 64)  # the phase menu's current-phase ring
    found = E.find_highlights(img)
    assert [a.centre for a in found] == [(1232, 380), (650, 740)]  # top down


def test_touching_zone_frames_are_found_one_by_one():
    """Free zones to place a card on: frames that share their sides merge into one blob."""
    img = board()
    xs = [647, 836, 1026, 1217, 1405]
    for x in xs:
        frame(img, (x, 737), w=200, h=200, t=12, colour=(205, 205, 80))
    assert [a.centre for a in E.find_highlights(img)] == [(x, 740) for x in L.ZONE_X]


def test_hand_glow_is_not_a_highlight():
    assert E.find_highlights(frame(board(), (1020, 1100))) == []


# --- signature ----------------------------------------------------------------------------------------------------
LIST_MEM = {"mode": 8, "accepting": True, "dialog_type": 0, "list_type": 3, "list_count": 4, "choices": 0}


def list_screen(n: int, ticked=(), lit: bool = False, art: int = 150) -> np.ndarray:
    img = button(card_list(board(), n), (1026, 1117), "enabled" if lit else "disabled")
    for i in ticked:
        tick(img, E.find_list_cards(img)[i].centre)
    img[100:250, 700:1300] = art  # board / animation content elsewhere
    return img


def test_signature_ignores_the_selection_and_the_cards_offered():
    a = E.signature(list_screen(4), LIST_MEM)
    assert a == E.signature(list_screen(4, ticked=[2], lit=True, art=60), LIST_MEM)
    assert a == E.signature(list_screen(5, art=200), {**LIST_MEM, "list_count": 5})  # a longer list of the same kind
    assert a.startswith("m8a1d0l3c2n0|list|")


def test_signature_tells_dialog_kinds_and_memory_states_apart():
    yesno = button(button(board(), (866, 658)), (1186, 658))
    radio = button(radios(board(), [413, 489]), (1026, 840), "disabled")
    prompt = button(button(card_list(board(), 1), (870, 1117), "enabled"), (1180, 1117), "disabled")
    effect = button(tabs(board()), (1026, 818))
    sigs = {E.signature(img, {"mode": 0}) for img in (yesno, radio, prompt, effect, list_screen(3), board())}
    assert len(sigs) == 6
    assert E.signature(prompt, {"mode": 4, "accepting": 1}) != E.signature(prompt, {"mode": 5, "accepting": 1})
    assert E.sig_class(E.signature(prompt, {})) == "prompt"
    assert E.memory_key({"mode": 0, "choices": [0x80000001, 0x80000002]}) == "m0a0d0l0c0n2"


# --- playbook -----------------------------------------------------------------------------------------------------
def test_playbook_is_seeded_persists_and_demotes(tmp_path):
    path = tmp_path / "pb" / "playbook.json"
    pb = E.Playbook(path)
    sig = E.signature(list_screen(3), LIST_MEM)
    assert pb.best(sig) == E.SEEDS["class:list"]  # nothing learnt: the seed of its class
    assert pb.recipes(sig)[0][0] == "seed"
    mine = [{"pick": "list_card"}, {"press": "button", "which": "rightmost"}]
    pb.record(sig, mine, True)
    again = E.Playbook(path)  # reloaded from disk
    assert again.best(sig) == mine and again.recipes(sig)[0][0] == "playbook"
    assert json.loads(path.read_text())["entries"][sig][0]["ok"] == 1
    again.record(sig, mine, False)
    again.record(sig, mine, False)  # 1 success, 2 failures: below the bar
    assert again.best(sig) == E.SEEDS["class:list"]
    # a recipe learnt under another memory state is offered for the same layout, before the seed
    other = E.signature(list_screen(3), {**LIST_MEM, "list_type": 7})
    pb2 = E.Playbook(None)
    pb2.record(sig, mine, True)
    assert pb2.recipes(other)[0] == ("layout", pb2.entries[sig][0])


def test_seeds_cover_the_hard_coded_scripts():
    assert {"class:yesno", "class:list", "class:prompt", "class:prompt#decline", "class:radio", "class:tabs",
            "class:position"} <= set(E.SEEDS)  # fmt: skip
    for recipe in E.SEEDS.values():  # relative to affordances, never coordinates
        assert all("xy" not in step for step in recipe)


# --- a fake MD -------------------------------------------------------------------------------------------------------
class FakeDialog:
    """A dialog drawn like MD's: option cards framed in yellow on the field (a click ticks / un-ticks one) and
    buttons with roles: "confirm" works only while exactly one card is ticked, "cancel" closes, "noop" does nothing.
    ``confirm_label`` is how the confirm button looks once usable (MD lights most in lime; this one stays white)."""

    def __init__(self, cards, buttons, confirm_label="white", danger_bait=False):
        self.cards, self.buttons, self.confirm_label = list(cards), list(buttons), confirm_label
        self.ticked: set[int] = set()
        self.closed: str | None = None  # "confirm 2" / "cancel"
        self.clicks: list[tuple[int, int]] = []
        self.danger_bait = danger_bait

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))
        if self.closed:
            return
        for i, (cx, cy) in enumerate(self.cards):
            if abs(x - cx) <= 75 and abs(y - cy) <= 90:
                self.ticked ^= {i}
                return
        for (bx, by), role in self.buttons:
            if abs(x - bx) <= 140 and abs(y - by) <= 30:
                if role == "cancel":
                    self.closed = "cancel"
                elif role == "confirm" and len(self.ticked) == 1:
                    self.closed = f"confirm {next(iter(self.ticked))}"
                return

    def capture(self) -> np.ndarray:
        img = board()
        if self.danger_bait:  # lit buttons where nothing may ever be clicked
            button(img, (90, 90), "enabled", size=(150, 60))
            button(img, (1800, 1110), "enabled", size=(200, 60))
        if self.closed:
            return img
        for i, xy in enumerate(self.cards):
            frame(img, xy)
            if i in self.ticked:
                tick(img, xy)
        for xy, role in self.buttons:
            label = {"cancel": "enabled", "noop": "white"}.get(role)
            if role == "confirm":
                label = self.confirm_label if len(self.ticked) == 1 else "disabled"
            button(img, xy, label)
        return img


CARDS = [L.my_zone("mzone", s) for s in (0, 2, 4)]
LEFT, RIGHT = (870, 1117), (1180, 1117)


def explorer(md, playbook):
    sleeps: list[float] = []
    ui = UI(md, md.capture, sleep=sleeps.append, echo=False)
    return E.Explorer(ui, playbook, settle=0.4, poll=0.4, enable_wait=0.8), ui


def test_explorer_learns_a_new_dialog_then_uses_the_playbook(tmp_path):
    """Pick a card, then a Confirm that never turns lime: the lime-button guess fails, the right button does
    nothing, each failed tick is undone, and Confirm (left) works. The second time it goes straight there."""
    path = tmp_path / "playbook.json"
    md = FakeDialog(CARDS, [(LEFT, "confirm"), (RIGHT, "noop")])
    ex, ui = explorer(md, E.Playbook(path))
    res = ex.run({"mode": 0}, progress=lambda: md.closed is not None, choice=1)
    assert res.ok and res.source == "explore" and md.closed == "confirm 1"
    assert res.recipe == [{"pick": "highlight"}, {"press": "button", "which": "index", "index": 0}]
    assert md.clicks == [CARDS[1], CARDS[1],  # tick, no lit button: un-tick
                         CARDS[1], RIGHT, CARDS[1],  # tick, the right button does nothing: un-tick
                         CARDS[1], LEFT]  # fmt: skip
    assert [c["why"].split(":")[0] for c in ui.take_clicks()][:2] == ["explore", "explore"]
    assert res.new_signature != res.signature and E.sig_class(res.new_signature) == "board"

    md2 = FakeDialog(CARDS, [(LEFT, "confirm"), (RIGHT, "noop")])
    ex2, _ = explorer(md2, E.Playbook(path))  # a later game: the playbook from disk
    res2 = ex2.run({"mode": 0}, progress=lambda: md2.closed is not None, choice=2)
    assert res2.ok and res2.source == "playbook" and md2.clicks == [CARDS[2], LEFT] and md2.closed == "confirm 2"
    assert [t["source"] for t in res2.tried] == ["playbook"]
    assert E.Playbook(path).entries[res.signature][0]["ok"] == 2


def test_a_failing_recipe_is_demoted_and_relearnt(tmp_path):
    path = tmp_path / "playbook.json"
    md = FakeDialog(CARDS, [(LEFT, "confirm"), (RIGHT, "noop")])
    pb = E.Playbook(path)
    explorer(md, pb)[0].run({}, progress=lambda: md.closed is not None)
    old = pb.best(E.signature(FakeDialog(CARDS, [(LEFT, "confirm"), (RIGHT, "noop")]).capture(), {}))
    # MD changed the dialog: same look, but now the right button confirms
    md = FakeDialog(CARDS, [(LEFT, "noop"), (RIGHT, "confirm")])
    ex, _ = explorer(md, pb)
    res = ex.run({}, progress=lambda: md.closed is not None)
    assert res.ok and res.source == "explore" and md.closed == "confirm 0"
    assert res.tried[0] == {"recipe": old, "source": "playbook", "ok": False}
    entries = {json.dumps(e["recipe"]): e for e in E.Playbook(path).entries[res.signature]}
    assert entries[json.dumps(old)]["fail"] == 1 and entries[json.dumps(res.recipe)]["ok"] == 1
    assert pb.best(res.signature) == res.recipe  # the new recipe now ranks first


def test_a_seed_answers_a_known_dialog_class_first(tmp_path):
    """MD's yes / no: the seed presses button #choice (0 = No on the left, 1 = Yes) without exploring."""
    md = FakeDialog([], [((866, 658), "cancel"), ((1186, 658), "cancel")])
    pb = E.Playbook(tmp_path / "pb.json")

    def capture():  # draw it like MD's yes / no: dark band behind two lit buttons
        img = md.capture()
        if not md.closed:
            img[520:600, 620:1430] = (20, 40, 60)
        return img

    ui = UI(md, capture, sleep=lambda s: None, echo=False)
    res = E.Explorer(ui, pb, settle=0.4).run({"mode": 0}, progress=lambda: md.closed is not None, choice=1)
    assert res.ok and res.source == "seed" and md.clicks == [(1186, 658)]
    assert E.sig_class(res.signature) == "yesno" and pb.entries[res.signature][0]["ok"] == 1


def test_declining_a_prompt_uses_cancel(tmp_path):
    md = FakeDialog([], [(LEFT, "cancel"), (RIGHT, "confirm")])

    def capture():
        img = md.capture()
        return img if md.closed else card_list(img, 1)

    ui = UI(md, capture, sleep=lambda s: None, echo=False)
    res = E.Explorer(ui, E.Playbook(None), settle=0.4).run({"mode": 5, "accepting": 1}, decline=True,
                                                            progress=lambda: md.closed is not None)  # fmt: skip
    assert res.ok and res.source == "seed" and md.closed == "cancel" and md.clicks == [LEFT]


def test_danger_zones_are_never_clicked_even_when_nothing_works():
    """A dialog nothing gets past, next to lit buttons in the danger zones: the budget runs out, no click lands in
    a danger zone, and a direct click there is refused."""
    md = FakeDialog(CARDS, [(LEFT, "noop"), (RIGHT, "noop")], danger_bait=True)
    ex, ui = explorer(md, E.Playbook(None))
    res = ex.run({}, progress=lambda: md.closed is not None)
    assert not res.ok and res.source == "none" and 0 < len(md.clicks) <= ex.max_clicks
    assert not any(E.is_dangerous(xy) for xy in md.clicks)
    with pytest.raises(U.Stuck, match="danger zone"):
        ex._click((60, 80), "the gear", res, {})


def test_screen_progress_without_a_memory_callback():
    md = FakeDialog(CARDS, [(LEFT, "confirm"), (RIGHT, "noop")], confirm_label="enabled")
    ex, _ = explorer(md, E.Playbook(None))
    res = ex.run({}, choice=0)  # progress: the signature changed (the dialog closed)
    assert res.ok and md.closed == "confirm 0"
    assert res.recipe == [{"pick": "highlight"}, {"press": "button", "which": "enabled"}]


def test_a_tick_noted_in_memory_is_not_mistaken_for_progress():
    md = FakeDialog(CARDS, [(LEFT, "confirm"), (RIGHT, "noop")], confirm_label="enabled")
    ex, _ = explorer(md, E.Playbook(None))
    res = ex.run({}, progress=lambda: bool(md.ticked) or md.closed is not None, choice=2)
    assert res.ok and md.closed == "confirm 2" and md.clicks == [CARDS[2], LEFT]


# --- real captures (only where the shared out/ directory is present) ------------------------------------------------
MDPLAY = Path(os.environ.get("YGORL_MDPLAY", Path(__file__).resolve().parents[1] / "out" / "mdplay"))
CAPTURES = sorted(glob.glob(str(MDPLAY / "stuck-*" / "screen.npy")))


@pytest.mark.skipif(not CAPTURES, reason="no Master Duel captures in out/mdplay")
def test_real_captures_one_layout_per_dialog_class():
    """On the stuck records of 2026-10-21/22: each dialog class has one visual key (the same dialog seen with
    other cards, other selections, mid-animation), and nothing is found in a danger zone."""
    keys: dict[str, set[str]] = {}
    for p in CAPTURES:
        img = np.load(p)
        affs = E.find_affordances(img)
        key = E.visual_key(img, affs)
        keys.setdefault(key.split("|")[0], set()).add(key)
        assert not any(E.is_dangerous(a.centre) for a in affs)
    assert all(len(v) == 1 for v in keys.values()), keys
    assert {"list", "prompt", "yesno", "radio", "tabs", "phase_menu", "highlight", "board"} <= set(keys)
