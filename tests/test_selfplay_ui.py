"""Solo player UI actuators and trace (design 10, Solo self-play), offline: a fake mouse and synthetic captures."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ygorl.md import layout as L
from ygorl.md.memory import read_duel
from ygorl.md.ui import UI, Stuck, command_order
from ygorl.selfplay import trace as T
from tests.test_md import ASH, BASE, BEWD, DM, OFF, PASSWORDS, SEM, build_memory

H, W = L.REF_H, L.REF_W


class FakeMouse:
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))


def make_ui(*frames: np.ndarray):
    """A UI over a fake mouse; ``capture`` returns ``frames`` in turn, then repeats the last one."""
    mouse, sleeps, seq = FakeMouse(), [], list(frames)

    def capture() -> np.ndarray:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return UI(mouse, capture, sleep=sleeps.append, echo=False), mouse, sleeps


def board(value: int = 180) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


YY, XX = np.mgrid[0:H, 0:W]


GLYPHS = {"summon": (21, (36, 128, 251)), "set": (16, (36, 128, 251)), "activate": (20, (190, 220, 60))}


def with_icons(img: np.ndarray, centres, kinds=None) -> np.ndarray:
    """MD command icons: dark discs (radius 44) with a glyph whose size and colour tell the command (blue on about
    half the inner disc for Summon, a quarter for Set, yellow-green for Activate). Default: Summon, then Set."""
    img = img.copy()
    kinds = kinds or ["summon"] + ["set"] * (len(centres) - 1)
    for (cx, cy), kind in zip(centres, kinds, strict=True):
        d = np.hypot(XX - cx, YY - cy)
        img[d <= 44] = (15, 20, 25)
        radius, colour = GLYPHS[kind]
        img[d <= radius] = colour
    return img


def with_menu(img: np.ndarray, buttons) -> np.ndarray:
    """The phase menu: a light ring of radius 90 at each button's centre."""
    img = img.copy()
    for b in buttons:
        cx, cy = L.PHASE_MENU[b]
        d = np.hypot(XX - cx, YY - cy)
        img[(d >= 86) & (d <= 94)] = (220, 220, 220)
    return img


def near(found, want, tol=8) -> bool:
    return len(found) == len(want) and all(
        abs(a - x) <= tol and abs(b - y) <= tol for (a, b), (x, y) in zip(found, want)
    )


# --- phase menu ---------------------------------------------------------------------------------------------------
def test_phase_menu_that_never_opens_is_retried_once_then_stuck():
    ui, mouse, sleeps = make_ui(board(30))
    with pytest.raises(Stuck, match="did not open"):
        ui.phase("battle")
    assert mouse.clicks == [L.PHASE_BADGE, L.PHASE_BADGE]  # never the End button's spot (the Graveyard)
    assert sleeps == [1.2] + [0.3] * 8 + [1.2] + [0.3] * 8  # polled while the menu could still fade in
    assert [c["why"] for c in ui.take_clicks()] == ["phase menu", "phase menu (again)"] and ui.clicks == []


def test_phase_menu_fading_in_is_waited_for():
    """The menu shows a few captures after the badge click: no second click (that would toggle it shut)."""
    menu = with_menu(board(30), ["battle", "main2", "end"])
    ui, mouse, sleeps = make_ui(board(30), board(30), board(30), menu)
    ui.phase("main2")
    assert mouse.clicks == [L.PHASE_BADGE, L.PHASE_MENU["main2"]]


def test_phase_menu_already_open_is_used_as_it_is():
    ui, mouse, _ = make_ui(with_menu(board(30), ["battle", "main2", "end"]))
    ui.phase("end")
    assert mouse.clicks == [L.PHASE_MENU["end"]]


def test_phase_menu_without_the_wanted_button():
    ui, mouse, _ = make_ui(board(30), with_menu(board(30), ["main2", "end"]))
    with pytest.raises(Stuck, match="no battle"):
        ui.phase("battle")
    assert mouse.clicks == [L.PHASE_BADGE]


# --- hand card commands -------------------------------------------------------------------------------------------
HAND_XY = L.hand_card(2, 5)
ROW = [(HAND_XY[0] - 90, 880), (HAND_XY[0] + 90, 880)]  # Summon, Set above the lifted card


def test_command_order():
    assert command_order(True, {"mset", "summon"}) == ["summon", "mset"]
    assert command_order(False, {"sset", "activate"}) == ["activate", "sset"]
    assert command_order(True, {"sset"}) == []


@pytest.mark.parametrize("kind,index", [("summon", 0), ("mset", 1)])
def test_hand_card_command_chosen_by_its_glyph(kind, index):
    opened = with_icons(board(), ROW)
    ui, mouse, sleeps = make_ui(board(), opened, opened)
    icons = ui.open_hand_card(HAND_XY, "hand Blue-Eyes")
    assert near(icons, ROW)
    target = ui.choose_command(icons, kind, command_order(True, {"summon", "mset"}))
    assert target == icons[index]
    assert mouse.clicks == [HAND_XY, icons[index]] and sleeps == [0.6, 1.5]
    ui.pick_zone("mzone", 2, "zone")
    assert mouse.clicks[-1] == L.my_zone("mzone", 2) and sleeps[-1] == 2.5


def test_hand_card_icons_that_come_late_are_polled_for():
    """The icons appear a few captures after the click: found by polling, no second click."""
    ui, mouse, sleeps = make_ui(board(), board(), board(), with_icons(board(), ROW))
    assert near(ui.open_hand_card(HAND_XY, "hand"), ROW)
    assert mouse.clicks == [HAND_XY] and sleeps == [0.6, 0.3, 0.3]


def test_hand_card_clicked_again_when_nothing_comes():
    """Nothing within the polling window (an animation swallowed the click): click again."""
    frames = [board()] * 12 + [with_icons(board(), ROW)]
    ui, mouse, _ = make_ui(*frames)
    assert near(ui.open_hand_card(HAND_XY, "hand"), ROW)
    assert mouse.clicks == [HAND_XY, HAND_XY]
    assert [c["why"] for c in ui.clicks] == ["hand", "hand (again)"]


def test_hand_card_left_open_keeps_its_icons():
    """Icons already there before both clicks: none are new, the ones above y 940 are taken."""
    opened = with_icons(board(), ROW)
    ui, _, _ = make_ui(opened)
    assert near(ui.open_hand_card(HAND_XY, "hand"), ROW)


def test_choose_command_set_only_and_refusals():
    order = ["summon", "mset"]
    set_only = with_icons(board(), [ROW[0]], ["set"])
    ui, mouse, _ = make_ui(set_only)
    assert ui.choose_command([ROW[0]], "mset", order) == ROW[0]  # MD offers only Set
    assert ui.choose_command([], "summon", order) is None  # no command at all: refused
    assert ui.choose_command([ROW[0]], "summon", order) is None  # no Summon among them: refused, nothing clicked
    assert mouse.clicks == [ROW[0]]
    with pytest.raises(Stuck, match="not among"):
        ui.choose_command(ROW, "activate", order)


def test_field_command_takes_the_icon_above_the_card():
    xy = (1038, 750)
    ui, mouse, sleeps = make_ui(board(), with_icons(board(), [(xy[0], xy[1] - 145)]))
    assert ui.attack(xy, "attacker")
    assert mouse.clicks[0] == xy and abs(mouse.clicks[1][0] - xy[0]) <= 8 and sleeps == [1.2, 1.8]
    ui2, mouse2, _ = make_ui(board())
    assert not ui2.pick_target(xy, "target") and mouse2.clicks == [xy]  # one click, no retry on the field


# --- lists, prompts, dialogs --------------------------------------------------------------------------------------
def list_panel(n: int) -> tuple[np.ndarray, list[int]]:
    img = np.full((H, W, 3), 15, np.uint8)  # the panel's dark background (also under the Select button)
    img[:800] = 180  # the board above the panel
    width = L.CARD_STEP * n - 20
    left = 1026 - width // 2
    img[820:1060, left : left + width] = L.TRAY_RGB
    centres = [round(left + (width - (n - 1) * L.CARD_STEP) / 2 + i * L.CARD_STEP) for i in range(n)]
    for x in centres:
        img[890:1030, x - 60 : x + 60] = (200, 160, 90)
    return img, centres


def test_list_selection_by_index():
    panel, centres = list_panel(3)
    ui, mouse, sleeps = make_ui(panel, board())  # the panel closes after Select
    assert ui.pick_list_cards([1]) == [(x, 960) for x in centres]
    assert mouse.clicks == [(centres[1], 960), (1026, 1116)] and sleeps == [0.8, 1.0]
    assert ui.clicks[0]["why"] == "#2 of 3 cards in the list"


def test_list_with_two_buttons_and_empty_list():
    panel, centres = list_panel(2)
    ui, mouse, _ = make_ui(panel)  # still a list after Select: the right button
    ui.pick_list_cards([0])
    assert mouse.clicks == [(centres[0], 960), (1026, 1116), (1180, 1117)]
    ui, _, _ = make_ui(board())
    with pytest.raises(Stuck, match="no card found"):
        ui.pick_list_cards([0])
    ui, _, _ = make_ui(panel)
    with pytest.raises(Stuck, match="2 cards in the list"):
        ui.pick_list_cards([2])


def test_prompt_and_dialog_buttons():
    ui, mouse, sleeps = make_ui(board())
    ui.respond_prompt(True)
    ui.respond_prompt(False)
    ui.pick_position()
    ui.yesno(False)
    ui.continue_draw()
    assert mouse.clicks == [L.PROMPT_CARD, L.PROMPT_ACTIVATE, L.PROMPT_CANCEL, (940, 956), (866, 658), L.PHASE_BADGE]
    assert sleeps == [0.8, 2.0, 1.5, 1.5, 1.5, 1.5]
    ui.yesno(True)
    assert mouse.clicks[-1] == (1186, 658)


def test_player_decisions_drive_the_ui(tmp_path):
    """The loop decides (first free zone, pass outside the battle) and only calls the UI layer."""
    from ygorl.selfplay.loop import Player

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    player = Player.__new__(Player)
    player.ui, mouse, sleeps = make_ui(board())
    player.trace = T.Trace(tmp_path)
    player.context = ""
    player.trace.begin(0, raw, SimpleNamespace(mode=4, accepting=1, player=0), "position")
    player.dialog(raw, "position")
    player.prompt(raw, SimpleNamespace(mode=4, accepting=1, player=0))  # own Main Phase 1: pass
    assert mouse.clicks == [(940, 956), L.my_zone("mzone", 0), L.PROMPT_CANCEL] and sleeps == [1.5, 2.5, 1.5]
    assert player.trace.rec["decision"] == {"by": "rule", "rule": "prompt-pass"}


# --- trace --------------------------------------------------------------------------------------------------------
def test_trace_round_trip_and_summary(tmp_path, capsys):
    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    inp = SimpleNamespace(mode=2, accepting=1, player=0)
    card = SimpleNamespace(code=PASSWORDS[BEWD])
    summon = SimpleNamespace(kind="summon", card=card)
    tr = T.Trace(tmp_path, run="r1", screen_every=2)
    tr.start(env="md-2026-09")

    tr.begin(0, raw, inp, None)
    assert tr.wants_screen(0) and not tr.wants_screen(1)
    tr.save_screen(board())
    tr.policy(summon, 0.6, "main1", [SimpleNamespace(action=summon, prob=0.6)])
    tr.observe(icons=[(np.int64(930), 880)])
    tr.refusal(summon, "MD refused: Blue-Eyes shows no command")
    tr.end_step([{"t": 0, "xy": [1020, 1100], "why": "hand"}])

    tr.begin(1, raw, inp, None)
    tr.policy(summon, 0.5, "main1")
    tr.end_step([{"t": 0, "xy": [1, 2], "why": "summon"}, {"t": 0, "xy": [3, 4], "why": "zone"}])

    tr.begin(2, replace(raw, turn=4), inp, None)
    tr.check(summon, None)
    tr.rule("prompt-pass")
    tr.end_step()

    tr.begin(3, replace(raw, turn=4), inp, "yesno")
    tr.check(SimpleNamespace(kind="attack", card=None), "attack: nothing changed")
    tr.stuck("the game does not show the policy's action")
    tr.close(exit=3)

    recs = list(T.read(tr.path))
    assert [r["kind"] for r in recs] == ["start", "step", "step", "step", "step", "end"]
    first = recs[1]
    assert first["turn"] == 4 and first["phase"] == "main1" and first["turn_player"] == "me"
    assert first["input"] == {"mode": 2, "accepting": True, "player": 0}
    assert first["decision"]["top"] == [["summon", PASSWORDS[BEWD], 0.6]]
    assert first["observed"] == {"icons": [[930, 880]]}
    assert (tmp_path / first["screen"]).is_file()
    assert np.load(tmp_path / first["screen"])["screen"].shape == (H // 2, W // 2, 3)
    mem = first["memory"]
    assert mem["me"]["hand"] == [BEWD, ASH] and mem["me"]["lp"] == 8000
    # the opponent: no hand cards, face-down cards hidden, face-up ones shown
    assert "hand" not in mem["opp"] and mem["opp"]["counts"]["hand"] == 4
    assert mem["opp"]["field"][0][0] == BEWD and mem["opp"]["field"][1][0] == 0 and mem["opp"]["field"][8][0] == 0
    assert DM not in [s[0] for s in mem["opp"]["field"] if s]
    assert recs[2]["clicks"][1]["why"] == "zone"

    s = T.summarize(recs)
    assert s["steps"] == 4 and s["clicks"] == 3 and s["stuck"] == 1
    assert s["decisions"] == {"policy:summon": 2, "rule:prompt-pass": 1}
    assert s["checks"] == {"ok": 1, "failed": 1} and s["failed_checks"] == {"attack": 1}
    assert s["refusals"] == {"MD refused: Blue-Eyes shows no command": 1}
    assert s["steps_per_turn"] == {4: 2, 5: 2} and s["end"]["exit"] == 3

    assert T.main(["SUMMARY", str(tr.path)]) == 0
    out = capsys.readouterr().out
    assert "steps 4, clicks 3, stuck 1" in out and "checks: ok 1, failed 1" in out and "4:2, 5:2" in out


def test_activate_icon_is_told_by_its_glyph_colour():
    """A monster with a hand effect: MD shows Summon, Activate and Set; Activate is found by colour, whatever its
    place in the row (King of the Swamp's Activate was clicked for its Summon by position)."""
    from ygorl.md.ui import is_activate_icon

    from ygorl.md.ui import icon_kind

    row = [(880, 880), (1050, 880), (1220, 880)]
    img = with_icons(board(), row, ["activate", "summon", "set"])
    assert [is_activate_icon(img, *p) for p in row] == [True, False, False]
    assert [icon_kind(img, *p) for p in row] == ["activate", "summon", "set"]
    ui, mouse, _ = make_ui(img, img, img)
    assert ui.choose_command(row, "summon", ["summon", "activate", "mset"]) == row[1]
    assert ui.choose_command(row, "activate", ["summon", "activate", "mset"]) == row[0]
    assert ui.choose_command(row, "mset", ["summon", "activate", "mset"]) == row[2]


def test_recovery_escalates_per_state_and_gives_up(tmp_path):
    """Self-recovery: level 1 clicks an empty spot, level 2 answers a shown list differently; past the limit it
    gives up. Every attempt is recorded."""
    from ygorl.selfplay.loop import NEUTRAL_SPOT, Input, Player

    player = Player.__new__(Player)
    ui, mouse, _ = make_ui(*([list_panel(3)[0]] * 40))
    player.ui = ui
    player.recoveries, player.recovery_count, player.forbidden, player.recent = {}, 0, set(), []
    player.expect = None
    player.record_stuck = lambda why: tmp_path / "stuck"
    player.trace = type("T", (), {"observe": lambda self, **k: None, "rule": lambda self, *a, **k: None})()
    raw = read_duel(build_memory(), BASE, OFF)
    inp = Input(0, False, 0)
    state = (0, "list", raw.phase, "board", 0)
    assert player.recover("first", raw, inp, "list", state)
    assert mouse.clicks[0] == NEUTRAL_SPOT
    assert player.recover("second", raw, inp, "list", state[:4] + (1,))  # same state despite a new refusal
    assert player.recoveries[state[:4]] == 2
    for _ in range(2):
        assert player.recover("again", raw, inp, "list", state)
    assert not player.recover("too many", raw, inp, "list", state)


def _panel(name_width: int, numbers: tuple[str, str, str]) -> np.ndarray:
    """A synthetic information panel: dark body, a white name bar, digits drawn as distinct block shapes."""
    from ygorl.md import panel as P

    img = np.full((1184, 2054, 3), 180, np.uint8)
    img[160:420, 30:450] = (8, 12, 14)
    x0, y0, _, y1 = P.NAME_BAND
    img[y0 + 4 : y1 - 4, x0 : x0 + name_width] = 255
    shapes = {"0": [(0, 0, 14, 4), (0, 18, 14, 22), (0, 0, 3, 22), (11, 0, 14, 22)], "1": [(6, 0, 9, 22)],
              "2": [(0, 0, 14, 4), (0, 9, 14, 13), (0, 18, 14, 22), (11, 0, 14, 11), (0, 11, 3, 22)],
              "4": [(0, 0, 3, 12), (0, 9, 14, 13), (11, 0, 14, 22)], "5": [(0, 0, 14, 4), (0, 9, 14, 13),
              (0, 18, 14, 22), (0, 0, 3, 11), (11, 11, 14, 22)]}  # fmt: skip
    for band, text in zip((P.LEVEL_BAND, P.ATK_BAND, P.DEF_BAND), numbers, strict=True):
        bx, by = band[0] + 6, band[1] + 5
        for i, ch in enumerate(text):
            for a, b, c, d in shapes[ch]:
                img[by + b : by + d, bx + 20 * i + a : bx + 20 * i + c] = 255
    return img


def test_panel_book_learns_names_and_digits_then_tells_cards_apart():
    from ygorl.md.panel import PanelBook

    stats = {1: (5, 2000, 1400), 2: (4, 1500, 1000), 3: (4, 1200, 1000)}
    book = PanelBook()
    a = _panel(120, ("5", "2000", "1400"))
    assert book.identify(a, [1, 2, 3], stats) == (None, "unknown")  # nothing learned yet
    book.learn(a, 1, stats[1])
    assert book.identify(a, [1, 2, 3], stats) == (1, "name")
    b = _panel(200, ("4", "1500", "1000"))  # an unseen name: told apart by its digits (5 -> card 2 only)
    assert book.identify(b, [1, 2, 3], stats) == (2, "stats")
    closed = board()
    assert book.identify(closed, [1, 2], stats) == (None, "closed")
