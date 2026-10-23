"""Solo player UI actuators and trace (design 10, Solo self-play), offline: a fake mouse and synthetic captures."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ygorl.md import layout as L
from ygorl.md.memory import read_duel
from ygorl.md.ui import POSITION_FACEUP_DEFENSE, UI, Stuck, command_order
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
def test_phase_menu_that_never_opens_is_retried_then_stuck():
    ui, mouse, sleeps = make_ui(board(30))
    with pytest.raises(Stuck, match="did not open"):
        ui.phase("battle")
    assert mouse.clicks == [L.PHASE_BADGE] * 3  # never the End button's spot (the Graveyard)
    assert sleeps == ([1.2] + [0.3] * 16) * 3  # polled while the menu could still fade in
    assert [c["why"] for c in ui.take_clicks()] == ["phase menu", "phase menu (attempt 2)", "phase menu (attempt 3)"]
    assert ui.clicks == [] and len(ui.evidence) == 9  # the frames seen instead, for the stuck record


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
    assert mouse.clicks == [HAND_XY, icons[index]] and sleeps[0] == 0.6 and sleeps[-1] == 1.5  # (+ fade-in wait)
    ui.pick_zone("mzone", 2, "zone")
    assert mouse.clicks[-1] == L.my_zone("mzone", 2) and sleeps[-1] == 2.5


def test_hand_card_icons_that_come_late_are_polled_for():
    """The icons appear a few captures after the click: found by polling, no second click."""
    ui, mouse, sleeps = make_ui(board(), board(), board(), with_icons(board(), ROW))
    assert near(ui.open_hand_card(HAND_XY, "hand"), ROW)
    assert mouse.clicks == [HAND_XY] and sleeps[:3] == [0.6, 0.3, 0.3]


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
    assert mouse.clicks[0] == xy and abs(mouse.clicks[1][0] - xy[0]) <= 8 and sleeps[0] == 1.2 and sleeps[-1] == 1.8
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


def test_declare_runs_the_search_and_picks_from_the_grey_tray():
    """Typing alone lists nothing: Return runs the search (in a box focused by a second click); its results sit on
    a grey tray."""
    from ygorl.md.ui import DECLARE_SEARCH

    panel, centres = list_panel(1)
    panel[820:1060][np.all(panel[820:1060] == L.TRAY_RGB, axis=2)] = L.DECLARE_TRAY
    ui, mouse, _ = make_ui(panel)
    typed, entered = [], []
    assert ui.declare("Dark Magician", typed.append, enter=lambda: entered.append(1)) == [(centres[0], 905)]
    assert typed == ["Dark Magician"] and entered == [1]
    assert mouse.clicks[:2] == [DECLARE_SEARCH, DECLARE_SEARCH]


def test_list_selection_by_index():
    panel, centres = list_panel(3)
    ui, mouse, sleeps = make_ui(panel, panel, panel, board())  # the panel closes after Select
    assert ui.pick_list_cards([1]) == [(x, 905) for x in centres]
    # 0.3: a second capture to see the cards have stopped sliding in
    assert mouse.clicks == [(centres[1], 905), (1026, 1116)] and sleeps == [0.3, 0.8, 1.0]
    assert ui.clicks[0]["why"] == "#2 of 3 cards in the list"


def test_list_with_two_buttons_and_empty_list():
    panel, centres = list_panel(2)
    ui, mouse, _ = make_ui(panel)  # still a list after Select: the right button
    ui.pick_list_cards([0])
    assert mouse.clicks == [(centres[0], 905), (1026, 1116), (1180, 1117)]
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
    assert sleeps[:4] == [0.8, 2.0, 1.5, 1.5] and sleeps[-1] == 1.5  # yesno settles first (polls in between)
    ui.yesno(True)
    assert mouse.clicks[-1] == (1186, 658)


def test_player_decisions_drive_the_ui(tmp_path):
    """The loop decides (first free zone, pass outside the battle) and only calls the UI layer."""
    from ygorl.selfplay.loop import Player

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    player = Player.__new__(Player)
    player.ui, mouse, sleeps = make_ui(board())
    player.trace = T.Trace(tmp_path)
    player.context, player.responded = "", None
    player.trace.begin(0, raw, SimpleNamespace(mode=4, accepting=1, player=0), "position")
    player.dialog(raw, "position")
    player.prompt(raw, SimpleNamespace(mode=4, accepting=1, player=0))  # own Main Phase 1: pass
    assert mouse.clicks == [(940, 956), L.my_zone("mzone", 0), L.PROMPT_CANCEL] and sleeps == [1.5, 2.5, 1.5]
    assert player.trace.rec["decision"] == {"by": "rule", "rule": "prompt-pass"}


def test_rebuilt_decisions_reach_the_ui(tmp_path):
    """Without a shadow the position and the prompt come from the rebuild (design 10 「识别与执行的系统化」): the
    policy's face-up Defense and its activation in the opponent's turn, recorded as {"by": "rebuild"}."""
    from ygorl.selfplay.decide import Choice
    from ygorl.selfplay.loop import PHASES, Player

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    player = Player.__new__(Player)
    player.ui, mouse, _ = make_ui(board())
    player.trace = T.Trace(tmp_path)
    player.context, player.responded = "", None
    player.request_now = lambda raw, list_shown=False: SimpleNamespace(params=(), candidates=())
    asked = []

    class Decider:
        def position(self, *args):
            asked.append(("position", args[2:4]))
            return Choice("position", pick=1, probs={0: 0.2, 1: 0.8}, reached=2, samples=2)

        def respond(self, raw, req):
            asked.append(("respond", raw.turn_player))
            return Choice("respond", pick=0, probs={0: 0.7, 1: 0.3}, reached=1, samples=2)

    player.decider = Decider()
    player.trace.begin(0, raw, SimpleNamespace(mode=0, accepting=0, player=0), "position")
    player.dialog(raw, "position")
    assert mouse.clicks == [POSITION_FACEUP_DEFENSE, L.my_zone("mzone", 0)]  # the Defense button, then the zone
    assert player.trace.rec["decision"]["by"] == "rebuild" and player.trace.rec["decision"]["pick"] == 1
    theirs = replace(raw, phase=PHASES.index("main1"), turn_player=1 - raw.me)
    inp = SimpleNamespace(mode=5, accepting=1, player=0)
    player.trace.begin(1, theirs, inp, None)
    player.prompt(theirs, inp)
    assert mouse.clicks[2:] == [L.PROMPT_CARD, L.PROMPT_ACTIVATE]
    assert player.trace.rec["decision"]["question"] == "respond"
    assert asked == [("position", (None, None)), ("respond", 1 - raw.me)]


def test_end_phase_prompt_passes_when_the_response_did_not_take(tmp_path):
    """Our End Phase: respond to a chain prompt once; the same prompt again with nothing of ours changed (MD greyed
    the Activate button) is passed."""
    from ygorl.selfplay.loop import PHASES, Player

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    raw = replace(raw, phase=PHASES.index("end"), turn_player=raw.me)
    player = Player.__new__(Player)
    player.ui, mouse, _ = make_ui(board())
    player.trace = T.Trace(tmp_path)
    player.context, player.responded = "", None
    inp = SimpleNamespace(mode=5, accepting=1, player=0)
    player.trace.begin(0, raw, inp, None)
    player.prompt(raw, inp)
    player.prompt(raw, inp)
    assert mouse.clicks == [L.PROMPT_CARD, L.PROMPT_ACTIVATE, L.PROMPT_CANCEL]


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
    state = (raw.turn, 0, "list", raw.phase, "board", 0)
    assert player.recover("first", raw, inp, "list", state)
    assert mouse.clicks[0] == NEUTRAL_SPOT
    assert player.recover("second", raw, inp, "list", state[:-1] + (1,))  # same state despite a new refusal
    assert player.recoveries[state[:-1]] == 2
    for _ in range(2):
        assert player.recover("again", raw, inp, "list", state)
    assert not player.recover("too many", raw, inp, "list", state)


def _panel(name_width: int, numbers: tuple[str, str, str]) -> np.ndarray:
    """A synthetic information panel: dark body, a white name bar, digits drawn as distinct block shapes."""
    from ygorl.md import panel as P

    img = np.full((1184, 2054, 3), 180, np.uint8)
    img[160:420, 30:450] = (8, 12, 14)
    img[P.RIM_Y, 30:450] = P.RIM_RGB  # the panel's grey top rim (Chinese layout)
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


def test_list_select_state_reads_the_label_colour():
    from ygorl.md.ui import LIST_SELECT_TEXT, list_select_state

    img = np.zeros((L.REF_H, L.REF_W, 3), np.uint8)
    assert list_select_state(img) == ""
    x0, y0, x1, y1 = LIST_SELECT_TEXT
    img[y0 + 10 : y0 + 20, x0 + 30 : x1 - 30] = (82, 102, 0)  # nothing ticked: dim olive label
    assert list_select_state(img) == "disabled"
    img[y0 + 10 : y0 + 20, x0 + 30 : x1 - 30] = (203, 255, 0)  # ticked: bright label
    assert list_select_state(img) == "enabled"


def test_radio_dialog_rows_and_pick():
    """MD's radio-button dialog (Pot of Extravagance: banish 3 or 6): rows from the yellow rings, first row then
    Confirm."""
    from ygorl.md.ui import RADIO_CONFIRM, RADIO_ROW_X, radio_rows, screen_dialog

    img = np.full((L.REF_H, L.REF_W, 3), 20, np.uint8)
    for cy in (413, 489):
        img[cy - 21 : cy + 22, 545:585] = (230, 220, 60)
    assert radio_rows(img) == [413, 489]
    assert screen_dialog(img) != "radio"  # rings without the dialog's Confirm label: a passing animation
    img[832:850, 1005:1047] = (82, 102, 0)  # 确认, dim until a row is picked
    assert screen_dialog(img) == "radio"
    ui, mouse, _ = make_ui(img)
    assert ui.pick_radio(0) == 2
    assert mouse.clicks == [(RADIO_ROW_X, 413), RADIO_CONFIRM]


def test_yesno_needs_its_button_labels():
    """A dark band alone is a dimmed or dark board (the lava stage) or the effect choice: "panel"; yes / no shows
    lime labels on both buttons."""
    from ygorl.md.ui import YESNO_NO, YESNO_YES, screen_dialog

    img = np.full((L.REF_H, L.REF_W, 3), 40, np.uint8)
    assert screen_dialog(img) == "panel"
    for x, y in (YESNO_NO, YESNO_YES):
        img[y - 8 : y + 8, x - 30 : x + 30] = (203, 255, 0)
    img[640:676, 840:900] = (203, 255, 0)  # zone outlines: lime around the spots, no black pill
    assert screen_dialog(img) == "panel"  # lime spots that are no buttons: zone outlines on a dark board
    for x, y in (YESNO_NO, YESNO_YES):  # the buttons: black pills (measured 300x58) with the lime labels
        img[y - 29 : y + 29, x - 150 : x + 150] = 0
        img[y - 7 : y + 8, x - 16 : x + 17] = (203, 255, 0)
    assert screen_dialog(img) == "yesno"


def _banner(img: np.ndarray, top: int) -> None:
    """MD's banner: a navy band across the board, lighter behind its white text."""
    img[top : top + 100, 466:1590] = (12, 40, 56)
    img[top + 15 : top + 75, 900:1150] = (32, 68, 84)
    img[top + 45 : top + 60, 950:1100] = 255


def test_banner_over_a_dark_board():
    """On the lava stage the board's middle row reads as a dark band: a top banner over it is still a banner."""
    from ygorl.md.ui import screen_dialog

    img = np.full((L.REF_H, L.REF_W, 3), 40, np.uint8)
    assert screen_dialog(img) == "panel"
    _banner(img, 255)
    assert screen_dialog(img) == "banner"


def test_banner_at_the_top_or_the_bottom_of_the_board():
    from ygorl.md.ui import screen_dialog

    board = np.full((L.REF_H, L.REF_W, 3), 110, np.uint8)
    assert screen_dialog(board) is None
    img = board.copy()
    img[270:300, 900:1150] = (32, 68, 84)  # navy and white at the top centre only: a monster's art and ATK digits
    img[300:315, 950:1100] = 255
    assert screen_dialog(img) is None
    for top in (255, 855):  # MD drops the banner low when the targets are in the opponent's rows
        img = board.copy()
        _banner(img, top)
        assert screen_dialog(img) == "banner"


def test_zone_outlined_finds_the_offered_spell_zone():
    from ygorl.md.ui import zone_outlined

    img = np.full((L.REF_H, L.REF_W, 3), 40, np.uint8)
    x, y = L.my_zone("szone", 2)
    assert not zone_outlined(img, x, y)
    img[y - 85 : y + 85, x - 95 : x + 95] = (130, 140, 40)  # the dim yellow-green outline (and fill)
    img[y - 70 : y + 70, x - 80 : x + 80] = 40
    assert zone_outlined(img, x, y) and not zone_outlined(img, *L.my_zone("szone", 0))


def test_new_icons_fall_back_to_what_appeared_on_a_dark_board():
    """On the dark lava board the Confirm icon's rim has no light ground around it: the circle test misses it, the
    patch that appeared with the click is taken instead."""
    before = np.full((L.REF_H, L.REF_W, 3), 30, np.uint8)
    after = before.copy()
    yy, xx = np.mgrid[-44:45, -44:45]
    disc = yy**2 + xx**2 <= 44**2
    after[250 - 44 : 250 + 45, 1030 - 44 : 1030 + 45][disc] = (40, 120, 230)
    assert L.find_icons(after, 600, 100, 1500, 500) == []
    assert L.new_icons(before, after, 600, 100, 1500, 500) == [(1030, 250)]


def test_blue_confirm_icon_found_by_colour():
    img = np.full((L.REF_H, L.REF_W, 3), 20, np.uint8)
    yy, xx = np.mgrid[-36:37, -36:37]
    ring = (yy**2 + xx**2 <= 36**2) & (yy**2 + xx**2 >= 28**2)
    img[221 - 36 : 221 + 37, 1027 - 36 : 1027 + 37][ring] = (40, 120, 230)
    assert L.blue_icons(img, 908, 125, 1168, 325) == [(1027, 221)]


def test_special_summon_icon_is_the_one_matching_no_other_command():
    """Apprentice Illusion Magician offered only Special Summon, whose glyph reads as Activate; Activate is not one
    of its legal commands, so that icon is taken."""
    icons = [(1020, 878)]
    img = with_icons(board(), icons, ["activate"])
    ui, mouse, _ = make_ui(img)
    assert ui.choose_command(icons, "spsummon", ["spsummon", "mset"], "Apprentice") == icons[0]
    ui, mouse, _ = make_ui(img)
    with pytest.raises(Stuck):
        ui.choose_command(icons, "spsummon", ["spsummon", "activate"], "x")  # could be either


def test_unknown_panel_is_not_taken_for_one_of_two_unlearned_spells():
    """Fist of Fate was Set for Millennium Revelation: both spells, neither learned. An unknown panel then names
    neither, and with two unknown positions elimination cannot decide either: the player declines (Stuck)."""
    from ygorl.selfplay.loop import Player

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    hand = [c.cid for c in raw.players[raw.me].piles["hand"]]
    player = Player.__new__(Player)
    player.ui, mouse, _ = make_ui(board())
    player.ui.open_hand_card = lambda xy, why: [(xy[0], 880)]
    player.trace = type("T", (), {"observe": lambda self, **k: None})()
    player.panel = type("P", (), {"names": {}, "identify": lambda self, img, h, s: (None, "unknown")})()
    player.selection = NoSelection()
    player.card_stats = lambda c: None
    player.card_kind = lambda c: "spell"
    player.name = str
    with pytest.raises(Stuck, match="not found"):
        player.verify_hand_card(raw, hand[0], L.hand_card(0, len(hand)), [(0, 0)])


class NoSelection:
    """Memory cannot tell the open card (still calibrating, and calibration finds nothing)."""

    last: dict = {}

    def opened(self, raw):
        return None

    def calibrated(self):
        return False


def test_open_card_is_read_from_memory():
    """The open-card object holds (card id, uid): found by the pair that changes between two openings, then read
    directly; the hand is searched by memory, not by the panel."""
    import struct

    from ygorl.md.memory import BufferMemory
    from ygorl.md.selection import Selection
    from ygorl.selfplay.loop import Player

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    cards = [(c.cid, c.uid) for c in raw.players[raw.me].piles["hand"]]
    assert len(cards) >= 2
    heap = bytearray(0x1000)
    struct.pack_into("<II", heap, 0x100, *cards[0])  # a card object: never changes
    struct.pack_into("<II", heap, 0x204, *cards[1])  # another one
    UI_AT = 0x804
    mem = BufferMemory({0x10000: heap})
    sel = Selection(mem, lambda: [(0x10000, 0x11000)])
    order = [1, 0]  # MD's screen order: position 0 holds the second card

    def open_at(i):
        struct.pack_into("<II", heap, UI_AT + 0x78, *struct.unpack_from("<II", heap, UI_AT))  # the one before
        struct.pack_into("<II", heap, UI_AT, *cards[order[i]])
        mem.regions[0x10000] = bytes(heap)

    player = Player.__new__(Player)
    player.selection = sel
    player.trace = type("T", (), {"observe": lambda self, **k: None})()
    player.name = str
    opened = []

    def open_hand_card(xy, why):
        i = [L.hand_card(k, len(cards)) for k in range(len(cards))].index(xy)
        opened.append(i)
        open_at(i)
        return [(xy[0], 880)]

    player.ui = type("U", (), {"open_hand_card": staticmethod(open_hand_card)})()
    # the policy wants cards[0]; the sort rule guessed position 0 (which holds cards[1])
    open_at(0)
    xy0 = L.hand_card(0, len(cards))
    got = player.verify_by_memory(raw, cards[0][0], xy0, [(xy0[0], 880)])
    assert sel.address == 0x10000 + UI_AT  # calibrated on the pair that changed
    assert got == [(L.hand_card(1, len(cards))[0], 880)] and opened[-1] == 1  # found at position 1


def test_open_card_slot_is_the_lower_of_two_equal_slots():
    """Opening one card twice leaves the same pair in the open slot and in the previous-card slot 0x78 above it."""
    import struct

    from ygorl.md.memory import BufferMemory
    from ygorl.md.selection import PREVIOUS, Selection

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    a, b = [(c.cid, c.uid) for c in raw.players[raw.me].piles["hand"]][:2]
    heap = bytearray(0x1000)
    for at in (0x400, 0x400 + PREVIOUS):
        struct.pack_into("<II", heap, at, *a)
    mem = BufferMemory({0x10000: bytes(heap)})
    sel = Selection(mem, lambda: [(0x10000, 0x11000)])
    assert sel.opened(raw) is None
    for at in (0x400, 0x400 + PREVIOUS):
        struct.pack_into("<II", heap, at, *b)
    mem.regions[0x10000] = bytes(heap)
    assert sel.opened(raw) == b and sel.address == 0x10400


def test_unlearned_names_are_told_apart_by_their_words():
    """Blaze Cannon and Soul Crossing are both ~200 px of name text; their words (measured 76 + 110 and 61 + 125
    px, 14 px apart) are not alike."""
    from ygorl.md.panel import name_fits, name_words

    def sig(words):
        out, x = np.zeros((27, 360), bool), 6
        for w in words:
            for gx in range(x, x + w, 18):  # letters 14 px wide, 4 px apart
                out[5:20, gx : min(gx + 14, x + w)] = True
            out[5:20, x + w - 3 : x + w] = True  # the last letter reaches the word's end
            x += w + 14
        return out

    blaze, soul = sig([76, 110]), sig([61, 125])
    assert name_words(blaze) == [76, 110] and name_words(soul) == [61, 125]
    assert name_fits(blaze, "Blaze Cannon") and not name_fits(blaze, "Soul Crossing")
    assert name_fits(soul, "Soul Crossing") and not name_fits(soul, "Blaze Cannon")
    assert name_fits(sig([107]), "Raigeki") and not name_fits(sig([107]), "Blaze Cannon")
    # a truncated name: the last visible word is cut, the words before it still count
    assert name_fits(sig([160, 101, 45]), "Apprentice Illusion Magician")
    assert not name_fits(sig([160, 101, 45]), "The Winged Dragon of Ra - Sphere Mode")


def test_list_pick_is_checked_on_the_panel():
    """MD's on-screen list order is not memory's (reversed here): the guessed position is ticked, the panel names
    another candidate, so it is unticked and the next position tried until the panel shows the wanted card."""
    from ygorl.md.request import Candidate
    from ygorl.md.request import Request as MDRequest
    from ygorl.selfplay.loop import Player

    img, centres = list_panel(3)
    player = Player.__new__(Player)
    player.ui, mouse, _ = make_ui(img)
    player.trace = type("T", (), {"observe": lambda self, **k: None})()
    player.card_stats = lambda c: None
    player.name = str
    player.passwords = {}
    cands = [11, 22, 33]
    screen = [33, 22, 11]  # what each position really holds

    def identify(_img, candidates, stats):
        x = mouse.clicks[-1][0]
        return screen[centres.index(x)], "stats"

    player.panel = type("P", (), {"identify": lambda self, img, c, s: identify(img, c, s)})()
    req = MDRequest(kind="mode0", mode=0, accepting=False, player=0,
                    candidates=tuple(Candidate(c, i, 0, "grave", i) for i, c in enumerate(cands)))  # fmt: skip
    index, ticked = player.find_in_list(req, 0)  # wants 11: the screen shows it last
    assert (index, ticked) == (2, True)
    y = L.LIST_CARD_Y
    assert mouse.clicks == [(centres[0], y), (centres[0], y), (centres[1], y), (centres[1], y), (centres[2], y)]


def test_classify_is_memory_first():
    """design 10 「识别与执行的系统化」: who holds the input and MD's request decide; the screen only fills in."""
    from ygorl.selfplay.loop import classify

    def k(**kw):
        base = dict(mine=True, ours=True, accepting=True, mode=2, dialog_type=0, choices=0, screen=None)
        return classify(**{**base, **kw})

    assert k() == "idle" and k(mode=3) == "battle" and k(mode=1) == "draw"
    assert k(ours=False, mine=False, screen="position") == "wait"  # the opponent's summon: never ours
    assert k(ours=False, mine=False, screen="notice") == "notice"  # nobody's: always answered
    assert k(screen="panel", dialog_type=6, choices=3, mine=False, accepting=False, mode=0) == "option"
    assert k(screen="panel", mine=False, accepting=False, mode=0) == "explore"  # a dimmed board, no request
    assert k(mode=7, dialog_type=10) == "position"  # position by memory, wherever MD draws it
    assert k(mode=7, screen="banner") == "place"
    assert k(mode=9, screen="banner") == "field_pick" and k(mode=9, context="tribute") == "tributes"
    assert k(mode=9, screen="radio") == "field_pick"
    assert (
        k(screen="radio") == "idle"
        and k(screen="radio", mine=False, accepting=False, mode=0, dialog_type=13) == "radio"
    )  # selection frames read as radio rings: memory wins
    assert k(mode=5, screen="list") == "prompt" and k(mode=0, mine=False, accepting=False, screen="list") == "list"
    assert k(mode=0, mine=False, accepting=False, screen="banner") == "settle"
    assert k(screen="phase_menu") == "idle"
    assert (
        k(screen="banner") == "idle" and k(screen="banner", zone_pending=True) == "zone_again"
    )  # a phase change under way: the decision uses the open menu


def test_list_pick_by_elimination():
    """The panel names every other distinct card but cannot name the wanted one (not learned): it is the one left;
    MD lists identical copies once, so the memory list can be longer than the screen's."""
    from ygorl.md.request import Candidate
    from ygorl.md.request import Request as MDRequest
    from ygorl.selfplay.loop import Player

    img, centres = list_panel(3)
    player = Player.__new__(Player)
    player.ui, mouse, _ = make_ui(img)
    player.trace = type("T", (), {"observe": lambda self, **k: None})()
    player.card_stats = lambda c: None
    player.name = str
    player.passwords = {}
    screen = [33, None, 22]  # position 1 shows the wanted card, which the panel cannot name

    def identify(_img, candidates, stats):
        got = screen[centres.index(mouse.clicks[-1][0])]
        return got, "name" if got else "unknown"

    player.panel = type("P", (), {"identify": lambda self, img, c, s: identify(img, c, s)})()
    req = MDRequest(kind="mode0", mode=0, accepting=False, player=0,
                    candidates=tuple(Candidate(c, i, 0, "grave", i) for i, c in enumerate([11, 22, 33, 22])))  # fmt: skip
    assert player.find_in_list(req, 0) == (1, False)


def test_graveyard_activation_checks_the_open_card_in_memory():
    import tempfile
    from pathlib import Path

    """The Graveyard strip shows the top card first; the card opened is read from memory before its Activate icon
    is clicked, and another card is never activated."""
    from types import SimpleNamespace

    from ygorl.selfplay.loop import Player

    raw = read_duel(build_memory(), BASE, replace(OFF, verified=True, semantics=SEM))
    grave = [SimpleNamespace(cid=111, uid=1), SimpleNamespace(cid=222, uid=2)]  # memory order: 222 on top
    me = raw.players[raw.me]
    raw = replace(raw, players=tuple(replace(p, piles={**p.piles, "grave": tuple(grave)}) if p is me else p
                                     for p in raw.players))  # fmt: skip
    shown = [222, 111]  # strip index -> card id
    state, clicks = {"open": None}, []
    ui = SimpleNamespace(
        open_graveyard=lambda: clicks.append("pile"),
        open_graveyard_card=lambda i, kind, why: (
            state.update(open=shown[i]) or ([(1500, 300 + i)], np.zeros((2, 2, 3)))
        ),
        click=lambda x, y, why, pause=0: clicks.append((x, y)),
        capture=lambda: np.zeros((2, 2, 3)),
        graveyard_command_greyed=lambda before, after: False,
    )
    player = Player.__new__(Player)
    player.ui, player.name, player.out = ui, str, Path(tempfile.mkdtemp())
    player.trace = SimpleNamespace(observe=lambda **k: None)
    player.game = SimpleNamespace(read=lambda: raw)
    player.selection = SimpleNamespace(read=lambda r: (state["open"], 0))
    refused = []
    player.refused = lambda r, a, why: refused.append(why)
    player.forbidden = set()
    action = SimpleNamespace(card=SimpleNamespace(loc=SimpleNamespace(sequence=0)))  # grave[0] = 111
    player.graveyard_activate(raw, action, 111)
    assert clicks == ["pile", (1500, 301)] and not refused  # strip #2 holds grave[0]; its icon clicked
    state["open"], clicks[:] = None, []
    player.selection = SimpleNamespace(read=lambda r: (222, 0))  # memory says another card is always open
    player.graveyard_activate(raw, action, 111)
    assert refused and all(c == "pile" or c[0] != 1500 for c in clicks)  # no icon of another card clicked


def test_prompt_cards_sit_side_by_side():
    """One offered card at PROMPT_CARD; two at x 940 and 1106 (measured: Sun God Unification and Dark Spell
    Regeneration offered together)."""
    assert L.prompt_cards(1) == [L.PROMPT_CARD]
    assert [x for x, _ in L.prompt_cards(2)] == [943, 1109]


def test_prompt_ticks_each_offered_card_until_memory_names_the_one_meant():
    """A prompt offering two cards: the second one is the policy's; it is found by reading memory per tick."""
    from ygorl.selfplay.loop import Player

    at = {xy: cid for xy, cid in zip(L.prompt_cards(2), (1, 2), strict=True)}
    state, clicks = {"open": None}, []
    player = Player.__new__(Player)
    player.ui = SimpleNamespace(click=lambda x, y, why, pause=0: clicks.append((x, y)) or state.update(
        open=at.get((x, y), state["open"])))  # fmt: skip
    player.game = SimpleNamespace(read=lambda: None)
    player.selection = SimpleNamespace(address=1, read=lambda raw: (state["open"], 9) if state["open"] else None)
    player.passwords = {1: 111, 2: 222}
    player.trace = SimpleNamespace(observe=lambda **k: None)
    assert player.tick_prompt_card(222) == 222 and state["open"] == 2
    state["open"], clicks[:] = None, []
    assert player.tick_prompt_card(333) == 111 and state["open"] == 1  # none meant: the first card, ticked again


def test_ra_pays_lp_only_when_it_can_attack():
    """Ra's optional pay-LP trigger (a one-card list of Ra in the monster zone) is taken only in our Main Phase 1
    after the duel's first turn; any other list is not this rule's."""
    from ygorl.selfplay.loop import RA, Player

    player = Player.__new__(Player)
    player.passwords = {7: RA, 8: 1}
    ra = SimpleNamespace(candidates=[SimpleNamespace(cid=7, location="mzone")])
    raw = SimpleNamespace(turn=0, phase=2, turn_player=0, me=0)  # PHASES[2] == "main1"
    assert player.ra_pays_lp(raw, ra) is False  # the first turn: Ra cannot attack
    assert player.ra_pays_lp(SimpleNamespace(**{**vars(raw), "turn": 3}), ra)
    other = SimpleNamespace(candidates=[SimpleNamespace(cid=8, location="mzone")])
    assert player.ra_pays_lp(raw, other) is None
