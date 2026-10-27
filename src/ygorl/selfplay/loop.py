"""Autonomous Solo player for Master Duel (design 10, Solo self-play). Off unless explicitly run.

One loop: wait until the game asks the client for something, work out what is being asked from memory (the
engine's input state) and from a few fixed screen signatures (MD's own dialogs), decide, click, and check the
result in memory. Main Phase decisions come from the policy through the advisor (:func:`advise`); Battle Phase and
prompts use small rules. Anything it does not recognise is a *stuck* record: a full-resolution capture, the memory
state and the context, written to ``--out`` before the player stops. Those records are how recognisers get added
until the player needs no help (the goal: a scripted-only player).

Decisions stay in the player; every click script lives in :mod:`ygorl.md.ui` (no decisions there). The player is
:class:`Player` here (its state, reading memory and the screen, the dispatch of each kind of question) composed of
mixins, one module each: :mod:`~ygorl.selfplay.commands` (Main Phase commands, hand-card identification),
:mod:`~ygorl.selfplay.lists` (card lists), :mod:`~ygorl.selfplay.prompts` (chain prompts and MD's dialogs),
:mod:`~ygorl.selfplay.battle` (attacks, Tributes, field picks), :mod:`~ygorl.selfplay.policy` (the policy's
decisions, masks, shadow lockstep), :mod:`~ygorl.selfplay.recovery` (failures by cause, recovery, restarts) and
:mod:`~ygorl.selfplay.series` (the step loop, series, self-audit); the pure helpers they share (``classify``,
``check_expectation``, the keys, MD's fixed spots) are in :mod:`~ygorl.selfplay.state`. Each loop step is one
record of the run's trace (:mod:`ygorl.selfplay.trace`, ``<out>/<run>.jsonl``): what was read, decided, clicked and
checked; ``python -m ygorl.selfplay.trace SUMMARY <trace>`` summarises a run.

Every click goes through :class:`ygorl.md.input.GatedMouse`, which re-checks the Solo gate (verified DLL,
human client, computer opponent) before each event.

    python -m ygorl.selfplay.loop --env md-2026-09 --policy CHECKPOINT [--out out/mdplay] [--max-steps 400]
        [--screen-every N]
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ygorl.md import layout as L
from ygorl.md.input import GatedMouse
from ygorl.md.memory import PHASES, MemoryReadError, RawDuel, load_md_cid
from ygorl.md.screen import ScreenError
from ygorl.md.screen import capture as capture_client
from ygorl.md.ui import (
    UI,
    Stuck,
    opp_monster_xy,
)
from ygorl.md.panel import PanelBook
from ygorl.md.request import read_request
from ygorl.md.selection import for_process as selection_for
from ygorl.md.client import Client
from ygorl.selfplay.decide import Decider, policy_scorer
from ygorl.selfplay.observe import ShadowObserver
from ygorl.selfplay.trace import Trace
from ygorl.selfplay.state import (
    INPUT_MODE,
    INPUT_ACCEPT,
    INPUT_PLAYER,
    FOCUS_CID,
    OPTION_DIALOG,
    OPTION_OK,
    LIST_NEXT,
    RA,
    FIELD_ICON_STEP,
    FIELD_ICON_DY,
    SLIFER,
    HAND_LOCATION,
    COMMAND_BITS,
    LIST_SHOWN,
    LIST_PAGES,
    list_more,
    option_tab,
    option_tabs,
    RESULT_WIN,
    NEUTRAL_SPOT,
    RESULT_RETRY,
    SETTINGS_GEAR,
    SETTINGS_RETRY,
    SETTINGS_RETRY_YES,
    MAIN,
    BATTLE,
    OPEN_PROMPT,
    CHAIN_PROMPT,
    SUMMON_PICK,
    PLACE,
    FIELD_PICK,
    FIELD_PICK_TRIBUTE,
    _progress,
    Input,
    Expectation,
    _cards,
    check_expectation,
    EFFECT_LIST_END,
    POSITION_DIALOG,
    RADIO_DIALOG,
    NEUTRAL,
    own_effect_nodes,
    classify,
    declares_a_card,
    plain_name,
    searchable,
    clickable,
    ATTACKS_MASKED,
    _action_key,
    _own_cards,
    _board_key,
)
from ygorl.selfplay.recovery import Failure
from ygorl.selfplay.commands import CommandsMixin
from ygorl.selfplay.lists import ListsMixin
from ygorl.selfplay.prompts import PromptsMixin
from ygorl.selfplay.battle import BattleMixin
from ygorl.selfplay.policy import PolicyMixin
from ygorl.selfplay.recovery import RecoveryMixin
from ygorl.selfplay.series import SeriesMixin

# the names the player's modules and tests import from here (see the module layout above)
__all__ = [
    "ATTACKS_MASKED",
    "BATTLE",
    "CHAIN_PROMPT",
    "COMMAND_BITS",
    "EFFECT_LIST_END",
    "Expectation",
    "FIELD_ICON_DY",
    "FIELD_ICON_STEP",
    "FIELD_PICK",
    "FIELD_PICK_TRIBUTE",
    "FOCUS_CID",
    "Failure",
    "HAND_LOCATION",
    "INPUT_ACCEPT",
    "INPUT_MODE",
    "INPUT_PLAYER",
    "Input",
    "LIST_NEXT",
    "LIST_PAGES",
    "LIST_SHOWN",
    "MAIN",
    "NEUTRAL",
    "NEUTRAL_SPOT",
    "OPEN_PROMPT",
    "OPTION_DIALOG",
    "OPTION_OK",
    "PHASES",
    "PLACE",
    "POSITION_DIALOG",
    "Player",
    "RA",
    "RADIO_DIALOG",
    "RESULT_RETRY",
    "RESULT_WIN",
    "SETTINGS_GEAR",
    "SETTINGS_RETRY",
    "SETTINGS_RETRY_YES",
    "SLIFER",
    "SUMMON_PICK",
    "_action_key",
    "_board_key",
    "_cards",
    "_own_cards",
    "_progress",
    "capture",
    "check_expectation",
    "classify",
    "clickable",
    "declares_a_card",
    "list_more",
    "main",
    "option_tab",
    "option_tabs",
    "own_effect_nodes",
    "plain_name",
    "searchable",
]


def capture(window) -> np.ndarray:
    """A capture in the reference frame (see :mod:`ygorl.md.layout`), whatever the window size."""
    return L.reference_image(capture_client(window))


@dataclass
class Player(CommandsMixin, ListsMixin, PromptsMixin, BattleMixin, PolicyMixin, RecoveryMixin, SeriesMixin):
    env: object
    policy: object
    out: Path
    samples: int = 4
    screen_every: int = 0  # save the reference-frame capture into the trace every N steps (0: off)
    mouse: GatedMouse = field(init=False)
    ui: UI = field(init=False)
    trace: Trace = field(init=False)

    expect: Expectation | None = None
    recent: list = field(default_factory=list)  # the last states seen, to stop instead of looping
    draw_attempts: int = 0
    draw_seen: tuple = (-1, -1)
    last_zone: tuple | None = None  # (location, sequence) of the zone picked for our last own action
    pending_panel: tuple | None = None  # (capture, card id) of the last opened hand card, learned once confirmed
    pending_list: tuple | None = None  # (panel capture, request) of the last list pick, learned once it resolves
    summon_level: int = 0  # level of the monster our current Summon / Set is for (0: none under way)
    field_picked: tuple = ((-1, -1), [])  # ((turn, phase), cards picked) of the field selection under way
    recoveries: dict = field(default_factory=dict)  # state -> self-recovery attempts this duel
    recovery_count: int = 0
    phase_seen: tuple = (-1, -1)
    forbidden: set = field(default_factory=set)  # ((turn, phase), action key) MD refused this phase

    def __post_init__(self) -> None:
        from ygorl.advisor.reconstruct import OpponentPrior
        from ygorl.engine.duel import default_cards

        self.mouse = GatedMouse()
        self.ui = UI(self.mouse, self._capture)
        self.game = self.mouse.game
        self.selection = selection_for(self.game.memory, self.game.pid)
        try:  # MD's client objects (command masks, prompt cards; design 10 「客户端的命令掩码」)
            self.client = Client.for_process(self.game.memory, self.game.pid)
        except Exception as exc:  # noqa: BLE001 - without it the screen decides, as before
            self.client = None
            print(f"  client: not readable ({exc}); commands judged on screen", flush=True)
        self.cards = default_cards()
        self.passwords = load_md_cid(self.env.artifact_path("md_cid.json"))
        self.prior = OpponentPrior.from_environment(self.env)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(self.out, screen_every=self.screen_every)
        self.observer = ShadowObserver(self.env.rule_flags, self.cards, self.prior, self._score)  # design 10
        scorer = policy_scorer(self.policy, self.cards, self.env.player.starting_lp) if self.policy else None
        # in-action decisions rebuilt from the current board
        self.decider = Decider(self.env.rule_flags, self.cards, self.passwords, self.prior, scorer=scorer,
                               samples=self.samples, stats=self.field_stats)  # fmt: skip
        self.panel = PanelBook(self.out / "panel_book.npz")  # learned across runs
        self.attacked: set[int] = set()  # uids that attacked (or could not) in this Battle Phase
        self.activated: set[tuple[int, str]] = set()  # (password, location) the policy activated off the field
        self.turn_seen = -1
        self.context = ""  # what the last action was, for sub-prompts
        self.responded = None  # the prompt state last answered with an activation

    # --- reading --------------------------------------------------------------------------------------------
    def _capture(self) -> np.ndarray:
        """A capture of the game; while the screen cannot be read at all (locked or blanked: XWayland refuses every
        GetImage, even of the root window) the player pauses and waits instead of acting blind."""
        waited = 0.0
        while True:
            try:
                return capture(self.mouse.window)
            except (subprocess.CalledProcessError, ScreenError):
                try:  # the window may have moved or been re-created
                    self.mouse.refresh_window()
                    return capture(self.mouse.window)
                except (subprocess.CalledProcessError, ScreenError):
                    pass
            if waited % 300 == 0:
                print(f"PAUSED: the screen cannot be captured (locked or blanked?), waited {waited:.0f} s", flush=True)
            time.sleep(10.0)
            waited += 10.0

    @property
    def state_base(self) -> int:
        """The duel state S: re-read every time, a new duel (Retry) allocates a new one."""
        return struct.unpack("<Q", self.game.memory.read(self.game.module.base + self.game.offsets.state_ptr, 8))[0]

    def input(self) -> Input:
        b = self.game.memory.read(self.state_base + INPUT_MODE, 0x40)
        return Input(struct.unpack_from("<I", b, 0)[0], bool(struct.unpack_from("<I", b, INPUT_ACCEPT - INPUT_MODE)[0]),
                     struct.unpack_from("<I", b, INPUT_PLAYER - INPUT_MODE)[0])  # fmt: skip

    def focus_cid(self) -> int:
        return struct.unpack("<I", self.game.memory.read(self.state_base + FOCUS_CID, 4))[0] & 0xFFFF

    def name(self, cid: int) -> str:
        pw = self.passwords.get(cid)
        return self.cards[pw].name if pw in self.cards else str(cid)

    def wait_turn(self, timeout: float = 120) -> tuple[RawDuel, Input, str | None]:
        """Wait until the client must act: an engine input addressed to it, or one of MD's dialogs (stable ~1 s).
        ``timeout`` counts from the last change of the duel in memory: the CPU's own turn can run for minutes of
        chains and animations (a Red-Eyes turn took over 4), and that is not the player being stuck."""
        end, stable, last, progress = time.time() + timeout, 0, None, None
        menu_checked = 0.0
        seen_dialogs: list = []  # the last few reads: a dialog whose button pulses reads as one only now and then
        while time.time() < end:
            if time.time() - menu_checked > 5:
                menu_checked = time.time()
                self.answer_turn_order()
            try:
                raw, inp = self.game.read(), self.input()
            except MemoryReadError:  # between duels (loading) the state can be gone for a moment
                time.sleep(1.0)
                continue
            now = _progress(raw, inp)
            if now != progress:
                progress, end = now, time.time() + timeout
            if raw.finished:  # the engine's own flag: LP reads can catch the obfuscated value mid-update
                return raw, inp, "over"
            mine = inp.accepting and inp.player == raw.me
            dialog = self.ui.dialog()  # MD's own dialog wins over the engine mode
            seen_dialogs = (seen_dialogs + [dialog])[-4:]
            if dialog is None:
                # a dialog read in the last few polls is still up (a Tribute question's pulsing Yes button made
                # the read flicker, the state never counted as stable, and the duel waited 120 s twice)
                dialog = next((d for d in reversed(seen_dialogs) if d is not None), None)
            if dialog == "panel" and not mine and self.read_dialog_type() != OPTION_DIALOG:
                dialog = None  # a dimmed board while MD works: nothing for us yet
            if dialog and dialog not in NEUTRAL and not self.dialog_is_ours(raw, inp, dialog):
                dialog = None  # the opponent holds the input: their dialog, not ours to answer
            key = (inp.mode, mine, dialog, raw.turn, raw.phase)
            stable = stable + 1 if key == last and (mine or dialog) else 0
            last = key
            if stable >= 3:
                return raw, inp, dialog
            time.sleep(0.35)
        raise Stuck(f"nothing to do for {timeout:.0f} s")

    def dialog_is_ours(self, raw: RawDuel, inp, dialog: str | None) -> bool:
        """Whether MD's dialog on screen is ours to answer. MD's own dialogs while the engine takes no input can read
        the opponent's input player (runs s68 / s69, our Artillery Catapult Turtle's Special Summon: its list, then
        its position, each waited 120 s): ours when memory confirms it (the client's list holds our cards; the
        dialog type is the position question)."""
        if inp.player == raw.me:
            return True
        if inp.accepting:
            return False
        if dialog == "list":
            return self.list_is_ours(raw)
        return dialog == "position" and self.read_dialog_type() == POSITION_DIALOG

    def answer_turn_order(self) -> bool:
        """MD's "go first or second?" menu before a duel (some stages ask it after Retry, and once it came only
        after the player had taken the restarted duel for running: it waited 120 s, then crashed reading the
        freed duel state). The menu's own pick (the user allowed menu clicks for restarts)."""
        from ygorl.md import menu
        from ygorl.md.input import MenuMouse

        try:
            img = self._capture()
            if menu.screen(img) != "turn_order":
                return False
            MenuMouse(self.game).click(*menu.target(img, "turn_order"))
        except Exception:  # noqa: BLE001 - no window / capture: nothing to answer
            return False
        print("  menu: turn_order", flush=True)
        self.trace.observe(menu="turn_order")
        return True

    def _memory_key(self) -> tuple:
        """What counts as MD moving on: the input state, the dialog and the board."""
        raw = self.game.read()
        req = self.request_now(raw)
        me, op = raw.players[raw.me], raw.players[1 - raw.me]
        return req.mode, req.accepting, req.dialog_type, req.raw.get("list_count", 0), _board_key(me, op)

    def explore(self, why: str, choice: int = 0, decline: bool = False) -> None:
        """A screen without a script (or one that stays stuck): let the explorer find what to click from the
        screen's affordances and the learned playbook (design 10 「边角情况由 player 自己学会」)."""
        from ygorl.md.explore import Explorer, Playbook

        if getattr(self, "explorer", None) is None:
            self.explorer = Explorer(self.ui, Playbook(self.out / "playbook.json"))

        def memory():
            r = self.request_now(self.game.read())
            return {"mode": r.mode, "accepting": r.accepting, "dialog_type": r.dialog_type,
                    "list_type": r.list_type, "list_count": r.raw.get("list_count", 0), "choices": len(r.choices)}  # fmt: skip

        before = self._memory_key()
        result = self.explorer.run(memory=memory, progress=lambda: self._memory_key() != before, choice=choice,
                                   decline=decline)  # fmt: skip
        self.trace.observe(explore={"why": why, **result.to_json()})
        print(f"  explore ({why}): {'ok' if result.ok else 'nothing worked'} via {result.source} "
              f"{result.recipe} [{result.signature}]", flush=True)  # fmt: skip
        if not result.ok:
            raise Stuck(f"exploration found nothing that works ({why})")

    STATE_BLOCK = 0x4000  # the duel state struct S: fields up to S+0x3d18 are known

    def save_memory_blocks(self) -> None:
        """The raw duel state (S) and its key struct (K) of every step, for decoding once-per-turn usage and
        turn-long effects offline; read-only, a failed read just skips the step."""
        try:
            mem, base, off = self.game.memory, self.game.module.base, self.game.offsets
            key = struct.unpack("<Q", mem.read(base + off.key_ptr, 8))[0]
            state = bytearray(mem.read(self.state_base, self.STATE_BLOCK))
            me = self.game.read().me
            # the opponent's hidden cards must never leave memory (design 10): blank their player block, the
            # uid -> card id table and the effect list (it names the cards its effects belong to, face-down too),
            # then put back the nodes of our own lists (the player list and our field cards'): our effects
            own = own_effect_nodes(bytes(state), me, off)
            opp = (1 - me) * off.player_stride
            state[opp : opp + off.player_stride] = bytes(off.player_stride)
            state[off.uid_table : EFFECT_LIST_END] = bytes(EFFECT_LIST_END - off.uid_table)
            for at, node in own.items():
                state[at : at + len(node)] = node
            self.trace.save_memory({"S": bytes(state), "K": mem.read(key, 0x400)})
        except Exception:  # noqa: BLE001 - diagnostics only
            pass

    def read_dialog_type(self) -> int:
        try:
            return struct.unpack("<I", self.game.memory.read(self.state_base + self.game.offsets.dialog_type, 4))[0]
        except Exception:  # noqa: BLE001
            return 0

    def request_now(self, raw: RawDuel, list_shown: bool = False):
        return read_request(self.game.memory, self.game.module.base, self.game.offsets, raw, list_shown=list_shown)

    # kind (classify) -> method(raw, inp); kinds not listed are MD's own dialogs, answered by :meth:`dialog`
    HANDLERS = {
        "idle": "on_idle", "battle": "on_battle", "draw": "on_draw", "summon_pick": "on_summon_pick",
        "prompt": "prompt", "place": "on_place", "zone_again": "on_zone_again", "tributes": "on_tributes",
        "field_pick": "on_field_pick", "settle": "on_settle", "wait": "on_wait", "explore": "on_explore",
        "position": "on_position",
    }  # fmt: skip

    def on_idle(self, raw: RawDuel, inp) -> None:
        self.main_phase(raw)

    def on_battle(self, raw: RawDuel, inp) -> None:
        self.battle(raw)

    def on_draw(self, raw: RawDuel, inp) -> None:
        self.draw_attempts = self.draw_attempts + 1 if self.phase_seen == self.draw_seen else 0
        self.draw_seen = self.phase_seen
        self.trace.rule("draw-continue", attempt=self.draw_attempts)
        self.ui.continue_draw(self.draw_attempts)

    def on_summon_pick(self, raw: RawDuel, inp) -> None:
        self.summon_pick(raw)

    def on_place(self, raw: RawDuel, inp) -> None:
        """Where to put a card (a summoned monster, or a Spell / Trap an effect places): MD outlines the zones it
        allows; the first outlined one, Monster Zones first, else the first free monster zone."""
        self.trace.rule("zone-first-free")
        self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)  # a resolve animation can still cover the zones
        key = (raw.turn, raw.phase, _own_cards(raw))
        if not self.zones_outlined() and getattr(self, "place_waited", None) != key:
            self.place_waited = key
            self.trace.observe(place="no zone outlined yet")
            return  # nothing to click yet: the loop comes back with the next state (once; then as before)
        location, zone = self.outlined_zone(raw)
        if location == "opp_mzone":
            self.ui.click(*opp_monster_xy(zone), "zone for the card (the opponent's Monster Zone)", 2.5)
            return
        self.ui.pick_zone(location, zone, f"zone for the card ({location})")
        self._placed(location, zone)

    def on_zone_again(self, raw: RawDuel, inp) -> None:
        self.trace.rule("zone-again")
        self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
        location = self.last_zone[0]
        # the leftmost free zone now (memory): a zone taken meanwhile would open the card in it instead
        zone = self.free_zone(raw, location)
        self.ui.pick_zone(location, zone, "zone again")
        self._placed(location, zone)  # the shadow takes the same zone and stays in step
        self.last_zone = None

    def on_tributes(self, raw: RawDuel, inp) -> None:
        self.pick_tributes(raw)  # all of them in one step: the board does not change in between
        self.context = ""

    def on_field_pick(self, raw: RawDuel, inp) -> None:
        self.field_pick(raw)

    def on_settle(self, raw: RawDuel, inp) -> None:
        self.trace.rule("banner-wait")
        self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)

    def on_wait(self, raw: RawDuel, inp) -> None:
        """The opponent holds the input: nothing on screen is ours to answer."""
        self.trace.observe(wait="the opponent holds the input")
        self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=4.0)

    def on_explore(self, raw: RawDuel, inp) -> None:
        self.explore(f"unknown input mode {inp.mode}")

    def on_position(self, raw: RawDuel, inp) -> None:
        self.dialog(raw, "position")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="autonomous Solo player for Master Duel (Solo vs computer only)")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/mdplay"))
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1500, help="most loop steps per duel")
    ap.add_argument("--games", type=int, default=1, help="duels in a row: Retry on MD's result screen in between")
    ap.add_argument("--screen-every", type=int, default=0, help="save a capture into the trace every N steps")
    args = ap.parse_args(argv)
    from ygorl.data.environment import load_environment
    from ygorl.train.checkpoint import load_actor

    env = load_environment(args.env)
    player = Player(env, load_actor(args.policy), args.out, args.samples, args.screen_every)
    player.trace.start(env=args.env, policy=str(args.policy), samples=args.samples, max_steps=args.max_steps)
    result: dict = {"exit": None}
    try:
        result["exit"] = player.run_games(args.games, args.max_steps)
        return result["exit"]
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        player.trace.close(**result)
        print(f"trace: {player.trace.path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
