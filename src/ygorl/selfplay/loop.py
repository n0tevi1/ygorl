"""Autonomous Solo player for Master Duel (design 10, Solo self-play). Off unless explicitly run.

One loop: wait until the game asks the client for something, work out what is being asked from memory (the
engine's input state) and from a few fixed screen signatures (MD's own dialogs), decide, click, and check the
result in memory. Main Phase decisions come from the policy through the advisor (:func:`advise`); Battle Phase and
prompts use small rules. Anything it does not recognise is a *stuck* record: a full-resolution capture, the memory
state and the context, written to ``--out`` before the player stops. Those records are how recognisers get added
until the player needs no help (the goal: a scripted-only player).

Decisions stay here; every click script lives in :mod:`ygorl.md.ui` (no decisions there). Each loop step is one
record of the run's trace (:mod:`ygorl.selfplay.trace`, ``<out>/<run>.jsonl``): what was read, decided, clicked and
checked; ``python -m ygorl.selfplay.trace SUMMARY <trace>`` summarises a run.

Every click goes through :class:`ygorl.md.input.GatedMouse`, which re-checks the Solo gate (verified DLL,
human client, computer opponent) before each event.

    python -m ygorl.selfplay.loop --env md-2026-09 --policy CHECKPOINT [--out out/mdplay] [--max-steps 400]
        [--screen-every N]
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import struct
from collections import Counter
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ygorl.engine import constants as C
from ygorl.md import layout as L
from ygorl.md.input import GatedMouse
from ygorl.md.memory import PHASES, MemoryReadError, RawDuel, load_md_cid, own_deck, solo_gate, to_snapshot
from ygorl.md.screen import ScreenError
from ygorl.md.screen import capture as capture_client
from ygorl.md.ui import (
    GY_STRIP_SHOWN,
    MINIMISED_RESTORE,
    PHASE_MENU_CANCEL,
    UI,
    Stuck,
    command_order,
    is_command_icon,
    my_monster_xy,
    notice_button,
    opp_monster_xy,
    radio_rows,
    yellow_highlight,
    zone_outlined,
)
from ygorl.md.panel import PanelBook, name_fits, name_signature, name_words, panel_kind
from ygorl.md.explore import find_buttons
from ygorl.md.request import read_request
from ygorl.md.selection import for_process as selection_for
from ygorl.selfplay.decide import Decider, policy_scorer
from ygorl.selfplay.observe import ShadowObserver
from ygorl.selfplay.trace import Trace

# --- input state (S + ...): written by duel.dll's command layer, read by DLL_DuelComGetCommandMask --------------
INPUT_MODE, INPUT_ACCEPT, INPUT_PLAYER, FOCUS_CID = 0x3CD0, 0x3CDC, 0x3D00, 0x3CF0
OPTION_DIALOG = 6  # the dialog type of MD's "which effect?" choice (DLL_DuelDlg*: tabs and an OK button)
OPTION_OK = (1024, 788)


def option_tab(index: int, count: int, y: int = 446) -> tuple[int, int]:
    """The tab of choice ``index`` among ``count`` in the effect-choice dialog (tabs span x 520-1540; row ``y``)."""
    return round(520 + (index + 0.5) * 1020 / count), y


def option_tabs(img: np.ndarray, count: int) -> tuple[int, int | None]:
    """The effect-choice dialog's tab row (y) and which tab is selected (the lime one), read from the screen. The
    row was once taken as y 418, the dialog's title: tab clicks did nothing and OK took the preselected effect
    (Millennium Revelation's first effect instead of the second)."""
    rows: dict[int, list[int]] = {}
    for y in range(400, 540, 2):
        for i in range(count):
            x = option_tab(i, count)[0]
            r = img[y, x - 60 : x + 60].astype(int)
            if ((r[:, 1] > 160) & (r[:, 0] > 110) & (r[:, 2] < 60)).mean() > 0.6:
                rows.setdefault(i, []).append(y)
    if not rows:
        return 446, None
    i = max(rows, key=lambda k: len(rows[k]))
    return (min(rows[i]) + max(rows[i])) // 2, i  # the middle of the lime tab


RESULT_WIN = 1  # DLL_DuelGetDuelResult for the client: 1 won (VICTORY, Confirm -> menus), 2 lost (Retry offered)
NEUTRAL_SPOT = (1300, 640)  # empty board between the rows: clicking it closes a card left open
RESULT_RETRY = (743, 1093)  # MD's result screen: Retry (left) and OK (right)
# the settings gear in a duel, its Retry and the confirmation's Yes: restart a duel the player gave up on
SETTINGS_GEAR, SETTINGS_RETRY, SETTINGS_RETRY_YES = (66, 95), (298, 979), (1218, 743)
MAIN, BATTLE, OPEN_PROMPT, CHAIN_PROMPT, SUMMON_PICK, PLACE, FIELD_PICK = 2, 3, 4, 5, 6, 7, 8  # observed modes
FIELD_PICK_TRIBUTE = 9  # "choose a monster to Tribute" (an effect's tribute: the explorer learned pick + Confirm)


def capture(window) -> np.ndarray:
    """A capture in the reference frame (see :mod:`ygorl.md.layout`), whatever the window size."""
    return L.reference_image(capture_client(window))


def _progress(raw: RawDuel, inp) -> tuple:
    """What changes in memory while a duel moves on: turn, phase, LP, who holds the input, and where every card is
    (by uid only: nothing about hidden cards is kept)."""
    return (
        raw.turn,
        raw.phase,
        inp.mode,
        inp.player,
        tuple(
            (
                p.lp,
                tuple(c.uid if c else 0 for c in p.slots),
                tuple((k, tuple(c.uid for c in v)) for k, v in sorted(p.piles.items())),
            )
            for p in raw.players
        ),
    )


@dataclass
class Input:
    mode: int
    accepting: bool
    player: int


@dataclass
class Expectation:
    """What the policy chose and the board it chose on, checked against memory once the action has resolved."""

    action: object
    before: RawDuel
    phase: str


def _cards(pl, location: str) -> list:
    if location == "hand":
        return list(pl.piles["hand"])
    if location == "mzone":
        return [c for c in pl.slots[:7] if c]
    if location == "szone":
        return [c for c in pl.slots[7:12] if c]
    return list(pl.piles[location])


def check_expectation(exp: Expectation, now: RawDuel, passwords: dict[int, int]) -> str | None:
    """None when ``now`` shows the chosen action happened, else what is inconsistent."""
    act, before = exp.action, exp.before
    me0, me1 = before.players[before.me], now.players[now.me]
    op0, op1 = before.players[1 - before.me], now.players[1 - now.me]
    kind = act.kind
    if kind in ("battle_phase", "main2", "end_phase"):
        want = {"battle_phase": "battle", "main2": "main2"}.get(kind)
        phase = PHASES[now.phase] if now.phase < len(PHASES) else str(now.phase)
        if now.turn != before.turn:
            return None if kind == "end_phase" else f"{kind}: the turn ended instead"
        if want is not None and phase != want:
            return f"{kind}: the phase is {phase}"
        if kind == "end_phase" and phase in ("main1", "battle"):
            return f"end_phase: still in {phase}"
        return None
    cid = next((c for c, p in passwords.items() if act.card is not None and p == act.card.code), None)

    def count(pl, loc, **kw):
        return sum(c.cid == cid and all(getattr(c, k) == v for k, v in kw.items()) for c in _cards(pl, loc))

    left_hand = count(me1, "hand") < count(me0, "hand")
    if kind == "summon":
        ok = left_hand and count(me1, "mzone", face=1, sideways=0) > count(me0, "mzone", face=1, sideways=0)
        return None if ok else "summon: no new face-up Attack Position monster from the hand"
    if kind == "mset":
        ok = left_hand and count(me1, "mzone", face=0) > count(me0, "mzone", face=0)
        return None if ok else "mset: no new face-down monster from the hand"
    if kind == "sset":
        ok = left_hand and count(me1, "szone", face=0) > count(me0, "szone", face=0)
        return None if ok else "sset: no new face-down Spell/Trap from the hand"
    if kind == "attack":
        changed = (op1.lp, me1.lp, len(_cards(op1, "mzone")), len(_cards(me1, "mzone"))) != (
            op0.lp, me0.lp, len(_cards(op0, "mzone")), len(_cards(me0, "mzone")))  # fmt: skip
        return None if changed else "attack: nothing changed (no damage, no monster left the field)"
    if kind in ("activate", "spsummon"):
        if _board_key(me0, op0) == _board_key(me1, op1):
            return f"{kind}: nothing changed on the board"
        loc = act.card.loc if act.card is not None else None
        if kind == "activate" and loc is not None and loc.location == C.LOCATION_MZONE:
            s0, s1 = me0.slots[loc.sequence], me1.slots[loc.sequence]
            turned = s0 is not None and s1 is not None and s0.cid == s1.cid and s0.sideways != s1.sideways
            rest0 = _board_key(me0, op0)[1:]
            rest1 = _board_key(me1, op1)[1:]
            if turned and rest0 == rest1 and [c for i, c in enumerate(me0.slots) if i != loc.sequence] == [
                    c for i, c in enumerate(me1.slots) if i != loc.sequence]:  # fmt: skip
                return "activate: only the monster's position changed (the Change Position icon was clicked)"
        return None
    return None  # reposition and others: not checked yet


EFFECT_LIST_END = 0x3C80  # S + uid_table (0x1bec) .. here: the uid table and the effect list (S + 0x2bec)
POSITION_DIALOG, RADIO_DIALOG = 10, 13  # DLL_DuelDlg* dialog types (6 is OPTION_DIALOG)
NEUTRAL = ("notice", "minimised")  # screens that belong to nobody's input: always answered


def own_effect_nodes(state: bytes, me: int, off) -> dict[int, bytes]:
    """{offset in S: node bytes} of the effect-list nodes reachable from our player list head and our field slots'
    list heads: what our own effects are (kept in the recorded memory; the rest of the table is blanked)."""
    blk = me * off.player_stride
    heads = [struct.unpack_from("<H", state, blk + off.player_effects)[0]]
    for i in range(13):
        at = blk + off.slot_base + i * off.slot_stride
        if struct.unpack_from("<H", state, at)[0]:
            heads.append(struct.unpack_from("<H", state, at + off.slot_chain)[0])
    out: dict[int, bytes] = {}
    for head in heads:
        while 0 < head < 0x100:
            at = off.effect_table + head * off.effect_stride
            if at in out or at + off.effect_stride > len(state):
                break
            out[at] = bytes(state[at : at + off.effect_stride])
            head = struct.unpack_from("<H", state, at + off.effect_next)[0]
    return out


def classify(*, mine: bool, ours: bool, accepting: bool, mode: int, dialog_type: int, choices: int,
             screen: str | None, context: str = "", zone_pending: bool = False) -> str:  # fmt: skip
    """What the step must answer (design 10 「识别与执行的系统化」): memory first (who holds the input, the input
    mode, MD's dialog type), the screen class only where memory cannot tell (MD's own dialogs while the engine
    input is closed) and as the locator. ``mine``: the engine waits for our input; ``ours``: the input (or MD's
    dialog) is the client's, accepting or not. Returns a key of :data:`Player.HANDLERS`, "wait" or "over"."""
    if screen == "over":
        return "over"
    if screen in NEUTRAL:
        return screen
    if not ours:
        return "wait"  # the opponent holds the input: whatever is on screen is theirs
    if screen == "phase_menu" and mine and mode in (MAIN, BATTLE):
        screen = None  # a phase change under way: the command decision uses (or closes) the open menu
    if screen == "panel":  # a dark band without yes / no labels: the effect choice when memory says so
        screen = "option" if dialog_type == OPTION_DIALOG and choices >= 2 else None
    if screen in ("yesno", "panel") and dialog_type == OPTION_DIALOG and choices >= 2:
        screen = "option"
    if screen == "radio" and dialog_type != RADIO_DIALOG:
        screen = None  # rings without MD's radio dialog: selection frames or zone outlines (clicked a set card)
    if mine and mode == SUMMON_PICK and screen in (None, "banner"):
        return "summon_pick"
    if mine and mode in (OPEN_PROMPT, CHAIN_PROMPT) and screen in (None, "list"):
        return "prompt"  # an activate-or-pass prompt panel looks like a list
    if mine and mode == PLACE and (screen == "position" or dialog_type == POSITION_DIALOG):
        return "position"
    if mine and mode == PLACE and screen in ("banner", "panel", None):
        return "place"  # the zone question can come while a chain's resolve animation still dims the board
    if mine and mode == MAIN and screen == "banner" and zone_pending:
        return "zone_again"  # MD still asks for the zone of our own Summon / Set
    if mine and mode in (MAIN, BATTLE) and screen == "banner":
        screen = None  # the engine waits for a command: no banner question is pending (the opponent's ATK / DEF
        # numbers on a dark board passed for a banner's text)
    if mine and mode in (FIELD_PICK, FIELD_PICK_TRIBUTE) and context == "tribute":
        return "tributes"
    if mine and mode in (FIELD_PICK, FIELD_PICK_TRIBUTE) and screen not in ("list", "yesno", "option", "notice"):
        return "field_pick"  # memory says a field pick; the selection frames can pass for other screens (radio)
    if screen == "banner" and not accepting:
        return "settle"  # MD's own text while the engine takes no input (a Summon animating)
    if screen:
        return screen  # MD's own dialogs: yesno, option, list, radio, position, phase_menu, declare, ...
    if mode == MAIN:
        return "idle"
    if mode == BATTLE:
        return "battle"
    if mode == 1:
        return "draw"
    return "explore"


@functools.cache
def declares_a_card(password: int) -> bool:
    """The card's script asks to declare a card name (``Duel.AnnounceCard``): MD's answer is a search box that needs
    typed text, and window-only input cannot type (The True Name stalled the duel)."""
    from ygorl import paths

    for d in paths.script_directories():
        f = Path(d) / f"c{password}.lua"
        if f.is_file():
            return "AnnounceCard" in f.read_text(errors="replace")
    return False


def plain_name(name: str) -> bool:
    """A card name of ASCII letters, digits and spaces only: typed whole, MD's search finds exactly it."""
    return all((ch.isascii() and ch.isalnum()) or ch == " " for ch in name)


def searchable(name: str) -> str:
    """The longest stretch of a card name that the keyboard can type (search matches substrings). Apostrophes and
    quotes split it: MD's names use other glyphs for them, and a search with the ASCII one finds nothing."""
    from ygorl.md.input import GatedMouse

    ok = lambda ch: ((ch.isascii() and ch.isalnum()) or ch in GatedMouse.KEYSYMS) and ch not in "'\""  # noqa: E731
    parts, cur = [], ""
    for ch in name:
        if ok(ch):
            cur += ch
        else:
            parts.append(cur)
            cur = ""
    parts.append(cur)
    return max(parts, key=len).strip() or name


def clickable(action) -> bool:
    """Whether the player can carry ``action`` out: not yet effects activated from the banished cards (the
    Graveyard's are, through its viewer: :meth:`Player.graveyard_activate`)."""
    card = action.card
    if action.kind == "activate" and card is not None and card.loc.location == C.LOCATION_REMOVED:
        return False
    return True


ATTACKS_MASKED = ("attack", "*")  # a forbidden entry masking every attack of a phase


def _action_key(action) -> tuple:
    card = action.card
    return (action.kind, card.code if card else 0, card.loc.location if card else 0, card.loc.sequence if card else 0)


def _own_cards(raw: RawDuel) -> tuple:
    """Our field (with chain marks) and hand: what an activation of ours would change."""
    me = raw.players[raw.me]
    return (tuple((c.cid, c.face, c.sideways, c.flags) if c else None for c in me.slots),
            tuple(c.cid for c in me.piles["hand"]))  # fmt: skip


def _board_key(me, op) -> tuple:
    """Everything an action of ours can visibly change; the piles' counts too (a search moves a card from a Deck
    to a hand without touching the field: Millennium Seeker's effect read as "nothing changed")."""
    return (tuple((c.cid, c.face, c.sideways) if c else None for c in me.slots),
            tuple(c.cid for c in me.piles["hand"]), tuple(c.cid for c in me.piles["grave"]),
            tuple((c.cid, c.face) if c else None for c in op.slots), op.lp, me.lp,
            len(me.piles["removed"]), len(op.piles["grave"]), tuple(sorted(me.counts.items())),
            tuple(sorted(op.counts.items())))  # fmt: skip


@dataclass
class Player:
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
        self.cards = default_cards()
        self.passwords = load_md_cid(self.env.artifact_path("md_cid.json"))
        self.prior = OpponentPrior.from_environment(self.env)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(self.out, screen_every=self.screen_every)
        self.observer = ShadowObserver(self.env.rule_flags, self.cards, self.prior, self._score)  # design 10
        scorer = policy_scorer(self.policy, self.cards, self.env.player.starting_lp) if self.policy else None
        self.decider = Decider(self.env.rule_flags, self.cards, self.passwords, self.prior, scorer=scorer,
                               samples=self.samples)  # in-action decisions rebuilt from the current board  # fmt: skip
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
        while time.time() < end:
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
            if dialog == "panel" and not mine and self.read_dialog_type() != OPTION_DIALOG:
                dialog = None  # a dimmed board while MD works: nothing for us yet
            if dialog and dialog not in NEUTRAL and inp.player != raw.me:
                dialog = None  # the opponent holds the input: their dialog, not ours to answer
            key = (inp.mode, mine, dialog, raw.turn, raw.phase)
            stable = stable + 1 if key == last and (mine or dialog) else 0
            last = key
            if stable >= 3:
                return raw, inp, dialog
            time.sleep(0.35)
        raise Stuck(f"nothing to do for {timeout:.0f} s")

    def learn_panel(self, raw: RawDuel) -> None:
        """The card that actually left the hand since the panel was captured is the card the panel showed —
        whether or not it was the one intended — so the panel is labelled with it (mistakes teach too)."""
        if self.pending_panel is None:
            return
        img, before = self.pending_panel
        self.pending_panel = None
        gone = Counter(c.cid for c in before.players[before.me].piles["hand"]) - Counter(
            c.cid for c in raw.players[raw.me].piles["hand"])  # fmt: skip
        if sum(gone.values()) == 1:
            (cid,) = gone
            self.panel.learn(img, cid, self.card_stats(cid))
            self.panel.save()
            self.trace.observe(panel_learned=cid)

    def learn_list(self, raw: RawDuel) -> None:
        """The list candidate that left its place since the pick is the card that was picked: label the panel
        captured on its position with it (right or wrong, so the next list of these cards can be read)."""
        if self.pending_list is None:
            return
        img, req = self.pending_list
        from ygorl.md.request import _locate

        moved = {c.cid for c in req.candidates if c.uid and _locate(raw, c.uid)[1:] != (c.location, c.sequence)}
        if len(moved) == 1:
            self.pending_list = None
            (cid,) = moved
            self.panel.learn(img, cid, self.card_stats(cid))
            self.panel.save()
            self.trace.observe(panel_learned_from_list=cid)

    def card_kind(self, cid: int) -> str:
        """The panel kind (:func:`ygorl.md.panel.panel_kind`) of a hand card."""
        k = self.cards.get(self.passwords.get(cid))
        if k is None:
            return ""
        return "monster" if k.is_monster else "spell" if k.type & C.TYPE_SPELL else "purple"

    def _only_one_of_kind(self, hand, kind: str) -> bool:
        return len({c for c in hand if self.card_kind(c) == kind}) == 1

    def card_stats(self, cid: int) -> tuple[int, int, int] | None:
        k = self.cards.get(self.passwords.get(cid))
        if k is None or not k.is_monster:
            return None
        return (k.level, k.attack, k.defense)

    def verify_hand_card(self, raw: RawDuel, cid: int, xy, icons):
        """Check that the opened hand card is ``cid`` (the hand order on screen is MD's own): from memory
        (:mod:`ygorl.md.selection`) once the open-card object is found, else on the information panel."""
        hand = [c.cid for c in raw.players[raw.me].piles["hand"]]
        told = self.verify_by_memory(raw, cid, xy, icons)
        if told is not None:
            return told
        return self.verify_on_panel(raw, cid, xy, icons, hand)

    def verify_by_memory(self, raw: RawDuel, cid: int, xy, icons):
        """The open card read from memory; calibrates first if needed (opens another hand card, then this one
        again). Returns the command icons of the card ``cid`` once it is open, or None when memory cannot tell."""
        hand = [c.cid for c in raw.players[raw.me].piles["hand"]]
        sel = self.selection.opened(raw)
        if sel is None and len(hand) > 1 and not self.selection.calibrated():
            other = next(p for p in (L.hand_card(i, len(hand)) for i in range(len(hand))) if p != xy)
            self.ui.open_hand_card(other, "calibrate: the open card, from memory")
            if self.selection.opened(raw) is not None:
                icons = self.ui.open_hand_card(xy, f"hand {self.name(cid)} (after calibrating)")
                sel = self.selection.opened(raw)
        if not self.selection.calibrated():
            self.trace.observe(calibration=self.selection.last)
        if sel is None:
            return None
        self.trace.observe(selected={"want": cid, "got": sel[0], "uid": sel[1], "at": list(xy)})
        if sel[0] == cid:
            return icons
        print(f"  memory: the open card is {self.name(sel[0])}, not {self.name(cid)}: searching the hand", flush=True)
        for i in range(len(hand)):
            pos = L.hand_card(i, len(hand))
            if pos == xy:
                continue
            found = self.ui.open_hand_card(pos, f"hand #{i + 1}: looking for {self.name(cid)}")
            sel = self.selection.opened(raw)
            if sel is None:
                return None  # the object moved: the panel decides this time
            if sel[0] == cid:
                self.trace.observe(selected={"want": cid, "found_at": i, "uid": sel[1]})
                return found
        raise Stuck(f"{self.name(cid)} not found in the hand (memory: no position opens it)")

    def verify_on_panel(self, raw: RawDuel, cid: int, xy, icons, hand: list[int]):
        """Check on MD's information panel that the opened card is ``cid``. A different card: try the other hand
        positions until the panel shows ``cid``. Unknown (nothing learned for these cards yet): keep the click, and
        learn from it once memory confirms the action."""
        stats = {c: self.card_stats(c) for c in hand}
        img = img0 = self.ui.capture()
        got, how = self.panel.identify(img, hand, stats)
        kind, want_kind = panel_kind(img), self.card_kind(cid)
        first_unknown = got is None and (not kind or kind == want_kind)
        self.trace.observe(panel={"want": cid, "got": got, "how": how, "kind": kind, "want_kind": want_kind,
                                  "words": name_words(name_signature(img))})  # fmt: skip
        if got == cid or (first_unknown and self.unlearned_is(img, cid, hand)):
            self.pending_panel = (img, raw)
            return icons
        if got is None:
            how = f"a {kind} panel"
        shown = self.name(got) if got is not None else how
        print(f"  panel shows {shown}, not {self.name(cid)}: searching the hand", flush=True)
        unknown = []  # positions whose panel is no card learned so far, of the kind wanted
        last_opened = None
        for i in range(len(hand)):
            pos = L.hand_card(i, len(hand))
            if pos == xy:
                continue
            found = self.ui.open_hand_card(pos, f"hand #{i + 1}: looking for {self.name(cid)}")
            last_opened = i
            img = self.ui.capture()
            seen, how = self.panel.identify(img, hand, stats)
            kind = panel_kind(img)
            if seen == cid or (seen is None and kind == want_kind and self.unlearned_is(img, cid, hand)):
                self.trace.observe(panel={"want": cid, "found_at": i, "how": how or kind})
                self.pending_panel = (img, raw)
                return found
            if seen is None and (not kind or kind == want_kind) and self.width_allows(img, cid):
                unknown.append((i, found, img))
        if first_unknown and self.width_allows(img0, cid):  # the first panel is a candidate too: rule it out
            unknown.append((-1, None, None))
        if cid not in self.panel.names and len(unknown) == 1 and unknown[0][1] is not None:
            # the wanted card was never learned, and every other position showed a card that is not it: by elimination
            i, found, img = unknown[0]
            if i != last_opened:  # open it again; a click on the card still open would close it
                pos = L.hand_card(i, len(hand))
                found = self.ui.open_hand_card(pos, f"hand #{i + 1}: {self.name(cid)} by elimination")
                img = self.ui.capture()
            self.trace.observe(panel={"want": cid, "found_at": i, "how": "elimination"})
            self.pending_panel = (img, raw)
            return found
        raise Stuck(f"{self.name(cid)} not found in the hand by its information panel")

    def width_allows(self, img: np.ndarray, cid: int) -> bool:
        """The name text on the panel has the words of ``cid``'s name (:func:`ygorl.md.panel.name_fits`)."""
        return name_fits(name_signature(img), self.name(cid))

    def unlearned_is(self, img: np.ndarray, cid: int, hand: list[int]) -> bool:
        """A panel no learned card matches shows ``cid`` when its kind and name words fit ``cid`` and no other
        unlearned card of that kind in hand fits as well (Fist of Fate was Set for Millennium Revelation: two
        spells, neither learned yet; Blaze Cannon and Soul Crossing differ in their words)."""
        kind = self.card_kind(cid)
        if not self.width_allows(img, cid):
            return False
        twins = [c for c in dict.fromkeys(hand) if c != cid and self.card_kind(c) == kind
                 and c not in self.panel.names and self.width_allows(img, c)]  # fmt: skip
        return not twins

    def hand_xy(self, raw: RawDuel, cid: int) -> tuple[int, int]:
        """Screen position of the hand card with MD id ``cid`` (MD sorts the hand on screen)."""
        hand = [c.cid for c in raw.players[raw.me].piles["hand"]]
        order = L.hand_order(hand, lambda c: self.passwords.get(c, 0), self.cards.get)
        return L.hand_card(order.index(hand.index(cid)), len(hand))

    def zones_outlined(self) -> bool:
        img = self.ui.capture()
        return any(yellow_highlight(img, *L.my_zone(loc, z)) or zone_outlined(img, *L.my_zone(loc, z))
                   for loc in ("mzone", "szone") for z in range(5))  # fmt: skip

    def outlined_zone(self, raw: RawDuel) -> tuple[str, int]:
        """The first zone MD outlines for a placement (Monster Zones, then Spell & Trap Zones)."""
        img = self.ui.capture()
        for location in ("mzone", "szone"):
            for zone in range(5):
                xy = L.my_zone(location, zone)
                if yellow_highlight(img, *xy) or zone_outlined(img, *xy):
                    return location, zone
        return "mzone", self.free_zone(raw, "mzone")

    def command_at_zone(self, location: str, zone: int, kind: str, order) -> None:
        """When the command click did not take, the zone click makes MD place the selected hand card on that zone
        and ask the command again there (Summon / Set icons above the zone): answer it with the same command."""
        x, y = L.my_zone(location, zone)
        self.ui.sleep(0.6)
        img = self.ui.capture()
        box = (max(x - 200, 0), max(y - 260, 0), min(x + 200, L.REF_W), y - 60)
        icons = []
        for p in sorted(set(L.find_icons(img, *box)) | set(L.blue_icons(img, *box))):
            if is_command_icon(img, *p) and all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 40 for q in icons):
                icons.append(p)  # the two detectors can both report one icon
        if icons:
            # the icons fade in: read their glyphs once drawn (a half-faded Summon reads as Set: Chocolate Magician
            # Girl was Set instead of Summoned)
            self.ui.wait_still(box, timeout=2.4, tolerance=2.0)
            self.trace.observe(command_at_zone=icons)
            self.ui.choose_command(icons, kind, order, "the card on the zone")

    def free_zone(self, raw: RawDuel, location: str) -> int:
        """The leftmost free zone (0-4) of ``location`` (mzone / szone): where a summon or Set is placed."""
        slots = raw.players[raw.me].slots
        rng = range(0, 5) if location == "mzone" else range(7, 12)
        for s in rng:
            if slots[s] is None:
                return s - (0 if location == "mzone" else 7)
        raise Stuck(f"no free {location}")

    # --- decisions ------------------------------------------------------------------------------------------
    def close_phase_menu_unless(self, kind: str) -> None:
        """An open phase menu covers the hand: close it unless the decision is a phase change."""
        if kind not in ("battle_phase", "main2", "end_phase") and self.ui.dialog() == "phase_menu":
            self.ui.click(*PHASE_MENU_CANCEL, "close the phase menu", 1.2)

    def main_phase(self, raw: RawDuel) -> None:
        best, legal = self.decide(raw)
        self.close_phase_menu_unless(best.kind)
        kinds = {}
        for s in legal:
            if s.action.card is not None:
                kinds.setdefault(s.action.card.code, set()).add(s.action.kind)
        self.context = best.kind
        if best.kind == "battle_phase":
            self.attacked.clear()
            return self.ui.phase("battle")
        if best.kind == "end_phase":
            return self.ui.phase("end")
        if best.card is None:
            raise Stuck(f"no click script for {best.kind}")
        cid = next((c for c, p in self.passwords.items() if p == best.card.code), None)
        where = best.card.loc

        if where.location in (C.LOCATION_MZONE, C.LOCATION_SZONE) and best.kind == "activate":
            if where.location == C.LOCATION_MZONE:
                xy = my_monster_xy(where.sequence)
            elif where.sequence < 5:
                xy = L.my_zone("szone", where.sequence)
            else:
                raise Stuck("no position measured for the Field Zone yet")
            if not self.ui.card_command(xy, f"field {self.name(cid)}", "activate", 2.0, kind="activate"):
                # a draw reveal or an effect's animation can cover the field: settle and look once more
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
                if not self.ui.card_command(xy, f"field {self.name(cid)} (again)", "activate", 2.0, kind="activate"):
                    return self.refused(raw, best, f"no Activate for {self.name(cid)} on the field")
            return
        if where.location == C.LOCATION_GRAVE and best.kind == "activate":
            return self.graveyard_activate(raw, best, cid)
        if where.location == C.LOCATION_MZONE and best.kind == "reposition":
            # a monster's change-of-position icon: only taken when it is the card's single icon (no glyph measured)
            xy = my_monster_xy(where.sequence)
            icons = self.ui.open_card(xy, f"field {self.name(cid)}")
            if len(icons) != 1:
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                return self.refused(
                    raw, best, f"{len(icons)} icons on {self.name(cid)}: cannot tell the position change"
                )
            self.ui.click(*icons[0], "change position", 1.5)
            return
        if where.location != C.LOCATION_HAND:
            raise Stuck(f"no click script for {best.kind} from location {where.location}")
        xy = self.hand_xy(raw, cid)
        icons = self.ui.open_hand_card(xy, f"hand {self.name(cid)}")
        if not icons:
            # before taking it for MD refusing: the hand may have been dimmed by an effect resolving; let the screen
            # settle, re-read the hand (positions) and try once more
            self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=10.0)
            raw = self.game.read()
            if not any(c.cid == cid for c in raw.players[raw.me].piles["hand"]):
                raise Stuck(f"{self.name(cid)} left the hand before it was used")
            xy = self.hand_xy(raw, cid)
            icons = self.ui.open_hand_card(xy, f"hand {self.name(cid)} (after the screen settled)")
        try:
            icons = self.verify_hand_card(raw, cid, xy, icons)
        except Stuck as exc:  # the card cannot be told apart from another one not learned yet: decline, do not guess
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
            return self.refused(raw, best, str(exc))
        self.trace.observe(icons=icons)
        card = self.cards[best.card.code]
        order = command_order(card.is_monster, kinds.get(best.card.code, set()))
        if self.ui.choose_command(icons, best.kind, order, self.name(cid)) is None:
            # an effect's resolve glow can pass for the only icon (Millennium Seeker's Summon refused while The True
            # Name glowed): let the board settle, open the card again and look once more before refusing
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
            self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
            icons = self.ui.open_hand_card(xy, f"hand {self.name(cid)} (again, after the board settled)")
            sel = self.selection.read(self.game.read()) if getattr(self, "selection", None) else None
            if (sel is not None and sel[0] != cid) or self.ui.choose_command(
                icons, best.kind, order, self.name(cid)
            ) is None:
                return self.refused(raw, best, f"{self.name(cid)} shows no command")
        if card.is_monster and best.kind == "activate":
            return  # a hand effect: no zone, the card may stay in hand; its prompts are handled by the loop
        self.summon_level = card.level if best.kind in ("summon", "mset") else 0
        if best.kind in ("summon", "mset") and card.level >= 5 and self.ui.dialog() == "yesno":
            # a Tribute Summon / Set asks to confirm first (message 335); the tributes are a field selection next
            self.ui.yesno(True, "yes: tribute")
            self.pick_tributes(raw)
            return
        if best.kind in ("spsummon", "activate") and self.pending_question():
            return  # MD asks first (a cost: Apprentice Illusion Magician's discard): the loop answers, then the zone
            # (a Normal Summon / Set goes to its zone straight away: a passing state once skipped that zone)
        location = "mzone" if card.is_monster else "szone"
        zone = self.free_zone(raw, location)
        self.ui.pick_zone(location, zone, f"zone for {best.kind}")
        self.last_zone = (location, zone)
        self._placed(location, zone)
        self.command_at_zone(location, zone, best.kind, order)
        before = sum(c.cid == cid for c in raw.players[raw.me].piles["hand"])
        for _ in range(10):
            after = self.game.read()
            if sum(c.cid == cid for c in after.players[after.me].piles["hand"]) < before:
                return
            time.sleep(0.5)
        if best.kind in ("spsummon", "activate") and self.pending_question():
            return  # still resolving: a cost or a choice is asked first
        raise Stuck(f"{best.kind} {self.name(cid)}: the card is still in hand after the clicks")

    def graveyard_activate(self, raw: RawDuel, best, cid: int) -> None:
        """An effect activated from our Graveyard (the user: it must be supported): open the Graveyard viewer, open
        the card (its strip position from the pile order in memory; which card is open is read from memory, the
        other positions tried if it is another), and click its Activate icon."""
        grave = raw.players[raw.me].piles["grave"]
        seq = best.card.loc.sequence
        guess = len(grave) - 1 - seq if 0 <= seq < len(grave) else 0  # the strip shows the top card first
        tries = [guess] + [i for i in range(len(grave)) if i != guess]
        self.gy_before = self.ui.capture()
        self.ui.open_graveyard()
        for i in [t for t in tries if t < GY_STRIP_SHOWN]:  # deeper cards need the strip scrolled: not yet
            icons, shot = self.ui.open_graveyard_card(
                i, "activate", f"graveyard #{i + 1}: looking for {self.name(cid)}"
            )
            sel = self.selection.read(self.game.read()) if getattr(self, "selection", None) else None
            self.trace.observe(graveyard={"want": cid, "index": i, "selected": sel and sel[0], "icons": icons})
            if sel is not None and sel[0] == cid:  # the icon's place was never measured: keep what MD showed
                np.save(self.out / f"graveyard-{time.strftime('%Y%m%d-%H%M%S')}-{i}.npy", shot)
            if sel is not None and sel[0] != cid:
                continue  # another card: try the next position
            if sel is None and i != guess:
                continue  # memory cannot tell, and this is not the position the pile order names
            if icons:
                self.ui.click(*icons[0], "activate (graveyard)", 2.0)
                return
            if sel is not None and self.ui.graveyard_command_greyed(self.gy_before, shot):
                # MD offers the command greyed out: its rules differ from the reconstruction's here (Ancient
                # Chant's Graveyard effect, greyed on a turn no copy was activated): masked for the rest of the turn
                self.forbidden.add(((raw.turn, "*"), _action_key(best)))
                self.trace.observe(graveyard_greyed=cid)
            break  # the right card (or the one the order names), and MD shows it no Activate
        self.ui.click(*NEUTRAL_SPOT, "close the graveyard", 0.8)
        self.refused(raw, best, f"no Activate for {self.name(cid)} in the Graveyard")

    def pending_question(self) -> bool:
        """MD is asking something of ours right after a command (a list, a dialog, a field pick), not idle."""
        time.sleep(0.8)
        inp = self.input()
        dialog = self.ui.dialog()
        # not the zone banner / place mode: those are the zone of this very command, picked next
        return dialog in ("list", "radio", "position", "yesno", "option", "notice") or (
            inp.accepting and inp.mode not in (MAIN, BATTLE, PLACE)
        )

    def refused(self, raw: RawDuel, action, why: str) -> None:
        """MD shows no command for an action the reconstruction allows: a state the snapshot does not carry
        (a turn-long restriction, a once-per-turn already used). Masked for this phase; the policy decides again."""
        key = (raw.turn, raw.phase)
        self.forbidden.add((key, _action_key(action)))
        if action.kind == "attack" and sum(1 for k, a in self.forbidden if k == key and a[0] == "attack") >= 2:
            # two monsters refused in one Battle Phase: a phase-wide attack limit the reconstruction cannot see (The
            # Dark Door counts the attacks already made this phase): no more attacks this phase
            self.forbidden.add((key, ATTACKS_MASKED))
            why += "; all attacks masked for this phase"
        self.trace.refusal(action, f"MD refused: {why}")
        print(f"  check {action.kind}: MD refused ({why}); masked for this phase", flush=True)
        self.expect = None

    def note_activation(self, action) -> None:
        """Remember a card activated this turn from off the field (hand, GY, ...): its once-per-turn activation is
        spent, and once the card is gone no slot flag says so (design 10 「重建能还原什么」)."""
        from ygorl.advisor.snapshot import LOCATIONS

        if action.kind != "activate" or action.card is None:
            return
        where = next((n for n, v in LOCATIONS.items() if v == action.card.loc.location), "")
        if where and where not in ("mzone", "szone"):
            self.activated.add((action.card.code, where))

    def decide(self, raw: RawDuel):
        """The policy's top action for the current Main or Battle Phase board (and every legal suggestion)."""
        from ygorl.advisor.advise import advise

        snap = to_snapshot(raw, self.passwords, frozenset(self.attacked), frozenset(self.activated))
        deck = own_deck(raw, self.passwords, self.cards)
        advice = advise(snap, deck, self.policy, self.prior, self.env.rule_flags, cards=self.cards,
                        samples=self.samples, seed=random.randrange(1 << 30), starting_lp=self.env.player.starting_lp)  # fmt: skip
        key = (raw.turn, raw.phase)
        no_attacks = (key, ATTACKS_MASKED) in self.forbidden
        allowed = [g for g in advice.suggestions if (key, _action_key(g.action)) not in self.forbidden
                   and ((raw.turn, "*"), _action_key(g.action)) not in self.forbidden
                   and not (no_attacks and g.action.kind == "attack") and clickable(g.action)]  # fmt: skip
        if not allowed:
            raise Stuck(f"no legal action left in the {snap.phase} reconstruction (MD refused the rest)")
        top = allowed[0]
        name = self.cards[top.action.card.code].name if top.action.card else ""
        print(f"policy ({snap.phase}): {top.action.kind} {name} p={top.prob:.2f}", flush=True)
        self.trace.policy(top.action, top.prob, snap.phase, allowed)
        self.trace.observe(shadow_build=self.observer.decided(top.action, snap, deck))
        self.expect = Expectation(top.action, raw, snap.phase)
        return top.action, allowed

    def battle(self, raw: RawDuel) -> None:
        """The policy chooses: attack with which monster, Main Phase 2 or End Phase, and the attack target (rebuilt
        from the current board, :meth:`ygorl.selfplay.decide.Decider.attack_target`). The rule when the rebuild
        cannot tell: the most valuable monster the attacker beats, else the weakest; direct when the opponent has no
        monster."""
        best, _ = self.decide(raw)
        self.close_phase_menu_unless(best.kind)
        self.context = best.kind
        if best.kind == "main2":
            return self.ui.phase("main2")
        if best.kind == "end_phase":
            return self.ui.phase("end")
        if best.kind != "attack" or best.card is None:
            raise Stuck(f"no click script for {best.kind} in the Battle Phase")
        me, opp = raw.players[raw.me], raw.players[1 - raw.me]
        slot = best.card.loc.sequence
        card = me.slots[slot]
        if card is None:
            raise Stuck(f"the policy attacks with an empty zone {slot}")
        self.attacked.add(card.uid)
        pw = self.passwords.get(card.cid)
        power = self.cards[pw].attack if pw in self.cards else 0
        targets = []
        for s, c in enumerate(opp.slots[:7]):
            if c is None:
                continue
            k = self.cards.get(self.passwords.get(c.cid)) if c.face else None
            value = (k.defense if c.sideways else k.attack) if k is not None else 1500  # face-down: average guess
            targets.append((value < power, value if value < power else -value, s))
        if not self.ui.attack(my_monster_xy(slot), f"attacker {self.name(card.cid)}"):
            return self.refused(raw, best, f"no Attack for {self.name(card.cid)}")
        if targets and self.input().mode == FIELD_PICK:
            s = self.rebuilt("target", raw, slot, frozenset(self.attacked - {card.uid}))
            if s not in {t[2] for t in targets}:
                _, _, s = max(targets)
                self.trace.sub("target", by="rule", rule="target-most-valuable-beaten", slot=s)
            self.trace.observe(target_slot=s)
            if not self.ui.pick_target(opp_monster_xy(s), f"target {s}"):
                raise Stuck("no Confirm icon above the attack target")
        self.context = f"attack lp {opp.lp}"

    def prompt(self, raw: RawDuel, inp: Input) -> None:
        """Chain / open-timing prompts: the policy through the shadow (inside our action), else through a rebuild
        that reaches the same turn and phase (:meth:`ygorl.selfplay.decide.Decider.respond`); the rule when neither
        can tell: respond to the opponent's attacks and in our own End Phase, else pass."""
        phase = PHASES[raw.phase] if raw.phase < len(PHASES) else ""
        theirs = raw.turn_player != raw.me
        # the rule: respond to the opponent's attacks, and to our own triggers in our Draw / Standby / End Phase
        # ("You drew card(s). Activate?" of Palladium Oracle Mahad, a free Special Summon, was passed)
        respond = (theirs and phase == "battle") or (not theirs and phase in ("draw", "standby", "end"))
        self.respond_card = None  # set by a rebuild below, not left over from an earlier one
        pick = None if theirs else self.lockstep("prompt", raw, lambda: self.request_now(raw))
        if pick is None:
            pick = self.rebuilt("respond", raw)
        if pick is not None:
            respond = pick == 0
        # the same prompt again with nothing of ours changed: the activation did not happen (MD can offer a card it
        # then greys the Activate button for), so pass instead of ticking it forever
        seen = (raw.turn, raw.phase, inp.mode, _own_cards(raw))
        if respond and seen == self.responded:
            respond = False
            self.trace.observe(prompt="the response did not take: passing")
        self.responded = seen if respond else None
        if pick is None:
            self.trace.rule("prompt-respond" if respond else "prompt-pass")
        want = getattr(self, "respond_card", None) if pick is not None else None
        self.respond_card = None
        if respond and want is not None:
            # the rebuild's answer is for the window, not for the card MD offers here: tick it, read which card
            # it is from memory, and pass when it is another one (Ultimate Divine-Beast was flipped for a Mirror
            # Force answer)
            self.ui.click(*L.PROMPT_CARD, "tick the offered card", 0.8)
            got = self.selection.read(self.game.read()) if getattr(self, "selection", None) else None
            got_pw = self.passwords.get(got[0]) if got else None
            self.trace.observe(prompt_card={"want": want, "got": got_pw})
            if got_pw is not None and got_pw != want:
                print(f"  prompt: MD offers {self.name(got[0])}, the policy meant another card: pass", flush=True)
                self.ui.respond_prompt(False)
                self.responded = None
                return
            self.ui.click(*L.PROMPT_ACTIVATE, "activate", 2.0)
            self.context = "respond"
            return
        self.ui.respond_prompt(respond)
        if respond:
            self.context = "respond"

    def _score(self, shadow, point):
        """The policy's logits on a shadow decision point (sub-picks inside our own action)."""
        from dataclasses import replace

        from ygorl.advisor.advise import _logits

        return _logits(replace(shadow.rec, point=point), self.policy, self.cards, self.env.player.starting_lp)

    def rebuilt(self, question: str, raw: RawDuel, *args):
        """The policy's answer to ``question`` ("target", "position", "respond") rebuilt from the current board
        (:class:`ygorl.selfplay.decide.Decider`); None when the rebuild cannot tell (the caller's rule decides).
        Recorded as the step's decision (the target: as the attack decision's ``target``)."""
        decider = getattr(self, "decider", None)  # tests build players without one
        if decider is None:
            return None
        if question == "target":
            choice = decider.attack_target(raw, *args)
        elif question == "position":
            action, before = (self.expect.action, self.expect.before) if self.expect is not None else (None, None)
            listed = getattr(self, "listed", (None, ()))
            shadow = getattr(getattr(self, "observer", None), "shadow", None)
            choice = decider.position(raw, self.request_now(raw), action, shadow, before,
                                      listed[1] if listed[0] is self.expect else ())  # fmt: skip
        else:
            choice = decider.respond(raw, self.request_now(raw))
            self.respond_card = choice.detail.get("card")  # the card the policy would activate (a password)
        record = choice.record()
        if choice.pick is None:
            self.trace.observe(rebuild=record)
            print(f"  rebuild ({question}): no decision ({'; '.join(choice.notes[:2])})", flush=True)
            return None
        if question == "target":
            self.trace.sub("target", by="rebuild", **record)
            if choice.shadow is not None:
                self.observer.adopt(choice.shadow, "attack")  # in lockstep again for what the attack asks next
        else:
            self.trace.rebuilt(**record)
        print(f"  rebuild ({question}): {choice.pick} {record['probs']} over {choice.reached}/{choice.samples}",
              flush=True)  # fmt: skip
        return choice.pick

    def lockstep(self, dialog: str, raw: RawDuel, req=None, count: int = 0, zone=None) -> int | None:
        """The option the policy picks for MD's ``dialog`` through the shadow duel (None: no shadow, or it does not
        match; the caller's rule decides). Recorded as the step's decision."""
        observer = getattr(self, "observer", None)  # tests build players without one
        if observer is None or observer.shadow is None:
            return None
        if callable(req):  # read only when there is a shadow to ask
            req = req()
        pick, record = observer.choose(dialog, req, raw.me, self.passwords, count=count, zone=zone)
        if record:
            self.trace.observe(lockstep=record)
        if pick is not None:
            self.trace.shadow(dialog, pick, action=record.get("action"))
            print(f"  shadow ({record.get('ygo')}): {dialog} option {pick + 1} of {record.get('options')}", flush=True)
        elif record:
            print(f"  shadow: {record.get('desync') or record.get('shadow_error')} ({record.get('ygo')})", flush=True)
        return pick

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
        self.ui.pick_zone(location, zone, f"zone for the card ({location})")
        self._placed(location, zone)

    def on_zone_again(self, raw: RawDuel, inp) -> None:
        self.trace.rule("zone-again")
        self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
        self.ui.pick_zone(*self.last_zone, "zone again")
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

    def dialog(self, raw: RawDuel, kind: str) -> None:
        if kind == "position":
            zone = self.free_zone(raw, "mzone")  # MD asks the zone next; the shadow takes the same before the position
            pick = self.lockstep("position", raw, zone=("mzone", zone))
            if pick is None:
                pick = self.rebuilt("position", raw)
            if pick == 1:
                self.ui.pick_position("faceup_defense")
            else:
                if pick is None:
                    self.trace.rule("position-faceup-attack")
                self.ui.pick_position("faceup_attack")
            self.ui.pick_zone("mzone", zone, "zone for the summon")
            self._placed("mzone", zone)
        elif kind == "option":
            req = self.request_now(raw)
            n = len(req.choices)
            pick = self.lockstep("option", raw, req)
            if pick is None:
                pick = self.intended_option(req)
            if pick is None:
                pick = next((i for i, c in enumerate(req.choices) if c >> 31), 0)  # the first enabled effect
                self.trace.rule("option-first-enabled", choices=[hex(c) for c in req.choices], pick=pick)
            self.trace.observe(option_choices=[hex(c) for c in req.choices])
            y, _ = option_tabs(self.ui.capture(), n)
            for attempt in range(3):
                self.ui.click(*option_tab(pick, n, y), f"effect {pick + 1} of {n}", 0.8)
                y, selected = option_tabs(self.ui.capture(), n)
                if selected == pick:
                    break
            else:
                raise Stuck(f"effect tab {pick + 1} of {n} does not stay selected")
            ok = [b.centre for b in find_buttons(self.ui.capture()) if b.label == "enabled" and b.centre[1] > y + 200]
            self.ui.click(*(min(ok, key=lambda c: abs(c[0] - 1026)) if ok else OPTION_OK), "ok", 1.5)
        elif kind == "radio":
            rows = len(radio_rows(self.ui.capture()))
            if not rows:  # a passing frame looked like the radio dialog
                self.trace.observe(radio="gone before the pick")
                return
            pick = self.lockstep("radio", raw, count=rows)
            if pick is None:
                self.trace.rule("radio-first")  # e.g. Pot of Extravagance: banish 3 (the first row) rather than 6
            self.trace.observe(radio_rows=self.ui.pick_radio(pick or 0))
        elif kind == "declare":
            # MD does not list every name ("Ancient Chant": "No applicable cards found"): the next one then
            for name in self.declare_names(raw)[:6]:
                self.trace.rule("declare-most-common-in-deck", name=name)
                try:
                    self.ui.declare(searchable(name), self.mouse.type_text, self.mouse.backspace, self.mouse.enter)
                    return
                except Stuck as exc:
                    self.trace.observe(declare_failed=str(exc))
            raise Stuck("no declarable name found among the Deck's cards")
        elif kind == "minimised":
            self.trace.rule("restore-minimised")
            self.ui.click(*MINIMISED_RESTORE, "restore the minimised panel", 1.5)
        elif kind == "notice":
            xy = notice_button(self.ui.capture())
            if xy is None:
                return  # gone already
            self.trace.rule("notice-confirm")
            self.ui.click(*xy, "confirm the notice", 1.2)
        elif kind == "phase_menu":
            self.trace.rule("phase-menu-cancel")
            self.ui.click(*PHASE_MENU_CANCEL, "close the phase menu", 1.2)
        elif kind == "yesno":
            pick = self.lockstep("yesno", raw)
            if pick is None and self.context == "summon" and self.summon_level < 5:
                # a Normal Summon / Set of Level <= 4 asks no Tribute question: this is no yes / no of ours
                self.trace.observe(yesno="ignored: a Level <= 4 summon asks no Tribute question")
                return
            if pick is None and self.context == "summon":
                pick = 0  # our own summon asks to confirm the Tribute Summon (message 335): the tributes come next
                self.trace.rule("yesno-yes-tribute")
            elif pick is None and self.context in ("activate", "spsummon", "respond"):
                # our own card asks about an optional part of the effect we chose to activate ("add the designated
                # card from the Deck to your hand?" of The True Name): take it
                pick = 0
                self.trace.rule("yesno-yes-own-effect")
            elif pick is None:
                self.trace.rule("yesno-no")
            self.ui.yesno(pick == 0)
            if self.context == "summon" and pick == 0:
                self.context = "tribute"
        elif kind == "list":
            if not L.list_cards(self.ui.capture()):  # a passing frame (a turn change) looked like a list
                self.trace.observe(list="gone before the pick")
                return
            shown = len(L.list_cards(self.ui.capture()))
            req = self.request_now(raw, list_shown=True)
            for _ in range(8):  # MD fills its list a moment after the panel shows: one stale candidate was read once
                if len(req.candidates) >= shown:
                    break
                time.sleep(0.3)
                raw = self.game.read()
                req = self.request_now(raw, list_shown=True)
            stale = len(req.candidates) < shown
            if stale:
                # (a choice without a list whose check is not modelled, e.g. Millennium Revelation's hand card;
                # Apprentice Illusion Magician's discard read a stale list until its check was, design 10)
                self.trace.observe(list_request={"stale": [[c.cid, c.location, c.sequence] for c in req.candidates],
                                                 "shown": shown})  # fmt: skip
            self.listed = (self.expect, tuple(c.cid for c in req.candidates))  # names a monster summoned next
            need = next((v for k, v in req.params if k == 8), 1)  # "choose N cards" (Fusion Materials)
            if 1 < need <= len(L.list_cards(self.ui.capture())):
                self.trace.rule("list-set-first-accepted", need=need)  # MD's list here is not the one in memory
                self.trace.observe(list_set=self.ui.pick_list_set(need))
                self.context = "list"
                return
            pick = None if stale else self.lockstep("list", raw, req)
            if pick is None:
                # a rule pick has no card in mind: the first card on screen (not memory's first candidate: the list
                # can scroll and its order differs, an audit found Dark Magician sent for the "first" Ra)
                self.trace.rule("list-first-on-screen")
                index, ticked = 0, False
            else:
                index, ticked = self.find_in_list(req, pick)
            self.trace.observe(list_cards=self.ui.pick_list_cards([index], ticked=ticked), list_index=index,
                               list_candidates=[[c.cid, c.location, c.sequence] for c in req.candidates])  # fmt: skip
            self.context = "list"
        else:
            # an animation (battle damage, a summon) can pass for a dialog: only explore what is still there
            self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
            if self.ui.dialog() != kind:
                self.trace.observe(dialog=f"{kind} gone after the screen settled")
                return
            self.explore(f"no script for the {kind} screen")

    def field_target(self, raw: RawDuel, lit) -> tuple[int, int] | None:
        """The field card the policy picks through the shadow for MD's field selection (tributes, targets), if the
        shadow matches and the card is one MD highlights; else None (the caller's rule decides)."""
        req = None
        try:
            req = self.request_now(raw)
        except Exception:  # noqa: BLE001 - no request: the rule decides
            return None
        pick = self.lockstep("list", raw, req)
        if pick is None:
            return None
        c = req.candidates[pick]
        xy = None
        if c.location == "mzone" and c.sequence is not None:
            xy = my_monster_xy(c.sequence) if c.player == raw.me else opp_monster_xy(c.sequence)
        elif c.location == "szone" and c.player == raw.me and c.sequence is not None and c.sequence < 5:
            xy = L.my_zone("szone", c.sequence)
        # frames are found a little off the slot's measured point ((1219, 740) for (1232, 750)): near is enough
        near = [q for q in lit if xy is not None and abs(q[0] - xy[0]) + abs(q[1] - xy[1]) < 60]
        if not near:
            self.observer.shadow = None  # it took a pick MD will not see: rebuild at the next command point
            self.trace.observe(lockstep_field=f"no highlighted card at {xy} for candidate {pick}")
            return None
        return near[0]

    def summon_pick(self, raw: RawDuel) -> None:
        """Mode 6 (MD's "summon a monster": an effect or a Tribute Summon asking for the monster): MD keeps no list
        for it (the list in memory is stale) and shows a Summon icon only on the hand monsters it offers. So the
        hand positions are opened one by one and the first that shows a Summon icon is summoned; which card that
        was is read from memory afterwards (no need to tell the cards apart on the panel, which failed for cards not
        learned yet: The Winged Dragon of Ra). Positions are tried strongest-guess first. The zone, tributes and any
        further dialog are the loop's next steps."""
        me = raw.players[raw.me]
        hand = [c.cid for c in me.piles["hand"]]

        def strength(cid):
            k = self.cards.get(self.passwords.get(cid))
            if k is None or not k.is_monster:
                return -1
            return k.attack if k.attack >= 0 else 1000 * k.level  # "?" ATK: count the level

        guessed = sorted(range(len(hand)), key=lambda i: strength(hand[i]), reverse=True)
        order = list(dict.fromkeys([L.hand_card(L.hand_order(hand, lambda c: self.passwords.get(c, 0),
                                                              self.cards.get).index(i), len(hand)) for i in guessed]
                                   + [L.hand_card(i, len(hand)) for i in range(len(hand))]))  # fmt: skip
        self.trace.rule("summon-pick-first-offered", hand=hand)
        for n, xy in enumerate(order):
            icons = self.ui.open_hand_card(xy, f"hand position {n + 1}: offered for the summon?")
            if icons and self.ui.choose_command(icons, "summon", ["summon", "mset"], "the offered monster") is not None:
                self.context = "summon"
                self.summon_level = 5  # unknown until memory says: let MD's own Tribute question through
                return
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
        raise Stuck(f"no hand card shows a Summon icon for the summon pick ({len(order)} positions tried)")

    def intended_option(self, req) -> int | None:
        """The tab of the effect the policy activated: its action's description (password * 16 + effect index) is
        the dialog choice's (:func:`ygorl.selfplay.observe.md_option_desc`). None when it names no tab."""
        from ygorl.selfplay.observe import md_option_desc

        action = self.expect.action if self.expect is not None else None
        if action is None or action.kind != "activate" or not action.description:
            return None
        tabs = [i for i, c in enumerate(req.choices) if md_option_desc(c, self.passwords) == action.description]
        if len(tabs) != 1:
            return None
        self.trace.rule("option-the-policy-activated", description=action.description, pick=tabs[0])
        return tabs[0]

    def declare_name(self, raw: RawDuel) -> str:
        return self.declare_names(raw)[0]

    def declare_names(self, raw: RawDuel) -> list[str]:
        """The cards to declare (The True Name and the like), best first: the cards with the most copies left in our
        Deck (our decklist minus what is visible), the likeliest top card. The policy does not decide it yet."""
        deck = own_deck(raw, self.passwords, self.cards)
        me = raw.players[raw.me]
        seen = Counter(self.passwords.get(c.cid) for z in ("hand", "grave", "removed") for c in me.piles[z])
        seen += Counter(self.passwords.get(c.cid) for c in me.slots if c is not None)
        left = Counter(deck.main) - seen
        # a name the keyboard types whole finds exactly that card; MD's search found nothing for "Ra's Disciple"
        # (its apostrophe is not the ASCII one), so plainly spelled names come first
        plain = lambda p: plain_name(self.cards[p].name)  # noqa: E731
        monster = lambda p: self.cards[p].is_monster  # noqa: E731 - monsters first: found by MD every time
        ranked = sorted(left, key=lambda p: (plain(p), monster(p), left[p], p), reverse=True)
        ranked = ranked or list(dict.fromkeys(deck.main))
        return list(dict.fromkeys(self.cards[p].name for p in ranked))

    def find_in_list(self, req, pick: int) -> tuple[int, bool]:
        """The screen position of candidate ``pick`` in MD's card list, and whether it is ticked already.

        MD's list is not memory's: the order differs (an audit found 4 of 4 picks off) and identical copies are
        listed once (14 candidates in memory, 6 cards on screen). So the screen cards are identified on the
        information panel (tick, read, untick) and matched to the wanted card by password, not by position; by
        elimination when the panel names every other distinct card. A pick that cannot be confirmed is recorded
        as unverified. The panel of the card finally picked is kept to be learned once memory shows what moved."""
        cards = L.list_cards(self.ui.capture())
        cands = [c.cid for c in req.candidates]
        n = len(cards)
        if not cands or not 0 <= pick < len(cands):
            return 0, False
        pw = lambda cid: self.passwords.get(cid, cid)  # noqa: E731
        want = cands[pick]
        if n <= 1 or len({pw(c) for c in cands}) <= 1:
            return 0, False  # one card, or only copies of one card: any position is the wanted card
        stats = {c: self.card_stats(c) for c in set(cands)}
        seen: dict[int, int | None] = {}
        panels = {}
        for i in range(n):
            self.ui.click(*cards[i], f"#{i + 1} of {n}: which card?", 0.9)
            img = self.ui.capture()
            got, how = self.panel.identify(img, list(dict.fromkeys(cands)), stats)
            seen[i], panels[i] = got, img
            if got is not None and pw(got) == pw(want):
                self.trace.observe(list_check={"want": want, "seen": seen, "by": how})
                self.pending_list = (img, req)
                return i, True
            self.ui.click(*cards[i], f"untick #{i + 1}", 0.6)
        unknown = [i for i, got in seen.items() if got is None]
        named = {pw(got) for got in seen.values() if got is not None}
        if len(unknown) == 1 and {pw(c) for c in cands} - named == {pw(want)}:
            i = unknown[0]  # every other distinct card was named: the one the panel could not name is it
            self.trace.observe(list_check={"want": want, "seen": seen, "by": "elimination"})
            self.pending_list = (panels[i], req)
            return i, False
        i = unknown[0] if unknown else 0
        self.trace.observe(list_check={"want": want, "seen": seen, "unverified": True})
        print(f"  list: {self.name(want)} not found on the panel; position {i + 1} is a guess", flush=True)
        self.pending_list = (panels[i], req)
        return i, False

    def pick_tributes(self, raw: RawDuel) -> None:
        """Tributes: the policy's picks through the shadow; without a shadow, highlighted monsters of the advisor's,
        lowest ATK first. Each is confirmed, until none is left highlighted."""
        me = raw.players[raw.me]
        taken: set = set()  # slots picked already: they stay framed, and a click would take the pick back
        for n in range(3):
            if n:
                for _ in range(8):  # the input mode flickers while a pick animates (a third Tribute for Ra was lost)
                    if self.input().mode == FIELD_PICK:
                        break
                    time.sleep(0.25)
                else:
                    return  # MD took the tributes: the summoned monster glowing in their zone is not another pick
            lit = set(self.ui.highlighted())
            mine = [(s, c) for s, c in enumerate(me.slots[:7]) if c and my_monster_xy(s) in lit and s not in taken]
            if not mine:
                return
            xy = self.field_target(self.game.read(), {my_monster_xy(s) for s, _ in mine})
            if xy is not None:
                if not self.ui.card_command(xy, "tribute (policy)", "confirm tribute", 2.0):
                    raise Stuck("no Confirm icon above the tribute")
                taken |= {s for s, _ in mine if my_monster_xy(s) == xy}
                continue
            self.trace.rule("tribute-lowest-atk")

            def atk(c):
                k = self.cards.get(self.passwords.get(c.cid))
                return k.attack if k is not None else 0

            s, c = min(mine, key=lambda t: atk(t[1]))
            if not self.ui.card_command(my_monster_xy(s), f"tribute {self.name(c.cid)}", "confirm tribute", 2.0):
                raise Stuck("no Confirm icon above the tribute")
            taken.add(s)

    @staticmethod
    def occupied(raw: RawDuel, spots) -> list:
        """The highlighted spots that hold a monster: lava glowing in an empty zone passed for a highlight."""
        me, op = raw.players[raw.me], raw.players[1 - raw.me]
        cards = [my_monster_xy(s) for s in range(7) if me.slots[s] is not None]
        cards += [opp_monster_xy(s) for s in range(7) if op.slots[s] is not None]
        return [xy for xy in spots if any(abs(xy[0] - c[0]) + abs(xy[1] - c[1]) < 80 for c in cards)]

    def offered_field_cards(self, raw: RawDuel) -> list[tuple[int, int]]:
        """Screen positions of the monsters MD's current request offers (memory), own row first."""
        try:
            req = self.request_now(raw)
        except Exception:  # noqa: BLE001 - no readable request: the frames decide
            return []
        out = []
        for c in req.candidates:
            if c.location == "mzone" and c.sequence is not None and 0 <= c.sequence < 7:
                slots = raw.players[c.player].slots if c.player in (0, 1) else ()
                if c.sequence < len(slots) and slots[c.sequence] is not None:
                    out.append(my_monster_xy(c.sequence) if c.player == raw.me else opp_monster_xy(c.sequence))
        self.trace.observe(offered_field=out)
        return sorted(out, key=lambda xy: -xy[1])

    def field_pick(self, raw: RawDuel) -> None:
        """Mode 8 outside an attack: pick the highlighted cards one by one and confirm each."""
        episode = (raw.turn, raw.phase)
        if self.field_picked[0] != episode:
            self.field_picked = (episode, [])
        picked = self.field_picked[1]
        near = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) < 40  # noqa: E731
        lit = self.occupied(raw, self.ui.highlighted())
        # no frame seen at all: the slots of the monsters MD's request offers (the opponent's gold-framed back row
        # hid the yellow frames of a Tribute for Ra - Sphere Mode). Only then: the request can be stale (it named an
        # opponent's monster during our own three-Tribute summon, and the click found no Confirm five times)
        if not [xy for xy in lit if not any(near(xy, p) for p in picked)]:
            lit = lit + [xy for xy in self.offered_field_cards(raw) if not any(near(xy, q) for q in lit)]
        if not [xy for xy in lit if not any(near(xy, p) for p in picked)]:
            # nothing seen and nothing in the request: every monster on the field not picked yet, opponent's first
            # (a Tribute's opponent monster sat under the banner, its frame hidden); only one that shows a Confirm
            # icon is taken (card_command)
            me, op = raw.players[raw.me], raw.players[1 - raw.me]
            lit = lit + [opp_monster_xy(s) for s in range(7) if op.slots[s] is not None]
            lit = lit + [my_monster_xy(s) for s in range(7) if me.slots[s] is not None]
            self.trace.rule("field-pick-every-monster")
        # cards picked earlier in this selection stay framed: clicking one again would take it back (a three-
        # Tribute summon picked its first Tribute over and over)
        fresh = [xy for xy in lit if not any(near(xy, p) for p in picked)]
        self.trace.observe(highlighted=lit, picked_before=picked)
        if not fresh:
            raise Stuck("a field selection with no highlighted card left to pick")
        xy = self.field_target(raw, fresh)
        if xy is None:
            # without the shadow: a target of the opponent's first (a destroy / target effect offering both sides
            # must not hit our own card); tributes are picked by :meth:`pick_tributes`, not here
            theirs = [p for p in fresh if p[1] < 640]
            xy = (theirs or fresh)[0]
            self.trace.rule("field-pick-opponent-first" if theirs else "field-pick-first")
        for target in [xy] + [q for q in fresh if q != xy]:
            if self.ui.card_command(target, "highlighted card", "confirm"):
                picked.append(target)
                return
            # no Confirm on this one (Millennium Seeker, framed among three Tribute candidates): another
            self.trace.observe(field_pick_no_confirm=list(target))
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
        raise Stuck("no Confirm icon above any selected card")

    # --- loop -----------------------------------------------------------------------------------------------
    def record_stuck(self, why: str) -> Path:
        d = self.out / f"stuck-{time.strftime('%Y%m%d-%H%M%S')}"
        n = 1
        while d.exists():  # two records in one second (a recovery that failed at once) ended the series
            d, n = self.out / f"stuck-{time.strftime('%Y%m%d-%H%M%S')}-{n}", n + 1
        d.mkdir(parents=True)
        np.save(d / "screen.npy", self.ui.capture())
        for i, frame in enumerate(self.ui.evidence):
            np.save(d / f"evidence-{i}.npy", frame)
        self.ui.evidence.clear()
        raw, inp = self.game.read(), self.input()
        step = dict(self.trace.rec or {}, clicks=list(self.ui.clicks))
        (d / "state.json").write_text(json.dumps({"why": why, "context": self.context, "input": inp.__dict__,
                                                  "turn": raw.turn + 1, "phase": raw.phase, "step": step,
                                                  "log": self.trace.recent(30)}, indent=1))  # fmt: skip
        self.trace.stuck(why, str(d))
        return d

    def run(self, max_steps: int = 400) -> int:
        try:
            return self._run(max_steps)
        finally:
            self.trace.end_step(self.ui.take_clicks())

    def run_games(self, games: int, max_steps: int) -> int:
        """Play ``games`` duels in a row: after each one, press Retry on MD's result screen (the same Solo duel
        again) and wait for the new duel. Stops at the first duel that does not finish on its own."""
        try:
            first = self.game.read()
        except Exception:  # noqa: BLE001 - no duel state: the menus
            first = None
        if first is None or first.finished:  # started on a result screen or in the stage menu: get a duel first
            won = first is not None and self._result() == RESULT_WIN
            self.next_duel_from_menu() if first is None or won else self.restart_duel()
        for g in range(games):
            print(f"=== game {g + 1}/{games}", flush=True)
            code = self.run(max_steps)
            raw = self.game.read()
            result = self._result()
            for _ in range(20):  # MD sets the result a moment after the finish flag (0 read twice in a series)
                if result in (1, 2) or code != 0:
                    break
                time.sleep(0.5)
                result = self._result()
            summary = {"game": g + 1, "exit": code, "finished": raw.finished, "result": result,
                       "lp": [raw.players[raw.me].lp, raw.players[1 - raw.me].lp], "turn": raw.turn + 1}  # fmt: skip
            self.trace.game(**summary)
            print(f"=== game {g + 1} summary {summary}", flush=True)
            if code != 0 and not raw.finished and g + 1 < games:
                # the player gave up on this duel: restart it from the settings menu and go on with the series
                try:
                    self.retry_from_settings()
                    continue
                except Stuck as exc:
                    print(f"  restart from the settings menu failed: {exc}", flush=True)
            if code != 0:
                return code
            if g + 1 < games:
                for attempt in range(3):  # MD can be slow to answer Retry (7 clicks once) or to load the duel
                    try:
                        if result == RESULT_WIN and attempt == 0:
                            self.next_duel_from_menu()  # a won Solo duel offers no Retry: through the stage menu
                        elif self._new_duel():
                            self.reset_game_state()
                        else:
                            self.restart_duel()
                        break
                    except Stuck as exc:
                        print(f"  restart attempt {attempt + 1}: {exc}", flush=True)
                else:
                    raise Stuck("no new duel after three restart attempts")
        return 0

    def _result(self) -> int:
        return struct.unpack("<I", self.game.memory.read(self._key() + self.game.offsets.duel_result, 4))[0]

    def _key(self) -> int:
        return struct.unpack("<Q", self.game.memory.read(self.game.module.base + self.game.offsets.key_ptr, 8))[0]

    def park_pointer(self) -> None:
        """Rest the (window-only) pointer on an empty spot of the board: left over a pile from the menus, its
        tooltip stood over the opponent's turn and the duel waited for 240 s."""
        try:
            self.mouse.refresh_window()
            self.mouse.move(*NEUTRAL_SPOT)
        except Exception:  # noqa: BLE001 - cosmetic
            pass

    def _new_duel(self) -> bool:
        try:
            raw = self.game.read()
        except Exception:  # noqa: BLE001 - the menus: no duel state
            return False
        return not raw.finished and raw.turn == 0

    def next_duel_from_menu(self, timeout: float = 300) -> None:
        """After a win: VICTORY's Confirm, the stage menu, the next stage (or the same one again) and Play, until a
        new duel runs (:mod:`ygorl.md.menu`; the user allowed these clicks)."""
        from ygorl.md.input import MenuMouse
        from ygorl.md.menu import MenuWalker

        time.sleep(6.0)  # the win animation
        walker = MenuWalker(MenuMouse(self.game).click, self._capture, self._new_duel)
        ok = walker.run(timeout)
        self.trace.observe(menu=[[k, list(xy)] for k, xy in walker.log])
        if not ok:
            raise Stuck(f"no new duel after the menus ({len(walker.log)} clicks)")
        self.mouse.refresh_window()
        self.park_pointer()
        self.reset_game_state()
        time.sleep(3.0)

    def retry_from_settings(self) -> None:
        """Restart a duel the player gave up on: the settings gear, Retry, Yes; then the new duel as after the
        result screen's Retry (a stage may ask who goes first)."""
        before = self.game.read()
        for attempt in range(3):
            # a focused text box (the declare search) swallows the gear's click: an empty spot takes the focus first
            self.ui.click(*NEUTRAL_SPOT, "an empty spot (drop the focus)", 1.2)
            self.ui.click(*SETTINGS_GEAR, "settings: restart the duel the player gave up on", 2.0)
            self.ui.click(*SETTINGS_RETRY, "retry", 2.0)
            self.ui.click(*SETTINGS_RETRY_YES, "yes: restart from the beginning", 3.0)
            self.ui.take_clicks()
            end = time.time() + 40
            while time.time() < end:  # the old duel's state lingers a moment: wait for the new one to replace it
                try:
                    now = self.game.read()
                except Exception:  # noqa: BLE001 - loading
                    return self.restart_duel()
                if _progress(now, Input(0, False, 0)) != _progress(before, Input(0, False, 0)):
                    return self.restart_duel()
                time.sleep(1.0)
            print(f"  settings restart attempt {attempt + 1}: the duel did not change", flush=True)
        raise Stuck("the settings menu's Retry did not restart the duel")

    def restart_duel(self, timeout: float = 300) -> None:
        """MD's result screen: press Retry, then wait until a new duel is running (not finished, turn 1)."""
        time.sleep(6.0)  # the win / loss animation
        end = time.time() + timeout
        last_click = 0.0
        while time.time() < end:
            try:
                raw = self.game.read()
            except Exception:
                raw = None
            if raw is not None and not raw.finished and raw.turn == 0:
                self.park_pointer()
                self.reset_game_state()
                time.sleep(3.0)
                return
            if time.time() - last_click > 8:
                from ygorl.md import menu
                from ygorl.md.input import MenuMouse

                img = self._capture()
                if menu.screen(img) == "turn_order":  # some stages ask who goes first before the duel again
                    MenuMouse(self.game).click(*menu.target(img, "turn_order"))
                    print("  menu: turn_order after Retry", flush=True)
                    last_click = time.time()
                elif raw is not None and raw.finished:
                    self.ui.click(*RESULT_RETRY, "retry the duel", 2.0)
                    self.ui.take_clicks()
                    last_click = time.time()
            time.sleep(1.0)
        raise Stuck("the duel did not restart after Retry")

    def _placed(self, location: str, zone: int) -> None:
        observer = getattr(self, "observer", None)  # observe only; tests build players without one
        if observer is not None:
            self.trace.observe(shadow_place=observer.placed(location, zone))

    MAX_RECOVERIES_PER_STATE, MAX_RECOVERIES_PER_DUEL = 4, 30

    def recover(self, why: str, raw: RawDuel, inp: Input, dialog: str | None, state) -> bool:
        """Self-recovery after a failure (design 10): record it, then escalate per state — (1) mask the action that
        led here, close what may be open, let the screen settle and decide again; (2) answer the shown dialog
        differently; (3) move on to the next phase. False (give up) after too many attempts."""
        key = state[:-1] if isinstance(state, tuple) and len(state) > 4 else state  # not the refusal count
        self.recoveries[key] = n = self.recoveries.get(key, 0) + 1
        self.recovery_count += 1
        record = self.record_stuck(why)
        print(f"RECOVER {n} ({self.recovery_count} this duel): {why} -> {record}", flush=True)
        self.trace.observe(recovery={"why": why, "level": n, "record": str(record)})
        if n > self.MAX_RECOVERIES_PER_STATE or self.recovery_count > self.MAX_RECOVERIES_PER_DUEL:
            print(f"STUCK: giving up after {n} recoveries on this state ({self.recovery_count} this duel)", flush=True)
            return False
        if self.expect is not None:  # the policy's last action did not happen as intended: do not repeat it now
            self.forbidden.add(((raw.turn, raw.phase), _action_key(self.expect.action)))
        self.expect, self.recent = None, []
        mine = inp.accepting and inp.player == raw.me
        try:
            if n == 1:
                self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot", 1.0)
                self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=10.0)
            elif n == 3 and (dialog not in (None, "phase_menu") or (mine and inp.mode not in (MAIN, BATTLE, 1))):
                self.explore(f"recovery: {why}", choice=1)  # the scripted alternatives did not help: explore
            elif n == 2 or n == 3:
                alt = n - 1  # the 1st / 2nd alternative
                if mine and inp.mode in (OPEN_PROMPT, CHAIN_PROMPT):  # its panel looks like a list: pass first
                    self.ui.respond_prompt(False)
                elif dialog == "radio":
                    self.ui.pick_radio(alt)
                elif dialog == "list":
                    cards = self.ui.pick_list_cards([alt])
                    self.trace.rule(f"recover-list-#{alt + 1}", cards=len(cards))
                elif dialog == "yesno":
                    self.ui.yesno(alt == 1, "recover: the other answer")
                elif dialog == "banner" or (mine and inp.mode == FIELD_PICK):
                    lit = self.occupied(raw, self.ui.highlighted())
                    if len(lit) > alt:
                        self.ui.card_command(lit[alt], "recover: another highlighted card", "confirm")
                elif mine and inp.mode in (MAIN, BATTLE):
                    # the failed action is masked: decide again rather than give up the phase (that comes at 4)
                    self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot, decide again", 1.0)
                else:
                    self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot", 1.0)
                    self.ui.continue_draw()
            else:
                if mine and inp.mode in (MAIN, BATTLE):
                    self.ui.phase("end")
                else:
                    self.ui.click(*NEUTRAL_SPOT, "recover: click an empty spot", 1.0)
        except Exception as exc:  # a recovery step that fails is just a failed attempt
            print(f"  recovery step failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(1.0)
        return True

    def reset_game_state(self) -> None:
        self.attacked, self.forbidden, self.recent, self.last_zone = set(), set(), [], None
        self.activated = set()
        self.recoveries, self.recovery_count = {}, 0
        self.field_picked = ((-1, -1), [])
        self.turn_seen, self.phase_seen, self.expect, self.context = -1, (-1, -1), None, ""
        self.responded = None

    def _run(self, max_steps: int) -> int:
        for step in range(max_steps):
            self.trace.end_step(self.ui.take_clicks())  # the previous step, with its clicks
            self.mouse.refresh_window()
            try:
                raw, inp, dialog = self.wait_turn()
            except Stuck as exc:  # nothing to do for a long time: MD may wait for a click we do not see
                raw, inp, dialog = self.game.read(), self.input(), self.ui.dialog()
                if not self.recover(str(exc), raw, inp, dialog, ("idle", raw.turn, raw.phase)):
                    return 3
                continue
            solo_gate(raw)
            self.trace.begin(step, raw, inp, dialog)
            req = read_request(self.game.memory, self.game.module.base, self.game.offsets, raw)
            if dialog == "panel" and not (req.dialog_type == OPTION_DIALOG and len(req.choices) >= 2):
                dialog = None  # a dark band without yes / no labels: a dimmed or dark board (classify maps the rest)
            if inp.accepting and inp.player == raw.me:
                self.trace.observe(md_request={"kind": req.kind, "candidates": [[c.cid, c.location, c.sequence, c.player]
                                                                                for c in req.candidates],
                                               "params": req.params, "min": req.min, "max": req.max})  # fmt: skip
                compared = self.observer.request(req, raw.me, self.passwords)
                if compared is not None:
                    self.trace.observe(shadow_vs_md=compared)
                    print(
                        f"  shadow vs MD: {compared.get('md')} / {compared.get('ygo', {}).get('decision')} "
                        f"consistent={compared.get('consistent')}",
                        flush=True,
                    )
            if self.trace.wants_screen(step):
                self.trace.save_screen(self.ui.capture())
            self.save_memory_blocks()
            if dialog == "over":
                print("duel over, LP", [p.lp for p in raw.players])
                return 0
            if (raw.turn, raw.phase) != self.phase_seen:
                # the engine accepts input before MD's turn / phase animations (the drawn card shown big) are over
                self.phase_seen = (raw.turn, raw.phase)
                time.sleep(1.0)
                self.ui.wait_still()  # e.g. the drawn card flying to the hand: hand positions count it already
            if raw.turn != self.turn_seen:
                self.turn_seen, self.attacked, self.activated = raw.turn, set(), set()
            if inp.mode not in (FIELD_PICK, FIELD_PICK_TRIBUTE):
                self.field_picked = ((-1, -1), [])  # the field selection is over: the next one starts fresh
            me, op = raw.players[raw.me], raw.players[1 - raw.me]
            # the turn too: the same prompt in two End Phases is not a loop; the last entry (refusals) is progress
            state = (
                raw.turn,
                inp.mode,
                dialog,
                raw.phase,
                _board_key(me, op),
                req.params,  # successive "activate?" windows at different timings look alike otherwise
                len(self.forbidden) + len(self.field_picked[1]),
            )  # a pick of a multi-pick is progress too
            self.recent = (self.recent + [state])[-8:]
            # the Draw Phase takes a few clicks (badge, Deck) by design: it has its own budget of attempts
            in_draw = inp.mode == 1 and dialog is None and self.draw_attempts < 6
            if len(self.recent) >= 3 and len(set(self.recent[-3:])) == 1 and not in_draw:
                if not self.recover("the same state three steps in a row", raw, inp, dialog, state):
                    return 3
                continue
            if self.recent.count(state) >= 3 and not in_draw:  # a cycle: answers that keep undoing each other
                self.recent = []
                if not self.recover("the same state three times in eight steps", raw, inp, dialog, state):
                    return 3
                continue
            phase = PHASES[raw.phase] if raw.phase < len(PHASES) else str(raw.phase)
            print(f"[{step}] turn {raw.turn + 1} {phase} mode {inp.mode} dialog {dialog} "
                  f"LP {raw.players[raw.me].lp}/{raw.players[1 - raw.me].lp}", flush=True)  # fmt: skip
            try:
                if self.expect is not None and not dialog and inp.mode in (MAIN, BATTLE):
                    problem = check_expectation(self.expect, raw, self.passwords)
                    self.trace.check(self.expect.action, problem)
                    self.learn_panel(raw)
                    self.learn_list(raw)
                    print(f"  check {self.expect.action.kind}: {'ok' if problem is None else problem}", flush=True)
                    if problem is None:
                        self.note_activation(self.expect.action)
                    self.expect = None
                    if problem is not None:
                        raise Stuck(f"the game does not show the policy's action: {problem}")
                mine = inp.accepting and inp.player == raw.me
                kind = classify(mine=mine, ours=inp.player == raw.me, accepting=inp.accepting, mode=inp.mode,
                                dialog_type=req.dialog_type, choices=len(req.choices), screen=dialog,
                                context=self.context, zone_pending=self.last_zone is not None)  # fmt: skip
                self.trace.observe(kind=kind)
                handler = self.HANDLERS.get(kind)
                if handler is None:
                    self.dialog(raw, kind)  # MD's own dialogs (yesno, option, list, radio, phase_menu, ...)
                else:
                    getattr(self, handler)(raw, inp)
            except Exception as exc:  # Stuck, or anything unexpected inside a step: recover, do not die
                why = str(exc) if isinstance(exc, Stuck) else f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, (Stuck, MemoryReadError)):
                    traceback.print_exc()
                if not self.recover(why, raw, inp, dialog, state):
                    return 3
        return 0


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
