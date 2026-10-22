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
import json
import random
import struct
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
from ygorl.md.ui import PHASE_MENU_CANCEL, UI, Stuck, command_order, my_monster_xy, opp_monster_xy
from ygorl.md.panel import PanelBook, panel_kind
from ygorl.md.request import read_request
from ygorl.selfplay.observe import ShadowObserver
from ygorl.selfplay.trace import Trace

# --- input state (S + ...): written by duel.dll's command layer, read by DLL_DuelComGetCommandMask --------------
INPUT_MODE, INPUT_ACCEPT, INPUT_PLAYER, FOCUS_CID = 0x3CD0, 0x3CDC, 0x3D00, 0x3CF0
OPTION_DIALOG = 6  # the dialog type of MD's "which effect?" choice (DLL_DuelDlg*: tabs and an OK button)
OPTION_OK = (1024, 788)


def option_tab(index: int, count: int) -> tuple[int, int]:
    """The tab of choice ``index`` among ``count`` in the effect-choice dialog (tabs span x 520-1540 at y 418)."""
    return round(520 + (index + 0.5) * 1020 / count), 418


NEUTRAL_SPOT = (1300, 640)  # empty board between the rows: clicking it closes a card left open
RESULT_RETRY = (743, 1093)  # MD's result screen: Retry (left) and OK (right)
MAIN, BATTLE, OPEN_PROMPT, CHAIN_PROMPT, SUMMON_PICK, PLACE, FIELD_PICK = 2, 3, 4, 5, 6, 7, 8  # observed modes


def capture(window) -> np.ndarray:
    """A capture in the reference frame (see :mod:`ygorl.md.layout`), whatever the window size."""
    return L.reference_image(capture_client(window))


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
        return None
    return None  # reposition and others: not checked yet


def _action_key(action) -> tuple:
    card = action.card
    return (action.kind, card.code if card else 0, card.loc.location if card else 0, card.loc.sequence if card else 0)


def _own_cards(raw: RawDuel) -> tuple:
    """Our field (with chain marks) and hand: what an activation of ours would change."""
    me = raw.players[raw.me]
    return (tuple((c.cid, c.face, c.sideways, c.flags) if c else None for c in me.slots),
            tuple(c.cid for c in me.piles["hand"]))  # fmt: skip


def _board_key(me, op) -> tuple:
    return (tuple((c.cid, c.face, c.sideways) if c else None for c in me.slots),
            tuple(c.cid for c in me.piles["hand"]), tuple(c.cid for c in me.piles["grave"]),
            tuple((c.cid, c.face) if c else None for c in op.slots), op.lp, me.lp,
            len(me.piles["removed"]), len(op.piles["grave"]))  # fmt: skip


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
        self.cards = default_cards()
        self.passwords = load_md_cid(self.env.artifact_path("md_cid.json"))
        self.prior = OpponentPrior.from_environment(self.env)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(self.out, screen_every=self.screen_every)
        self.observer = ShadowObserver(self.env.rule_flags, self.cards, self.prior)  # observe only (design 10)
        self.panel = PanelBook(self.out / "panel_book.npz")  # learned across runs
        self.attacked: set[int] = set()  # uids that attacked (or could not) in this Battle Phase
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

    def wait_turn(self, timeout: float = 240) -> tuple[RawDuel, Input, str | None]:
        """Wait until the client must act: an engine input addressed to it, or one of MD's dialogs (stable ~1 s)."""
        end, stable, last = time.time() + timeout, 0, None
        while time.time() < end:
            try:
                raw, inp = self.game.read(), self.input()
            except MemoryReadError:  # between duels (loading) the state can be gone for a moment
                time.sleep(1.0)
                continue
            if raw.finished:  # the engine's own flag: LP reads can catch the obfuscated value mid-update
                return raw, inp, "over"
            mine = inp.accepting and inp.player == raw.me
            dialog = self.ui.dialog()  # MD's own dialog wins over the engine mode
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
        from collections import Counter

        gone = Counter(c.cid for c in before.players[before.me].piles["hand"]) - Counter(
            c.cid for c in raw.players[raw.me].piles["hand"])  # fmt: skip
        if sum(gone.values()) == 1:
            (cid,) = gone
            self.panel.learn(img, cid, self.card_stats(cid))
            self.panel.save()
            self.trace.observe(panel_learned=cid)

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
        """Check on MD's information panel that the opened card is ``cid`` (the hand order on screen is inferred).
        A different card: try the other hand positions until the panel shows ``cid``. Unknown (nothing learned for
        these cards yet): keep the click, and learn from it once memory confirms the action."""
        hand = [c.cid for c in raw.players[raw.me].piles["hand"]]
        stats = {c: self.card_stats(c) for c in hand}
        img = self.ui.capture()
        got, how = self.panel.identify(img, hand, stats)
        kind, want_kind = panel_kind(img), self.card_kind(cid)
        self.trace.observe(panel={"want": cid, "got": got, "how": how, "kind": kind, "want_kind": want_kind})
        if got == cid or (got is None and (not kind or kind == want_kind)):
            self.pending_panel = (img, raw)
            return icons
        if got is None:
            how = f"a {kind} panel"
        shown = self.name(got) if got is not None else how
        print(f"  panel shows {shown}, not {self.name(cid)}: searching the hand", flush=True)
        for i in range(len(hand)):
            pos = L.hand_card(i, len(hand))
            if pos == xy:
                continue
            found = self.ui.open_hand_card(pos, f"hand #{i + 1}: looking for {self.name(cid)}")
            img = self.ui.capture()
            seen, how = self.panel.identify(img, hand, stats)
            kind = panel_kind(img)
            if seen == cid or (seen is None and kind == want_kind and self._only_one_of_kind(hand, want_kind)):
                self.trace.observe(panel={"want": cid, "found_at": i, "how": how or kind})
                self.pending_panel = (img, raw)
                return found
        raise Stuck(f"{self.name(cid)} not found in the hand by its information panel")

    def hand_xy(self, raw: RawDuel, cid: int) -> tuple[int, int]:
        """Screen position of the hand card with MD id ``cid`` (MD sorts the hand on screen)."""
        hand = [c.cid for c in raw.players[raw.me].piles["hand"]]
        order = L.hand_order(hand, lambda c: self.passwords.get(c, 0), self.cards.get)
        return L.hand_card(order.index(hand.index(cid)), len(hand))

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
            if not self.ui.card_command(xy, f"field {self.name(cid)}", "activate", 2.0):
                return self.refused(raw, best, f"no Activate for {self.name(cid)} on the field")
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
        icons = self.verify_hand_card(raw, cid, xy, icons)
        self.trace.observe(icons=icons)
        card = self.cards[best.card.code]
        order = command_order(card.is_monster, kinds.get(best.card.code, set()))
        if self.ui.choose_command(icons, best.kind, order, self.name(cid)) is None:
            return self.refused(raw, best, f"{self.name(cid)} shows no command")
        if card.is_monster and best.kind == "activate":
            return  # a hand effect: no zone, the card may stay in hand; its prompts are handled by the loop
        if best.kind in ("summon", "mset") and self.ui.dialog() == "yesno":
            # a Tribute Summon / Set asks to confirm first (message 335); the tributes are a field selection next
            self.ui.yesno(True, "yes: tribute")
            self.pick_tributes(raw)
            return
        location = "mzone" if card.is_monster else "szone"
        zone = self.free_zone(raw, location)
        self.ui.pick_zone(location, zone, f"zone for {best.kind}")
        self.last_zone = (location, zone)
        self._placed(location, zone)
        before = sum(c.cid == cid for c in raw.players[raw.me].piles["hand"])
        for _ in range(10):
            after = self.game.read()
            if sum(c.cid == cid for c in after.players[after.me].piles["hand"]) < before:
                return
            time.sleep(0.5)
        raise Stuck(f"{best.kind} {self.name(cid)}: the card is still in hand after the clicks")

    def refused(self, raw: RawDuel, action, why: str) -> None:
        """MD shows no command for an action the reconstruction allows: a state the snapshot does not carry
        (a turn-long restriction, a once-per-turn already used). Masked for this phase; the policy decides again."""
        self.forbidden.add(((raw.turn, raw.phase), _action_key(action)))
        self.trace.refusal(action, f"MD refused: {why}")
        print(f"  check {action.kind}: MD refused ({why}); masked for this phase", flush=True)
        self.expect = None

    def decide(self, raw: RawDuel):
        """The policy's top action for the current Main or Battle Phase board (and every legal suggestion)."""
        from ygorl.advisor.advise import advise

        snap = to_snapshot(raw, self.passwords, frozenset(self.attacked))
        deck = own_deck(raw, self.passwords, self.cards)
        advice = advise(snap, deck, self.policy, self.prior, self.env.rule_flags, cards=self.cards,
                        samples=self.samples, seed=random.randrange(1 << 30), starting_lp=self.env.player.starting_lp)  # fmt: skip
        key = (raw.turn, raw.phase)
        allowed = [g for g in advice.suggestions if (key, _action_key(g.action)) not in self.forbidden]
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
        """The policy chooses: attack with which monster, Main Phase 2 or End Phase. The target is a rule: the most
        valuable monster the attacker beats, else the weakest; direct when the opponent has no monster."""
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
            _, _, s = max(targets)
            self.trace.observe(target_slot=s)
            if not self.ui.pick_target(opp_monster_xy(s), f"target {s}"):
                raise Stuck("no Confirm icon above the attack target")
        self.context = f"attack lp {opp.lp}"

    def prompt(self, raw: RawDuel, inp: Input) -> None:
        """Chain / open-timing prompts: respond to the opponent's attacks and in our own End Phase, else pass."""
        phase = PHASES[raw.phase] if raw.phase < len(PHASES) else ""
        theirs = raw.turn_player != raw.me
        respond = (theirs and phase == "battle") or (not theirs and phase == "end")
        # the same prompt again with nothing of ours changed: the activation did not happen (MD can offer a card it
        # then greys the Activate button for), so pass instead of ticking it forever
        seen = (raw.turn, raw.phase, inp.mode, _own_cards(raw))
        if respond and seen == self.responded:
            respond = False
            self.trace.observe(prompt="the response did not take: passing")
        self.responded = seen if respond else None
        self.trace.rule("prompt-respond" if respond else "prompt-pass")
        self.ui.respond_prompt(respond)
        if respond:
            self.context = "respond"

    def dialog(self, raw: RawDuel, kind: str) -> None:
        if kind == "position":
            self.trace.rule("position-faceup-attack")
            self.ui.pick_position("faceup_attack")
            zone = self.free_zone(raw, "mzone")
            self.ui.pick_zone("mzone", zone, "zone for the summon")
            self._placed("mzone", zone)
        elif kind == "option":
            req = read_request(self.game.memory, self.game.module.base, self.game.offsets, raw)
            n = len(req.choices)
            pick = next((i for i, c in enumerate(req.choices) if c >> 31), 0)  # the first enabled effect
            self.trace.rule("option-first-enabled", choices=[hex(c) for c in req.choices], pick=pick)
            self.ui.click(*option_tab(pick, n), f"effect {pick + 1} of {n}", 0.8)
            self.ui.click(*OPTION_OK, "ok", 1.5)
        elif kind == "phase_menu":
            self.trace.rule("phase-menu-cancel")
            self.ui.click(*PHASE_MENU_CANCEL, "close the phase menu", 1.2)
        elif kind == "yesno":
            self.trace.rule("yesno-no")
            self.ui.yesno(False)
        elif kind == "list":
            self.trace.rule("list-first")
            self.trace.observe(list_cards=self.ui.pick_list_cards([0]))
            self.context = "list"
        else:
            raise Stuck(f"no script for the {kind} screen (context {self.context!r})")

    def pick_tributes(self, raw: RawDuel) -> None:
        """Tributes: highlighted monsters of the advisor's, lowest ATK first, each confirmed, until none is left
        highlighted. (A rule for now; the shadow duel will let the policy decide.)"""
        me = raw.players[raw.me]
        self.trace.rule("tribute-lowest-atk")
        for _ in range(3):
            lit = set(self.ui.highlighted())
            mine = [(s, c) for s, c in enumerate(me.slots[:7]) if c and my_monster_xy(s) in lit]
            if not mine:
                return

            def atk(c):
                k = self.cards.get(self.passwords.get(c.cid))
                return k.attack if k is not None else 0

            s, c = min(mine, key=lambda t: atk(t[1]))
            if not self.ui.card_command(my_monster_xy(s), f"tribute {self.name(c.cid)}", "confirm tribute", 2.0):
                raise Stuck("no Confirm icon above the tribute")

    def field_pick(self, raw: RawDuel) -> None:
        """Mode 8 outside an attack: pick the highlighted cards one by one and confirm each."""
        lit = self.ui.highlighted()
        self.trace.rule("field-pick-first")
        self.trace.observe(highlighted=lit)
        if not lit:
            raise Stuck("a field selection with no highlighted card found")
        if not self.ui.card_command(lit[0], "highlighted card", "confirm"):
            raise Stuck("no Confirm icon above the selected card")

    # --- loop -----------------------------------------------------------------------------------------------
    def record_stuck(self, why: str) -> Path:
        d = self.out / f"stuck-{time.strftime('%Y%m%d-%H%M%S')}"
        d.mkdir(parents=True)
        np.save(d / "screen.npy", self.ui.capture())
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
        for g in range(games):
            print(f"=== game {g + 1}/{games}", flush=True)
            code = self.run(max_steps)
            raw = self.game.read()
            result = struct.unpack("<I", self.game.memory.read(self._key() + self.game.offsets.duel_result, 4))[0]
            summary = {"game": g + 1, "exit": code, "finished": raw.finished, "result": result,
                       "lp": [raw.players[raw.me].lp, raw.players[1 - raw.me].lp], "turn": raw.turn + 1}  # fmt: skip
            self.trace.game(**summary)
            print(f"=== game {g + 1} summary {summary}", flush=True)
            if code != 0:
                return code
            if g + 1 < games:
                self.restart_duel()
        return 0

    def _key(self) -> int:
        return struct.unpack("<Q", self.game.memory.read(self.game.module.base + self.game.offsets.key_ptr, 8))[0]

    def restart_duel(self, timeout: float = 150) -> None:
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
                self.reset_game_state()
                time.sleep(3.0)
                return
            if raw is not None and raw.finished and time.time() - last_click > 8:
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
        key = state[:4] if isinstance(state, tuple) and len(state) > 4 else state  # not the refusal count
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
            elif n == 2 or n == 3:
                alt = n - 1  # the 1st / 2nd alternative
                if mine and inp.mode in (OPEN_PROMPT, CHAIN_PROMPT):  # its panel looks like a list: pass first
                    self.ui.respond_prompt(False)
                elif dialog == "list":
                    cards = self.ui.pick_list_cards([alt])
                    self.trace.rule(f"recover-list-#{alt + 1}", cards=len(cards))
                elif dialog == "yesno":
                    self.ui.yesno(alt == 1, "recover: the other answer")
                elif dialog == "banner" or (mine and inp.mode == FIELD_PICK):
                    lit = self.ui.highlighted()
                    if len(lit) > alt:
                        self.ui.card_command(lit[alt], "recover: another highlighted card", "confirm")
                elif mine and inp.mode in (MAIN, BATTLE):
                    self.ui.phase("end")
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
        self.recoveries, self.recovery_count = {}, 0
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
            if dialog == "yesno" and req.dialog_type == OPTION_DIALOG and len(req.choices) >= 2:
                dialog = "option"  # an effect choice: looks like yes / no on screen, told apart in memory
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
            if dialog == "over":
                print("duel over, LP", [p.lp for p in raw.players])
                return 0
            if (raw.turn, raw.phase) != self.phase_seen:
                # the engine accepts input before MD's turn / phase animations (the drawn card shown big) are over
                self.phase_seen = (raw.turn, raw.phase)
                time.sleep(1.0)
                self.ui.wait_still()  # e.g. the drawn card flying to the hand: hand positions count it already
            if raw.turn != self.turn_seen:
                self.turn_seen, self.attacked = raw.turn, set()
            me, op = raw.players[raw.me], raw.players[1 - raw.me]
            state = (inp.mode, dialog, raw.phase, _board_key(me, op), len(self.forbidden))  # a refusal is progress
            self.recent = (self.recent + [state])[-3:]
            if len(self.recent) == 3 and len(set(self.recent)) == 1:
                if not self.recover("the same state three steps in a row", raw, inp, dialog, state):
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
                    print(f"  check {self.expect.action.kind}: {'ok' if problem is None else problem}", flush=True)
                    self.expect = None
                    if problem is not None:
                        raise Stuck(f"the game does not show the policy's action: {problem}")
                mine = inp.accepting and inp.player == raw.me
                if dialog == "phase_menu" and mine and inp.mode in (MAIN, BATTLE):
                    dialog = None  # a phase change under way: the decision below uses (or closes) the open menu
                if mine and inp.mode in (OPEN_PROMPT, CHAIN_PROMPT) and dialog in (None, "list"):
                    self.prompt(raw, inp)  # an activate-or-pass prompt panel looks like a list
                elif dialog == "banner" and mine and inp.mode == PLACE:
                    # where to put a monster (after a Tribute Summon: message 327, the free zones highlighted)
                    self.trace.rule("zone-first-free")
                    zone = self.free_zone(raw, "mzone")
                    self.ui.pick_zone("mzone", zone, "zone for the summon")
                    self._placed("mzone", zone)
                elif dialog == "banner" and mine and inp.mode == MAIN and self.last_zone is not None:
                    # MD still asks for the zone of our own Summon / Set (the click came while the banner animated in)
                    self.trace.rule("zone-again")
                    self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
                    self.ui.pick_zone(*self.last_zone, "zone again")
                    self.last_zone = None
                elif dialog == "banner" and mine and inp.mode == FIELD_PICK:
                    self.field_pick(raw)  # the banner states what to pick on the field (tributes, targets)
                elif dialog:
                    self.dialog(raw, dialog)
                elif inp.mode == MAIN:
                    self.main_phase(raw)
                elif inp.mode == BATTLE:
                    self.battle(raw)
                elif inp.mode == FIELD_PICK:
                    self.field_pick(raw)
                elif inp.mode == 1:
                    self.draw_attempts = self.draw_attempts + 1 if self.phase_seen == self.draw_seen else 0
                    self.draw_seen = self.phase_seen
                    self.trace.rule("draw-continue", attempt=self.draw_attempts)
                    self.ui.continue_draw(self.draw_attempts)
                else:
                    raise Stuck(f"unknown input mode {inp.mode}")
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
