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
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ygorl.md import layout as L
from ygorl.md.input import GatedMouse
from ygorl.md.memory import PHASES, RawDuel, load_md_cid, own_deck, solo_gate, to_snapshot
from ygorl.md.screen import capture as capture_client
from ygorl.md.ui import UI, Stuck, command_order, my_monster_xy, opp_monster_xy
from ygorl.selfplay.trace import Trace

# --- input state (S + ...): written by duel.dll's command layer, read by DLL_DuelComGetCommandMask --------------
INPUT_MODE, INPUT_ACCEPT, INPUT_PLAYER, FOCUS_CID = 0x3CD0, 0x3CDC, 0x3D00, 0x3CF0
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
    phase_seen: tuple = (-1, -1)
    forbidden: set = field(default_factory=set)  # ((turn, phase), action key) MD refused this phase

    def __post_init__(self) -> None:
        from ygorl.advisor.reconstruct import OpponentPrior
        from ygorl.engine.duel import default_cards

        self.mouse = GatedMouse()
        self.ui = UI(self.mouse, lambda: capture(self.mouse.window))
        self.game = self.mouse.game
        self.cards = default_cards()
        self.passwords = load_md_cid(self.env.artifact_path("md_cid.json"))
        self.prior = OpponentPrior.from_environment(self.env)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(self.out, screen_every=self.screen_every)
        self.state_base = struct.unpack("<Q", self.game.memory.read(self.game.module.base + self.game.offsets.state_ptr, 8))[0]  # fmt: skip
        self.attacked: set[int] = set()  # uids that attacked (or could not) in this Battle Phase
        self.turn_seen = -1
        self.context = ""  # what the last action was, for sub-prompts

    # --- reading --------------------------------------------------------------------------------------------
    def input(self) -> Input:
        b = self.game.memory.read(self.state_base + INPUT_MODE, 0x40)
        return Input(struct.unpack_from("<I", b, 0)[0], bool(struct.unpack_from("<I", b, INPUT_ACCEPT - INPUT_MODE)[0]),
                     struct.unpack_from("<I", b, INPUT_PLAYER - INPUT_MODE)[0])  # fmt: skip

    def focus_cid(self) -> int:
        return struct.unpack("<I", self.game.memory.read(self.state_base + FOCUS_CID, 4))[0] & 0xFFFF

    def name(self, cid: int) -> str:
        pw = self.passwords.get(cid)
        return self.cards[pw].name if pw in self.cards else str(cid)

    def wait_turn(self, timeout: float = 600) -> tuple[RawDuel, Input, str | None]:
        """Wait until the client must act: an engine input addressed to it, or one of MD's dialogs (stable ~1 s)."""
        end, stable, last = time.time() + timeout, 0, None
        while time.time() < end:
            raw, inp = self.game.read(), self.input()
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
        raise Stuck("nothing to do for a long time")

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
    def main_phase(self, raw: RawDuel) -> None:
        best, legal = self.decide(raw)
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
        from ygorl.engine import constants as C

        if where.location in (C.LOCATION_MZONE, C.LOCATION_SZONE) and best.kind == "activate":
            if where.location == C.LOCATION_MZONE:
                xy = my_monster_xy(where.sequence)
            elif where.sequence < 5:
                xy = L.my_zone("szone", where.sequence)
            else:
                raise Stuck("no position measured for the Field Zone yet")
            if not self.ui.card_command(xy, f"field {self.name(cid)}", "activate", 2.0):
                raise Stuck(f"no Activate icon above {self.name(cid)} on the field")
            return
        if where.location != C.LOCATION_HAND:
            raise Stuck(f"no click script for {best.kind} from location {where.location}")
        xy = self.hand_xy(raw, cid)
        icons = self.ui.open_hand_card(xy, f"hand {self.name(cid)}")
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
        self.ui.pick_zone(location, self.free_zone(raw, location), f"zone for {best.kind}")
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
        self.expect = Expectation(top.action, raw, snap.phase)
        return top.action, allowed

    def battle(self, raw: RawDuel) -> None:
        """The policy chooses: attack with which monster, Main Phase 2 or End Phase. The target is a rule: the most
        valuable monster the attacker beats, else the weakest; direct when the opponent has no monster."""
        best, _ = self.decide(raw)
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
        self.trace.rule("prompt-respond" if respond else "prompt-pass")
        self.ui.respond_prompt(respond)
        if respond:
            self.context = "respond"

    def dialog(self, raw: RawDuel, kind: str) -> None:
        if kind == "position":
            self.trace.rule("position-faceup-attack")
            self.ui.pick_position("faceup_attack")
            self.ui.pick_zone("mzone", self.free_zone(raw, "mzone"), "zone for the summon")
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

    def _run(self, max_steps: int) -> int:
        for step in range(max_steps):
            self.trace.end_step(self.ui.take_clicks())  # the previous step, with its clicks
            raw, inp, dialog = self.wait_turn()
            solo_gate(raw)
            self.trace.begin(step, raw, inp, dialog)
            if self.trace.wants_screen(step):
                self.trace.save_screen(self.ui.capture())
            if dialog == "over":
                print("duel over, LP", [p.lp for p in raw.players])
                return 0
            if (raw.turn, raw.phase) != self.phase_seen:
                # the engine accepts input before MD's turn / phase animations (the drawn card shown big) are over
                self.phase_seen = (raw.turn, raw.phase)
                time.sleep(2.5)
            if raw.turn != self.turn_seen:
                self.turn_seen, self.attacked = raw.turn, set()
            me, op = raw.players[raw.me], raw.players[1 - raw.me]
            state = (inp.mode, dialog, raw.phase, _board_key(me, op), len(self.forbidden))  # a refusal is progress
            self.recent = (self.recent + [state])[-3:]
            if len(self.recent) == 3 and len(set(self.recent)) == 1:
                print(f"STUCK: no progress -> {self.record_stuck('the same state three steps in a row')}", flush=True)
                return 3
            phase = PHASES[raw.phase] if raw.phase < len(PHASES) else str(raw.phase)
            print(f"[{step}] turn {raw.turn + 1} {phase} mode {inp.mode} dialog {dialog} "
                  f"LP {raw.players[raw.me].lp}/{raw.players[1 - raw.me].lp}", flush=True)  # fmt: skip
            try:
                if self.expect is not None and not dialog and inp.mode in (MAIN, BATTLE):
                    problem = check_expectation(self.expect, raw, self.passwords)
                    self.trace.check(self.expect.action, problem)
                    print(f"  check {self.expect.action.kind}: {'ok' if problem is None else problem}", flush=True)
                    self.expect = None
                    if problem is not None:
                        raise Stuck(f"the game does not show the policy's action: {problem}")
                mine = inp.accepting and inp.player == raw.me
                if mine and inp.mode in (OPEN_PROMPT, CHAIN_PROMPT) and dialog in (None, "list"):
                    self.prompt(raw, inp)  # an activate-or-pass prompt panel looks like a list
                elif dialog == "banner" and mine and inp.mode == PLACE:
                    # where to put a monster (after a Tribute Summon: message 327, the free zones highlighted)
                    self.trace.rule("zone-first-free")
                    self.ui.pick_zone("mzone", self.free_zone(raw, "mzone"), "zone for the summon")
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
                    self.trace.rule("draw-continue")
                    self.ui.continue_draw()
                else:
                    raise Stuck(f"unknown input mode {inp.mode}")
            except Stuck as exc:
                print(f"STUCK: {exc} -> {self.record_stuck(str(exc))}", flush=True)
                return 3
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="autonomous Solo player for Master Duel (Solo vs computer only)")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/mdplay"))
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--screen-every", type=int, default=0, help="save a capture into the trace every N steps")
    args = ap.parse_args(argv)
    from ygorl.data.environment import load_environment
    from ygorl.train.checkpoint import load_actor

    env = load_environment(args.env)
    player = Player(env, load_actor(args.policy), args.out, args.samples, args.screen_every)
    player.trace.start(env=args.env, policy=str(args.policy), samples=args.samples, max_steps=args.max_steps)
    result: dict = {"exit": None}
    try:
        result["exit"] = player.run(args.max_steps)
        return result["exit"]
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        player.trace.close(**result)
        print(f"trace: {player.trace.path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
