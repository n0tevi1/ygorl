"""Autonomous Solo player for Master Duel (design 10, Solo self-play). Off unless explicitly run.

One loop: wait until the game asks the client for something, work out what is being asked from memory (the
engine's input state) and from a few fixed screen signatures (MD's own dialogs), decide, click, and check the
result in memory. Main Phase decisions come from the policy through the advisor (:func:`advise`); Battle Phase and
prompts use small rules. Anything it does not recognise is a *stuck* record: a full-resolution capture, the memory
state and the context, written to ``--out`` before the player stops. Those records are how recognisers get added
until the player needs no help (the goal: a scripted-only player).

Every click goes through :class:`ygorl.advisor.mdinput.GatedMouse`, which re-checks the Solo gate (verified DLL,
human client, computer opponent) before each event.

    python -m ygorl.advisor.mdplay --env md-2026-09 --policy CHECKPOINT [--out out/mdplay] [--max-steps 400]
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

from ygorl.advisor import mdlayout as L
from ygorl.advisor.mdinput import GatedMouse
from ygorl.advisor.mdmem import PHASES, RawDuel, load_md_cid, own_deck, solo_gate, to_snapshot
from ygorl.advisor.mdscreen import capture as capture_client

# --- input state (S + ...): written by duel.dll's command layer, read by DLL_DuelComGetCommandMask --------------
INPUT_MODE, INPUT_ACCEPT, INPUT_PLAYER, FOCUS_CID = 0x3CD0, 0x3CDC, 0x3D00, 0x3CF0
MAIN, BATTLE, OPEN_PROMPT, CHAIN_PROMPT, SUMMON_PICK, FIELD_PICK = 2, 3, 4, 5, 6, 8  # observed modes


def capture(window) -> np.ndarray:
    """A capture in the reference frame (see :mod:`ygorl.advisor.mdlayout`), whatever the window size."""
    return L.reference_image(capture_client(window))


class Stuck(RuntimeError):
    """The player cannot continue on its own; the record says why."""


@dataclass
class Input:
    mode: int
    accepting: bool
    player: int


def box_mean(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return img[y0:y1, x0:x1].reshape(-1, 3).mean(0)


def screen_dialog(img: np.ndarray) -> str | None:
    """MD UI dialogs that are not engine input states (measured on the 2054x1184 window)."""
    pos_l, pos_r = box_mean(img, 925, 940, 955, 970), box_mean(img, 1100, 940, 1130, 970)
    if pos_l[0] < 30 and pos_l[1] < 60 and pos_r[0] < 30 and pos_r[1] < 70:
        return "position"
    band, mid = box_mean(img, 620, 520, 1430, 560), box_mean(img, 900, 560, 1150, 600)
    if band.mean() < 100 and mid.mean() < 120:
        return "yesno"
    if box_mean(img, 950, 1100, 1100, 1130).max() < 30:
        return "list"
    banner = box_mean(img, 900, 270, 1150, 300)
    if banner[0] < 60 and banner[2] < 110 and banner.mean() < 80:
        return "banner"
    return None


def yellow_highlight(img: np.ndarray, x: int, y: int, w: int = 130, h: int = 165) -> bool:
    """A card outlined in MD's yellow selection frame around (x, y)."""
    x0, x1, y0, y1 = x - w // 2, x + w // 2, y - h // 2, y + h // 2
    ring = np.concatenate([img[y0 : y0 + 8, x0:x1].reshape(-1, 3), img[y1 - 8 : y1, x0:x1].reshape(-1, 3),
                           img[y0:y1, x0 : x0 + 8].reshape(-1, 3), img[y0:y1, x1 - 8 : x1].reshape(-1, 3)])  # fmt: skip
    yellow = (ring[:, 0] > 190) & (ring[:, 1] > 170) & (ring[:, 2] < 110)
    return yellow.mean() > 0.25


# screen centres of field cards (window pixels); the opponent's row is mirrored
MY_MZONE_Y, MY_EMZ_Y, OPP_MZONE_Y = 750, 540, 380
EMZ_X = (844, 1232)


def my_monster_xy(slot: int) -> tuple[int, int]:
    return (L.ZONE_X[slot], MY_MZONE_Y) if slot < 5 else (EMZ_X[slot - 5], MY_EMZ_Y)


def opp_monster_xy(slot: int) -> tuple[int, int]:
    return (L.ZONE_X[4 - slot], OPP_MZONE_Y) if slot < 5 else (EMZ_X[6 - slot], MY_EMZ_Y)


@dataclass
class Player:
    env: object
    policy: object
    out: Path
    samples: int = 4
    mouse: GatedMouse = field(init=False)
    log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        from ygorl.advisor.reconstruct import OpponentPrior
        from ygorl.engine.duel import default_cards

        self.mouse = GatedMouse()
        self.game = self.mouse.game
        self.cards = default_cards()
        self.passwords = load_md_cid(self.env.artifact_path("md_cid.json"))
        self.prior = OpponentPrior.from_environment(self.env)
        self.out.mkdir(parents=True, exist_ok=True)
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
            dialog = screen_dialog(capture(self.mouse.window)) if not mine else None
            key = (inp.mode, mine, dialog, raw.turn, raw.phase)
            stable = stable + 1 if key == last and (mine or dialog) else 0
            last = key
            if stable >= 3:
                return raw, inp, dialog
            time.sleep(0.35)
        raise Stuck("nothing to do for a long time")

    # --- acting ---------------------------------------------------------------------------------------------
    def click(self, x: int, y: int, why: str, pause: float = 1.0) -> None:
        self.log.append({"t": round(time.time(), 2), "click": [x, y], "why": why})
        print(f"  click ({x}, {y}) {why}", flush=True)
        self.mouse.click(x, y)
        time.sleep(pause)

    def phase(self, which: str) -> None:
        self.click(*L.PHASE_BADGE, "phase menu", 1.2)
        self.click(*L.PHASE_MENU[which], which, 2.0)

    def hand_xy(self, raw: RawDuel, cid: int) -> tuple[int, int]:
        """Screen position of the hand card with MD id ``cid`` (MD sorts the hand on screen)."""
        hand = [c.cid for c in raw.players[raw.me].piles["hand"]]
        order = L.hand_order(hand, lambda c: self.passwords.get(c, 0), self.cards.get)
        return L.hand_card(order.index(hand.index(cid)), len(hand))

    def open_card(self, xy: tuple[int, int], why: str, *, hand: bool = False) -> list[tuple[int, int]]:
        """Click a card and return the command icons that click opened: a row above a hand card, an icon above a
        field card (compared with a capture from just before, so card art is never taken for an icon)."""
        x, y = xy
        before = capture(self.mouse.window)
        self.click(x, y, why, 1.2)
        after = capture(self.mouse.window)
        if hand:
            return L.new_icons(before, after, max(x - 380, 0), 790, min(x + 380, L.REF_W), 975)
        return L.new_icons(before, after, max(x - 130, 0), max(y - 260, 0), min(x + 130, L.REF_W), y - 60)

    def open_and_click_icon(self, xy: tuple[int, int], why_card: str, why: str, pause: float = 1.8) -> bool:
        icons = self.open_card(xy, why_card)
        if not icons:
            return False
        self.click(*min(icons, key=lambda p: abs(p[0] - xy[0])), why, pause)
        return True

    def free_zone(self, raw: RawDuel, location: str) -> tuple[int, int]:
        slots = raw.players[raw.me].slots
        rng = range(0, 5) if location == "mzone" else range(7, 12)
        for s in rng:
            if slots[s] is None:
                return L.my_zone(location, s - (0 if location == "mzone" else 7))
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
            return self.phase("battle")
        if best.kind == "end_phase":
            return self.phase("end")
        if best.card is None:
            raise Stuck(f"no click script for {best.kind}")
        cid = next((c for c, p in self.passwords.items() if p == best.card.code), None)
        where = best.card.loc
        from ygorl.engine import constants as C

        if where.location != C.LOCATION_HAND:
            raise Stuck(f"no click script for {best.kind} from location {where.location}")
        xy = self.hand_xy(raw, cid)
        icons = self.open_card(xy, f"hand {self.name(cid)}", hand=True)
        card = self.cards[best.card.code]
        options = kinds.get(best.card.code, set())
        if card.is_monster:
            order = [k for k in ("summon", "spsummon", "mset") if k in options]
        else:
            order = [k for k in ("activate", "sset") if k in options]
        if best.kind not in order:
            raise Stuck(f"{best.kind} not among the buttons {order}")
        if len(icons) == len(order):
            target = icons[order.index(best.kind)]
        elif len(icons) == 1 and best.kind in ("mset", "sset") and order[-1] == best.kind:
            target = icons[0]  # MD offers only Set here although the reconstruction also allowed more
        else:
            raise Stuck(f"{len(icons)} command icons for {self.name(cid)}, expected {order}")
        self.click(*target, best.kind, 1.5)
        location = "mzone" if card.is_monster else "szone"
        self.click(*self.free_zone(raw, location), f"zone for {best.kind}", 2.5)
        before = sum(c.cid == cid for c in raw.players[raw.me].piles["hand"])
        for _ in range(10):
            after = self.game.read()
            if sum(c.cid == cid for c in after.players[after.me].piles["hand"]) < before:
                return
            time.sleep(0.5)
        raise Stuck(f"{best.kind} {self.name(cid)}: the card is still in hand after the clicks")

    def decide(self, raw: RawDuel):
        """The policy's top action for the current Main or Battle Phase board (and every legal suggestion)."""
        from ygorl.advisor.advise import advise

        snap = to_snapshot(raw, self.passwords, frozenset(self.attacked))
        deck = own_deck(raw, self.passwords, self.cards)
        advice = advise(snap, deck, self.policy, self.prior, self.env.rule_flags, cards=self.cards,
                        samples=self.samples, seed=random.randrange(1 << 30), starting_lp=self.env.player.starting_lp)  # fmt: skip
        if not advice.suggestions:
            raise Stuck(f"no legal action in the {snap.phase} reconstruction")
        top = advice.suggestions[0]
        name = self.cards[top.action.card.code].name if top.action.card else ""
        print(f"policy ({snap.phase}): {top.action.kind} {name} p={top.prob:.2f}", flush=True)
        self.log.append({"decide": top.action.kind, "card": top.action.card.code if top.action.card else 0,
                         "p": top.prob, "phase": snap.phase})  # fmt: skip
        return top.action, advice.suggestions

    def battle(self, raw: RawDuel) -> None:
        """The policy chooses: attack with which monster, Main Phase 2 or End Phase. The target is a rule: the most
        valuable monster the attacker beats, else the weakest; direct when the opponent has no monster."""
        best, _ = self.decide(raw)
        self.context = best.kind
        if best.kind == "main2":
            return self.phase("main2")
        if best.kind == "end_phase":
            return self.phase("end")
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
        x, y = my_monster_xy(slot)
        if not self.open_and_click_icon((x, y), f"attacker {self.name(card.cid)}", "attack"):
            return  # MD offers no Attack command (Gate of Darkness, already attacked): the policy is asked again
        if targets and self.input().mode == FIELD_PICK:
            _, _, s = max(targets)
            if not self.open_and_click_icon(opp_monster_xy(s), f"target {s}", "confirm target", 2.0):
                raise Stuck("no Confirm icon above the attack target")
        self.context = f"attack lp {opp.lp}"

    def prompt(self, raw: RawDuel, inp: Input) -> None:
        """Chain / open-timing prompts: respond to the opponent's attacks and in our own End Phase, else pass."""
        phase = PHASES[raw.phase] if raw.phase < len(PHASES) else ""
        theirs = raw.turn_player != raw.me
        respond = (theirs and phase == "battle") or (not theirs and phase == "end")
        if respond:
            self.click(*L.PROMPT_CARD, "tick the offered card", 0.8)
            self.click(*L.PROMPT_ACTIVATE, "activate", 2.0)
            self.context = "respond"
        else:
            self.click(*L.PROMPT_CANCEL, "pass", 1.5)

    def dialog(self, raw: RawDuel, kind: str) -> None:
        if kind == "position":
            self.click(940, 956, "face-up attack", 1.5)
            self.click(*self.free_zone(raw, "mzone"), "zone for the summon", 2.5)
        elif kind == "yesno":
            self.click(866, 658, "no", 1.5)
        elif kind == "list":
            self.click(1026, 960, "first card of the list", 0.8)
            self.click(1026, 1116, "select", 1.0)
            if screen_dialog(capture(self.mouse.window)) == "list":
                self.click(1180, 1117, "select (two-button list)", 1.5)
            self.context = "list"
        else:
            raise Stuck(f"no script for the {kind} screen (context {self.context!r})")

    def field_pick(self, raw: RawDuel) -> None:
        """Mode 8 outside an attack: pick the highlighted cards one by one and confirm each."""
        img = capture(self.mouse.window)
        spots = [my_monster_xy(s) for s in range(7)] + [opp_monster_xy(s) for s in range(7)]
        lit = [xy for xy in spots if yellow_highlight(img, *xy)]
        if not lit:
            raise Stuck("a field selection with no highlighted card found")
        x, y = lit[0]
        if not self.open_and_click_icon((x, y), "highlighted card", "confirm"):
            raise Stuck("no Confirm icon above the selected card")

    # --- loop -----------------------------------------------------------------------------------------------
    def record_stuck(self, why: str) -> Path:
        d = self.out / f"stuck-{time.strftime('%Y%m%d-%H%M%S')}"
        d.mkdir(parents=True)
        np.save(d / "screen.npy", capture(self.mouse.window))
        raw, inp = self.game.read(), self.input()
        (d / "state.json").write_text(json.dumps({"why": why, "context": self.context, "input": inp.__dict__,
                                                  "turn": raw.turn + 1, "phase": raw.phase, "log": self.log[-30:]},
                                                 indent=1))  # fmt: skip
        return d

    def run(self, max_steps: int = 400) -> int:
        for step in range(max_steps):
            raw, inp, dialog = self.wait_turn()
            solo_gate(raw)
            if dialog == "over":
                print("duel over, LP", [p.lp for p in raw.players])
                return 0
            if raw.turn != self.turn_seen:
                self.turn_seen, self.attacked = raw.turn, set()
            phase = PHASES[raw.phase] if raw.phase < len(PHASES) else str(raw.phase)
            print(f"[{step}] turn {raw.turn + 1} {phase} mode {inp.mode} dialog {dialog} "
                  f"LP {raw.players[raw.me].lp}/{raw.players[1 - raw.me].lp}", flush=True)  # fmt: skip
            try:
                if dialog:
                    self.dialog(raw, dialog)
                elif inp.mode == MAIN:
                    self.main_phase(raw)
                elif inp.mode == BATTLE:
                    self.battle(raw)
                elif inp.mode in (OPEN_PROMPT, CHAIN_PROMPT):
                    self.prompt(raw, inp)
                elif inp.mode == FIELD_PICK:
                    self.field_pick(raw)
                elif inp.mode == 1:
                    time.sleep(1.0)
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
    args = ap.parse_args(argv)
    from ygorl.data.environment import load_environment
    from ygorl.train.checkpoint import load_actor

    player = Player(load_environment(args.env), load_actor(args.policy), args.out, args.samples)
    try:
        return player.run(args.max_steps)
    finally:
        (args.out / f"log-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(player.log, indent=0))


if __name__ == "__main__":
    raise SystemExit(main())
