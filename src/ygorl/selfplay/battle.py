"""Battle Phase and field picks of the Solo player (:class:`ygorl.selfplay.loop.Player`): attacks and their
targets, the battle checks (a target stronger than shown), Tributes and field selections."""

from __future__ import annotations

import random
import time
from dataclasses import replace


from ygorl.md import layout as L
from ygorl.md.memory import PHASES, MemoryReadError, RawDuel, own_deck, to_snapshot
from ygorl.md.client import CommandBit
from ygorl.md.request import Candidate
from ygorl.md.ui import (
    Stuck,
    my_monster_xy,
    my_spell_xy,
    opp_monster_xy,
    opp_spell_xy,
)
from ygorl.selfplay.state import (
    NEUTRAL_SPOT,
    FIELD_PICK,
    Expectation,
)


class BattleMixin:
    """Attacks, attack targets, Tributes and field picks."""

    def note_battle(self, exp: Expectation, now: RawDuel) -> None:
        """An attack that cost us LP while its target stayed on the field: the target is stronger than the
        reconstruction thinks (an equip it does not restore: Time Wizard with Red-Eyes Black Dragon Sword was
        attacked three times by a 1100 monster). No more attacks into it this turn."""
        target = getattr(self, "attack_target_uid", 0)
        self.attack_target_uid = 0
        if exp.action.kind != "attack" or not target:
            return
        me0, me1 = exp.before.players[exp.before.me], now.players[now.me]
        stays = any(c is not None and c.uid == target for c in now.players[1 - now.me].slots[:7])
        if me1.lp < me0.lp and stays and getattr(self, "attack_expected_win", True):
            # only a battle the shown values said we win (a loss the values foresaw tells nothing new: flagging
            # it redirected the next attack into a stronger monster and lost the duel)
            self.strong_targets = getattr(self, "strong_targets", set()) | {(now.turn, target)}
            self.trace.observe(strong_target=target)
            print("  battle: the target survived and we took damage; no more attacks into it this turn", flush=True)

    def field_stats(self, raw: RawDuel) -> dict[int, tuple[int, int]]:
        """uid -> (ATK, DEF) MD shows for the face-up monsters (memory: the client's card views, equips and other
        continuous effects included; :meth:`ygorl.md.client.Client.field_stats`); {} when not readable."""
        client = getattr(self, "client", None)
        if client is None:
            return {}
        try:
            return client.field_stats(raw)
        except (MemoryReadError, OSError):
            return {}

    def still_attacks(self, raw: RawDuel) -> bool:
        """Whether the policy's top Battle Phase action on ``raw`` is an attack by the attacker of the replay."""
        from ygorl.advisor.advise import advise

        uid, code = self.attacker
        snap = to_snapshot(raw, self.passwords, frozenset(self.attacked - {uid}), frozenset(self.activated),
                           self.field_stats(raw))  # fmt: skip
        try:
            advice = advise(snap, own_deck(raw, self.passwords, self.cards), self.policy, self.prior,
                            self.env.rule_flags, cards=self.cards, samples=self.samples,
                            seed=random.randrange(1 << 30), starting_lp=self.env.player.starting_lp)  # fmt: skip
        except Exception as exc:  # noqa: BLE001
            self.trace.observe(replay=f"no decision: {type(exc).__name__}: {exc}")
            return False
        top = advice.suggestions[0].action if advice.suggestions else None
        self.trace.observe(replay={"top": [top.kind, top.card.code if top and top.card else 0] if top else None})
        return top is not None and top.kind == "attack" and top.card is not None and top.card.code == code

    def battle(self, raw: RawDuel) -> None:
        """The policy chooses: attack with which monster, Main Phase 2 or End Phase, and the attack target (rebuilt
        from the current board, :meth:`ygorl.selfplay.decide.Decider.attack_target`). The rule when the rebuild
        cannot tell: the most valuable monster the attacker beats, else the weakest; direct when the opponent has no
        monster."""
        best, legal = self.decide(raw)
        self.close_phase_menu_unless(best.kind)
        self.context = best.kind
        if best.kind == "main2":
            return self.ui.phase("main2")
        if best.kind == "end_phase":
            return self.ui.phase("end")
        if best.kind == "activate" and best.card is not None:
            # an effect activated in the Battle Phase (Gaia the Fierce Knight Origin, run s76): the same commands as
            # in a Main Phase
            return self.main_phase(raw, decided=(best, legal))
        if best.kind != "attack" or best.card is None:
            # nothing clicked: masked for this phase, the policy decides again (not a generic recovery)
            return self.refused(raw, best, f"no click script for {best.kind} in the Battle Phase")
        me, opp = raw.players[raw.me], raw.players[1 - raw.me]
        slot = best.card.loc.sequence
        card = me.slots[slot]
        if card is None:
            raise Stuck(f"the policy attacks with an empty zone {slot}")
        self.attacked.add(card.uid)
        pw = self.passwords.get(card.cid)
        shown = self.field_stats(raw)  # memory: what MD shows (equips included); the database values else
        power = shown[card.uid][0] if card.uid in shown else (self.cards[pw].attack if pw in self.cards else 0)
        targets = []
        strong = getattr(self, "strong_targets", set())
        self._values = {}
        for s, c in enumerate(opp.slots[:7]):
            if c is None or (raw.turn, c.uid) in strong:
                continue
            k = self.cards.get(self.passwords.get(c.cid)) if c.face else None
            value = (k.defense if c.sideways else k.attack) if k is not None else 1500  # face-down: average guess
            if c.face and c.uid in shown:
                value = shown[c.uid][1] if c.sideways else shown[c.uid][0]
            targets.append((value < power, value if value < power else -value, s))
            values = getattr(self, "_values", {})
            values[s] = value
            self._values = values
        if not targets and any(c is not None for c in opp.slots[:7]):
            # every monster it could attack beat an attack of ours this turn: not again
            return self.refused(raw, best, "every target survived an attack of ours this turn")
        if self.command_by_mask(raw, best, card.cid, my_monster_xy(slot), "attack"):
            pass
        elif getattr(self, "mask_refused", False):
            self.attacked.discard(card.uid)
            return None
        elif not self.ui.attack(my_monster_xy(slot), f"attacker {self.name(card.cid)}"):
            # the previous battle can still be animating (a second Ra's Disciple was refused while it glowed ready
            # to attack): let the board settle and try once more
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
            self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
            if not self.ui.attack(my_monster_xy(slot), f"attacker {self.name(card.cid)} (again)"):
                return self.refused(raw, best, f"no Attack for {self.name(card.cid)}")
        if targets and self.input().mode == FIELD_PICK:
            s = self.rebuilt("target", raw, slot, frozenset(self.attacked - {card.uid}))
            if s not in {t[2] for t in targets}:
                _, _, s = max(targets)
                self.trace.sub("target", by="rule", rule="target-most-valuable-beaten", slot=s)
            self.trace.observe(target_slot=s)
            self.attack_slot = s
            self.attack_target_uid = opp.slots[s].uid if opp.slots[s] is not None else 0
            self.attack_expected_win = getattr(self, "_values", {}).get(s, 0) < power
            if not self.ui.pick_target(opp_monster_xy(s), f"target {s}"):
                raise Stuck("no Confirm icon above the attack target")
        self.context = f"attack lp {opp.lp}"
        self.attacker = (card.uid, best.card.code)
        self.attack_board = (raw.turn, tuple(c is not None for c in opp.slots[:7]))  # (a replay: these change)

    def replay_pick(self, raw: RawDuel) -> bool:
        """The new attack target after a replay, rebuilt like a declared attack's (run s89: a rule took the
        opponent's first monster, a 2800 one, and the attacker was destroyed). False when the rebuild cannot tell."""
        me, opp = raw.players[raw.me], raw.players[1 - raw.me]
        slot = next((i for i, c in enumerate(me.slots[:7]) if c is not None and c.uid == self.attacker[0]), None)
        if slot is None:
            return False
        s = self.rebuilt("target", raw, slot, frozenset(self.attacked - {self.attacker[0]}))
        if s is None or not 0 <= s < 7 or opp.slots[s] is None:
            return False
        self.trace.observe(target_slot=s, replay=True)
        return self.ui.pick_target(opp_monster_xy(s), f"target {s} (replay)")

    def replay_answer(self, raw: RawDuel) -> bool:
        """Yes to MD's replay question? The policy must still attack with the attacker on the new board; and when the
        opponent's monster zones did not change (the declared target became one MD forbids: Corrosive Scales, run
        s91, where the rebuild still saw it attackable), another target must have been wanted at the declaration
        (its rebuilt probability at least 0.5), else the attack ends here."""
        if not self.still_attacks(raw):
            return False
        if self.is_replay(raw):
            return True
        opp = raw.players[1 - raw.me]
        declared = getattr(self, "attack_slot", None)
        others = [float(p) for s, p in getattr(self, "target_probs", {}).items()
                  if str(s).isdigit() and int(s) != declared and 0 <= int(s) < 7 and opp.slots[int(s)] is not None]  # fmt: skip
        return bool(others) and max(others) >= 0.5

    def is_replay(self, raw: RawDuel) -> bool:
        """Whether a yes / no now is MD's replay question ("continue to attack?"): in the Battle Phase of the attack,
        the attacker still on our field, and the opponent's monster zones changed since it was declared."""
        attacker, board = getattr(self, "attacker", None), getattr(self, "attack_board", None)
        if attacker is None or board is None or board[0] != raw.turn or raw.phase != PHASES.index("battle"):
            return False
        me, opp = raw.players[raw.me], raw.players[1 - raw.me]
        if not any(c is not None and c.uid == attacker[0] for c in me.slots[:7]):
            return False
        return tuple(c is not None for c in opp.slots[:7]) != board[1]

    def field_target(self, raw: RawDuel, lit) -> tuple[int, int] | None:
        """The field card the policy picks through the shadow for MD's field selection (tributes, targets): its spot,
        which the caller confirms by its Confirm icon (MD shows one only on a card it allows); None when the shadow
        does not match (the caller's rule decides)."""
        req = None
        try:
            req = self.request_now(raw)
        except Exception:  # noqa: BLE001 - no request: the rule decides
            return None
        spots = self.lit_candidates(raw, lit)
        framed = {(c.player, c.sequence) for c, _ in spots}
        on_field = [c for c in req.candidates if c.location == "mzone" and (c.player, c.sequence) in framed]
        if not req.candidates:
            # no list in memory (a target "on the field"): every card on the field is a candidate, both players'
            # Monster, Spell & Trap and Field Zones (run s93: only the three framed monsters were offered, the
            # shadow's six targets shrank to one and Spiral Discharge destroyed our own monster)
            req = replace(req, candidates=tuple(c for c, _ in self.field_candidates(raw)))
            self.trace.observe(field_candidates=[[c.cid, c.player, c.location, c.sequence] for c in req.candidates])
        elif spots and len(on_field) < len(spots):
            # memory's request does not name every framed card (a Tribute's check has no list, or it is stale and
            # named only one of three Tributes): the cards MD frames, read from memory at their places, are the
            # candidates (a Tribute for Obelisk fell to a rule that never takes the opponent's monster)
            req = replace(req, candidates=tuple(c for c, _ in spots))
            self.trace.observe(field_candidates=[[c.cid, c.player, c.sequence] for c, _ in spots])
        pick = self.lockstep("list", raw, req)
        if pick is None or pick < 0:  # (a declined optional trigger is no field pick)
            return None
        c = req.candidates[pick]
        xy = self.field_spot(raw, c)
        if xy is None:
            self.observer.shadow = None
            self.trace.observe(lockstep_field=f"no measured spot for candidate {pick}")
            return None
        # frames are found a little off the slot's measured point ((1219, 740) for (1232, 750)): a frame near it is
        # clicked where it was seen; an unframed card (its frame hidden under an attack arrow) at its own spot
        near = [q for q in lit if abs(q[0] - xy[0]) + abs(q[1] - xy[1]) < 60]
        self.field_shadow_spot = near[0] if near else xy  # (no Confirm there: the shadow is dropped by the caller)
        return self.field_shadow_spot

    def field_spot(self, raw: RawDuel, c) -> tuple[int, int] | None:
        """The screen spot of a field candidate of either player (Monster, Spell & Trap or Field Zone)."""
        if c.sequence is None:
            return None
        mine = c.player == raw.me
        if c.location == "mzone":
            return my_monster_xy(c.sequence) if mine else opp_monster_xy(c.sequence)
        if c.location == "szone" and 0 <= c.sequence <= 5:
            return my_spell_xy(c.sequence) if mine else opp_spell_xy(c.sequence)
        return None

    def field_candidates(self, raw: RawDuel) -> list:
        """(candidate, spot) for every card on the field, both players: card ids of ours and face-up ones only."""
        out = []
        for player in (raw.me, 1 - raw.me):
            for i, c in enumerate(raw.players[player].slots):
                if c is None:
                    continue  # (Extra Monster Zones 5 / 6 included: a Link monster there was the only target, s95)
                loc, seq = ("mzone", i) if i < 7 else ("szone", i - 7)
                cid = c.cid if (player == raw.me or c.face) else 0
                cand = Candidate(cid, c.uid, player, loc, seq)
                xy = self.field_spot(raw, cand)
                if xy is not None:
                    out.append((cand, xy))
        return out

    def lit_candidates(self, raw: RawDuel, lit) -> list:
        """(candidate, spot) for every monster card MD frames: its player / zone / sequence and card from memory."""
        out = []
        for player, spot in ((raw.me, my_monster_xy), (1 - raw.me, opp_monster_xy)):
            slots = raw.players[player].slots
            for s in range(7):
                c = slots[s]
                xy = spot(s)
                if c is None or not any(abs(xy[0] - q[0]) + abs(xy[1] - q[1]) < 80 for q in lit):
                    continue
                cid = c.cid if (player == raw.me or c.face) else 0  # an opponent's face-down card: a place only
                out.append((Candidate(cid, c.uid, player, "mzone", s), xy))
        return out

    def pick_tributes(self, raw: RawDuel) -> None:
        """Tributes: the policy's picks through the shadow; without a shadow, highlighted monsters of the advisor's,
        lowest ATK first. Each is confirmed, until none is left highlighted."""
        me = raw.players[raw.me]
        taken: set = set()  # slots picked already: they stay framed, and a click would take the pick back
        unpickable: set = set()  # spots that showed no Confirm (not a Tribute MD allows): never tried again
        n = 0
        for _attempt in range(6):
            if n >= 3:
                return
            # the input mode flickers while a pick animates (a third Tribute for Ra was lost); the first pick waits
            # too, longer (a request read before MD is in field-pick mode is stale: the shadow was dropped)
            for _ in range(8 if n else 20):
                if self.input().mode == FIELD_PICK:
                    break
                time.sleep(0.25)
            else:
                if n:
                    return  # MD took the tributes: the summoned monster glowing in their zone is not another pick
                # MD never asked for a Tribute (its yes / no was "Normal Summon without Tributing?": Charging Gaia,
                # run s87, summoned without one while a monster was clicked as a Tribute): nothing to pick
                self.trace.observe(tributes="MD asked for none")
                return
            lit = set(self.ui.highlighted())
            # frames found off the measured spot (Apprentice Illusion Magician's at (644, 739) for (650, 750)) count
            # within 80 px; the monsters MD's request offers (memory) count too: an exact match missed one, and the
            # lowest-ATK rule tributed a 2500 monster instead of a 2000 one
            near = lambda xy: any(abs(xy[0] - p[0]) + abs(xy[1] - p[1]) < 80 for p in lit)  # noqa: E731
            # the request's monsters only when no frame is seen (it can be stale: it named our Ra's Disciple, which
            # Sphere Mode's Tributes exclude, and the duel was given up on its missing Confirm)
            offered = set(self.offered_field_cards(self.game.read())) if n == 0 and not lit else set()
            mine = [(s, c) for s, c in enumerate(me.slots[:7])
                    if c and (near(my_monster_xy(s)) or my_monster_xy(s) in offered) and s not in taken]  # fmt: skip
            # the opponent's framed monsters too (Soul Crossing lets one be tributed)
            theirs = [s for s in range(7) if raw.players[1 - raw.me].slots[s] is not None
                      and near(opp_monster_xy(s)) and ("op", s) not in taken]  # fmt: skip
            self.trace.observe(tribute_candidates=[s for s, _ in mine], tribute_theirs=theirs, tribute_lit=sorted(lit))
            # every pick's view, not only the last (an audit could not tell whether a hidden frame was offered)
            self.tribute_views = getattr(self, "tribute_views", [])[-8:] + [
                {"mine": [s for s, _ in mine], "theirs": theirs, "lit": sorted(lit)}
            ]
            self.trace.observe(tribute_views=self.tribute_views)
            if not mine and not theirs:
                return
            mine = [(s, c) for s, c in mine if my_monster_xy(s) not in unpickable]
            theirs = [s for s in theirs if opp_monster_xy(s) not in unpickable]
            if not mine and not theirs:
                return
            spots = {my_monster_xy(s) for s, _ in mine} | {opp_monster_xy(s) for s in theirs}
            xy = self.field_target(self.game.read(), spots)
            if xy is not None:
                if not self.ui.card_command(xy, "tribute (policy)", "confirm tribute", 2.0):
                    # MD does not take this one: the shadow is out of step; the next candidate, never this again
                    self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                    unpickable.add(self.nearest_spot(xy, spots))
                    if getattr(self, "observer", None) is not None:
                        self.observer.shadow = None
                    self.trace.observe(tribute_unpickable=list(xy))
                    continue
                n += 1
                taken |= {
                    s for s, _ in mine if abs(my_monster_xy(s)[0] - xy[0]) + abs(my_monster_xy(s)[1] - xy[1]) < 80
                }
                taken |= {("op", s) for s in theirs
                          if abs(opp_monster_xy(s)[0] - xy[0]) + abs(opp_monster_xy(s)[1] - xy[1]) < 80}  # fmt: skip
                continue
            if not mine:
                return  # only the opponent's are framed and no policy decision: not taken by a rule
            self.trace.rule("tribute-lowest-atk")

            def atk(c):
                k = self.cards.get(self.passwords.get(c.cid))
                return k.attack if k is not None else 0

            s, c = min(mine, key=lambda t: atk(t[1]))
            if not self.ui.card_command(my_monster_xy(s), f"tribute {self.name(c.cid)}", "confirm tribute", 2.0):
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                unpickable.add(my_monster_xy(s))
                continue
            taken.add(s)
            n += 1

    @staticmethod
    def nearest_spot(xy, spots):
        return min(spots, key=lambda p: abs(p[0] - xy[0]) + abs(p[1] - xy[1])) if spots else xy

    def monster_attack_at(self, raw: RawDuel, xy) -> int:
        """The ATK of our monster drawn at ``xy`` (a large number when none is there)."""
        me = raw.players[raw.me]
        for s in range(7):
            c = me.slots[s]
            m = my_monster_xy(s)
            if c is not None and abs(m[0] - xy[0]) + abs(m[1] - xy[1]) < 80:
                k = self.cards.get(self.passwords.get(c.cid))
                return k.attack if k is not None else 0
        return 1 << 20

    @staticmethod
    def occupied(raw: RawDuel, spots) -> list:
        """The highlighted spots that hold a monster: lava glowing in an empty zone passed for a highlight."""
        me, op = raw.players[raw.me], raw.players[1 - raw.me]
        cards = [my_monster_xy(s) for s in range(7) if me.slots[s] is not None]
        cards += [opp_monster_xy(s) for s in range(7) if op.slots[s] is not None]
        cards += [my_spell_xy(s) for s in range(6) if me.slots[7 + s] is not None]
        cards += [opp_spell_xy(s) for s in range(6) if op.slots[7 + s] is not None]
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
        self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=4.0)
        if self.ui.dialog() == "list":
            # a card list was still sliding in when the step was classified (run s87: Spiral Spear Strike's
            # "discard 1 card from your hand", mode 8): the loop answers it as a list
            self.trace.observe(field_pick="a card list came up: answered as a list")
            return
        if getattr(self, "replay_target", False):
            self.replay_target = False
            if self.replay_pick(raw):
                return
        episode = (raw.turn, raw.phase)
        if self.field_picked[0] != episode:
            self.field_picked = (episode, [])
        picked = self.field_picked[1]
        near = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) < 80  # noqa: E731 (a picked frame read 40 px off once counted as fresh)
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
            if theirs:
                xy = theirs[0]
                self.trace.rule("field-pick-opponent-first")
            else:
                # our own monsters only (a Tribute for a cost, Soul Crossing's): the lowest ATK first (Apprentice
                # Illusion Magician, 2000, was tributed while a 1000 Ra's Disciple was framed too)
                xy = min(fresh, key=lambda p: self.monster_attack_at(raw, p))
                self.trace.rule("field-pick-own-lowest-atk")
        for target in [xy] + [q for q in fresh if q != xy]:
            if self.ui.card_command(target, "highlighted card", "confirm"):
                picked.append(target)
                return
            if target == getattr(self, "field_shadow_spot", None):
                self.observer.shadow = None  # the shadow's pick shows no Confirm: MD does not allow it
                self.field_shadow_spot = None
            # no Confirm on this one (Millennium Seeker, framed among three Tribute candidates): another
            self.trace.observe(field_pick_no_confirm=list(target))
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
        # no framed card took it: a frame can be hidden (an attack arrow over our Set card asked as a cost, run
        # s59). Every other card on the field is opened and taken only when MD's command mask (memory) says it
        # can be selected (the Decide bit)
        tried = [xy] + fresh
        for target in self.field_spots(raw):
            if any(abs(target[0] - q[0]) + abs(target[1] - q[1]) < 80 for q in tried + picked):
                continue
            selectable = self.selectable_in_memory(target)
            if selectable:
                if self.ui.card_command(target, "a card memory says can be selected", "confirm"):
                    self.trace.rule("field-pick-by-command-mask", spot=list(target))
                    picked.append(target)
                    return
            elif selectable is None:
                # no command object for the card (our Field Zone's Galloping Gaia as Curse of Dragonfire's
                # target, run s72): in a field selection a card shows only Confirm, and only when it can be picked
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                if self.ui.card_command(target, "a card whose Confirm shows", "confirm"):
                    self.trace.rule("field-pick-by-confirm-icon", spot=list(target))
                    picked.append(target)
                    return
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
        raise Stuck("no Confirm icon above any selected card")

    def field_spots(self, raw: RawDuel) -> list:
        """Screen spots of every card on the field: ours first (Spell & Trap row, then monsters), then theirs."""
        me, op = raw.players[raw.me], raw.players[1 - raw.me]
        out = [my_spell_xy(s) for s in range(6) if me.slots[7 + s] is not None]
        out += [my_monster_xy(s) for s in range(7) if me.slots[s] is not None]
        out += [opp_monster_xy(s) for s in range(7) if op.slots[s] is not None]
        out += [opp_spell_xy(s) for s in range(6) if op.slots[7 + s] is not None]
        return out

    def selectable_in_memory(self, xy) -> bool | None:
        """Open the card at ``xy`` and read MD's command mask for it: whether the Decide bit (a card choice
        without a list) is set. False when the client is not readable; None when the card has no command object."""
        client = getattr(self, "client", None)
        if client is None:
            return False
        self.ui.click(*xy, "open: may this card be selected?", 0.9)
        try:
            oc = client.open_command(self.game.read().me)
        except Exception:  # noqa: BLE001
            return False
        ok = oc is not None and bool(oc.mask & CommandBit.DECIDE)
        self.trace.observe(selectable={"spot": list(xy), "mask": oc.mask if oc is not None else None, "ok": ok})
        return None if oc is None else ok
