"""Main Phase commands of the Solo player (:class:`ygorl.selfplay.loop.Player`): which hand card is open
(memory, the client's lists, the learned panel), the zones, the click scripts of :meth:`main_phase`, and the
command checks in memory (:meth:`command_by_mask`, :meth:`took_effect`, the Graveyard viewer)."""

from __future__ import annotations

from collections import Counter
import time

import numpy as np

from ygorl.engine import constants as C
from ygorl.md import layout as L
from ygorl.md.memory import MemoryReadError, RawDuel
from ygorl.md.ui import (
    GY_ICON,
    GY_STRIP_SHOWN,
    PHASE_MENU_CANCEL,
    Stuck,
    command_order,
    icon_kind,
    is_command_icon,
    my_monster_xy,
    opp_monster_xy,
    yellow_highlight,
    zone_outlined,
)
from ygorl.md.panel import name_fits, name_signature, name_words, panel_kind
from ygorl.selfplay.state import (
    FIELD_ICON_STEP,
    FIELD_ICON_DY,
    HAND_LOCATION,
    COMMAND_BITS,
    NEUTRAL_SPOT,
    MAIN,
    BATTLE,
    PLACE,
    POSITION_DIALOG,
    _action_key,
    _board_key,
)
from ygorl.selfplay.recovery import Failure


class CommandsMixin:
    """Main Phase commands: identify the open hand card, pick zones, click and check commands."""

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
        told = self.verify_by_client(raw, cid, xy, icons)
        if told is not None:
            return told
        told = self.verify_by_memory(raw, cid, xy, icons)
        if told is not None:
            return told
        return self.verify_on_panel(raw, cid, xy, icons, hand)

    def open_hand_cid(self, raw: RawDuel, before=None) -> int | None:
        """The card id of the hand card MD has open now (memory: the client's CommandOperation, no calibration);
        None when not readable. ``before``: a (location, index, cid) read before the click, which must change."""
        client = getattr(self, "client", None)
        if client is None:
            return None
        for _ in range(6):
            try:
                oc = client.open_command(raw.me)
            except Exception:  # noqa: BLE001
                return None
            if oc is not None and oc.location == HAND_LOCATION and oc.cid and (oc.location, oc.index, oc.cid) != before:
                return oc.cid
            time.sleep(0.25)
        return None

    def verify_by_client(self, raw: RawDuel, cid: int, xy, icons):
        """The opened hand card told by the client's open command (memory, needs no calibration: a duel's first
        hand card fell to the information panel, which could not name Millennium Revelation); the other hand
        positions are opened when it is another card. None when the client cannot tell."""
        got = self.open_hand_cid(raw)
        if got is None:
            return None
        self.trace.observe(selected={"want": cid, "got": got, "by": "client", "at": list(xy)})
        if got == cid:
            return icons
        hand = raw.players[raw.me].piles["hand"]
        for i in range(len(hand)):
            pos = L.hand_card(i, len(hand))
            if pos == xy:
                continue
            found = self.ui.open_hand_card(pos, f"hand #{i + 1}: looking for {self.name(cid)}")
            got = self.open_hand_cid(raw)
            if got == cid:
                self.trace.observe(selected={"want": cid, "found_at": i, "by": "client"})
                return found
        return None  # the memory read and the panel decide

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
        slots = raw.players[raw.me].slots
        for location in ("mzone", "szone"):
            for zone in range(5):
                if slots[zone if location == "mzone" else 7 + zone] is not None:
                    continue  # occupied (memory): a lit card's frame there is no outline (run s64: Gaia's zone)
                xy = L.my_zone(location, zone)
                if yellow_highlight(img, *xy) or zone_outlined(img, *xy):
                    return location, zone
        # the opponent's Monster Zones (Ra - Sphere Mode is Special Summoned to the opponent's field): the banner
        # hides their outlines' tops, so their bottom edge (y +70) tells
        # (every free zone of theirs outlined, at least two: one or two lit edges were lava on 31 of 909 screens)
        free = [z for z in range(5) if raw.players[1 - raw.me].slots[z] is None]

        def edge(zone):
            x, y = opp_monster_xy(zone)
            band = img[y + 64 : y + 76, x - 90 : x + 90].reshape(-1, 3).astype(int)
            return ((band[:, 0] > 100) & (band[:, 1] > 110) & (band[:, 2] < 90)).mean() > 0.25

        if len(free) >= 2 and all(edge(z) for z in free):
            return "opp_mzone", free[0]
        # no outline seen: the card MD places (its request names it, a mix entry of kind 5) tells the row (Dark
        # Spell Regeneration's Spell was clicked toward a Monster Zone)
        try:
            cid = next((v for k, v in self.request_now(raw).params if k == 5), 0)
        except Exception:  # noqa: BLE001
            cid = 0
        card = self.cards.get(self.passwords.get(cid, 0))
        location = "szone" if card is not None and not card.is_monster else "mzone"
        return location, self.free_zone(raw, location)

    def command_at_zone(self, location: str, zone: int, kind: str, order) -> None:
        """When the command click did not take, the zone click makes MD place the selected hand card on that zone
        and ask the command again there (Summon / Set icons above the zone): answer it with the same command."""
        x, y = L.my_zone(location, zone)
        self.ui.sleep(0.6)
        if self.read_dialog_type() == POSITION_DIALOG or self.ui.dialog() == "position":
            # the command took: MD asks the summoned monster's position, whose two icons stand above the zone too
            # (run s67: Lord Gaia's Special Summon, its position icons read as two Set icons, stuck)
            return
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
        self.last_zone = None  # an earlier action's zone is never this one's (a stale one opened a Set card twice)
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
            else:
                xy = L.my_zone("szone", where.sequence)  # (5: the Field Zone)
            if self.activate_by_mask(raw, best, cid, xy):
                return
            if getattr(self, "mask_refused", False):
                return
            if not self.ui.card_command(xy, f"field {self.name(cid)}", "activate", 2.0, kind="activate"):
                # a draw reveal or an effect's animation can cover the field: settle and look once more
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
                if not self.ui.card_command(xy, f"field {self.name(cid)} (again)", "activate", 2.0, kind="activate"):
                    return self.refused(raw, best, f"no Activate for {self.name(cid)} on the field")
            return
        if where.location == C.LOCATION_MZONE and best.kind == "summon":
            # a Gemini monster already on the field is Normal Summoned again (Red-Eyes Archfiend of Lightning): its
            # Summon icon above it, no zone
            xy = my_monster_xy(where.sequence)
            if not self.ui.card_command(xy, f"field {self.name(cid)}", "summon (gemini)", 2.0, kind="summon"):
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                return self.refused(raw, best, f"no Summon for {self.name(cid)} on the field")
            return
        if where.location == C.LOCATION_GRAVE and best.kind == "activate":
            return self.graveyard_activate(raw, best, cid)
        if where.location == C.LOCATION_MZONE and best.kind == "reposition":
            # a monster's change-of-position icon: only taken when it is the card's single icon (no glyph measured)
            xy = my_monster_xy(where.sequence)
            if self.command_by_mask(raw, best, cid, xy, "reposition"):
                return
            if getattr(self, "mask_refused", False):
                return
            icons = self.ui.open_card(xy, f"field {self.name(cid)}")
            oc = self.open_command(raw, cid)
            want = COMMAND_BITS["reposition"]
            if oc is not None and not oc.mask & want:
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                return self.refused(
                    raw, best, f"{self.name(cid)}: MD's command mask {oc.mask:#x} has no position change"
                )
            if oc is not None and not icons:
                # MD offers it (memory) but no icon was seen yet: the board can still be animating; look again
                self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
                icons = self.ui.open_card(xy, f"field {self.name(cid)} (again: MD offers the position change)")
            nth = [i for i, c in enumerate(oc.commands) if want & (1 << c)] if oc is not None else []
            if nth and len(icons) == len(oc.commands):
                self.trace.observe(open_command={"mask": oc.mask, "commands": list(oc.commands), "icon": nth[0]})
                icons = [sorted(icons)[nth[0]]]
            if len(icons) > 1:
                # beside an Activate icon (Millennium Seeker): the one that is not Activate
                img = self.ui.capture()
                icons = [p for p in icons if icon_kind(img, *p) != "activate"]
            if len(icons) != 1:
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                return self.refused(
                    raw, best, f"{len(icons)} icons on {self.name(cid)}: cannot tell the position change"
                )
            self.ui.click(*icons[0], "change position", 1.5)
            return
        if where.location != C.LOCATION_HAND:
            # no click script yet (a Special Summon from the Extra Deck, run s67): nothing was clicked, so the action
            # is masked for this phase and the policy decides again, rather than a generic recovery
            return self.refused(raw, best, f"no click script for {best.kind} from location {where.location}")
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
        oc = self.open_command(raw, cid)
        if oc is not None:
            want = COMMAND_BITS.get(best.kind, 0)
            if not oc.mask & want:
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                return self.refused(raw, best, f"{self.name(cid)}: MD's command mask {oc.mask:#x} has no {best.kind}")
            nth = [i for i, c in enumerate(oc.commands) if want & (1 << c)]
            if nth and len(icons) > len(oc.commands):
                # more icons than commands: a field card's glow in the icon band (a Special Summon clicked
                # Red-Eyes Archfiend's glow off the row): keep the icons on the row the others share
                row_y = sorted(p[1] for p in icons)[len(icons) // 2]
                icons = [p for p in icons if abs(p[1] - row_y) <= 20]
            if nth and len(icons) != len(oc.commands):
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
                icons = self.ui.open_hand_card(xy, f"hand {self.name(cid)} (again: {len(oc.commands)} commands)")
                if len(icons) > len(oc.commands):
                    row_y = sorted(p[1] for p in icons)[len(icons) // 2]
                    icons = [p for p in icons if abs(p[1] - row_y) <= 20]
                if len(icons) != len(oc.commands) and best.kind == "spsummon":
                    # Special Summon has no glyph of its own: never guessed when MD's command list is known
                    self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                    return self.refused(raw, best, f"{len(icons)} icons for {len(oc.commands)} commands")
            if nth and len(icons) == len(oc.commands):
                # the icons left to right are MD's command list in order: the icon is known, not guessed by glyph
                self.trace.observe(open_command={"mask": oc.mask, "commands": list(oc.commands), "icon": nth[0]})
                before = self.snapshot_state()
                self.ui.click(*sorted(icons)[nth[0]], best.kind, 0.5)
                if not self.took_effect(before):
                    # checked in memory at once: no effect, so the card is opened again and its icon clicked once
                    # more (never a third time: the cause is reported)
                    self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                    self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
                    again = self.ui.open_hand_card(xy, f"hand {self.name(cid)} (again: the click had no effect)")
                    if len(again) > len(oc.commands):
                        row_y = sorted(p[1] for p in again)[len(again) // 2]
                        again = [p for p in again if abs(p[1] - row_y) <= 20]
                    if len(again) != len(oc.commands) or self.open_hand_cid(raw) not in (cid, None):
                        self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                        raise Failure("command-no-effect", f"{best.kind} {self.name(cid)} from the hand", best)
                    before = self.snapshot_state()
                    self.ui.click(*sorted(again)[nth[0]], f"{best.kind} (again)", 0.5)
                    if not self.took_effect(before):
                        self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                        raise Failure("command-no-effect", f"{best.kind} {self.name(cid)} from the hand", best)
                self.ui.sleep(1.0)
                icons = None
        if icons is not None and self.ui.choose_command(icons, best.kind, order, self.name(cid)) is None:
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
        if card.type & C.TYPE_FIELD:
            # a Field Spell goes to the Field Zone by itself: no Spell & Trap zone to click (one was clicked, empty
            # by luck, and the shadow recorded the card there)
            return
        location = "mzone" if card.is_monster else "szone"
        if best.kind in ("spsummon", "activate") and self.pending_question():
            self.last_zone = (location, self.free_zone(raw, location))  # the zone, should MD ask it later
            return  # MD asks first (a cost: Apprentice Illusion Magician's discard): the loop answers, then the zone
            # (a Normal Summon / Set goes to its zone straight away: a passing state once skipped that zone)
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

    def activate_by_mask(self, raw: RawDuel, best, cid: int, xy) -> bool:
        return self.command_by_mask(raw, best, cid, xy, "activate")

    def snapshot_state(self) -> tuple:
        """What a command changes when MD takes it (memory): the input (mode, accepting) and the board."""
        try:
            raw, inp = self.game.read(), self.input()
        except MemoryReadError:
            return (None, None)
        return ((inp.mode, inp.accepting), _board_key(raw.players[raw.me], raw.players[1 - raw.me]))

    def took_effect(self, before: tuple, timeout: float = 3.0) -> bool:
        """Whether a click just made had its effect: memory's input or board moved away from ``before``
        (:meth:`snapshot_state`), or one of MD's dialogs came up; polled for ``timeout`` seconds."""
        end = time.time() + timeout
        while time.time() < end:
            now = self.snapshot_state()
            if now[0] is not None and (now[0] != before[0] or now[1] != before[1]):
                return True
            if self.ui.dialog() not in (None, "phase_menu"):
                return True
            time.sleep(0.25)
        return False

    def command_by_mask(self, raw: RawDuel, best, cid: int, xy, kind: str) -> bool:
        """A field card's command (activate, attack) judged by MD's command mask (memory): refused when the mask
        lacks it (``self.mask_refused``), else its icon clicked by command index. Icons blink and a badge can pass
        for one (Ra's attack clicked Blaze Cannon's badge; Millennium Seeker's Activate was refused twice while
        offered): when the icons seen do not match the command list, the card is opened again, and failing that
        the icon's place is computed from the command index (one icon above the card, two at x -86 / +86).
        False when the mask is not readable: the glyph search decides."""
        self.mask_refused = False
        if getattr(self, "client", None) is None:
            return False
        icons = self.ui.open_card(xy, f"field {self.name(cid)}")
        oc = self.open_command(raw, cid)
        if oc is None:
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
            return False
        want = COMMAND_BITS[kind]
        if not oc.mask & want:
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
            self.mask_refused = True
            self.refused(raw, best, f"{self.name(cid)}: MD's command mask {oc.mask:#x} has no {kind}")
            return False
        nth = [i for i, c in enumerate(oc.commands) if want & (1 << c)][0]
        n = len(oc.commands)
        # an icon must sit where the command list puts it (one command: straight above the card; Obelisk's attack
        # clicked a mark 88 px to the right six times and never attacked)
        ax = L.MY_FIELD_ICON_X if tuple(xy) == L.MY_FIELD_ZONE else xy[0]  # (the Field Zone's icon is off the card)
        expect = [round(ax + (i - (n - 1) / 2) * FIELD_ICON_STEP) for i in range(n)]
        icons = [p for p in icons if min(abs(p[0] - e) for e in expect) <= 45]
        if len(icons) != n:
            # the card is open (memory says so): its icons blink, so look again over a blink cycle rather than
            # clicking the card again, which closed it (Ra's attack was then clicked on nothing eight times)
            icons = self.blinking_icons(xy, expect) or icons
        computed = (round(ax + (nth - (n - 1) / 2) * FIELD_ICON_STEP), xy[1] + FIELD_ICON_DY)
        # each click is checked in memory within seconds; one without effect is followed by the next way to reach
        # the icon (never the same one twice): the icon seen, its computed place, the card opened again
        if len(icons) != n:
            try:  # what an open card shows when its icons are not found (a Divine-Beast with Divine Evolution)
                np.save(self.out / f"command-open-{time.strftime('%Y%m%d-%H%M%S')}.npy", self.ui.capture())
            except Exception:  # noqa: BLE001
                pass
        tries = ([("seen", sorted(icons)[nth])] if len(icons) == n else []) + [
            ("computed", computed),
            ("reopened", None),
        ]
        # then the icon's x at other heights (a Divine-Beast with Divine Evolution drew its Attack icon elsewhere:
        # every such attack was clicked at the usual height and none took), each checked in memory
        tries += [(f"computed dy {dy}", (computed[0], xy[1] + dy)) for dy in (-125, -175, -200, -100)]
        done = []
        for how, target in tries:
            if target is None:
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
                self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
                again = [p for p in self.ui.open_card(xy, f"field {self.name(cid)} (again)")
                         if min(abs(p[0] - e) for e in expect) <= 45]  # fmt: skip
                again = again if len(again) == n else self.blinking_icons(xy, expect)
                # after the card was opened again, its icon is clicked even where an earlier click went (that one
                # may have landed while the card was closed: Obelisk's attack failed four turns in a row)
                target = sorted(again)[nth] if len(again) == n else computed
            elif target in done:
                continue
            if how.startswith("computed dy") and self.open_command(raw, cid) is None:
                self.ui.open_card(xy, f"field {self.name(cid)} (open again for another height)")
            done.append(target)
            before = self.snapshot_state()
            self.ui.click(*target, kind, 0.5)
            ok = self.took_effect(before)
            self.trace.observe(open_command={"mask": oc.mask, "commands": list(oc.commands), "icon": nth,
                                             "icons_seen": len(icons), "how": how, "took": ok,
                                             "tried": [list(t) for t in done]})  # fmt: skip
            if ok:
                return True
        try:  # what the screen showed, for the next look at why no click took
            np.save(self.out / f"command-failure-{time.strftime('%Y%m%d-%H%M%S')}.npy", self.ui.capture())
        except Exception:  # noqa: BLE001
            pass
        self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
        raise Failure("command-no-effect", f"{kind} {self.name(cid)}: no click on its icon had an effect", best)

    def blinking_icons(self, xy, expect: list[int]) -> list:
        """The open field card's icons where its command list puts them (``expect``: their x), looked for over a
        blink cycle (about 4 s); [] when not all of them were seen at once."""
        n = len(expect)
        y0, y1 = xy[1] + FIELD_ICON_DY - 90, xy[1] + FIELD_ICON_DY + 60  # (a band: the height varies)
        box = (min(expect) - 70, y0, max(expect) + 70, y1)
        for _ in range(12):
            img = self.ui.capture()
            found = L.find_icons(img, *box) + L.blue_icons(img, *box) + L.green_icons(img, *box)
            seen = []
            for p in sorted(p for p in found if min(abs(p[0] - e) for e in expect) <= 45):
                if all(abs(p[0] - q[0]) > 60 for q in seen):
                    seen.append(p)
            if len(seen) == n:
                return seen
            self.ui.sleep(0.35)
        return []

    def open_command(self, raw: RawDuel, cid: int):
        """MD's command mask of the card open now (memory), when it is card ``cid`` of ours; else None."""
        client = getattr(self, "client", None)
        if client is None:
            return None
        for _ in range(6):
            try:
                oc = client.open_command(raw.me)
            except Exception:  # noqa: BLE001
                return None
            if oc is not None and oc.cid == cid:
                return oc
            time.sleep(0.25)
        return None

    def graveyard_activate(self, raw: RawDuel, best, cid: int) -> None:
        """An effect activated from our Graveyard (the user: it must be supported): open the Graveyard viewer, open
        the card (its strip position from the pile order in memory; which card is open is read from memory, the
        other positions tried if it is another), and click its Activate icon."""
        grave = raw.players[raw.me].piles["grave"]
        seq = best.card.loc.sequence
        guess = len(grave) - 1 - seq if 0 <= seq < len(grave) else 0  # the strip shows the top card first
        tries = [guess] + [i for i in range(len(grave)) if i != guess]
        if guess >= GY_STRIP_SHOWN:
            # deeper than the strip shows (it is not scrolled yet): refused for the turn without opening the viewer
            # (Ancient Chant at position 10 was tried twice a turn on cards 1-5)
            self.forbidden.add(((raw.turn, "*"), _action_key(best)))
            return self.refused(raw, best, f"{self.name(cid)} is #{guess + 1} in the Graveyard, past the strip")
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
            if sel is None:
                sel = (cid, 0)  # the position the pile order names: taken as the card (the turn-wide mask below)
            oc = self.open_command(raw, cid) if sel is not None else None
            if oc is not None:
                # memory: MD's command mask for the open Graveyard card decides, not the glyph (lava passed
                # for two Activate icons, and the click did nothing)
                self.trace.observe(open_command={"mask": oc.mask, "commands": list(oc.commands)})
                if not oc.mask & COMMAND_BITS["activate"]:
                    icons = []
                elif icons:
                    icons = [min(icons, key=lambda p: abs(p[0] - GY_ICON[0]) + abs(p[1] - GY_ICON[1]))]
                else:
                    icons = [GY_ICON]  # offered: its measured place above the pile
            else:
                # no command mask for a Graveyard card (the viewer does not set it): only an icon at the measured
                # place counts (lava glow 84 px off it passed for Ancient Chant's Activate again)
                icons = [p for p in icons if abs(p[0] - GY_ICON[0]) + abs(p[1] - GY_ICON[1]) <= 40]
            if icons:
                self.ui.click(*icons[0], "activate (graveyard)", 2.0)
                return
            if sel is not None:
                # memory says it is the card, and MD shows no Activate (greyed, or none at all): its rules differ
                # from the reconstruction's here (Ancient Chant's Graveyard effect): masked for the rest of the turn,
                # not just the phase (the policy picked it again seven times in run s25)
                self.forbidden.add(((raw.turn, "*"), _action_key(best)))
                self.trace.observe(graveyard_greyed=self.ui.graveyard_command_greyed(self.gy_before, shot))
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

    def _placed(self, location: str, zone: int) -> None:
        observer = getattr(self, "observer", None)  # observe only; tests build players without one
        if observer is not None:
            self.trace.observe(shadow_place=observer.placed(location, zone))
