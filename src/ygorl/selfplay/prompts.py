"""Prompts and MD's dialogs for the Solo player (:class:`ygorl.selfplay.loop.Player`): chain / open-timing
prompts, the client's prompt list, the dialogs (position, yes / no, effect choice, declare a name, ...) and the
summon pick."""

from __future__ import annotations

from collections import Counter
import time
from dataclasses import replace


from ygorl.md import layout as L
from ygorl.md.memory import PHASES, RawDuel, own_deck
from ygorl.md.ui import (
    COIN_OK,
    COIN_RINGS,
    MINIMISED_RESTORE,
    PHASE_MENU_CANCEL,
    Stuck,
    notice_button,
    radio_rows,
)
from ygorl.md.explore import find_buttons
from ygorl.md.request import Candidate
from ygorl.selfplay.observe import md_option_desc, option_of_description
from ygorl.selfplay.state import (
    OPTION_OK,
    RA,
    SLIFER,
    HAND_LOCATION,
    option_tab,
    option_tabs,
    NEUTRAL_SPOT,
    OPEN_PROMPT,
    Input,
    plain_name,
    searchable,
    _own_cards,
)


class PromptsMixin:
    """Chain prompts and MD's dialogs."""

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
        pick = None if theirs else self.lockstep("prompt", raw, lambda: self.prompt_request(raw))
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
        calibrated = getattr(getattr(self, "selection", None), "address", None) is not None
        listed = self.prompt_list()  # the cards offered (memory: the client's list; no calibration needed)
        # only monsters: a hand trigger is a monster's (Millennium Seeker, Mahad); a Quick-Play Spell offered at an
        # open timing (Soul Crossing) is a play the policy passed twice, and taking it by this rule lost a duel
        in_hand = [c for c in listed if c.location == HAND_LOCATION and self.passwords.get(c.cid)
                   and getattr(self.cards.get(self.passwords[c.cid]), "is_monster", False)]  # fmt: skip
        if (pick is None and not respond and inp.mode == OPEN_PROMPT and in_hand
                and getattr(self, "hand_trigger", None) != seen):  # fmt: skip
            # an optional trigger of a card in our hand (Millennium Seeker's free Special Summon): take it
            card = self.passwords[in_hand[0].cid]
            self.trace.rule("prompt-respond-hand-trigger", card=in_hand[0].cid, source="client list")
            if self.tick_prompt_card(card) == card:
                self.listed = (self.expect, (in_hand[0].cid,))  # names the monster whose position is asked next
                self.ui.click(*L.PROMPT_ACTIVATE, "activate", 2.0)
                self.context, self.hand_trigger = "respond", seen  # the same prompt again: it did not take, pass
                return
        if (pick is None and not respond and inp.mode == OPEN_PROMPT and calibrated and not listed
                and getattr(self, "hand_trigger", None) != seen):  # fmt: skip
            # an optional trigger of a card in our hand (Millennium Seeker's "You have received damage. Activate?"
            # in the opponent's turn: a free Special Summon was passed): tick it, read it from memory, take it when
            # it is a hand card (not a Set Trap, which the open timing would waste)
            self.ui.click(*L.PROMPT_CARD, "tick the offered card", 0.8)
            now = self.game.read()
            got = self.selection.read(now)
            hand = {c.uid for c in now.players[now.me].piles["hand"]}
            if got is not None and got[1] in hand:
                self.trace.rule("prompt-respond-hand-trigger", card=got[0])
                self.listed = (self.expect, (got[0],))  # names the monster whose position is asked next
                self.ui.click(*L.PROMPT_ACTIVATE, "activate", 2.0)
                self.context, self.hand_trigger = "respond", seen  # the same prompt again: it did not take, pass
                return
        if pick is None:
            self.trace.rule("prompt-respond" if respond else "prompt-pass")
        want = getattr(self, "respond_card", None) if pick is not None else None
        self.respond_card = None
        if respond and want is None and listed and self.passwords.get(listed[0].cid):
            # the rule responds: to the card MD lists first, ticked through memory so the card activated is known
            # (Palladium Oracle Mahad's position question after a rule response fell to a rule)
            want = self.passwords[listed[0].cid]
        if respond and want is not None:
            # the rebuild's answer is for the window, not for the card MD offers here: tick it, read which card
            # it is from memory, and pass when it is another one (Ultimate Divine-Beast was flipped for a Mirror
            # Force answer)
            got_pw = self.tick_prompt_card(want)
            if got_pw is not None and got_pw != want:
                print(f"  prompt: MD offers {got_pw}, the policy meant another card: pass", flush=True)
                self.ui.respond_prompt(False)
                self.responded = None
                return
            self.ui.click(*L.PROMPT_ACTIVATE, "activate", 2.0)
            self.context = "respond"
            if getattr(self, "prompted", 0):
                # the card activated names the monster a summon's position question is about next (Millennium
                # Seeker's own Special Summon took the Attack Position rule: "the summoned monster is not known")
                self.listed = (self.expect, (self.prompted,))
            return
        self.ui.respond_prompt(respond)
        if respond:
            self.context = "respond"

    def observe_client_list(self) -> None:
        """Record the client's card list (memory) next to what the screen shows: whether its order is the screen's
        (then lists need no tick to identify a card) is checked on these records."""
        client = getattr(self, "client", None)
        if client is None:
            return
        try:
            got = client.list_cards(self.game.read())
        except Exception as exc:  # noqa: BLE001
            got = f"{type(exc).__name__}: {exc}"
        if isinstance(got, tuple):
            got = {"type": got[0], "cards": [[c.cid, c.location, c.index, c.uid] for c in got[1]]}
        self.trace.observe(client_list=got)

    def prompt_request(self, raw: RawDuel):
        """MD's request with the prompt's offered cards as candidates (memory: the client's list; the engine's
        request holds none for open prompts, and the shadow's "pass" once answered a prompt for another card:
        Obelisk's End Phase send was cancelled)."""
        req = self.request_now(raw)
        listed = self.prompt_list()
        if listed and not req.candidates:
            req = replace(req, candidates=tuple(Candidate(c.cid, c.uid, c.player, "", None) for c in listed))
        return req

    def prompt_list(self) -> list:
        """The cards MD's open / chain prompt offers (memory, the client's list), [] when not readable."""
        client = getattr(self, "client", None)
        if client is None:
            return []
        try:
            raw = self.game.read()
            if self.request_now(raw).kind not in ("open", "chain"):
                return []
            return client.prompt_cards(raw) or []
        except Exception:  # noqa: BLE001
            return []

    def tick_prompt_card(self, want: int) -> int | None:
        """Tick the card a response prompt offers that the policy means (password ``want``): MD can offer several
        side by side (Sun God Unification and Dark Spell Regeneration: only the first was ever ticked, and the
        prompt passed). Which card is ticked is read from memory; the first one's password when none is ``want``."""
        sel = getattr(self, "selection", None)
        first, first_xy, tried = None, None, set()
        listed = self.prompt_list()
        order = []
        if listed:
            # the prompt's own list (memory): how many cards and which is where (assumed in the list's order)
            spots = L.prompt_cards(len(listed))
            order = [spots[i] for i, c in enumerate(listed) if self.passwords.get(c.cid) == want]
            order += [xy for xy in spots if xy not in order]
            self.trace.observe(prompt_list=[c.cid for c in listed])
            # the client's list (memory) names the card at each place: tick the meant one; the open-card read only
            # overrides it when it names another card of ours (an uncalibrated or stale read, None, does not:
            # Millennium Seeker's free Summon was passed twice for want of that read)
            self.ui.click(*order[0], "tick the offered card (the client's list names it)", 0.8)
            cid = listed[spots.index(order[0])].cid
            got = sel.read(self.game.read()) if sel is not None and sel.address is not None else None
            if got is not None and got[0] != cid:
                cid = got[0]
            self.prompted = cid
            got_pw = self.passwords.get(cid)
            self.trace.observe(prompt_card={"want": want, "got": got_pw, "by": "client list", "read": got and got[0]})
            return got_pw
        for count in range(1, 5):
            for xy in order if count == 1 and order else [] if order else L.prompt_cards(count):
                if xy in tried:
                    continue
                tried.add(xy)
                self.ui.click(*xy, "tick the offered card", 0.8)
                got = sel.read(self.game.read()) if sel is not None else None
                got_pw = self.passwords.get(got[0]) if got else None
                self.prompted = got[0] if got else 0
                self.trace.observe(prompt_card={"want": want, "got": got_pw, "at": xy})
                if got_pw == want or sel is None or sel.address is None:
                    return got_pw  # the card meant, or memory cannot tell another card
                if first is None and got_pw is not None:
                    first, first_xy = got_pw, xy
        if first_xy is not None:
            self.ui.click(*first_xy, "tick the first card again", 0.8)
        return first

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
            # memory: a tab without the enabled bit is an effect spent this turn; remembered, so no later decision
            # takes it, and the policy's next activation of this card among the enabled tabs is taken instead (a
            # greyed tab could not be selected, and a rule then paid a cost the policy never chose)
            disabled = {i for i, c in enumerate(req.choices) if not c >> 31}
            code = self.expect.action.card.code if self.expect is not None and self.expect.action.card else 0
            spent = getattr(self, "spent_effects", set())
            for i in disabled:
                desc = md_option_desc(req.choices[i], self.passwords)
                if code and desc:
                    spent.add((raw.turn, code, desc))
            self.spent_effects = spent
            if pick is not None and pick in disabled:
                observer = getattr(self, "observer", None)
                if observer is not None:
                    observer.shadow = None  # it took the spent effect: out of step from here
                pick = None
                for g in getattr(self, "last_allowed", []):
                    a = g.action
                    if a.kind == "activate" and a.card is not None and a.card.code == code:
                        tab = option_of_description(req.choices, getattr(a, "description", 0), self.passwords)
                        if tab is not None and tab not in disabled:
                            pick = tab
                            self.trace.rule("option-policy-next-enabled", pick=pick, p=round(g.prob, 4))
                            break
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
        elif kind == "coin":
            # a coin call (Time Wizard): either face is a guess with the same odds; no policy decides it
            self.trace.rule("coin-call-first")
            self.ui.click(*COIN_RINGS[0], "call the coin (left face)", 0.8)
            self.ui.click(*COIN_OK, "ok", 2.0)
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
            if pick is None and self.context in ("summon", "tribute"):
                # our own summon asks to confirm the Tribute Summon (message 335): the tributes come next; still
                # up after a yes that did not take ("tribute"), it is the same question, not one to answer no
                pick = 0
                self.trace.rule("yesno-yes-tribute")
            elif pick is None and self.context in ("activate", "spsummon", "respond"):
                # our own card asks about an optional part of the effect we chose to activate ("add the designated
                # card from the Deck to your hand?" of The True Name): take it
                pick = 0
                self.trace.rule("yesno-yes-own-effect")
            elif pick is None and self.context.startswith("attack") and getattr(self, "attacker", None):
                # a replay ("the opponent's monsters changed: continue the attack?"): the policy on the new board,
                # its attacker not counted as having attacked: yes when it still attacks with it
                pick = 0 if self.still_attacks(raw) else 1
                self.trace.rule("replay-by-policy", attack=pick == 0)
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
                # the client holds the list MD shows (memory): its own cards are the candidates (Millennium
                # Revelation's cost list read stale and a rule sent Obelisk while the policy meant Ra)
                req, stale = self.request_from_client_list(raw, req, shown)
            if stale:
                # (a choice without a list whose check is not modelled, e.g. Millennium Revelation's hand card;
                # Apprentice Illusion Magician's discard read a stale list until its check was, design 10)
                self.trace.observe(list_request={"stale": [[c.cid, c.location, c.sequence] for c in req.candidates],
                                                 "shown": shown})  # fmt: skip
            self.listed = (self.expect, tuple(c.cid for c in req.candidates))  # names a monster summoned next
            self.observe_client_list()
            need = next((v for k, v in req.params if k == 8), 1)  # "choose N cards" (Fusion Materials)
            if 1 < need <= len(L.list_cards(self.ui.capture())):
                self.trace.rule("list-set-first-accepted", need=need)  # MD's list here is not the one in memory
                self.trace.observe(list_set=self.ui.pick_list_set(need))
                self.context = "list"
                return
            hidden = self.hidden_in_lists()
            exclude = frozenset(i for i, c in enumerate(req.candidates)
                                if [self.passwords.get(c.cid), c.location] in hidden)  # fmt: skip
            if exclude:
                self.trace.observe(list_hidden=sorted(exclude))
            pick = None if stale else self.lockstep("list", raw, req, exclude=exclude)
            if pick is None and self.ra_pays_lp(raw, req) is False:
                # The Winged Dragon of Ra's optional "pay LP so that you have 100" trigger (a one-card list of Ra
                # itself, Cancel on the left): taken by "first on screen", it paid 8000 -> 100 on turn 1, when Ra
                # cannot attack, and the duel was lost next turn. Declined unless Ra can attack this turn
                self.trace.rule("ra-pay-lp-declined")
                self.ui.click(*L.PROMPT_CANCEL, "cancel: keep the LP", 1.5)
                self.context = "list"
                return
            deck = [i for i, c in enumerate(req.candidates) if c.location == "deck"]
            if pick is None and self.ra_pays_lp(raw, req):
                self.trace.rule("ra-pay-lp-taken")
                index, ticked = 0, False
            elif pick is None and deck and len({self.passwords.get(c.cid) for c in req.candidates}) == 1:
                # copies of one card in different places (Mahad: the hand's Dark Magician or the Deck's): the
                # Deck's, so the hand keeps its card
                self.trace.rule("list-copies-from-the-deck")
                index, ticked = self.find_in_list(req, deck[0])
            elif pick is None:
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

    def ra_pays_lp(self, raw: RawDuel, req) -> bool | None:
        """For the one-card list of The Winged Dragon of Ra's pay-LP trigger: whether to pay (Ra can attack this
        turn: our Main Phase 1, not the duel's first turn); None for any other list."""
        if len(req.candidates) != 1 or req.candidates[0].location != "mzone":
            return None
        if self.passwords.get(req.candidates[0].cid) != RA:
            return None
        phase = PHASES[raw.phase] if raw.phase < len(PHASES) else ""
        return raw.turn_player == raw.me and phase == "main1" and raw.turn > 0

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
            pw = self.passwords.get(cid)
            k = self.cards.get(pw)
            if k is None or not k.is_monster:
                return -1
            if pw == SLIFER:
                return 1000 * (len(hand) - 1)  # 1000 per card in hand once it has left it
            if k.attack < 0:
                return 0  # another "?" ATK (Ra: its tributes' total, unknown here): not ahead of a known one
            return k.attack

        guessed = sorted(range(len(hand)), key=lambda i: strength(hand[i]), reverse=True)
        order = list(dict.fromkeys([L.hand_card(L.hand_order(hand, lambda c: self.passwords.get(c, 0),
                                                              self.cards.get).index(i), len(hand)) for i in guessed]
                                   + [L.hand_card(i, len(hand)) for i in range(len(hand))]))  # fmt: skip
        want = self.summon_pick_by_policy(raw)
        if want is not None:
            # the policy's monster (through the shadow): the positions are opened until the client says that card
            # is the open one (both Ra cards have ? ATK, and a guess summoned Sphere Mode)
            for n, xy in enumerate(order):
                icons = self.ui.open_hand_card(xy, f"hand position {n + 1}: {self.name(want)}?")
                if self.open_hand_cid(raw) == want and icons and self.ui.choose_command(
                    icons, "summon", ["summon", "mset"], self.name(want)
                ) is not None:  # fmt: skip
                    self.context, self.summon_level = "summon", 5
                    return
                self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
        self.trace.rule("summon-pick-first-offered", hand=hand)
        for n, xy in enumerate(order):
            icons = self.ui.open_hand_card(xy, f"hand position {n + 1}: offered for the summon?")
            if icons and self.ui.choose_command(icons, "summon", ["summon", "mset"], "the offered monster") is not None:
                self.context = "summon"
                self.summon_level = 5  # unknown until memory says: let MD's own Tribute question through
                return
            self.ui.click(*NEUTRAL_SPOT, "close the card", 0.8)
        raise Stuck(f"no hand card shows a Summon icon for the summon pick ({len(order)} positions tried)")

    def summon_pick_by_policy(self, raw: RawDuel) -> int | None:
        """The MD card id of the hand monster the policy summons for MD's summon pick, through the shadow (its
        SelectCard over the hand's monsters); None without a shadow decision."""
        me = raw.players[raw.me]
        cands = tuple(Candidate(c.cid, c.uid, raw.me, "hand", i) for i, c in enumerate(me.piles["hand"])
                      if getattr(self.cards.get(self.passwords.get(c.cid)), "is_monster", False))  # fmt: skip
        if len({c.cid for c in cands}) < 2:
            return None
        try:
            req = replace(self.request_now(raw), candidates=cands)
        except Exception:  # noqa: BLE001
            return None
        pick = self.lockstep("list", raw, req)
        return cands[pick].cid if pick is not None and 0 <= pick < len(cands) else None

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
