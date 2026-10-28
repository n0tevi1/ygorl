"""Policy decisions of the Solo player (:class:`ygorl.selfplay.loop.Player`): the policy's top action through
the advisor, masked by what MD offers and refused; in-action picks through the shadow duel (:meth:`lockstep`) or
rebuilt from the current board (:meth:`rebuilt`)."""

from __future__ import annotations

import random
import time


from ygorl.engine import constants as C
from ygorl.md import layout as L
from ygorl.md.memory import RawDuel, own_deck, to_snapshot
from ygorl.md.ui import Stuck
from ygorl.md.client import aura_kinds
from ygorl.selfplay.state import (
    Expectation,
    clickable,
    ATTACKS_MASKED,
    _action_key,
)


class PolicyMixin:
    """The policy's decisions, masked by MD; shadow lockstep and rebuilt sub-decisions."""

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

    def md_offers(self, raw: RawDuel):
        """A test of whether MD offers an action, from the cards it highlights (memory: the client's command
        groups per card, :meth:`ygorl.md.client.Client.highlighted`): a hand / field card MD does not light has no
        command, and a lit card only the commands of its groups. Anything else (phases, Graveyard cards, a
        highlight not readable yet) passes; masked actions are recorded in the trace."""
        client = getattr(self, "client", None)
        lit: dict = {}
        if client is not None:
            for _ in range(10):  # MD lights the cards 0.2-2 s after the input mode changes
                try:
                    lit = client.highlighted(raw)
                except Exception:  # noqa: BLE001
                    lit = {}
                if lit:
                    break
                time.sleep(0.3)
        if not lit and getattr(client, "views_complete", False):
            # nothing lit yet: right after the Draw Phase memory is in Main Phase 1 while the client still draws
            # (its "Turn N Draw" badge up, nothing lit for seconds; run s83 masked every card and skipped whole
            # turns). Only an empty highlight on a settled screen counts as "no card has a command"
            self.ui.wait_still((0, 0, L.REF_W, L.REF_H), timeout=6.0)
            for _ in range(10):
                try:
                    lit = client.highlighted(self.game.read())
                except Exception:  # noqa: BLE001
                    lit = {}
                if lit:
                    break
                time.sleep(0.3)
        if not lit and not getattr(client, "views_complete", False):
            return lambda action: True  # (the highlight not readable: no test)
        if not lit:
            self.trace.observe(md_nothing_lit=True)  # every card masked by an empty highlight (audited)
        # every own card's view read and none lit: MD offers no hand / field card a command (a set Pot of
        # Extravagance the rebuild could activate in Main Phase 2, run s64: MD refused it twice)
        me = raw.players[raw.me]
        masked = []

        def uids(card) -> set:
            loc = card.loc
            if loc.location == C.LOCATION_HAND:
                return {c.uid for c in me.piles["hand"] if self.passwords.get(c.cid) == card.code}
            if loc.location == C.LOCATION_MZONE and 0 <= loc.sequence < 7:
                slot = me.slots[loc.sequence]
            elif loc.location == C.LOCATION_SZONE and 0 <= loc.sequence < 6:
                slot = me.slots[7 + loc.sequence]
            else:
                return set()
            return {slot.uid} if slot is not None else set()

        def offered(action) -> bool:
            card = getattr(action, "card", None)
            if card is None or card.loc.location not in (C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE):
                return True
            ids = uids(card)
            if not ids:
                return True
            kinds = set().union(*(aura_kinds(lit[u]) for u in ids if u in lit)) if ids & set(lit) else set()
            ok = action.kind in kinds or (action.kind == "reposition" and "flip" in kinds)
            if not ok:
                masked.append([action.kind, card.code])
            return ok

        def record(action) -> bool:
            ok = offered(action)
            self.trace.observe(md_masked=masked)
            return ok

        return record

    def effect_spent(self, raw: RawDuel, action) -> bool:
        """An activation of an effect MD showed greyed out this turn (its option tab without the enabled bit:
        Millennium Revelation's once-per-turn effects, by card name across copies)."""
        if action.kind != "activate" or action.card is None or not getattr(action, "description", 0):
            return False
        return (raw.turn, action.card.code, action.description) in getattr(self, "spent_effects", set())

    def decide(self, raw: RawDuel):
        """The policy's top action for the current Main or Battle Phase board (and every legal suggestion)."""
        from ygorl.advisor.advise import advise

        stats = self.field_stats(raw)
        snap = to_snapshot(raw, self.passwords, frozenset(self.attacked), frozenset(self.activated), stats)
        # recorded: what the rebuild restores (auditors could not tell otherwise)
        self.trace.observe(field_stats={str(u): list(v) for u, v in stats.items()})
        deck = own_deck(raw, self.passwords, self.cards)
        advice = advise(snap, deck, self.policy, self.prior, self.env.rule_flags, cards=self.cards,
                        samples=self.samples, seed=random.randrange(1 << 30), starting_lp=self.env.player.starting_lp)  # fmt: skip
        key = (raw.turn, raw.phase)
        no_attacks = (key, ATTACKS_MASKED) in self.forbidden
        offered = self.md_offers(raw)
        allowed = [g for g in advice.suggestions if offered(g.action)
                   and (key, _action_key(g.action)) not in self.forbidden
                   and ((raw.turn, "*"), _action_key(g.action)) not in self.forbidden
                   and not (no_attacks and g.action.kind == "attack") and clickable(g.action)
                   and not self.effect_spent(raw, g.action)]  # fmt: skip
        self.last_allowed = allowed
        if not allowed:
            raise Stuck(f"no legal action left in the {snap.phase} reconstruction (MD refused the rest)")
        top = allowed[0]
        name = self.cards[top.action.card.code].name if top.action.card else ""
        print(f"policy ({snap.phase}): {top.action.kind} {name} p={top.prob:.2f}", flush=True)
        self.trace.policy(top.action, top.prob, snap.phase, allowed)
        self.trace.observe(shadow_build=self.observer.decided(top.action, snap, deck))
        self.expect = Expectation(top.action, raw, snap.phase)
        return top.action, allowed

    def _score(self, shadow, point):
        """The policy's logits on a shadow decision point (sub-picks inside our own action)."""
        from dataclasses import replace

        from ygorl.advisor.advise import _logits

        return _logits(replace(shadow.rec, point=point), self.policy, self.cards, self.env.player.starting_lp)

    def rebuilt(self, question: str, raw: RawDuel, *args, offered: tuple[int, ...] | None = None):
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
            extra = getattr(self, "extra_pick", (None, 0))
            if extra[1] and extra[0] is self.expect:
                listed = (self.expect, (extra[1],))  # the Extra Deck monster picked in this action names it
            shadow = getattr(getattr(self, "observer", None), "shadow", None)
            choice = decider.position(raw, self.request_now(raw), action, shadow, before,
                                      listed[1] if listed[0] is self.expect else ())  # fmt: skip
        else:
            # the cards MD offers (memory: the client's prompt list): the window's other activations are not this
            # prompt's (Millennium Seeker's trigger was passed on Sun God Unification's probabilities)
            if offered is None:
                listed = self.prompt_list()
                offered = tuple(p for c in listed if (p := self.passwords.get(c.cid))) or None
            choice = decider.respond(raw, self.request_now(raw), offered=offered)
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

    def lockstep(self, dialog: str, raw: RawDuel, req=None, count: int = 0, zone=None,
                 exclude: frozenset = frozenset()) -> int | None:  # fmt: skip
        """The option the policy picks for MD's ``dialog`` through the shadow duel (None: no shadow, or it does not
        match; the caller's rule decides). Recorded as the step's decision."""
        observer = getattr(self, "observer", None)  # tests build players without one
        if observer is None or observer.shadow is None:
            return None
        if callable(req):  # read only when there is a shadow to ask
            req = req()
        pick, record = observer.choose(dialog, req, raw.me, self.passwords, count=count, zone=zone, exclude=exclude)
        if record:
            self.trace.observe(lockstep=record)
        if pick is not None:
            self.trace.shadow(dialog, pick, action=record.get("action"))
            print(f"  shadow ({record.get('ygo')}): {dialog} option {pick + 1} of {record.get('options')}", flush=True)
        elif record:
            print(f"  shadow: {record.get('desync') or record.get('shadow_error')} ({record.get('ygo')})", flush=True)
        return pick
