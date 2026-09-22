"""A deterministic heuristic baseline: be proactive, attack what you can beat.

``GreedyAgent`` only looks at the current decision (plus a little per-turn
memory); it never tracks the board. Card stats come from the card database
(base ATK/DEF, not current values). Rules, in priority order:

* main phase: activate > special summon > normal summon (highest ATK) >
  set monster > set spell/trap > battle phase > end phase;
* battle phase: direct attack > attack with the highest ATK > activate >
  main phase 2 > end phase;
* attack target: the strongest target whose ATK (attack position) or DEF
  (defense position) the attacker's ATK strictly exceeds; if none, cancel the
  attack (and do not try that attacker again this turn), or, when it cannot be
  cancelled, the weakest target;
* chain whenever possible, answer yes, choose attack position when ATK >= DEF;
* card selections: keep selecting (tributes: lowest ATK first, otherwise
  seeded random), ``finish`` only when nothing else is offered, never unselect;
* anything else: a seeded random choice.

The same activation (card, location, effect) is used at most
``max_repeats`` times per turn, and at most ``max_turn_actions`` proactive
main-phase actions are taken per turn, so effects that can be activated and
cancelled repeatedly cannot stall a duel.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action

if TYPE_CHECKING:
    from ygorl.engine.duel import DecisionPoint

HINTMSG_ATTACKTARGET = 549  # constant.lua; MSG_HINT(HINT_SELECTMSG) before the attack target selection

_IDLE_ORDER = ("activate", "spsummon", "summon", "mset", "sset", "battle_phase", "end_phase")
_BY_ATK = ("summon", "spsummon", "mset")


def _key(a: Action) -> tuple:
    loc = a.card.loc if a.card is not None else None
    where = (loc.controller, loc.location, loc.sequence) if loc is not None else None
    return (a.kind, a.card.code if a.card is not None else 0, where, a.description)


class GreedyAgent:
    name = "greedy"

    def __init__(self, seed: int | None = None, *, cards: Mapping[int, Any] | None = None,
                 max_repeats: int = 3, max_turn_actions: int = 100) -> None:  # fmt: skip
        self.rng = random.Random(seed)
        self._cards = cards
        self.max_repeats = max_repeats
        self.max_turn_actions = max_turn_actions
        self._turn: int | None = None
        self._used: Counter[tuple] = Counter()  # activations/chains this turn
        self._turn_actions = 0
        self._blocked: set[tuple] = set()  # attackers whose attack we cancelled this turn
        self._attacker: Action | None = None

    # -- card stats --------------------------------------------------------
    @property
    def cards(self) -> Mapping[int, Any]:
        if self._cards is None:
            from ygorl.engine.duel import default_cards

            self._cards = default_cards()
        return self._cards

    def _attack(self, code: int) -> int:
        c = self.cards.get(code)
        return c.attack if c is not None else 0

    def _defense(self, code: int) -> int:
        c = self.cards.get(code)
        return c.defense if c is not None else 0

    # -- entry point -------------------------------------------------------
    def act(self, point: DecisionPoint) -> int:
        if point.turn != self._turn:
            self._turn, self._turn_actions = point.turn, 0
            self._used.clear()
            self._blocked.clear()
        decision = point.decision
        attacker, self._attacker = self._attacker, None
        handler: Callable[[DecisionPoint], int] | None = {
            M.SelectIdleCmd: self._idle,
            M.SelectBattleCmd: self._battle,
            M.SelectChain: self._chain,
            M.SelectEffectYn: self._yes,
            M.SelectYesNo: self._yes,
            M.SelectPosition: self._position,
            M.SelectUnselectCard: self._select,
            M.SelectTribute: self._tribute,
            M.SelectSum: self._select,
        }.get(type(decision))
        if isinstance(decision, M.SelectCard):
            if attacker is not None and _is_attack_target(point):
                return self._target(point, attacker)
            handler = self._select
        if handler is None:
            return self.rng.randrange(len(point.actions))
        return handler(point)

    # -- helpers -----------------------------------------------------------
    def _fresh(self, a: Action) -> bool:
        return self._used[_key(a)] < self.max_repeats

    def _take(self, point: DecisionPoint, a: Action) -> int:
        if a.kind in ("activate", "chain"):
            self._used[_key(a)] += 1
        return point.actions.index(a)

    def _first(self, actions: list[Action], kind: str) -> Action | None:
        for a in actions:
            if a.kind == kind and (a.kind not in ("activate", "chain") or self._fresh(a)):
                return a
        return None

    def _strongest(self, actions: list[Action], kind: str) -> Action | None:
        cands = [a for a in actions if a.kind == kind]
        return max(cands, key=lambda a: self._attack(a.card.code) if a.card else 0, default=None)

    def _fallback(self, point: DecisionPoint) -> int:
        return self.rng.randrange(len(point.actions))

    # -- decisions ---------------------------------------------------------
    def _idle(self, point: DecisionPoint) -> int:
        acts = point.actions
        proactive = self._turn_actions < self.max_turn_actions
        for kind in _IDLE_ORDER:
            if not proactive and kind not in ("battle_phase", "end_phase"):
                continue
            a = self._strongest(acts, kind) if kind in _BY_ATK else self._first(acts, kind)
            if a is not None:
                if kind not in ("battle_phase", "end_phase"):
                    self._turn_actions += 1
                return self._take(point, a)
        return self._fallback(point)

    def _battle(self, point: DecisionPoint) -> int:
        acts = point.actions
        attacks = [a for a in acts if a.kind == "attack" and _key(a) not in self._blocked]
        if attacks:
            a = max(attacks, key=lambda a: (a.value, self._attack(a.card.code)))  # direct first, then ATK
            self._attacker = a
            return self._take(point, a)
        for kind in ("activate", "main2", "end_phase"):
            a = self._first(acts, kind)
            if a is not None:
                return self._take(point, a)
        return self._fallback(point)

    def _target(self, point: DecisionPoint, attacker: Action) -> int:
        acts = point.actions
        power = self._attack(attacker.card.code)

        def stat(a: Action) -> int:
            pos = a.card.loc.position
            return self._attack(a.card.code) if pos & C.POS_ATTACK else self._defense(a.card.code)

        targets = [a for a in acts if a.kind == "select"]
        beaten = [a for a in targets if power > stat(a)]
        if beaten:
            return acts.index(max(beaten, key=stat))
        cancel = next((a for a in acts if a.kind == "cancel"), None)
        if cancel is not None:
            self._blocked.add(_key(attacker))
            return acts.index(cancel)
        if targets:
            return acts.index(min(targets, key=stat))
        return self._fallback(point)

    def _chain(self, point: DecisionPoint) -> int:
        a = self._first(point.actions, "chain") or self._first(point.actions, "pass")
        return self._take(point, a) if a is not None else self._fallback(point)

    def _yes(self, point: DecisionPoint) -> int:
        return next(i for i, a in enumerate(point.actions) if a.kind == "yes")

    def _position(self, point: DecisionPoint) -> int:
        d: M.SelectPosition = point.decision
        prefer = ((C.POS_FACEUP_ATTACK, C.POS_FACEUP_DEFENSE, C.POS_FACEDOWN_DEFENSE, C.POS_FACEDOWN_ATTACK)
                  if self._attack(d.code) >= self._defense(d.code) else
                  (C.POS_FACEUP_DEFENSE, C.POS_FACEDOWN_DEFENSE, C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_ATTACK))  # fmt: skip
        values = [a.value for a in point.actions]
        for pos in prefer:
            if pos in values:
                return values.index(pos)
        return self._fallback(point)

    def _select(self, point: DecisionPoint, key: Callable[[Action], Any] | None = None) -> int:
        acts = point.actions
        selects = [a for a in acts if a.kind == "select"]
        if selects:
            a = min(selects, key=key) if key is not None else self.rng.choice(selects)
            return acts.index(a)
        for kind in ("finish", "cancel"):
            a = self._first(acts, kind)
            if a is not None:
                return acts.index(a)
        return self._fallback(point)

    def _tribute(self, point: DecisionPoint) -> int:
        return self._select(point, key=lambda a: self._attack(a.card.code))


def _is_attack_target(point: DecisionPoint) -> bool:
    return any(isinstance(e, M.Hint) and e.hint_type == C.HINT_SELECTMSG and e.data == HINTMSG_ATTACKTARGET
               for e in point.events)  # fmt: skip
