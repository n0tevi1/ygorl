"""The ygopro shadow duel next to the Solo player (design 10, 迁移步骤 3 and 「边角情况由 player 自己学会」).

When the player commits to a policy action, a :class:`~ygorl.selfplay.shadow.ShadowDuel` is built from the same
snapshot and the same action is applied to it. From then on, until the next command point, the shadow is kept in
lockstep with MD:

- :meth:`ShadowObserver.request` compares MD's accepting requests (:mod:`ygorl.md.request`) with the shadow's next
  decision (:func:`ygorl.selfplay.bridge.align`) and records the result (observe only);
- :meth:`ShadowObserver.choose` answers MD's own dialogs inside the action (effect choice, card list, yes / no,
  radio rows, position, activate-or-pass): the dialog's options are paired with the shadow's decision, the policy
  scores the shadow's decision point, the chosen ygopro action is mapped back to MD's option and the shadow is
  stepped with it. Most of these dialogs come while MD's engine input is closed (mode 0), which is why the request
  comparison alone never saw them.

The mapping tolerates what MD is known to do differently (design 10 「识别与执行的系统化」, lockstep audit):

- MD asks a Summon's position before its zone (ygopro: zone, then position), and the cost / target of a Spell before
  its zone: a shadow waiting at its own zone pick takes the zone the player will click (``zone``) or the first
  free one, then answers MD's question;
- MD asks the effect of a card with several effects in a dialog after the card is clicked, where ygopro offers one
  activate action per effect: the policy's activate action already chose it (its description names the tab);
- MD asks a list ygopro did not ask (one candidate, or copies the shadow filled in by itself): the shadow's own
  answer (:attr:`ShadowObserver.pending`) gives the pick and the shadow stays;
- an optional trigger ygopro asks as yes / no (``SelectEffectYn``) is MD's activate-or-pass prompt, or MD merges it
  into the list of the cards it would use (Ra's Disciple: a list of Disciples, cancel = no): the list means yes.

Any other dialog that does not match the shadow's decision drops the shadow (recorded; the caller falls back to a
rebuild from the current board (:mod:`ygorl.selfplay.decide`) or its rule, and the shadow is rebuilt at the next
command point). Errors are recorded, never raised.
"""

from __future__ import annotations

import random
import traceback
from collections.abc import Mapping

import numpy as np

from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.md.request import Request as MDRequest
from ygorl.selfplay.bridge import DECISIONS, Option, Request, align, find, location_name
from ygorl.selfplay.shadow import ShadowDuel

# MD request kind -> bridge kind. Mode 7 is both the position dialog and the zone pick (after a Tribute Summon);
# the dialog type tells them apart (10: the position dialog with its buttons).
POSITION_DIALOG = 10
DECLINE = ("pass", "no", "cancel", "finish")
SHUFFLED = ("deck", "extra")  # piles whose order the shadow does not share with MD (it shuffles them)

# MD dialog (as the player names it) -> the shadow decisions that ask the same question
DIALOG_DECISIONS: dict[str, tuple] = {
    "list": DECISIONS["select_card"],
    "option": (M.SelectOption,),
    "radio": (M.SelectOption, M.AnnounceNumber),
    "yesno": DECISIONS["yesno"],
    "prompt": (M.SelectChain, M.SelectEffectYn),
    "position": (M.SelectPosition,),
}
# the position dialog's buttons, left to right
POSITION_BUTTONS = (C.POS_FACEUP_ATTACK, C.POS_FACEUP_DEFENSE)


def md_option(c, me: int, passwords: Mapping[int, int]) -> Option:
    """MD candidate ``c`` as a bridge option. The shadow shuffles the cards left in the Deck and Extra Deck (their
    order is hidden), so a candidate there is any copy of its card: MD's pile index names no shadow card (an audit
    found Foolish Burial's Deck list paired with 0 or 1 shadow option out of 14)."""
    sequence = None if c.location in SHUFFLED else c.sequence
    return Option(password=passwords.get(c.cid, 0), location=c.location, sequence=sequence,
                  controller=0 if c.player == me else 1)  # fmt: skip


def to_bridge(req: MDRequest, me: int, passwords: Mapping[int, int]) -> Request | None:
    """MD's request in the bridge's terms (None for kinds the bridge has no counterpart for: draw, summon_pick)."""
    if req.kind in ("idle", "battle"):
        kind = req.kind
    elif req.kind in ("open", "chain"):
        kind = "chain"
    elif req.kind == "select_card":
        kind = "select_card"
    elif req.kind == "position":
        kind = "position" if req.dialog_type == POSITION_DIALOG else "place"
    else:
        return None
    options = [md_option(c, me, passwords) for c in req.candidates if c.cid]
    return Request(kind=kind, options=options, min=req.min or 1, max=req.max or 1)


def describe(point) -> dict:
    if point is None:
        return {"decision": None}
    return {"decision": type(point.decision).__name__, "player": point.player,
            "actions": [[a.kind, a.card.code if a.card else 0, location_name(a.card.loc.location) if a.card else ""]
                        for a in point.actions[:12]]}  # fmt: skip


def md_option_desc(choice: int, passwords: Mapping[int, int]) -> int | None:
    """A dialog choice (enabled << 31 | default << 30 | text id) as a ygopro option description: card strings are
    MD card id | (10 + index) << 16 -> ``aux.Stringid(password, index)`` = password << 20 | index (Project Ignis'
    utility.lua); system strings keep their number (not verified to agree with ygopro's, so they only pair by
    order)."""
    text = choice & 0x3FFFFFFF
    index, cid = text >> 16, text & 0xFFFF
    if index >= 10 and cid in passwords:
        return passwords[cid] << 20 | (index - 10)
    return None


def option_of_description(choices, description: int, passwords: Mapping[int, int]) -> int | None:
    """The tab of MD's effect-choice dialog whose card string is ``description`` (an activate action's), if one."""
    if not description:
        return None
    tabs = [i for i, c in enumerate(choices) if md_option_desc(c, passwords) == description]
    return tabs[0] if len(tabs) == 1 else None


def option_pairs(md: list[int | None], ygo: list[int]) -> dict[int, int] | None:
    """MD option index -> ygopro option index: by description where both name one, else by order (equal counts)."""
    pairs = {i: ygo.index(d) for i, d in enumerate(md) if d is not None and d in ygo}
    if len(pairs) == len(md):
        return pairs
    if len(md) == len(ygo):
        return dict(enumerate(range(len(ygo))))
    return pairs or None


class ShadowObserver:
    def __init__(self, rule_flags: int, cards, prior, scorer=None) -> None:
        """``scorer(shadow, point) -> logits`` scores a shadow decision point (the policy); None: the first option."""
        self.rule_flags, self.cards, self.prior, self.scorer = rule_flags, cards, prior, scorer
        self.shadow: ShadowDuel | None = None
        self.after: str = ""  # the policy action the pending comparison follows
        self.auto: list = []
        self.activated = 0  # the description of the activate action the shadow took (the effect MD asks a tab for)
        self.pending: list[dict] = []  # the advisor's own auto answers MD may still ask (one candidate, copies)

    def decided(self, action, snapshot, own_deck) -> dict:
        """The player commits to ``action`` on ``snapshot``: build the shadow and apply it."""
        self.shadow, self.after, self.auto, self.pending = None, action.kind, [], []
        self.activated = action.description if action.kind == "activate" else 0
        try:
            shadow = ShadowDuel.from_snapshot(snapshot, own_deck, self.prior, self.rule_flags, cards=self.cards,
                                              rng=random.randrange(1 << 30))  # fmt: skip
            point = shadow.point
            index = next((i for i, a in enumerate(point.actions) if _same(a, action)), None)
            if index is None:
                return {"shadow": "the shadow does not offer the policy's action", "point": describe(point)}
            self._log(shadow.step(index))
            self.shadow = shadow
            return {"shadow": "ok", "auto": self.auto}
        except Exception as exc:  # observe only: never break the player
            return {"shadow_error": f"{type(exc).__name__}: {exc}", "where": traceback.format_exc(limit=3)}

    def adopt(self, shadow: ShadowDuel, after: str) -> None:
        """Continue in lockstep with ``shadow``, a fresh rebuild already answered up to MD's current question (the
        attack target: :func:`ygorl.selfplay.decide.attack_target`)."""
        self.shadow, self.after, self.auto, self.pending, self.activated = shadow, after, [], [], 0

    def _log(self, steps) -> None:
        for s in steps:
            self.auto.append(_auto(s))
            if s.player == 0 and s.reason in ("single", "equivalent") and s.action.card is not None:
                self.pending.append({"card": s.action.card.code, "location": location_name(s.action.card.loc.location),
                                     "decision": s.decision})  # fmt: skip

    def placed(self, location: str, sequence: int) -> dict | None:
        """The player picked zone ``sequence`` of ``location`` as part of its action: the shadow picks it too
        (its own SelectPlace), so the next comparison is with the question after the zone."""
        shadow = self.shadow
        if shadow is None or shadow.point is None or type(shadow.point.decision).__name__ != "SelectPlace":
            return None
        try:
            from ygorl.engine import constants as C

            loc = {"mzone": C.LOCATION_MZONE, "szone": C.LOCATION_SZONE}[location]
            want = loc << 8 | sequence  # a place action's value: controller << 16 | location << 8 | sequence
            index = next((i for i, a in enumerate(shadow.point.actions)
                          if a.kind == "place" and a.value & 0xFFFF == want and a.value >> 16 == 0), None)  # fmt: skip
            if index is None:
                return {"placed": f"the shadow has no {location} {sequence}"}
            self._log(shadow.step(index))
            return {"placed": f"{location} {sequence}"}
        except Exception as exc:
            self.shadow = None
            return {"shadow_error": f"{type(exc).__name__}: {exc}"}

    def request(self, req: MDRequest, me: int, passwords: Mapping[int, int]) -> dict | None:
        """An accepting MD request inside the action: align it with the shadow's next decision. A command point ends
        the action (the shadow is dropped and rebuilt with the next policy action)."""
        if self.shadow is None:
            return None
        shadow = self.shadow
        if req.kind in ("idle", "battle", "draw") or shadow.done:
            self.shadow = None
        try:
            ask = to_bridge(req, me, passwords)
            point = shadow.point
            out = {"after": self.after, "md": req.kind, "md_candidates": len(req.candidates), "ygo": describe(point),
                   "auto": self.auto, "shadow_stopped": shadow.stopped or ""}  # fmt: skip
            if ask is None or point is None:
                return out
            if ask.kind == "chain" and type(point.decision).__name__ == "SelectEffectYn":
                ask = Request(kind="yesno", options=[], min=1, max=1)  # an optional trigger: MD asks it as a prompt
            a = align(ask, point)
            # command points: MD keeps no option list in memory, so only the kind can be compared
            consistent = a.kind_match if ask.kind in ("idle", "battle", "yesno") else a.consistent
            out.update(kind_match=a.kind_match, pairs=len(a.pairs), md_only=len(a.md_only), ygo_only=len(a.ygo_only),
                       consistent=consistent)  # fmt: skip
            return out
        except Exception as exc:
            return {"shadow_error": f"{type(exc).__name__}: {exc}", "where": traceback.format_exc(limit=3)}

    def choose(self, dialog: str, req: MDRequest | None, me: int, passwords: Mapping[int, int], *, count: int = 0,
               zone: tuple[str, int] | None = None, exclude: frozenset = frozenset()) -> tuple[int | None, dict]:  # fmt: skip
        """The option of MD's ``dialog`` the policy picks through the shadow, and a trace record; (None, record)
        when there is no shadow or it does not match (then the shadow is dropped).

        Options are numbered as the player clicks them: "list" the list candidates in order, "option" the dialog's
        tabs, "radio" the rows top down (``count`` of them), "yesno" 0 yes / 1 no, "prompt" 0 activate the offered
        card / 1 pass, "position" the buttons left to right (:data:`POSITION_BUTTONS`). ``zone``: the zone the
        player will pick for the card in question (MD asks it after the position / the cost). ``exclude``: MD
        options its screen does not show (a list's candidate MD never displays), not picked."""
        shadow = self.shadow
        if shadow is None or shadow.point is None:
            return None, {}
        point = shadow.point
        record = {"lockstep": dialog, "ygo": type(point.decision).__name__}
        try:
            if isinstance(point.decision, M.SelectPlace) and point.player == 0 and dialog in DIALOG_DECISIONS:
                placed = self._place_first(point, zone)  # MD asks this before the zone
                if placed is None:
                    self.shadow = None
                    return None, {**record, "desync": "the shadow's zone pick has no free zone"}
                record["placed_first"] = placed
                point = shadow.point
                if point is None:
                    self.shadow = None
                    return None, {**record, "desync": f"the shadow stopped ({shadow.stopped})"}
                record["ygo"] = type(point.decision).__name__
            if dialog == "list" and isinstance(point.decision, M.SelectEffectYn) and point.player == 0:
                # MD merges an optional trigger's "activate?" into the list of its cards (cancel = no): the list
                # means the trigger is used; the policy's own preference is recorded
                yes = next((i for i in point.choosable if point.actions[i].kind == "yes"), None)
                if yes is not None:
                    if self.scorer is not None:
                        logits = self.scorer(shadow, point)
                        kept = [i for i in point.choosable if i < len(logits)]
                        z = np.exp(np.array([logits[i] for i in kept]) - max(logits[i] for i in kept))
                        record["policy_yes"] = round(float(z[kept.index(yes)] / z.sum()), 4) if yes in kept else None
                    self._log(shadow.step(yes))
                    point = shadow.point
                    if point is None:
                        self.shadow = None
                        return None, {**record, "desync": f"the shadow stopped ({shadow.stopped})"}
                    record["ygo"] = type(point.decision).__name__
            if not isinstance(point.decision, DIALOG_DECISIONS.get(dialog, ())):
                pick, how = self._tolerated(dialog, req, passwords)
                if pick is not None:  # MD asks what the shadow answered already: the shadow stays
                    return pick, {**record, "pick": pick, "by": how}
                self.shadow = None
                return None, {**record, "desync": "the shadow asks something else"}
            pairs = {m: a for m, a in self._pairs(dialog, req, point, me, passwords, count).items() if m not in exclude}
            if not pairs:
                pick, how = self._tolerated(dialog, req, passwords)
                if pick is not None:
                    return pick, {**record, "pick": pick, "by": how}
                self.shadow = None
                return None, {**record, "desync": "no MD option matches a shadow option"}
            logits = self.scorer(shadow, point) if self.scorer is not None else None
            md, act = max(pairs.items(), key=lambda kv: (logits[kv[1]] if logits is not None else 0.0, -kv[0]))
            action = point.actions[act]
            self.pending = []  # MD asked on: what the shadow filled in before is not asked any more
            self._log(shadow.step(act))
            return md, {**record, "pick": md, "options": len(pairs), "action": [action.kind, action.card.code if
                        action.card else 0, action.value]}  # fmt: skip
        except Exception as exc:
            self.shadow = None
            return None, {**record, "shadow_error": f"{type(exc).__name__}: {exc}"}

    def _place_first(self, point, zone: tuple[str, int] | None) -> str | None:
        """Step the shadow's own zone pick: ``zone`` when it offers it, else its first free zone."""
        acts = [i for i in point.choosable if point.actions[i].kind == "place" and point.actions[i].value >> 16 == 0]
        if not acts:
            return None
        index = acts[0]
        if zone is not None and zone[0] in ("mzone", "szone"):
            loc = {"mzone": C.LOCATION_MZONE, "szone": C.LOCATION_SZONE}[zone[0]]
            index = next((i for i in acts if point.actions[i].value & 0xFFFF == loc << 8 | zone[1]), index)
        value = point.actions[index].value
        self._log(self.shadow.step(index))
        return f"{location_name(value >> 8 & 0xFF)} {value & 0xFF}"

    def _tolerated(self, dialog: str, req, passwords) -> tuple[int | None, str]:
        """MD's option for a question the shadow does not ask (any more), when the shadow's history answers it."""
        if dialog == "option" and req is not None:
            tab = option_of_description(req.choices, self.activated, passwords)
            if tab is not None:
                self.activated = 0
                return tab, "activate"  # the policy's activate action chose the effect
        if dialog == "list" and req is not None and req.candidates:
            for k, done in enumerate(self.pending):
                hit = next((i for i, c in enumerate(req.candidates) if passwords.get(c.cid) == done["card"]
                            and c.location == done["location"]), None)  # fmt: skip
                if hit is not None:
                    del self.pending[: k + 1]
                    return hit, "auto"  # the shadow filled this in (one candidate, or copies of one card)
            if len({passwords.get(c.cid) for c in req.candidates}) == 1:
                return 0, "forced"  # one card (or copies of it): nothing to decide
        return None, ""

    def _pairs(self, dialog, req, point, me, passwords, count) -> dict[int, int]:
        """MD option index -> shadow action index."""
        acts = point.actions
        choosable = list(point.choosable)
        if dialog == "list":
            pairs = {}
            for i, c in enumerate(req.candidates if req is not None else ()):
                a = find(point, md_option(c, me, passwords))
                if a is not None and a not in pairs.values():
                    pairs[i] = a
            return pairs
        if dialog in ("option", "radio"):
            ygo = [i for i in choosable if acts[i].kind in ("option", "number")]
            if dialog == "option":
                md = [md_option_desc(c, passwords) for c in (req.choices if req is not None else ())]
            else:
                md = [None] * count
            got = option_pairs(md, [acts[i].description if acts[i].kind == "option" else None for i in ygo])
            return {m: ygo[y] for m, y in (got or {}).items()}
        if dialog == "yesno":
            kinds = {acts[i].kind: i for i in choosable}
            return {m: kinds[k] for m, k in ((0, "yes"), (1, "no")) if k in kinds}
        if dialog == "prompt":
            out = {}
            offered = req.candidates[0] if req is not None and req.candidates else None
            if isinstance(point.decision, M.SelectEffectYn):  # an optional trigger: activate = yes
                if offered is not None and passwords.get(offered.cid) != point.decision.card.code:
                    return {}
                kinds = {acts[i].kind: i for i in choosable}
                return {m: kinds[k] for m, k in ((0, "yes"), (1, "no")) if k in kinds}
            for i in choosable:
                a = acts[i]
                if a.kind in DECLINE:
                    out.setdefault(1, i)
                elif a.card is not None and (offered is None or passwords.get(offered.cid) == a.card.code):
                    out.setdefault(0, i)
            if offered is not None and 0 not in out:
                return {}  # MD offers a card the shadow cannot activate here: its "pass" is not about this card
            return out
        if dialog == "position":
            values = {acts[i].value: i for i in choosable}
            return {m: values[p] for m, p in enumerate(POSITION_BUTTONS) if p in values}
        return {}


def _same(a, b) -> bool:
    if a.kind != b.kind or (a.card is None) != (b.card is None):
        return False
    if a.card is None:
        return True
    # an activation is also its effect: Millennium Revelation's two activations differ only by description, and
    # the shadow stepped the other one than the policy chose
    return (
        a.card.code == b.card.code
        and a.card.loc.location == b.card.loc.location
        and a.card.loc.sequence == b.card.loc.sequence
        and (a.kind != "activate" or getattr(a, "description", 0) == getattr(b, "description", 0))
    )


def _auto(step) -> dict:
    return {
        "player": step.player,
        "decision": step.decision,
        "reason": step.reason,
        "could_respond": step.could_respond,
    }


__all__ = ["DIALOG_DECISIONS", "POSITION_BUTTONS", "SHUFFLED", "ShadowObserver", "md_option", "md_option_desc", "option_of_description",
           "option_pairs", "to_bridge"]  # fmt: skip
