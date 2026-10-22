"""Observe-only shadow duel next to the Solo player (design 10, 迁移步骤 3): does ygopro ask what MD asks?

When the player commits to a policy action, a :class:`~ygorl.selfplay.shadow.ShadowDuel` is built from the same
snapshot and the same action is applied to it. At MD's next request for the client, the request
(:mod:`ygorl.md.request`) is aligned with the shadow's next decision (:func:`ygorl.selfplay.bridge.align`) and the
result goes to the trace. Nothing here changes what the player does; any error is recorded, never raised.

v1 compares the first question after each policy action. Lockstep through every later pick (feeding the shadow the
player's own choices) comes when the decisions move to the shadow.
"""

from __future__ import annotations

import random
import traceback
from collections.abc import Mapping

from ygorl.md.request import Request as MDRequest
from ygorl.selfplay.bridge import Option, Request, align, location_name
from ygorl.selfplay.shadow import ShadowDuel

# MD request kind -> bridge kind. Mode 7 is both the position dialog and the zone pick (after a Tribute Summon);
# the dialog type tells them apart (10: the position dialog with its buttons).
POSITION_DIALOG = 10


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
    options = [Option(password=passwords.get(c.cid, 0), location=c.location, sequence=c.sequence,
                      controller=0 if c.player == me else 1) for c in req.candidates if c.cid]  # fmt: skip
    return Request(kind=kind, options=options, min=req.min or 1, max=req.max or 1)


def describe(point) -> dict:
    if point is None:
        return {"decision": None}
    return {"decision": type(point.decision).__name__, "player": point.player,
            "actions": [[a.kind, a.card.code if a.card else 0, location_name(a.card.loc.location) if a.card else ""]
                        for a in point.actions[:12]]}  # fmt: skip


class ShadowObserver:
    def __init__(self, rule_flags: int, cards, prior) -> None:
        self.rule_flags, self.cards, self.prior = rule_flags, cards, prior
        self.shadow: ShadowDuel | None = None
        self.after: str = ""  # the policy action the pending comparison follows
        self.auto: list = []

    def decided(self, action, snapshot, own_deck) -> dict:
        """The player commits to ``action`` on ``snapshot``: build the shadow and apply it."""
        self.shadow, self.after, self.auto = None, action.kind, []
        try:
            shadow = ShadowDuel.from_snapshot(snapshot, own_deck, self.prior, self.rule_flags, cards=self.cards,
                                              rng=random.randrange(1 << 30))  # fmt: skip
            point = shadow.point
            index = next((i for i, a in enumerate(point.actions) if _same(a, action)), None)
            if index is None:
                return {"shadow": "the shadow does not offer the policy's action", "point": describe(point)}
            self.auto = [_auto(s) for s in shadow.step(index)]
            self.shadow = shadow
            return {"shadow": "ok", "auto": self.auto}
        except Exception as exc:  # observe only: never break the player
            return {"shadow_error": f"{type(exc).__name__}: {exc}", "where": traceback.format_exc(limit=3)}

    def request(self, req: MDRequest, me: int, passwords: Mapping[int, int]) -> dict | None:
        """MD's first request after the action: align it with the shadow's next decision (once)."""
        if self.shadow is None:
            return None
        shadow, self.shadow = self.shadow, None
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


def _same(a, b) -> bool:
    if a.kind != b.kind or (a.card is None) != (b.card is None):
        return False
    if a.card is None:
        return True
    return (
        a.card.code == b.card.code
        and a.card.loc.location == b.card.loc.location
        and a.card.loc.sequence == b.card.loc.sequence
    )


def _auto(step) -> dict:
    return {
        "player": step.player,
        "decision": step.decision,
        "reason": step.reason,
        "could_respond": step.could_respond,
    }


__all__ = ["ShadowObserver", "to_bridge"]
