"""Opt-in inference experiment: bound canceled optional summon reentry.

This is a policy restriction, not an engine legality rule. An abandoned summon
may have a valid alternative material choice. Never enable by default or infer
engine-state equality from equal model observations alone.
"""

from collections import Counter
from hashlib import sha256

import numpy as np

from ygorl.engine import messages as M


def _fingerprint(observation):
    digest = sha256()
    for key, value in sorted(observation.items()):
        array = np.asarray(value)
        digest.update(repr((key, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.digest()


class SummonReentryGuard:
    """Allow ``budget`` aborted attempts at an unchanged idle menu.

    Call observe() for EVERY player's point, restrict() only for the candidate,
    and on_decision() for EVERY chosen action. Create a fresh guard per duel.
    Only idle -> SelectUnselectCard -> cancel -> identical idle is recognized.
    Any other event/decision/action, player, turn or phase invalidates history.
    Even this narrow no-event cycle is a heuristic, not a proof of uselessness.
    """

    def __init__(self, budget=1):
        if type(budget) is not int or budget < 1:
            raise ValueError("at least one summon attempt must remain")
        self.budget = budget
        self.reset()

    def reset(self):
        self.context = None
        self.pending = None
        self.canceled = False
        self.counts = Counter()
        self.root = None
        self.observed_index = None
        self.restricted_index = None

    def observe(self, point):
        context = (point.player, point.turn, point.phase, point.lp)
        # Deliberately narrower than tracker._NOT_EVENTS: retries, reveals,
        # chains and unfamiliar messages must not silently count as no progress.
        if (
            context != self.context
            or not isinstance(point.decision, (M.SelectIdleCmd, M.SelectUnselectCard))
            or any(not isinstance(e, (M.SelectIdleCmd, M.SelectUnselectCard, M.Hint)) for e in point.events)
        ):
            self.reset()
        self.context = context
        self.observed_index = point.index
        self.restricted_index = None

    def restrict(self, point, observation):
        if self.observed_index != point.index:
            raise ValueError("observe every point before restricting")
        if self.restricted_index == point.index:
            raise ValueError("restrict only once per point")
        self.restricted_index = point.index
        mask = observation["action_mask"]
        if not np.any(mask):
            raise ValueError("empty baseline action mask")
        info = {"aborted_attempt": 0, "blocked": [], "sole_exit_preserved": 0}
        if not isinstance(point.decision, M.SelectIdleCmd):
            return observation, info
        root = _fingerprint(observation)
        if root != self.root:
            self.counts.clear()
        if self.pending and self.canceled and self.pending[0] == root:
            self.counts[self.pending[1]] += 1
            info["aborted_attempt"] = 1
        self.pending = None
        self.canceled = False
        self.root = root
        indices = [
            i
            for i, action in enumerate(point.actions)
            if i < len(mask) and mask[i] and action.kind == "spsummon" and self.counts[action] >= self.budget
        ]
        if not indices:
            return observation, info
        changed = mask.copy()
        changed[indices] = False
        if not np.any(changed):
            info["sole_exit_preserved"] = 1
            return observation, info
        info["blocked"] = indices
        return {**observation, "action_mask": changed}, info

    def on_decision(self, point, index):
        if self.observed_index != point.index:
            raise ValueError("observe every point before acting")
        action = point.actions[index]
        if isinstance(point.decision, M.SelectIdleCmd):
            if action.kind == "spsummon" and self.restricted_index == point.index:
                self.pending = (self.root, action)
                self.canceled = False
                return
        elif isinstance(point.decision, M.SelectUnselectCard) and self.pending:
            if not self.canceled and action.kind in ("select", "unselect", "cancel"):
                self.canceled = action.kind == "cancel"
                return
        self.reset()
