"""Read-only combat diagnostics; never registered as a training/evaluation opponent.

A forced attack must seed Greedy's next-target context. Public-current-stat mode
is an optional sensitivity probe, not a complete combat/effect solver.
"""

from dataclasses import replace

from ygorl.agents.greedy import HIDDEN_STAT, GreedyAgent, _key
from ygorl.engine import constants as C


class CombatProbe(GreedyAgent):
    def __init__(self, seed, *, cards, board=None):
        super().__init__(seed, cards=cards)
        self.board = board

    def prime(self, point, index):
        """Initialize a fresh chooser after the driver forces the first action."""
        if self._turn is not None:
            raise ValueError("prime requires a fresh probe")
        action = point.actions[index]
        if action.kind not in ("attack", "battle_phase"):
            raise ValueError("expected forced attack or battle entry")
        self._turn = point.turn
        self._attacker = action if action.kind == "attack" else None

    def _visible(self, action):
        if self.board is None or action.card is None:
            return None
        loc = action.card.loc
        if loc.location != C.LOCATION_MZONE:
            return None
        row = self.board()[loc.controller]
        if not 0 <= loc.sequence < len(row):
            return None
        card = row[loc.sequence]
        # Even a mistakenly unfiltered provider must not disclose a facedown card.
        if not card or not card.get("position", 0) & C.POS_FACEUP:
            return None
        if card.get("code") != action.card.code:
            return None
        return card

    def _power(self, action):
        card = self._visible(action)
        return card["attack"] if card is not None else self._attack(action.card.code)

    def _battle(self, point):
        if self.board is None:
            return super()._battle(point)
        attacks = [a for a in point.actions if a.kind == "attack" and _key(a) not in self._blocked]
        if not attacks:
            return super()._battle(point)
        action = max(attacks, key=lambda a: (a.value, self._power(a)))
        self._attacker = action
        return self._take(point, action)

    def _target(self, point, attacker):
        if self.board is None:
            return super()._target(point, attacker)
        power = self._power(attacker)

        def stat(action):
            if not action.card.code or not action.card.loc.position & C.POS_FACEUP:
                return HIDDEN_STAT
            card = self._visible(action)
            if card is not None:
                return card["attack"] if card["position"] & C.POS_ATTACK else card["defense"]
            return self._attack(action.card.code) if action.card.loc.position & C.POS_ATTACK \
                else self._defense(action.card.code)  # fmt: skip

        targets = [a for a in point.actions if a.kind == "select"]
        beaten = [a for a in targets if power > stat(a)]
        if beaten:
            return point.actions.index(max(beaten, key=stat))
        cancel = next((a for a in point.actions if a.kind == "cancel"), None)
        if cancel is not None:
            self._blocked.add(_key(attacker))
            return point.actions.index(cancel)
        return point.actions.index(min(targets, key=stat)) if targets else self._fallback(point)


def masked_action(chooser, point, mask):
    """Choose among policy-supported rows and return the original engine index.

    The encoded mask can be shorter than the engine's candidate list. Filter
    before acting so rejected guesses do not mutate chooser context or RNG.
    """
    indices = [i for i in range(min(len(mask), len(point.actions))) if mask[i]]
    if not indices:
        raise ValueError("combat probe has no policy-supported action")
    filtered = replace(point, actions=[point.actions[i] for i in indices])
    return indices[chooser.act(filtered)]
