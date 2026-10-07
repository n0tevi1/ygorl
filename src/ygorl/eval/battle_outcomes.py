"""Ordinary battle arithmetic as a diagnostic baseline, never a tactical mask."""

from __future__ import annotations


def ordinary_battle(attacker, target=None):
    """Return damage [self, opponent] and destruction flags, or None if unsupported.

    Inputs are public encoder card rows. This baseline deliberately omits all card
    text/effects (piercing, immunity, damage changes, battle-position exceptions).
    Actual native outcomes must remain the teacher when assessing those effects.
    """
    if len(attacker) != 23 or (target is not None and len(target) != 23):
        raise ValueError("expected encoded card rows of width 23")
    if attacker[7] != 1 or attacker[4] != 0 or attacker[6] != 1:
        return None
    if target is not None and (target[7] != 1 or target[4] != 1 or target[6] not in (1, 3)):
        return None
    rows = [attacker] + ([] if target is None else [target])
    if any(not 0 <= r[i] < 65535 for r in rows for i in (17, 18)):
        return None  # clipped/unsupported values cannot establish the true comparison
    atk = int(attacker[17])
    if target is None:
        return {"damage": [0, atk], "destroy": [0, 0]}
    value = int(target[17] if target[6] == 1 else target[18])
    diff = atk - value
    if target[6] == 3:
        return {"damage": [max(-diff, 0), 0], "destroy": [0, int(diff > 0)]}
    return {
        "damage": [max(-diff, 0), max(diff, 0)],
        "destroy": [int(diff < 0 or (diff == 0 and atk > 0)), int(diff > 0 or (diff == 0 and atk > 0))],
    }
