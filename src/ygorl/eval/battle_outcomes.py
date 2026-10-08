"""Ordinary battle arithmetic as a diagnostic baseline, never a tactical mask."""

from __future__ import annotations

import math


def battle_relation_features(numeric):
    """Six public relations from [aATK,aDEF,tATK,tDEF,attack,defense].

    Stats use the caller's common scale (the probe uses /4000). No outcome,
    identity, or hidden information is consumed. Relations ignore card effects.
    """
    if len(numeric) != 6 or not all(math.isfinite(float(x)) for x in numeric):
        raise ValueError("expected six finite public numeric features")
    atk, _, tatk, tdef, attack, defense = map(float, numeric)
    if attack not in (0, 1) or defense not in (0, 1) or attack + defense > 1:
        raise ValueError("target position flags must be exclusive booleans")
    direct = 1 - attack - defense
    margin = atk - attack * tatk - defense * tdef
    return [margin, max(margin, 0), max(-margin, 0), float(margin == 0), direct, direct * atk]


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


def decompose_battle_outcome(declaration, battle, damage):
    """Post-outcome supervision, NEVER declaration-time model features.

    ``damage`` is [self, opponent] damage after Battle through the local endpoint.
    The native Battle message supplies calculation-time stats and destruction
    flags. Deltas describe observations, not causal effect identities or value.
    Unsupported calculation-time states return None, not a zero residual.
    """

    def row(atk, defense, opponent, position):
        result = [0] * 23
        result[4] = opponent
        result[6] = {1: 1, 4: 3}.get(position, 0)
        result[7] = 1
        result[17:19] = [atk, defense]
        return result

    attacker = row(battle["attacker_atk"], battle["attacker_def"], 0, battle["attacker"]["position"])
    target = None
    if battle["target"]["location"] != 0:
        target = row(battle["target_atk"], battle["target_def"], 1, battle["target"]["position"])
    arithmetic = ordinary_battle(attacker, target)
    if arithmetic is None:
        return None
    destroyed = [int(bool(battle["attacker_destroyed"])), int(bool(battle["target_destroyed"]))]
    return {
        "battle_arithmetic": arithmetic,
        "numeric_damage_delta": [b - a for a, b in zip(declaration["damage"], arithmetic["damage"], strict=True)],
        "non_arithmetic_damage_delta": [b - a for a, b in zip(arithmetic["damage"], damage, strict=True)],
        "destruction_delta": [b - a for a, b in zip(arithmetic["destroy"], destroyed, strict=True)],
    }
