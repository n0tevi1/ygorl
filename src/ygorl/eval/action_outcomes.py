"""Observed action intervals for review, without tactical correctness labels.

Trace rows contain events *before* the selected action. A battle can overlap a
chain: their outcomes are not additive. Missing terminal events remain unknown.
"""

from __future__ import annotations


def action_intervals(rows):
    """Summarize selected effect/attack intervals in an archived decision trace.

    Returns ``(intervals, coverage)``. Card movement counts exclude draws, which are
    reported separately; neither is treated as a reward. A selected action may fail
    or be cancelled: only native closing events make an interval complete.
    """
    active = {}
    result = []
    coverage = {"damage_outside_tracked_intervals": [0, 0], "trace_rows": 0}

    def finish(kind, end, reason):
        interval = active.pop(kind)
        interval.update(end=end, stop=reason, complete=reason in ("ChainEnd", "DamageStepEnd", "Win"))
        result.append(interval)

    last = -1
    for row in rows:
        index = row["index"]
        if index <= last:
            raise ValueError("trace decision indices must increase")
        last = index
        coverage["trace_rows"] += int(row["chosen"] is not None)
        for event in row["events"]:
            name = event["type"]
            if name in ("NewPhase", "NewTurn", "Win"):
                for kind in tuple(active):
                    finish(kind, index, name)
            if name == "Damage" and not active:
                coverage["damage_outside_tracked_intervals"][event["player"]] += event["amount"]
            for interval in active.values():
                if name in ("Damage", "PayLpCost", "Recover"):
                    key = {"Damage": "damage", "PayLpCost": "lp_cost", "Recover": "recover"}[name]
                    interval[key][event["player"]] += event["amount"]
                elif name == "Draw":
                    interval["draws"][event["player"]] += len(event["cards"])
                elif name == "Move":
                    before, after = event["previous"], event["current"]
                    if before["location"] == 2 and (
                        after["location"] != 2 or after["controller"] != before["controller"]
                    ):
                        interval["hand_exits"][before["controller"]] += 1
                    if after["location"] == 2 and (
                        before["location"] != 2 or before["controller"] != after["controller"]
                    ):
                        interval["hand_entries"][after["controller"]] += 1
                elif name == "Battle":
                    interval["battles"].append(event)
                elif name in ("ChainDisabled", "ChainNegated"):
                    interval["negations"].append(event)
            if name == "ChainEnd" and "chain" in active:
                finish("chain", index, name)
            if name == "DamageStepEnd" and "attack" in active:
                finish("attack", index, name)
        action = row["chosen"]
        if action is None:  # explicit final events, without inventing another decision/action
            continue
        kind = "attack" if action["kind"] == "attack" else "chain" if action["kind"] in ("activate", "chain") else None
        if kind is not None:
            if kind == "attack" and kind in active:
                finish(kind, index, "next_attack_without_end")
            if kind not in active:
                active[kind] = {
                    "kind": kind,
                    "start": index,
                    "initiator": row["player"],
                    "action": action,
                    "damage": [0, 0],
                    "lp_cost": [0, 0],
                    "recover": [0, 0],
                    "hand_entries": [0, 0],
                    "hand_exits": [0, 0],
                    "draws": [0, 0],
                    "battles": [],
                    "negations": [],
                    "responses": [],
                }
            active[kind]["responses"].append({"decision": index, "player": row["player"], "action": action})
    for kind in tuple(active):
        finish(kind, last, "trace_end_unobserved_terminal")
    return sorted(result, key=lambda x: (x["start"], x["kind"])), coverage
