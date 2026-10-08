"""Observational warning signals; neither tactical labels nor reward shaping."""

import json
import math
import statistics
from collections import Counter

from ygorl.eval.action_outcomes import action_intervals


def length_stats(values):
    xs = sorted(values)
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p90": xs[math.ceil(0.9 * len(xs)) - 1],
        "p99": xs[math.ceil(0.99 * len(xs)) - 1],
        "max": xs[-1],
        "le4": sum(x <= 4 for x in xs),
        "gt10": sum(x > 10 for x in xs),
        "gt20": sum(x > 20 for x in xs),
    }


def summarize_trace(rows, candidate):
    counts = Counter()
    repeats, findings, chain, turn_actions = Counter(), [], {}, Counter()
    idle_segment = 0
    for row in rows:
        events = row["events"]
        for e in events:
            if e["type"] == "Chaining":
                chain = {k: v for k, v in chain.items() if k < e["chain_count"]}
                chain[e["chain_count"]] = e
            elif e["type"] == "ChainEnd":
                chain.clear()
        if events:  # tracker.events excludes hint/wait/decision messages
            repeats.clear()
            idle_segment = 0
        action = row["chosen"]
        if action is None:
            continue
        idle_segment += 1
        counts["max_no_event_decisions"] = max(counts["max_no_event_decisions"], idle_segment)
        key = json.dumps(
            [row["turn"], row["phase"], row["player"], row["lp"], row["board"], row["options"], action], sort_keys=True
        )
        repeats[key] += 1
        counts["max_same_choice_without_event"] = max(counts["max_same_choice_without_event"], repeats[key])
        if repeats[key] == 8:
            findings.append(
                {
                    "kind": "no_event_repeat",
                    "decision": row["index"],
                    "player": row["player"],
                    "action": action,
                    "repetitions": 8,
                }
            )
        if row["player"] != candidate:
            continue
        counts["candidate_decisions"] += 1
        counts["undo_chosen"] += int(row["choice"] in row["undo"])
        counts["cancel_chosen"] += int(action["kind"] == "cancel")
        turn_actions[f"{row['turn']}:{action['kind']}"] += 1
        legal = [a for i, a in enumerate(row["options"]) if row["mask"][i]]
        ash = [
            i
            for i, a in enumerate(row["options"])
            if row["mask"][i] and a["kind"] == "chain" and a["card"] and a["card"]["code"] == 14558127
        ]
        if ash:
            top = chain[max(chain)] if chain else None
            owner = "unknown" if top is None else "self" if top["triggering_controller"] == candidate else "opponent"
            counts[f"ash_{owner}_opportunities"] += 1
            counts[f"ash_{owner}_chosen"] += int(row["choice"] in ash)
            if owner != "opponent":
                findings.append(
                    {
                        "kind": "ash_" + owner,
                        "decision": row["index"],
                        "lp": row["lp"],
                        "chosen": row["choice"] in ash,
                        "probability": sum(row["probs"][i] for i in ash),
                        "chain": list(chain.values()),
                    }
                )
        if action["kind"] in ("main2", "end_phase") and any(a["kind"] == "attack" for a in legal):
            counts["end_battle_with_attack_available"] += 1
        if action["kind"] in ("battle_phase", "end_phase") and any(
            a["kind"] in ("summon", "spsummon", "activate") for a in legal
        ):
            counts["leave_main_with_proactive_option"] += 1
    intervals, coverage = action_intervals(rows)
    for interval in intervals:
        if interval["initiator"] != candidate:
            continue
        counts[interval["kind"] + "_intervals"] += 1
        if not interval["complete"]:
            counts[interval["kind"] + "_unobserved_endpoint"] += 1
        elif interval["damage"][candidate]:
            findings.append(
                {"kind": interval["kind"] + "_self_damage", "decision": interval["start"], "interval": interval}
            )
    return {"counts": dict(counts), "findings": findings, "turn_actions": dict(turn_actions), "coverage": coverage}
