"""Replay natural action prefixes and explicitly bounded decline alternatives.

Observed original continuations retain recorded choices. Alternatives use passive
responses and stop at the next idle/battle command. These are not solved values.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mine_action_outcomes import read, sha
from ygorl import _core
from ygorl.agents.checkpoint import CheckpointAgent
from ygorl.data import load_environment
from ygorl.engine.duel import DuelSession
from ygorl.engine.replay import Replay
from ygorl.nets.batch import to_tensors
from ygorl.train.registration import atomic_json

HEALTH = ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")


def events_of(events):
    # Archived JSON represents tuple-valued native event fields as lists.
    return json.loads(json.dumps([{"type": type(e).__name__, **asdict(e)} for e in events]))


def branch(root, panel, source, case, route):
    base = panel / "games" / f"{case['game']:04d}"
    with gzip.open(base / "trace.jsonl.gz", "rt") as f:
        trace = [json.loads(line) for line in f]
    replay = Replay.load(base / "replay.json.gz")
    duel = replay.duel(load_environment("md-2026-09"))
    mirror = CheckpointAgent(source, 0)
    mirror.on_duel_start(duel)
    session = DuelSession(duel)
    target, events, choices, stop, initial_lp = None, [], [], None, None
    try:
        while (p := session.point) is not None:
            assert mirror.host.player() == p.player and len(mirror.host.actions()) == len(p.actions)
            if p.index > case["start"]:
                events.extend(events_of(p.events))
            if p.index < case["start"]:
                old = trace[p.index]
                idx = old["choice"]
                assert p.player == old["player"] and asdict(p.actions[idx]) == old["chosen"]
                assert events_of(p.events) == old["events"]
            else:
                obs = mirror.host.observe()
                legal = np.flatnonzero(obs["action_mask"]).tolist()
                if p.index == case["start"]:
                    assert p.player == case["learner"]
                    old = trace[p.index]
                    assert events_of(p.events) == old["events"]
                    original = old["choice"]
                    assert asdict(p.actions[original]) == case["action"]
                    alternative = next(
                        (i for kind in ("pass", "main2", "end_phase") for i in legal if p.actions[i].kind == kind), None
                    )
                    with torch.no_grad():
                        logits = mirror.policy.net(to_tensors({k: v[None] for k, v in obs.items()})).logits
                        probs = logits.softmax(-1)[0].tolist()
                    target = {
                        "decision": p.index,
                        "lp": list(p.lp),
                        "original": original,
                        "alternative": alternative,
                        "options": [asdict(a) for a in p.actions],
                        "probs": probs,
                        "obs": {k: v.tolist() for k, v in obs.items()},
                    }
                    initial_lp = list(p.lp)
                    if alternative is None and route == "decline":
                        stop = "unsupported_no_decline"
                        break
                    idx = original if route == "original" else alternative
                elif route == "original":
                    if p.index == case["end"]:
                        stop = "recorded_interval_end"
                        break
                    old = trace[p.index]
                    idx = old["choice"]
                    assert p.player == old["player"] and asdict(p.actions[idx]) == old["chosen"]
                    assert events_of(p.events) == old["events"]
                elif type(p.decision).__name__ in ("SelectIdleCmd", "SelectBattleCmd"):
                    stop = "next_action_command"
                    break
                elif p.index - case["start"] >= 128:
                    stop = "decision_budget"
                    break
                else:
                    idx = next((i for kind in ("pass", "yes") for i in legal if p.actions[i].kind == kind), legal[0])
                choices.append(
                    {"decision": p.index, "player": p.player, "choice": idx, "action": asdict(p.actions[idx])}
                )
            session.act(idx)
            mirror.on_decision(p, idx)
        result = session.result() if session.done else session.tracker.result
        assert not any(getattr(result, k) for k in HEALTH)
        if session.done:
            events.extend(events_of(session.tracker.events))
            stop = "terminal:" + result.reason
        assert target is not None and stop is not None
        damage = [sum(e["amount"] for e in events if e["type"] == "Damage" and e["player"] == p) for p in range(2)]
        # Original endpoint row can carry later messages beyond ChainEnd/DamageStepEnd;
        # compare the closed interval separately, never treat the entire row as that interval.
        close = "ChainEnd" if case["kind"] == "chain" else "DamageStepEnd"
        bounded = events[: next((i + 1 for i, e in enumerate(events) if e["type"] == close), len(events))]
        original_damage = [
            sum(e["amount"] for e in bounded if e["type"] == "Damage" and e["player"] == p) for p in range(2)
        ]
        if route == "original":
            assert original_damage == case["damage"]
        out = {
            "game": case["game"],
            "start": case["start"],
            "kind": case["kind"],
            "route": route,
            "source_trace_sha256": sha(base / "trace.jsonl.gz"),
            "source_replay_sha256": sha(base / "replay.json.gz"),
            "target": target,
            "stop": stop,
            "health": {k: getattr(result, k) for k in HEALTH},
            "initial_lp": initial_lp,
            "endpoint_lp": list(session.tracker.lp),
            "damage": damage,
            "closed_interval_damage": original_damage if route == "original" else None,
            "winner_deck": result.winner if session.done else None,
            "events": events,
            "choices": choices,
        }
        atomic_json(root / f"{case['game']:04d}-{case['start']:04d}-{route}.json", out)
        return out
    finally:
        session.close()
        mirror.host = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    out = root / "reviews"
    out.mkdir(exist_ok=False)
    panel = Path(read(root / "identity.json")["panel"])
    source = Path(read(panel / "identity.json")["source"])
    atomic_json(
        out / "identity.json",
        {
            "driver_sha256": sha(__file__),
            "native_sha256": sha(_core.__file__),
            "source_sha256": sha(source),
            "protocol_sha256": sha(root / "protocol.json"),
            "selection_sha256": sha(root / "selected.json"),
            "supplement_protocol_sha256": sha(root / "supplement-protocol.json"),
            "supplement_selection_sha256": sha(root / "supplement-selected.json"),
            "parent_report_sha256": sha(root / "report.json"),
            "environment": load_environment("md-2026-09").stamp(),
        },
    )
    torch.set_num_threads(1)
    rows = []
    try:
        for case in read(root / "selected.json") + read(root / "supplement-selected.json"):
            b = {route: branch(out, panel, source, case, route) for route in ("original", "decline")}
            assert b["original"]["target"] == b["decline"]["target"]
            target = b["original"]["target"]
            row = {
                "game": case["game"],
                "start": case["start"],
                "kind": case["kind"],
                "initiator_code": case["action"]["card"]["code"],
                "learner": case["learner"],
                "original_probability": target["probs"][target["original"]],
                "alternative_kind": target["options"][target["alternative"]]["kind"]
                if target["alternative"] is not None
                else None,
                "branches": {
                    k: {f: v[f] for f in ("stop", "initial_lp", "endpoint_lp", "damage", "winner_deck")}
                    for k, v in b.items()
                },
                "files": {k: sha(out / f"{case['game']:04d}-{case['start']:04d}-{k}.json") for k in b},
            }
            rows.append(row)
            print(row, flush=True)
        atomic_json(out / "report.json", {"identity_sha256": sha(out / "identity.json"), "cases": rows})
    except Exception as exc:
        atomic_json(out / "STOP.json", {"error": repr(exc), "completed": len(rows)})
        raise


if __name__ == "__main__":
    main()
