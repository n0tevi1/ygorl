"""Bounded native regression forks for battle mechanisms; not policy training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import battle_effect_residual as predictors
from battle_outcome_probe import HEALTH, coords
from mine_action_outcomes import read, sha
from review_action_outcomes import events_of
from ygorl import _core
from ygorl.agents.checkpoint import CheckpointAgent
from ygorl.data import load_environment
from ygorl.engine.duel import DuelSession
from ygorl.engine.replay import Replay
from ygorl.eval.battle_outcomes import decompose_battle_outcome
from ygorl.train.registration import atomic_json


def digest_obs(obs, case):
    h = hashlib.sha256()
    for k, v in sorted(obs.items()):
        h.update(k.encode())
        h.update(v.tobytes())
    h.update(str((case["attacker_row"], case["target_row"])).encode())
    return h.hexdigest()


def select(root, p):
    assert not (root / "selection.json").exists()
    candidates = []
    for pool, path in enumerate(p["roots"]):
        collection = Path(path).resolve()
        _, meta = predictors.common.dataset(collection)
        panel = Path(read(collection / "protocol.json")["panel"]).resolve()
        for m in meta:
            c = decompose_battle_outcome(m["baseline"], m["battles"][0], m["damage_relative"])
            if c is None:
                continue
            group = (
                "train"
                if m["pair"] % 10 < 6 and not m["heldout"]
                else "validation"
                if m["pair"] % 10 < 8 and not m["heldout"]
                else "development_holdout"
            )
            if pool == len(p["roots"]) - 1:
                group = "previous_confirmation"
            candidates.append(
                {
                    **m,
                    "pool": pool,
                    "collection": str(collection),
                    "panel": str(panel),
                    "components": c,
                    "prior_membership": group,
                }
            )
    used, selected = set(), []
    categories = ["numeric_damage_delta", "non_arithmetic_damage_delta", "destruction_delta", "ordinary"]
    for category in categories:
        rows = [
            m
            for m in candidates
            if (m["baseline_agrees"] if category == "ordinary" else any(m["components"][category]))
        ]
        rows.sort(
            key=lambda m: (
                not bool(m["damage_relative"][0]),
                m["pool"] != len(p["roots"]) - 1,
                m["pool"],
                m["game"],
                m["decision"],
            )
        )
        pairs = set()
        count = 0
        for m in rows:
            key = (m["pool"], m["game"], m["decision"])
            pair = tuple(sorted(m["codes"]))
            if key in used or pair in pairs:
                continue
            selected.append({**m, "category": category, "case_id": len(selected)})
            used.add(key)
            pairs.add(pair)
            count += 1
            if count == p["per_category"]:
                break
        assert count == p["per_category"], (category, count)
    atomic_json(root / "selection.json", selected)
    paths = [
        Path(__file__),
        Path("src/ygorl/eval/battle_outcomes.py"),
        Path("tools/review_action_outcomes.py"),
        Path("tools/battle_effect_residual.py"),
        Path("src/ygorl/nets/battle_residual.py"),
        Path(_core.__file__),
        Path(p["source"]),
        root / "protocol.json",
        root / "selection.json",
    ]
    paths += [Path(x) / "collection.json" for x in p["roots"]]
    for s in read(Path(p["predictors"]) / "selection.json"):
        path = Path(p["predictors"]) / f"{s['arm']}-{s['seed']}.pt"
        assert sha(path) == s["sha256"]
        paths.append(path)
    paths.append(Path(p["predictors"]) / "selection.json")
    for m in selected:
        paths.extend(
            Path(m["panel"]) / "games" / f"{m['game']:04d}" / name
            for name in ["trace.jsonl.gz", "replay.json.gz", "spec.json", "result.json"]
        )
    atomic_json(
        root / "identity.json",
        {"bindings": {str(x.resolve()): sha(x) for x in paths}, "environment": load_environment("md-2026-09").stamp()},
    )
    print("SELECTED", len(selected), "cases", flush=True)


@torch.no_grad()
def branch(root, p, case, route, fork=None):
    directory = root / "cases" / f"{case['case_id']:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (route + ("" if fork is None else f"-{fork['decision']}") + ".json")
    assert not path.exists()
    base = Path(case["panel"]) / "games" / f"{case['game']:04d}"
    with gzip.open(base / "trace.jsonl.gz", "rt") as f:
        trace = [json.loads(line) for line in f]
    replay = Replay.load(base / "replay.json.gz")
    duel = replay.duel(load_environment("md-2026-09"))
    mirror = CheckpointAgent(p["source"], 0)
    mirror.on_duel_start(duel)
    session = DuelSession(duel)
    start = case["decision"]
    learner = int(replay.first)
    events = []
    choices = []
    forks = []
    stop = None
    declaration = None
    fork_obs = None
    intervened = False

    def consume(new):
        nonlocal stop
        for e in new:
            events.append(e)
            if e["type"] in ("DamageStepEnd", "NewPhase", "NewTurn"):
                stop = e["type"]
                break

    try:
        while (point := session.point) is not None:
            assert mirror.host.player() == point.player and len(mirror.host.actions()) == len(point.actions)
            if point.index > start:
                consume(events_of(point.events))
                if stop:
                    break
            obs = mirror.host.observe()
            legal = np.flatnonzero(obs["action_mask"]).tolist()
            if point.index == start:
                declaration = digest_obs(obs, case)
                stored = Path(case["collection"]) / "games" / f"{case['game']:04d}" / "observations.npz"
                with np.load(stored) as f:
                    for k in f.files:
                        np.testing.assert_array_equal(obs[k], f[k][case["row"]])
                if route == "original":
                    np.savez_compressed(directory / "declaration.npz", **obs)
            exact = route == "original" or (point.index <= (fork["decision"] if fork else start))
            if exact:
                old = trace[point.index]
                idx = old["choice"]
                assert point.player == old["player"] and asdict(point.actions[idx]) == old["chosen"]
                assert events_of(point.events) == old["events"]
            if point.index >= start:
                decline = next((i for kind in ("pass", "no") for i in legal if point.actions[i].kind == kind), None)
                if (
                    route == "original"
                    and point.index > start
                    and point.actions[idx].kind in ("chain", "activate", "yes")
                    and decline is not None
                ):
                    forks.append(
                        {
                            "decision": point.index,
                            "player": point.player,
                            "original": idx,
                            "decline": decline,
                            "action": asdict(point.actions[idx]),
                            "before_battle": not any(e["type"] == "Battle" for e in events),
                        }
                    )
                if route == "decline_attack" and point.index == start:
                    idx = next(
                        (i for kind in ("main2", "end_phase") for i in legal if point.actions[i].kind == kind), None
                    )
                    if idx is None:
                        stop = "unsupported_no_decline"
                        break
                    intervened = True
                elif fork and point.index == fork["decision"]:
                    fork_obs = digest_obs(obs, case)
                    assert idx == fork["original"]
                    idx = fork["decline"] if route == "fork_decline" else fork["original"]
                    assert idx in legal
                    intervened = True
                elif not exact:
                    if type(point.decision).__name__ in ("SelectIdleCmd", "SelectBattleCmd"):
                        stop = "next_action_command"
                        break
                    if point.index - start >= p["decision_budget"]:
                        stop = "decision_budget"
                        break
                    idx = next(
                        (i for kind in ("pass", "no", "finish", "yes") for i in legal if point.actions[i].kind == kind),
                        legal[0],
                    )
                choices.append(
                    {
                        "decision": point.index,
                        "player": point.player,
                        "choice": idx,
                        "action": asdict(point.actions[idx]),
                    }
                )
            session.act(idx)
            mirror.on_decision(point, idx)
        if session.done:
            consume(events_of(session.tracker.events))
            if stop is None:
                stop = "terminal"
        result = session.result() if session.done else session.tracker.result
        assert not any(getattr(result, k) for k in HEALTH)
        assert declaration is not None and stop is not None
        battles = [e for e in events if e["type"] == "Battle"]
        attacks = [e for e in events if e["type"] == "Attack"]
        damage = [0, 0]
        seen = False
        for e in events:
            if e["type"] == "Battle":
                seen = True
            elif e["type"] == "Damage" and seen:
                damage[int(e["player"] != learner)] += e["amount"]
        same_pair = len(battles) == len(attacks) == 1 and all(
            coords(battles[0][key]) == coords(case["battles"][0][key]) for key in ("attacker", "target")
        )
        comparable = stop in ("DamageStepEnd", "terminal") and same_pair
        components = decompose_battle_outcome(case["baseline"], battles[0], damage) if comparable else None
        destroyed = (
            [int(bool(battles[0][k])) for k in ("attacker_destroyed", "target_destroyed")] if comparable else None
        )
        if route == "original":
            assert comparable and damage == case["damage_relative"] and destroyed == case["destroy"]
            assert components == case["components"]
        if fork:
            assert intervened
        out = {
            "case_id": case["case_id"],
            "route": route,
            "fork": fork,
            "declaration_sha256": declaration,
            "fork_observation_sha256": fork_obs,
            "intervened": intervened,
            "stop": stop,
            "comparable": comparable,
            "damage": damage if comparable else None,
            "destroy": destroyed,
            "components": components,
            "events": events,
            "choices": choices,
            "optional_forks": forks,
            "health": {k: getattr(result, k) for k in HEALTH},
        }
        atomic_json(path, out)
        return out
    finally:
        session.close()
        mirror.host = None


def collect(root, p):
    assert all(sha(k) == v for k, v in read(root / "identity.json")["bindings"].items())
    assert not (root / "collection.json").exists()
    records = []
    for case in read(root / "selection.json"):
        original = branch(root, p, case, "original")
        decline = branch(root, p, case, "decline_attack")
        assert original["declaration_sha256"] == decline["declaration_sha256"]
        forks = original["optional_forks"]
        prefer_before = case["category"] != "non_arithmetic_damage_delta"
        forks = sorted(forks, key=lambda f: (f["before_battle"] != prefer_before, f["decision"]))[: p["max_forks"]]
        pairs = []
        for fork in forks:
            a = branch(root, p, case, "fork_original", fork)
            b = branch(root, p, case, "fork_decline", fork)
            assert a["declaration_sha256"] == b["declaration_sha256"] == original["declaration_sha256"]
            assert a["fork_observation_sha256"] == b["fork_observation_sha256"]
            pairs.append(
                {
                    "fork": fork,
                    "both_comparable": a["comparable"] and b["comparable"],
                    "damage_if_activated": a["damage"],
                    "damage_if_declined": b["damage"],
                    "destroy_if_activated": a["destroy"],
                    "destroy_if_declined": b["destroy"],
                }
            )
        records.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "prior_membership": case["prior_membership"],
                "original_damage": original["damage"],
                "decline_stop": decline["stop"],
                "optional_fork_count": len(forks),
                "pairs": pairs,
            }
        )
        atomic_json(root / "progress.json", {"done": len(records), "total": len(read(root / "selection.json"))})
        print("CASE", case["case_id"], case["category"], "damage", original["damage"], "forks", len(forks), flush=True)
    files = {str(f.relative_to(root)): sha(f) for f in sorted((root / "cases").rglob("*")) if f.is_file()}
    atomic_json(
        root / "collection.json", {"records": records, "files": files, "selection_sha256": sha(root / "selection.json")}
    )


def evaluate(root, p):
    assert not (root / "report.json").exists()
    meta = read(root / "selection.json")
    collection = read(root / "collection.json")
    data = []
    for m in meta:
        with np.load(Path(m["collection"]) / "games" / f"{m['game']:04d}" / "features.npz") as f:
            data.append({k: torch.from_numpy(f[k][m["row"]].copy()) for k in f.files})
    data = {k: torch.stack([x[k] for x in data]) for k in data[0]}
    xn, xc = predictors.features(data)
    models = read(Path(p["predictors"]) / "selection.json")
    results = []
    raw = []
    for s in models:
        path = Path(p["predictors"]) / f"{s['arm']}-{s['seed']}.pt"
        assert sha(path) == s["sha256"]
        state = torch.load(path, weights_only=True)
        model = predictors.build(s["arm"])
        model.load_state_dict(state["model"])
        with torch.no_grad():
            out, gate = predictors.forward(
                model,
                s["arm"],
                (xn - state["numeric_mean"]) / state["numeric_scale"],
                (xc - state["context_mean"]) / state["context_scale"],
            )
        raw.append(out.numpy())
        results.append(
            {
                "arm": s["arm"],
                "seed": s["seed"],
                "predicted_direction": out[:, :4].argmax(-1).tolist(),
                "self_damage_truth": [m["damage_relative"][0] > 0 for m in meta],
            }
        )
    np.savez_compressed(root / "predictions.npz", outputs=np.stack(raw))
    conflicts = [
        {"case_id": r["case_id"], **pair}
        for r in collection["records"]
        for pair in r["pairs"]
        if pair["both_comparable"]
        and (
            pair["damage_if_activated"] != pair["damage_if_declined"]
            or pair["destroy_if_activated"] != pair["destroy_if_declined"]
        )
    ]
    atomic_json(
        root / "report.json",
        {
            "collection_sha256": sha(root / "collection.json"),
            "predictions_sha256": sha(root / "predictions.npz"),
            "cases": len(meta),
            "native_branches": sum(2 + 2 * r["optional_fork_count"] for r in collection["records"]),
            "same_declaration_different_outcomes": conflicts,
            "frozen_predictor_regression": results,
            "interpretation": "retrospective mechanism regression; not an independent test or solved action values",
        },
    )
    print("DONE", len(meta), "cases;", len(conflicts), "same-input outcome differences", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["select", "collect", "evaluate", "all"], default="all")
    a = parser.parse_args()
    root = a.output.resolve()
    p = read(root / "protocol.json")
    torch.set_num_threads(1)
    try:
        if a.phase in ("select", "all"):
            select(root, p)
        if a.phase in ("collect", "all"):
            collect(root, p)
        if a.phase in ("evaluate", "all"):
            evaluate(root, p)
    except Exception as exc:
        atomic_json(root / "STOP.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
