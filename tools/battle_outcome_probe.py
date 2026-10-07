"""Frozen battle-outcome probes on replayed, publicly visible attack-target pairs.

Collection preserves hidden/unsupported cases as coverage records. Predictor
training uses only eligible public rows and never updates the source actor.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mine_action_outcomes import read, sha
from review_action_outcomes import events_of
from ygorl import _core
from ygorl.agents.checkpoint import CheckpointAgent
from ygorl.data import load_environment
from ygorl.engine.duel import DuelSession
from ygorl.engine.replay import Replay
from ygorl.eval.battle_outcomes import ordinary_battle
from ygorl.nets.batch import to_tensors
from ygorl.train.registration import atomic_json

HEALTH = ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")


def heldout(a, b):
    pair = ":".join(map(str, sorted((a, b))))
    return int(hashlib.sha256(("battle-pair-v1:" + pair).encode()).hexdigest(), 16) % 4 == 0


def coords(loc):
    return tuple(loc[k] for k in ("controller", "location", "sequence"))


@torch.no_grad()
def collect_game(job):
    root_s, panel_s, source_s, record = job
    root, panel = Path(root_s), Path(panel_s)
    torch.set_num_threads(1)
    base = panel / "games" / f"{record['game']:04d}"
    for name in ("replay.json.gz", "trace.jsonl.gz", "spec.json", "result.json"):
        assert sha(base / name) == record["files"][name]
    with gzip.open(base / "trace.jsonl.gz", "rt") as f:
        trace = [json.loads(s) for s in f]
    replay = Replay.load(base / "replay.json.gz")
    duel = replay.duel(load_environment("md-2026-09"))
    mirror = CheckpointAgent(source_s, 0)
    mirror.on_duel_start(duel)
    net, vocab = mirror.policy.net.eval(), mirror.policy.vocab
    session = DuelSession(duel)
    learner = int(replay.first)
    pending = None
    rows, features, raw_obs = [], [], []

    def finish(stop):
        nonlocal pending
        if pending is None:
            return
        q = pending
        pending = None
        meta = {
            "game": record["game"],
            "pair": record["pair"],
            "decision": q["decision"],
            "stop": stop,
            "damage": q["damage"],
            "battles": q["battles"],
            "attacks": q["attacks"],
        }
        reason = None
        if stop not in ("DamageStepEnd", "terminal"):
            reason = "incomplete"
        elif len(q["battles"]) != 1 or len(q["attacks"]) != 1:
            reason = "missing_or_multiple_battle_attack"
        obs = q["obs"]
        ai = int(obs["actions"][q["action"], 1]) - 1
        if ai < 0:
            reason = reason or "missing_attacker_row"
        target_row = None
        ti = None
        if reason is None:
            attack, battle = q["attacks"][0], q["battles"][0]
            if coords(attack["card"]) != coords(battle["attacker"]) or coords(attack["target"]) != coords(
                battle["target"]
            ):
                reason = "changed_coordinates"
            elif coords(q["card_loc"]) != coords(battle["attacker"]):
                reason = "changed_attacker"
            if battle["target"]["location"] != 0:
                loc = battle["target"]
                ids = [
                    i
                    for i, c in enumerate(obs["cards"])
                    if c[1] == 3 and c[2] == loc["sequence"] and c[4] == int(loc["controller"] != learner)
                ]
                if len(ids) != 1:
                    reason = reason or "missing_target_row"
                else:
                    ti = ids[0]
                    target_row = obs["cards"][ti]
            if reason is None:
                baseline = ordinary_battle(obs["cards"][ai], target_row)
                if baseline is None:
                    reason = "hidden_or_unsupported_public_state"
        if reason is None:
            attacker = obs["cards"][ai]
            target = target_row if target_row is not None else np.zeros(23, dtype=np.int64)
            codes = [vocab.password(int(attacker[0])), vocab.password(int(target[0])) if ti is not None else 0]
            damage = [q["damage"][learner], q["damage"][1 - learner]]
            destroy = [int(bool(battle["attacker_destroyed"])), int(bool(battle["target_destroyed"]))]
            f = net.features(to_tensors({k: v[None] for k, v in obs.items()}))
            pair = torch.cat(
                [f.context[0], f.cards[0, ai], f.cards[0, ti] if ti is not None else torch.zeros_like(f.context[0])]
            ).numpy()
            numeric = np.asarray(
                [
                    attacker[17] / 4000,
                    attacker[18] / 4000,
                    target[17] / 4000,
                    target[18] / 4000,
                    float(target[6] == 1),
                    float(target[6] == 3),
                ],
                dtype=np.float32,
            )
            meta.update(
                eligible=True,
                row=len(features),
                codes=codes,
                heldout=heldout(*codes),
                direct=ti is None,
                damage_relative=damage,
                destroy=destroy,
                baseline=baseline,
                baseline_agrees=baseline == {"damage": damage, "destroy": destroy},
                attacker_row=ai,
                target_row=ti,
            )
            features.append(
                {
                    "pair": pair,
                    "numeric": numeric,
                    "damage": np.asarray(damage, dtype=np.float32),
                    "destroy": np.asarray(destroy, dtype=np.float32),
                }
            )
            raw_obs.append(obs)
        else:
            meta.update(eligible=False, exclusion=reason)
        rows.append(meta)

    def consume(events):
        nonlocal pending
        for e in events:
            if pending is None:
                continue
            name = e["type"]
            if name == "Attack":
                pending["attacks"].append(e)
            elif name == "Battle":
                pending["battles"].append(e)
            elif name == "Damage" and pending["battles"]:
                pending["damage"][e["player"]] += e["amount"]
            elif name == "DamageStepEnd":
                finish(name)
            elif name in ("NewPhase", "NewTurn"):
                finish(name)

    try:
        while (p := session.point) is not None:
            consume(events_of(p.events))
            old = trace[p.index]
            idx = old["choice"]
            assert p.player == old["player"] and asdict(p.actions[idx]) == old["chosen"]
            assert events_of(p.events) == old["events"]
            assert mirror.host.player() == p.player and len(mirror.host.actions()) == len(p.actions)
            if p.player == learner and p.actions[idx].kind == "attack":
                finish("next_attack")
                pending = {
                    "decision": p.index,
                    "action": idx,
                    "card_loc": asdict(p.actions[idx].card.loc),
                    "obs": {k: v.copy() for k, v in mirror.host.observe().items()},
                    "attacks": [],
                    "battles": [],
                    "damage": [0, 0],
                }
            session.act(idx)
            mirror.on_decision(p, idx)
        consume(events_of(session.tracker.events))
        finish("terminal")
        result = session.result()
        assert result.reason == "win" and not any(getattr(result, k) for k in HEALTH)
        assert result.responses == replay.responses
        expected = read(base / "result.json")
        assert result.winner == expected["winner"] and list(result.lp) == expected["lp"]
        out = root / "games" / f"{record['game']:04d}"
        out.mkdir(exist_ok=False)
        atomic_json(out / "rows.json", rows)
        if features:
            np.savez_compressed(out / "features.npz", **{k: np.stack([x[k] for x in features]) for k in features[0]})
            np.savez_compressed(out / "observations.npz", **{k: np.stack([x[k] for x in raw_obs]) for k in raw_obs[0]})
        record = {
            "game": record["game"],
            "pair": record["pair"],
            "attacks": len(rows),
            "eligible": len(features),
            "health": {k: getattr(result, k) for k in HEALTH},
            "responses_exact": True,
            "files": {p.name: sha(p) for p in out.iterdir()},
        }
        atomic_json(out / "done.json", record)
        return record
    finally:
        session.close()
        mirror.host = None


def collect(root, protocol, workers):
    panel = Path(protocol["panel"]).resolve()
    assert not (root / "identity.json").exists(), "preserve prior run"
    records = read(panel / "collection.json")["games"]
    lo, hi = protocol["game_ids"]
    records = [r for r in records if lo <= r["game"] <= hi]
    assert len(records) == hi - lo + 1
    atomic_json(
        root / "identity.json",
        {
            "driver_sha256": sha(__file__),
            "baseline_sha256": sha("src/ygorl/eval/battle_outcomes.py"),
            "events_helper_sha256": sha("tools/review_action_outcomes.py"),
            "native_sha256": sha(_core.__file__),
            "source_sha256": sha(protocol["source"]),
            "collection_sha256": sha(panel / "collection.json"),
            "protocol_sha256": sha(root / "protocol.json"),
            "environment": load_environment("md-2026-09").stamp(),
        },
    )
    (root / "games").mkdir()
    jobs = [(str(root), str(panel), protocol["source"], r) for r in records]
    done = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for row in pool.imap_unordered(collect_game, jobs):
            done.append(row)
            atomic_json(root / "progress.json", {"completed": len(done), "total": len(records)})
            if len(done) % 16 == 0:
                print("COLLECT", len(done), "/", len(records), flush=True)
    atomic_json(
        root / "collection.json",
        {"identity_sha256": sha(root / "identity.json"), "games": sorted(done, key=lambda x: x["game"])},
    )


def metrics(output, damage, destroy):
    target = (damage[:, 0] > 0).long() + 2 * (damage[:, 1] > 0).long()
    pred = output[:, :4].argmax(-1)
    confusion = torch.bincount(target * 4 + pred, minlength=16).reshape(4, 4)
    present = confusion.sum(1) > 0
    return {
        "n": len(target),
        "direction_accuracy": float((pred == target).float().mean()),
        "direction_balanced_accuracy": float((confusion.diag()[present] / confusion.sum(1)[present]).float().mean()),
        "direction_confusion": confusion.tolist(),
        "damage_mae_lp": float((output[:, 4:6].clamp_min(0) * 4000 - damage).abs().mean()),
        "destruction_exact_accuracy": float(((output[:, 6:] > 0) == destroy.bool()).all(1).float().mean()),
    }


def loss(output, damage, destroy):
    direction = (damage[:, 0] > 0).long() + 2 * (damage[:, 1] > 0).long()
    return (
        F.cross_entropy(output[:, :4], direction)
        + F.smooth_l1_loss(output[:, 4:6], damage / 4000)
        + F.binary_cross_entropy_with_logits(output[:, 6:], destroy)
    )


def fit(root, protocol):
    assert not (root / "fit-identity.json").exists(), "preserve fitted run"
    collection = read(root / "collection.json")
    meta, chunks = [], []
    for g in collection["games"]:
        folder = root / "games" / f"{g['game']:04d}"
        assert all(sha(folder / k) == v for k, v in g["files"].items())
        rows = read(folder / "rows.json")
        if g["eligible"]:
            with np.load(folder / "features.npz") as d:
                chunks.append({k: torch.from_numpy(d[k].copy()) for k in d.files})
            meta.extend(x for x in rows if x["eligible"])
    d = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}

    def split(m):
        return "train" if m["pair"] % 10 < 6 else "val" if m["pair"] % 10 < 8 else "test"

    ids = {
        s: torch.tensor([i for i, m in enumerate(meta) if split(m) == s and (s == "test" or not m["heldout"])])
        for s in ("train", "val", "test")
    }
    assert all(len(x) >= 30 for x in ids.values()), {k: len(v) for k, v in ids.items()}
    train_pairs = {tuple(sorted(meta[i]["codes"])) for i in ids["train"].tolist()}
    assert all(tuple(sorted(m["codes"])) not in train_pairs for m in meta if m["heldout"])
    width = d["pair"].shape[1] + 6
    xs = {
        "numeric": F.pad(d["numeric"], (0, width - 6)),
        "frozen_pair": F.pad(d["pair"], (0, 6)),
        "combined": torch.cat([d["pair"], d["numeric"]], -1),
    }
    atomic_json(
        root / "fit-identity.json",
        {
            "driver_sha256": sha(__file__),
            "collection_sha256": sha(root / "collection.json"),
            "protocol_sha256": sha(root / "protocol.json"),
            "counts": {k: len(v) for k, v in ids.items()},
            "input_width": width,
        },
    )
    candidates = []
    for mode, x in xs.items():
        mean, scale = x[ids["train"]].mean(0), x[ids["train"]].std(0, correction=0).clamp_min(0.001)
        normalized = (x - mean) / scale
        for seed in protocol["fit_seeds"]:
            torch.manual_seed(seed)
            head = torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
            optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
            rng = torch.Generator().manual_seed(seed + 1)
            best, history = None, []
            for step in range(1, 257):
                idx = ids["train"][torch.randint(len(ids["train"]), (128,), generator=rng)]
                optimizer.zero_grad()
                value = loss(head(normalized[idx]), d["damage"][idx], d["destroy"][idx])
                assert torch.isfinite(value)
                value.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
                optimizer.step()
                if step in (64, 128, 256):
                    with torch.no_grad():
                        val = float(
                            loss(head(normalized[ids["val"]]), d["damage"][ids["val"]], d["destroy"][ids["val"]])
                        )
                    history.append({"step": step, "validation_loss": val})
                    if best is None or val < best[0]:
                        best = (val, step, copy.deepcopy(head.state_dict()))
            path = root / f"{mode}-{seed}.pt"
            torch.save(
                {"head": best[2], "mean": mean, "scale": scale, "selected_step": best[1], "seed": seed, "mode": mode},
                path,
            )
            candidates.append(
                {"mode": mode, "seed": seed, "selected_step": best[1], "history": history, "sha256": sha(path)}
            )
    atomic_json(root / "selection.json", candidates)  # freeze all selections before test scoring
    results = []
    for candidate in candidates:
        state = torch.load(root / f"{candidate['mode']}-{candidate['seed']}.pt", weights_only=True)
        head = torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
        head.load_state_dict(state["head"])
        with torch.no_grad():
            pred = head((xs[candidate["mode"]] - state["mean"]) / state["scale"])
        groups = {"test": ids["test"].tolist()}
        for key, check in {
            "heldout_pairs": lambda m: m["heldout"],
            "seen_hash_pairs": lambda m: not m["heldout"],
            "direct": lambda m: m["direct"],
            "monster": lambda m: not m["direct"],
            "ordinary_agreement": lambda m: m["baseline_agrees"],
            "effect_or_numeric_exception": lambda m: not m["baseline_agrees"],
        }.items():
            groups[key] = [i for i in groups["test"] if check(meta[i])]
        results.append(
            {
                "mode": candidate["mode"],
                "seed": candidate["seed"],
                "selected_step": candidate["selected_step"],
                "metrics": {k: metrics(pred[v], d["damage"][v], d["destroy"][v]) for k, v in groups.items() if v},
            }
        )
    baseline = torch.zeros(len(meta), 8)
    for i, m in enumerate(meta):
        b = m["baseline"]
        direction = int(b["damage"][0] > 0) + 2 * int(b["damage"][1] > 0)
        baseline[i, direction] = 1
        baseline[i, 4:6] = torch.tensor(b["damage"]) / 4000
        baseline[i, 6:] = torch.tensor(b["destroy"]) * 2 - 1
    all_rows = [x for g in collection["games"] for x in read(root / "games" / f"{g['game']:04d}" / "rows.json")]
    report = {
        "fit_identity_sha256": sha(root / "fit-identity.json"),
        "selection_sha256": sha(root / "selection.json"),
        "coverage": dict(Counter("eligible" if x["eligible"] else x["exclusion"] for x in all_rows)),
        "split_counts": {k: len(v) for k, v in ids.items()},
        "results": results,
        "ordinary_baseline": {k: metrics(baseline[v], d["damage"][v], d["destroy"][v]) for k, v in groups.items() if v},
        "test_heldout_floor_met": len(groups["heldout_pairs"]) >= 30,
    }
    atomic_json(root / "report.json", report)
    atomic_json(root / "sample-index.json", meta)
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "splits": report["split_counts"],
                "baseline": report["ordinary_baseline"]["test"],
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--phase", choices=("collect", "fit", "all"), default="all")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    root = a.output.resolve()
    protocol = read(root / "protocol.json")
    try:
        if a.phase in ("collect", "all"):
            collect(root, protocol, a.workers)
        if a.phase in ("fit", "all"):
            fit(root, protocol)
    except Exception as exc:
        atomic_json(root / "STOP.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
