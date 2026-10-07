"""Registered frozen-feature masker pilot on archived Ash data and fresh shadow games.

Requires the archived ash-correction-pilot/extra research artifacts (including their
collection/branch-verification drivers); their hashes are bound in identity.json.
No PPO changes and no filtered actions are executed. Run --phase all once in a new
output directory containing protocol.json. Refuses to overwrite completed phases.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ygorl import _core
from ygorl.agents.registry import agent_factory
from ygorl.data import load_environment
from ygorl.engine.duel import DuelConfig
from ygorl.eval.agent_matrix import cell_specs, pairing_slots, sample_pairings
from ygorl.nets.action_filter import ActionRiskHead, soft_filter
from ygorl.nets.batch import to_tensors
from ygorl.train.checkpoint import load_policy
from ygorl.train.heuristic_demos import load_data
from ygorl.train.registration import atomic_json


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def archive(pilot):
    sys.path.insert(0, str(Path(pilot).resolve()))
    return tuple(importlib.import_module(n) for n in ("collect", "fit", "label"))


def heldout(code):
    return int(hashlib.sha256(f"chain-relation-v1:{code}".encode()).hexdigest(), 16) % 4 == 0


def subset(data, indices):
    idx = torch.tensor(indices, dtype=torch.long)
    return {
        "obs": {k: v[idx] for k, v in data["obs"].items()},
        "meta": [data["meta"][i] for i in indices],
        "good": data["good"][idx],
        "bad": data["bad"][idx],
    }


@torch.no_grad()
def cache(net, data):
    parts = {k: [] for k in ("context", "actions", "logits")}
    for start in range(0, len(data["meta"]), 32):
        obs = {k: v[start : start + 32] for k, v in data["obs"].items()}
        f = net.features(obs)
        ids = torch.arange(len(f.context))
        ms = data["meta"][start : start + 32]
        pairs = torch.tensor(
            [[m["ash"][0], next(i for i, a in enumerate(m["options"]) if a["kind"] == "pass")] for m in ms]
        )
        parts["context"].append(f.context)
        parts["actions"].append(f.actions[ids[:, None], pairs])
        parts["logits"].append(net.logits(f))
    return {k: torch.cat(v) for k, v in parts.items()}


def targets(data):
    # Columns always Ash / pass. These labels are only branch-verified local preferences.
    return torch.tensor([[1.0, 0.0] if m["owner"] == "self" else [0.0, 1.0] for m in data["meta"]])


def groups(data):
    return {
        owner: torch.tensor([i for i, m in enumerate(data["meta"]) if m["owner"] == owner])
        for owner in ("self", "opponent")
    }


def balanced_bce(pred, target, gs):
    per = F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean(1)
    return torch.stack([per[idx].mean() for idx in gs.values()]).mean()


def classification(scores, data, threshold):
    result = {}
    y = targets(data).bool()
    for owner, idx in groups(data).items():
        if len(idx) == 0:
            result[owner] = {"n": 0}
            continue
        s, truth = scores[idx], y[idx]
        result[owner] = {
            "n": len(idx),
            "good_flag_count": int((s[~truth] > threshold).sum()),
            "bad_flag_count": int((s[truth] > threshold).sum()),
            "good_flag_rate": int((s[~truth] > threshold).sum()) / len(idx),
            "bad_flag_rate": int((s[truth] > threshold).sum()) / len(idx),
        }
    return result


def gate(metrics, floor=1):
    if set(metrics) != {"self", "opponent"}:
        return False
    return all(
        v["n"] >= floor and v["good_flag_count"] * 20 <= v["n"] and v["bad_flag_count"] * 2 >= v["n"]
        for v in metrics.values()
    )


def fit(root, pilot):
    collect, oldfit, _ = archive(pilot)
    protocol = read(root / "protocol.json")
    assert not (root / "identity.json").exists(), "preserve registered run; use a new output directory"
    loaded = load_policy(collect.SOURCE)
    net = loaded.net.eval().requires_grad_(False)
    data = {split: oldfit.dataset(split) for split in ("train", "val")}
    # Card split is additional to (never replaces) the existing deal-group split.
    data = {
        split: subset(d, [i for i, m in enumerate(d["meta"]) if not heldout(m["chain"][-1]["code"])])
        for split, d in data.items()
    }
    for d in data.values():
        assert all(len(ids) >= 6 for ids in groups(d).values())
    bindings = {
        str(p.resolve()): sha(p)
        for p in (
            Path(__file__),
            Path(_core.__file__),
            Path("src/ygorl/nets/action_filter.py"),
            Path(collect.SOURCE),
            root / "protocol.json",
        )
    }
    for folder in (Path(pilot), Path(pilot).parent / "ash-correction-extra-2026-10-07"):
        for name in ("collection.json", "labels-development.json", "label-identity.json"):
            bindings[str((folder / name).resolve())] = sha(folder / name)
    for name in ("collect.py", "fit.py", "label.py"):
        bindings[str((Path(pilot) / name).resolve())] = sha(Path(pilot) / name)
    identity = {
        "bindings": bindings,
        "environment": load_environment("md-2026-09").stamp(),
        "source": str(collect.SOURCE),
        "protocol": protocol,
        "counts": {k: {g: len(ids) for g, ids in groups(d).items()} for k, d in data.items()},
    }
    atomic_json(root / "identity.json", identity)
    c = {k: cache(net, d) for k, d in data.items()}
    train, val = data["train"], data["val"]
    tc, vc = c["train"], c["val"]
    yt, yv = targets(train), targets(val)
    gs, vg = groups(train), groups(val)
    heads, reports = [], []
    for sid, seed in enumerate(protocol["fit_seeds"]):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        head = ActionRiskHead(tc["context"].shape[-1])
        head.fit_normalization(head.inputs(tc["context"], tc["actions"]).flatten(0, 1))
        opt = torch.optim.Adam(head.parameters(), lr=0.001)
        best, history = None, []
        for step in range(1, 257):
            idx = torch.tensor(np.concatenate([rng.choice(v.numpy(), 16) for v in gs.values()]))
            pred = head(tc["context"][idx], tc["actions"][idx])
            loss = F.binary_cross_entropy_with_logits(pred, yt[idx])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if step in (32, 64, 128, 256):
                with torch.no_grad():
                    vloss = float(balanced_bce(head(vc["context"], vc["actions"]), yv, vg))
                history.append({"step": step, "validation_bce": vloss})
                if best is None or vloss < best[0]:
                    best = (vloss, step, copy.deepcopy(head.state_dict()))
        head.load_state_dict(best[2])
        head.eval().requires_grad_(False)
        heads.append(head)
        path = root / f"head-{sid}.pt"
        torch.save(
            {
                "state": head.state_dict(),
                "d_model": tc["context"].shape[-1],
                "seed": seed,
                "selected_step": best[1],
                "identity_sha256": sha(root / "identity.json"),
            },
            path,
        )
        reports.append({"seed": seed, "nodes": history, "selected_step": best[1], "sha256": sha(path)})
    with torch.no_grad():
        scores = torch.stack([h(vc["context"], vc["actions"]).sigmoid() for h in heads]).mean(0)
    choices = [{"threshold": t, "metrics": classification(scores, val, t)} for t in protocol["thresholds"]]
    eligible = [x for x in choices if gate(x["metrics"])]
    threshold = eligible[0]["threshold"] if eligible else None
    atomic_json(
        root / "selection.json",
        {
            "identity_sha256": sha(root / "identity.json"),
            "heads": reports,
            "threshold": threshold,
            "validation_choices": choices,
            "enabled_for_shadow_analysis": threshold is not None,
        },
    )
    print("FIT COMPLETE", identity["counts"], "threshold", threshold, flush=True)


def verify_identity(root):
    identity = read(root / "identity.json")
    assert all(sha(p) == digest for p, digest in identity["bindings"].items()), "registered source/data changed"
    assert read(root / "selection.json")["identity_sha256"] == sha(root / "identity.json")
    return identity


def specs(root):
    identity = read(root / "identity.json")
    env = load_environment("md-2026-09")
    assert env.stamp() == identity["environment"]
    protocol = identity["protocol"]
    seed = protocol["new_panel_seed"]
    decks = [m.deck for m in sorted(env.meta_decks, key=lambda x: x.deck.name)]
    assert len(decks) == 20
    cfg = DuelConfig.from_environment(env)
    slots = pairing_slots(decks, sample_pairings(20, protocol["new_deal_groups"], seed), seed, cfg)
    return cell_specs(agent_factory("policy:" + identity["source"]), agent_factory("greedy"), slots, env, cfg)


def collect_game(job):
    root, pilot, game, spec = job
    collect, _, _ = archive(pilot)
    collect.ROOT = root
    collect.SEED = read(root / "protocol.json")["new_panel_seed"]
    report = collect.play((game, spec))
    # Every new group is a shadow test group; archived collection split rules do not apply.
    path = root / "games" / f"{game:04d}"
    s = read(path / "spec.json")
    s["split"] = "shadow"
    atomic_json(path / "spec.json", s)
    report["split"] = "shadow"
    report["files"]["spec.json"] = sha(path / "spec.json")
    atomic_json(path / "done.json", report)
    return report


def collect_fresh(root, pilot, workers):
    verify_identity(root)
    assert not (root / "collection.json").exists()
    collect, _, _ = archive(pilot)
    games = specs(root)
    atomic_json(
        root / "panel-identity.json",
        {
            "selection_sha256": sha(root / "selection.json"),
            "specs": [collect.spec_json(s) for s in games],
            "executed_policy": "unmodified source actor",
        },
    )
    records = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for result in pool.imap_unordered(collect_game, [(root, pilot, i, s) for i, s in enumerate(games)]):
            records.append(result)
            atomic_json(root / "progress.json", {"phase": "collect", "done": len(records), "total": len(games)})
            if len(records) % 32 == 0:
                print("COLLECT", len(records), "/", len(games), flush=True)
    atomic_json(
        root / "collection.json",
        {"games": sorted(records, key=lambda r: r["game"]), "panel_identity_sha256": sha(root / "panel-identity.json")},
    )


def label_game(job):
    root, pilot, game, spec = job
    _, _, label = archive(pilot)
    label.ROOT = root
    (root / "branches").mkdir(exist_ok=True)
    return label.verify((game, spec))


def label_fresh(root, pilot, workers):
    verify_identity(root)
    assert not (root / "labels.json").exists()
    games = specs(root)
    records = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for result in pool.imap_unordered(label_game, [(root, pilot, i, s) for i, s in enumerate(games)]):
            records.append(result)
            atomic_json(root / "progress.json", {"phase": "label", "done": len(records), "total": len(games)})
            if len(records) % 64 == 0:
                print("LABEL", len(records), "/", len(games), flush=True)
    atomic_json(
        root / "labels.json",
        {"collection_sha256": sha(root / "collection.json"), "games": sorted(records, key=lambda r: r["game"])},
    )


def fresh_data(root):
    obs, meta, good, bad = [], [], [], []
    labelled = {r["game"]: r for r in read(root / "labels.json")["games"]}
    for game in read(root / "collection.json")["games"]:
        folder = root / "games" / f"{game['game']:04d}"
        assert all(sha(folder / name) == digest for name, digest in game["files"].items())
        if not (folder / "windows.npz").exists():
            continue
        d, _ = load_data(folder / "windows.npz")
        assert read(folder / "labels.json") == labelled[game["game"]]
        labels = {r["row"]: r for r in labelled[game["game"]]["labels"]}
        for i, m in enumerate(d.meta):
            r = labels.get(i)
            if r:
                assert sha(r["branch_file"]) == r["branch_sha256"]
            obs.append({k: v[i] for k, v in d.obs.items()})
            meta.append(
                m | {"pair": game["pair"], "labelled": r is not None, "heldout_card": heldout(m["chain"][-1]["code"])}
            )
            good.append(r["good"] if r else -1)
            bad.append(r["bad"] if r else -1)
    return {
        "obs": to_tensors({k: np.stack([o[k] for o in obs]) for k in obs[0]}),
        "meta": meta,
        "good": torch.tensor(good),
        "bad": torch.tensor(bad),
    }


def analyze(root, pilot):
    verify_identity(root)
    assert not (root / "report.json").exists()
    collect, _, label = archive(pilot)
    selection = read(root / "selection.json")
    net = load_policy(collect.SOURCE).net.eval().requires_grad_(False)
    heads = []
    for sid, rec in enumerate(selection["heads"]):
        path = root / f"head-{sid}.pt"
        assert sha(path) == rec["sha256"]
        state = torch.load(path, weights_only=True)
        head = ActionRiskHead(state["d_model"])
        head.load_state_dict(state["state"])
        heads.append(head.eval().requires_grad_(False))
    data = fresh_data(root)
    c = cache(net, data)
    with torch.no_grad():
        scores = torch.stack([h(c["context"], c["actions"]).sigmoid() for h in heads]).mean(0)
    # Disabled candidates still report shadow scores at 0.9, but never qualify the gate.
    threshold = selection["threshold"] if selection["threshold"] is not None else 0.9
    mask = data["obs"]["action_mask"].bool()
    supported = torch.zeros_like(mask)
    allscores = torch.zeros_like(c["logits"])
    for i, m in enumerate(data["meta"]):
        pair = [m["ash"][0], next(j for j, a in enumerate(m["options"]) if a["kind"] == "pass")]
        allscores[i, pair] = scores[i]
        if mask[i].sum() == 2 and m["chain"][-1]["code"] in label.SEARCH:
            supported[i, pair] = True
    out = soft_filter(c["logits"], allscores, mask, supported, threshold=threshold)
    before, after = c["logits"].softmax(-1), out.logits.softmax(-1)
    rows = []
    for i, m in enumerate(data["meta"]):
        ash = m["ash"][0]
        rows.append(
            {
                "game": m["game"],
                "decision": m["decision"],
                "pair": m["pair"],
                "owner": m["owner"],
                "own_turn": m["player"] == m["turn_player"],
                "heldout_card": m["heldout_card"],
                "top_card": m["chain"][-1]["code"],
                "labelled": m["labelled"],
                "supported": bool(supported[i, ash]),
                "scores_ash_pass": scores[i].tolist(),
                "base_ash_probability": float(before[i, ash]),
                "shadow_ash_probability": float(after[i, ash]),
            }
        )
    atomic_json(root / "shadow-rows.json", rows)
    slices = {}
    for name, predicate in {
        "all_labelled": lambda m: m["labelled"],
        "heldout_cards": lambda m: m["labelled"] and m["heldout_card"],
        "cross_turn": lambda m: m["labelled"] and ((m["owner"] == "self") != (m["player"] == m["turn_player"])),
        "supported_labelled": lambda m: m["labelled"] and bool(mask[data["meta"].index(m)].sum() == 2),
    }.items():
        idx = [i for i, m in enumerate(data["meta"]) if predicate(m)]
        if not idx:
            slices[name] = {"n": 0}
            continue
        d = subset(data, idx)
        metrics = classification(scores[idx], d, threshold)
        for owner in ("self", "opponent"):
            js = [i for i in idx if data["meta"][i]["owner"] == owner]
            if js:
                a = torch.tensor([data["meta"][i]["ash"][0] for i in js])
                metrics[owner].update(
                    base_ash_probability=float(before[js, a].mean()), shadow_ash_probability=float(after[js, a].mean())
                )
        slices[name] = metrics
    # Cluster uncertainty by deal group for the primary labelled error/coverage metrics.
    rng = np.random.default_rng(2026100774)
    intervals = {}
    for owner in ("self", "opponent"):
        selected = [r for r in rows if r["labelled"] and r["owner"] == owner]
        pairs = sorted({r["pair"] for r in selected})
        if not pairs:
            continue
        counts = np.array(
            [
                [
                    sum(r["pair"] == p for r in selected),
                    sum(
                        r["scores_ash_pass"][1 if owner == "self" else 0] > threshold
                        for r in selected
                        if r["pair"] == p
                    ),
                    sum(
                        r["scores_ash_pass"][0 if owner == "self" else 1] > threshold
                        for r in selected
                        if r["pair"] == p
                    ),
                ]
                for p in pairs
            ]
        )
        draws = counts[rng.integers(len(pairs), size=(20000, len(pairs)))].sum(1)
        intervals[owner] = {
            "deal_groups": len(pairs),
            "good_flag_ci95": np.quantile(draws[:, 1] / draws[:, 0], [0.025, 0.975]).tolist(),
            "bad_flag_ci95": np.quantile(draws[:, 2] / draws[:, 0], [0.025, 0.975]).tolist(),
        }
    report = {
        "identity_sha256": sha(root / "identity.json"),
        "selection_sha256": sha(root / "selection.json"),
        "shadow_rows_sha256": sha(root / "shadow-rows.json"),
        "games": len(read(root / "collection.json")["games"]),
        "windows": len(rows),
        "labelled_windows": sum(r["labelled"] for r in rows),
        "threshold": selection["threshold"],
        "slices": slices,
        "cluster_intervals": intervals,
        "fresh_local_gate": selection["threshold"] is not None and gate(slices["all_labelled"], 30),
        "co_training_qualified": False,
        "limitations": "No executed filtering or strength comparison. Unlabelled actions are unknown. No independently reviewed legitimate self-negation exception set; this blocks co-training. Card holdout concerns masker fitting only. Bootstrap zero errors cannot establish zero population error.",
    }
    atomic_json(root / "report.json", report)
    print("SHADOW COMPLETE", json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot-root", type=Path, required=True)
    p.add_argument("--phase", choices=("fit", "collect", "label", "analyze", "all"), default="all")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    root, pilot = a.output.resolve(), a.pilot_root.resolve()
    try:
        for phase in ("fit", "collect", "label", "analyze"):
            if a.phase not in ("all", phase):
                continue
            if phase == "fit":
                fit(root, pilot)
            elif phase == "collect":
                collect_fresh(root, pilot, a.workers)
            elif phase == "label":
                label_fresh(root, pilot, a.workers)
            else:
                analyze(root, pilot)
    except Exception as exc:
        atomic_json(root / "STOP.json", {"phase": a.phase, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
