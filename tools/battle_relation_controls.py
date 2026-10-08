"""Frozen 2x2 relation/class-weight controls with new unmodified-policy games."""

from __future__ import annotations

import argparse
import copy
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import action_filter_pilot as panel_tools
import battle_outcome_probe as probe
from mine_action_outcomes import read, sha
from ygorl import _core
from ygorl.data import load_environment
from ygorl.eval.battle_outcomes import battle_relation_features
from ygorl.train.registration import atomic_json


def dataset(root):
    meta, chunks = [], []
    for g in read(root / "collection.json")["games"]:
        base = root / "games" / f"{g['game']:04d}"
        assert all(sha(base / n) == digest for n, digest in g["files"].items())
        if g["eligible"]:
            with np.load(base / "features.npz") as f:
                chunks.append({k: torch.from_numpy(f[k].copy()) for k in f.files})
            meta.extend(m for m in read(base / "rows.json") if m["eligible"])
    return {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}, meta


def inputs(data, relational):
    relations = torch.tensor([battle_relation_features(v) for v in data["numeric"].tolist()])
    return torch.cat([data["pair"], data["numeric"], relations if relational else torch.zeros_like(relations)], -1)


def head(width=396):
    return torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))


def direction(damage):
    return (damage[:, 0] > 0).long() + 2 * (damage[:, 1] > 0).long()


def objective(out, damage, destroy, weights=None):
    return (
        F.cross_entropy(out[:, :4], direction(damage), weight=weights)
        + F.smooth_l1_loss(out[:, 4:6], damage / 4000)
        + F.binary_cross_entropy_with_logits(out[:, 6:], destroy)
    )


def state_digest(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def train(root, p):
    parent = Path(p["parent"]).resolve()
    assert not (root / "identity.json").exists(), "preserve experiment"
    data, meta = dataset(parent)
    train_ids = torch.tensor([i for i, m in enumerate(meta) if m["pair"] % 10 < 6 and not m["heldout"]])
    val_ids = torch.tensor([i for i, m in enumerate(meta) if 6 <= m["pair"] % 10 < 8 and not m["heldout"]])
    assert (len(train_ids), len(val_ids)) == (558, 177)
    counts = torch.bincount(direction(data["damage"][train_ids]), minlength=4)
    present = counts > 0
    class_weights = torch.ones(4)
    class_weights[present] = len(train_ids) / (present.sum() * counts[present].float())
    bindings = {
        str(path.resolve()): sha(path)
        for path in [
            Path(__file__),
            Path("tools/battle_outcome_probe.py"),
            Path("tools/action_filter_pilot.py"),
            Path("tools/review_action_outcomes.py"),
            Path("src/ygorl/eval/battle_outcomes.py"),
            Path(p["source"]),
            Path(_core.__file__),
            root / "protocol.json",
            parent / "collection.json",
        ]
    }
    pilot = Path("out/research/ash-correction-pilot-2026-10-07").resolve()
    for name in ("collect.py", "fit.py", "label.py"):
        bindings[str(pilot / name)] = sha(pilot / name)
    atomic_json(
        root / "identity.json",
        {
            "bindings": bindings,
            "environment": load_environment("md-2026-09").stamp(),
            "train": len(train_ids),
            "val": len(val_ids),
            "direction_counts": counts.tolist(),
            "class_weights": class_weights.tolist(),
        },
    )
    selections = []
    for arm in p["arms"]:
        x = inputs(data, arm.startswith("relations"))
        mean, scale = x[train_ids].mean(0), x[train_ids].std(0, correction=0).clamp_min(0.001)
        x = (x - mean) / scale
        weights = class_weights if arm.endswith("balanced") else None
        for seed in p["fit_seeds"]:
            torch.manual_seed(seed)
            model = head()
            init = state_digest(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            rng = torch.Generator().manual_seed(seed + 1)
            sampler_hash = hashlib.sha256()
            best, nodes = None, []
            for step in range(1, 257):
                idx = train_ids[torch.randint(len(train_ids), (128,), generator=rng)]
                sampler_hash.update(idx.numpy().tobytes())
                optimizer.zero_grad()
                loss = objective(model(x[idx]), data["damage"][idx], data["destroy"][idx], weights)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if step in (64, 128, 256):
                    with torch.no_grad():
                        val = float(objective(model(x[val_ids]), data["damage"][val_ids], data["destroy"][val_ids]))
                    nodes.append({"step": step, "validation_unweighted_loss": val})
                    if best is None or val < best[0]:
                        best = (val, step, copy.deepcopy(model.state_dict()))
            path = root / f"{arm}-{seed}.pt"
            torch.save(
                {
                    "head": best[2],
                    "mean": mean,
                    "scale": scale,
                    "arm": arm,
                    "seed": seed,
                    "selected_step": best[1],
                    "class_weights": weights,
                },
                path,
            )
            selections.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "selected_step": best[1],
                    "nodes": nodes,
                    "initial_state_sha256": init,
                    "sampling_sha256": sampler_hash.hexdigest(),
                    "sha256": sha(path),
                }
            )
    for seed in p["fit_seeds"]:
        for key in ("initial_state_sha256", "sampling_sha256"):
            assert len({s[key] for s in selections if s["seed"] == seed}) == 1
    atomic_json(root / "selection.json", selections)
    print("FIT FROZEN", len(selections), "heads", flush=True)


def fresh(root, p, workers):
    bindings = read(root / "identity.json")["bindings"]
    assert all(sha(k) == v for k, v in bindings.items()), "registered input changed"
    assert all(sha(root / f"{s['arm']}-{s['seed']}.pt") == s["sha256"] for s in read(root / "selection.json"))
    out = root / "fresh-panel"
    out.mkdir(exist_ok=False)
    atomic_json(out / "protocol.json", p)
    atomic_json(
        out / "identity.json",
        {
            "source": p["source"],
            "protocol": p,
            "environment": load_environment("md-2026-09").stamp(),
            "frozen_selection_sha256": sha(root / "selection.json"),
            "parent_identity_sha256": sha(root / "identity.json"),
        },
    )
    pilot = Path("out/research/ash-correction-pilot-2026-10-07").resolve()
    archived, _, _ = panel_tools.archive(pilot)
    assert sha(archived.SOURCE) == sha(p["source"])
    specs = panel_tools.specs(out)
    assert len(specs) == p["new_games"]
    atomic_json(
        out / "panel-identity.json",
        {
            "specs": [archived.spec_json(s) for s in specs],
            "frozen_selection_sha256": sha(root / "selection.json"),
            "executed_policy": "unmodified source actor",
        },
    )
    records = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for rec in pool.imap_unordered(panel_tools.collect_game, [(out, pilot, i, s) for i, s in enumerate(specs)]):
            records.append(rec)
            atomic_json(out / "progress.json", {"done": len(records), "total": len(specs)})
            if len(records) % 16 == 0:
                print("FRESH GAMES", len(records), "/", len(specs), flush=True)
    atomic_json(
        out / "collection.json",
        {"games": sorted(records, key=lambda x: x["game"]), "panel_identity_sha256": sha(out / "panel-identity.json")},
    )
    dest = root / "confirmation"
    dest.mkdir(exist_ok=False)
    cp = {
        "panel": str(out),
        "source": p["source"],
        "game_ids": [0, len(specs) - 1],
        "stage": "new frozen-source outcome confirmation; no fitting",
    }
    atomic_json(dest / "protocol.json", cp)
    probe.collect(dest, cp, workers)


def extra_metrics(out, damage, destroy):
    result = probe.metrics(out, damage, destroy)
    actual = damage[:, 0] > 0
    pred = (out[:, :4].argmax(-1) & 1).bool()
    result.update(
        self_damage_n=int(actual.sum()),
        self_damage_recall=float(pred[actual].float().mean()) if actual.any() else None,
        self_damage_false_positive_rate=float(pred[~actual].float().mean()) if (~actual).any() else None,
        self_damage_false_positive_count=int((pred & ~actual).sum()),
    )
    return result


def evaluate(root, p):
    assert not (root / "report.json").exists(), "preserve report"
    data, meta = dataset(root / "confirmation")
    models, outputs = read(root / "selection.json"), []
    groups = {"all": list(range(len(meta)))}
    for name, check in {
        "heldout_pairs": lambda m: m["heldout"],
        "monster": lambda m: not m["direct"],
        "direct": lambda m: m["direct"],
        "ordinary_agreement": lambda m: m["baseline_agrees"],
        "exceptions": lambda m: not m["baseline_agrees"],
    }.items():
        groups[name] = [i for i, m in enumerate(meta) if check(m)]
    results = []
    for rec in models:
        path = root / f"{rec['arm']}-{rec['seed']}.pt"
        assert sha(path) == rec["sha256"]
        state = torch.load(path, weights_only=True)
        model = head()
        model.load_state_dict(state["head"])
        with torch.no_grad():
            out = model((inputs(data, rec["arm"].startswith("relations")) - state["mean"]) / state["scale"])
        outputs.append(out.numpy())
        results.append(
            {
                "arm": rec["arm"],
                "seed": rec["seed"],
                "selected_step": rec["selected_step"],
                "metrics": {
                    k: extra_metrics(out[v], data["damage"][v], data["destroy"][v]) for k, v in groups.items() if v
                },
            }
        )
    pred = np.stack(outputs)
    np.savez_compressed(
        root / "confirmation-predictions.npz",
        outputs=pred,
        damage=data["damage"].numpy(),
        destroy=data["destroy"].numpy(),
    )
    atomic_json(root / "confirmation-index.json", meta)
    baseline = torch.zeros(len(meta), 8)
    for i, m in enumerate(meta):
        b = m["baseline"]
        cls = int(b["damage"][0] > 0) + 2 * int(b["damage"][1] > 0)
        baseline[i, cls] = 1
        baseline[i, 4:6] = torch.tensor(b["damage"]) / 4000
        baseline[i, 6:] = torch.tensor(b["destroy"]) * 2 - 1
    report = {
        "selection_sha256": sha(root / "selection.json"),
        "confirmation_collection_sha256": sha(root / "confirmation/collection.json"),
        "confirmation_prediction_sha256": sha(root / "confirmation-predictions.npz"),
        "groups": {k: len(v) for k, v in groups.items()},
        "results": results,
        "ordinary_baseline": {
            k: extra_metrics(baseline[v], data["damage"][v], data["destroy"][v]) for k, v in groups.items() if v
        },
    }
    atomic_json(root / "report.json", report)
    print("CONFIRM COMPLETE", report["groups"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("fit", "fresh", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    a = parser.parse_args()
    root = a.output.resolve()
    p = read(root / "protocol.json")
    torch.set_num_threads(1)
    try:
        if a.phase in ("fit", "all"):
            train(root, p)
        if a.phase in ("fresh", "all"):
            fresh(root, p, a.workers)
        if a.phase in ("evaluate", "all"):
            evaluate(root, p)
    except Exception as exc:
        atomic_json(root / "STOP.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
