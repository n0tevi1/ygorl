"""Frozen numeric/context residual controls followed by fresh native confirmation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import battle_relation_controls as common
from mine_action_outcomes import read, sha
from ygorl import _core
from ygorl.data import load_environment
from ygorl.nets.battle_residual import BattleResidualHead
from ygorl.train.registration import atomic_json


def development(protocol):
    chunks, meta = [], []
    for pool, path in enumerate(protocol["development_roots"]):
        data, rows = common.dataset(Path(path))
        chunks.append(data)
        meta.extend({**m, "pool": pool} for m in rows)
    return {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}, meta


def features(data):
    context = common.inputs(data, True)
    numeric = context.clone()
    numeric[:, :384] = 0
    return numeric, context


def build(arm, numeric=None):
    if arm == "numeric":
        return common.head()
    if arm.startswith("residual"):
        return BattleResidualHead(common.head() if numeric is None else numeric)
    return torch.nn.Sequential(torch.nn.Linear(396, 64), torch.nn.ReLU(), torch.nn.Linear(64, 9))


def forward(model, arm, numeric, context):
    if arm == "numeric":
        return model(numeric), None
    if arm.startswith("residual"):
        return model(numeric, context)
    out = model(context)
    return out[:, :8], out[:, 8]


def outcome_loss(out, damage, destroy):
    return (
        F.cross_entropy(out[:, :4], common.direction(damage), reduction="none")
        + F.smooth_l1_loss(out[:, 4:6], damage / 4000, reduction="none").mean(-1)
        + F.binary_cross_entropy_with_logits(out[:, 6:], destroy, reduction="none").mean(-1)
    )


def fit(root, p):
    assert not (root / "identity.json").exists(), "preserve fit"
    data, meta = development(p)
    ids = torch.tensor([i for i, m in enumerate(meta) if m["pair"] % 10 < 6 and not m["heldout"]])
    val = torch.tensor([i for i, m in enumerate(meta) if 6 <= m["pair"] % 10 < 8 and not m["heldout"]])
    exceptional = torch.tensor([not m["baseline_agrees"] for m in meta])
    count = torch.bincount(exceptional[ids].long(), minlength=2)
    assert count.min() > 0
    weights = len(ids) / (2 * count.float())
    pos_weight = count[0] / count[1]
    xn, xc = features(data)
    norms = {}
    for name, x in [("numeric", xn), ("context", xc)]:
        norms[name + "_mean"] = x[ids].mean(0)
        norms[name + "_scale"] = x[ids].std(0, correction=0).clamp_min(0.001)
    xn = (xn - norms["numeric_mean"]) / norms["numeric_scale"]
    xc = (xc - norms["context_mean"]) / norms["context_scale"]
    files = [
        Path(__file__),
        Path("tools/battle_relation_controls.py"),
        Path("tools/battle_outcome_probe.py"),
        Path("tools/action_filter_pilot.py"),
        Path("tools/review_action_outcomes.py"),
        Path("src/ygorl/eval/battle_outcomes.py"),
        Path("src/ygorl/nets/battle_residual.py"),
        Path(_core.__file__),
        Path(p["source"]),
        root / "protocol.json",
    ]
    files += [Path(v) / "collection.json" for v in p["development_roots"]]
    pilot = Path("out/research/ash-correction-pilot-2026-10-07")
    files += [pilot / v for v in ["collect.py", "fit.py", "label.py"]]
    identity = {
        "bindings": {str(f.resolve()): sha(f) for f in files},
        "environment": load_environment("md-2026-09").stamp(),
        "train": len(ids),
        "validation": len(val),
        "development_total": len(meta),
        "training_deviation_counts": count.tolist(),
        "deviation_weights": weights.tolist(),
        "training_damage_deviations": sum(
            meta[i]["baseline"]["damage"] != meta[i]["damage_relative"] for i in ids.tolist()
        ),
        "validation_deviations": int(exceptional[val].sum()),
    }
    atomic_json(root / "identity.json", identity)
    atomic_json(root / "development-index.json", meta)
    selection, diagnostics = [], []
    for seed in p["fit_seeds"]:
        numeric_base = None
        for arm in p["arms"]:
            torch.manual_seed(seed)
            model = build(arm, numeric_base)
            initial = common.state_digest(model.state_dict())
            optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad], lr=p["learning_rate"])
            rng = torch.Generator().manual_seed(seed + 1)
            best, nodes = None, []
            for step in range(p["steps"] + 1):
                if step:
                    ix = ids[torch.randint(len(ids), (128,), generator=rng)]
                    out, gate = forward(model, arm, xn[ix], xc[ix])
                    per_row = outcome_loss(out, data["damage"][ix], data["destroy"][ix])
                    w = weights[exceptional[ix].long()] if arm.endswith("balanced") else torch.ones_like(per_row)
                    loss = (per_row * w).sum() / w.sum()
                    if gate is not None:
                        loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                            gate, exceptional[ix].float(), pos_weight=pos_weight
                        )
                    assert torch.isfinite(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                if step in p["nodes"]:
                    with torch.no_grad():
                        vo, _ = forward(model, arm, xn[val], xc[val])
                        score = float(outcome_loss(vo, data["damage"][val], data["destroy"][val]).mean())
                    nodes.append({"step": step, "validation_unweighted_outcome_loss": score})
                    if best is None or score < best[0]:
                        best = score, step, copy.deepcopy(model.state_dict())
            model.load_state_dict(best[2])
            if arm == "numeric":
                numeric_base = copy.deepcopy(model)
            if arm.startswith("residual"):
                assert common.state_digest(model.numeric.state_dict()) == common.state_digest(numeric_base.state_dict())
            path = root / f"{arm}-{seed}.pt"
            torch.save({"model": best[2], **norms, "arm": arm, "seed": seed, "selected_step": best[1]}, path)
            selection.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "selected_step": best[1],
                    "nodes": nodes,
                    "initial_sha256": initial,
                    "sha256": sha(path),
                }
            )
            with torch.no_grad():
                diag = {}
                for split, indices in [("train", ids), ("validation", val)]:
                    out, gate = forward(model, arm, xn[indices], xc[indices])
                    diag[split] = metrics(
                        out, gate, data["damage"][indices], data["destroy"][indices], exceptional[indices]
                    )
            diagnostics.append({"arm": arm, "seed": seed, "metrics": diag})
            print("FIT", arm, seed, "selected", best[1], flush=True)
    for seed in p["fit_seeds"]:
        rows = [s for s in selection if s["seed"] == seed and s["arm"].startswith("residual")]
        assert len({s["initial_sha256"] for s in rows}) == 1
    atomic_json(root / "selection.json", selection)
    atomic_json(root / "development-diagnostics.json", diagnostics)
    print("FROZEN", len(selection), "candidates", flush=True)


def metrics(out, gate, damage, destroy, exceptional):
    result = common.extra_metrics(out, damage, destroy)
    if gate is not None:
        predicted = gate > 0
        result["deviation_detection"] = {
            "positive_n": int(exceptional.sum()),
            "recall": float(predicted[exceptional].float().mean()) if exceptional.any() else None,
            "false_positive_rate": float(predicted[~exceptional].float().mean()) if (~exceptional).any() else None,
            "threshold": 0.5,
        }
    return result


def evaluate(root, p):
    assert not (root / "report.json").exists(), "preserve confirmation"
    assert all(sha(k) == v for k, v in read(root / "identity.json")["bindings"].items())
    data, meta = common.dataset(root / "confirmation")
    xn, xc = features(data)
    ex = torch.tensor([not m["baseline_agrees"] for m in meta])
    groups = {"all": list(range(len(meta)))}
    predicates = {
        "ordinary": lambda m: m["baseline_agrees"],
        "exceptions": lambda m: not m["baseline_agrees"],
        "damage_exceptions": lambda m: m["baseline"]["damage"] != m["damage_relative"],
        "self_damage_exceptions": lambda m: (
            m["baseline"]["damage"] != m["damage_relative"] and m["damage_relative"][0] > 0
        ),
        "heldout_pairs": lambda m: m["heldout"],
        "monster": lambda m: not m["direct"],
        "direct": lambda m: m["direct"],
    }
    for k, predicate in predicates.items():
        groups[k] = [i for i, m in enumerate(meta) if predicate(m)]
    results, predictions, gates = [], [], []
    for s in read(root / "selection.json"):
        path = root / f"{s['arm']}-{s['seed']}.pt"
        assert sha(path) == s["sha256"]
        state = torch.load(path, weights_only=True)
        model = build(s["arm"])
        model.load_state_dict(state["model"])
        with torch.no_grad():
            out, gate = forward(
                model,
                s["arm"],
                (xn - state["numeric_mean"]) / state["numeric_scale"],
                (xc - state["context_mean"]) / state["context_scale"],
            )
        predictions.append(out.numpy())
        gates.append(np.full(len(meta), np.nan) if gate is None else gate.numpy())
        results.append(
            {
                **s,
                "metrics": {
                    k: metrics(out[v], None if gate is None else gate[v], data["damage"][v], data["destroy"][v], ex[v])
                    for k, v in groups.items()
                    if v
                },
            }
        )
    np.savez_compressed(
        root / "predictions.npz",
        outputs=np.stack(predictions),
        gates=np.stack(gates),
        damage=data["damage"].numpy(),
        destroy=data["destroy"].numpy(),
    )
    atomic_json(root / "confirmation-index.json", meta)
    atomic_json(
        root / "report.json",
        {
            "selection_sha256": sha(root / "selection.json"),
            "collection_sha256": sha(root / "confirmation/collection.json"),
            "predictions_sha256": sha(root / "predictions.npz"),
            "groups": {k: len(v) for k, v in groups.items()},
            "results": results,
        },
    )
    print("CONFIRM COMPLETE", {k: len(v) for k, v in groups.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["fit", "fresh", "evaluate", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve()
    p = read(root / "protocol.json")
    torch.set_num_threads(1)
    try:
        if args.phase in ("fit", "all"):
            fit(root, p)
        if args.phase in ("fresh", "all"):
            common.fresh(root, p, args.workers)
        if args.phase in ("evaluate", "all"):
            evaluate(root, p)
    except Exception as exc:
        atomic_json(root / "STOP.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
