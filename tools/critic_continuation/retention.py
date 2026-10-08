"""Fixed old-BC-task diagnostics; read only sealed checkpoints, never intervene in PPO."""

import importlib.util
import json
import time

import numpy as np
import torch

from run import NODES, ROOT, check_stop, checked_node, node_record, sha, verify_identity
from ygorl.train.checkpoint import load_actor_critic
from ygorl.train.registration import atomic_json

LEGACY = ROOT.parent / "terminal-critic-retention-2026-10-06"
spec = importlib.util.spec_from_file_location("legacy_retention", LEGACY / "run.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
OUTPUT = ROOT / "retention"


def preflight():
    report = json.loads((legacy.CONFIRM / "confirm/report.json").read_text())
    ro, _ = legacy.dataset_batch(report["files"][0])
    source = ROOT.parent / "terminal-critic-policy-long-2026-10-06/seed-0/warm/run/checkpoints/update_00000128.pt"
    model = load_actor_critic(source).model.eval()
    q, v = legacy.forward(model, ro, slice(0, 64))
    old = torch.load(LEGACY / "seed-0-warm-u128-actual.pt", weights_only=True)["prediction"]
    for name, value in (("q", q), ("v", v)):
        torch.testing.assert_close(value, old[name][:64], atol=2e-5, rtol=2e-4)
    return {"rows": 64, "matches_sealed_u128_predictions": True}


def score(sid, arm, update):
    node = checked_node(sid, arm, update)
    model = load_actor_critic(node["checkpoint"]).model.eval()
    report = json.loads((legacy.CONFIRM / "confirm/report.json").read_text())
    assert report["games"] == 256 and report["rows"] == 81546 and report["all_healthy"]
    predictions, datasets = [], []
    for item in report["files"]:
        check_stop()
        ro, data = legacy.dataset_batch(item)
        q, v = [], []
        for start in range(0, item["rows"], 128):
            qq, vv = legacy.forward(model, ro, slice(start, start + 128))
            q.append(qq)
            v.append(vv)
        predictions.append({"q": torch.cat(q), "v": torch.cat(v)})
        datasets.append(data)
        atomic_json(
            OUTPUT / "progress.json",
            {
                "stage": "scoring",
                "seed": sid,
                "arm": arm,
                "update": update,
                "batch": item["batch"],
                "updated_unix": time.time(),
            },
        )
    pred, data = legacy.core.concatenate(predictions), legacy.core.concatenate(datasets)
    assert all(len(v) == 81546 and torch.isfinite(v).all() for v in pred.values())
    label = f"seed-{sid}-{arm}-u{update}"
    target = OUTPUT / f"{label}.pt"
    assert not target.exists()
    legacy.core.save_tensor({"prediction": pred, "data": data}, target)
    atomic_json(
        OUTPUT / f"{label}.json",
        {
            "study_sha256": sha(ROOT / "identity.json"),
            "node": node,
            "predictions_sha256": sha(target),
            "metrics": legacy.core.game_metrics(pred, data),
        },
    )
    print("RETENTION", label, flush=True)


def analyze():
    rows = {}
    for sid in range(3):
        for arm in ("cold", "warm"):
            for u in NODES:
                label = f"seed-{sid}-{arm}-u{u}"
                row = json.loads((OUTPUT / f"{label}.json").read_text())
                assert row["predictions_sha256"] == sha(OUTPUT / f"{label}.pt")
                stored = torch.load(OUTPUT / f"{label}.pt", weights_only=True)
                # Recompute per-game errors from saved predictions/terminal labels, not saved scalar summaries.
                actual = legacy.core.game_metrics(stored["prediction"], stored["data"])
                assert actual == row["metrics"]
                rows[label] = row
    rng = np.random.default_rng(2026100803)
    si, di = rng.integers(0, 3, (20000, 3)), rng.integers(0, 128, (20000, 128))
    results = {}
    for arm in ("cold", "warm"):
        for head in ("q", "v"):
            delta = np.array(
                [
                    np.array(rows[f"seed-{i}-{arm}-u512"]["metrics"][head]["game_mse"])
                    - np.array(rows[f"seed-{i}-{arm}-u128"]["metrics"][head]["game_mse"])
                    for i in range(3)
                ]
            )
            delta = delta.reshape(3, 128, 2).mean(2)
            boot = delta[si[:, :, None], di[:, None, :]].mean((1, 2))
            results[f"{arm}_{head}_512_minus_128"] = {
                "mean": float(delta.mean()),
                "per_seed": delta.mean(1).tolist(),
                "crossed_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
    atomic_json(
        OUTPUT / "analysis.json",
        {
            "study_sha256": sha(ROOT / "identity.json"),
            "nodes": rows,
            "contrasts": results,
            "bootstrap_seed": 2026100803,
            "interpretation": "Fixed BC task only; not on-policy calibration or proof of harmful forgetting",
        },
    )


def main():
    torch.set_num_threads(1)
    verify_identity()
    OUTPUT.mkdir(exist_ok=True)
    deadline = time.monotonic() + 48 * 3600
    done = set()
    while len(done) < 18:
        check_stop()
        assert time.monotonic() < deadline
        for sid in range(3):
            for arm in ("cold", "warm"):
                for u in NODES:
                    job = (sid, arm, u)
                    if job not in done and node_record(*job).exists():
                        score(*job)
                        done.add(job)
        atomic_json(OUTPUT / "progress.json", {"stage": "waiting", "completed": len(done), "updated_unix": time.time()})
        if len(done) < 18:
            time.sleep(30)
    verify_identity()
    analyze()
    atomic_json(OUTPUT / "progress.json", {"stage": "complete", "completed": 18, "updated_unix": time.time()})


if __name__ == "__main__":
    main()
