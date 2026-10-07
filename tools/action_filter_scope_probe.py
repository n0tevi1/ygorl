"""Paired shadow scope ablation using frozen scores on already opened panels.

Reconstruct actual actor logits; do not infer missing probabilities from rounded
stored softmax values. Compare dual-candidate and Ash-only penalties without fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ygorl.nets.action_filter import soft_filter
from ygorl.train.checkpoint import load_policy
from ygorl.train.registration import atomic_json

# This script is run from tools/, making the archived-data reader available here.
from action_filter_pilot import cache, fresh_data


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@torch.no_grad()
def compare(panel):
    report = read(panel / "report.json")
    assert report["shadow_rows_sha256"] == sha(panel / "shadow-rows.json")
    recorded = read(panel / "shadow-rows.json")
    keyed = {(r["game"], r["decision"]): r for r in recorded}
    data = fresh_data(panel)
    identity = read(panel / "identity.json")
    net = load_policy(identity["source"]).net.eval()
    cached = cache(net, data)
    logits = cached["logits"]
    legal = data["obs"]["action_mask"].bool()
    scores = torch.zeros_like(logits)
    both, single = torch.zeros_like(legal), torch.zeros_like(legal)
    ash_ids, rows = [], []
    for i, m in enumerate(data["meta"]):
        r = keyed[(m["game"], m["decision"])]
        pair = [m["ash"][0], next(j for j, a in enumerate(m["options"]) if a["kind"] == "pass")]
        scores[i, pair] = torch.tensor(r["scores_ash_pass"])
        if r["supported"]:
            assert legal[i].sum() == 2
            both[i, pair] = True
            single[i, pair[0]] = True
        ash_ids.append(pair[0])
        rows.append(r)
    ids = torch.arange(len(rows))
    threshold = report["threshold"]
    assert threshold is not None
    original = logits.softmax(-1)[ids, ash_ids]
    dual = soft_filter(logits, scores, legal, both, threshold=threshold).logits.softmax(-1)[ids, ash_ids]
    one = soft_filter(logits, scores, legal, single, threshold=threshold).logits.softmax(-1)[ids, ash_ids]
    err = max(abs(float(dual[i]) - r["shadow_ash_probability"]) for i, r in enumerate(rows))
    assert err < 2e-6, err
    assert (one <= original + 1e-7).all()
    output = {
        "panel": str(panel),
        "report_sha256": sha(panel / "report.json"),
        "windows": len(rows),
        "dual_reconstruction_max_error": err,
        "only_ash_max_probability_increase": float((one - original).max()),
        "supported_windows": sum(r["supported"] for r in rows),
        "supported_unlabelled_windows": sum(r["supported"] and not r["labelled"] for r in rows),
        "groups": {},
    }
    for owner in ("self", "opponent"):
        idx = [i for i, r in enumerate(rows) if r["labelled"] and r["owner"] == owner]
        output["groups"][owner] = {
            "n": len(idx),
            "original": float(original[idx].mean()),
            "ash_and_pass": float(dual[idx].mean()),
            "ash_only": float(one[idx].mean()),
        }
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", action="append", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    assert not a.output.exists(), "preserve prior report"
    rows = [compare(p.resolve()) for p in a.panel]
    result = {
        "driver_sha256": sha(__file__),
        "panels": rows,
        "interpretation": "Post-hoc scope ablation; no executed filtering or refitting. Monotonicity is a distribution property, not proof that suppressing Ash is correct.",
    }
    atomic_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
