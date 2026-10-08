"""Read-only independent audit; bootstrap is recomputed with multiplicity weights."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit_statistics(a, scores):
    rng = np.random.default_rng(a["bootstrap_seed"])
    seeds = rng.integers(0, 3, (20000, 3))
    deals = rng.integers(0, 64, (20000, 64))
    sw = np.stack([(seeds == i).sum(1) for i in range(3)], axis=1) / 3
    dw = np.stack([(deals == i).sum(1) for i in range(64)], axis=1) / 64
    initial = scores["initial"][:3].mean(axis=(0, 2))
    for update in (128, 256, 512):
        warm = np.stack([scores[f"seed-{i}-warm-u{update}"][:3].mean(axis=(0, 2)) for i in range(3)])
        cold = np.stack([scores[f"seed-{i}-cold-u{update}"][:3].mean(axis=(0, 2)) for i in range(3)])
        for name, delta in (
            ("warm_minus_cold", warm - cold),
            ("warm_minus_initial", warm - initial),
            ("cold_minus_initial", cold - initial),
        ):
            saved = a["contrasts"][str(update)][name]
            boot = np.einsum("bi,ij,bj->b", sw, delta, dw)
            np.testing.assert_allclose(np.quantile(boot, [0.025, 0.975]), saved["crossed_ci95"], atol=1e-14)
            conditional = dw @ delta.mean(0)
            np.testing.assert_allclose(
                np.quantile(conditional, [0.025, 0.975]), saved["conditional_deal_ci95"], atol=1e-14
            )
            np.testing.assert_allclose(delta.mean(), saved["mean"], atol=1e-14)
            np.testing.assert_allclose(delta.mean(1), saved["per_training_seed"], atol=1e-14)
    for arm in ("cold", "warm"):
        for panel, selection in (("", slice(0, 3)), ("historical_", slice(3, 4))):
            delta = np.stack(
                [
                    (scores[f"seed-{i}-{arm}-u512"][selection] - scores[f"seed-{i}-{arm}-u128"][selection]).mean((0, 2))
                    for i in range(3)
                ]
            )
            saved = a["growth"][f"{arm}_{panel}512_minus_128"]
            boot = np.einsum("bi,ij,bj->b", sw, delta, dw)
            np.testing.assert_allclose(np.quantile(boot, [0.025, 0.975]), saved["crossed_ci95"], atol=1e-14)
            np.testing.assert_allclose(
                np.quantile(dw @ delta.mean(0), [0.025, 0.975]), saved["conditional_deal_ci95"], atol=1e-14
            )
            np.testing.assert_allclose(delta.mean(), saved["mean"], atol=1e-14)
            np.testing.assert_allclose(delta.mean(1), saved["per_training_seed"], atol=1e-14)
    p = a["contrasts"]["512"]
    proceed = (
        a["growth"]["warm_512_minus_128"]["crossed_ci95"][0] > 0
        and all(v > 0 for v in a["growth"]["warm_512_minus_128"]["per_training_seed"])
        and a["growth"]["warm_historical_512_minus_128"]["crossed_ci95"][0] > 0
    )
    assert proceed == a["criterion_to_test_longer_training"]
    return p, proceed


def audit(root):
    root = Path(root)
    a = json.loads((root / "analysis.json").read_text())
    assert a["replicates"] == 20000 and a["training_seeds"] == 3 and a["deal_clusters"] == 64
    assert not (root / "STOP.json").exists()
    assert a["study_sha256"] == sha(root / "identity.json")
    assert a["evaluation_report_sha256"] == sha(root / "evaluation-report.json")
    assert a["paired_scores_sha256"] == sha(root / "paired-scores.npz")
    report = json.loads((root / "evaluation-report.json").read_text())
    assert report["all_healthy"] and report["games"] == 19456
    scores = np.load(root / "paired-scores.npz")
    baselines = ("greedy", "old-256x2", "initial-128x2", "historical-rl")
    paired = {}
    games = 0
    for label in scores.files:
        assert scores[label].shape == (4, 64, 4)
        for b, opponent in enumerate(baselines):
            p = root / "evaluation" / label / f"{opponent}.jsonl"
            assert sha(p) == a["raw_sha256"][str(p.relative_to(root))]
            rows = [json.loads(s) for s in p.read_text().splitlines()]
            assert len(rows) == 256 and [r["game_id"] for r in rows] == list(range(256))
            rs = [r["result"] for r in rows]
            assert all(
                r["reason"] == "win"
                and r["winner"] in (0, 1)
                and not any(
                    r.get(k) for k in ("retries", "unknown_messages", "undecodable_messages", "script_errors", "error")
                )
                for r in rs
            )
            assert [r["pair"] for r in rs] == np.repeat(np.arange(64), 4).tolist()
            inputs = [
                {k: r[k] for k in ("agent_seeds", "deck_a", "deck_b", "config")}
                | {k: r["result"][k] for k in ("pair", "seed", "first")}
                for r in rows
            ]
            if opponent not in paired:
                paired[opponent] = inputs
            assert paired[opponent] == inputs
            actual = np.array([r["winner"] == 0 for r in rs]).reshape(64, 4)
            np.testing.assert_array_equal(actual, scores[label][b])
            cell = json.loads(p.with_suffix(".json").read_text())
            assert int(actual.sum()) == cell["wins"]
            np.testing.assert_allclose(actual.mean(), a["rates"][label][opponent], atol=1e-14)
            games += len(rows)
        np.testing.assert_allclose(scores[label][:3].mean(), a["rates"][label]["mean"], atol=1e-14)
    assert len(scores.files) == 19 and games == a["games"] == 19456
    totals = dict(updates=0, rows=0, games=0, truncated=0, errors=0)
    producer = json.loads((root / "producer-report.json").read_text())
    assert len(producer["phases"]) == len(a["training_phases"])
    for item in producer["phases"]:
        phase_root = root / f"seed-{item['seed_id']}" / item["arm"]
        assert sha(phase_root / "report.json") == item["report_sha256"]
        phase = json.loads((phase_root / "report.json").read_text())
        assert phase == a["training_phases"][f"seed-{item['seed_id']}-{item['arm']}"]
        assert phase["study_sha256"] == a["study_sha256"]
        assert phase["identity_sha256"] == sha(phase_root / "identity.json")
        assert phase["first_rollout_sha256"] == sha(phase_root / "first-rollout.pt")
        c = phase["counters"]
        assert c["updates"] == 512 and c["rows"] == 8388608 and c["errors"] == 0
        added = phase["added"]
        assert added["updates"] == 384 and added["rows"] == 6291456
        assert added["games"] < 100 or added["truncated"] / added["games"] <= 0.01
        for node in phase["nodes"]:
            assert node == json.loads((phase_root / "nodes" / f"update_{node['update']:08}.json").read_text())
            assert sha(node["checkpoint"]) == node["sha256"]
        for k in totals:
            assert phase["added"][k] == phase["counters"][k] - phase["source_counters"][k]
            totals[k] += phase["added"][k]
    assert totals["errors"] == 0 and totals["updates"] == 2304 and totals["rows"] == 37748736
    p, proceed = audit_statistics(a, scores)
    return {
        "verified_unix": time.time(),
        "analysis_sha256": sha(root / "analysis.json"),
        "auditor_sha256": sha(Path(__file__)),
        "raw_games_and_scores_match": True,
        "crossed_and_conditional_bootstrap_reproduced": True,
        "training": totals,
        "games": games,
        "criterion_to_test_longer_training": proceed,
        "primary": p,
        "growth": a["growth"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("output")
    args = parser.parse_args()
    p = Path(args.output)
    assert not p.exists()
    result = audit(args.study)
    p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
