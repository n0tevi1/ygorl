"""Fixed update512 endpoint and paired 512-minus-128 growth; paired training-seed and deal-cluster uncertainty."""

import json

import numpy as np

from run import BASELINES, ROOT, check_stop, checked_node, phase_root, sha, verify_identity
from ygorl.eval.arena import GameRecord
from ygorl.train.registration import atomic_json


def validate_existing():
    data = json.loads((ROOT / "analysis.json").read_text())
    assert data["study_sha256"] == sha(ROOT / "identity.json")
    assert data["evaluation_report_sha256"] == sha(ROOT / "evaluation-report.json")
    assert data["paired_scores_sha256"] == sha(ROOT / "paired-scores.npz")
    for relative, digest in data["raw_sha256"].items():
        assert sha(ROOT / relative) == digest
    for sid in range(3):
        for arm in ("cold", "warm"):
            root = phase_root(sid, arm)
            phase = json.loads((root / "report.json").read_text())
            assert phase == data["training_phases"][f"seed-{sid}-{arm}"]
            assert phase["identity_sha256"] == sha(root / "identity.json")
            for node in phase["nodes"]:
                assert checked_node(sid, arm, node["update"]) == node
            assert sha(root / "first-rollout.pt") == phase["first_rollout_sha256"]


def statistics(scores):
    rng = np.random.default_rng(2026100805)
    seed_indices = rng.integers(0, 3, (20000, 3))
    deal_indices = rng.integers(0, 64, (20000, 64))

    def contrast(delta):
        assert delta.shape == (3, 64)
        crossed = delta[seed_indices[:, :, None], deal_indices[:, None, :]].mean(axis=(1, 2))
        conditional = delta.mean(0)[deal_indices].mean(1)
        return {
            "mean": float(delta.mean()),
            "per_training_seed": delta.mean(1).tolist(),
            "crossed_ci95": np.quantile(crossed, [0.025, 0.975]).tolist(),
            "conditional_deal_ci95": np.quantile(conditional, [0.025, 0.975]).tolist(),
        }

    nodes = {}
    for u in (128, 256, 512):
        warm = np.stack([scores[f"seed-{sid}-warm-u{u}"][:3].mean(axis=(0, 2)) for sid in range(3)])
        cold = np.stack([scores[f"seed-{sid}-cold-u{u}"][:3].mean(axis=(0, 2)) for sid in range(3)])
        initial = scores["initial"][:3].mean(axis=(0, 2))
        nodes[u] = {
            "warm_minus_cold": contrast(warm - cold),
            "warm_minus_initial": contrast(warm - initial),
            "cold_minus_initial": contrast(cold - initial),
        }
    growth = {}
    for arm in ("cold", "warm"):
        delta = np.stack(
            [
                (scores[f"seed-{sid}-{arm}-u512"][:3] - scores[f"seed-{sid}-{arm}-u128"][:3]).mean(axis=(0, 2))
                for sid in range(3)
            ]
        )
        growth[f"{arm}_512_minus_128"] = contrast(delta)
        history_delta = np.stack(
            [(scores[f"seed-{sid}-{arm}-u512"][3] - scores[f"seed-{sid}-{arm}-u128"][3]).mean(1) for sid in range(3)]
        )
        growth[f"{arm}_historical_512_minus_128"] = contrast(history_delta)
    proceed = (
        growth["warm_512_minus_128"]["crossed_ci95"][0] > 0
        and all(v > 0 for v in growth["warm_512_minus_128"]["per_training_seed"])
        and growth["warm_historical_512_minus_128"]["crossed_ci95"][0] > 0
    )
    return nodes, growth, proceed


def main():
    verify_identity()
    check_stop()
    if (ROOT / "analysis.json").exists():
        validate_existing()
        print("EXISTING ANALYSIS VERIFIED", flush=True)
        return
    report = json.loads((ROOT / "evaluation-report.json").read_text())
    assert report["all_healthy"] and report["games"] == 19456
    assert report["study_sha256"] == sha(ROOT / "identity.json")
    expected_inputs = {}
    scores, rates, counts = {}, {}, {}
    raw_digests = {}
    for label, cells in report["cells"].items():
        assert len(cells) == len(BASELINES) == 4
        cell_scores = []
        rates[label], counts[label] = {}, {}
        for baseline, cell in zip(BASELINES, cells, strict=True):
            raw = ROOT / "evaluation" / label / f"{baseline}.jsonl"
            assert sha(raw) == cell["raw_sha256"]
            raw_digests[str(raw.relative_to(ROOT))] = cell["raw_sha256"]
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            assert len(rows) == 256 and [r["game_id"] for r in rows] == list(range(256))
            records = [GameRecord(**r["result"]) for r in rows]
            assert all(r.healthy and r.reason == "win" and r.winner in (0, 1) for r in records)
            assert [r.pair for r in records] == list(np.repeat(np.arange(64), 4))
            common = [
                {k: r[k] for k in ("agent_seeds", "deck_a", "deck_b", "config")}
                | {k: r["result"][k] for k in ("pair", "seed", "first")}
                for r in rows
            ]
            if baseline not in expected_inputs:
                expected_inputs[baseline] = common
            assert common == expected_inputs[baseline], "paired game inputs differ across candidates"
            values = np.array([r.winner == 0 for r in records], dtype=float).reshape(64, 4)
            assert int(values.sum()) == cell["wins"]
            cell_scores.append(values)
            rates[label][baseline] = float(values.mean())
            counts[label][baseline] = cell["candidate_counts"]
        scores[label] = np.stack(cell_scores)  # fixed opponent, deal, four seat/deck assignments
        rates[label]["mean"] = float(scores[label][:3].mean())
    assert len(scores) == 19
    phases = {}
    for sid in range(3):
        for arm in ("cold", "warm"):
            root = phase_root(sid, arm)
            phase = json.loads((root / "report.json").read_text())
            assert phase["study_sha256"] == sha(ROOT / "identity.json")
            assert phase["identity_sha256"] == sha(root / "identity.json")
            assert phase["counters"]["updates"] == 512 and phase["counters"]["rows"] == 8388608
            for node in phase["nodes"]:
                assert checked_node(sid, arm, node["update"]) == node
            assert sha(root / "first-rollout.pt") == phase["first_rollout_sha256"]
            phases[f"seed-{sid}-{arm}"] = phase
    nodes, growth, proceed = statistics(scores)
    result = {
        "study_sha256": sha(ROOT / "identity.json"),
        "evaluation_report_sha256": sha(ROOT / "evaluation-report.json"),
        "raw_sha256": raw_digests,
        "rates": rates,
        "contrasts": nodes,
        "growth": growth,
        "candidate_counts": counts,
        "training_phases": phases,
        "criterion_to_test_longer_training": proceed,
        "bootstrap_seed": 2026100805,
        "replicates": 20000,
        "training_seeds": 3,
        "deal_clusters": 64,
        "games": 19456,
        "limitations": [
            "Three PPO seeds and critic initializations share one supervised corpus; training-data uncertainty is not covered",
            "Equal PPO budgets; warm uses a shared 512-game corpus and one supervised epoch per critic; discovery costs excluded from PPO budget",
            "Restored optimizer, reference, pool, counters and RNG; active duels restart",
            "Small fixed opponent set in the training deck pool; not a top-tier agent evaluation",
            "Update128 fixed growth comparator; fixed seed0 historical RL opponent; no checkpoint selection",
            "Material cancellation counters depend on each policy's visited states",
        ],
    }
    assert not (ROOT / "analysis.json").exists()
    np.savez_compressed(ROOT / "paired-scores.npz", **scores)
    result["paired_scores_sha256"] = sha(ROOT / "paired-scores.npz")
    atomic_json(ROOT / "analysis.json", result)
    print(json.dumps({"rates": rates, "contrasts": nodes, "growth": growth, "continue": proceed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
