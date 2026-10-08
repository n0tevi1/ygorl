"""The precision pilot retains partial evidence, pairs deal clusters, and never auto-promotes."""

import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
pe = importlib.import_module("precision_evaluate")


def rows(wins_per_cluster=2):
    return [
        {
            "game_id": i,
            "agent_seeds": [2 * (i // 4), 2 * (i // 4) + 1],
            "deck_a": {"name": "a"},
            "deck_b": {"name": "b"},
            "config": {"shuffle_decks": False},
            "agent_a": "policy:test.pt",
            "agent_b": "greedy",
            "candidate_counts": {"material_cancel_chosen": 0},
            "result": asdict(
                pe.GameRecord(
                    pair=i // 4,
                    first=i % 2,
                    seed=i // 4,
                    winner=int(i % 4 >= wins_per_cluster),
                    reason="win",
                    turns=4,
                    decisions=20,
                    win_reason=1,
                )
            ),
        }
        for i in range(256)
    ]


def save_cell(path, data, ident=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".jsonl")
    raw.write_text("".join(json.dumps(row) + "\n" for row in data))
    wins = sum(r["result"]["winner"] == 0 for r in data)
    pe.atomic(
        path,
        {
            "identity": ident or {},
            "raw_sha256": pe.sha(raw),
            "games": len(data),
            "wins": wins,
            "win_rate": wins / len(data),
            "all_healthy": True,
        },
    )


def test_crossed_bootstrap_retains_seed_uncertainty_and_shared_clusters():
    constant = pe.crossed_bootstrap(np.full((3, 64), 0.25), seed=7, reps=2000)
    assert constant["mean"] == 0.25 and constant["ci95"] == [0.25, 0.25]
    seed_effects = np.repeat(np.array([-1, 0, 1])[:, None], 64, axis=1)
    result = pe.crossed_bootstrap(seed_effects, seed=7, reps=20000)
    assert result["mean"] == 0 and result["per_seed"] == [-1, 0, 1]
    assert result["ci95"] == [-1, 1]  # 64 deal clusters cannot manufacture 192 independent seed replicates
    shared = np.tile(np.r_[np.zeros(32), np.ones(32)], (3, 1))
    expected_rng = np.random.default_rng(7)
    expected_rng.integers(3, size=(2000, 3))
    cluster_ids = expected_rng.integers(64, size=(2000, 64))
    expected = np.quantile(shared[0, cluster_ids].mean(axis=1), [0.025, 0.975])
    result = pe.crossed_bootstrap(shared, seed=7, reps=2000)
    np.testing.assert_array_equal(result["ci95"], expected)


@pytest.mark.parametrize("corruption", ["hash", "duplicate", "unhealthy", "score", "identity", "partial"])
def test_completed_cell_rejects_corruption(tmp_path, corruption):
    target = tmp_path / "cell.json"
    data = rows()
    if corruption == "duplicate":
        data[-1]["game_id"] = 0
    if corruption == "unhealthy":
        data[-1]["result"]["script_errors"] = 1
    save_cell(target, data, {"run": "expected"})
    raw = target.with_suffix(".jsonl")
    if corruption in ("hash", "partial"):
        raw.write_text(raw.read_text().rstrip("\n"))
        if corruption == "partial":
            report = pe.read_json(target)
            report["raw_sha256"] = pe.sha(raw)
            pe.atomic(target, report)
    if corruption == "score":
        report = pe.read_json(target)
        report["wins"] = 0
        pe.atomic(target, report)
    with pytest.raises(ValueError):
        pe.read_cell(target, {"run": "changed" if corruption == "identity" else "expected"})


class Pool:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    def imap(self, fn, jobs, chunksize):
        self.calls += 1
        assert len(jobs) == 256 and chunksize == 1
        return iter(self.data)


def cell_setup(root, monkeypatch):
    pe.atomic(root / "identity.json", {"study": 1})
    pe.atomic(root / "evaluator/identity.json", {"local": 1})
    checkpoint = root / "candidate.pt"
    checkpoint.write_bytes(b"frozen")
    monkeypatch.setattr(pe, "agent_factory", lambda spec: spec)
    monkeypatch.setattr(pe, "cell_specs", lambda *a: [None] * 256)
    return {"checkpoint": str(checkpoint), "sha256": pe.sha(checkpoint)}, {
        "name": "greedy",
        "spec": "greedy",
        "checkpoint_sha256": "",
    }


def test_completed_cell_resumes_without_reroll(tmp_path, monkeypatch):
    candidate, opponent = cell_setup(tmp_path, monkeypatch)
    pool = Pool(rows())
    first = pe.evaluate_cell(tmp_path, pool, candidate, "seed-0-fp32", opponent, [], None, None)
    second = pe.evaluate_cell(tmp_path, pool, candidate, "seed-0-fp32", opponent, [], None, None)
    assert pool.calls == 1 and first == second and first["win_rate"] == 0.5


def test_health_stop_preserves_partial_raw_and_refuses_reroll(tmp_path, monkeypatch):
    candidate, opponent = cell_setup(tmp_path, monkeypatch)
    data = rows()
    data[2]["result"].update(reason="decision_limit", winner=None)
    pool = Pool(data)
    with pytest.raises(RuntimeError, match="health/limit"):
        pe.evaluate_cell(tmp_path, pool, candidate, "seed-0-fp32", opponent, [], None, None)
    raw = tmp_path / "evaluation/seed-0-fp32/greedy.jsonl"
    assert len(raw.read_text().splitlines()) == 3 and not raw.with_suffix(".json").exists()
    assert pe.read_json(tmp_path / "STOP.json")["row"]["game_id"] == 2
    digest = pe.sha(raw)
    (tmp_path / "STOP.json").unlink()  # operator clearing STOP alone never authorizes replacing partial evidence
    with pytest.raises(ValueError, match="partial cell"):
        pe.evaluate_cell(tmp_path, pool, candidate, "seed-0-fp32", opponent, [], None, None)
    assert pe.sha(raw) == digest and pool.calls == 1


def test_analysis_pairs_baselines_endpoints_and_rejects_panel_drift(tmp_path):
    pe.atomic(tmp_path / "evaluator/identity.json", {"local": 1})
    registration = {
        "evaluation": {
            "bootstrap_seed": 2026100823,
            "bootstrap_replicates": 20000,
            "opponents": [{"name": n} for n in ("greedy", "bc", "rl")],
        }
    }
    assert pe.analyze(tmp_path, registration) is None
    for seed in range(3):
        for arm, count in (("start", 1), ("fp32", 2), ("bf16", 3)):
            for opponent in ("greedy", "bc", "rl"):
                save_cell(tmp_path / f"evaluation/seed-{seed}-{arm}/{opponent}.json", rows(count))
    result = pe.analyze(tmp_path, registration)
    assert result["games"] == 6912 and result["all_healthy"] and not result["auto_promote"]
    assert result["primary_bf16_minus_fp32"]["mean"] == 0.25
    assert result["primary_bf16_minus_fp32"]["ci95"] == [0.25, 0.25]
    assert result["improvement_from_start"]["bf16"]["mean"] == 0.5
    assert result["improvement_from_start"]["fp32"]["mean"] == 0.25
    assert all(result["quality_gate"].values())
    drifted = rows(3)
    drifted[0]["agent_seeds"][0] = 999
    save_cell(tmp_path / "evaluation/seed-2-bf16/rl.json", drifted)
    with pytest.raises(ValueError, match="panel differs"):
        pe.analyze(tmp_path, registration)


def test_completed_endpoint_must_be_verified_and_at_registered_boundary(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    checkpoint = snapshot / "checkpoint.pt"
    checkpoint.write_bytes(b"endpoint")
    job = {"job_id": "seed-0-fp32", "seed": 0, "learner_precision": "fp32"}
    identity = {"registered": 1}
    controller = {"jobs": {job["job_id"]: identity}}
    record = {
        **job,
        "snapshot": str(snapshot),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": pe.sha(checkpoint),
        "identity": identity,
    }
    assert pe.completed_candidate(tmp_path, job, controller) is None
    pe.atomic(tmp_path / "completed/seed-0-fp32.json", record)
    checked = []
    monkeypatch.setattr(pe, "validate_precision", lambda directory, ident: checked.append((directory, ident)))
    monkeypatch.setattr(pe, "load_checkpoint", lambda p: {"counters": {"updates": 156}})
    with pytest.raises(ValueError, match="u160"):
        pe.completed_candidate(tmp_path, job, controller)
    monkeypatch.setattr(pe, "load_checkpoint", lambda p: {"counters": {"updates": 160}})
    assert pe.completed_candidate(tmp_path, job, controller)["update"] == 160 and checked
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        pe.completed_candidate(tmp_path, job, controller)


def test_once_before_controller_does_not_create_controller_identity(tmp_path):
    assert pe.main(["--root", str(tmp_path), "--once"]) == 0
    assert not (tmp_path / "identity.json").exists()
    assert pe.read_json(tmp_path / "evaluator/status.json")["status"] == "waiting-for-controller"


def test_worker_wait_obeys_wall_deadline(tmp_path, monkeypatch):
    clock = iter([0, 31, 61])
    monkeypatch.setattr(pe.time, "monotonic", lambda: next(clock))
    waits = []

    class Pending:
        def next(self, timeout):
            waits.append(timeout)
            raise pe.mp.TimeoutError

    with pytest.raises(TimeoutError, match="wall budget"):
        list(pe.receive_rows(tmp_path, Pending(), 60))
    assert waits == [30, 29]


def test_play_disables_inherited_autocast_and_preserves_failure_trace(tmp_path, monkeypatch):
    """An actual CPU linear kernel proves evaluation precision even under an ambient BF16 scope."""
    observed = []

    class Inner:
        host = object()

    class Duel:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, candidate, opponent):
            result = torch.nn.Linear(4, 4)(torch.ones(1, 4))
            observed.append((result.dtype, torch.is_autocast_enabled("cpu")))
            raise RuntimeError("fixture failure after inference")

    monkeypatch.setattr(pe, "Duel", Duel)
    monkeypatch.setattr(pe, "spec_of", lambda factory: "policy:fixture")
    from ygorl.cards.ydk import Deck

    spec = SimpleNamespace(
        pair=0,
        first=0,
        seed=1,
        agent_seeds=(2, 3),
        agent_a=lambda seed: Inner(),
        agent_b=lambda seed: object(),
        env=None,
        deck_a=Deck(main=(1,), name="a"),
        deck_b=Deck(main=(2,), name="b"),
        config=pe.DuelConfig(),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        row = pe.play((0, spec, str(tmp_path)))
    assert observed == [(torch.float32, False)]
    assert not pe.healthy(row) and pe.sha(row["failure_trace"]["path"]) == row["failure_trace"]["sha256"]
