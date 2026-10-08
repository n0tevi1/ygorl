"""Offline admission and fencing checks for paired, preemptible precision jobs."""

import gzip
import importlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
pc = importlib.import_module("precision_colab")
pw = importlib.import_module("precision_worker")
cc = importlib.import_module("colab_checkpoint")
TrainConfig = pw.TrainConfig


def registration():
    return {
        "run_id": "test-precision",
        "origin_update": 128,
        "target_update": 160,
        "snapshot_every": 4,
        "lease_seconds": 300,
        "max_hours": 6,
        "max_units": 12,
        "max_allocations": 8,
        "jobs": [
            {
                "job_id": f"seed-{seed}-{precision}",
                "seed": seed,
                "learner_precision": precision,
                "checkpoint": f"/seed-{seed}.pt",
                "checkpoint_sha256": str(seed) * 64,
            }
            for seed, precision in ((0, "fp32"), (0, "bf16"), (1, "bf16"), (1, "fp32"), (2, "fp32"), (2, "bf16"))
        ],
    }


def test_registered_job_order_pairing_and_budgets():
    value = registration()
    pc.validate_registration(value)
    value["jobs"][1]["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="paired initial"):
        pc.validate_registration(value)
    value = registration()
    value["max_allocations"] = 9
    with pytest.raises(ValueError, match="budget"):
        pc.validate_registration(value)
    value = registration()
    value["jobs"] = value["jobs"][::-1]
    with pytest.raises(ValueError, match="order"):
        pc.validate_registration(value)


def test_only_approved_configuration_changes_and_resume_precision_identity():
    cfg = TrainConfig(decks=("/old/branded.ydk",), bf16=False).to_dict()
    fp32 = pw.expected_config(cfg, "fp32")
    bf16 = pw.expected_config(cfg, "bf16")
    assert fp32["bf16"] is False and bf16["bf16"] is False
    assert {k for k in fp32 if fp32[k] != bf16[k]} == {"learner_precision"}
    changed = {k for k in cfg if cfg[k] != fp32[k]}
    assert changed <= {
        "decks",
        "device",
        "bf16",
        "learner_precision",
        "log_games",
        "register_every",
        "register_matrix",
        "eval_every",
        "checkpoint_every",
    }
    assert pw.digest(fp32) != pw.digest(bf16)
    with pytest.raises(ValueError, match="precision"):
        pw.expected_config(cfg, "fp16")
    cfg["bf16"] = True
    with pytest.raises(ValueError, match="collection"):
        pw.expected_config(cfg, "fp32")


def test_lease_checks_job_generation_expiry_and_absolute_deadline(tmp_path):
    lease = tmp_path / "lease.json"
    cc.atomic(lease, {"job_id": "seed-0-fp32", "generation": 4, "expires_unix": 400})
    job = {"lease": str(lease), "job_id": "seed-0-fp32", "generation": 4, "deadline_unix": 350}
    pw.check_lease(job, now=349)
    for override in ({"generation": 3}, {"job_id": "seed-0-bf16"}, {"deadline_unix": 300}):
        with pytest.raises(RuntimeError, match="expired or fenced"):
            pw.check_lease({**job, **override}, now=349)
    with pytest.raises(RuntimeError, match="expired or fenced"):
        pw.check_lease(job, now=400)


def test_server_session_matching_does_not_confuse_prefix_or_banner():
    listing = "[colab] Banner\n[study-vm1] e | Hardware: L4 | Variant: GPU\n[study-vm10] e | Hardware: L4"
    assert pc.active_names(listing) == {"study-vm1", "study-vm10"}
    assert "study-vm" not in pc.active_names(listing)
    assert pc.active_names("[colab] No active sessions found on server.") == set()
    with pytest.raises(RuntimeError, match="cannot interpret"):
        pc.active_names("")


def snapshot(root):
    cfg = pw.expected_config(TrainConfig(decks=("branded.ydk",)).to_dict(), "fp32")
    env = {"environment": "fixture", "fingerprint": "sealed"}
    identity = {
        "run_id": "study/seed-0-fp32",
        "environment": env,
        "config_sha256": pw.digest(cfg),
        "origin_update": 128,
        "target_update": 160,
        "origin_counts": {"games": 100},
        "source_checkpoint_sha256": "source",
        "learner_precision": "fp32",
    }
    counters = {"updates": 132, "games": 104}
    state = {
        "format": "ygorl-ppo-1",
        "config": cfg,
        "environment": env,
        "counters": counters,
        "learner": {"model": {}, "reference": {}, "optimizer": {}, "updates": 132},
        "rng": {"torch": torch.tensor([1]), "collector": torch.tensor([1]), "cuda": torch.tensor([1])},
        "pool": {},
        "schedule": {},
        "net_config": {},
        "vocab": "",
    }
    root.mkdir()
    torch.save(state, root / "checkpoint.pt")
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps({"total": {"updates": i, "games": i - 28}}) + "\n" for i in range(129, 133))
    )
    cc.atomic(
        root / "manifest.json",
        {
            "format": "ygorl-colab-snapshot-1",
            "run_id": identity["run_id"],
            "environment": env,
            "origin_update": 128,
            "counters": counters,
            "files": {
                n: {"bytes": (root / n).stat().st_size, "sha256": cc.sha(root / n)}
                for n in ("checkpoint.pt", "metrics.jsonl")
            },
        },
    )
    with gzip.open(root / "games.jsonl.gz", "wt") as stream:
        for i in range(129, 133):
            stream.write(json.dumps({"update": i, **env}) + "\n")
    cc.atomic(
        root / "precision-manifest.json",
        {
            "identity": identity,
            "base_sha256": cc.sha(root / "manifest.json"),
            "games_sha256": cc.sha(root / "games.jsonl.gz"),
            "games_bytes": (root / "games.jsonl.gz").stat().st_size,
        },
    )
    return identity


def test_precision_snapshot_rejects_wrong_arm_and_incomplete_games(tmp_path):
    directory = tmp_path / "snapshot"
    identity = snapshot(directory)
    pw.validate_precision(directory, identity)
    with pytest.raises(ValueError, match="identity"):
        pw.validate_precision(directory, {**identity, "learner_precision": "bf16"})
    with gzip.open(directory / "games.jsonl.gz", "wt") as stream:
        stream.write(json.dumps({"update": 132, **identity["environment"]}) + "\n")
    manifest = json.loads((directory / "precision-manifest.json").read_text())
    manifest.update(
        games_sha256=cc.sha(directory / "games.jsonl.gz"), games_bytes=(directory / "games.jsonl.gz").stat().st_size
    )
    cc.atomic(directory / "precision-manifest.json", manifest)
    with pytest.raises(ValueError, match="game log/checkpoint boundary"):
        pw.validate_precision(directory, identity)


def test_stop_file_prevents_lease_refresh_or_allocation(tmp_path):
    pilot = object.__new__(pc.PrecisionPilot)
    pilot.root = tmp_path
    (tmp_path / "STOP.json").write_text("{}")
    with pytest.raises(RuntimeError, match="STOP"):
        pilot.guard()


def test_cleanup_refuses_an_unowned_session():
    pilot = object.__new__(pc.PrecisionPilot)
    pilot.owned = ["owned"]
    with pytest.raises(ValueError, match="unowned"):
        pilot.release("someone-else")


def test_failed_server_listing_cannot_confirm_cleanup(tmp_path, monkeypatch):
    pilot = object.__new__(pc.PrecisionPilot)
    pilot.owned = ["owned"]
    pilot.root = tmp_path
    pilot.cli = "colab"
    pilot.serial = 0
    pilot.cli_call = lambda *args, **kwargs: ""

    def unavailable(*args, **kwargs):
        raise pc.subprocess.CalledProcessError(1, ["colab", "sessions"])

    monkeypatch.setattr(pc.subprocess, "run", unavailable)
    with pytest.raises(pc.subprocess.CalledProcessError):
        pilot.release("owned")
