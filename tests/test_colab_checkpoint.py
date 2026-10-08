"""Recovery accepts only an intact checkpoint/metric prefix and keeps the previous good generation."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "colab_checkpoint", Path(__file__).parents[1] / "tools/colab_checkpoint.py"
)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
ENV = {"environment": "fixture", "fingerprint": "sealed"}


def snapshot(root, update=2, value=1):
    root.mkdir()
    counters = {"updates": update, "rows": update * 8}
    state = {
        "format": "ygorl-ppo-1",
        "counters": counters,
        "environment": ENV,
        "pool": {},
        "schedule": {},
        "config": {},
        "net_config": {},
        "vocab": "",
        "learner": {"model": {"x": torch.tensor(value)}, "reference": {}, "optimizer": {}, "updates": update},
        "rng": {k: torch.tensor([0], dtype=torch.uint8) for k in ("torch", "collector", "cuda")},
    }
    torch.save(state, root / "checkpoint.pt")
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps({"total": {"updates": u, "rows": u * 8}}) + "\n" for u in range(1, update + 1))
    )
    manifest = {
        "format": "ygorl-colab-snapshot-1",
        "run_id": "test-run",
        "environment": ENV,
        "origin_update": 0,
        "counters": counters,
        "files": {
            name: {"bytes": (root / name).stat().st_size, "sha256": cc.sha(root / name)}
            for name in ("checkpoint.pt", "metrics.jsonl")
        },
    }
    cc.atomic(root / "manifest.json", manifest)
    return root


def rehash(root, name):
    m = json.loads((root / "manifest.json").read_text())
    m["files"][name] = {"bytes": (root / name).stat().st_size, "sha256": cc.sha(root / name)}
    cc.atomic(root / "manifest.json", m)


def test_partial_download_preserves_last_good_checkpoint(tmp_path):
    store = tmp_path / "store"
    with cc.run_lock(store):
        accepted = cc.promote(snapshot(tmp_path / "first"), store, "test-run", ENV)
        before = (store / "latest.json").read_bytes()
        bad = snapshot(tmp_path / "partial", 4)
        (bad / "checkpoint.pt").write_bytes((bad / "checkpoint.pt").read_bytes()[:40])
        with pytest.raises(ValueError, match="incomplete or corrupt"):
            cc.promote(bad, store, "test-run", ENV)
        assert (store / "latest.json").read_bytes() == before
        assert cc.validate(accepted, "test-run", ENV)["counters"]["updates"] == 2


@pytest.mark.parametrize("updates", [[1, 1, 3], [1, 3], [1, 2, 3, 4]])
def test_metric_holes_duplicates_and_uncommitted_suffix_rejected(tmp_path, updates):
    path = snapshot(tmp_path / "candidate", 3)
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps({"total": {"updates": u, "rows": u * 8}}) + "\n" for u in updates)
    )
    rehash(path, "metrics.jsonl")
    with pytest.raises(ValueError, match="missing or duplicate"):
        cc.validate(path, "test-run", ENV)


def test_stale_and_conflicting_workers_cannot_replace_latest(tmp_path):
    store = tmp_path / "store"
    with cc.run_lock(store):
        cc.promote(snapshot(tmp_path / "first", 4), store, "test-run", ENV)
        before = (store / "latest.json").read_bytes()
        with pytest.raises(ValueError, match="older checkpoint"):
            cc.promote(snapshot(tmp_path / "stale", 2), store, "test-run", ENV)
        with pytest.raises(ValueError, match="conflicting workers"):
            cc.promote(snapshot(tmp_path / "conflict", 4, 99), store, "test-run", ENV)
        assert (store / "latest.json").read_bytes() == before


def test_run_lock_rejects_second_controller(tmp_path):
    with cc.run_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with cc.run_lock(tmp_path):
                pytest.fail("second controller acquired the run")


def test_state_counters_must_match_manifest_and_metric_boundary(tmp_path):
    path = snapshot(tmp_path / "candidate", 2)
    state = torch.load(path / "checkpoint.pt", weights_only=True)
    state["counters"]["rows"] += 1
    torch.save(state, path / "checkpoint.pt")
    rehash(path, "checkpoint.pt")
    with pytest.raises(ValueError, match="counters/environment mismatch"):
        cc.validate(path, "test-run", ENV)


def test_wrong_run_or_environment_rejected(tmp_path):
    path = snapshot(tmp_path / "candidate")
    with pytest.raises(ValueError, match="identity mismatch"):
        cc.validate(path, "another-run", ENV)
    with pytest.raises(ValueError, match="identity mismatch"):
        cc.validate(path, "test-run", {"environment": "other"})
