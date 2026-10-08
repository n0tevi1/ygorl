"""Immutable, verified checkpoint generations for preemptible research workers (#89)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def run_lock(store):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    with (store / "run.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def validate(directory, run_id, environment):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    if m["format"] != "ygorl-colab-snapshot-1" or m["run_id"] != run_id or m["environment"] != environment:
        raise ValueError("snapshot identity mismatch")
    if set(m["files"]) != {"checkpoint.pt", "metrics.jsonl"}:
        raise ValueError("unexpected snapshot files")
    for name, record in m["files"].items():
        p = directory / name
        if p.stat().st_size != record["bytes"] or sha(p) != record["sha256"]:
            raise ValueError(f"incomplete or corrupt {name}")
    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    required = {"learner", "pool", "schedule", "rng", "config", "net_config", "vocab", "environment", "counters"}
    if state.get("format") != "ygorl-ppo-1" or not required <= state.keys():
        raise ValueError("not a full training checkpoint")
    if not {"model", "reference", "optimizer", "updates"} <= state["learner"].keys():
        raise ValueError("missing learner state")
    if not {"torch", "collector", "cuda"} <= state["rng"].keys():
        raise ValueError("missing RNG state")
    if state["counters"] != m["counters"] or state["environment"] != environment:
        raise ValueError("checkpoint counters/environment mismatch")
    raw = (directory / "metrics.jsonl").read_text()
    if raw and not raw.endswith("\n"):
        raise ValueError("partial metrics record")
    rows = [json.loads(s) for s in raw.splitlines()]
    u = state["counters"]["updates"]
    if [r["total"]["updates"] for r in rows] != list(range(m["origin_update"] + 1, u + 1)):
        raise ValueError("metrics contain missing or duplicate committed updates")
    if rows and rows[-1]["total"] != state["counters"]:
        raise ValueError("metrics/checkpoint boundary differs")
    return m


def promote(staging, store, run_id, environment):
    """Caller holds run_lock; only a complete validated generation can replace the latest pointer."""
    staging, store = Path(staging), Path(store)
    manifest = validate(staging, run_id, environment)
    update = manifest["counters"]["updates"]
    digest = manifest["files"]["checkpoint.pt"]["sha256"]
    pointer = store / "latest.json"
    if pointer.exists():
        old = json.loads(pointer.read_text())
        if old["run_id"] != run_id or update < old["update"]:
            raise ValueError("wrong run or older checkpoint cannot replace latest")
        if update == old["update"]:
            if digest != old["sha256"]:
                raise ValueError("conflicting workers published the same update")
            return store / old["path"]
    destination = store / "snapshots" / f"u{update:08d}-{digest[:16]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if validate(destination, run_id, environment) != manifest:
            raise ValueError("existing generation differs")
    else:
        os.replace(staging, destination)
    atomic(pointer, {"run_id": run_id, "update": update, "sha256": digest, "path": str(destination.relative_to(store))})
    return destination


def seal(trainer, root, run_id, origin_update, generation, rows):
    """Publish checkpoint and its exact metric prefix together; manifests are written last."""
    from ygorl.train.checkpoint import save_checkpoint

    state = trainer.state_dict()
    update = state["counters"]["updates"]
    directory = Path(root) / "snapshots" / f"g{generation}-u{update:08d}"
    directory.mkdir(parents=True, exist_ok=False)
    save_checkpoint(state, directory / "checkpoint.pt")
    with (directory / "metrics.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    m = {
        "format": "ygorl-colab-snapshot-1",
        "run_id": run_id,
        "generation": generation,
        "origin_update": origin_update,
        "environment": state["environment"],
        "counters": state["counters"],
        "files": {
            name: {"sha256": sha(directory / name), "bytes": (directory / name).stat().st_size}
            for name in ("checkpoint.pt", "metrics.jsonl")
        },
    }
    atomic(directory / "manifest.json", m)
    validate(directory, run_id, state["environment"])
    return directory
