"""Run a registered transfer pilot on one local GPU with durable epoch boundaries."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from colab_checkpoint import atomic, sha


def verify_epoch(root, seed, arm, epoch):
    directory = root / "jobs" / f"seed-{seed}-{arm}"
    progress = json.loads((directory / "progress.json").read_text())
    digest = sha(directory / "latest.pt")
    assert digest == progress["checkpoint_sha256"], "checkpoint digest mismatch"
    state = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
    assert state["identity"] == {"registration_sha256": sha(root / "registration.json"), "seed": seed, "arm": arm}
    assert state["epoch"] == epoch == progress["epoch"] and len(state["history"]) == epoch
    assert {"model", "optimizer", "torch_rng", "cuda_rng"} <= state.keys()
    assert [row["epoch"] for row in state["history"]] == list(range(1, epoch + 1))
    assert json.loads((directory / "metrics.json").read_text()) == state["history"]
    return digest


def run(root):
    root = root.resolve()
    repo = Path(__file__).resolve().parents[1]
    identity = json.loads((root / "identity.json").read_text())
    reg = json.loads((root / "registration.json").read_text())
    assert sha(root / "registration.json") == identity["registration_sha256"]
    for name, digest in identity["code"].items():
        assert sha(repo / name) == digest, name
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    runtime = {"torch": torch.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0)}
    assert runtime == identity["runtime"], "local runtime changed"
    assert reg["backend"] == "local-single-gpu" and 0 < reg["max_hours"] <= 2
    assert not (root / "STOP.json").exists() and not (root / "jobs").exists(), "fresh local run only"
    deadline = min(reg["deadline_unix"], time.time() + reg["max_hours"] * 3600)
    done = threading.Event()
    stage = {"stage": "starting"}

    def heartbeat():
        while not done.is_set():
            atomic(root / "pipeline-status.json", {**stage, "updated_unix": time.time()})
            done.wait(15)

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    try:
        reports = []
        total = reg["epochs"] + reg["adapt_epochs"]
        for seed in reg["seeds"]:
            arms = reg["arms"] if seed % 2 == 0 else reg["arms"][::-1]
            for arm in arms:
                directory = root / "jobs" / f"seed-{seed}-{arm}"
                directory.mkdir(parents=True)
                stage.update(stage="training", seed=seed, arm=arm)
                for epoch in range(1, total + 1):
                    assert not (root / "STOP.json").exists(), "external STOP"
                    remaining = deadline - time.time()
                    assert remaining > 0, "wall budget expired"
                    stage["epoch"] = epoch
                    with (directory / f"epoch-{epoch}.log").open("x") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(repo / "tools/card_transfer_worker.py"),
                                "--root",
                                str(root),
                                "--seed",
                                str(seed),
                                "--arm",
                                arm,
                                "--device",
                                "cuda",
                                "--stop-after",
                                str(epoch),
                            ],
                            check=True,
                            timeout=remaining,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONPATH=str(repo / "src")),
                        )
                    digest = verify_epoch(root, seed, arm, epoch)
                    if epoch == reg["epochs"]:
                        shutil.copy2(directory / "latest.pt", directory / "zero-shot.pt")
                    event = dict(
                        stage="epoch_verified",
                        seed=seed,
                        arm=arm,
                        epoch=epoch,
                        checkpoint_sha256=digest,
                        time=time.time(),
                    )
                    with (root / "events.jsonl").open("a") as f:
                        f.write(json.dumps(event) + "\n")
                    print(json.dumps(event), flush=True)
                report = json.loads((directory / "report.json").read_text())
                assert report["healthy"] and report["checkpoint_sha256"] == digest
                assert report["epochs"] == total
                reports.append(report)
        atomic(
            root / "report.json",
            dict(
                study_sha256=sha(root / "identity.json"),
                healthy=True,
                jobs=len(reports),
                runtime=runtime,
                scope=reg["scope"],
            ),
        )
        stage["stage"] = "complete"
    except BaseException as exc:
        atomic(
            root / ("pipeline-failure.json" if (root / "STOP.json").exists() else "STOP.json"),
            dict(reason=repr(exc), time=time.time()),
        )
        stage["stage"] = "stopped"
        raise
    finally:
        done.set()
        beat.join()
        atomic(root / "pipeline-status.json", {**stage, "updated_unix": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with (args.root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.root)
