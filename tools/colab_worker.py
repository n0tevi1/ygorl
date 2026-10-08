"""Bounded CUDA recovery pilot; checkpoints, lease, health and restore checks are explicit."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from colab_checkpoint import atomic, run_lock, seal, sha
from ygorl.train.checkpoint import load_checkpoint
from ygorl.train.trainer import TrainConfig, Trainer


def equal(a, b, path="state"):
    if isinstance(b, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for k in b:
            equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            equal(x, y, f"{path}.{i}")
    else:
        assert a == b, path


class CheckedTrainer(Trainer):
    def _collect(self):
        if time.time() > self.deadline:
            raise RuntimeError("worker deadline exceeded")
        lease = json.loads(self.lease.read_text())
        if lease["generation"] != self.generation or time.time() > lease["expires_unix"]:
            raise RuntimeError("worker lease expired or fenced")
        ro = super()._collect()
        for game in ro.games:
            if game.reason == "error" or any(
                game.result.get(k)
                for k in ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")
            ):
                self._log_games([game])
                self._log_errors([game])
                raise RuntimeError("unhealthy rollout; no optimizer update accepted")
        return ro

    def step(self):
        row = super().step()
        if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
            raise RuntimeError("nonfinite metrics")
        games = self.counters["games"] - self.start_counts["games"]
        truncated = self.counters["truncated"] - self.start_counts["truncated"]
        if games >= 100 and truncated / games > 0.01:
            raise RuntimeError("new training truncations exceed 1%")
        return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    job = json.loads(Path(args.config).read_text())
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.version.cuda and not torch.version.hip
    assert torch.__version__.split("+")[0] == "2.14.0"
    assert sha(job["checkpoint"]) == job["checkpoint_sha256"]
    state = load_checkpoint(job["checkpoint"])
    cfg = TrainConfig.from_dict(state["config"])
    paths = tuple(str(Path("/content/ygorl/environments/md-2026-09/meta") / Path(d).name) for d in cfg.decks)
    assert all(sha(path) == job["deck_sha256"][Path(path).name] for path in paths)
    cfg = replace(cfg, decks=paths, device="cuda")
    state["config"] = cfg.to_dict()
    if job["migration"]:
        torch.cuda.manual_seed_all(2026100810)
        state["rng"]["cuda"] = torch.cuda.get_rng_state()
    with run_lock(out):
        trainer = CheckedTrainer(cfg, out / "run", state=state, log=lambda s: print(s, flush=True))
        actual = trainer.state_dict()
        for k in actual:
            equal(actual[k], state[k], k)
        assert trainer.environment.stamp() == job["environment"]
        assert state["counters"]["updates"] < job["target_update"]
        rows = [json.loads(s) for s in Path(job["metrics"]).read_text().splitlines()] if job.get("metrics") else []
        assert [r["total"]["updates"] for r in rows] == list(
            range(job["origin_update"] + 1, state["counters"]["updates"] + 1)
        )
        trainer.start_counts = dict(trainer.counters)
        trainer.deadline = time.time() + 1800
        trainer.lease = Path(job["lease"])
        trainer.generation = job["generation"]
        audit = {
            "restored_update": trainer.counters["updates"],
            "all_saved_fields_equal": True,
            "migration_cuda_rng_reset": job["migration"],
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "source_sha256": job["checkpoint_sha256"],
            "active_duels_restored": False,
            "started_unix": time.time(),
        }
        atomic(out / "restore.json", audit)
        start = time.monotonic()
        latest = None
        try:
            while trainer.counters["updates"] < job["target_update"]:
                trainer.train(max_updates=1)
                metrics = [json.loads(s) for s in (out / "run/metrics.jsonl").read_text().splitlines()]
                row = metrics[-1]
                # Snapshot admission happens after step() logs; publish the post-bookkeeping counters.
                row["total"] = dict(trainer.counters)
                rows.append(row)
                update = trainer.counters["updates"]
                if update % 2 == 0 or update == job["target_update"]:
                    latest = seal(trainer, out, job["run_id"], job["origin_update"], job["generation"], rows)
                atomic(
                    out / "status.json",
                    {
                        "stage": "complete" if update == job["target_update"] else "running",
                        "generation": job["generation"],
                        "updated_unix": time.time(),
                        "counters": dict(trainer.counters),
                        "latest": str(latest) if latest else None,
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )
        except BaseException as e:
            atomic(
                out / "status.json",
                {
                    "stage": "failed",
                    "error": repr(e),
                    "updated_unix": time.time(),
                    "latest": str(latest) if latest else None,
                    "generation": job["generation"],
                },
            )
            raise


if __name__ == "__main__":
    main()
