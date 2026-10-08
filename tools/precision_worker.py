"""Frozen paired precision jobs on a leased, preemptible CUDA worker."""

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch

from colab_checkpoint import atomic, run_lock, seal, sha, validate
from colab_worker import equal
from ygorl.train.checkpoint import load_checkpoint
from ygorl.train.trainer import TrainConfig, Trainer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_config(source, precision, repo="/content/ygorl"):
    if precision not in ("fp32", "bf16"):
        raise ValueError("unregistered learner precision")
    cfg = TrainConfig.from_dict(source)
    if cfg.bf16:
        raise ValueError("source collection must be FP32")
    if cfg.steps * cfg.num_envs != 16384:
        raise ValueError("source rollout budget differs from registered 16384 rows")
    return replace(
        cfg,
        decks=tuple(str(Path(repo) / "environments/md-2026-09/meta" / Path(d).name) for d in cfg.decks),
        device="cuda",
        bf16=False,
        learner_precision=precision,
        log_games=True,
        register_every=0,
        register_matrix=None,
        eval_every=0,
        checkpoint_every=0,
    ).to_dict()


def check_lease(job, now=None):
    now = time.time() if now is None else now
    lease = json.loads(Path(job["lease"]).read_text())
    if (
        lease.get("generation") != job["generation"]
        or lease.get("job_id") != job["job_id"]
        or now >= min(lease["expires_unix"], job["deadline_unix"])
    ):
        raise RuntimeError("worker lease expired or fenced")


def validate_precision(directory, identity):
    directory = Path(directory)
    base = validate(directory, identity["run_id"], identity["environment"])
    extra = json.loads((directory / "precision-manifest.json").read_text())
    if extra["identity"] != identity or extra["base_sha256"] != sha(directory / "manifest.json"):
        raise ValueError("precision snapshot identity mismatch")
    games = directory / "games.jsonl.gz"
    if sha(games) != extra["games_sha256"] or games.stat().st_size != extra["games_bytes"]:
        raise ValueError("corrupt game log")
    state = load_checkpoint(directory / "checkpoint.pt")
    if digest(state["config"]) != identity["config_sha256"]:
        raise ValueError("resumed precision/config differs")
    if not identity["origin_update"] < base["counters"]["updates"] <= identity["target_update"]:
        raise ValueError("snapshot outside registered update range")
    count, previous = 0, identity["origin_update"]
    with gzip.open(games, "rt") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("partial game log")
            row = json.loads(line)
            if not identity["origin_update"] < row["update"] <= base["counters"]["updates"]:
                raise ValueError("game log outside checkpoint prefix")
            if row["update"] < previous:
                raise ValueError("game log out of order")
            if any(row.get(k) != identity["environment"].get(k) for k in ("environment", "fingerprint")):
                raise ValueError("game environment differs")
            previous = row["update"]
            count += 1
    if count != base["counters"]["games"] - identity["origin_counts"]["games"]:
        raise ValueError("game log/checkpoint boundary differs")
    return extra


def seal_precision(trainer, out, job, rows):
    path = seal(trainer, out, job["run_id"], job["origin_update"], job["generation"], rows)
    source = Path(out) / "run/games.jsonl.gz"
    if source.exists():
        shutil.copyfile(source, path / "games.jsonl.gz")
    else:
        with gzip.open(path / "games.jsonl.gz", "wb"):
            pass
    atomic(
        path / "precision-manifest.json",
        {
            "identity": job["identity"],
            "base_sha256": sha(path / "manifest.json"),
            "games_sha256": sha(path / "games.jsonl.gz"),
            "games_bytes": (path / "games.jsonl.gz").stat().st_size,
        },
    )
    validate_precision(path, job["identity"])
    return path


class PrecisionTrainer(Trainer):
    @contextmanager
    def _autocast(self, *, learner=False):
        with super()._autocast(learner=learner):
            label = "learner" if learner else "collection_and_bootstrap"
            enabled = torch.is_autocast_enabled("cuda")
            expected = learner and self.cfg.learner_precision == "bf16"
            if enabled != expected:
                raise RuntimeError("actual autocast context differs from assigned precision")
            self.precision_evidence.setdefault(label, {"autocast": enabled, "linear_dtypes": []})
            self.precision_context = label
            try:
                yield
            finally:
                self.precision_context = None

    def _collect(self):
        check_lease(self.job)
        result = super()._collect()
        bad = [
            g
            for g in result.games
            if g.reason == "error"
            or any(
                g.result.get(k)
                for k in ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")
            )
        ]
        if bad:
            self._log_games(bad)
            self._log_errors(bad)
            raise RuntimeError("unhealthy rollout; optimizer update rejected")
        return result

    def step(self):
        row = super().step()
        if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
            raise RuntimeError("nonfinite metrics")
        if any(not torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise RuntimeError("nonfinite model parameters")
        games = self.counters["games"] - self.job["identity"]["origin_counts"]["games"]
        truncated = self.counters["truncated"] - self.job["identity"]["origin_counts"]["truncated"]
        if games >= 100 and truncated / games > 0.01:
            raise RuntimeError("cumulative new training truncations exceed 1%")
        return row


def run(job):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def watchdog():
        while not stop.wait(5):
            try:
                check_lease(job)
            except BaseException as exc:
                atomic(out / "status.json", {"stage": "failed", "error": repr(exc)})
                os._exit(75)

    check_lease(job)
    threading.Thread(target=watchdog, daemon=True).start()
    latest = None
    try:
        if not torch.cuda.is_available() or torch.version.hip or torch.__version__.split("+")[0] != "2.14.0":
            raise RuntimeError("registered CUDA/PyTorch platform differs")
        if torch.cuda.device_count() != 1 or "L4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("registered single L4 hardware differs")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if sha(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError("input checkpoint hash differs")
        state = load_checkpoint(job["checkpoint"])
        if job["migration"]:
            if job["checkpoint_sha256"] != job["identity"]["source_checkpoint_sha256"]:
                raise ValueError("migration source differs")
            config = expected_config(state["config"], job["learner_precision"])
            if state["counters"] != job["identity"]["origin_counts"]:
                raise ValueError("migration counters differ")
            torch.cuda.manual_seed_all(2026100820 + job["seed"])
            state["rng"]["cuda"] = torch.cuda.get_rng_state()
            state["config"] = config
        if digest(state["config"]) != job["identity"]["config_sha256"]:
            raise ValueError("resumed precision/config differs")
        cfg = TrainConfig.from_dict(state["config"])
        if any(sha(p) != job["deck_sha256"][Path(p).name] for p in cfg.decks):
            raise ValueError("deck hash differs")
        if not job["origin_update"] <= state["counters"]["updates"] < job["target_update"]:
            raise ValueError("input update outside registered range")
        rows = [json.loads(s) for s in Path(job["metrics"]).read_text().splitlines()] if job.get("metrics") else []
        if [r["total"]["updates"] for r in rows] != list(
            range(job["origin_update"] + 1, state["counters"]["updates"] + 1)
        ):
            raise ValueError("resumed metric prefix differs")
        with run_lock(out):
            trainer = PrecisionTrainer(cfg, out / "run", state=state, log=lambda s: print(s, flush=True))
            for key, value in trainer.state_dict().items():
                equal(value, state[key], key)
            if trainer.environment.stamp() != job["environment"]:
                raise ValueError("environment differs")
            trainer.job = job
            trainer.precision_evidence = {}
            trainer.precision_context = None
            if job.get("games"):
                shutil.copyfile(job["games"], out / "run/games.jsonl.gz")

            def record_dtype(module, args, output):
                context = trainer.precision_context
                if context and isinstance(output, torch.Tensor):
                    values = trainer.precision_evidence[context]["linear_dtypes"]
                    if str(output.dtype) not in values:
                        values.append(str(output.dtype))

            hooks = [
                module.register_forward_hook(record_dtype)
                for module in trainer.model.modules()
                if isinstance(module, torch.nn.Linear)
            ]
            audit = {
                "all_saved_fields_equal": True,
                "migration_cuda_rng_reset": job["migration"],
                "migration_seed": 2026100820 + job["seed"],
                "restored_update": trainer.counters["updates"],
                "gpu": torch.cuda.get_device_name(0),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "active_duels_restored": False,
                "identity": job["identity"],
                "config": cfg.to_dict(),
                "started_unix": time.time(),
            }
            atomic(out / "restore.json", audit)
            start = time.monotonic()
            while trainer.counters["updates"] < job["target_update"]:
                trainer.train(max_updates=1)
                row = json.loads((out / "run/metrics.jsonl").read_text().splitlines()[-1])
                row["total"] = dict(trainer.counters)
                rows.append(row)
                if hooks:
                    for hook in hooks:
                        hook.remove()
                    hooks = []
                    for context, expected_dtype in (
                        ("collection_and_bootstrap", "torch.float32"),
                        ("learner", "torch.bfloat16" if cfg.learner_precision == "bf16" else "torch.float32"),
                    ):
                        evidence = trainer.precision_evidence.get(context, {})
                        if evidence.get("linear_dtypes") != [expected_dtype]:
                            raise RuntimeError(f"actual {context} linear dtype differs: {evidence}")
                    atomic(out / "precision-evidence.json", trainer.precision_evidence)
                update = trainer.counters["updates"]
                if (update - job["origin_update"]) % job["snapshot_every"] == 0 or update == job["target_update"]:
                    latest = seal_precision(trainer, out, job, rows)
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
    except BaseException as exc:
        atomic(
            out / "status.json",
            {
                "stage": "failed",
                "error": repr(exc),
                "latest": str(latest) if latest else None,
                "generation": job["generation"],
                "updated_unix": time.time(),
            },
        )
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
