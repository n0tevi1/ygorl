"""Training throughput: ``Trainer.step()`` (collect + update) timings on one device and configuration.

  uv run python tools/bench_train.py --device cuda --envs 128 --steps 16 [--updates 6]

Runs ``--updates`` steps with evaluation, snapshots and checkpoints off, drops the first (warm-up) and reports the
mean collect / update seconds, decisions per second while collecting and rows per second overall; ``--json``
appends one line per run for docs/benchmarks.md. On an AMD GPU the mean ``gpu_busy_percent``
over the measured steps is included.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class GpuBusy:
    """Mean of the amdgpu ``gpu_busy_percent`` counter sampled every 10 ms (None without an AMD GPU)."""

    PATHS = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with self.PATHS[0].open() as f:
            while not self._stop.wait(0.01):
                f.seek(0)
                self.samples.append(int(f.read()))

    def start(self) -> None:
        if self.PATHS:
            self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        return round(statistics.fmean(self.samples), 1) if self.samples else None


def _commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()  # fmt: skip
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("decks", nargs="*", type=Path, help=".ydk files (default: tests/decks/*.ydk)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--updates", type=int, default=6)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--min-batch", type=int, default=None)
    p.add_argument("--env-threads", type=int, default=2)
    p.add_argument("--collect-threads", type=int, default=2)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--event-length", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--no-target-kl", action="store_true", help="always run every epoch (fixed update cost)")
    p.add_argument(
        "--overlap",
        action="store_true",
        help="experimental: collect the next rollout while updating (one update stale)",
    )
    p.add_argument("--bf16", action="store_true", help="experimental: bf16 autocast on a GPU")
    p.add_argument("--label", default="", help="free-form tag stored in the JSON line")
    p.add_argument("--json", type=Path, default=None, help="append the summary as one JSON line")
    args = p.parse_args()

    from ygorl.train.ppo import PPOConfig
    from ygorl.train.trainer import TrainConfig, Trainer
    import torch

    decks = [str(d) for d in (args.decks or sorted((ROOT / "tests" / "decks").glob("*.ydk")))]
    net = {"d_model": args.d_model, "n_heads": 4, "board_layers": args.layers, "history_layers": args.layers}
    ppo = PPOConfig(epochs=args.epochs, minibatch_size=args.minibatch, target_kl=None if args.no_target_kl else 0.01)
    cfg = TrainConfig(decks=tuple(decks), num_envs=args.envs, steps=args.steps, min_batch=args.min_batch,
                      env_threads=args.env_threads, collect_threads=args.collect_threads,
                      torch_threads=args.torch_threads, event_length=args.event_length, net=net, ppo=ppo,
                      device=args.device, overlap_collect=args.overlap, bf16=args.bf16, snapshot_every=0,
                      checkpoint_every=0, eval_every=0)  # fmt: skip
    busy = GpuBusy()
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(cfg, tmp, log=None)
        trainer.learner.timing = True  # time/<section> of each update (#72)
        for i in range(args.updates):
            if i == 1:
                busy.start()  # after the warm-up step
            t0 = time.perf_counter()
            r = trainer.step()
            r["step_s"] = time.perf_counter() - t0
            records.append(r)
            print(
                f"step {i}: collect {r['collect_s']:.2f} s ({r['decisions_per_s']:.0f} decisions/s), "
                f"update {r['update_s']:.2f} s ({r['minibatches']} minibatches), {r['rows'] / r['step_s']:.0f} rows/s"
            )
    kept = records[1:] or records

    def mean(key):
        return round(statistics.fmean(r[key] for r in kept), 3)

    summary = {"label": args.label, "device": args.device, "envs": args.envs, "steps": args.steps,
               "min_batch": args.min_batch, "env_threads": args.env_threads, "collect_threads": args.collect_threads,
               "torch_threads": args.torch_threads, "overlap": args.overlap, "bf16": args.bf16, "d_model": args.d_model,
               "layers": args.layers, "epochs": args.epochs, "minibatch": args.minibatch, "rows": kept[0]["rows"],
               "collect_s": mean("collect_s"), "update_s": mean("update_s"), "step_s": mean("step_s"),
               "minibatches": mean("minibatches"), "decisions_per_s": round(mean("decisions_per_s")),
               "rows_per_s": round(sum(r["rows"] for r in kept) / sum(r["step_s"] for r in kept)),
               "gpu_busy": busy.stop()}  # fmt: skip
    sections = sorted({k for r in kept for k in r if k.startswith("time/")})
    summary["update_breakdown"] = {k[5:]: mean(k) for k in sections}
    summary["commit"] = _commit()
    summary["rocm"] = {"hip": torch.version.hip, "aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
                       "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION")} if torch.version.hip else None  # fmt: skip
    total = summary["update_breakdown"].get("total") or 1.0
    print("update breakdown: " + ", ".join(f"{k} {v:.3f} s ({v / total:.0%})" for k, v in summary["update_breakdown"].items()
                                           if k != "total") + f"; total {total:.3f} s")  # fmt: skip
    print(json.dumps(summary))
    if args.json is not None:
        with args.json.open("a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
