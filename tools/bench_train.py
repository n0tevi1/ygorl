"""Training throughput: ``Trainer.step()`` (collect + update) timings on one device and configuration.

  uv run python tools/bench_train.py --device cuda --envs 128 --steps 16 [--updates 6]

Runs ``--updates`` steps with evaluation, snapshots and checkpoints off, drops the first (warm-up) and reports the
mean collect / update seconds, decisions per second while collecting and rows per second overall; ``--json``
appends one line per run for docs/benchmarks.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
    p.add_argument("--json", type=Path, default=None, help="append the summary as one JSON line")
    args = p.parse_args()

    from ygorl.train.ppo import PPOConfig
    from ygorl.train.trainer import TrainConfig, Trainer

    decks = [str(d) for d in (args.decks or sorted((ROOT / "tests" / "decks").glob("*.ydk")))]
    net = {"d_model": args.d_model, "n_heads": 4, "board_layers": args.layers, "history_layers": args.layers}
    ppo = PPOConfig(epochs=args.epochs, minibatch_size=args.minibatch, target_kl=None if args.no_target_kl else 0.01)
    cfg = TrainConfig(decks=tuple(decks), num_envs=args.envs, steps=args.steps, min_batch=args.min_batch,
                      env_threads=args.env_threads, collect_threads=args.collect_threads,
                      torch_threads=args.torch_threads, event_length=args.event_length, net=net, ppo=ppo,
                      device=args.device, snapshot_every=0, checkpoint_every=0, eval_every=0)  # fmt: skip
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(cfg, tmp, log=None)
        for i in range(args.updates):
            t0 = time.perf_counter()
            r = trainer.step()
            r["step_s"] = time.perf_counter() - t0
            records.append(r)
            print(f"step {i}: collect {r['collect_s']:.2f} s ({r['decisions_per_s']:.0f} decisions/s), "
                  f"update {r['update_s']:.2f} s ({r['minibatches']} minibatches), {r['rows'] / r['step_s']:.0f} rows/s")
    kept = records[1:] or records

    def mean(key):
        return round(statistics.fmean(r[key] for r in kept), 3)

    summary = {"device": args.device, "envs": args.envs, "steps": args.steps, "env_threads": args.env_threads,
               "collect_threads": args.collect_threads, "torch_threads": args.torch_threads, "d_model": args.d_model,
               "layers": args.layers, "epochs": args.epochs, "minibatch": args.minibatch, "rows": kept[0]["rows"],
               "collect_s": mean("collect_s"), "update_s": mean("update_s"), "minibatches": mean("minibatches"),
               "decisions_per_s": round(mean("decisions_per_s")),
               "rows_per_s": round(sum(r["rows"] for r in kept) / sum(r["step_s"] for r in kept))}  # fmt: skip
    print(json.dumps(summary))
    if args.json is not None:
        with args.json.open("a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
