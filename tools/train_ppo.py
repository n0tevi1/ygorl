"""PPO self-play training (T4b.4): current policy + snapshot pool + keep-best on a deck pool.

Usage:
  uv run python tools/train_ppo.py DECK... [--minutes 60] [--updates N] [--out DIR] [options]
  uv run python tools/train_ppo.py --resume RUN/checkpoints/latest.pt [--minutes 60]
  uv run python tools/train_ppo.py --summary RUN/metrics.jsonl

DECK is a .ydk file or a directory of .ydk files (the 10 test decks: tests/decks). Decks are checked like
``ygorl arena`` does (structural rules, or the environment's pool / banlist with --env). The run directory
(default out/train/<timestamp>; with --env: environments/<version>/artifacts/train/<name>) receives
config.json, vocab.json, metrics.jsonl, eval.jsonl, checkpoints/ and best.pt (docs/training.md §8). Play a
checkpoint with ``ygorl arena ... --agent-a policy:RUN/best.pt`` or ``ygorl duel ... --agent-a policy:...``.
Needs the train extra (``uv sync --extra train``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """The tool's own options plus every training-config flag (ygorl.train.cli; needs PyTorch)."""
    from ygorl.train import cli

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("decks", type=Path, nargs="*", metavar="DECK", help=".ydk file or directory of .ydk files")
    p.add_argument("--out", type=Path, default=None, help="run directory")
    p.add_argument("--name", default=None, help="run name under the environment's artifacts/train/ (with --env)")
    p.add_argument("--resume", type=Path, default=None, metavar="CKPT", help="continue from a checkpoint")
    p.add_argument("--summary", type=Path, default=None, metavar="METRICS", help="print a metrics.jsonl summary")
    p.add_argument("--minutes", type=float, default=None, help="wall-clock budget")
    p.add_argument("--updates", type=int, default=None, help="number of PPO updates")
    cli.add_arguments(p)
    return p


def config_from_args(args, decks: list[str]):
    from ygorl.train import cli

    return cli.from_args(args, decks)


def main(argv: list[str] | None = None) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("train_ppo: PyTorch is missing; run `uv sync --extra train`", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    from ygorl.commands import CommandError, load_decks, load_env
    from ygorl.train.trainer import Trainer, summarize_metrics

    if args.summary is not None:
        print(json.dumps(summarize_metrics(args.summary), indent=2))
        return 0
    try:
        if args.resume is not None:
            trainer = Trainer.resume(args.resume, args.out)
        else:
            if not args.decks:
                raise CommandError("give at least one DECK (or --resume)")
            env = load_env(args.env)
            paths = []
            for path in args.decks:
                paths.extend(sorted(path.glob("*.ydk")) if path.is_dir() else [path])
            load_decks([Path(p) for p in paths], env)  # legality, as in `ygorl arena`
            cfg = config_from_args(args, [str(p) for p in paths])
            stamp = time.strftime("%Y%m%d-%H%M%S")
            if args.out is not None:
                out = args.out
            elif env is not None:
                out = env.artifact_path("train", args.name or stamp)
            else:
                out = Path("out") / "train" / (args.name or stamp)
            trainer = Trainer(cfg, out)
    except (CommandError, ValueError, OSError) as exc:
        print(f"train_ppo: error: {exc}", file=sys.stderr)
        return 2
    print(f"run directory: {trainer.run_dir}")
    print(json.dumps(dataclasses.asdict(trainer.net_config)))
    model = trainer.model.nets[0] if trainer.cfg.seat_split else trainer.model
    print(
        model.actor.parameter_report() + ("\n(x2: seat-split, one network per seat)" if trainer.cfg.seat_split else "")
    )
    from ygorl.train.rollout import RolloutStalled

    try:
        trainer.train(max_updates=args.updates, max_minutes=args.minutes)
    except RolloutStalled as exc:
        # latest.pt is saved; a stuck engine thread would hang the normal shutdown (joining the env pool)
        print(f"error: {exc}\nresume with --resume {trainer.run_dir / 'checkpoints' / 'latest.pt'}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    print(json.dumps(summarize_metrics(trainer.run_dir / "metrics.jsonl"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
