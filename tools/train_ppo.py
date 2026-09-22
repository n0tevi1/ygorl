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
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("decks", type=Path, nargs="*", metavar="DECK", help=".ydk file or directory of .ydk files")
    p.add_argument("--pairings", default="cross", choices=("all", "cross", "mirror"),
                   help="deck pairings to sample (default cross: distinct decks only)")  # fmt: skip
    p.add_argument("--env", default=None, metavar="PATH|VERSION", help="environment (rules; stamped into checkpoints)")
    p.add_argument("--out", type=Path, default=None, help="run directory")
    p.add_argument("--name", default=None, help="run name under the environment's artifacts/train/ (with --env)")
    p.add_argument("--resume", type=Path, default=None, metavar="CKPT", help="continue from a checkpoint")
    p.add_argument("--summary", type=Path, default=None, metavar="METRICS", help="print a metrics.jsonl summary")
    p.add_argument("--minutes", type=float, default=None, help="wall-clock budget")
    p.add_argument("--updates", type=int, default=None, help="number of PPO updates")
    g = p.add_argument_group("environment and rollout")
    g.add_argument("--envs", type=int, default=32, help="environment slots = rollout columns B (default 32)")
    g.add_argument("--env-threads", type=int, default=2, help="C++ worker threads (default 2)")
    g.add_argument("--steps", type=int, default=64, help="rows per column per rollout, T (default 64)")
    g.add_argument("--event-length", type=int, default=64, help="event tokens per observation (default 64)")
    g.add_argument("--max-turns", type=int, default=None)
    g.add_argument("--max-decisions", type=int, default=None)
    g = p.add_argument_group("network")
    g.add_argument("--d-model", type=int, default=64)
    g.add_argument("--layers", type=int, default=1, help="board and history Transformer layers (default 1)")
    g.add_argument("--history", default="transformer", choices=("transformer", "lstm", "none"))
    g.add_argument("--no-id-embedding", action="store_true", help="drop the per-card ID embedding")
    g.add_argument("--separate-critic", action="store_true", help="critic gets its own trunk")
    g.add_argument("--no-privileged", action="store_true", help="non-privileged critic (ablation)")
    g = p.add_argument_group("PPO")
    g.add_argument("--objective", default="ppo_clip")
    g.add_argument("--estimator", default="vrpo", choices=("vrpo", "gae"))
    g.add_argument("--entropy", type=float, default=0.05, help="entropy coefficient (design: 0.05-0.2)")
    g.add_argument("--kl-ref", type=float, default=0.05, help="KL coefficient to the EMA reference")
    g.add_argument("--ema", type=float, default=0.02, help="reference EMA rate per update")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--epochs", type=int, default=2)
    g.add_argument("--minibatch", type=int, default=512)
    g.add_argument("--bc-prior", default=None, metavar="CKPT", help="BC checkpoint used as a KL prior")
    g.add_argument("--kl-prior", type=float, default=0.0, help="KL coefficient to the BC prior")
    g.add_argument("--init-from", default=None, metavar="CKPT", help="initialize the actor from a checkpoint")
    g = p.add_argument_group("league and evaluation")
    g.add_argument("--selfplay-fraction", type=float, default=0.75)
    g.add_argument("--pool-size", type=int, default=8)
    g.add_argument("--snapshot-every", type=int, default=10)
    g.add_argument("--checkpoint-every", type=int, default=10)
    g.add_argument("--eval-every", type=int, default=25)
    g.add_argument("--eval-pairs", type=int, default=8, help="paired seeds per deck pairing and baseline")
    g.add_argument("--eval-opponents", default="greedy,random")
    g.add_argument("--keep-best-by", default="greedy")
    g.add_argument("--eval-workers", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--torch-threads", type=int, default=4)
    g.add_argument("--collect-threads", type=int, default=2)
    return p


def config_from_args(args, decks: list[str]):
    from ygorl.train.ppo import PPOConfig
    from ygorl.train.trainer import TrainConfig

    net = {"d_model": args.d_model, "n_heads": 4, "board_layers": args.layers, "history_layers": args.layers,
           "history": args.history, "id_embedding": not args.no_id_embedding}  # fmt: skip
    ppo = PPOConfig(objective=args.objective, estimator=args.estimator, entropy_coef=args.entropy,
                    kl_ref_coef=args.kl_ref, reference_ema=args.ema, lr=args.lr, epochs=args.epochs,
                    minibatch_size=args.minibatch, kl_prior_coef=args.kl_prior)  # fmt: skip
    return TrainConfig(decks=tuple(decks), pairings=args.pairings, env=args.env, max_turns=args.max_turns,
                       max_decisions=args.max_decisions, num_envs=args.envs, env_threads=args.env_threads,
                       steps=args.steps, event_length=args.event_length, net=net,
                       privileged_critic=not args.no_privileged, shared_backbone=not args.separate_critic, ppo=ppo,
                       selfplay_fraction=args.selfplay_fraction, pool_size=args.pool_size,
                       snapshot_every=args.snapshot_every, checkpoint_every=args.checkpoint_every,
                       eval_every=args.eval_every, eval_pairs=args.eval_pairs,
                       eval_opponents=tuple(s for s in args.eval_opponents.split(",") if s),
                       keep_best_by=args.keep_best_by, eval_workers=args.eval_workers, seed=args.seed,
                       torch_threads=args.torch_threads, collect_threads=args.collect_threads,
                       bc_prior=args.bc_prior, init_from=args.init_from)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("train_ppo: PyTorch is missing; run `uv sync --extra train`", file=sys.stderr)
        return 2
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
    print(trainer.model.actor.parameter_report())
    trainer.train(max_updates=args.updates, max_minutes=args.minutes)
    print(json.dumps(summarize_metrics(trainer.run_dir / "metrics.jsonl"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
