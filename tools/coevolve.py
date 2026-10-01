"""Co-evolution loop: evolution rounds, RL training segments and value-model refits in turn (#151,
ygorl.build.coevolve; docs/tuning.md「共同进化循环」).

  .venv/bin/python tools/coevolve.py --env md-2026-09 --checkpoint CKPT --state DIR \\
      --deck therion=BASE.ydk ... [--cycles 3] [--updates 50] --deck-model MODEL.pt \\
      [--paired SRC[@CKPT] ...] [--warm-start PATH ...] [--learned-generator 8] [--rules 2] \\
      [--value-model-weight 0] [--device cuda] [--envs 256] [--evolve-arg=--budget=20000 ...]

Each cycle: one round of ``tools/evolve_decks.py`` on the current version of every ``--deck`` (state ``DIR/evo``; its
manifest ``DIR/evo/manifest.json`` is the training's deck pool), a training segment continuing ``--checkpoint`` to
``start + cycle × --updates`` updates in ``DIR/train`` with ``--deck-pool`` and ``--log-games``, and a refit of the
value model (``tools/fit_value_model.py``) on the ``--paired`` sources, the evolution state and the training games,
for the new policy (``DIR/valuemodel/cycle_NN.pt``). A value model is fitted before the first round as well.

Interrupted? Rerun the same command: finished stages are skipped (``DIR/coevo.json``), the evolution round resumes
from its game logs and the training segment from ``DIR/train/checkpoints/latest.pt``. Stage output goes to
``DIR/logs/``. The summary (games per +1 pp of validated gain) is printed and written to ``DIR/summary.json``.

``train-segment`` is the training stage on its own (run by the loop in a fresh process).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def train_segment(argv: list[str]) -> int:
    """``train-segment --from CKPT --run DIR --until N --deck-pool MANIFEST [--eval-every N]``: continue a run
    to update N in DIR (from DIR's latest checkpoint when there is one), the manifest as its evolved deck pool and
    every game logged; saves ``DIR/checkpoints/update_N.pt`` and prints its path last."""
    ap = argparse.ArgumentParser(prog="coevolve.py train-segment")
    ap.add_argument("--from", dest="source", required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--until", type=int, required=True)
    ap.add_argument("--deck-pool", type=Path, required=True)
    ap.add_argument("--eval-every", type=int, default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    from ygorl.train.rollout import RolloutStalled
    from ygorl.train.trainer import Trainer

    latest = args.run / "checkpoints" / "latest.pt"
    source = latest if latest.is_file() else Path(args.source)
    over = {"deck_pool": str(args.deck_pool.resolve()), "log_games": True}
    if args.eval_every is not None:
        over["eval_every"] = args.eval_every
    trainer = Trainer.resume(source, args.run, **over)
    have = trainer.counters["updates"]
    print(f"from {source} at update {have}, to {args.until}", flush=True)
    if have > args.until:
        raise SystemExit(f"{source} is past update {args.until}")
    try:
        if have < args.until:
            trainer.train(max_updates=args.until - have)
    except RolloutStalled as exc:
        print(f"error: {exc}; rerun to resume", file=sys.stderr, flush=True)
        os._exit(3)
    if trainer.counters["updates"] != args.until:
        raise SystemExit(f"stopped at update {trainer.counters['updates']} before {args.until}: rerun to resume")
    path = trainer.save(f"update_{args.until:06d}.pt")
    print(path, flush=True)
    return 0


class CommandStages:
    """The real stages: subprocesses of this repository's tools (module docstring)."""

    def __init__(self, args) -> None:
        self.a = args
        self.dir = args.state
        self.evo = self.dir / "evo"
        self.run = self.dir / "train"
        self.logs = self.dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.py = sys.executable
        self.base_run = os.path.normpath(str(Path(args.checkpoint).parent.parent))

    def evolve(self, cycle, checkpoint, value_model, parents):
        from ygorl.build.coevolve import EvolveResult, accepted_children, check_round, round_done, run_command

        a = self.a
        check_round(self.evo, cycle)
        if not round_done(self.evo, cycle):
            cmd = [self.py, str(ROOT / "tools" / "evolve_decks.py"), "--env", a.env, "--checkpoint", checkpoint,
                   "--state", str(self.evo), "--device", a.device, "--envs", str(a.envs), "--seed", str(a.seed),
                   "--deck-model", str(a.deck_model)]  # fmt: skip
            for f in parents.values():
                cmd += ["--parent", f]
            if a.learned_generator:
                cmd += ["--learned-generator", str(a.learned_generator)]
            if a.rules:
                cmd += ["--rules", str(a.rules)]
            if value_model:
                cmd += ["--value-model", value_model, "--value-model-weight", str(a.value_model_weight)]
            for w in a.warm_start:
                cmd += ["--warm-start", w]
            cmd += a.evolve_arg
            run_command(cmd, self.logs / f"evolve-{cycle:02d}.log")
        games, accepted = accepted_children(self.evo, cycle, parents)
        return EvolveResult(cycle, games, accepted)

    def train(self, cycle, checkpoint, until):
        from ygorl.build.coevolve import run_command

        cmd = [self.py, str(ROOT / "tools" / "coevolve.py"), "train-segment", "--from", checkpoint, "--run",
               str(self.run), "--until", str(until), "--deck-pool", str(self.evo / "manifest.json")]  # fmt: skip
        if self.a.eval_every is not None:
            cmd += ["--eval-every", str(self.a.eval_every)]
        (self.evo).mkdir(parents=True, exist_ok=True)
        if not (self.evo / "manifest.json").is_file():  # no round has written it yet: an empty pool
            (self.evo / "manifest.json").write_text(json.dumps({"format": "ygorl-deck-pool", "version": 1,
                                                                "decks": []}) + "\n")  # fmt: skip
        run_command(cmd, self.logs / f"train-{cycle:02d}.log")
        path = self.run / "checkpoints" / f"update_{until:06d}.pt"
        if not path.is_file():
            raise RuntimeError(f"the training segment did not write {path}")
        return str(path)

    def fit(self, cycle, checkpoint):
        from ygorl.build.coevolve import run_command

        a = self.a
        out = self.dir / "valuemodel" / f"cycle_{cycle:02d}.pt"
        cmd = [self.py, str(ROOT / "tools" / "fit_value_model.py"), "--env", a.env, "--deck-model", str(a.deck_model),
               "--target-checkpoint", checkpoint, "--same-run", self.base_run, "--same-run", str(self.run),
               "--out", str(out), "--seed", str(a.seed)]  # fmt: skip
        for p in a.paired:
            cmd += ["--paired", p]
        if (self.evo / "lineage.jsonl").is_file():
            cmd += ["--paired", str(self.evo)]
        if (self.run / "games.jsonl.gz").is_file():
            cmd += ["--games", str(self.run)]
        cmd += a.fit_arg
        run_command(
            cmd, self.logs / f"fit-{cycle:02d}.log", env={"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
        )
        return str(out)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "train-segment":
        return train_segment(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--checkpoint", required=True, help="the starting policy (RUN/checkpoints/update_N.pt)")
    ap.add_argument("--state", type=Path, required=True, help="co-evolution directory")
    ap.add_argument("--deck", action="append", default=[], metavar="SLOT=FILE", help="a deck of the pool")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--updates", type=int, default=50, help="training updates per cycle")
    ap.add_argument("--deck-model", type=Path, required=True)
    ap.add_argument("--paired", action="append", default=[], metavar="SRC[@CKPT]",
                    help="paired evaluations for the value model (tools/fit_value_model.py)")  # fmt: skip
    ap.add_argument("--warm-start", action="append", default=[], help="card-value model warm start (evolve_decks)")
    ap.add_argument("--learned-generator", type=int, default=8)
    ap.add_argument("--rules", type=int, default=0)
    ap.add_argument("--value-model-weight", type=float, default=0.0,
                    help="the value model's weight until the calibration table has evidence (default 0: the table "
                    "decides; a positive weight is an experiment: it ranked worse than the card-value model in "
                    "cross-validation)")  # fmt: skip
    ap.add_argument("--no-initial-fit", action="store_true", help="no value model in the first round")
    ap.add_argument("--evolve-arg", action="append", default=[], help="extra evolve_decks.py argument (repeatable)")
    ap.add_argument("--fit-arg", action="append", default=[], help="extra fit_value_model.py argument (repeatable)")
    ap.add_argument("--eval-every", type=int, default=None, help="training evaluation interval (default: the run's)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.deck:
        ap.error("give at least one --deck SLOT=FILE")

    from ygorl.build.coevolve import CoEvolution, CoevoConfig
    from ygorl.build.edit_labels import CheckpointRef

    decks = {}
    for d in args.deck:
        slot, _, f = d.partition("=")
        if not f or not Path(f).is_file():
            ap.error(f"--deck {d}: SLOT=FILE with an existing file")
        decks[slot] = f
    start = CheckpointRef.from_path(args.checkpoint).update
    if start is None:
        ap.error("--checkpoint must be RUN/checkpoints/update_N.pt (its update starts the count)")
    cfg = CoevoConfig(cycles=args.cycles, updates=args.updates, start_update=start, decks=decks,
                      initial_fit=not args.no_initial_fit)  # fmt: skip

    def say(msg):
        print(msg, flush=True)

    loop = CoEvolution(args.state, CommandStages(args), cfg, args.checkpoint, log=say)
    summary = loop.run()
    (args.state / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for row in summary["cycles"]:
        say(f"cycle {row['cycle']}: {row['games']} evolution games, {row['accepted']} accepted "
            f"(validated {row['validated_gain'] * 100:+.2f} pp)")  # fmt: skip
    gpp = summary["games_per_pp"]
    say(f"total: {summary['games']} games, {summary['accepted']} accepted, "
        + (f"{gpp:.0f} games per +1 pp validated" if gpp else "no validated gain"))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
