"""Collection throughput with and without window branching (T4/T5; docs/training.md §8.1 「窗口分支」).

  uv run --no-sync python tools/bench_branch_collect.py RUN/config.json --init-from CKPT \\
      --line-recovery FIRE.jsonl [--envs 128 --steps 128 --rounds 2] [--json OUT.jsonl]

Builds the run's training setup (its decks, scripted lines, league shares) on the CPU, one learner snapshot in the
pool, and collects ``--rounds`` rollouts per arm, the arms interleaved so machine load drifts hit them alike:
``plain`` (no snapshots, no branching: what training does today), ``branch`` (window branching, the default
``BranchConfig``: K = 2, M = 1, board horizon, windows with a recovery line only) and ``branch_all`` (the same without
the recovery requirement: every qualifying window on the line's turn), ``branch_quarter`` (that, with
``fraction = 0.25``). Reports learner rows per second of collection,
branched windows and branch decisions per rollout; only collection is timed (no update, no GPU).
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", help="a run's config.json (decks, lines, league shares, network)")
    p.add_argument("--init-from", required=True, help="checkpoint the acting network starts from")
    p.add_argument("--line-recovery", action="append", default=[], help="--fire records of the lines (repeatable)")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--env-threads", type=int, default=None, help="default: the run's")
    p.add_argument("--collect-threads", type=int, default=None, help="default: the run's")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--arms", default="plain,branch,branch_all,branch_quarter")
    p.add_argument("--json", default=None, help="append one line per arm and round")
    a = p.parse_args(argv)

    import torch

    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.eval.arena import derive_seed
    from ygorl.train.rollout import RolloutCollector
    from ygorl.train.trainer import TrainConfig, Trainer

    data = json.loads(Path(a.config).read_text())
    cfg = TrainConfig.from_dict(data)
    cfg = replace(cfg, device="cpu", bf16=False, num_envs=a.envs, steps=a.steps, eval_every=0, checkpoint_every=0,
                  snapshot_every=0, log_games=False, trace_games=False, init_from=a.init_from, critic_warmup=0,
                  line_recovery=tuple(a.line_recovery) or cfg.line_recovery,
                  env_threads=a.env_threads or cfg.env_threads,
                  collect_threads=a.collect_threads or cfg.collect_threads)  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(cfg, tmp, log=print)
    trainer.pool.add(trainer.model, update=0)
    torch.set_num_threads(cfg.collect_threads)
    arms = {
        "plain": None,
        "branch": cfg.branch_config(),
        "branch_all": replace(cfg.branch_config(), require_recovery=False),
        "branch_quarter": replace(cfg.branch_config(), require_recovery=False, fraction=0.25),
    }
    collectors = {}
    for name in a.arms.split(","):
        branch = arms[name]
        env = EncodedVecEnv(cfg.num_envs, cfg.env_threads, cards=trainer.cards, vocab=trainer.vocab,
                            privileged=cfg.privileged_critic, event_length=cfg.event_length, skip_forced=cfg.skip_forced,
                            record_actions=True, selection_history=trainer.net_config.selection_history,
                            attack_history=trainer.net_config.attack_history, cancel_budget=cfg.cancel_budget,
                            snapshots=branch is not None)  # fmt: skip
        collectors[name] = RolloutCollector(env, trainer.model, trainer.schedule, cfg.steps,
                                            opponents=trainer.schedule.opponent, seed=derive_seed(cfg.seed, 3),
                                            min_batch=cfg.min_batch, branch=branch)  # fmt: skip
    out = []
    for r in range(a.rounds):
        for name, col in collectors.items():
            t0 = time.perf_counter()
            ro = col.collect()
            seconds = time.perf_counter() - t0
            rec = {"arm": name, "round": r, "rows": ro.players.numel(), "seconds": round(seconds, 2),
                   "rows_per_s": round(ro.players.numel() / seconds, 1), "decisions": ro.decisions,
                   "branch_windows": ro.branch_windows, "branch_decisions": ro.branch_decisions,
                   "branch_valid_rows": int(ro.branch_valid.sum()) if ro.branch_valid is not None else 0,
                   "line_games": sum(g.line is not None for g in ro.games), "games": len(ro.games)}  # fmt: skip
            print(json.dumps(rec), flush=True)
            out.append(rec)
            if a.json:
                with open(a.json, "a") as f:
                    f.write(json.dumps({**rec, "config": a.config, "init_from": a.init_from}) + "\n")
    for name in collectors:
        rates = [x["rows_per_s"] for x in out if x["arm"] == name]
        print(f"{name}: {statistics.mean(rates):.1f} rows/s over {len(rates)} rollouts")
    return out


if __name__ == "__main__":
    main()
