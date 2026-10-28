"""Throughput of the look-ahead opponent (ygorl.train.lookahead; first-seat plan §13.4).

Evaluation driver (``play``): the learner (CKPT) moves first against OPPONENT on the panel's meta decks; the same
games are played with the plain opponent and with the opponent looking ahead on its turn 2 (best-of-K), with
``--forks`` spare slots the candidates roll out side by side. Reports wall time per game, look-ahead decisions per
game (qualifying / rolled out / changed), decisions stepped in rollouts, forward calls and rows, and the added wall
time per rolled-out decision (amortised over the slots running in parallel).

  uv run --no-sync python tools/lookahead_bench.py CKPT OPPONENT [--games 256] [--k 8] [--envs 128] [--forks 64]

Training collector (``--collect RUN/config.json``): CKPT is the acting network, ``--pool`` copies of it are the
snapshot pool, no scripted lines (``--lookahead-share 0.5`` then deals look-ahead games as often as a run with
``line_share 0.5`` and share 1 does); one rollout per arm and round, the arms interleaved: ``plain`` (no look-ahead),
``seq`` (look-ahead on the game's own slot, no look-only rounds: the collector as first built) and ``look`` (forks and
look-only rounds). Rows per second of collection only (no update). ``YGORL_GPU_FRACTION`` caps the GPU allocator.

  uv run --no-sync python tools/lookahead_bench.py CKPT --collect RUN/config.json [--device cuda] [--rounds 2]
"""

from __future__ import annotations

import argparse
import json
import os
import time


def collect(a) -> int:
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    import torch

    if a.device == "cuda" and os.environ.get("YGORL_GPU_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["YGORL_GPU_FRACTION"]))
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.eval.arena import derive_seed
    from ygorl.train.rollout import RolloutCollector
    from ygorl.train.trainer import TrainConfig, Trainer

    cfg = TrainConfig.from_dict(json.loads(Path(a.collect).read_text()))
    cfg = replace(cfg, device=a.device, bf16=a.device == "cuda", num_envs=a.envs, steps=a.steps, eval_every=0,
                  checkpoint_every=0, snapshot_every=0, log_games=False, trace_games=False, init_from=a.ckpt,
                  critic_warmup=0, line_demos=(), line_recovery=(), line_share=0.0, line_first_share=0.0,
                  lookahead_share=a.lookahead_share, lookahead_k=a.k, env_threads=a.env_threads or cfg.env_threads)  # fmt: skip
    arms = {"plain": (0, 0, 0), "seq": (a.lookahead_share, 0, 0), "look": (a.lookahead_share, a.forks, a.look_rounds)}
    runs = {}
    for name in a.arms.split(","):
        share, forks, rounds = arms[name]
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(cfg, tmp, log=lambda *x: None)
        trainer.schedule.lookahead_share = share or 1e-12  # the same deals in every arm (the draw is made either way)
        for i in range(a.pool):
            trainer.pool.add(trainer.model, update=i)
        env = EncodedVecEnv(cfg.num_envs + forks, cfg.env_threads, cards=trainer.cards, vocab=trainer.vocab,
                            privileged=True, event_length=cfg.event_length, skip_forced=cfg.skip_forced,
                            record_actions=True, selection_history=trainer.net_config.selection_history,
                            attack_history=trainer.net_config.attack_history, cancel_budget=cfg.cancel_budget,
                            snapshots=share > 0)  # fmt: skip
        col = RolloutCollector(env, trainer.model, trainer.schedule, cfg.steps, opponents=trainer.schedule.opponent,
                               seed=derive_seed(cfg.seed, 3), min_batch=cfg.min_batch, device=trainer.device,
                               forks=forks, look_rounds=rounds)  # fmt: skip
        runs[name] = (trainer, col)
    torch.set_num_threads(cfg.collect_threads)
    report = []
    for r in range(a.rounds):
        for name, (trainer, col) in runs.items():
            t0 = time.perf_counter()
            with trainer._autocast():
                ro = col.collect()
            if a.device == "cuda":
                torch.cuda.synchronize()
            s = time.perf_counter() - t0
            rec = {"arm": name, "round": r, "seconds": round(s, 1), "rows": ro.players.numel(), "rows_per_s": round(ro.players.numel() / s, 1),
                   "games": len(ro.games), "decisions": ro.decisions, "lookahead_decisions": ro.lookahead_decisions,
                   "lookahead_branched": ro.lookahead_branched, "lookahead_changed": ro.lookahead_changed,
                   "lookahead_steps": ro.lookahead_steps,
                   "fork_replays": col._forks.replays if col._forks is not None else 0}  # fmt: skip
            print(json.dumps(rec), flush=True)
            report.append(rec)
    for name in runs:
        recs = [x for x in report if x["arm"] == name]
        seconds = sum(x["seconds"] for x in recs)
        print(
            f"{name}: {sum(x['rows'] for x in recs) / seconds:.1f} rows/s over {len(recs)} rollouts ({seconds:.0f} s)"
        )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("opponent", nargs="?")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--candidates", choices=("sample", "top"), default="sample")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--forks", type=int, default=64, help="spare slots for look-ahead forks (0: one slot per game)")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026102702)
    p.add_argument("--out", default=None)
    p.add_argument("--collect", default=None, metavar="CONFIG", help="benchmark the training collector of this run")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--lookahead-share", type=float, default=0.5)
    p.add_argument("--look-rounds", type=int, default=4)
    p.add_argument("--env-threads", type=int, default=None)
    p.add_argument("--arms", default="plain,seq,look")
    a = p.parse_args(argv)
    if a.collect:
        return collect(a)
    if a.opponent is None:
        p.error("the evaluation benchmark needs OPPONENT")

    import torch

    torch.set_num_threads(a.torch_threads)
    from dataclasses import replace

    from ygorl.data import load_environment
    from ygorl.engine.duel import DuelConfig, default_cards
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.env.pool import GameSpec
    from ygorl.eval.agent_matrix import pairing_slots, sample_pairings
    from ygorl.train.checkpoint import load_actor
    from ygorl.train.lookahead import GameSetup, LookaheadConfig, SeatPlayer, play

    environment = load_environment("md-2026-09")
    cards = default_cards()
    learner, opp = load_actor(a.ckpt), load_actor(a.opponent)
    config = DuelConfig.from_environment(environment, max_decisions=4000)
    decks = [m.deck for m in sorted(environment.meta_decks, key=lambda m: m.deck.name)]
    slots = pairing_slots(decks, sample_pairings(len(decks), a.games, a.seed), a.seed, config)  # shuffled per slot
    config = replace(config, shuffle_decks=False)
    look = LookaheadConfig(k=a.k, first_turn=2, last_turn=2, own_turn=True, candidates=a.candidates)
    env = EncodedVecEnv(a.envs + a.forks, a.threads, cards=cards, vocab=learner.vocab, privileged=True,
                        event_length=learner.event_length, skip_forced=True, snapshots=True,
                        selection_history=learner.net_config.selection_history,
                        attack_history=learner.net_config.attack_history)  # fmt: skip
    report = {"games": a.games, "k": a.k, "candidates": a.candidates, "envs": a.envs, "forks": a.forks,
              "threads": a.threads, "torch_threads": a.torch_threads}  # fmt: skip
    arms = [("plain", None, 0), ("lookahead", look, a.forks)]
    if a.forks:
        arms.append(("lookahead_one_slot", look, 0))
    for name, cfg, forks in arms:
        games = [GameSetup(GameSpec(seed=s, deck_a=pair[0], deck_b=pair[1], first=0, config=config),
                           (SeatPlayer(learner.net), SeatPlayer(opp.net, cfg)), key=s) for s, pair, _ in slots]  # fmt: skip
        t0 = time.perf_counter()
        outcomes, timing = play(env, games, slots=a.envs, forks=forks, min_batch=a.envs // 2)
        wall = time.perf_counter() - t0
        q = sum(o.stats.qualifying for o in outcomes)
        b = sum(o.stats.branched for o in outcomes)
        report[name] = {"seconds": wall, "seconds_per_game": wall / len(games), "qualifying_per_game": q / len(games),
                        "branched_per_game": b / len(games),
                        "changed_per_game": sum(o.stats.changed for o in outcomes) / len(games),
                        "rollout_steps_per_branched": sum(o.stats.rollout_steps for o in outcomes) / max(1, b),
                        "main_decisions_per_game": timing["main_decisions"] / len(games),
                        "forward_rows": timing["forward_rows"], "forward_calls": timing["forward_calls"],
                        "forward_seconds": timing["forward_seconds"], "fork_replays": timing["fork_replays"],
                        "learner_first_score": sum(o.score(0) for o in outcomes) / len(games),
                        "actions": [o.actions for o in outcomes]}  # fmt: skip
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "actions"}), flush=True)
    for name, _, _ in arms[2:]:
        assert report[name]["actions"] == report["lookahead"]["actions"], "forks changed the games"
    for name, _, _ in arms:
        del report[name]["actions"]
    extra = report["lookahead"]["seconds"] - report["plain"]["seconds"]
    report["added_seconds_per_branched_decision"] = extra / max(1, report["lookahead"]["branched_per_game"] * a.games)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
