"""Throughput of the look-ahead opponent (ygorl.train.lookahead; first-seat plan §13).

The learner (CKPT) moves first against OPPONENT on the panel's meta decks; the same games are played with the plain
opponent and with the opponent looking ahead on its turn 2 (best-of-K). Reports wall time per game, look-ahead
decisions per game (qualifying / rolled out / changed), decisions stepped in rollouts, and the added wall time per
rolled-out decision (amortised over the slots running in parallel).

  uv run --no-sync python tools/lookahead_bench.py CKPT OPPONENT [--games 256] [--k 8] [--envs 128] [--threads 16]
"""

from __future__ import annotations

import argparse
import json
import time


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("opponent")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--candidates", choices=("sample", "top"), default="sample")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026102702)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

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
    env = EncodedVecEnv(a.envs, a.threads, cards=cards, vocab=learner.vocab, privileged=True,
                        event_length=learner.event_length, skip_forced=True, snapshots=True,
                        selection_history=learner.net_config.selection_history,
                        attack_history=learner.net_config.attack_history)  # fmt: skip
    report = {"games": a.games, "k": a.k, "envs": a.envs, "threads": a.threads, "torch_threads": a.torch_threads}
    for name, cfg in (("plain", None), ("lookahead", look)):
        games = [GameSetup(GameSpec(seed=s, deck_a=pair[0], deck_b=pair[1], first=0, config=config),
                           (SeatPlayer(learner.net), SeatPlayer(opp.net, cfg)), key=s) for s, pair, _ in slots]  # fmt: skip
        t0 = time.perf_counter()
        outcomes, timing = play(env, games)
        wall = time.perf_counter() - t0
        q = sum(o.stats.qualifying for o in outcomes)
        b = sum(o.stats.branched for o in outcomes)
        report[name] = {"seconds": wall, "seconds_per_game": wall / len(games), "qualifying_per_game": q / len(games),
                        "branched_per_game": b / len(games),
                        "changed_per_game": sum(o.stats.changed for o in outcomes) / len(games),
                        "rollout_steps_per_branched": sum(o.stats.rollout_steps for o in outcomes) / max(1, b),
                        "main_decisions_per_game": timing["main_decisions"] / len(games),
                        "forward_rows": timing["forward_rows"], "forward_seconds": timing["forward_seconds"],
                        "learner_first_score": sum(o.score(0) for o in outcomes) / len(games)}  # fmt: skip
        print(name, json.dumps(report[name]), flush=True)
    extra = report["lookahead"]["seconds"] - report["plain"]["seconds"]
    report["added_seconds_per_branched_decision"] = extra / max(1, report["lookahead"]["branched_per_game"] * a.games)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
