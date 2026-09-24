"""Policy-vs-policy evaluation over deck pools on the batched C++ path (ygorl.eval.batched, docs/evaluation.md).

  uv run python tools/eval_batched.py CKPT --decks DIR [--opponents DIR] [--opponent-checkpoint CKPT2] \\
      [--pairings 200] [--pairs 1] [--device cuda] [--envs 256] [--out report.json]

The policy of CKPT pilots deck i of ``--decks``, the policy of ``--opponent-checkpoint`` (default: the same one)
pilots deck j of ``--opponents`` (default: the same pool); ``--pairings`` (i, j) pairs with i != j are drawn once
with ``--seed``, each played ``--pairs`` times from both sides on the arena's paired seeds. Prints the overall win
rate of CKPT's side with its Wilson interval and the rate per deck it piloted; ``--out`` writes the report
(``ArenaReport.to_dict()`` plus ``per_deck`` and ``stats``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def deck_files(spec: str) -> list[Path]:
    p = Path(spec)
    return sorted(p.glob("*.ydk")) if p.is_dir() else sorted(Path().glob(spec))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--decks", required=True, help="directory or glob of .ydk files piloted by CHECKPOINT")
    ap.add_argument("--opponents", default=None, help="directory or glob of opponent decks (default: --decks)")
    ap.add_argument("--opponent-checkpoint", default=None)
    ap.add_argument("--pairings", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-decisions", type=int, default=4000)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np
    import torch

    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import DuelConfig
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.eval.arena import derive_seed, summarize
    from ygorl.eval.batched import paired_specs, play_policies
    from ygorl.train.checkpoint import load_actor, vocab_passwords

    mine = [load_ydk(p) for p in deck_files(args.decks)]
    theirs = [load_ydk(p) for p in deck_files(args.opponents)] if args.opponents else mine
    if not mine or not theirs:
        raise SystemExit("no decks found")
    if theirs is mine and len(mine) < 2:
        raise SystemExit("one deck against itself: give at least two decks or --opponents")
    pol = load_actor(args.checkpoint)
    opp = load_actor(args.opponent_checkpoint) if args.opponent_checkpoint else pol
    if vocab_passwords(opp.vocab) != vocab_passwords(pol.vocab) or opp.event_length != pol.event_length:
        raise SystemExit("the two checkpoints need the same card vocab and event length")
    device = torch.device(args.device)
    net_a = pol.net.to(device)
    net_b = net_a if opp is pol else opp.net.to(device)

    rng = np.random.default_rng(derive_seed(args.seed, 0))
    cells: list[tuple[int, int]] = []
    while len(cells) < args.pairings:
        i, j = int(rng.integers(len(mine))), int(rng.integers(len(theirs)))
        if mine is not theirs or i != j:
            cells.append((i, j))
    config = DuelConfig(max_decisions=args.max_decisions)
    per_cell = 2 * args.pairs
    specs = [s for k, (i, j) in enumerate(cells)
             for s in paired_specs(mine[i], theirs[j], args.pairs, derive_seed(args.seed, 1, k), config)]  # fmt: skip
    env = EncodedVecEnv(min(args.envs, len(specs)), args.threads, vocab=pol.vocab, event_length=pol.event_length,
                        skip_forced=True)  # fmt: skip
    records, stats = play_policies(env, specs, net_a, net_b, device=device, greedy=args.greedy, pairs_per_spec=2)
    rep = summarize(records, agent_a=f"policy:{args.checkpoint}", agent_b=f"policy:{args.opponent_checkpoint or args.checkpoint}",
                    deck_a="*", deck_b="*", seed=args.seed)  # fmt: skip
    by_deck: dict[str, list[float]] = defaultdict(list)
    for k, (i, _j) in enumerate(cells):
        for r in records[k * per_cell : (k + 1) * per_cell]:
            by_deck[mine[i].name].append(1.0 if r.winner == 0 else 0.5 if r.winner is None else 0.0)
    per_deck = {d: {"games": len(v), "win_rate": sum(v) / len(v)} for d, v in sorted(by_deck.items())}
    print(f"{rep.games} games in {stats['seconds']:.1f}s ({stats['games_per_s']:.1f} games/s, "
          f"{stats['decisions_per_s']:,.0f} decisions/s, {stats['forwards']} forwards): win rate {rep.win_rate:.3f} "
          f"({rep.ci[0]:.3f}-{rep.ci[1]:.3f}), first {rep.as_first.win_rate:.3f}, second {rep.as_second.win_rate:.3f}, "
          f"reasons {rep.reasons}")  # fmt: skip
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**rep.to_dict(), "per_deck": per_deck, "stats": stats}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
