"""Record GreedyAgent's decisions after turn 1 as behaviour-cloning samples (docs/bc.md「补救实验」).

Usage: uv run --frozen python tools/greedy_demos.py [DECK...] --out out/greedy_demos/train.npz
           [--opponents random,greedy] [--pairs 1] [--seed 7001] [--min-turn 2] [--event-length 128] [--workers 2]

Plays GreedyAgent (seat a, both going first and second) against each opponent on every ordered pairing of the
decks (default tests/decks, mirrors included), ``--pairs`` paired seeds each, and records every non-forced
decision of the Greedy seat from turn ``--min-turn`` on (``ygorl.train.heuristic_demos``). The seeds come from
``--seed`` (default 7001), not from the arena's evaluation seed 0, so the recorded games are not the games
``ygorl arena`` evaluates on. Writes an .npz that ``tools/train_bc.py --extra`` reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("decks", type=Path, nargs="*", default=[ROOT / "tests" / "decks"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--opponents", default="random,greedy", help="comma-separated agent specs (default random,greedy)")
    p.add_argument("--pairs", type=int, default=1, help="paired seeds per deck pairing and opponent (2 games each)")
    p.add_argument("--seed", type=int, default=7001)
    p.add_argument("--min-turn", type=int, default=2)
    p.add_argument("--event-length", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="only the first N games (smoke test)")
    args = p.parse_args(argv)

    from ygorl.agents import GreedyAgent
    from ygorl.agents.registry import AgentSpec
    from ygorl.cards.cdb import CardVocab
    from ygorl.commands import duel_config, load_decks
    from ygorl.engine.duel import default_cards
    from ygorl.eval.arena import Arena, derive_seed
    from ygorl.train.heuristic_demos import SUBSETS, record_games, save_data

    decks = load_decks(args.decks)
    vocab = CardVocab.from_db(default_cards())  # same construction as tools/train_bc.py
    specs = []
    for k, opponent in enumerate(s for s in args.opponents.split(",") if s):
        arena = Arena(AgentSpec("greedy"), AgentSpec(opponent), config=duel_config(None, None))
        for i, a in enumerate(decks):
            for j, b in enumerate(decks):
                specs.extend(arena.game_specs(a, b, args.pairs, derive_seed(args.seed, k, i, j)))
    if args.limit:
        specs = specs[: args.limit]
    t0 = time.time()
    data, games = record_games(specs, GreedyAgent, vocab, event_length=args.event_length, min_turn=args.min_turn,
                               workers=args.workers)  # fmt: skip
    seconds = time.time() - t0
    counts = {name: sum(map(f, data.meta)) for name, f in SUBSETS.items()}
    kinds = Counter(f"{m['decision']}/{m['kind']}" for m in data.meta)
    info = {"decks": [d.name for d in decks], "opponents": args.opponents, "pairs": args.pairs, "seed": args.seed,
            "min_turn": args.min_turn, "event_length": args.event_length, "games": len(games),
            "seconds": round(seconds, 1), "subsets": counts, "skipped": dict(data.skipped),
            "greedy_win_rate": {opp: sum(g.get("winner") == 0 for g in games if g.get("opponent") == opp)
                                / max(1, sum(g.get("opponent") == opp for g in games))
                                for opp in {g.get("opponent") for g in games}}}  # fmt: skip
    save_data(args.out, data, **info)
    args.out.with_suffix(".games.json").write_text(json.dumps({"info": info, "games": games,
                                                               "kinds": dict(kinds.most_common())}, indent=1))  # fmt: skip
    print(json.dumps(info, indent=1))
    print(f"{len(data)} samples from {len(games)} games in {seconds:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
