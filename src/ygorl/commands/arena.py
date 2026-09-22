"""`ygorl arena DECK...`: paired-seed games of agent a against agent b over deck pairings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ygorl.commands import (
    CommandError,
    add_env_option,
    add_max_turns_option,
    agents_help,
    duel_config,
    load_decks,
    load_env,
)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "arena",
        help="win rate of one agent against another on paired seeds",
        description="Agent a pilots each deck on its side against each deck on agent b's side: without --vs both "
        "sides use the same decks (every ordered pairing, mirrors included), with --vs agent a pilots DECK... and "
        "agent b the --vs decks. --games is spread evenly over the pairings, rounded to whole pairs (two games per "
        "seed, first player swapped). Prints agent a's win rate with its Wilson interval (see docs/cli.md).",
    )
    p.add_argument("decks", type=Path, nargs="+", metavar="DECK", help=".ydk file or directory of .ydk files")
    p.add_argument("--vs", type=Path, nargs="+", default=None, metavar="DECK", help="decks of agent b (default: DECK...)")
    p.add_argument("--agent-a", default="greedy", metavar="AGENT", help=f"agent a (default greedy): {agents_help()}")
    p.add_argument("--agent-b", default="random", metavar="AGENT", help="agent b (default random)")
    p.add_argument("--games", type=int, default=200, metavar="N", help="total games (default 200)")
    p.add_argument("--workers", type=int, default=1, metavar="N", help="worker processes (default 1)")
    p.add_argument("--seed", type=int, default=0, help="arena seed (default 0)")
    p.add_argument("--confidence", type=float, default=0.95, help="level of the Wilson interval (default 0.95)")
    add_env_option(p, "environment: rules, LP and hand size; stamped into the report (default: none)")
    add_max_turns_option(p)
    p.add_argument("--out", type=Path, default=None, metavar="PATH", help="write the pooled report with every game as JSON")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from ygorl.commands import make_factory
    from ygorl.eval.arena import Arena, derive_seed, merge

    side_a = load_decks(args.decks)
    side_b = load_decks(args.vs) if args.vs is not None else side_a
    env = load_env(args.env)
    config = duel_config(env, args.max_turns)
    if args.games < 1 or args.workers < 1:
        raise CommandError("--games and --workers must be at least 1")
    if not 0 < args.confidence < 1:
        raise CommandError("--confidence must be between 0 and 1")
    factory_a, factory_b = make_factory(args.agent_a), make_factory(args.agent_b)

    cells = [(a, b, derive_seed(args.seed, i, j)) for i, a in enumerate(side_a) for j, b in enumerate(side_b)]
    pairs = max(1, round(args.games / (2 * len(cells))))
    arena = Arena(factory_a, factory_b, env=env, config=config, workers=args.workers, confidence=args.confidence)
    t0 = time.time()
    reports = arena.run_many(cells, pairs)
    elapsed = time.time() - t0
    pooled = merge(reports, deck_a=_label(side_a), deck_b=_label(side_b))

    workers = f"{args.workers} worker{'s' if args.workers != 1 else ''}"
    lines = [
        f"{len(cells)} deck pairing{'s' if len(cells) != 1 else ''} x {2 * pairs} games = {pooled.games} games "
        f"in {elapsed:.1f}s with {workers} (environment: {env.version if env is not None else 'none'})",
        pooled.summary(),
        f"reasons: {pooled.reasons}; mean turns {pooled.mean_turns:.1f}",
    ]
    for title, key in ((f"{args.agent_a} win rate by the deck it piloted", "deck_a"),
                       (f"{args.agent_a} win rate by the opponent's deck", "deck_b")):  # fmt: skip
        groups: dict[str, list] = {}
        for rep in reports:
            groups.setdefault(getattr(rep, key), []).append(rep)
        if len(groups) > 1:
            lines += ["", f"{title}:"]
            width = max(len(n) for n in groups)
            for name, reps in groups.items():
                r = merge(reps, deck_a=name)
                lines.append(f"  {name:<{width}}  {r.win_rate:.3f}  ({r.ci[0]:.3f}-{r.ci[1]:.3f}, {r.games} games)")
    if args.out is not None:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(pooled.to_dict(), indent=1) + "\n", encoding="utf-8")
        except OSError as exc:
            raise CommandError(str(exc)) from None
        lines += ["", f"report written to {args.out}"]
    unhealthy = pooled.errors or pooled.retries or pooled.unknown_messages
    if unhealthy:
        lines.append(f"warning: {pooled.errors} games raised, {pooled.retries} retries, "
                     f"{pooled.unknown_messages} unknown messages")  # fmt: skip
    print("\n".join(lines))
    return 1 if unhealthy else 0


def _label(decks) -> str:
    """Deck label of one side in the pooled summary: its name if the side has one deck, else ``*``."""
    names = {d.name for d in decks}
    return names.pop() if len(names) == 1 else "*"
