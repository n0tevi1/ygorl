"""`ygorl matrix DECK...`: deck-vs-deck win-rate matrix, its Nash mixture and alpha-rank."""

from __future__ import annotations

import argparse
import re
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
        "matrix",
        help="deck-vs-deck matchup matrix with Nash mixture and alpha-rank",
        description="Play every pair of decks with the same agent on both sides (paired seeds), then solve the "
        "meta game: Nash mixture and alpha-rank. With --env and no DECK the environment's meta decks are used and "
        "the result is written to environments/<version>/artifacts/matrix/NAME.json (see docs/cli.md).",
    )
    p.add_argument("decks", type=Path, nargs="*", metavar="DECK",
                   help=".ydk file or directory of .ydk files (default: the meta decks of --env)")  # fmt: skip
    p.add_argument("--agent", default="greedy", metavar="AGENT", help=f"agent piloting every deck: {agents_help()}")
    p.add_argument("--games", type=int, default=20, metavar="N",
                   help="games per deck pair, rounded up to an even number (default 20)")  # fmt: skip
    p.add_argument("--workers", type=int, default=1, metavar="N", help="worker processes (default 1)")
    p.add_argument("--seed", type=int, default=0, help="matrix seed (default 0)")
    p.add_argument("--confidence", type=float, default=0.95, help="level of the per-cell Wilson interval (default 0.95)")
    p.add_argument("--alpha", type=float, default=None, help="alpha-rank selection intensity (default 10)")
    p.add_argument("--population", type=int, default=None, metavar="M", help="alpha-rank population size (default 50)")
    add_env_option(p, "environment: rules and meta decks; the result is stamped with it (default: none)")
    add_max_turns_option(p)
    p.add_argument("--out", type=Path, default=None, metavar="PATH", help="write the result as JSON to PATH")
    p.add_argument("--name", default=None,
                   help="artifact name under the environment's artifacts/matrix/ (default: the agent)")  # fmt: skip
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from ygorl.commands import make_factory
    from ygorl.eval.matchup import DEFAULT_ALPHA, DEFAULT_POPULATION, analyze, build_matrix

    env = load_env(args.env)
    if args.decks:
        decks = {}
        for deck in load_decks(args.decks):
            if deck.name in decks:
                raise CommandError(f"duplicate deck name {deck.name!r}: deck names (file stems) must be unique")
            decks[deck.name] = deck
    elif env is not None and env.meta_decks:
        decks = {md.name: md.deck for md in env.meta_decks}
    else:
        raise CommandError("no decks: pass DECK... or an --env with meta decks")
    if len(decks) < 2:
        raise CommandError("a matrix needs at least two decks")
    if args.games < 1 or args.workers < 1:
        raise CommandError("--games and --workers must be at least 1")
    if args.name is not None and env is None:
        raise CommandError("--name names an environment artifact; pass --env (or use --out)")
    config = duel_config(env, args.max_turns)
    factory = make_factory(args.agent)
    alpha = DEFAULT_ALPHA if args.alpha is None else args.alpha
    population = DEFAULT_POPULATION if args.population is None else args.population

    pairs = (args.games + 1) // 2
    t0 = time.time()
    matrix = build_matrix(decks, factory, pairs=pairs, seed=args.seed, env=env, config=config, workers=args.workers,
                          confidence=args.confidence)  # fmt: skip
    elapsed = time.time() - t0
    try:
        meta = analyze(matrix, alpha=alpha, population_size=population)
    except ValueError as exc:
        raise CommandError(str(exc)) from None

    n = len(matrix.decks)
    cells = n * (n - 1) // 2
    workers = f"{args.workers} worker{'s' if args.workers != 1 else ''}"
    lines = [
        f"{n} decks, {cells} deck pairs x {2 * pairs} games = {cells * 2 * pairs} games in {elapsed:.1f}s "
        f"with {workers}; agent {args.agent}, seed {args.seed} (environment: {env.version if env is not None else 'none'})",
        "win rate of the row deck against the column deck (draws count half); "
        f"alpha-rank alpha {alpha:g}, population {population}",
        "",
    ]
    lines += _table(matrix.decks, matrix.win_rate, meta.nash, meta.alpha_rank)
    if matrix.errors:
        lines.append(f"\nwarning: {matrix.errors} games raised (counted as draws)")
    try:
        if args.out is not None:
            path = meta.save(args.out)
        elif env is not None:
            path = meta.save(env=env, name=args.name or re.sub(r"[^A-Za-z0-9._-]+", "-", args.agent))
        else:
            path = None
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None
    if path is not None:
        lines += ["", f"written to {path}"]
    print("\n".join(lines))
    return 1 if matrix.errors else 0


def _table(names, win_rate, nash, alpha_rank) -> list[str]:
    first = max(len("deck"), *map(len, names))
    widths = [max(5, len(x)) for x in names]
    head = f"{'deck':<{first}}  " + "  ".join(f"{x:>{w}}" for x, w in zip(names, widths)) + "   nash  alpha_rank"
    rows = [head]
    for i, name in enumerate(names):
        cells = ("-" if i == j else f"{win_rate[i][j]:.3f}" for j in range(len(names)))
        rows.append(f"{name:<{first}}  " + "  ".join(f"{c:>{w}}" for c, w in zip(cells, widths))
                    + f"  {nash[i]:.3f}  {alpha_rank[i]:>10.3f}")  # fmt: skip
    return rows
