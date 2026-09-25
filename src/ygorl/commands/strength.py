"""`ygorl strength AGENT...`: agent-vs-agent matchup matrix over a deck pool, ranked by alpha-rank and Nash."""

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
from ygorl.commands.matrix import ARTIFACT_NAME_RE, _table

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "strength",
        help="agent-vs-agent matchup matrix: which agent plays best",
        description="Play every pair of AGENTs on the same sample of deck pairings from the deck pool (each pairing "
        "with both deck assignments and both players first: 4 games), then rank the agents by alpha-rank and the "
        "Nash mixture. With --env the result is written to environments/<version>/artifacts/agent-matrix/NAME.json "
        "(see docs/cli.md).",
    )
    p.add_argument("agents", nargs="+", metavar="AGENT",
                   help=f"[NAME=]SPEC, at least two; NAME defaults to the spec: {agents_help()}")  # fmt: skip
    p.add_argument("--decks", type=Path, nargs="+", default=None, metavar="DECK",
                   help=".ydk files or directories: the deck pool (default: the meta decks of --env)")  # fmt: skip
    p.add_argument("--pairings", type=int, default=50, metavar="N",
                   help="deck pairings sampled from the pool; 4 games each per agent pair (default 50)")  # fmt: skip
    p.add_argument("--workers", type=int, default=1, metavar="N", help="worker processes (default 1)")
    p.add_argument("--seed", type=int, default=0, help="seed of the pairing sample and the games (default 0)")
    p.add_argument(
        "--confidence", type=float, default=0.95, help="level of the per-cell Wilson interval (default 0.95)"
    )
    p.add_argument("--alpha", type=float, default=None, help="alpha-rank selection intensity (default 10)")
    p.add_argument("--population", type=int, default=None, metavar="M", help="alpha-rank population size (default 50)")
    add_env_option(p, "environment: rules and meta decks; the result is stamped with it (default: none)")
    add_max_turns_option(p)
    p.add_argument("--out", type=Path, default=None, metavar="PATH", help="write the result as JSON to PATH")
    p.add_argument("--name", default=None,
                   help="artifact name under the environment's artifacts/agent-matrix/ (default: agents)")  # fmt: skip
    p.set_defaults(func=run)


def parse_agents(args: list[str]) -> dict[str, str]:
    """``[NAME=]SPEC`` arguments -> {name: spec}. A ``NAME=`` prefix has no ':' (specs like ``policy:P@t=1`` do)."""
    out: dict[str, str] = {}
    for arg in args:
        name, sep, spec = arg.partition("=")
        if not sep or ":" in name:
            name, spec = arg, arg
        if not spec:
            raise CommandError(f"agent {arg!r} has an empty spec")
        if sep and ":" not in name and not NAME_RE.match(name):
            raise CommandError(f"agent name {name!r}: letters, digits, '.', '_' and '-', not starting with '.' or '-'")
        if name in out:
            raise CommandError(f"duplicate agent name {name!r}: give each agent a NAME= prefix")
        out[name] = spec
    if len(out) < 2:
        raise CommandError("a strength matrix needs at least two agents")
    return out


def run(args: argparse.Namespace) -> int:
    from ygorl.commands import make_factory
    from ygorl.eval.agent_matrix import build_agent_matrix
    from ygorl.eval.matchup import DEFAULT_ALPHA, DEFAULT_POPULATION

    env = load_env(args.env)
    if args.name is not None:
        if env is None:
            raise CommandError("--name names an environment artifact; pass --env (or use --out)")
        if not ARTIFACT_NAME_RE.match(args.name):
            raise CommandError(f"--name {args.name!r} must be a plain file name: letters, digits, '.', '_' and '-', "
                               "not starting with '.' or '-' (no path separators)")  # fmt: skip
    agents = parse_agents(args.agents)
    if args.decks:
        decks = {}
        for deck in load_decks(args.decks, env):
            if deck.name in decks:
                raise CommandError(f"duplicate deck name {deck.name!r}: deck names (file stems) must be unique")
            decks[deck.name] = deck
    elif env is not None and env.meta_decks:
        decks = {md.name: md.deck for md in env.meta_decks}
    else:
        raise CommandError("no deck pool: pass --decks or an --env with meta decks")
    if len(decks) < 2:
        raise CommandError("the deck pool needs at least two decks")
    if args.pairings < 1 or args.workers < 1:
        raise CommandError("--pairings and --workers must be at least 1")
    factories = {name: make_factory(spec) for name, spec in agents.items()}
    alpha = DEFAULT_ALPHA if args.alpha is None else args.alpha
    population = DEFAULT_POPULATION if args.population is None else args.population

    t0 = time.time()
    try:
        matrix = build_agent_matrix(factories, decks, pairings=args.pairings, seed=args.seed, env=env,
                                    config=duel_config(env, args.max_turns), workers=args.workers,
                                    confidence=args.confidence, alpha=alpha, population_size=population)  # fmt: skip
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    elapsed = time.time() - t0

    n = len(matrix.agents)
    cells, games = n * (n - 1) // 2, 4 * args.pairings
    lines = [
        f"{n} agents, {cells} agent pairs x {games} games = {cells * games} games in {elapsed:.1f}s; "
        f"{len(decks)} decks, {args.pairings} pairings, seed {args.seed} "
        f"(environment: {env.version if env is not None else 'none'})",
        "win rate of the row agent against the column agent (draws count half, errors left out); "
        f"alpha-rank alpha {alpha:g}, population {population}",
        "",
        "ranking: " + " > ".join(matrix.ranking()),
        "",
    ]
    lines += _table(matrix.agents, matrix.win_rate, matrix.nash, matrix.alpha_rank, label="agent")
    if matrix.total_errors():
        lines.append(f"\nwarning: {matrix.total_errors()} games ended in an error (left out of the win rates)")
    try:
        if args.out is not None:
            path = matrix.save(args.out)
        elif env is not None:
            path = matrix.save(env=env, name=args.name or "agents")
        else:
            path = None
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None
    if path is not None:
        lines += ["", f"written to {path}"]
    print("\n".join(lines))
    return 1 if matrix.total_errors() else 0
