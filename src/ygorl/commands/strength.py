"""`ygorl strength AGENT...`: agent-vs-agent matchup matrix over a deck pool, ranked by alpha-rank and Nash."""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import replace
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
        "Nash mixture. With --env the result is written to environments/<version>/artifacts/agent-matrix/NAME.json. "
        "If the output (--out, or that artifact) already exists, the AGENTs are added to it: only their games are "
        "played, on the matrix's own deck pairings, seed and rules (see docs/cli.md).",
    )
    p.add_argument("agents", nargs="+", metavar="AGENT",
                   help=f"[NAME=]SPEC, at least two; NAME defaults to the spec: {agents_help()}")  # fmt: skip
    p.add_argument("--decks", type=Path, nargs="+", default=None, metavar="DECK",
                   help=".ydk files or directories: the deck pool (default: the meta decks of --env)")  # fmt: skip
    p.add_argument("--pairings", type=int, default=None, metavar="N",
                   help="deck pairings sampled from the pool; 4 games each per agent pair (default 50; when "
                        "extending, the matrix's own)")  # fmt: skip
    p.add_argument("--workers", type=int, default=1, metavar="N", help="worker processes (default 1)")
    p.add_argument("--device", default="auto", metavar="DEV",
                   help="play policy-vs-policy cells on the batched C++ path with the networks on DEV (cuda, cpu); "
                        "'none' plays everything in the arena; default auto: cuda if available, else none. An "
                        "existing matrix is extended on the path it was built with")  # fmt: skip
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the pairing sample and the games (default 0; when extending, the matrix's own)",
    )
    p.add_argument(
        "--confidence", type=float, default=None, help="level of the per-cell Wilson interval (default 0.95)"
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
    return out


DEVICE_RE = re.compile(r"^(auto|none|cpu|cuda(:[0-9]+)?)$")


def resolve_device(spec: str) -> str | None:
    """--device: None for the arena, else a torch device string."""
    if not DEVICE_RE.match(spec):
        raise CommandError(f"--device {spec!r}: auto, none, cpu, cuda or cuda:N")
    if spec == "none":
        return None
    if spec != "auto":
        return spec
    try:
        import torch
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else None


def run(args: argparse.Namespace) -> int:
    from ygorl.commands import make_factory
    from ygorl.eval.agent_matrix import ARTIFACT_DIR, AgentMatrix, build_agent_matrix, extend_agent_matrix
    from ygorl.eval.matchup import DEFAULT_ALPHA, DEFAULT_POPULATION

    env = load_env(args.env)
    if args.name is not None:
        if env is None:
            raise CommandError("--name names an environment artifact; pass --env (or use --out)")
        if not ARTIFACT_NAME_RE.match(args.name):
            raise CommandError(f"--name {args.name!r} must be a plain file name: letters, digits, '.', '_' and '-', "
                               "not starting with '.' or '-' (no path separators)")  # fmt: skip
    agents = parse_agents(args.agents)
    target = args.out if args.out is not None else (
        env.artifact_path(ARTIFACT_DIR, f"{args.name or 'agents'}.json") if env is not None else None)  # fmt: skip
    extending = target is not None and Path(target).exists()
    if not extending and len(agents) < 2:
        raise CommandError("a strength matrix needs at least two agents (or an existing matrix to add them to)")
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
    if args.workers < 1:
        raise CommandError("--workers must be at least 1")
    factories = {name: make_factory(spec) for name, spec in agents.items()}
    alpha = DEFAULT_ALPHA if args.alpha is None else args.alpha
    population = DEFAULT_POPULATION if args.population is None else args.population

    device = resolve_device(args.device)
    t0 = time.time()
    old = None
    try:
        if extending:
            old = AgentMatrix.load(target, env=env)
            for flag, given, have in (("--pairings", args.pairings, len(old.pairings)), ("--seed", args.seed, old.seed),
                                      ("--max-turns", args.max_turns, old.max_turns), ("--alpha", args.alpha, old.alpha),
                                      ("--population", args.population, old.population_size),
                                      ("--confidence", args.confidence, old.confidence)):  # fmt: skip
                if given is not None and given != have:
                    raise CommandError(f"{flag} {given} differs from the matrix at {target} ({have}): "
                                       "extend it with its own settings or write a new matrix")  # fmt: skip
            config = replace(duel_config(env, old.max_turns), max_decisions=old.max_decisions)
            if old.batched and device is None:
                print("note: the matrix was built on the batched path, so it is extended on it (on the CPU)",
                      file=sys.stderr)  # fmt: skip
            matrix = extend_agent_matrix(
                old,
                factories,
                decks,
                env=env,
                config=config,
                workers=args.workers,
                device=(device or "cpu") if old.batched else None,
            )
        else:
            pairings = 50 if args.pairings is None else args.pairings
            if pairings < 1:
                raise CommandError("--pairings must be at least 1")
            matrix = build_agent_matrix(factories, decks, pairings=pairings, seed=args.seed or 0, env=env,
                                        config=duel_config(env, args.max_turns), workers=args.workers,
                                        confidence=0.95 if args.confidence is None else args.confidence, alpha=alpha, population_size=population,
                                        device=device)  # fmt: skip
    except (ValueError, OSError, RuntimeError, ImportError) as exc:  # EnvironmentConfigError is a ValueError
        raise CommandError(str(exc)) from None  # e.g. no CUDA device, or no PyTorch for --device cpu
    elapsed = time.time() - t0

    n, k = len(matrix.agents), len(matrix.pairings)
    added = [a for a in matrix.agents if old is None or a not in old.agents]
    m = n - len(added)
    cells = n * (n - 1) // 2 - m * (m - 1) // 2
    head = f"added {', '.join(added) or 'nothing'} to {target}: " if old is not None else ""
    lines = [
        f"{head}{n} agents, {cells} agent pairs played x {4 * k} games = {cells * 4 * k} games in {elapsed:.1f}s; "
        f"{len(decks)} decks, {k} pairings, seed {matrix.seed} "
        f"(environment: {env.version if env is not None else 'none'})",
        "win rate of the row agent against the column agent (draws count half, errors left out); "
        f"alpha-rank alpha {matrix.alpha:g}, population {matrix.population_size}; policy-vs-policy cells: "
        + ("batched C++ path" if matrix.batched else "arena"),
        "",
        "ranking (alpha-rank, then Nash, then mean win rate in brackets): "
        + " > ".join(f"{a} ({matrix.mean_win_rate()[a]:.3f})" for a in matrix.ranking()),
        "",
    ]
    lines += _table(matrix.agents, matrix.win_rate, matrix.nash, matrix.alpha_rank, label="agent")
    if matrix.total_errors():
        lines.append(f"\nwarning: {matrix.total_errors()} games ended in an error (left out of the win rates)")
    try:
        path = matrix.save(target) if target is not None else None
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None
    if path is not None:
        lines += ["", f"written to {path}"]
    print("\n".join(lines))
    return 1 if matrix.total_errors() else 0
