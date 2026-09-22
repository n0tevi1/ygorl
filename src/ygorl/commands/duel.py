"""`ygorl duel A.ydk B.ydk`: play one game between two decks and print how it ended."""

from __future__ import annotations

import argparse
from pathlib import Path

from ygorl.commands import (
    CommandError,
    add_env_option,
    add_max_turns_option,
    agents_help,
    describe_result,
    duel_config,
    load_decks,
    load_env,
)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "duel",
        help="play one game between two decks",
        description="Play one game: agent a pilots deck A, agent b pilots deck B. Agent a is seeded with SEED "
        "and agent b with SEED + 1, so the same arguments replay the same game (see docs/cli.md).",
    )
    p.add_argument("deck_a", type=Path, metavar="A.ydk", help="deck of side a")
    p.add_argument("deck_b", type=Path, metavar="B.ydk", help="deck of side b")
    p.add_argument("--seed", type=int, default=0, help="duel seed (default 0)")
    p.add_argument("--first", choices=("a", "b"), default="a", help="side that moves first (default a)")
    p.add_argument("--agent-a", default="random", metavar="AGENT", help=f"agent of side a: {agents_help()}")
    p.add_argument("--agent-b", default="random", metavar="AGENT", help="agent of side b (default random)")
    add_env_option(p, "environment: rules, LP and hand size (default: none, Master Rule 5 defaults)")
    add_max_turns_option(p)
    p.add_argument("--save-replay", type=Path, default=None, metavar="PATH", help="write the replay (.json or .json.gz)")
    p.add_argument("--yrpx", type=Path, default=None, metavar="PATH", help="export an EDOPro replay (.yrpX)")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from ygorl.commands import make_factory
    from ygorl.engine.duel import Duel
    from ygorl.engine.replay import Replay

    deck_a, deck_b = load_decks([args.deck_a, args.deck_b])
    env = load_env(args.env)
    config = duel_config(env, args.max_turns)
    factory_a, factory_b = make_factory(args.agent_a), make_factory(args.agent_b)
    first = 0 if args.first == "a" else 1
    duel = Duel(args.seed, env, deck_a, deck_b, config=config, first=first)
    result = duel.run(factory_a(args.seed), factory_b(args.seed + 1))

    decks, agents = (deck_a.name, deck_b.name), (args.agent_a, args.agent_b)
    winner, reason = describe_result(result.winner, result.reason, result.win_reason)
    who = f" ({decks[result.winner]}, {agents[result.winner]})" if result.winner is not None else ""
    lines = [
        f"duel       {decks[0]} ({agents[0]}) vs {decks[1]} ({agents[1]}): seed {args.seed}, "
        f"{args.first} moves first, environment {env.version if env is not None else 'none'}",
        f"winner     {winner}{who}",
        f"reason     {reason}",
        f"turns      {result.turns}",
        f"lp         a {result.lp[0]}, b {result.lp[1]}",
        f"decisions  {result.decisions}",
    ]
    health = {"retries": result.retries, "unknown messages": result.unknown_messages,
              "undecodable messages": result.undecodable_messages, "script errors": len(result.script_errors)}  # fmt: skip
    problems = ", ".join(f"{n} {k}" for k, n in health.items() if n)
    if problems or result.error:
        lines.append(f"warnings   {problems}{'; ' if problems and result.error else ''}{result.error}")
    if args.save_replay is not None or args.yrpx is not None:
        replay = Replay.from_duel(duel, result)
        try:
            if args.save_replay is not None:
                args.save_replay.parent.mkdir(parents=True, exist_ok=True)
                replay.save(args.save_replay)
                lines.append(f"replay     {args.save_replay}")
            if args.yrpx is not None:
                args.yrpx.parent.mkdir(parents=True, exist_ok=True)
                replay.to_yrpx(args.yrpx, names=(deck_a.name, deck_b.name), env=env)
                lines.append(f"yrpX       {args.yrpx}")
        except (OSError, ValueError) as exc:
            raise CommandError(str(exc)) from None
    print("\n".join(lines))
    return 0

