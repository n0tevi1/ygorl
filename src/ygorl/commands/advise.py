"""`ygorl advise SNAPSHOT.json --deck D --policy P --env E`: read-only move suggestions for a board (design 09)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ygorl.commands import CommandError, add_env_option, deck_arg, load_decks, load_env


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "advise",
        help="suggest actions for a board snapshot (read-only Master Duel advisor)",
        description="Reconstruct a board snapshot (JSON, docs/design/10-master-duel.md) in the engine, sample the "
        "hidden cards from the environment's meta decks, and print the policy's averaged action distribution. "
        "Never sends anything to a game.",
    )
    p.add_argument("snapshot", type=Path, help="board snapshot (.json)")
    p.add_argument("--deck", type=deck_arg, required=True, metavar="DECK",
                   help="the advisor's own deck list (.ydk file or ydke:// link)")  # fmt: skip
    p.add_argument("--policy", type=Path, default=None, metavar="PATH",
                   help="policy checkpoint, PPO or BC (needs the train extra; without it: list the legal actions)")  # fmt: skip
    p.add_argument("--samples", type=int, default=8, metavar="N", help="hidden-card samples (default 8)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    p.add_argument("--top", type=int, default=10, metavar="K", help="suggestions to print (default 10)")
    add_env_option(p, "environment: rules, and the meta decks the opponent is sampled from (required)")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from ygorl.advisor.advise import advise, format_advice
    from ygorl.advisor.reconstruct import OpponentPrior, ReconstructError
    from ygorl.advisor.snapshot import SnapshotError, load_snapshot
    from ygorl.engine.duel import default_cards

    if args.env is None:
        raise CommandError("advise needs --env (the opponent is sampled from its meta decks)")
    if args.samples < 1:
        raise CommandError("--samples must be at least 1")
    env = load_env(args.env)
    (deck,) = load_decks([args.deck], env)
    try:
        snapshot = load_snapshot(args.snapshot)
        policy = None
        if args.policy is not None:
            from ygorl.train.checkpoint import load_actor

            policy = load_actor(args.policy)
        prior = OpponentPrior.from_environment(env)
        cards = default_cards()
        result = advise(snapshot, deck, policy, prior, env.rule_flags, cards=cards, samples=args.samples,
                        seed=args.seed, starting_lp=env.player.starting_lp)  # fmt: skip
    except (SnapshotError, ReconstructError, ValueError, OSError) as exc:
        raise CommandError(str(exc)) from None
    print(format_advice(result, cards, top=args.top, deck_names=[m.name for m in env.meta_decks]))
    return 0
