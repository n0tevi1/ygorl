"""`ygorl advise SNAPSHOT.json --deck D --policy P --env E`: read-only move suggestions for a board (design 10)."""

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
    p.add_argument("snapshot", type=Path, nargs="?", help="board snapshot (.json); omit with --attach")
    p.add_argument(
        "--attach",
        action="store_true",
        help="read the board from the running Master Duel client instead (Linux/Proton, read-only)",
    )
    p.add_argument(
        "--raw", action="store_true", help="with --attach: only print what was read, to check it against the screen"
    )
    p.add_argument("--save", type=Path, default=None, metavar="PATH", help="with --attach: also write the snapshot")
    p.add_argument("--deck", type=deck_arg, default=None, metavar="DECK",
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

    if args.attach == (args.snapshot is not None):
        raise CommandError("give either a snapshot file or --attach")
    if (args.raw or args.save) and not args.attach:
        raise CommandError("--raw and --save need --attach")
    if args.env is None:
        raise CommandError("advise needs --env (the opponent is sampled from its meta decks)")
    if args.deck is None and not args.raw:
        raise CommandError("advise needs --deck (the advisor's own deck list)")
    if args.samples < 1:
        raise CommandError("--samples must be at least 1")
    env = load_env(args.env)
    if args.attach:
        snapshot = _attach(args, env)
        if snapshot is None:
            return 0
    (deck,) = load_decks([args.deck], env)
    try:
        if not args.attach:
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


def _attach(args: argparse.Namespace, env):
    """Snapshot from the running client (None after --raw, which only prints)."""
    from ygorl.advisor.mdmem import MemoryReadError, attach, format_raw, load_md_cid, to_snapshot
    from ygorl.advisor.mdoffsets import OffsetError
    from ygorl.advisor.snapshot import SnapshotError
    from ygorl.engine.duel import default_cards

    cid_path = env.artifact_path("md_cid.json")
    if not cid_path.is_file():
        raise CommandError(f"{cid_path} is missing: run tools/build_md_cid.py {env.version}")
    passwords = load_md_cid(cid_path)
    try:
        game = attach()
        raw = game.read()
    except MemoryReadError as exc:
        raise CommandError(exc.strerror or str(exc)) from None
    except OffsetError as exc:
        raise CommandError(f"duel.dll: {exc}") from None
    off = game.offsets
    print(
        f"# pid {game.pid}, duel.dll {off.dll_sha256[:16]} at {game.module.base:#x}, buildid {game.buildid or '?'}, "
        f"offsets {'verified' if off.verified else 'derived, NOT verified'}"
    )
    if args.raw:
        cards = default_cards()
        names = {cid: cards[pw].name for cid, pw in passwords.items() if pw in cards}
        print(format_raw(raw, names))
        return None
    try:
        snapshot = to_snapshot(raw, passwords)
    except SnapshotError as exc:
        raise CommandError(str(exc)) from None
    if args.save:
        args.save.write_text(snapshot.dumps(), encoding="utf-8")
    return snapshot
