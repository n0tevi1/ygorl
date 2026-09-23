"""`ygorl env build md-YYYY-MM` / `ygorl env check VERSION`: build and inspect environments (docs/data.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ygorl.commands import CommandError


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "env",
        help="build or check an environment (card pool, banlist, meta decks)",
        description="Build a Master Duel environment from YGOPRODECK, masterduelmeta and Yugipedia, or check an "
        "existing one (see docs/data.md and docs/environments.md).",
    )
    sub = p.add_subparsers(dest="env_command", metavar="<action>", required=True)

    b = sub.add_parser("build", help="fetch the sources and generate environments/<version>/",
                       description="Fetch missing raw files into <env>/raw/ (git-ignored), parse them into the "
                       "environment files and validate the result, meta decks included.")  # fmt: skip
    b.add_argument("version", help="environment version, md-YYYY-MM[-revision]")
    b.add_argument("--root", type=Path, default=None, help="environments directory (default $YGORL_ENVIRONMENTS "
                   "or ./environments)")  # fmt: skip
    b.add_argument("--raw", type=Path, default=None, metavar="DIR", help="raw source files (default <env>/raw)")
    b.add_argument("--offline", action="store_true", help="use the raw files as they are; never fetch")
    b.add_argument("--refresh", action="store_true", help="fetch every raw file again")
    b.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                   help="start of the meta window (default: the previous build's, else the date of the last Master "
                   "Duel banlist update)")  # fmt: skip
    b.add_argument("--min-share", type=float, default=None, metavar="F",
                   help="smallest share of a deck type kept as a meta deck (default: the previous build's, else 0.01)")  # fmt: skip
    b.add_argument("--max-decks", type=int, default=None, metavar="N",
                   help="at most N meta decks (default: the previous build's, else 20)")  # fmt: skip
    b.add_argument("--no-relations", action="store_true", help="skip the Yugipedia relations")
    b.add_argument("--reviewed-by", default=None, metavar="NAME",
                   help="record that NAME checked the banlist against the in-game list")  # fmt: skip
    b.set_defaults(func=run_build)

    c = sub.add_parser("check", help="validate an environment and print its summary")
    c.add_argument("env", metavar="PATH|VERSION", help="environment directory or version")
    c.set_defaults(func=run_check)


def run_build(args: argparse.Namespace) -> int:
    from ygorl.data.build import BuildError, BuildOptions, build
    from ygorl.data.environment import default_root
    from ygorl.data.fetch import FetchError

    if args.refresh and args.offline:
        raise CommandError("--refresh and --offline exclude each other")
    if args.min_share is not None and not 0 <= args.min_share <= 1 or args.max_decks is not None and args.max_decks < 0:
        raise CommandError("--min-share must be in [0, 1] and --max-decks non-negative")
    opts = BuildOptions(version=args.version, root=args.root or default_root(), raw_dir=args.raw,
                        offline=args.offline, refresh=args.refresh, since=args.since, min_share=args.min_share,
                        max_decks=args.max_decks, relations=not args.no_relations, reviewed_by=args.reviewed_by)  # fmt: skip
    try:
        result = build(opts)
    except (BuildError, FetchError, OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None
    s = result.stats
    ban = ", ".join(f"{v} {k}" for k, v in s["banlist"].items())
    print(f"built      {result.path}")
    print(f"pool       {s['pool']} cards ({len(s['pool_unmapped'])} unmapped, {len(s['pool_by_name'])} matched by name)")
    print(f"banlist    {ban} ({len(s['banlist_unmapped'])} unmapped, review {s['review']})")
    print(f"meta       {s['meta_decks']} decks, share {s['meta_share']:.1%} of {s['meta_counted']} counted lists "
          f"({len(s['meta_skipped'])} popular types without a legal list)")  # fmt: skip
    if s["relations"] is not None:
        print(f"relations  {s['relations']} cards")
    for w in result.warnings:
        print(f"warning    {w}")
    return 0


def run_check(args: argparse.Namespace) -> int:
    from ygorl.commands import load_env

    env = load_env(args.env)
    limits = env.banlist.limits
    counts = ", ".join(f"{sum(1 for v in limits.values() if v == k)} {name}"
                       for k, name in ((0, "forbidden"), (1, "limited"), (2, "semi-limited")))  # fmt: skip
    review = ((env.manifest.get("review") or {}).get("banlist") or {}).get("status", "n/a")
    print(f"environment {env.version} ({env.format}) at {env.root}")
    print(f"fingerprint {env.fingerprint[:16]}")
    print(f"pool        {len(env.card_pool)} cards")
    print(f"banlist     {env.banlist.name}: {counts} (review {review})")
    share = sum(m.share for m in env.meta_decks)
    print(f"meta        {len(env.meta_decks)} decks, share {share:.1%}, all legal")
    for m in env.meta_decks:
        print(f"  {m.share:6.1%}  {m.name}")
    return 0
