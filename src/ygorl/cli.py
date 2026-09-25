"""Command line entry point (`ygorl`).

Each subcommand lives in its own module under :mod:`ygorl.commands` and
exposes ``add_parser(subparsers)``, which registers the subcommand and sets
``func`` (``args -> exit code``) as a parser default. To add one, write the
module and list it in ``COMMANDS``. Modules should import heavy dependencies
inside ``func`` so ``ygorl --help`` stays fast.
"""

from __future__ import annotations

import argparse
import sys

from ygorl import __version__
from ygorl.commands import CommandError, arena, branch, duel, env, matrix, replay, strength

COMMANDS = (duel, arena, matrix, strength, replay, branch, env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ygorl", description="Yu-Gi-Oh! reinforcement-learning engine.")
    parser.add_argument("--version", action="version", version=f"ygorl {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for module in COMMANDS:
        module.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except CommandError as exc:
        print(f"ygorl {args.command}: error: {exc}", file=sys.stderr)
        return 2
