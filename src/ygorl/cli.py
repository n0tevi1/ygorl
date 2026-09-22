"""Command line entry point (`ygorl`). Subcommands arrive with T3.4."""

from __future__ import annotations

import argparse

from ygorl import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ygorl")
    parser.add_argument("--version", action="version", version=f"ygorl {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
