"""Subcommands of the `ygorl` command line (one module each, see :mod:`ygorl.cli`).

Shared option helpers live here. They import heavy modules inside the
functions so that building the parser (``ygorl --help``) stays fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SIDES = {0: "a", 1: "b", None: "draw"}
WIN_REASONS = {1: "lp", 2: "deck_out"}  # MSG_WIN reasons written by the core; 0x10+ come from card effects


class CommandError(Exception):
    """A user-facing error: printed as ``ygorl <command>: error: ...`` with exit code 2."""


def agents_help() -> str:
    from ygorl.agents.registry import available_agents

    return ", ".join(f"{name} ({desc})" if desc else name for name, desc in available_agents().items())


def add_env_option(p: argparse.ArgumentParser, help: str) -> None:
    p.add_argument("--env", default=None, metavar="PATH|VERSION", help=help)


def add_max_turns_option(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-turns", type=int, default=None, metavar="N",
                   help="end a game as a draw after N turns (default 200)")  # fmt: skip


def load_env(spec: str | None):
    """The environment named by ``--env`` (a directory or a version under the environments root), or None."""
    if spec is None:
        return None
    from ygorl.data import EnvironmentConfigError, load_environment

    try:
        return load_environment(spec)
    except (EnvironmentConfigError, OSError) as exc:
        raise CommandError(str(exc)) from None


def replay_env(replay, spec: str | None):
    """``--env`` if given, else the environment recorded in ``replay`` (looked up by version), else None."""
    if spec is not None or replay.environment is None:
        return load_env(spec)
    version = replay.environment["version"]
    try:
        return load_env(version)
    except CommandError as exc:
        raise CommandError(f"{exc}; pass --env PATH for environment {version}") from None


def load_replay(path: Path):
    from ygorl.engine.replay import Replay

    try:
        return Replay.load(path)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise CommandError(f"{path}: {exc}" if str(path) not in str(exc) else str(exc)) from None


def make_factory(spec: str):
    """Picklable named agent factory for ``spec`` (see :func:`ygorl.agents.registry.agent_factory`)."""
    from ygorl.agents.registry import agent_factory

    try:
        return agent_factory(spec)
    except ValueError as exc:
        raise CommandError(str(exc)) from None


def load_decks(paths: list[Path]) -> list:
    """Decks from ``.ydk`` files and directories (every ``*.ydk`` inside, sorted by name), named by file stem."""
    from ygorl.cards.ydk import YdkError, load_ydk

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.glob("*.ydk"))
            if not found:
                raise CommandError(f"no .ydk files in {path}")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise CommandError(f"deck file not found: {path}")
    try:
        return [load_ydk(f) for f in files]
    except (YdkError, OSError, UnicodeDecodeError) as exc:
        raise CommandError(str(exc)) from None


def duel_config(env, max_turns: int | None):
    """The environment's rules (or the defaults) with ``--max-turns`` applied."""
    from ygorl.engine.duel import DuelConfig

    overrides = {} if max_turns is None else {"max_turns": max_turns}
    if max_turns is not None and max_turns < 1:
        raise CommandError("--max-turns must be at least 1")
    return DuelConfig.from_environment(env, **overrides) if env is not None else DuelConfig(**overrides)


def describe_result(winner, reason: str, win_reason) -> tuple[str, str]:
    """(winner side, reason with the MSG_WIN cause) for reports."""
    detail = ""
    if reason == "win" and win_reason is not None:
        detail = f" ({WIN_REASONS.get(win_reason, f'card effect 0x{win_reason:x}')})"
    return SIDES[winner], reason + detail
