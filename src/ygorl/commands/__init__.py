"""Subcommands of the `ygorl` command line (one module each, see :mod:`ygorl.cli`)."""


class CommandError(Exception):
    """A user-facing error: printed as ``ygorl <command>: error: ...`` with exit code 2."""
