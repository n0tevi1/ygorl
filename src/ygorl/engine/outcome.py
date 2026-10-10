"""How a duel ended: the result reasons both hosts report, and what each one means.

``DuelResult.reason`` (Python host) and the ``reason`` of a C++ host's result dict are one of the names below. A game
that ends by the rules (``WIN``, ``END``) is a real outcome; one the host had to stop (a limit or an error) is not, and
consumers that care (training targets, demonstration checks, replays) ask this module instead of listing reasons:

- :data:`HOST_LIMITS`: the host stopped the duel at its turn or decision limit;
- :data:`TRUNCATED`: no real end (the limits and errors): training bootstraps such games, demonstrations and replays
  that hit one are not trusted;
- :func:`winner`: who is credited, by engine seat (the C++ mirror is ``Tracker::winner`` in ``csrc/host.cpp``).
"""

from __future__ import annotations

from collections.abc import Sequence

WIN = "win"  # a MSG_WIN: a player won by the rules (LP, deck out, a card effect)
END = "end"  # the core ended the duel without a MSG_WIN
TURN_LIMIT = "turn_limit"  # DuelConfig.max_turns passed: the higher LP wins (a rule of the format)
DECISION_LIMIT = "decision_limit"  # DuelConfig.max_decisions reached: stops loops, always a draw
ERROR = "error"  # a script error, an engine loop, an undecodable decision, repeated MSG_RETRY: a draw
LOG_EXHAUSTED = "log_exhausted"  # replay only: the engine asked for more responses than were recorded

REASONS = (WIN, END, TURN_LIMIT, DECISION_LIMIT, ERROR, LOG_EXHAUSTED)
HOST_LIMITS = frozenset({TURN_LIMIT, DECISION_LIMIT})
TRUNCATED = HOST_LIMITS | {ERROR}


def winner(reason: str, rules_winner: int | None, lp: Sequence[int]) -> int | None:
    """The engine seat credited with a duel that stopped for ``reason``; ``rules_winner`` is the MSG_WIN player (None
    without one) and ``lp`` is in engine-seat order. The turn limit is a rule (the higher LP wins, equal LP is a
    draw); the decision limit only stops a loop and an error is no outcome: both are draws, so that looping or
    crashing while ahead never counts as a win."""
    if reason == TURN_LIMIT:
        return None if lp[0] == lp[1] else (0 if lp[0] > lp[1] else 1)
    if reason in TRUNCATED:
        return None
    return rules_winner


__all__ = ["DECISION_LIMIT", "END", "ERROR", "HOST_LIMITS", "LOG_EXHAUSTED", "REASONS", "TRUNCATED", "TURN_LIMIT",
           "WIN", "winner"]  # fmt: skip
