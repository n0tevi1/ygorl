"""Result reasons and the winner rule (ygorl.engine.outcome), shared by the Python and C++ hosts."""

import pytest

from ygorl.engine import outcome as O


@pytest.mark.parametrize("reason, rules_winner, lp, expected", [
    (O.WIN, 1, (8000, 0), 1),
    (O.WIN, None, (0, 0), None),  # both players lost at once
    (O.END, None, (100, 50), None),
    (O.TURN_LIMIT, None, (100, 50), 0),  # the higher LP wins
    (O.TURN_LIMIT, None, (-200, 50), 1),  # LP can be negative
    (O.TURN_LIMIT, None, (50, 50), None),
    (O.DECISION_LIMIT, None, (8000, 100), None),  # a loop, never a win
    (O.ERROR, 0, (8000, 100), None),
])  # fmt: skip
def test_who_is_credited(reason, rules_winner, lp, expected):
    assert O.winner(reason, rules_winner, lp) == expected


def test_truncations_are_the_host_stops():
    assert O.HOST_LIMITS == {O.TURN_LIMIT, O.DECISION_LIMIT}
    assert O.TRUNCATED == {O.TURN_LIMIT, O.DECISION_LIMIT, O.ERROR}
    assert not {O.WIN, O.END} & O.TRUNCATED and set(O.REASONS) >= O.TRUNCATED
