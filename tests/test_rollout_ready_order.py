"""Worker completion order must not reassign random draws within a ready batch."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from ygorl.train.rollout import Assignment, RolloutCollector  # noqa: E402
from ygorl.train.toy import NimEnv, NimModel  # noqa: E402


class ReverseReady(NimEnv):
    def recv(self, min_events=1, timeout=None):
        return list(reversed(super().recv(min_events, timeout)))


@pytest.mark.parametrize("complete", [False, True])
def test_ready_order_preserves_actions_refills_and_continuation(complete):
    torch.set_num_threads(1)

    def collect(env_type):
        torch.manual_seed(107)
        model, opponent = NimModel(), NimModel()
        deals = itertools.count()

        def next_game():
            i = next(deals)
            return Assignment((5, 15, 8, 11)[i % 4], opponent=1 if i % 3 == 0 else None, learner_seat=i % 2)

        collector = RolloutCollector(
            env_type(4),
            model,
            next_game,
            16,
            opponents=lambda _: opponent,
            seed=811,
            min_batch=4,
            complete_games=complete,
        )
        return [collector.collect(), collector.collect()]

    forward, reverse = collect(NimEnv), collect(ReverseReady)
    for a, b in zip(forward, reverse, strict=True):
        for key in a.obs:
            torch.testing.assert_close(a.obs[key], b.obs[key], atol=0, rtol=0)
        for key in (
            "players",
            "actions",
            "probs",
            "q",
            "values",
            "rewards",
            "dones",
            "truncated",
            "bootstrap_player",
            "bootstrap_probs",
            "bootstrap_q",
            "bootstrap_value",
        ):
            torch.testing.assert_close(getattr(a, key), getattr(b, key), atol=0, rtol=0)
        assert a.games == b.games
        assert a.opponent_decisions == b.opponent_decisions
