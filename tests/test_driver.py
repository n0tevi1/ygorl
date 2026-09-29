"""The game driver (ygorl.env.driver) against a fake batched environment."""

from dataclasses import dataclass

import pytest

from ygorl.env.driver import drive


@dataclass
class Event:
    env_id: int
    player: int = 0
    result: dict | None = None


class FakeEnv:
    """A game spec is its length: that many decisions (players alternate), then a result with the actions taken.

    A negative spec cannot start. ``recv`` returns every ready event, as a real environment does once ``min_events``
    are ready; with ``max_events`` it returns at most that many, and none on every third call (a timeout), fewer than
    ``min_events`` asked for. It asserts the driver never waits for more events than games are running.
    """

    def __init__(self, num_envs: int, max_events: int | None = None) -> None:
        self.num_envs = num_envs
        self.max_events = max_events
        self.games: dict[int, list] = {}  # env -> [spec, actions]
        self.running = 0  # games reset whose result has not been returned by recv yet
        self.ready: list[Event] = []
        self.resets: list[tuple[int, int]] = []  # (env, spec) in reset order
        self.waits: list[int] = []

    def reset(self, env_id: int, spec: int) -> None:
        assert env_id not in self.games, "a slot was reset while its game was running"
        if spec < 0:
            raise ValueError(f"bad spec {spec}")
        self.resets.append((env_id, spec))
        self.games[env_id] = [spec, []]
        self.running += 1
        self._emit(env_id)

    def step(self, env_id: int, action) -> None:
        self.games[env_id][1].append(action)
        self._emit(env_id)

    def recv(self, min_events: int) -> list[Event]:
        assert 1 <= min_events <= self.running
        self.waits.append(min_events)
        n = len(self.ready)
        if self.max_events is not None:
            n = 0 if len(self.waits) % 3 == 0 else min(n, self.max_events)
        out, self.ready = self.ready[:n], self.ready[n:]
        self.running -= sum(ev.result is not None for ev in out)
        return out

    def _emit(self, env_id: int) -> None:
        spec, actions = self.games[env_id]
        if len(actions) < spec:
            self.ready.append(Event(env_id, player=len(actions) % 2))
        else:
            del self.games[env_id]
            self.ready.append(Event(env_id, result={"spec": spec, "actions": actions}))


def _play(env, specs, **kw):
    results, batches = {}, []

    def decide(ready):
        batches.append(len(ready))
        return [(game.index, game.steps) for game, _ in ready]

    def on_result(game, result):
        assert game.index not in results
        results[game.index] = (game.env_id, game.steps, result)

    n = drive(env, specs, decide, on_result, **kw)
    return results, batches, n


def test_every_spec_is_played_once_and_slots_are_reused():
    env = FakeEnv(3)
    specs = [2, 0, 5, 1, 3, 4, 2]
    results, _, n = _play(env, specs)
    assert sorted(results) == list(range(len(specs)))
    for i, (_, steps, res) in results.items():
        # the game's own decisions, in order, with the per-game step counter
        assert res == {"spec": specs[i], "actions": [(i, s) for s in range(specs[i])]} and steps == specs[i]
    assert n == sum(specs)
    # three slots only; the first three specs start first and every later one reuses a freed slot
    assert [e for e, _ in env.resets[:3]] == [0, 1, 2] and {e for e, _ in env.resets} == {0, 1, 2}
    assert [s for _, s in env.resets] == specs and not env.games


@pytest.mark.parametrize("specs", [[], [1], [3, 2]])
def test_fewer_specs_than_slots(specs):
    results, _, n = _play(FakeEnv(4), specs)
    assert sorted(results) == list(range(len(specs))) and n == sum(specs)


def test_launch_failures_are_recorded_and_the_slot_takes_the_next_spec():
    env = FakeEnv(2)
    errors = []
    specs = [-1, 2, -2, -3, 1, -4]  # the queue ends with a failure: the slot is simply freed
    results, _, _ = _play(env, specs, on_error=lambda i, spec, exc: errors.append((i, spec, str(exc))))
    assert errors == [(0, -1, "bad spec -1"), (2, -2, "bad spec -2"), (3, -3, "bad spec -3"), (5, -4, "bad spec -4")]
    assert sorted(results) == [1, 4] and [s for _, s in env.resets] == [2, 1]


def test_launch_failures_propagate_without_a_handler():
    with pytest.raises(ValueError, match="bad spec"):
        _play(FakeEnv(2), [1, -1])


def test_decisions_are_batched_up_to_min_batch():
    env = FakeEnv(4)
    specs = [3] * 4 + [1]
    _, batches, n = _play(env, specs, min_batch=4)
    # four slots each answer a decision per round; the waits never exceed the running games
    assert batches[:3] == [4, 4, 4] and sum(batches) == n == 13
    assert env.waits[0] == 4 and env.waits[-1] == 1


def test_start_builds_per_game_state():
    env = FakeEnv(2)
    seen = []

    def decide(ready):
        for game, ev in ready:
            game.state[ev.player] += 1  # per-seat counters, as batched evaluation keeps them
        return [0] * len(ready)

    drive(env, [3, 4, 1], decide, lambda game, res: seen.append((game.index, game.state)),
          start=lambda i, spec: [0, 0])  # fmt: skip
    assert sorted(seen) == [(0, [2, 1]), (1, [2, 2]), (2, [1, 0])]


@pytest.mark.parametrize("max_events", [1, 2])
def test_partial_and_empty_recv_still_finishes_every_game(max_events):
    env = FakeEnv(3, max_events=max_events)
    specs = [2, 0, 5, 1, 3, 4, 2]
    results, batches, n = _play(env, specs, min_batch=3)
    assert sorted(results) == list(range(len(specs))) and n == sum(specs) and not env.games
    for i, (_, steps, res) in results.items():
        assert res == {"spec": specs[i], "actions": [(i, s) for s in range(specs[i])]} and steps == specs[i]
    assert max(batches) <= max_events and 3 in env.waits  # smaller batches than asked for, and some empty rounds
