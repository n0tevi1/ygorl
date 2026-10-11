"""The game driver: play a list of game specs on the slots of a batched environment (docs/evaluation.md「批量评估」).

Every batched player of whole games (:func:`ygorl.eval.batched.play_policies`, :meth:`EncodedVecEnv.play
<ygorl.env.encoded.EncodedVecEnv.play>`, :func:`ygorl.env.pool.run_games`) is the same loop: fill every slot from a
queue of specs, collect the ready events, answer the decisions in one batch, and when a game ends record it and start
the next spec in the freed slot. :func:`drive` owns that loop; the callers only say how to answer a batch of
decisions and what to keep of a finished game.

- **Environment**: anything with ``num_envs``, ``reset(env_id, spec)``, ``step(env_id, action)`` and
  ``recv(min_events)`` whose events carry ``env_id`` and ``result`` (``None`` while a decision is pending), i.e.
  :class:`ygorl.env.encoded.EncodedVecEnv`; :func:`ygorl.env.pool.run_games` adapts :class:`VecDuelEnv`.
- **Launch failures**: with ``on_error``, a spec whose ``reset`` raises is handed to it and the next spec is tried
  in the same slot (evaluation records it as an ``exception`` game); without it the exception propagates.
- **Batching**: each round waits for ``min(min_batch, running games)`` events; finished games are recorded and
  their slots refilled first, then ``decide`` answers all the ready decisions of the round at once and they are
  stepped in event order.
- **Abandoning**: ``decide`` may answer :data:`ABANDON` instead of an action: the game is dropped where it stands
  (no ``on_result``) and its slot starts the next spec, e.g. to read only the first decision of every game.
- **Stalls**: with ``stall_timeout``, a round that gets no event for that many seconds raises :class:`EnvStalled`
  naming the silent slots (an engine stuck inside a duel would otherwise hang the caller forever). The environment's
  ``recv`` must then take ``timeout=`` (seconds). :class:`StallWatch` is the watchdog, shared with the collector.

The self-play :class:`ygorl.train.rollout.RolloutCollector` keeps its own loop (it holds columns at ``T`` rows) on the
same :class:`StallWatch`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any


class EnvStalled(RuntimeError):
    """No environment slot produced an event for the stall timeout: an engine is stuck inside a duel. Its worker
    thread never returns, so destroying the environment (joining its threads) would hang too: the caller should save
    what it needs and end the process with ``os._exit`` (tools/train_ppo.py does)."""


class StallWatch:
    """``env.recv`` with a watchdog. :meth:`started` marks a slot's game start; :meth:`recv` waits for events like
    ``env.recv`` but raises ``error`` (an :class:`EnvStalled`) after ``timeout`` seconds without any, naming the
    slots that owe events, longest silent first; ``describe(env_id)`` adds what is known about a slot's game.
    ``timeout=None`` waits forever (plain ``env.recv``)."""

    def __init__(self, env, timeout: float | None, *, describe: Callable[[int], str] | None = None,
                 error: type[EnvStalled] = EnvStalled) -> None:  # fmt: skip
        self.env, self.timeout, self.describe, self.error = env, timeout, describe, error
        self.last: dict[int, float] = {}  # slot -> time of its last event (or game start)

    def started(self, env_id: int) -> None:
        self.last[env_id] = time.monotonic()

    def recv(self, n: int, owing: Iterable[int]) -> list:
        """At least ``n`` events (``env.recv(n)``); ``owing``: the slots that still owe one (for the report)."""
        if self.timeout is None:
            return self.env.recv(n)
        start = time.monotonic()
        while True:
            got = self.env.recv(n, timeout=min(60.0, self.timeout))
            now = time.monotonic()
            for ev in got:
                self.last[ev.env_id] = now
            if got:
                return got
            if now - start >= self.timeout:
                raise self.error(self.report(now, owing))

    def report(self, now: float, owing: Iterable[int]) -> str:
        waiting = sorted(owing, key=lambda e: self.last.get(e, 0.0))
        lines = [f"env {e}: silent {now - self.last.get(e, now):.0f}s"
                 + (f", {self.describe(e)}" if self.describe is not None else "") for e in waiting[:8]]  # fmt: skip
        return (f"no environment event for {self.timeout:.0f}s while {len(waiting)} environment(s) still owe events "
                "(an engine stuck inside a duel?); longest silent:\n  " + "\n  ".join(lines))  # fmt: skip


@dataclass
class Game:
    """One game running in an environment slot."""

    index: int  # position of the spec in ``specs``
    spec: Any
    env_id: int
    steps: int = 0  # decisions answered so far in this game
    state: Any = None  # the caller's per-game data (``start(index, spec)``), e.g. the two agents


ABANDON = object()  # ``decide``'s answer for a game to drop at this decision (see the module docstring)

Decide = Callable[[list[tuple[Game, Any]]], Sequence[Any]]


def drive(env, specs: Sequence[Any], decide: Decide, on_result: Callable[[Game, Any], None], *, min_batch: int = 1,
          start: Callable[[int, Any], Any] | None = None,
          on_error: Callable[[int, Any, Exception], None] | None = None,
          stall_timeout: float | None = None) -> int:  # fmt: skip
    """Play every spec on ``env``'s slots; returns the number of decisions answered.

    ``decide([(game, event), ...]) -> actions`` answers the round's ready decisions (one action per event, in order,
    or :data:`ABANDON`);
    ``on_result(game, result)`` receives each finished game with the event's result. ``start(index, spec)``, if
    given, builds ``Game.state`` before the game is reset. ``on_error(index, spec, exc)`` and ``stall_timeout``: see
    the module docstring.
    """
    queue = iter(enumerate(specs))
    running: dict[int, Game] = {}
    decisions = 0

    def describe(env_id: int) -> str:
        game = running.get(env_id)
        return "" if game is None else f"spec {game.index}, {game.steps} decisions"

    watch = StallWatch(env, stall_timeout, describe=describe)

    def launch(env_id: int) -> bool:
        for i, spec in queue:
            game = Game(i, spec, env_id, state=start(i, spec) if start is not None else None)
            try:
                env.reset(env_id, spec)
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(i, spec, exc)
                continue
            running[env_id] = game
            watch.started(env_id)
            return True
        running.pop(env_id, None)
        return False

    for env_id in range(env.num_envs):
        launch(env_id)
    min_batch = max(1, min_batch)
    while running:
        ready = []
        for ev in watch.recv(min(min_batch, len(running)), owing=running):
            game = running[ev.env_id]
            if ev.result is not None:
                on_result(game, ev.result)
                launch(ev.env_id)
            else:
                ready.append((game, ev))
        if not ready:
            continue
        actions = decide(ready)
        for (game, ev), action in zip(ready, actions, strict=True):
            if action is ABANDON:
                launch(ev.env_id)
                continue
            game.steps += 1
            decisions += 1
            env.step(ev.env_id, action)
    return decisions


__all__ = ["ABANDON", "Decide", "EnvStalled", "Game", "StallWatch", "drive"]
