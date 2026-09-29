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

The self-play :class:`ygorl.train.rollout.RolloutCollector` keeps its own loop: it holds columns at ``T`` rows and
watches for stalled engines, which whole-game playing does not need.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class Game:
    """One game running in an environment slot."""

    index: int  # position of the spec in ``specs``
    spec: Any
    env_id: int
    steps: int = 0  # decisions answered so far in this game
    state: Any = None  # the caller's per-game data (``start(index, spec)``), e.g. the two agents


Decide = Callable[[list[tuple[Game, Any]]], Sequence[Any]]


def drive(env, specs: Sequence[Any], decide: Decide, on_result: Callable[[Game, Any], None], *, min_batch: int = 1,
          start: Callable[[int, Any], Any] | None = None,
          on_error: Callable[[int, Any, Exception], None] | None = None) -> int:  # fmt: skip
    """Play every spec on ``env``'s slots; returns the number of decisions answered.

    ``decide([(game, event), ...]) -> actions`` answers the round's ready decisions (one action per event, in order);
    ``on_result(game, result)`` receives each finished game with the event's result. ``start(index, spec)``, if
    given, builds ``Game.state`` before the game is reset. ``on_error(index, spec, exc)``: see the module docstring.
    """
    queue = iter(enumerate(specs))
    running: dict[int, Game] = {}
    decisions = 0

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
            return True
        running.pop(env_id, None)
        return False

    for env_id in range(env.num_envs):
        launch(env_id)
    min_batch = max(1, min_batch)
    while running:
        ready = []
        for ev in env.recv(min(min_batch, len(running))):
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
            game.steps += 1
            decisions += 1
            env.step(ev.env_id, action)
    return decisions


__all__ = ["Decide", "Game", "drive"]
