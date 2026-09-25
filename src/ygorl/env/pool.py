"""Vectorized duel environment on top of the C++ DuelPool (T2.1).

``VecDuelEnv`` runs many duels at once: the core work (processing up to the
next decision) happens on C++ worker threads without the GIL, while the
host-side bookkeeping reuses :class:`ygorl.engine.duel.DuelTracker`, so a game
played here is identical to ``Duel(...).run(...)`` with the same seed and
agents. Sub-steps of multi-selects are answered locally and never reach the
core until the decision is complete.

Asynchronous, envpool-style usage::

    env = VecDuelEnv(num_envs=64, num_threads=8)
    for i in range(64):
        env.start(i, next_spec())
    while True:
        for ev in env.recv(min_events=16):
            if ev.result is not None:
                ...                                   # game over; start another
            else:
                env.act(ev.env_id, policy(ev.point))  # answer the decision
"""

from __future__ import annotations

import dataclasses
import os
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ygorl import _core
from ygorl.cards.ydk import Deck
from ygorl.engine.duel import (
    DecisionPoint,
    Duel,
    DuelConfig,
    DuelResult,
    DuelTracker,
    default_cards,
    default_scripts,
    expand_seed,
)


@dataclass(frozen=True)
class GameSpec:
    seed: int
    deck_a: Deck
    deck_b: Deck
    first: int = 0
    config: DuelConfig = field(default_factory=DuelConfig)


def paired_specs(specs: Iterable[GameSpec]) -> list[GameSpec]:
    """Opening balance (先后攻配平): every spec twice, with deck_a going first and then second."""
    return [dataclasses.replace(spec, first=first) for spec in specs for first in (0, 1)]


@dataclass
class EnvEvent:
    env_id: int
    point: DecisionPoint | None = None  # a decision to answer with act()
    result: DuelResult | None = None  # the game is over


_IDLE, _RUNNING, _DECIDING = 0, 1, 2


class VecDuelEnv:
    def __init__(self, num_envs: int, num_threads: int | None = None, cards=None, scripts=None,
                 record_steps: bool = False) -> None:  # fmt: skip
        self.cards = cards if cards is not None else default_cards()
        scripts = scripts if scripts is not None else default_scripts()
        threads = num_threads or max(1, min(num_envs, os.cpu_count() or 1))
        self._pool = _core.DuelPool(num_envs, threads, self.cards.to_core(), scripts)
        self.num_envs = num_envs
        self.num_threads = threads
        self.record_steps = record_steps
        self._trackers: list[DuelTracker | None] = [None] * num_envs
        self._specs: list[GameSpec | None] = [None] * num_envs
        self._state = [_IDLE] * num_envs
        self._ready: deque[EnvEvent] = deque()

    # -- control -----------------------------------------------------------
    def start(self, env_id: int, spec: GameSpec) -> None:
        """Begin a new game in ``env_id`` (abandoning any game waiting for a decision there)."""
        self._check_id(env_id)
        if self._state[env_id] == _RUNNING:
            raise RuntimeError(f"env {env_id} is busy")
        duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=self.cards, config=spec.config, first=spec.first,
                    record_steps=self.record_steps)  # fmt: skip
        self._trackers[env_id] = duel.tracker()
        self._specs[env_id] = spec
        p = spec.config.player
        player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
        decks = [(list(main), list(extra)) for main, extra in duel.loaded_decks()]
        self._state[env_id] = _RUNNING
        self._pool.start(env_id, expand_seed(spec.seed), spec.config.rule_flags, player, player, decks)

    def act(self, env_id: int, action: int, probs=None) -> None:
        """Answer the pending decision point of ``env_id`` with an action index."""
        self._check_id(env_id)
        tracker = self._trackers[env_id]
        if self._state[env_id] != _DECIDING or tracker is None:
            raise RuntimeError(f"no decision is pending in env {env_id}")
        response = tracker.act(action, probs)
        if response is not None:
            self._state[env_id] = _RUNNING
            self._pool.respond(env_id, response)
        else:
            self._emit(env_id)

    def recv(self, min_events: int = 1, timeout: float | None = None) -> list[EnvEvent]:
        """Collect events; waits for at least ``min_events`` unless nothing more can arrive."""
        events = list(self._ready)
        self._ready.clear()
        need = max(0, min_events - len(events))
        timeout_ms = -1 if timeout is None else int(timeout * 1000)
        if need == 0:
            timeout_ms, need = 0, 0
        for env_id, status, buf, logs, error in self._pool.recv(need, timeout_ms):
            tracker = self._trackers[env_id]
            if status < 0:
                tracker.stop("error", error)
            else:
                tracker.on_buffer(buf, status, logs)
            self._emit(env_id)
        events.extend(self._ready)
        self._ready.clear()
        return events

    def spec(self, env_id: int) -> GameSpec | None:
        return self._specs[env_id]

    def close(self) -> None:
        self._pool = None

    # -- internals ---------------------------------------------------------
    def _check_id(self, env_id: int) -> None:
        if not 0 <= env_id < self.num_envs:
            raise IndexError(f"env id {env_id} out of range 0..{self.num_envs - 1}")

    def _emit(self, env_id: int) -> None:
        tracker = self._trackers[env_id]
        response = tracker.auto_response()  # the host answers for a restricted opponent (curriculum)
        if response is not None:
            self._state[env_id] = _RUNNING
            self._pool.respond(env_id, response)
            return
        point = tracker.point() if not tracker.done else None
        if point is not None:
            self._state[env_id] = _DECIDING
            self._ready.append(EnvEvent(env_id, point=point))
        elif tracker.done:
            self._state[env_id] = _IDLE
            self._pool.close_env(env_id)
            self._ready.append(EnvEvent(env_id, result=tracker.finish()))
        else:  # the core only continued (cannot happen: jobs run to a stop)
            raise RuntimeError(f"env {env_id} stopped without a decision or a result")


AgentFactory = Callable[[int, GameSpec], tuple]


def run_games(specs: Sequence[GameSpec], agent_factory: AgentFactory, num_envs: int = 8, num_threads: int | None = None,
              cards=None, scripts=None, record_steps: bool = False) -> list[DuelResult]:  # fmt: skip
    """Play every spec on a :class:`VecDuelEnv`; ``agent_factory(i, spec) -> (agent_a, agent_b)``.

    Results come back in spec order and equal ``Duel(...).run(...)`` played one by one.
    """
    env = VecDuelEnv(min(num_envs, max(1, len(specs))), num_threads, cards, scripts, record_steps)
    results: list[DuelResult | None] = [None] * len(specs)
    queue = iter(enumerate(specs))
    seats: dict[int, tuple[int, tuple]] = {}

    def launch(env_id: int) -> bool:
        nxt = next(queue, None)
        if nxt is None:
            return False
        i, spec = nxt
        a, b = agent_factory(i, spec)
        seats[env_id] = (i, (a, b) if spec.first == 0 else (b, a))
        env.start(env_id, spec)
        return True

    active = sum(launch(e) for e in range(env.num_envs))
    while active:
        for ev in env.recv(min_events=1):
            i, seat = seats[ev.env_id]
            if ev.result is not None:
                results[i] = ev.result
                if not launch(ev.env_id):
                    active -= 1
            else:
                agent = seat[ev.point.player]
                env.act(ev.env_id, agent.act(ev.point), getattr(agent, "last_probs", None))
    env.close()
    return results  # type: ignore[return-value]
