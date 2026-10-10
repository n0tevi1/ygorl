"""Vectorized environment with encoded observations, stepped entirely in C++ (T2.2 / T2.7).

Unlike :class:`ygorl.env.VecDuelEnv` (Python tracker, typed decision points),
``EncodedVecEnv`` keeps the tracker, the action state machines and the
observation encoder on the C++ worker threads: Python only receives the fixed
shape arrays of docs/encoding.md and answers with an action index. This is the
interface training uses.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ygorl import _core
from ygorl.cards.cdb import CardVocab
from ygorl.engine.duel import Duel, default_cards, default_scripts, expand_seed
from ygorl.env.driver import Game, drive
from ygorl.env.events import DEFAULT_EVENT_LENGTH
from ygorl.env.pool import GameSpec

MASK64 = (1 << 64) - 1


def chooser(seed: int, step: int, n: int) -> int:
    """Deterministic pseudo-random action index in [0, n) (splitmix64 of seed and step); for tests/benchmarks."""
    z = (seed * 0x9E3779B97F4A7C15 + step + 1) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) % n


@dataclass
class EncodedEvent:
    env_id: int
    player: int
    obs: dict[str, np.ndarray] | None  # the next decision to answer with step()
    result: dict | None  # the game is over (engine-player order)
    # Training mode only (privileged=True): opponent ground truth for the critic / belief losses
    # (docs/encoding.md). Kept out of ``obs`` so the actor cannot see it; None in inference mode.
    privileged: dict[str, np.ndarray] | None = None


class EncodedVecEnv:
    """``event_length``: event tokens per observation (``events`` / ``event_mask``, T2.4); 0 leaves them out."""

    def __init__(self, num_envs: int, num_threads: int | None = None, cards=None, scripts=None,
                 vocab: CardVocab | None = None, privileged: bool = False,
                 event_length: int = DEFAULT_EVENT_LENGTH, skip_forced: bool = False,
                 record_actions: bool = False, selection_history: bool = False, cancel_budget: int = 32) -> None:  # fmt: skip
        """``privileged=True`` is training mode: events also carry ``privileged`` (opponent ground truth).

        The default (inference mode) never computes it. Evaluation and play must use the default.
        ``cancel_budget`` restricts repeated target/material cancellation for BOTH engine seats.
        32 preserves existing behavior; 0/1/4 are experimental. It changes the native observation
        mask before forced-action skipping, policy sampling and rollout storage, retaining a sole exit.
        ``event_length`` is the number of event tokens per observation (docs/encoding.md; 0 = none).
        ``skip_forced=True`` plays every decision with exactly one choosable row (one legal action, or only
        equivalent copies of one; docs/encoding.md) inside the C++ loop, so
        only decisions with a real choice come back (the games are identical; ``step`` counts differ).
        ``record_actions=True`` retains explicit step indices across calls until each game ends. Limits/errors
        carry ``action_indices`` for replay with the same ``skip_forced`` setting; healthy wins discard the trace.
        """
        if type(cancel_budget) is not int or cancel_budget not in (0, 1, 4, 32):
            raise ValueError("cancel_budget must be 0, 1, 4 or 32")
        self.cancel_budget = cancel_budget
        self.cards = cards if cards is not None else default_cards()
        self.vocab = vocab if vocab is not None else CardVocab.from_db(self.cards)
        passwords = [self.vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(self.vocab))]
        threads = num_threads or max(1, min(num_envs, os.cpu_count() or 1))
        if selection_history and event_length <= 0:
            raise ValueError("selection history requires a nonempty event window")
        self._pool = _core.HostPool(num_envs, threads, self.cards.to_core(),
                                    scripts if scripts is not None else default_scripts(), passwords,
                                    privileged, event_length, skip_forced, selection_history, cancel_budget)  # fmt: skip
        self.num_envs = num_envs
        self.num_threads = threads
        self.privileged = privileged
        self.selection_history = selection_history
        self.event_length = event_length
        self.skip_forced = skip_forced
        self._action_traces: dict[int, list[int]] | None = {} if record_actions else None

    def reset(self, env_id: int, spec: GameSpec) -> None:
        if spec.config.curriculum != "full" or spec.config.augmented_start:
            raise NotImplementedError("EncodedVecEnv supports only curriculum='full' without augmented_start; "
                                      "curriculum modes (T2.6) run on VecDuelEnv / Duel.run")  # fmt: skip
        duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=self.cards, config=spec.config, first=spec.first)
        cfg = spec.config
        p = cfg.player
        player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
        decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
        self._pool.reset(env_id, expand_seed(spec.seed), cfg.rule_flags, player, player, decks, cfg.max_turns,
                         cfg.max_decisions)  # fmt: skip
        if self._action_traces is not None:
            self._action_traces[env_id] = []

    def step(self, env_id: int, action: int) -> None:
        self._pool.step(env_id, action)
        if self._action_traces is not None:
            self._action_traces[env_id].append(int(action))

    def recv(self, min_events: int = 1, timeout: float | None = None) -> list[EncodedEvent]:
        timeout_ms = -1 if timeout is None else int(timeout * 1000)
        events = []
        for e, done, player, obs, result, priv in self._pool.recv(min_events, timeout_ms):
            if done and self._action_traces is not None:
                actions = self._action_traces.pop(e, [])
                if result["reason"] in ("error", "turn_limit", "decision_limit") or any(
                    result.get(k)
                    for k in ("error", "retries", "unknown_messages", "script_errors", "undecodable_messages")
                ):
                    result["action_indices"] = actions
            events.append(EncodedEvent(e, player, obs, result, priv))
        return events

    def pending(self) -> int:
        return self._pool.pending()

    def play(self, specs: Sequence[GameSpec], choose: Callable[[int, int, int], int]) -> list[dict]:
        """Play every spec with ``choose(seed, step, n) -> k``, which picks the ``k``-th of the ``n`` rows the mask
        leaves (equivalent copies are masked, docs/encoding.md); results in spec order."""
        results: list[dict | None] = [None] * len(specs)

        def decide(ready: list[tuple[Game, EncodedEvent]]) -> list[int]:
            actions = []
            for game, ev in ready:
                legal = np.flatnonzero(ev.obs["action_mask"])
                actions.append(int(legal[choose(game.spec.seed, game.steps, len(legal))]))
            return actions

        def on_result(game: Game, result: dict) -> None:
            results[game.index] = result

        drive(self, specs, decide, on_result)
        return results  # type: ignore[return-value]
