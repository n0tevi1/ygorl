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
from ygorl.engine.duel import Duel, default_cards, default_scripts
from ygorl.engine.outcome import TRUNCATED
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
                 record_actions: bool = False, selection_history: bool = False, cancel_budget: int = 32, attack_history: bool = False, record_games: bool = False, snapshots: bool = False) -> None:  # fmt: skip
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
        ``record_games=True`` enables recording for every outcome, plus per-action ``action_model_updates``.
        The caller sets ``trace_model_update`` before collection; None means an unspecified model version.
        ``snapshots=True`` enables :meth:`snapshot` / :meth:`restore` / :meth:`release` (docs/engine.md
        「步进环境快照」): every game's core lives in a private arena (about 3 MiB more per env, plus about 3 MiB per
        live snapshot), and a finished game stays restorable in its env until the next reset.
        """
        if type(cancel_budget) is not int or cancel_budget not in (0, 1, 4, 32):
            raise ValueError("cancel_budget must be 0, 1, 4 or 32")
        self.cancel_budget = cancel_budget
        self.cards = cards if cards is not None else default_cards()
        self.vocab = vocab if vocab is not None else CardVocab.from_db(self.cards)
        passwords = self.vocab.passwords()
        threads = num_threads or max(1, min(num_envs, os.cpu_count() or 1))
        if (selection_history or attack_history) and event_length <= 0:
            raise ValueError("selection history requires a nonempty event window")
        self._pool = _core.HostPool(num_envs, threads, self.cards.to_core(),
                                    scripts if scripts is not None else default_scripts(), passwords,
                                    privileged, event_length, skip_forced, selection_history, cancel_budget, attack_history,
                                    snapshots)  # fmt: skip
        self.num_envs = num_envs
        self.num_threads = threads
        self.privileged = privileged
        self.selection_history = selection_history
        self.attack_history = attack_history
        self.event_length = event_length
        self.skip_forced = skip_forced
        self.record_games = record_games
        self.trace_model_update: int | None = None
        self._action_traces: dict[int, list[int]] | None = {} if record_actions or record_games else None
        self._action_updates: dict[int, list[int | None]] = {}
        self.snapshots = snapshots
        # Per snapshot id: the env's action trace / model updates when it was taken (restored with it).
        self._snapshot_traces: dict[int, tuple[list[int] | None, list[int | None] | None]] = {}
        self._held: list[EncodedEvent] = []  # events received alongside a refused prefix (returned by the next recv)
        self._final_responses: dict[int, list[bytes]] = {}  # per env: the response log its last finished game reported

    def reset(self, env_id: int, spec: GameSpec, prefix: Sequence[int] | None = None) -> None:
        """Queue a new game of ``spec`` in ``env_id``; :meth:`recv` returns its first decision.

        ``prefix`` (prefix start, docs/engine.md 「前缀起局」): action indices the worker plays right after the start,
        exactly as successive :meth:`step` calls with this env's ``skip_forced`` would (one per decision that comes
        back to Python), without events in between; :meth:`recv` then returns the decision after the last one, or
        the result (with ``prefix_applied``, the actions played) if the game ends inside the prefix. An action that
        is not a choosable row of its decision's ``action_mask`` makes :meth:`recv` raise ValueError for the env
        (which is then left without a game).
        """
        if spec.config.curriculum != "full" or spec.config.augmented_start:
            raise NotImplementedError("EncodedVecEnv supports only curriculum='full' without augmented_start; "
                                      "curriculum modes (T2.6) run on VecDuelEnv / Duel.run")  # fmt: skip
        duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, cards=self.cards, config=spec.config, first=spec.first,
                    core_seed=spec.core_seed)  # fmt: skip
        cfg = spec.config
        p = cfg.player
        player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
        decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
        actions = [] if prefix is None else [int(a) for a in prefix]
        if any(a < 0 or a >= 1 << 32 for a in actions):
            raise ValueError("prefix action indices must be non-negative 32-bit integers")
        self._pool.reset(env_id, duel.core_seed, cfg.rule_flags, player, player, decks, cfg.max_turns,
                         cfg.max_decisions, actions)  # fmt: skip
        self._final_responses.pop(env_id, None)
        if self._action_traces is not None:
            self._action_traces[env_id] = list(actions)
        if self.record_games:
            self._action_updates[env_id] = [None] * len(actions)  # prefix actions come from no model

    def step(self, env_id: int, action: int) -> None:
        self._pool.step(env_id, action)
        self._final_responses.pop(env_id, None)
        if self._action_traces is not None:
            self._action_traces[env_id].append(int(action))
        if self.record_games:
            self._action_updates[env_id].append(self.trace_model_update)

    def snapshot(self, env_id: int) -> int:
        """Queue a snapshot of ``env_id``'s current decision and return its id.

        Like :meth:`step` it is a job: the env stays busy until :meth:`recv` returns its event, the unchanged
        current observation. The env must be idle at a decision of a running game (RuntimeError otherwise).
        """
        sid = self._pool.snapshot(env_id)
        trace = self._action_traces.get(env_id) if self._action_traces is not None else None
        updates = self._action_updates.get(env_id) if self.record_games else None
        self._snapshot_traces[sid] = (
            None if trace is None else list(trace),
            None if updates is None else list(updates),
        )
        return sid

    def restore(self, env_id: int, sid: int) -> None:
        """Queue a return of ``env_id`` to snapshot ``sid``; :meth:`recv` then returns the observation it was taken at
        (byte for byte). Only snapshots of the env's current game restore, finished or not: a snapshot of another env
        or of an earlier game raises ValueError. The snapshot stays valid until :meth:`release`."""
        self._pool.restore(env_id, sid)
        self._final_responses.pop(env_id, None)
        trace, updates = self._snapshot_traces[sid]
        if self._action_traces is not None and trace is not None:
            self._action_traces[env_id] = list(trace)
        if self.record_games and updates is not None:
            self._action_updates[env_id] = list(updates)

    def release(self, env_id: int, sid: int) -> None:
        """Free snapshot ``sid`` of ``env_id`` (immediately; no event). Snapshots are never freed implicitly."""
        self._pool.release(env_id, sid)
        self._snapshot_traces.pop(sid, None)

    def responses(self, env_id: int) -> list[bytes]:
        """The engine responses of ``env_id``'s game so far (``set_response`` payloads, as ``result["responses"]``).

        With the game's spec they make a :class:`~ygorl.engine.replay.Replay` (:meth:`Replay.from_spec`) whose
        :meth:`~ygorl.engine.replay.Replay.play` re-runs the game to the start of the current decision (a
        multi-step decision in progress has no response yet). The env must be idle (its last event received);
        after the game ended it is the final log.
        """
        if env_id in self._final_responses:
            return list(self._final_responses[env_id])
        return list(self._pool.responses(env_id))

    def snapshot_count(self) -> int:
        return self._pool.snapshot_count()

    def snapshot_bytes(self) -> int:
        """Bytes held by the live snapshots' core images (the tracker copies add a few KiB each)."""
        return self._pool.snapshot_bytes()

    def arena_escapes(self) -> int:
        """Allocations that escaped the arenas of this env's duels; must stay 0 for exact snapshots."""
        return self._pool.arena_escapes()

    def recv(self, min_events: int = 1, timeout: float | None = None) -> list[EncodedEvent]:
        """At least ``min_events`` events (fewer once ``timeout`` seconds pass). A refused prefix (:meth:`reset`) raises
        ValueError; the events received with it are returned by the next call."""
        events, self._held = self._held, []
        timeout_ms = -1 if timeout is None else int(timeout * 1000)
        need = max(0, min_events - len(events))
        if need or self._pool.pending():
            events += [self._event(*raw) for raw in self._pool.recv(need, timeout_ms if need else 0)]
        for i, ev in enumerate(events):
            if ev.result is not None and ev.result.get("prefix_error"):
                self._held = events[:i] + events[i + 1 :]
                raise ValueError(f"env {ev.env_id}: illegal prefix: {ev.result['error']}")
        return events

    def _event(self, e: int, done: bool, player: int, obs, result, priv) -> EncodedEvent:
        if not done:
            return EncodedEvent(e, player, obs, result, priv)
        if result.pop("prefix_error", False):  # the env has no game: nothing to record
            if self._action_traces is not None:
                self._action_traces.pop(e, None)
            self._action_updates.pop(e, None)
            return EncodedEvent(e, player, None, {**result, "prefix_error": True})
        self._final_responses[e] = result["responses"]
        if self._action_traces is not None:
            actions = self._action_traces.pop(e, [])
            updates = self._action_updates.pop(e, []) if self.record_games else []
            if "prefix_applied" in result:  # the game ended inside the prefix: the rest was never played
                del actions[result["prefix_applied"] :], updates[result["prefix_applied"] :]
            if (
                self.record_games
                or result["reason"] in TRUNCATED
                or any(
                    result.get(k)
                    for k in ("error", "retries", "unknown_messages", "script_errors", "undecodable_messages")
                )
            ):
                result["action_indices"] = actions
            if self.record_games:
                result["action_model_updates"] = updates
        return EncodedEvent(e, player, obs, result, priv)

    def pending(self) -> int:
        return self._pool.pending() + len(self._held)

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
