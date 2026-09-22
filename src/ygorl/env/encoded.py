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
    def __init__(self, num_envs: int, num_threads: int | None = None, cards=None, scripts=None,
                 vocab: CardVocab | None = None, privileged: bool = False) -> None:  # fmt: skip
        """``privileged=True`` is training mode: events also carry ``privileged`` (opponent ground truth).

        The default (inference mode) never computes it. Evaluation and play must use the default.
        """
        self.cards = cards if cards is not None else default_cards()
        self.vocab = vocab if vocab is not None else CardVocab.from_db(self.cards)
        passwords = [self.vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(self.vocab))]
        threads = num_threads or max(1, min(num_envs, os.cpu_count() or 1))
        self._pool = _core.HostPool(num_envs, threads, self.cards.to_core(),
                                    scripts if scripts is not None else default_scripts(), passwords,
                                    privileged)  # fmt: skip
        self.num_envs = num_envs
        self.num_threads = threads
        self.privileged = privileged

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

    def step(self, env_id: int, action: int) -> None:
        self._pool.step(env_id, action)

    def recv(self, min_events: int = 1, timeout: float | None = None) -> list[EncodedEvent]:
        timeout_ms = -1 if timeout is None else int(timeout * 1000)
        return [EncodedEvent(e, player, obs, result, priv)
                for e, _done, player, obs, result, priv in self._pool.recv(min_events, timeout_ms)]  # fmt: skip

    def pending(self) -> int:
        return self._pool.pending()

    def play(self, specs: Sequence[GameSpec], choose: Callable[[int, int, int], int]) -> list[dict]:
        """Play every spec with ``choose(seed, step, n_legal) -> index``; results in spec order."""
        results: list[dict | None] = [None] * len(specs)
        queue = iter(enumerate(specs))
        running: dict[int, list] = {}  # env -> [spec index, step]

        def launch(env_id: int) -> bool:
            nxt = next(queue, None)
            if nxt is None:
                return False
            running[env_id] = [nxt[0], 0]
            self.reset(env_id, nxt[1])
            return True

        active = sum(launch(e) for e in range(self.num_envs))
        while active:
            for ev in self.recv(1):
                i, step = running[ev.env_id]
                if ev.result is not None:
                    results[i] = ev.result
                    if not launch(ev.env_id):
                        active -= 1
                    continue
                n = int(ev.obs["action_mask"].sum())
                running[ev.env_id][1] = step + 1
                self.step(ev.env_id, choose(specs[i].seed, step, n))
        return results  # type: ignore[return-value]
